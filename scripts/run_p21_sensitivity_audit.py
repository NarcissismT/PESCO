#!/usr/bin/env python3
"""Audit reward-weight sensitivity and family-wise directions for P2.1.

This is a diagnostic-only secondary analysis.  It consumes retained evaluator
rows from isolated P2.1 policy runs, applies the same global atomic-reward weight
draw to every method, and reports rank stability, family-wise deltas, and the
explicit confirmation/validity denominators.  It never opens the v0.5 final
bundle and cannot authorize a formal comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p2_experiments import (
    _perturbed_winner_stability,
    normalized_regret,
)
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


METHOD_DIRS = {
    "SFT": "sft",
    "PESCO-BranchOnly": "branch_only",
    "PESCO-NoFlipLoss": "no_flip",
    "PESCO-Full": "full",
}


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _retained_rows(methods_dir: Path, split: str) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], list[str]]:
    rows_by_method: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    missing: list[str] = []
    # The diagnostic CLI writes one aggregate ``p21_result.json`` containing
    # one summary row per method/split.  Accept that layout directly, while
    # retaining support for the older isolated-per-method directory layout.
    aggregate_path = methods_dir / "p21_result.json"
    aggregate_payload: Mapping[str, Any] | None = None
    if aggregate_path.is_file():
        loaded = _load(aggregate_path)
        if isinstance(loaded, Mapping):
            aggregate_payload = loaded
    for method, slug in METHOD_DIRS.items():
        if aggregate_payload is not None:
            path = aggregate_path
            aggregate = next(
                (
                    row for row in aggregate_payload.get("records", [])
                    if isinstance(row, Mapping)
                    and row.get("split") == split
                    and row.get("method") == method
                ),
                None,
            )
        else:
            path = methods_dir / slug / "p21_result.json"
            if not path.is_file():
                missing.append(method)
                continue
            payload = _load(path)
            aggregate = next(
                (
                    row for row in payload.get("records", [])
                    if isinstance(row, Mapping) and row.get("split") == split
                ),
                None,
            )
        retained = aggregate.get("records") if isinstance(aggregate, Mapping) else None
        if not isinstance(retained, list) or not retained:
            missing.append(method)
            continue
        parsed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in retained:
            if not isinstance(row, Mapping):
                continue
            key = (str(row.get("question_id", "")), str(row.get("world_id", "")))
            action = str(row.get("selected_action", ""))
            if key[0] and key[1] and action:
                parsed[key] = dict(row)
        if parsed:
            rows_by_method[method] = parsed
        else:
            missing.append(method)
    return rows_by_method, missing


def _component_payload(dataset: DecisionDataset, rows_by_method: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]], split: str) -> tuple[list[Any], tuple[str, ...], list[str]]:
    examples = [example for example in dataset.examples if str(example.split) == str(split)]
    names: set[str] = set()
    problems: list[str] = []
    for example in examples:
        components = example.metadata.get("reward_components")
        if not isinstance(components, Mapping):
            problems.append(f"{example.question_id}/{example.world_id}:missing_reward_components")
            continue
        for action_terms in components.values():
            if not isinstance(action_terms, Mapping) or not action_terms:
                problems.append(f"{example.question_id}/{example.world_id}:malformed_reward_components")
                continue
            names.update(str(key) for key in action_terms)
    for method, rows in rows_by_method.items():
        absent = [
            f"{method}:{example.question_id}/{example.world_id}"
            for example in examples
            if (str(example.question_id), str(example.world_id)) not in rows
        ]
        problems.extend(absent)
    return examples, tuple(sorted(names)), problems


def _method_metrics(
    examples: list[Any],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    weights: Mapping[str, float],
) -> tuple[float, float, dict[str, float]]:
    regrets: list[float] = []
    raw_regrets: list[float] = []
    by_question: dict[str, list[float]] = {}
    for example in examples:
        row = rows[(str(example.question_id), str(example.world_id))]
        selected_action = str(row.get("selected_action", ""))
        components = example.metadata.get("reward_components", {})
        values = {
            str(action): sum(
                float(value) * float(weights.get(str(term), 1.0))
                for term, value in (terms.items() if isinstance(terms, Mapping) else ())
            )
            for action, terms in components.items()
        }
        if selected_action not in values or not values:
            continue
        action_names = list(values)
        selected_index = action_names.index(selected_action)
        ordered = [values[name] for name in action_names]
        raw = max(ordered) - ordered[selected_index]
        normalized = normalized_regret(ordered, selected_index)
        raw_regrets.append(float(raw))
        regrets.append(float(normalized))
        by_question.setdefault(str(example.question_id), []).append(float(normalized))
    return (
        sum(regrets) / len(regrets) if regrets else float("nan"),
        sum(raw_regrets) / len(raw_regrets) if raw_regrets else float("nan"),
        {question: sum(values) / len(values) for question, values in by_question.items() if values},
    )


def _rank(values: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda key: (float(values[key]), key)))


def _family_directions(
    examples: list[Any],
    question_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    families: dict[str, str] = {}
    for example in examples:
        families[str(example.question_id)] = str(example.metadata.get("family", "unknown"))
    output: dict[str, Any] = {}
    for left, right in (("PESCO-Full", "PESCO-NoFlipLoss"), ("PESCO-Full", "SFT")):
        by_family: dict[str, list[float]] = {}
        for question_id, family in families.items():
            if question_id not in question_metrics.get(left, {}) or question_id not in question_metrics.get(right, {}):
                continue
            by_family.setdefault(family, []).append(
                question_metrics[left][question_id] - question_metrics[right][question_id]
            )
        key = f"{left}_minus_{right}_normalized_regret"
        output[key] = {
            family: {
                "question_count": len(values),
                "mean_delta": sum(values) / len(values) if values else None,
                "full_better_question_n": sum(value < 0.0 for value in values),
                "full_worse_question_n": sum(value > 0.0 for value in values),
                "tie_question_n": sum(value == 0.0 for value in values),
            }
            for family, values in sorted(by_family.items())
        }
    return output


def _safety_snapshot(
    rows_by_method: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, rows in rows_by_method.items():
        def count(key: str) -> int:
            return sum(bool(row.get(key)) for row in rows.values())
        def integer_sum(key: str) -> int:
            return sum(int(row.get(key, 0) or 0) for row in rows.values())
        eligible = integer_sum("selected_confirmation_eligible_n")
        passed = integer_sum("selected_confirmation_passed_n")
        output[method] = {
            "row_count": len(rows),
            "selected_confirmation_observed_n": integer_sum("selected_confirmation_observed_n"),
            "selected_confirmation_eligible_n": eligible,
            "selected_confirmation_passed_n": passed,
            "selected_confirmation_rate": passed / eligible if eligible else None,
            "selected_invalid_branch_n": count("selected_branch_invalid"),
            "invalid_local_optimization_n": count("invalid_local_optimization"),
            "erroneous_repair_n": count("erroneous_repair_action"),
        }
    return output


def run(
    dataset_path: Path,
    methods_dir: Path,
    output_dir: Path,
    *,
    split: str = "promotion",
    replicates: int = 1000,
    seed: int = 202629,
) -> dict[str, Any]:
    dataset = DecisionDataset.from_json(dataset_path)
    rows_by_method, missing = _retained_rows(methods_dir, split)
    examples, component_names, problems = _component_payload(dataset, rows_by_method, split)
    if missing or problems:
        result = {
            "schema_version": "pesco_tier1_p21_sensitivity_audit_v0.1",
            "status": "fail_closed_missing_retained_rows",
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "split": split,
            "missing_methods": missing,
            "problem_count": len(problems),
            "problems_sample": problems[:20],
            "input_dataset": str(dataset_path),
            "methods_dir": str(methods_dir),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sensitivity_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return result

    zero_weights = {name: 1.0 for name in component_names}
    base_norm: dict[str, float] = {}
    base_raw: dict[str, float] = {}
    question_metrics: dict[str, dict[str, float]] = {}
    for method, rows in rows_by_method.items():
        norm, raw, questions = _method_metrics(examples, rows, zero_weights)
        base_norm[method] = norm
        base_raw[method] = raw
        question_metrics[method] = questions
    baseline_norm_rank = _rank(base_norm)
    baseline_raw_rank = _rank(base_raw)
    rng = random.Random(int(seed))
    norm_rank_matches = 0
    raw_rank_matches = 0
    pairwise_norm_agreement = 0
    pairwise_raw_agreement = 0
    total_pairwise = 0
    replicate_rows: list[dict[str, Any]] = []
    methods = tuple(sorted(rows_by_method))
    for replicate in range(max(1, int(replicates))):
        weights = {name: rng.uniform(0.80, 1.20) for name in component_names}
        norm_values: dict[str, float] = {}
        raw_values: dict[str, float] = {}
        for method, rows in rows_by_method.items():
            norm_values[method], raw_values[method], _ = _method_metrics(examples, rows, weights)
        norm_rank = _rank(norm_values)
        raw_rank = _rank(raw_values)
        norm_rank_matches += int(norm_rank == baseline_norm_rank)
        raw_rank_matches += int(raw_rank == baseline_raw_rank)
        for index, left in enumerate(methods):
            for right in methods[index + 1:]:
                base_norm_order = base_norm[left] <= base_norm[right]
                draw_norm_order = norm_values[left] <= norm_values[right]
                base_raw_order = base_raw[left] <= base_raw[right]
                draw_raw_order = raw_values[left] <= raw_values[right]
                pairwise_norm_agreement += int(base_norm_order == draw_norm_order)
                pairwise_raw_agreement += int(base_raw_order == draw_raw_order)
                total_pairwise += 1
        if replicate < 10:
            replicate_rows.append({
                "replicate": replicate,
                "weights": weights,
                "normalized_regret": norm_values,
                "raw_regret": raw_values,
                "normalized_rank": list(norm_rank),
                "raw_rank": list(raw_rank),
            })
    stability = _perturbed_winner_stability(
        dataset,
        tolerance=0.02,
        replicates=min(100, max(1, int(replicates))),
        seed=int(seed) + 1,
    )
    result = {
        "schema_version": "pesco_tier1_p21_sensitivity_audit_v0.1",
        "status": "completed_cpu_diagnostic",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "split": split,
        "question_count": len({str(example.question_id) for example in examples}),
        "example_count": len(examples),
        "methods": list(methods),
        "component_names": list(component_names),
        "weight_range": [0.80, 1.20],
        "replicates": max(1, int(replicates)),
        "seed": int(seed),
        "baseline_ranking": {
            "normalized_regret": list(baseline_norm_rank),
            "raw_regret": list(baseline_raw_rank),
            "normalized_regret_values": base_norm,
            "raw_regret_values": base_raw,
        },
        "global_weight_rank_stability": {
            "normalized_regret_exact_rank_fraction": norm_rank_matches / max(1, int(replicates)),
            "raw_regret_exact_rank_fraction": raw_rank_matches / max(1, int(replicates)),
            "pairwise_normalized_order_agreement": pairwise_norm_agreement / max(1, total_pairwise),
            "pairwise_raw_order_agreement": pairwise_raw_agreement / max(1, total_pairwise),
            "pair_count_per_replicate": len(methods) * max(0, len(methods) - 1) // 2,
            "definition": "one shared independent uniform weight draw per atomic reward term, applied to every method",
        },
        "family_direction_audit": _family_directions(examples, question_metrics),
        "safety_receipt_snapshot": _safety_snapshot(rows_by_method),
        "environment_reward_winner_stability": stability,
        "sample_replicates": replicate_rows,
        "limitations": [
            "Policy actions are frozen from one retained CPU diagnostic run; this is not retraining under perturbed rewards.",
            "The analysis is diagnostic-only and does not open v0.5 final comparison or model scaling.",
            "Safety snapshot is descriptive; preregistered non-inferiority requires paired multi-seed formal receipts.",
        ],
        "input_dataset_digest": _digest(dataset_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sensitivity_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    input_paths = [dataset_path]
    if (methods_dir / "p21_result.json").is_file():
        input_paths.append(methods_dir / "p21_result.json")
    else:
        input_paths.extend(methods_dir / slug / "p21_result.json" for slug in METHOD_DIRS.values())
    manifest = build_run_manifest(
        experiment="tier1_p21_reward_weight_sensitivity_audit",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[Path(__file__), ROOT / "research_strategy_optimization/evaluation/tier1_p2_experiments.py"],
        data_paths=input_paths,
        seeds={"weight_perturbation": int(seed)},
        status=result["status"],
        diagnostics={"diagnostic_only": True, "formal_comparison_authorized": False, "split": split},
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--methods-dir", type=Path, required=True, help="isolated runs created with --retain-example-records")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p21_sensitivity_audit")
    parser.add_argument("--split", default="promotion")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202629)
    args = parser.parse_args(argv)
    result = run(args.dataset, args.methods_dir, args.output_dir, split=args.split, replicates=args.replicates, seed=args.seed)
    print(json.dumps({
        "output": str(args.output_dir),
        "status": result.get("status"),
        "global_weight_rank_stability": result.get("global_weight_rank_stability"),
        "family_direction_audit": result.get("family_direction_audit"),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "completed_cpu_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())

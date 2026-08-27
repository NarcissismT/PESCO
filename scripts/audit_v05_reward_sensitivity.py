#!/usr/bin/env python3
"""Audit v0.5 atomic-reward winner stability without evaluating a model.

The v0.5 evaluator bundle already contains one receipt for every
question/world/action/exploration-seed tuple.  This script aggregates the eight
exploration receipts for each action, then applies one shared ±20% draw to every
atomic reward term across all actions.  It is an evaluator-side structural
diagnostic only: it never reads policy logits, never selects a baseline, and
never authorizes final model comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ACTIONS = (
    "continue_current_method",
    "add_samples_or_seeds",
    "repair_data_split",
    "switch_to_alternative_method",
)
EXPECTED_EXPLORATION_SEEDS = (17, 29, 41, 53, 67, 71, 83, 97)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _top1_top2_gap(values: list[float]) -> float:
    ordered = sorted((float(value) for value in values), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else float("nan")


def _load_rows(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("environment receipt payload must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("environment receipt rows are missing or malformed")
    return payload, list(rows)


def audit(
    receipts_path: Path,
    *,
    replicates: int = 1000,
    seed: int = 202629,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    payload, rows = _load_rows(receipts_path)
    supplied_receipt_digest = payload.get("receipt_digest")
    recomputed_receipt_digest = _digest({
        key: value for key, value in payload.items() if key != "receipt_digest"
    })
    groups: dict[tuple[str, str], dict[str, dict[int, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    problems: list[str] = []
    if supplied_receipt_digest != recomputed_receipt_digest:
        problems.append("top_level_receipt_digest_mismatch")
    term_names: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id_audit", ""))
        world_id = str(row.get("world_id_audit", ""))
        action = str(row.get("action", ""))
        try:
            exploration_seed = int(row.get("exploration_seed"))
        except (TypeError, ValueError, OverflowError):
            problems.append(f"{question_id}/{world_id}/{action}:bad_seed")
            continue
        if not question_id or not world_id or action not in EXPECTED_ACTIONS:
            problems.append(f"{question_id}/{world_id}/{action}:bad_identity")
            continue
        if exploration_seed not in EXPECTED_EXPLORATION_SEEDS:
            problems.append(f"{question_id}/{world_id}/{action}:{exploration_seed}:unexpected_seed")
            continue
        if exploration_seed in groups[(question_id, world_id)][action]:
            problems.append(f"{question_id}/{world_id}/{action}:{exploration_seed}:duplicate")
            continue
        components = row.get("reward_components")
        if not isinstance(components, Mapping) or not components:
            problems.append(f"{question_id}/{world_id}/{action}:{exploration_seed}:missing_components")
            continue
        try:
            utility = float(row["utility"])
            numeric = {str(key): float(value) for key, value in components.items()}
        except (KeyError, TypeError, ValueError, OverflowError):
            problems.append(f"{question_id}/{world_id}/{action}:{exploration_seed}:non_numeric")
            continue
        if any(not math.isfinite(value) for value in numeric.values()) or not math.isfinite(utility):
            problems.append(f"{question_id}/{world_id}/{action}:{exploration_seed}:non_finite")
            continue
        if not math.isclose(sum(numeric.values()), utility, rel_tol=0.0, abs_tol=1e-12):
            problems.append(f"{question_id}/{world_id}/{action}:{exploration_seed}:utility_mismatch")
            continue
        term_names.update(numeric)
        groups[(question_id, world_id)][action][exploration_seed] = {
            "components": numeric,
            "utility": utility,
            "split": str(row.get("split", "")),
        }

    expected_group_count = int(payload.get("world_count_collected", 0) or 0)
    complete_groups: list[tuple[tuple[str, str], dict[str, dict[int, Mapping[str, Any]]]]] = []
    for key, actions in sorted(groups.items()):
        for action in EXPECTED_ACTIONS:
            observed = actions.get(action, {})
            if set(observed) != set(EXPECTED_EXPLORATION_SEEDS):
                problems.append(f"{key[0]}/{key[1]}/{action}:seed_coverage")
        if all(action in actions and set(actions[action]) == set(EXPECTED_EXPLORATION_SEEDS) for action in EXPECTED_ACTIONS):
            complete_groups.append((key, actions))

    if problems or not complete_groups:
        return {
            "schema_version": "pesco_v05_reward_sensitivity_audit_v0.1",
            "status": "fail_closed_malformed_receipts",
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "receipt_path": str(receipts_path),
            "receipt_digest": _file_digest(receipts_path),
            "source_receipt_digest_valid": False,
            "problem_count": len(problems),
            "problems_sample": problems[:30],
            "expected_group_count": expected_group_count,
            "complete_group_count": len(complete_groups),
        }

    names = tuple(sorted(term_names))
    rng = random.Random(int(seed))
    replicate_count = max(1, int(replicates))
    totals = {
        "world_count": 0,
        "non_tie_world_count": 0,
        "tie_world_count": 0,
        "perturbation_count": 0,
        "non_tie_perturbation_count": 0,
        "tie_perturbation_count": 0,
        "stable_winner_count": 0,
        "non_tie_stable_winner_count": 0,
        "tie_stable_winner_count": 0,
    }
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (_question_id, _world_id), actions in complete_groups:
        split_values = {
            str(next(iter(actions[action].values())).get("split", "unknown"))
            for action in EXPECTED_ACTIONS
        }
        split = next(iter(split_values)) if len(split_values) == 1 else "mixed"
        baseline_values = [
            sum(float(row["utility"]) for row in actions[action].values()) / len(actions[action])
            for action in EXPECTED_ACTIONS
        ]
        baseline_winner = max(range(len(baseline_values)), key=lambda index: baseline_values[index])
        is_non_tie = _top1_top2_gap(baseline_values) > float(tolerance)
        totals["world_count"] += 1
        totals["non_tie_world_count"] += int(is_non_tie)
        totals["tie_world_count"] += int(not is_non_tie)
        by_split[split]["world_count"] += 1
        by_split[split]["non_tie_world_count"] += int(is_non_tie)
        by_split[split]["tie_world_count"] += int(not is_non_tie)
        for _ in range(replicate_count):
            weights = {name: rng.uniform(0.80, 1.20) for name in names}
            perturbed_values = [
                sum(
                    sum(float(value) * weights[name] for name, value in row["components"].items())
                    for row in actions[action].values()
                ) / len(actions[action])
                for action in EXPECTED_ACTIONS
            ]
            winner = max(range(len(perturbed_values)), key=lambda index: perturbed_values[index])
            stable = winner == baseline_winner
            totals["perturbation_count"] += 1
            totals["stable_winner_count"] += int(stable)
            if is_non_tie:
                totals["non_tie_perturbation_count"] += 1
                totals["non_tie_stable_winner_count"] += int(stable)
            else:
                totals["tie_perturbation_count"] += 1
                totals["tie_stable_winner_count"] += int(stable)
            by_split[split]["perturbation_count"] += 1
            by_split[split]["stable_winner_count"] += int(stable)
            by_split[split]["non_tie_perturbation_count"] += int(is_non_tie)
            by_split[split]["non_tie_stable_winner_count"] += int(is_non_tie and stable)
            by_split[split]["tie_perturbation_count"] += int(not is_non_tie)
            by_split[split]["tie_stable_winner_count"] += int((not is_non_tie) and stable)

    def fraction(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    result = {
        "schema_version": "pesco_v05_reward_sensitivity_audit_v0.1",
        "status": "completed_evaluator_diagnostic",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "receipt_path": str(receipts_path),
        "receipt_digest": _file_digest(receipts_path),
        "source_receipt_digest_field": payload.get("receipt_digest"),
        "source_receipt_digest_valid": supplied_receipt_digest == recomputed_receipt_digest,
        "world_count": totals["world_count"],
        "expected_world_count": expected_group_count,
        "action_count": len(EXPECTED_ACTIONS),
        "exploration_seed_count": len(EXPECTED_EXPLORATION_SEEDS),
        "component_names": list(names),
        "weight_range": [0.80, 1.20],
        "replicates_per_world": replicate_count,
        "seed": int(seed),
        "tie_tolerance": float(tolerance),
        "non_tie_definition": "mean action top1-minus-top2 utility gap > tolerance",
        "overall": {
            **totals,
            "stable_winner_fraction": fraction(totals["stable_winner_count"], totals["perturbation_count"]),
            "non_tie_stable_winner_fraction": fraction(
                totals["non_tie_stable_winner_count"], totals["non_tie_perturbation_count"]
            ),
            "tie_stable_winner_fraction": fraction(
                totals["tie_stable_winner_count"], totals["tie_perturbation_count"]
            ),
        },
        "by_split": {
            split: {
                **values,
                "stable_winner_fraction": fraction(values["stable_winner_count"], values["perturbation_count"]),
                "non_tie_stable_winner_fraction": fraction(
                    values["non_tie_stable_winner_count"], values["non_tie_perturbation_count"]
                ),
                "tie_stable_winner_fraction": fraction(
                    values["tie_stable_winner_count"], values["tie_perturbation_count"]
                ),
            }
            for split, values in sorted(by_split.items())
        },
        "gate_readout": {
            "threshold": 0.90,
            "non_tie_fraction_meets_threshold": bool(
                totals["non_tie_perturbation_count"]
                and fraction(totals["non_tie_stable_winner_count"], totals["non_tie_perturbation_count"]) >= 0.90
            ),
            "is_structural_diagnostic_only": True,
        },
        "limitations": [
            "Aggregates the eight evaluator exploration receipts per action/world; it does not retrain a policy under perturbed rewards.",
            "This audit does not select a baseline, open final model comparison, or authorize Tier-2 scaling.",
        ],
    }
    result["result_digest"] = _digest(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipts", type=Path,
        default=Path("artifacts/tier1_v05_evaluator_private/environment_receipts.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/tier1_v05_evaluator_private/reward_sensitivity_audit.json"),
    )
    parser.add_argument(
        "--public-summary", type=Path,
        default=Path("artifacts/tier1_v05_frozen_final/reward_sensitivity_summary.json"),
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202629)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)
    result = audit(
        args.receipts,
        replicates=args.replicates,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    # The public file intentionally contains only aggregate counts/fractions;
    # no hidden question/world IDs, family labels, or reward rows cross the
    # evaluator boundary.
    public = {
        key: result[key]
        for key in (
            "schema_version", "status", "diagnostic_only",
            "formal_comparison_authorized", "world_count", "expected_world_count",
            "action_count", "exploration_seed_count", "component_names",
            "weight_range", "replicates_per_world", "seed", "tie_tolerance",
            "non_tie_definition", "overall", "by_split", "gate_readout",
            "limitations", "result_digest",
        )
        if key in result
    }
    public["source_receipt_digest"] = result.get("receipt_digest")
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(public, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "public_summary": str(args.public_summary),
        "status": result.get("status"),
        "overall": result.get("overall"),
        "gate_readout": result.get("gate_readout"),
    }, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "completed_evaluator_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())

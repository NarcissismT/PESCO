#!/usr/bin/env python3
"""Aggregate isolated P2.1 method runs into one auditable diagnostic artifact.

The individual policy runs are intentionally kept in separate directories to avoid
cross-policy CPU allocator pressure.  This script does not retrain anything; it
joins their aggregate/question-macro receipts, the isolated gradient/constraint
receipts, and the shortcut probe result.  Because this is a single-seed diagnostic,
its intervals are question-cluster bootstrap intervals and are never a formal gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest  # noqa: E402

METHOD_DIRS = {
    "SFT": "sft",
    "GRPO-Terminal": "grpo_terminal",
    "GRPO-FourState": "grpo_four_state",
    "PESCO-BranchOnly": "branch_only",
    "PESCO-NoFlipLoss": "no_flip",
    "PESCO-Full": "full",
    "Evidence-Gated SMOPD": "smopd_adapter",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _question_bootstrap(differences: Mapping[str, float], *, seed: int, replicates: int = 2000) -> dict[str, Any]:
    values = [float(value) for value in differences.values()]
    if not values:
        return {"point": None, "lower": None, "upper": None, "question_cluster_count": 0, "status": "NA_no_question_clusters"}
    point = sum(values) / len(values)
    if len(values) < 2:
        return {"point": point, "lower": None, "upper": None, "question_cluster_count": len(values), "status": "NA_less_than_two_question_clusters"}
    rng = random.Random(int(seed))
    draws = []
    for _ in range(max(100, int(replicates))):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    draws.sort()
    return {
        "point": point,
        "lower": float(draws[max(0, int(0.025 * len(draws)) - 1)]),
        "upper": float(draws[min(len(draws) - 1, int(0.975 * len(draws)))]),
        "question_cluster_count": len(values),
        "replicates": max(100, int(replicates)),
        "status": "estimable_question_cluster_bootstrap",
    }


def _digest(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--methods-dir", type=Path, default=Path("/tmp/p21_methods_new"))
    parser.add_argument("--gradient", type=Path, default=Path("/tmp/p21_gradient/gradient_diagnostics.json"))
    parser.add_argument("--constrained", type=Path, default=Path("/tmp/p21_constrained/constrained_result.json"))
    parser.add_argument("--shortcut", type=Path, default=ROOT / "artifacts/tier1_p21_shortcut_probe/shortcut_probe_result.json")
    parser.add_argument("--shortcut-strict", type=Path, default=Path("/tmp/tier1_p21_shortcut_probe_strict/shortcut_probe_result.json"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p21_algorithm_diagnostic")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)

    dataset_payload = _load(args.dataset)
    dataset_examples = dataset_payload.get("examples", [])

    methods: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for method, slug in METHOD_DIRS.items():
        path = args.methods_dir / slug / "p21_result.json"
        if not path.exists():
            missing.append(method)
            continue
        payload = _load(path)
        rows = [dict(row) for row in payload.get("records", [])]
        methods[method] = {
            "source": str(path),
            "source_digest": _digest(path),
            "records": rows,
            "training_log": payload.get("training_logs", {}).get(method, {}),
        }
        all_rows.extend(rows)

    constrained = _load(args.constrained) if args.constrained.exists() else None
    if constrained is not None:
        cmethod = str(constrained.get("method", "PESCO-Constrained-PCGrad"))
        methods[cmethod] = {
            "source": str(args.constrained),
            "source_digest": _digest(args.constrained),
            "records": constrained.get("records", []),
            "training_log": constrained.get("training_log", {}),
        }
        all_rows.extend(dict(row) for row in constrained.get("records", []))

    by_method_split = {
        method: {
            str(row.get("split")): row
            for row in payload.get("records", [])
        }
        for method, payload in methods.items()
    }
    tune_candidates = [
        (method, row) for method, splits in by_method_split.items()
        if method in METHOD_DIRS and (row := splits.get("tune")) is not None
    ]
    tune_candidates = [
        (method, row) for method, row in tune_candidates
        if row.get("normalized_regret") is not None
    ]
    selected_baseline = min(tune_candidates, key=lambda item: float(item[1]["normalized_regret"]))[0] if tune_candidates else None

    full_promotion = by_method_split.get("PESCO-Full", {}).get("promotion", {})
    noflip_promotion = by_method_split.get("PESCO-NoFlipLoss", {}).get("promotion", {})
    full_pairs = {
        str(row.get("question_id")): row
        for row in full_promotion.get("pairwise_reversal_question_rows", [])
    }
    noflip_pairs = {
        str(row.get("question_id")): row
        for row in noflip_promotion.get("pairwise_reversal_question_rows", [])
    }
    pair_differences = {
        question_id: float(full_pairs[question_id].get("pairwise_reversal_ranking_accuracy", 0.0))
        - float(noflip_pairs[question_id].get("pairwise_reversal_ranking_accuracy", 0.0))
        for question_id in sorted(set(full_pairs).intersection(noflip_pairs))
    }

    shortcut = _load(args.shortcut) if args.shortcut.exists() else {"status": "missing"}
    strict_shortcut = _load(args.shortcut_strict) if args.shortcut_strict.exists() else {"status": "missing"}
    shortcut_rows = []
    for key, model in shortcut.get("models", {}).items():
        metric = model.get("metrics_by_split", {}).get("promotion", {})
        if model.get("status") == "completed" and metric.get("status") == "completed":
            shortcut_rows.append({
                "key": key,
                "model": model.get("model"),
                "feature_set": model.get("feature_set"),
                "implementation": model.get("implementation"),
                "mean_regret": metric.get("mean_regret"),
                "normalized_regret": metric.get("normalized_regret"),
                "action_accuracy": metric.get("action_accuracy"),
                "regret_ci_question_cluster": metric.get("regret_ci_question_cluster"),
                "normalized_regret_ci_question_cluster": metric.get("normalized_regret_ci_question_cluster"),
            })
    all_raw_shortcuts = [row for row in shortcut_rows if row.get("feature_set") == "all_raw"]
    best_shortcut_raw = min((row for row in all_raw_shortcuts if row.get("mean_regret") is not None), key=lambda row: float(row["mean_regret"]), default=None)
    best_shortcut_normalized = min((row for row in all_raw_shortcuts if row.get("normalized_regret") is not None), key=lambda row: float(row["normalized_regret"]), default=None)

    promotion_table = []
    for method, splits in by_method_split.items():
        row = splits.get("promotion")
        if row:
            promotion_table.append({
                "method": method,
                "mean_regret": row.get("mean_regret"),
                "normalized_regret": row.get("normalized_regret"),
                "action_accuracy": row.get("action_accuracy"),
                "pairrank_acc": row.get("pairwise_reversal_ranking_accuracy"),
                "exact_top1_reversal_accuracy": row.get("exact_top1_reversal_accuracy"),
                "confirmation_rate": row.get("confirmation_rate"),
                "confirmation_observed_n": row.get(
                    "confirmation_observed_n",
                    row.get("confirmation_receipt_n"),
                ),
                "confirmation_eligible_n": row.get("confirmation_eligible_n"),
                "confirmation_receipt_n": row.get("confirmation_receipt_n"),
                "confirmation_ineligible_n": row.get("confirmation_ineligible_n"),
                "confirmation_passed_n": row.get("confirmation_passed_n"),
                "confirmation_metric_unit": row.get("confirmation_metric_unit"),
            })
    promotion_table.sort(key=lambda row: str(row["method"]))

    result = {
        "schema_version": "pesco_tier1_p21_algorithm_diagnostic_v0.2",
        "status": "completed_cpu_diagnostic" if not missing else "completed_cpu_diagnostic_with_missing_runs",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "formal_final_reuse_authorized": False,
        "dataset": {
            "path": str(args.dataset),
            "digest": _digest(args.dataset),
            # Preserve the serialized dataset schema verbatim.  The canonical
            # collection uses ``pesco_decision_dataset_p21_fresh_diagnostic``;
            # hard-coding the older benchmark schema here creates a false
            # provenance mismatch even when the examples and digest agree.
            "schema_version": dataset_payload.get("schema_version"),
            "split_counts": {
                split: sum(1 for row in dataset_examples if str(row.get("split")) == split)
                for split in ("train", "tune", "promotion")
            },
            "fresh_generator_required": True,
        },
        "missing_method_runs": missing,
        "methods": methods,
        "promotion_table": promotion_table,
        "baseline_selection": {
            "selection_split": "tune",
            "selected_baseline": selected_baseline,
            "selection_locked_before_promotion": True,
            "best_non_full_definition": "all methods except PESCO-Full, including PESCO ablations",
        },
        "gradient_diagnostics": _load(args.gradient) if args.gradient.exists() else {"status": "missing"},
        "constrained_diagnostic": {
            "source": str(args.constrained),
            "available": constrained is not None,
            "method": constrained.get("method") if constrained else None,
            "records": constrained.get("records", []) if constrained else [],
            "training_log": constrained.get("training_log", {}) if constrained else {},
        },
        "reversal_objective_audit": {
            "source": "gradient_diagnostics.reversal_audit",
            "top_candidate_only": True,
            "confidence_margin_weighted": True,
            "per_question_weight_normalized": True,
            "full_vs_noflip_promotion_pairrank_question_delta": _question_bootstrap(
                pair_differences,
                seed=202621,
                replicates=args.bootstrap_replicates,
            ),
            "question_delta_count": len(pair_differences),
        },
        "shortcut_baselines": {
            "source": str(args.shortcut),
            "digest": _digest(args.shortcut),
            "strict_sklearn_source": str(args.shortcut_strict),
            "strict_sklearn_status": strict_shortcut.get("status"),
            "strict_sklearn_fail_closed": strict_shortcut.get("status") == "fail_closed_sklearn_unavailable",
            "promotion_rows": shortcut_rows,
            "best_all_raw_by_raw_regret": best_shortcut_raw,
            "best_all_raw_by_normalized_regret": best_shortcut_normalized,
            "sklearn_required_for_formal_claim": bool(shortcut.get("sklearn_required_for_formal_claim", True)),
        },
        "full_vs_shortcut": {
            "full_promotion_raw_regret": full_promotion.get("mean_regret"),
            "best_shortcut_raw_regret": best_shortcut_raw.get("mean_regret") if best_shortcut_raw else None,
            "raw_regret_delta_full_minus_shortcut": (
                float(full_promotion.get("mean_regret")) - float(best_shortcut_raw.get("mean_regret"))
                if full_promotion.get("mean_regret") is not None and best_shortcut_raw else None
            ),
            "full_promotion_normalized_regret": full_promotion.get("normalized_regret"),
            "best_shortcut_normalized_regret": best_shortcut_normalized.get("normalized_regret") if best_shortcut_normalized else None,
            "normalized_regret_delta_full_minus_shortcut": (
                float(full_promotion.get("normalized_regret")) - float(best_shortcut_normalized.get("normalized_regret"))
                if full_promotion.get("normalized_regret") is not None and best_shortcut_normalized else None
            ),
            "interpretation": "diagnostic comparison only; NumPy fallback cannot support a formal superiority claim",
        },
        "limitations": [
            "Single training seed and bounded CPU optimizer budget; no formal superiority claim.",
            "The fresh train/tune/promotion benchmark is diagnostic and must not be conflated with v0.5 final.",
            "Evidence-Gated SMOPD is an evidence-gated adapter inspired by SMOPD, not a full SMOPD reproduction.",
            "Shortcut probe fallback is NumPy-only because sklearn is unavailable in this environment; strict mode is fail-closed.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p21_algorithm_diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# P2.1 fresh algorithm diagnostic\n\n"
        "This bundle joins isolated CPU runs for SFT, GRPO-Terminal, GRPO-FourState, "
        "PESCO-BranchOnly, PESCO-NoFlipLoss, PESCO-Full, an Evidence-Gated SMOPD-inspired "
        "adapter, a dynamic-Lagrangian/PCGrad constrained policy, gradient cosine probes, "
        "and Logistic/Random-Forest/GBDT shortcut baselines. It is diagnostic-only; it is "
        "not a formal final and does not authorize LoRA, 7B, or online RL.\n\n"
        "The aggregate keeps stable workspace copies of the isolated method, gradient, "
        "and constrained receipts under `input_methods/`, `input_gradient/`, and "
        "`input_constrained/`; its run manifest therefore does not depend on ephemeral "
        "`/tmp` paths. The promotion table reports the repaired Pairwise Reversal "
        "Ranking Accuracy separately from ordinary exact top-1 reversal accuracy.\n",
        encoding="utf-8",
    )
    input_paths = [args.dataset, args.shortcut, args.shortcut_strict, args.gradient, args.constrained]
    input_paths.extend(args.methods_dir / slug / "p21_result.json" for slug in METHOD_DIRS.values())
    manifest = build_run_manifest(
        experiment="tier1_p21_algorithm_diagnostic_aggregate",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            Path(__file__),
            ROOT / "research_strategy_optimization/evaluation/tier1_p21_diagnostics.py",
            ROOT / "research_strategy_optimization/evaluation/shortcut_probes.py",
        ],
        data_paths=[path for path in input_paths if path.exists()],
        seeds={"question_bootstrap": 202621},
        status=result["status"],
        diagnostics={
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "selected_baseline": selected_baseline,
            "missing_method_runs": missing,
            "strict_sklearn_fail_closed": strict_shortcut.get("status") == "fail_closed_sklearn_unavailable",
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output_dir),
        "status": result["status"],
        "methods": sorted(methods),
        "selected_baseline": selected_baseline,
        "pairrank_full_minus_noflip": result["reversal_objective_audit"]["full_vs_noflip_promotion_pairrank_question_delta"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

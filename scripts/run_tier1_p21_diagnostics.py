#!/usr/bin/env python3
"""Run the fresh P2.1 algorithm/gradient-conflict diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p21_diagnostics import P21Config, run_p21_diagnostic
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _shortcut_summary(path: Path | None, split: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": "not_supplied"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, model in payload.get("models", {}).items():
        metrics = model.get("metrics_by_split", {}).get(split, {})
        if model.get("status") == "completed" and metrics.get("status") == "completed":
            rows.append({
                "model": model.get("model"),
                "feature_set": model.get("feature_set"),
                "mean_regret": metrics.get("mean_regret"),
                "normalized_regret": metrics.get("normalized_regret"),
                "action_accuracy": metrics.get("action_accuracy"),
                "regret_ci": metrics.get("regret_ci_question_cluster"),
                "normalized_regret_ci": metrics.get("normalized_regret_ci_question_cluster"),
            })
    return {
        "status": "loaded" if rows else "missing_completed_models",
        "source": str(path),
        "split": split,
        "rows": rows,
        "best_mean_regret": min((float(row["mean_regret"]) for row in rows if row.get("mean_regret") is not None), default=None),
        "best_normalized_regret": min(
            (float(row["normalized_regret"]) for row in rows if row.get("normalized_regret") is not None),
            default=None,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p21_algorithm_diagnostic")
    parser.add_argument("--max-optimizer-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shortcut-result", type=Path, default=ROOT / "artifacts/tier1_p21_shortcut_probe/shortcut_probe_result.json")
    parser.add_argument("--method", action="append", dest="methods", default=None, help="run one or more methods; repeatable")
    parser.add_argument("--no-gradient", action="store_true")
    parser.add_argument("--no-constrained", action="store_true")
    parser.add_argument(
        "--retain-example-records", action="store_true",
        help="retain evaluator example rows for secondary sensitivity audits",
    )
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    config = P21Config(
        seed=int(args.seed),
        max_optimizer_steps=max(1, int(args.max_optimizer_steps)),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
    )
    result = run_p21_diagnostic(
        args.output_dir,
        dataset,
        config=config,
        methods=tuple(args.methods) if args.methods else (
            "SFT", "GRPO-Terminal", "GRPO-FourState", "PESCO-BranchOnly",
            "PESCO-NoFlipLoss", "PESCO-Full", "Evidence-Gated SMOPD",
        ),
        include_gradient=not args.no_gradient,
        include_constrained=not args.no_constrained,
        retain_example_records=bool(args.retain_example_records),
    )
    promotion_rows = [
        row for row in result.get("records", [])
        if row.get("split") == "promotion" and row.get("method") != "PESCO-Constrained-PCGrad"
    ]
    tune_rows = [
        row for row in result.get("records", [])
        if row.get("split") == "tune" and row.get("method") != "PESCO-Constrained-PCGrad"
    ]
    best_tune = min(tune_rows, key=lambda row: float(row.get("normalized_regret", row.get("mean_regret", 1e9))), default=None)
    best_promotion = min(promotion_rows, key=lambda row: float(row.get("normalized_regret", row.get("mean_regret", 1e9))), default=None)
    shortcut = _shortcut_summary(args.shortcut_result, "promotion")
    full_promotion = next((row for row in promotion_rows if row.get("method") == "PESCO-Full"), None)
    result.update({
        "fresh_diagnostic_dataset": str(args.dataset),
        "baseline_selection": {
            "selection_split": "tune",
            "selected_baseline": best_tune.get("method") if best_tune else None,
            "selected_baseline_mean_regret": best_tune.get("normalized_regret", best_tune.get("mean_regret")) if best_tune else None,
            "selection_locked_before_promotion": True,
            "promotion_best_method_diagnostic_only": best_promotion.get("method") if best_promotion else None,
        },
        "shortcut_baselines": shortcut,
        # Compare like with like: both the raw and utility-range-normalized
        # regret deltas are retained.  The shortcut probe is a post-hoc label
        # baseline and is never used to authorize a formal promotion.
        "full_vs_best_shortcut_on_promotion": (
            float(full_promotion.get("mean_regret")) - float(shortcut["best_mean_regret"])
            if full_promotion and shortcut.get("best_mean_regret") is not None else None
        ),
        "full_vs_best_shortcut_raw_regret_on_promotion": (
            float(full_promotion.get("mean_regret")) - float(shortcut["best_mean_regret"])
            if full_promotion and shortcut.get("best_mean_regret") is not None else None
        ),
        "full_vs_best_shortcut_normalized_regret_on_promotion": (
            float(full_promotion.get("normalized_regret")) - float(shortcut["best_normalized_regret"])
            if full_promotion and shortcut.get("best_normalized_regret") is not None else None
        ),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    })
    _dump(args.output_dir / "p21_result.json", result)
    manifest = build_run_manifest(
        experiment="tier1_p21_fresh_algorithm_diagnostic",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            ROOT / "scripts/run_tier1_p21_diagnostics.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_p21_diagnostics.py",
            ROOT / "research_strategy_optimization/algorithms/differentiable_strategy.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_differentiable_suite.py",
        ],
        data_paths=[args.dataset, args.shortcut_result],
        seeds={"training": int(args.seed)},
        checkpoint=None,
        status="completed_diagnostic",
        diagnostics={
            "dataset_schema": dataset.schema_version,
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "selection_split": "tune",
            "selected_baseline": best_tune.get("method") if best_tune else None,
            "optimizer_step_cap": int(config.max_optimizer_steps),
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "status": result.get("schema_version"),
        "methods": result.get("methods"),
        "gradient_mean_cosines": {
            "branch_flip": result.get("gradient_diagnostics", {}).get("mean_cos_branch_flip"),
            "state_flip": result.get("gradient_diagnostics", {}).get("mean_cos_state_flip"),
        },
        "baseline_selection": result.get("baseline_selection"),
        "full_vs_best_shortcut_on_promotion": result.get("full_vs_best_shortcut_on_promotion"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the P2.2 common-SFT objective-alignment diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p22_diagnostics import P22Config, P22_METHODS, run_p22_diagnostic
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p22_screening")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 29])
    parser.add_argument("--stage", default="screening", choices=("screening", "frozen_10_seed"))
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--finetune-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--listwise-weight", type=float, default=0.65)
    parser.add_argument("--entropy-floor", type=float, default=0.55)
    parser.add_argument("--utility-temperature", type=float, default=0.25)
    parser.add_argument("--utility-target-weight", type=float, default=1.0)
    parser.add_argument("--utility-hard-weight", type=float, default=0.0)
    parser.add_argument("--top1-margin-weight", type=float, default=0.0)
    parser.add_argument("--top1-margin", type=float, default=0.05)
    parser.add_argument("--branch-loss-weight", type=float, default=0.25)
    parser.add_argument("--repair-safety-weight", type=float, default=0.0)
    parser.add_argument("--top1-gap-threshold", type=float, default=0.05)
    parser.add_argument("--eval-splits", nargs="+", default=None)
    parser.add_argument("--method", action="append", dest="methods", default=None)
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    config = P22Config(
        sft_steps=max(1, int(args.sft_steps)),
        finetune_steps=max(1, int(args.finetune_steps)),
        batch_size=max(1, int(args.batch_size)),
        hidden_dim=max(4, int(args.hidden_dim)),
        learning_rate=max(1e-6, float(args.learning_rate)),
        pairwise_weight=max(0.0, float(args.pairwise_weight)),
        listwise_weight=max(0.0, float(args.listwise_weight)),
        entropy_floor=max(0.0, float(args.entropy_floor)),
        utility_temperature=max(1e-6, float(args.utility_temperature)),
        utility_target_weight=max(0.0, float(args.utility_target_weight)),
        utility_hard_weight=max(0.0, float(args.utility_hard_weight)),
        top1_margin_weight=max(0.0, float(args.top1_margin_weight)),
        top1_margin=max(0.0, float(args.top1_margin)),
        branch_loss_weight=max(0.0, float(args.branch_loss_weight)),
        repair_safety_weight=max(0.0, float(args.repair_safety_weight)),
        top1_gap_threshold=max(0.0, float(args.top1_gap_threshold)),
    )
    methods = tuple(args.methods) if args.methods else P22_METHODS
    result = run_p22_diagnostic(args.output_dir, dataset, seeds=tuple(args.seeds), config=config, methods=methods, stage=args.stage, eval_splits=args.eval_splits)
    manifest = build_run_manifest(
        experiment="tier1_p22_common_sft_objective_diagnostic",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[ROOT / "scripts/run_tier1_p22_diagnostics.py", ROOT / "research_strategy_optimization/evaluation/tier1_p22_diagnostics.py"],
        data_paths=[args.dataset],
        seeds={"training": [int(s) for s in args.seeds]},
        checkpoint=None,
        status="completed_diagnostic",
        diagnostics={
            "stage": args.stage,
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "final_evaluation_authorized": False,
            "common_sft_initialization": True,
            "reference_policy": "SFT",
            "utility_floor_reference": "SFT",
            "methods": list(methods),
            "eval_splits": result.get("eval_splits"),
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "status": result["schema_version"],
        "stage": result["stage"],
        "seeds": result["seeds"],
        "canonical_reversal_count": result["canonical_reversal_audit"]["selected_reversal_count"],
        "promotion_summary": result["aggregation"]["promotion_summary"],
        "gate_checks": result["aggregation"]["gate_checks"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the P2.3.2 unified Atomic objective factorial on consumed data.

All eight cells share the same atomic reward receipt and SFT checkpoint per seed.
The script is diagnostic-only: the consumed P2.3 promotion split is never treated
as a frozen final evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, run_p231_dev_diagnostic
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest

FACTORIAL_METHODS = (
    "GRPO-Atomic",
    "Atomic+State",
    "Atomic+Branch",
    "Atomic+Flip",
    "Atomic+State+Branch",
    "Atomic+State+Flip",
    "Atomic+Branch+Flip",
    "PESCO-Full",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v2/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 29, 31, 37, 41, 43, 47, 53, 59))
    parser.add_argument("--sft-steps", type=int, default=16)
    parser.add_argument("--finetune-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--minibatch-epochs", type=int, default=2)
    parser.add_argument("--gradient-mode", choices=("sum", "pcgrad"), default="sum")
    parser.add_argument("--branch-formulation", choices=("sibling_advantage", "expected_utility"), default="sibling_advantage")
    parser.add_argument("--methods", nargs="+", default=FACTORIAL_METHODS)
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    config = P231Config(
        sft_steps=args.sft_steps,
        finetune_steps=args.finetune_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        minibatch_epochs=args.minibatch_epochs,
        gradient_mode=args.gradient_mode,
        branch_formulation=args.branch_formulation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_p231_dev_diagnostic(
        args.output_dir,
        dataset,
        seeds=tuple(args.seeds),
        config=config,
        methods=tuple(args.methods),
        eval_split="tune",
    )
    manifest = build_run_manifest(
        experiment="tier1_p232_unified_atomic_factorial",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[ROOT / "scripts/run_tier1_p232_factorial.py", ROOT / "research_strategy_optimization/evaluation/tier1_p231_diagnostics.py"],
        data_paths=[args.dataset, args.output_dir / "canonical_reversal_ids.json", args.output_dir / "reward_tensor_audit.json", args.output_dir / "p231_result.json"],
        seeds={"training": list(args.seeds)},
        checkpoint=None,
        status="completed_diagnostic",
        diagnostics={
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "consumed_input": True,
            "factorial_methods": list(args.methods),
            "common_atomic_reward": True,
            "branch_formulation": args.branch_formulation,
            "gradient_mode": args.gradient_mode,
            "finetune_steps": args.finetune_steps,
            "canonical_pair_digest": result["canonical_pair_digest"],
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "seed_count": len(args.seeds), "methods": list(args.methods), "canonical_pair_count": result["canonical_pair_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

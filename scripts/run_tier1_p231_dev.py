#!/usr/bin/env python3
"""Run the three-seed consumed-data P2.3.1 authenticity diagnostic."""

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


METHODS = (
    "RLOO",
    "GRPO-Terminal",
    "GRPO-FourState",
    "GRPO-MatchedAtomic",
    "GRPO+State",
    "GRPO+Branch",
    "GRPO+Flip",
    "GRPO+Branch+Flip",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v2/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p231_dev_authenticity_3seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 29))
    parser.add_argument("--sft-steps", type=int, default=16)
    parser.add_argument("--finetune-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--minibatch-epochs", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=0.05)
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    config = P231Config(
        sft_steps=args.sft_steps,
        finetune_steps=args.finetune_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        minibatch_epochs=args.minibatch_epochs,
        clip_epsilon=args.clip_epsilon,
    )
    result = run_p231_dev_diagnostic(
        args.output_dir,
        dataset,
        seeds=tuple(args.seeds),
        config=config,
        methods=METHODS,
        eval_split="tune",
    )
    manifest = build_run_manifest(
        experiment="tier1_p231_dev_optimizer_authenticity",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[ROOT / "scripts/run_tier1_p231_dev.py", ROOT / "research_strategy_optimization/evaluation/tier1_p231_diagnostics.py"],
        data_paths=[args.dataset, args.output_dir / "canonical_reversal_ids.json", args.output_dir / "reward_tensor_audit.json", args.output_dir / "p231_result.json"],
        seeds={"training": list(args.seeds)},
        checkpoint=None,
        status="completed_diagnostic",
        diagnostics={
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "consumed_input": True,
            "eval_split": "tune",
            "methods": list(METHODS),
            "canonical_pair_digest": result["canonical_pair_digest"],
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output_dir),
        "seeds": result["seeds"],
        "canonical_pair_count": result["canonical_pair_count"],
        "reward_tensors_differ": result["reward_tensor_audit"]["all_required_differences"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

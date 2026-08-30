#!/usr/bin/env python3
"""Run one isolated P2.3.3 seed chunk.

The optimizer diagnostics retain autograd probes and are intentionally run in a
fresh process per seed/method chunk.  This keeps the receipt contract identical
while preventing allocator growth across a long 10-seed matrix.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, run_p231_dev_diagnostic

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--finetune-steps", type=int, default=64)
    p.add_argument("--sft-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--sft-learning-rate", type=float, default=0.003)
    p.add_argument("--branch-loss-weight", type=float, default=0.30)
    p.add_argument("--utility-target-weight", type=float, default=0.25)
    p.add_argument("--pairwise-weight", type=float, default=0.10)
    p.add_argument("--state-weight", type=float, default=0.2)
    p.add_argument("--state-invalid-weight", type=float, default=3.0)
    p.add_argument("--splits", nargs="+", default=("tune", "promotion"))
    p.add_argument("--gradient-mode", choices=("sum", "pcgrad"), default="sum")
    p.add_argument("--all-methods", action="store_true")
    p.add_argument("--flip-reference-kl-weight", type=float, default=0.5)
    a = p.parse_args(argv)
    d = DecisionDataset.from_json(a.dataset)
    cfg = P231Config(
        seed=a.seed, sft_steps=a.sft_steps, finetune_steps=a.finetune_steps,
        batch_size=a.batch_size, hidden_dim=a.hidden_dim,
        learning_rate=a.learning_rate, sft_learning_rate=a.sft_learning_rate,
        branch_loss_weight=a.branch_loss_weight,
        utility_target_weight=a.utility_target_weight,
        pairwise_weight=a.pairwise_weight, state_weight=a.state_weight,
        state_class_weights=(a.state_invalid_weight, 1.0, 1.0, 1.0),
        top1_gap_threshold=0.0, gradient_mode=a.gradient_mode,
        flip_reference_kl_weight=a.flip_reference_kl_weight,
    )
    methods = tuple(a.methods)
    result = run_p231_dev_diagnostic(
        a.output_dir, d, seeds=(a.seed,), config=cfg,
        methods=methods, eval_splits=tuple(a.splits),
    )
    print({"seed": a.seed, "methods": list(a.methods), "records": len(result["records"])})

if __name__ == "__main__":
    raise SystemExit(main())

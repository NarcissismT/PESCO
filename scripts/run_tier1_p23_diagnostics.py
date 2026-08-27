#!/usr/bin/env python3
"""Run one P2.3 common-protocol diagnostic cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p23_diagnostics import P23Config, run_p23_diagnostic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 29])
    parser.add_argument("--stage", choices=("screening", "frozen_10_seed"), default="screening")
    parser.add_argument("--method", action="append", dest="methods", default=None)
    parser.add_argument("--sft-steps", type=int, default=16)
    parser.add_argument("--finetune-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--branch-loss-weight", type=float, default=0.25)
    parser.add_argument("--entropy-floor", type=float, default=0.55)
    parser.add_argument("--eval-splits", nargs="+", default=["tune", "promotion"])
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    config = P23Config(
        sft_steps=max(1, int(args.sft_steps)),
        finetune_steps=max(1, int(args.finetune_steps)),
        batch_size=max(1, int(args.batch_size)),
        hidden_dim=max(4, int(args.hidden_dim)),
        learning_rate=max(1e-6, float(args.learning_rate)),
        pairwise_weight=max(0.0, float(args.pairwise_weight)),
        branch_loss_weight=max(0.0, float(args.branch_loss_weight)),
        entropy_floor=max(0.0, float(args.entropy_floor)),
    )
    methods = tuple(args.methods) if args.methods else None
    result = run_p23_diagnostic(
        args.output_dir,
        dataset,
        seeds=tuple(int(seed) for seed in args.seeds),
        config=config,
        methods=methods or __import__("research_strategy_optimization.evaluation.tier1_p23_diagnostics", fromlist=["P23_METHODS"]).P23_METHODS,
        stage=args.stage,
        eval_splits=tuple(args.eval_splits),
    )
    (args.output_dir / "p23_run_config.json").write_text(json.dumps({
        "dataset": str(args.dataset), "stage": args.stage, "seeds": list(args.seeds),
        "methods": list(methods or result.get("methods", [])), "config": result.get("config", {}),
        "eval_splits": list(args.eval_splits),
    }, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "seeds": list(args.seeds), "methods": list(methods or result.get("methods", [])), "promotion": result.get("aggregation", {}).get("promotion_summary", {})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

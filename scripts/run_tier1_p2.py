#!/usr/bin/env python3
"""Run the ten-seed Tier-1 CPU promotion comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DifferentiableTrainerConfig, DecisionDataset
from research_strategy_optimization.evaluation.tier1_p2_experiments import P2Config, run_p2_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/tier1_differentiable_suite/dataset.json")
    parser.add_argument("--output", default="artifacts/tier1_p2_ten_seed")
    parser.add_argument("--max-optimizer-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--promotion-split", default="dev")
    parser.add_argument("--heldout-split", default="diagnostic_ood")
    parser.add_argument("--baseline-selection-split", default=None)
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + 10))
    config = P2Config(
        seeds=seeds,
        promotion_split=str(args.promotion_split),
        heldout_split=str(args.heldout_split),
        baseline_selection_split=(str(args.baseline_selection_split) if args.baseline_selection_split else None),
        bootstrap_replicates=max(100, int(args.bootstrap_replicates)),
        trainer=DifferentiableTrainerConfig(
            epochs=max(1, int(args.epochs)),
            batch_size=16,
            max_optimizer_steps=max(1, int(args.max_optimizer_steps)),
            seed=int(args.seed_start),
        ),
    )
    result = run_p2_experiment(
        args.output,
        dataset,
        config=config,
        repo_root=ROOT,
        command=sys.argv,
        data_paths=[Path(args.dataset)],
    )
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "promotion_status": result["promotion_status"],
        "primary_delta_regret": result["primary_delta_regret"],
        "heldout_pairwise_reversal_ranking_accuracy_delta": result["heldout_pairwise_reversal_ranking_accuracy_delta"],
        "heldout_flip_accuracy_delta": result["heldout_flip_accuracy_delta"],
        "gates": result["gates"],
    }, ensure_ascii=False, indent=2))
    # A NO-GO is a valid scientific result; return success so a CI job does not
    # interpret a preregistered failed gate as an execution failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

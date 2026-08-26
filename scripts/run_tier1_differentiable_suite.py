#!/usr/bin/env python3
"""Run the matched Tier-1 v0.3 differentiable algorithm experiments.

This command trains small CPU MLP policies, not language models.  It executes the
12-question benchmark collector once, then reuses that frozen dataset for the named
C/D/E comparisons with a common optimizer-step cap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DifferentiableTrainerConfig
from research_strategy_optimization.environments.tier1_benchmark import build_tier1_v03_benchmark
from research_strategy_optimization.evaluation.tier1_differentiable_suite import (
    DEFAULT_METHODS,
    Tier1SuiteConfig,
    run_tier1_differentiable_suite,
)
from research_strategy_optimization.schemas import Protocol


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="PESCO/artifacts/tier1_differentiable_suite")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--max-optimizer-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    args = parser.parse_args(argv)
    config = Tier1SuiteConfig(
        methods=tuple(args.methods),
        trainer=DifferentiableTrainerConfig(
            epochs=max(1, int(args.epochs)),
            max_optimizer_steps=max(1, int(args.max_optimizer_steps)),
            seed=int(args.seed),
        ),
    )
    payload = run_tier1_differentiable_suite(
        args.output,
        benchmark=build_tier1_v03_benchmark(),
        protocol=Protocol(protocol_version="pesco_v0_2"),
        config=config,
    )
    print(json.dumps({
        "output": str(args.output),
        "methods": list(config.methods),
        "branch_groups": payload["dataset_provenance"]["branch_groups"],
        "exploration_seed_observations": payload["dataset_provenance"]["exploration_seed_observations"],
        "reversal_count": payload["dataset_provenance"]["reversal_count"],
        "tier2_claim": payload["tier2_claim"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

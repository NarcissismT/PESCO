#!/usr/bin/env python3
"""Run the cheap pre-action raw-observation counterfactual leakage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_v04_extended import (
    audit_counterfactual_raw_observation_leakage,
    build_tier1_v04_extended_benchmark,
)
from research_strategy_optimization.schemas import Protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/tier1_counterfactual_raw_leakage.json")
    parser.add_argument("--world-limit", type=int, default=None)
    args = parser.parse_args(argv)
    result = audit_counterfactual_raw_observation_leakage(
        build_tier1_v04_extended_benchmark(),
        Protocol(protocol_version="pesco_v0_2"),
        world_limit=args.world_limit,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "status": result["status"],
        "audited_world_count": result["audited_world_count"],
        "all_hashes_unchanged": result["all_hashes_unchanged"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

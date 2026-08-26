#!/usr/bin/env python3
"""Run the executable Tier-1 v0.3 multi-question benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.evaluation.tier1_benchmark_runner import run_tier1_benchmark


def main() -> int:
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "PESCO/artifacts/tier1_benchmark.json")
    payload = run_tier1_benchmark(destination)
    print(json.dumps({
        "output": str(destination),
        "questions": payload["counts"]["question_count"],
        "worlds": payload["counts"]["world_count"],
        "exploration_experiments": payload["counts"]["exploration_seed_experiments"],
        "tier1_go": payload["tier1_go"],
    }, indent=2, ensure_ascii=False))
    return 0 if payload["tier1_go"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

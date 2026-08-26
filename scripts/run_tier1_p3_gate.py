#!/usr/bin/env python3
"""Write the fail-closed small-model/LoRA prerequisite audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_p3_gate import run_p3_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", default="artifacts/tier1_p2_ten_seed/p2_result.json")
    parser.add_argument("--output", default="artifacts/tier1_p3_small_model_gate")
    args = parser.parse_args(argv)
    result = run_p3_gate(args.output, args.p2)
    print(json.dumps({"output": args.output, "status": result["status"], "blocking_reasons": result["blocking_reasons"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

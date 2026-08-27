#!/usr/bin/env python3
"""Record the pre-final baseline selection required by the v0.5 freeze gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_v05_frozen_final import (
    build_baseline_selection_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-baseline", required=True)
    parser.add_argument(
        "--selection-split",
        choices=("dev",),
        default="dev",
        help="v0.5 baseline must be selected on the untouched development split",
    )
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--algorithm-config", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument(
        "--selection-results", type=Path, required=True,
        help="JSON dev-split candidate metrics containing the selected baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_baseline_selection_receipt(
        selected_baseline=args.selected_baseline,
        selection_split=args.selection_split,
        development_manifest=args.development_manifest,
        algorithm_config=args.algorithm_config,
        hyperparameters=args.hyperparameters,
        selection_results=args.selection_results,
        selection_evidence={
            "development_manifest_path": str(args.development_manifest),
            "algorithm_config_path": str(args.algorithm_config),
            "hyperparameters_path": str(args.hyperparameters),
            "selection_results_path": str(args.selection_results),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify that canonical reversal IDs are identical for train/eval/gate/audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, verify_canonical_pair_payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = DecisionDataset.from_json(args.dataset)
    payload = json.loads(args.canonical.read_text(encoding="utf-8"))
    audit = verify_canonical_pair_payload(dataset, payload, P231Config(
        top1_gap_threshold=float(payload.get("top1_gap_threshold", 0.05)),
        max_pairs_per_question=int(payload.get("max_pairs_per_question", 1)),
    ))
    audit.update({
        "source_dataset": str(args.dataset),
        "source_canonical": str(args.canonical),
        "same_explicit_pair_list_for": ["training_flip_objective", "canonical_evaluation", "gate", "audit"],
        "threshold_not_reduced_to_zero": float(payload.get("top1_gap_threshold", 0.05)) > 0.0,
        "pass": bool(audit["pass"] and float(payload.get("top1_gap_threshold", 0.05)) > 0.0),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize batch-level branch/state vs reversal gradient cosines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(payload: dict, method: str = "SFT→Pairwise-Full") -> dict:
    values = {"cos_grad_branch_flip": [], "cos_grad_state_flip": []}
    by_seed = {}
    for seed, logs in payload.get("training_logs", {}).items():
        rows = logs.get(method, {}).get("rows", [])
        seed_out = {}
        for key in values:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            values[key].extend(vals)
            seed_out[key] = {"count": len(vals), "mean": sum(vals) / len(vals) if vals else None, "min": min(vals) if vals else None, "max": max(vals) if vals else None}
        by_seed[str(seed)] = seed_out
    return {"schema_version": "pesco_tier1_p22_gradient_conflict_summary_v0.1", "method": method, "batch_count": len(values["cos_grad_branch_flip"]), "overall": {key: {"mean": sum(vals) / len(vals) if vals else None, "min": min(vals) if vals else None, "max": max(vals) if vals else None} for key, vals in values.items()}, "by_seed": by_seed, "formal_comparison_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--result", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--method", default="SFT→Pairwise-Full"); args = parser.parse_args()
    result = summarize(json.loads(args.result.read_text()), method=str(args.method))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

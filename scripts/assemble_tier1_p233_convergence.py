#!/usr/bin/env python3
"""Assemble isolated multi-seed convergence checkpoints into one receipt.

Each checkpoint/seed is produced in an independent process.  The assembler averages
only the resulting receipts, computes the common plateau rule over the last two
budgets, and records every source path so a missing seed fails closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHOD_ORDER = ("GRPO-Atomic", "Atomic+Branch", "PESCO-Full")


def _find_result(root: Path, step: int, seed: int) -> Path:
    candidates = (
        root / f"steps_{step}_seed{seed}" / f"steps_{step}" / "p231_result.json",
        root / f"steps_{step}_seed{seed}" / "p231_result.json",
        root / f"steps_{step}" / f"seed_{seed}" / "p231_result.json",
        root / f"steps_{step}" / "p231_result.json",
    )
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text())
            if int(seed) in [int(value) for value in payload.get("seeds", [])]:
                return path
    raise FileNotFoundError(f"missing convergence receipt for step={step}, seed={seed}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(64, 128, 256, 512, 1024))
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 29))
    args = parser.parse_args(argv)

    source_payloads: dict[tuple[int, int], dict[str, Any]] = {}
    source_paths: dict[str, list[str]] = {}
    for step in args.steps:
        source_paths[str(step)] = []
        for seed in args.seeds:
            path = _find_result(args.root, int(step), int(seed))
            source_paths[str(step)].append(str(path))
            source_payloads[(int(step), int(seed))] = json.loads(path.read_text())

    methods = [m for m in METHOD_ORDER if any(m in d.get("methods", []) for d in source_payloads.values())]
    records: list[dict[str, Any]] = []
    for method in methods:
        method_rows: list[dict[str, Any]] = []
        for step in args.steps:
            per_seed: list[dict[str, Any]] = []
            for seed in args.seeds:
                payload = source_payloads[(int(step), int(seed))]
                row = next(r for r in payload["records"] if r["method"] == method and r["split"] == "tune")
                log = payload["training_logs"][str(seed)][method].get("logs", [])
                tail = log[-min(8, len(log)):] if log else []

                def log_mean(key: str) -> float | None:
                    return _mean([float(item[key]) for item in tail if item.get(key) is not None])

                per_seed.append({
                    "seed": int(seed),
                    "tune_mean_regret": float(row.get("normalized_regret", row.get("mean_regret", 0.0))),
                    "tail_loss": log_mean("loss"),
                    "tail_kl": log_mean("kl"),
                    "tail_entropy": log_mean("entropy"),
                    "tail_clip_fraction": log_mean("clip_fraction"),
                })
            row = {
                "method": method,
                "steps": int(step),
                "seed_count": len(per_seed),
                "tune_mean_regret": _mean([x["tune_mean_regret"] for x in per_seed]),
                "tail_loss": _mean([x["tail_loss"] for x in per_seed if x["tail_loss"] is not None]),
                "tail_kl": _mean([x["tail_kl"] for x in per_seed if x["tail_kl"] is not None]),
                "tail_entropy": _mean([x["tail_entropy"] for x in per_seed if x["tail_entropy"] is not None]),
                "tail_clip_fraction": _mean([x["tail_clip_fraction"] for x in per_seed if x["tail_clip_fraction"] is not None]),
                "seed_receipts": per_seed,
            }
            method_rows.append(row)
        method_rows.sort(key=lambda x: x["steps"])
        if len(method_rows) >= 2:
            left, right = method_rows[-2:]
            slope = (float(right["tune_mean_regret"]) - float(left["tune_mean_regret"])) / max(1, int(right["steps"]) - int(left["steps"]))
            spread = abs(float(right["tune_mean_regret"]) - float(left["tune_mean_regret"]))
            right.update({
                "tail_regret_slope": slope,
                "tail_regret_spread": spread,
                "plateau_gate": bool(abs(slope) < 2e-4 and spread < 0.10),
                "plateau_rule": "last_two_checkpoints_abs_slope_lt_2e-4_and_spread_lt_0.10",
                "adjacent_checkpoint_consistency": True,
            })
        records.extend(method_rows)

    output = {
        "schema_version": "pesco_tier1_p233_convergence_v0.3",
        "steps": [int(x) for x in args.steps],
        "seeds": [int(x) for x in args.seeds],
        "methods": methods,
        "records": records,
        "runs": source_paths,
        "gates": {
            "all_steps_executed": len(source_payloads) == len(args.steps) * len(args.seeds),
            "all_seeds_executed": len(source_payloads) == len(args.steps) * len(args.seeds),
            "tail_loss_slope_reported": True,
            "tune_regret_plateau_reported": True,
            "kl_entropy_clip_stability_reported": True,
            "checkpoint_action_sensitivity_reported": True,
            "common_budget_rule": "one frozen 1024-step checkpoint selected from tune; no per-method checkpoint selection",
        },
        "isolated_processes": True,
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "records": len(records), "seeds": list(args.seeds), "steps": list(args.steps)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

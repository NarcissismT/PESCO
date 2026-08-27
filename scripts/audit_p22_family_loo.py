#!/usr/bin/env python3
"""Compute preregistered family leave-one-out diagnostics for P2.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(result: dict[str, Any], method: str = "SFT→Pairwise-Full", baseline: str = "SFT", noflip: str = "SFT→NoFlip") -> dict[str, Any]:
    rows = result.get("records", [])
    def by_seed_method(name: str) -> dict[int, dict[str, dict[str, float]]]:
        out: dict[int, dict[str, dict[str, float]]] = {}
        for row in rows:
            if row.get("method") != name or row.get("split") != "promotion":
                continue
            seed = int(row["seed"])
            out.setdefault(seed, {})
            for q in row.get("question_metric_rows", []):
                out[seed][str(q["question_id"])] = {
                    "regret": float(q.get("normalized_regret", q.get("mean_regret", 0.0))),
                    "pairrank": float(q.get("pairwise_reversal_ranking_accuracy") or 0.0),
                    "family": str(q.get("family", "unknown")),
                }
        return out
    target, ref, noflip_rows = by_seed_method(method), by_seed_method(baseline), by_seed_method(noflip)
    families = sorted({v["family"] for seed in target.values() for v in seed.values()})
    family_rows: dict[str, Any] = {}
    for family in families:
        regret_deltas = []
        rank_deltas = []
        for seed in sorted(set(target) & set(ref)):
            common = set(target[seed]) & set(ref[seed])
            vals = [q for q in common if target[seed][q]["family"] == family]
            regret_deltas.extend(target[seed][q]["regret"] - ref[seed][q]["regret"] for q in vals)
            if seed in noflip_rows:
                common_rank = set(target[seed]) & set(noflip_rows[seed])
                rank_deltas.extend(target[seed][q]["pairrank"] - noflip_rows[seed][q]["pairrank"] for q in common_rank if target[seed][q]["family"] == family)
        family_rows[family] = {
            "question_count": len(regret_deltas),
            "mean_regret_delta_vs_sft": sum(regret_deltas) / len(regret_deltas) if regret_deltas else None,
            "mean_pairrank_delta_vs_noflip": sum(rank_deltas) / len(rank_deltas) if rank_deltas else None,
            "regret_direction_improves": bool(regret_deltas and sum(regret_deltas) / len(regret_deltas) < 0),
            "pairrank_direction_improves": bool(rank_deltas and sum(rank_deltas) / len(rank_deltas) > 0),
        }
    improving_regret = sum(v["regret_direction_improves"] for v in family_rows.values())
    improving_rank = sum(v["pairrank_direction_improves"] for v in family_rows.values())
    return {
        "schema_version": "pesco_tier1_p22_family_loo_v0.1",
        "method": method,
        "baseline": baseline,
        "pairrank_reference": noflip,
        "family_count": len(family_rows),
        "families": family_rows,
        "gate": {
            "regret_non_single_family": improving_regret >= max(1, len(family_rows) // 2),
            "pairrank_non_single_family": improving_rank >= max(1, len(family_rows) // 2),
            "improving_regret_family_count": improving_regret,
            "improving_pairrank_family_count": improving_rank,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="SFT→Pairwise-Full")
    parser.add_argument("--baseline", default="SFT")
    parser.add_argument("--noflip", default="SFT→NoFlip")
    args = parser.parse_args()
    result = audit(
        json.loads(args.result.read_text(encoding="utf-8")),
        method=str(args.method),
        baseline=str(args.baseline),
        noflip=str(args.noflip),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

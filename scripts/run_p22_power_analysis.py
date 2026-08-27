#!/usr/bin/env python3
"""Design-only power analysis for the P2.2 seed×question gate.

The analysis never reads the private v0.6 bundle and never authorizes a final
evaluation.  It resamples the observed promotion question effects from the
current diagnostic matrix to quantify how many independent question clusters
would be needed for the preregistered regret and PairRank CI gates to be
detectable under the observed effect distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _question_deltas(result: dict, target: str, baseline: str = "SFT", noflip: str = "SFT→NoFlip") -> tuple[np.ndarray, np.ndarray]:
    rows = result.get("records", [])
    by_method: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for row in rows:
        if row.get("split") != "promotion":
            continue
        method = str(row.get("method")); seed = int(row.get("seed"))
        by_method.setdefault(method, {})[seed] = {
            str(q["question_id"]): {
                "regret": float(q.get("normalized_regret", q.get("mean_regret", 0.0))),
                "pairrank": float(q.get("pairwise_reversal_ranking_accuracy") or 0.0),
            }
            for q in row.get("question_metric_rows", [])
        }
    target_rows, base_rows, noflip_rows = by_method[target], by_method[baseline], by_method[noflip]
    regret = []
    pairrank = []
    for seed in sorted(set(target_rows) & set(base_rows) & set(noflip_rows)):
        common = sorted(set(target_rows[seed]) & set(base_rows[seed]))
        regret.extend(target_rows[seed][q]["regret"] - base_rows[seed][q]["regret"] for q in common)
        rank_delta = float(result["aggregation"]["promotion_summary"][target]["pairrank_acc"]) - float(result["aggregation"]["promotion_summary"][noflip]["pairrank_acc"])
        pairrank.extend(rank_delta for _ in common)
    return np.asarray(regret, dtype=float), np.asarray(pairrank, dtype=float)


def analyze(result: dict, *, target: str, question_counts: tuple[int, ...], replicates: int, seed: int) -> dict:
    regret, pairrank = _question_deltas(result, target)
    rng = np.random.default_rng(int(seed))
    observed_q = max(1, len(regret) // max(1, len(result.get("seeds", []))))
    projections = {}
    for count in question_counts:
        regret_gate = 0; pairrank_gate = 0
        for _ in range(max(1, int(replicates))):
            sampled_regret = rng.choice(regret, size=max(1, int(count)) * max(1, len(result.get("seeds", []))), replace=True)
            sampled_pairrank = rng.choice(pairrank, size=max(1, int(count)) * max(1, len(result.get("seeds", []))), replace=True)
            # Approximate the two-layer CI with percentile resampling of the
            # projected question-level effects.  This is a planning estimate,
            # not a formal result and is labelled as such in the receipt.
            regret_boot = (
                rng.choice(sampled_regret, size=(300, sampled_regret.size), replace=True).mean(axis=1)
                if sampled_regret.size else np.array([0.0])
            )
            pairrank_boot = (
                rng.choice(sampled_pairrank, size=(300, sampled_pairrank.size), replace=True).mean(axis=1)
                if sampled_pairrank.size else np.array([0.0])
            )
            if float(np.quantile(regret_boot, 0.975)) < 0.0:
                regret_gate += 1
            if float(np.quantile(pairrank_boot, 0.025)) > 0.0:
                pairrank_gate += 1
        projections[str(count)] = {
            "regret_ci_upper_below_zero_power": regret_gate / max(1, int(replicates)),
            "pairrank_ci_lower_above_zero_power": pairrank_gate / max(1, int(replicates)),
            "projected_question_clusters": int(count),
        }
    return {
        "schema_version": "pesco_tier1_p22_power_analysis_v0.1",
        "design_only": True,
        "formal_comparison_authorized": False,
        "final_evaluation_authorized": False,
        "target_method": target,
        "source_stage": result.get("stage"),
        "observed_question_clusters_per_seed": observed_q,
        "observed_regret_delta_mean": float(regret.mean()) if regret.size else None,
        "observed_pairrank_delta_mean": float(pairrank.mean()) if pairrank.size else None,
        "replicates": int(replicates),
        "seed": int(seed),
        "projection_method": "question-level effect resampling; percentile planning approximation, not a formal inferential claim",
        "projections": projections,
        "note": "Run before any future private-final model evaluation; current v0.6 remains private and unevaluated because the P2.2 gate is blocked.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="SFT→Listwise+PCGrad")
    parser.add_argument("--question-counts", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=911)
    args = parser.parse_args()
    output = analyze(json.loads(args.result.read_text(encoding="utf-8")), target=str(args.target), question_counts=tuple(args.question_counts), replicates=int(args.replicates), seed=int(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate P2.3.1 diagnostic records with seed×question bootstrap and family LOO."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _two_layer(values_by_seed, *, seed=23, replicates=2000):
    seeds = sorted(values_by_seed)
    questions = sorted({q for row in values_by_seed.values() for q in row})
    flat = [v for row in values_by_seed.values() for v in row.values()]
    if not flat:
        return {"point": None, "lower": None, "upper": None, "method": "not_estimable"}
    point = sum(flat) / len(flat)
    if len(seeds) < 2 or len(questions) < 2:
        return {"point": point, "lower": point, "upper": point, "method": "degenerate_seed_question_bootstrap"}
    rng = random.Random(int(seed)); draws = []
    for _ in range(max(1, int(replicates))):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_questions = [rng.choice(questions) for _ in questions]
        sample = [values_by_seed[s][q] for s in sampled_seeds for q in sampled_questions if q in values_by_seed[s]]
        if sample:
            draws.append(sum(sample) / len(sample))
    draws.sort()
    return {"point": point, "lower": draws[max(0, int(0.025 * len(draws)) - 1)], "upper": draws[min(len(draws) - 1, int(0.975 * len(draws)))], "method": "two_layer_seed_then_question_bootstrap_percentile_95", "seed_count": len(seeds), "question_count": len(questions), "replicates": len(draws)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    records = result.get("records", [])
    methods = result.get("methods", [])
    summaries = {}
    all_family_rows = defaultdict(lambda: defaultdict(list))
    for method in methods:
        rows = [row for row in records if row.get("method") == method]
        regret = defaultdict(dict); pairrank = defaultdict(dict); non_tie_acc = defaultdict(dict)
        family_rows = defaultdict(list)
        for row in rows:
            seed = int(row["seed"])
            for qrow in row.get("question_metric_rows", []):
                qid = str(qrow["question_id"])
                if qrow.get("normalized_regret") is not None:
                    regret[seed][qid] = float(qrow["normalized_regret"])
                if qrow.get("pairwise_reversal_ranking_accuracy") is not None:
                    pairrank[seed][qid] = float(qrow["pairwise_reversal_ranking_accuracy"])
                family_rows[str(qrow.get("family", ""))].append(qrow)
            for record in row.get("records", []):
                if record.get("non_tie"):
                    non_tie_acc[seed][str(record["question_id"])] = float(bool(record.get("action_correct", False)))
        family_mean = {}
        for family, qrows in family_rows.items():
            vals = [float(q["normalized_regret"]) for q in qrows if q.get("normalized_regret") is not None]
            family_mean[family] = sum(vals) / len(vals) if vals else None
        summaries[method] = {
            "seed_question_normalized_regret_ci": _two_layer(regret, seed=101, replicates=args.bootstrap_replicates),
            "seed_question_pairrank_ci": _two_layer(pairrank, seed=102, replicates=args.bootstrap_replicates),
            "non_tie_action_accuracy_ci": _two_layer(non_tie_acc, seed=103, replicates=args.bootstrap_replicates),
            "family_mean_normalized_regret": family_mean,
            "family_leave_one_out": {},
            "record_count": len(rows),
        }
        for index, family in enumerate(sorted(family_rows)):
            excluded_questions = {str(qrow["question_id"]) for qrow in family_rows[family]}
            loo_values = {
                seed: {qid: value for qid, value in row.items() if qid not in excluded_questions}
                for seed, row in regret.items()
            }
            summaries[method]["family_leave_one_out"][family] = _two_layer(
                loo_values, seed=200 + index, replicates=args.bootstrap_replicates
            )
    out = {
        "schema_version": "pesco_tier1_p231_dev_aggregate_v0.1",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "canonical_pair_digest": result.get("canonical_pair_digest"),
        "canonical_pair_contract": result.get("canonical_pair_contract"),
        "bootstrap_definition": "seed_then_question two-layer percentile 95%",
        "methods": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"methods": list(summaries), "canonical_pair_digest": out["canonical_pair_digest"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate the P2.3.3-r1 pure-factor receipts.

The aggregate never re-evaluates a policy and never reads audit labels.  Every
effect is computed from seed-by-question question rows emitted by the evaluator;
reversal effects use the continuous two-endpoint PairRank score while binary
accuracy remains available in the source receipts.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


CELLS = (
    "GRPO-Atomic", "Atomic+State", "Atomic+Branch", "Atomic+Flip",
    "Atomic+State+Branch", "Atomic+State+Flip", "Atomic+Branch+Flip", "PESCO-Full",
)


def _bootstrap(matrix: Mapping[int, Mapping[str, float]], reps: int, seed: int) -> dict[str, Any]:
    seeds = sorted(int(s) for s in matrix)
    questions = sorted({str(q) for row in matrix.values() for q in row})
    values = [float(matrix[s][q]) for s in seeds for q in questions if q in matrix[s]]
    if not values:
        return {"point": None, "lower": None, "upper": None, "seed_count": 0, "question_count": 0, "replicates": 0, "method": "seed_then_question_bootstrap"}
    point = sum(values) / len(values)
    if len(seeds) < 2:
        return {"point": point, "lower": point, "upper": point, "seed_count": len(seeds), "question_count": len(questions), "replicates": 0, "method": "degenerate_seed_then_question_bootstrap"}
    rng = random.Random(int(seed)); draws = []
    for _ in range(max(100, int(reps))):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_questions = [rng.choice(questions) for _ in questions] if len(questions) > 1 else questions
        x = [float(matrix[s][q]) for s in sampled_seeds for q in sampled_questions if q in matrix[s]]
        if x:
            draws.append(sum(x) / len(x))
    draws.sort()
    return {"point": point, "lower": draws[max(0, int(.025 * len(draws)) - 1)], "upper": draws[min(len(draws) - 1, int(.975 * len(draws)))], "seed_count": len(seeds), "question_count": len(questions), "replicates": len(draws), "method": "seed_then_question_bootstrap_percentile_95"}


def _matrix(records: list[Mapping[str, Any]], method: str, split: str, key: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in records:
        if row.get("method") != method or row.get("split") != split:
            continue
        values = {}
        for q in row.get("question_metric_rows", []):
            value = q.get(key)
            if value is not None:
                values[str(q.get("question_id"))] = float(value)
        out[int(row["seed"])] = values
    return out


def _delta(left: Mapping[int, Mapping[str, float]], right: Mapping[int, Mapping[str, float]]) -> dict[int, dict[str, float]]:
    return {s: {q: float(left[s][q]) - float(right[s][q]) for q in set(left[s]) & set(right[s])} for s in set(left) & set(right)}


def _comparison(records: list[Mapping[str, Any]], left: str, right: str, split: str, key: str, reps: int, seed: int) -> dict[str, Any]:
    lm = _matrix(records, left, split, key); rm = _matrix(records, right, split, key)
    return {"left": left, "right": right, "metric": key, "delta_left_minus_right": _bootstrap(_delta(lm, rm), reps, seed)}


def _seed_metric(records: list[Mapping[str, Any]], method: str, split: str, key: str) -> dict[int, dict[str, float]]:
    out = {}
    for row in records:
        if row.get("method") == method and row.get("split") == split and row.get(key) is not None:
            out[int(row["seed"])] = {"__seed__": float(row[key])}
    return out


def _means(records: list[Mapping[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    methods = sorted({str(r.get("method")) for r in records if r.get("split") == split})
    for method in methods:
        rows = [r for r in records if r.get("split") == split and r.get("method") == method]
        regrets = [float(r.get("normalized_regret", r.get("mean_regret", 0.0))) for r in rows]
        ranks = [float(r.get("pairwise_reversal_ranking_score", 0.0) or 0.0) for r in rows]
        f1 = [float(r.get("state_macro_f1", 0.0)) for r in rows]
        out[method] = {"seed_count": len({int(r["seed"]) for r in rows}), "mean_normalized_regret": sum(regrets) / len(regrets) if regrets else None, "mean_pairrank_score": sum(ranks) / len(ranks) if ranks else None, "mean_pairrank_binary_accuracy": sum(float(r.get("pairwise_reversal_ranking_accuracy", 0.0) or 0.0) for r in rows) / len(rows) if rows else None, "mean_state_macro_f1": sum(f1) / len(f1) if f1 else None}
    return out


def _family_means(records: list[Mapping[str, Any]], split: str, method: str, key: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.get("method") != method or row.get("split") != split:
            continue
        for q in row.get("question_metric_rows", []):
            if q.get(key) is not None:
                acc[str(q.get("family", "unknown"))].append(float(q[key]))
    return {family: sum(values) / len(values) for family, values in acc.items() if values}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="promotion")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--selected-baseline", default=None)
    args = parser.parse_args(argv)
    payload = json.loads(args.result.read_text(encoding="utf-8")); records = list(payload.get("records", []))
    means = _means(records, args.split)
    comparisons: dict[str, dict[str, Any]] = {}
    pairs = (("PESCO-Full", "Atomic+State+Flip", "Full-ASF", "regret"), ("PESCO-Full", "Atomic+State+Branch", "Full-ASB", "pairrank"), ("PESCO-Full", "Atomic+State+Branch", "Full-ASB-regret", "regret"), ("PESCO-Full", "Atomic+Branch+Flip", "Full-ABF-state", "state"))
    for i, (left, right, name, kind) in enumerate(pairs):
        key = {"regret": "normalized_regret", "pairrank": "pairwise_reversal_ranking_score", "state": "state_macro_f1"}[kind]
        comparisons[name] = (_comparison(records, left, right, args.split, key, args.bootstrap_replicates, 233 + i) if kind != "state" else {"left": left, "right": right, "metric": key, "delta_left_minus_right": _bootstrap(_delta(_seed_metric(records,left,args.split,key), _seed_metric(records,right,args.split,key)), args.bootstrap_replicates, 237 + i)})
    # Four branch main effects are pooled question-wise before bootstrap; this is
    # the preregistered factorial main effect, not a cherry-picked cell.
    effects = []
    for left, right in (("Atomic+Branch", "GRPO-Atomic"), ("Atomic+State+Branch", "Atomic+State"), ("Atomic+Branch+Flip", "Atomic+Flip"), ("PESCO-Full", "Atomic+State+Flip")):
        effects.append(_delta(_matrix(records, left, args.split, "normalized_regret"), _matrix(records, right, args.split, "normalized_regret")))
    pooled: dict[int, dict[str, float]] = defaultdict(dict)
    for matrix in effects:
        for seed, values in matrix.items():
            for q, value in values.items():
                pooled[seed].setdefault(q, []).append(float(value))
    pooled_mean = {s: {q: sum(v) / len(v) for q, v in vals.items()} for s, vals in pooled.items()}
    comparisons["factorial-branch-main-effect"] = {"left": "Branch-on", "right": "Branch-off", "metric": "normalized_regret", "component_effects": [("Atomic+Branch", "GRPO-Atomic"), ("Atomic+State+Branch", "Atomic+State"), ("Atomic+Branch+Flip", "Atomic+Flip"), ("PESCO-Full", "Atomic+State+Flip")], "delta_left_minus_right": _bootstrap(pooled_mean, args.bootstrap_replicates, 239)}
    selected = args.selected_baseline
    # Freeze the non-Full baseline on the registered tune split.  Promotion
    # performance is never consulted for model/baseline selection.
    tune_means = _means(records, "tune")
    if not selected:
        selected = min((m for m in tune_means if m != "PESCO-Full" and m != "SFT"), key=lambda m: float(tune_means[m].get("mean_normalized_regret", float("inf"))), default=None)
    if selected:
        comparisons["full_vs_best_nonfull"] = _comparison(records, "PESCO-Full", selected, args.split, "normalized_regret", args.bootstrap_replicates, 251)
    full_families = _family_means(records, args.split, "PESCO-Full", "normalized_regret")
    asf_families = _family_means(records, args.split, "Atomic+State+Flip", "normalized_regret")
    family_direction = {family: full_families[family] < asf_families.get(family, float("inf")) for family in full_families}
    direction = {}
    full_matrix = _matrix(records, "PESCO-Full", args.split, "normalized_regret"); asf_matrix = _matrix(records, "Atomic+State+Flip", args.split, "normalized_regret")
    for seed in sorted(set(full_matrix) & set(asf_matrix)):
        lv = [full_matrix[seed][q] for q in set(full_matrix[seed]) & set(asf_matrix[seed])]; rv = [asf_matrix[seed][q] for q in set(full_matrix[seed]) & set(asf_matrix[seed])]
        direction[str(seed)] = {"full_better": bool(sum(lv) / len(lv) < sum(rv) / len(rv)) if lv else False, "delta": (sum(lv) / len(lv) - sum(rv) / len(rv)) if lv else None}
    state_rows = [r for r in records if r.get("method") == "PESCO-Full" and r.get("split") == args.split]
    state_recall = {state: [float(r.get("state_recall", {}).get(state)) for r in state_rows if r.get("state_recall", {}).get(state) is not None] for state in ("supported", "refuted", "insufficient", "invalid")}
    budget_receipts = payload.get("training_logs", {})
    budget = [budget_receipts.get(str(seed), {}).get(method, {}).get("budget_contract", {}) for seed in payload.get("seeds", []) for method in CELLS]
    budget_keys = ("policy_rollout_calls", "counterfactual_branch_calls", "exploration_seed_executions", "confirmation_seed_executions", "optimizer_steps", "forward_backward_flops")
    budget_equal = bool(budget) and len({tuple(b.get(k) for k in budget_keys) for b in budget}) == 1
    out = {"schema_version": "pesco_tier1_p233_r1_aggregate_v1", "receipt_derived": True, "diagnostic_only": True, "formal_comparison_authorized": False, "split": args.split, "methods": means, "tune_selection_means": tune_means, "selected_baseline_frozen_from_tune": selected, "comparisons": comparisons, "seed_direction": direction, "at_least_8_of_10_seed_directions_positive": sum(bool(v["full_better"]) for v in direction.values()) / max(1, len(direction)), "family_direction": family_direction, "family_majority_direction_positive": sum(family_direction.values()) / max(1, len(family_direction)), "state_recall": state_recall, "min_state_recall": min((min(v) for v in state_recall.values() if v), default=None), "budget_receipts": budget, "environment_execution_budget_matched": budget_equal, "pairrank_metric": "question-macro continuous two-endpoint probability margin; binary accuracy retained in evaluator receipts", "bootstrap": {"replicates": args.bootstrap_replicates, "seed_question_two_layer": True}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"); print(json.dumps({"output": str(args.output), "selected_baseline": selected, "comparisons": list(comparisons)})); return 0


if __name__ == "__main__":
    raise SystemExit(main())

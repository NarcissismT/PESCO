#!/usr/bin/env python3
"""Audit tie strata and relative (not global) reward-weight stability."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


GROUPS = {
    "replication": ("confirmation_bonus", "replicate_bonus"),
    "validity": ("validity_gate", "repair_protocol_bonus", "heldout_split_bonus"),
    "cost": ("execution_cost_penalty",),
    "discovery": ("mechanism_transition_bonus", "switch_success_bonus", "switch_failure_penalty", "sample_precision_bonus", "state_resolution_bonus"),
}
ACTION_ORDER = (
    "continue_current_method",
    "add_samples_or_seeds",
    "repair_data_split",
    "switch_to_alternative_method",
)


def _gap(values):
    ordered = sorted((float(value) for value in values), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0


def _components(example):
    raw = example.get("metadata", {}).get("reward_components")
    if not isinstance(raw, dict):
        return None
    actions = list(ACTION_ORDER)
    result = []
    for action in actions:
        terms = raw.get(action)
        if not isinstance(terms, dict):
            return None
        result.append({str(key): float(value) for key, value in terms.items()})
    return actions, result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    rng = random.Random(int(args.seed))
    rows = []
    counters = Counter()
    missing = 0
    for example in dataset.get("examples", []):
        parsed = _components(example)
        if parsed is None:
            missing += 1
            continue
        actions, terms = parsed
        base_values = [sum(values.values()) for values in terms]
        recorded = [float(value) for value in example.get("branch_utilities", [])]
        if len(recorded) != len(base_values) or any(abs(a - b) > 1e-6 for a, b in zip(recorded, base_values)):
            missing += 1
            continue
        gap = _gap(base_values)
        non_tie = gap > float(args.tolerance)
        base_winner = max(range(len(base_values)), key=lambda index: base_values[index])
        perturbation_winners = []
        for _ in range(max(1, int(args.replicates))):
            weights = {group: (0.8 if rng.random() < 0.5 else 1.2) for group in GROUPS}
            values = []
            for action_terms in terms:
                value = 0.0
                for key, term in action_terms.items():
                    group = next((name for name, names in GROUPS.items() if key in names), "discovery")
                    value += float(term) * float(weights[group])
                values.append(value)
            winner = max(range(len(values)), key=lambda index: values[index])
            perturbation_winners.append(winner)
        stable = sum(winner == base_winner for winner in perturbation_winners)
        counters["world_count"] += 1
        counters["non_tie_world_count"] += int(non_tie)
        counters["tie_world_count"] += int(not non_tie)
        counters["non_tie_stable_n"] += int(stable if non_tie else 0)
        counters["tie_stable_n"] += int(stable if not non_tie else 0)
        rows.append({
            "split": str(example.get("split", "unknown")),
            "question_id": example.get("question_id"),
            "world_id": example.get("world_id"),
            "top1_minus_top2_gap": gap,
            "non_tie": non_tie,
            "base_winner": actions[base_winner],
            "relative_weight_groups": GROUPS,
            "perturbation_count": len(perturbation_winners),
            "stable_count": stable,
            "stable_fraction": stable / max(1, len(perturbation_winners)),
        })
    reps = max(1, int(args.replicates))
    out = {
        "schema_version": "pesco_tier1_p231_relative_reward_stability_audit_v0.2",
        "tolerance": float(args.tolerance),
        "non_tie_definition": "top1_minus_top2 > tolerance",
        "perturbation_definition": "independent relative +/-20% weights by replication/validity/cost/discovery groups",
        "relative_weight_groups": GROUPS,
        "replicate_n_per_example": reps,
        "counts": dict(counters),
        "missing_or_malformed_receipt_n": missing,
        "non_tie_stable_winner_fraction": counters["non_tie_stable_n"] / max(1, counters["non_tie_world_count"] * reps),
        "tie_stable_winner_fraction": counters["tie_stable_n"] / max(1, counters["tie_world_count"] * reps),
        "status": "completed" if not missing else "failed_closed_missing_atomic_receipts",
        "authorized": not missing and counters["non_tie_world_count"] > 0 and counters["tie_world_count"] > 0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": out["status"], "authorized": out["authorized"], "counts": out["counts"]}, ensure_ascii=False))
    return 0 if out["authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

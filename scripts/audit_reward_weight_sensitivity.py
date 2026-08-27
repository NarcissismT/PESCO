#!/usr/bin/env python3
"""Audit ±20% reward-component perturbations on the frozen diagnostic worlds.

This is an evaluator-side structural audit.  It recomputes branch-utility winners
from atomic reward components and reports winner stability by split/family.  A
method-ranking gate is intentionally not inferred when policy action receipts are
not present; that distinction is recorded explicitly in the output.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def audit(dataset_path: Path, delta: float = 0.20) -> dict:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("examples", [])
    component_names = sorted({name for row in rows for name in row.get("metadata", {}).get("reward_components", {}).get("continue_current_method", {})})
    base_winners = {}
    scenarios = {"joint_up": {name: 1.0 + delta for name in component_names}, "joint_down": {name: 1.0 - delta for name in component_names}}
    for name in component_names:
        scenarios[f"{name}_up"] = {key: 1.0 for key in component_names}
        scenarios[f"{name}_up"][name] = 1.0 + delta
        scenarios[f"{name}_down"] = {key: 1.0 for key in component_names}
        scenarios[f"{name}_down"][name] = 1.0 - delta
    changed = defaultdict(int)
    total = defaultdict(int)
    family_changed = defaultdict(lambda: defaultdict(int))
    for index, row in enumerate(rows):
        comps = row.get("metadata", {}).get("reward_components", {})
        actions = list(comps)
        base_values = {action: sum(float(value) for value in comps[action].values()) for action in actions}
        base = max(actions, key=lambda action: base_values[action])
        base_winners[index] = base
        for scenario, weights in scenarios.items():
            values = {action: sum(float(value) * weights.get(name, 1.0) for name, value in comps[action].items()) for action in actions}
            winner = max(actions, key=lambda action: values[action])
            total[scenario] += 1
            if winner != base:
                changed[scenario] += 1
                family_changed[scenario][str(row.get("metadata", {}).get("family", "unknown"))] += 1
    stability = {scenario: {"changed_count": changed[scenario], "row_count": total[scenario], "stability_rate": 1.0 - changed[scenario] / total[scenario] if total[scenario] else None, "changed_by_family": dict(family_changed[scenario])} for scenario in scenarios}
    return {
        "schema_version": "pesco_reward_weight_sensitivity_v0.1",
        "delta": float(delta),
        "component_names": component_names,
        "scenarios": stability,
        "method_ranking_available": False,
        "method_ranking_note": "P2.2 matrix stores evaluator metrics without per-action atomic receipts; this audit is structural and cannot authorize method-ranking claims.",
        "formal_comparison_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

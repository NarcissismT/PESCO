#!/usr/bin/env python3
"""Re-score selected actions under shared ±20% atomic reward perturbations.

The diagnostic matrix retains selected action/world IDs, while the evaluator
dataset retains atomic reward components.  This script joins those two audited
records and checks whether the ranking of methods by promotion normalized regret
is invariant to global reward-component perturbations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(dataset_path: Path, result_path: Path, delta: float = 0.20) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    examples = {str(row.get("world_id")): row for row in dataset.get("examples", [])}
    rows = [row for row in result.get("records", []) if row.get("split") == "promotion"]
    methods = sorted({str(row.get("method")) for row in rows})
    component_names = sorted({str(name) for row in examples.values() for action in row.get("metadata", {}).get("reward_components", {}).values() for name in action})
    scenarios = {
        "base": {name: 1.0 for name in component_names},
        "joint_up": {name: 1.0 + delta for name in component_names},
        "joint_down": {name: 1.0 - delta for name in component_names},
    }
    for name in component_names:
        scenarios[f"{name}_up"] = {key: 1.0 for key in component_names}; scenarios[f"{name}_up"][name] = 1.0 + delta
        scenarios[f"{name}_down"] = {key: 1.0 for key in component_names}; scenarios[f"{name}_down"][name] = 1.0 - delta
    scores = {}
    for scenario, weights in scenarios.items():
        scores[scenario] = {}
        for method in methods:
            values = []
            for row in rows:
                if row.get("method") != method:
                    continue
                compact_rows = row.get("selected_action_rows", [])
                for compact in compact_rows:
                    example = examples.get(str(compact.get("world_id")))
                    if example is None:
                        continue
                    action = str(compact.get("selected_action")); components = example.get("metadata", {}).get("reward_components", {}).get(action)
                    if not isinstance(components, dict):
                        continue
                    all_components = example.get("metadata", {}).get("reward_components", {})
                    selected = sum(float(value) * weights.get(str(name), 1.0) for name, value in components.items())
                    best = max(sum(float(value) * weights.get(str(name), 1.0) for name, value in action_components.items()) for action_components in all_components.values())
                    worst = min(sum(float(value) * weights.get(str(name), 1.0) for name, value in action_components.items()) for action_components in all_components.values())
                    values.append((best - selected) / max(best - worst, 1e-8))
            scores[scenario][method] = sum(values) / len(values) if values else None
    method_ranking_available = bool(all(scores["base"].get(method) is not None for method in methods))
    def ranking(scenario: str) -> list[str]:
        return sorted(methods, key=lambda method: (scores[scenario].get(method) is None, scores[scenario].get(method) if scores[scenario].get(method) is not None else float("inf")))
    base_order = ranking("base")
    stable = {scenario: (ranking(scenario) == base_order if method_ranking_available else None) for scenario in scenarios}
    return {
        "schema_version": "pesco_tier1_p22_method_reward_sensitivity_v0.1",
        "delta": float(delta),
        "method_count": len(methods),
        "methods": methods,
        "normalized_regret_by_scenario": scores,
        "ranking_by_scenario": {scenario: ranking(scenario) for scenario in scenarios},
        "ranking_stable_by_scenario": stable,
        "method_ranking_available": method_ranking_available,
        "method_ranking_stable": bool(method_ranking_available and all(stable.values())),
        "formal_comparison_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--result", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    output = audit(args.dataset, args.result); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(output, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

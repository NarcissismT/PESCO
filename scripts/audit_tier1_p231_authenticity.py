#!/usr/bin/env python3
"""Evaluate P2.3.1 authenticity/convergence gates on the three-seed run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run(result_path: Path, output_path: Path, shortcut_path: Path | None = None) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    logs = result.get("training_logs", {})
    methods = [method for method in result.get("methods", []) if method != "SFT"]
    record_actions = defaultdict(dict)
    for row in result.get("records", []):
        record_actions[(int(row.get("seed", -1)), str(row.get("method")))] = {
            str(item.get("world_id")): str(item.get("selected_action"))
            for item in row.get("records", [])
        }
    all_rows = [row for seed in logs.values() for method, item in seed.items() if method != "checkpoint" for row in item.get("logs", [])]
    reward_audit = result.get("reward_tensor_audit", {})
    reward_differences = bool(reward_audit.get("all_required_differences"))
    rloo_grpo_different = False
    clip_ratios_seen = False
    clip_effective = False
    branch_changed = False
    flip_changed = False
    branch_action_changed = False
    flip_action_changed = False
    convergence = True
    budget_values = defaultdict(set)
    per_method = {}
    for seed, seed_logs in logs.items():
        for method in methods:
            item = seed_logs.get(method)
            if not item:
                convergence = False
                continue
            rows = item.get("logs", [])
            finite_rows = all(_finite(row.get("loss")) and _finite(row.get("gradient_norm")) and _finite(row.get("entropy")) for row in rows)
            if not rows or not finite_rows:
                convergence = False
            losses = [float(row["loss"]) for row in rows]
            loss_delta = losses[-1] - losses[0] if losses else None
            ratio_delta = max((float(row.get("importance_ratio_max_abs_delta", 0.0)) for row in rows), default=0.0)
            clip_fraction = max((float(row.get("clip_fraction", 0.0)) for row in rows), default=0.0)
            clip_ratios_seen = clip_ratios_seen or ratio_delta > 1e-6
            clip_effective = clip_effective or clip_fraction > 0.0
            for row in rows:
                budget_values[method].add((int(row.get("rollout_sample_count", -1)), int(row.get("environment_budget_units", -1))))
            per_method[f"{seed}:{method}"] = {
                "optimizer": item.get("optimizer"),
                "reward_name": item.get("reward_name"),
                "initial_sft_digest": item.get("initial_sft_digest"),
                "final_policy_digest": item.get("final_policy_digest"),
                "optimizer_parameter_delta": bool(item.get("optimizer_parameter_delta")),
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_delta": loss_delta,
                "loss_finite": finite_rows,
                "ratio_max_abs_delta": ratio_delta,
                "clip_fraction_max": clip_fraction,
                "minibatch_epochs": item.get("minibatch_epochs"),
                "rollout_frozen": all(bool(row.get("frozen_rollout")) for row in rows),
                "rollout_budget_values": sorted(budget_values[method]),
            }
            if method == "GRPO-Terminal":
                rloo = seed_logs.get("RLOO", {})
                rloo_grpo_different = rloo_grpo_different or item.get("final_policy_digest") != rloo.get("final_policy_digest")
            if method in {"GRPO+Branch", "GRPO+Branch+Flip"}:
                branch_changed = branch_changed or bool(item.get("optimizer_parameter_delta"))
                branch_action_changed = branch_action_changed or bool(set(record_actions[(int(seed), method)].items()) ^ set(record_actions[(int(seed), "SFT")].items()))
            if method in {"GRPO+Flip", "GRPO+Branch+Flip"}:
                flip_changed = flip_changed or bool(item.get("optimizer_parameter_delta"))
                flip_action_changed = flip_action_changed or bool(set(record_actions[(int(seed), method)].items()) ^ set(record_actions[(int(seed), "SFT")].items()))
    common_sft = True
    for seed, seed_logs in logs.items():
        digests = {str(seed_logs.get(method, {}).get("initial_sft_digest", "")) for method in methods}
        common_sft = common_sft and len(digests) == 1 and bool(next(iter(digests), ""))
    budget_fair = bool(budget_values) and all(len(values) == 1 for values in budget_values.values())
    # Neural-vs-shortcut closeness is diagnostic only when a strict sklearn probe
    # exists.  A fallback probe is explicitly not enough for a promotion claim.
    shortcut_available = False
    fallback_shortcut_available = False
    shortcut_best_regret = None
    if shortcut_path and shortcut_path.exists():
        shortcut = json.loads(shortcut_path.read_text(encoding="utf-8"))
        shortcut_available = bool(shortcut.get("dependency", {}).get("sklearn", {}).get("available"))
        fallback_shortcut_available = bool(shortcut.get("fallback_used") or shortcut.get("models"))
        if shortcut_available or fallback_shortcut_available:
            values = []
            for key, item in shortcut.get("models", {}).items():
                metrics = item.get("metrics_by_split", {}).get("tune", {})
                if metrics.get("normalized_regret") is not None:
                    values.append(float(metrics["normalized_regret"]))
            shortcut_best_regret = min(values) if values else None
    neural_tune = [float(row.get("normalized_regret")) for row in result.get("records", []) if row.get("method") not in {"SFT"} and row.get("normalized_regret") is not None]
    neural_best_regret = min(neural_tune) if neural_tune else None
    neural_close = bool(shortcut_best_regret is not None and neural_best_regret is not None and neural_best_regret <= shortcut_best_regret + 0.10)
    gate = {
        "reward_tensors_differ": reward_differences,
        "rloo_vs_grpo_parameter_updates_differ": rloo_grpo_different,
        "grpo_ratio_observed": clip_ratios_seen,
        "grpo_clip_effective": clip_effective,
        "branch_changes_parameters": branch_changed,
        "flip_changes_parameters": flip_changed,
        "branch_changes_actions": branch_action_changed,
        "flip_changes_actions": flip_action_changed,
        "convergence_logs_finite": convergence,
        "common_sft_checkpoint": common_sft,
        "matched_rollout_budget": budget_fair,
        "strict_shortcut_probe_available": shortcut_available,
        "neural_close_to_shortcut": neural_close,
    }
    out = {
        "schema_version": "pesco_tier1_p231_authenticity_gate_v0.1",
        "status": "GO_P2_3_1_10SEED_AUTHORIZED" if all(gate.values()) else "NO_GO_P2_3_1",
        "gate": gate,
        "required_rule": "all authenticity gates must pass before any 10-seed or promotion-v3 evaluation",
        "shortcut_best_tune_normalized_regret": shortcut_best_regret,
        "neural_best_tune_normalized_regret": neural_best_regret,
        "fallback_shortcut_available": fallback_shortcut_available,
        "per_seed_method": per_method,
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "promotion_v3_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": out["status"], "gate": gate}, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortcut", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run(args.result, args.output, args.shortcut)
    return 0 if result["status"].startswith("GO_") else 2


if __name__ == "__main__":
    raise SystemExit(main())

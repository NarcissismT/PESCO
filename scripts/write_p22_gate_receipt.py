#!/usr/bin/env python3
"""Write a fail-closed P2.2 promotion gate receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(result: dict, family_loo: dict, shortcut: dict, reward_sensitivity: dict | None = None, *, method: str = "SFT→Pairwise-Full") -> dict:
    summary = result.get("aggregation", {}).get("promotion_summary", {})
    checks = result.get("aggregation", {}).get("gate_checks", {}).get(method, {})
    sft = summary.get("SFT", {})
    noflip = summary.get("SFT→NoFlip", {})
    target = summary.get(method, {})
    shortcut_rows = shortcut.get("models", {})
    # Strict sklearn runs use a top-level ``completed_sklearn`` status while
    # each model row is marked ``completed``.  Accept only explicitly
    # completed (non-fallback) rows; NumPy fallback rows are diagnostic-only
    # and must not silently authorize the shortcut gate.
    def promotion_normalized_regret(row: dict) -> float | None:
        """Read the strict probe's canonical promotion metric.

        The probe stores metrics under ``metrics_by_split.promotion``; older
        diagnostic receipts used a flattened ``promotion_normalized_regret``
        field.  Accept both layouts, but never infer a value from a failed or
        NumPy-fallback model.
        """
        value = row.get("promotion_normalized_regret")
        if value is None:
            value = (row.get("metrics_by_split", {}).get("promotion", {}) or {}).get("normalized_regret")
        return float(value) if value is not None else None

    shortcut_regrets = []
    completed_shortcut_models = set()
    for row in shortcut_rows.values():
        value = promotion_normalized_regret(row)
        if (
            value is not None
            and str(row.get("status", "")).startswith("completed")
            and str(row.get("implementation", "")).lower() == "sklearn"
        ):
            shortcut_regrets.append(value)
            completed_shortcut_models.add(str(row.get("model", "")))
    required_switch_gate = bool(
        target.get("required_switch_rate") is not None
        and target["required_switch_rate"] > 0.041
        # The preregistered wording is “better than SFT or NoFlip”: beating the
        # weaker comparator is sufficient, while the absolute historical floor is
        # still enforced above.  Requiring max(SFT, NoFlip) would silently turn
        # the documented OR gate into an AND gate.
        and target["required_switch_rate"] > min(float(sft.get("required_switch_rate") or 0), float(noflip.get("required_switch_rate") or 0))
    )
    safety_fields = ("confirmation_rate", "erroneous_repair_rate", "validity_rate")
    safety_complete = all(
        row.get(field) is not None
        for row in (target, sft, noflip)
        for field in safety_fields
    )
    safety_gate = bool(
        safety_complete
        and float(target["confirmation_rate"]) >= min(float(sft["confirmation_rate"]), float(noflip["confirmation_rate"]))
        and float(target["erroneous_repair_rate"]) <= max(float(sft["erroneous_repair_rate"]), float(noflip["erroneous_repair_rate"]))
        and float(target["validity_rate"]) >= min(float(sft["validity_rate"]), float(noflip["validity_rate"]))
    )
    required_shortcut_models = {"logistic_regression", "gradient_boosting"}
    shortcut_gate = bool(
        required_shortcut_models.issubset(completed_shortcut_models)
        and shortcut_regrets
        and target.get("mean_regret") is not None
        and float(target["mean_regret"]) < min(shortcut_regrets)
    )
    core_gate = bool(checks.get("regret_gate") and checks.get("pairrank_gate"))
    canonical_top1_gate = bool(checks.get("canonical_top1_gate"))
    family_gate = bool(family_loo.get("gate", {}).get("regret_non_single_family") and family_loo.get("gate", {}).get("pairrank_non_single_family"))
    reward_structural_gate = bool(reward_sensitivity and all(float(v.get("stability_rate") or 0.0) >= 0.95 for v in reward_sensitivity.get("scenarios", {}).values()))
    reward_method_gate = bool(reward_sensitivity and reward_sensitivity.get("method_ranking_available") and reward_sensitivity.get("method_ranking_stable"))
    gates = {
        "matrix_complete": bool(result.get("matrix_complete")),
        "regret_vs_sft": bool(checks.get("regret_gate")),
        "pairrank_vs_noflip": bool(checks.get("pairrank_gate")),
        "canonical_top1_reversal": canonical_top1_gate,
        "required_switch": required_switch_gate,
        "safety_confirmation_validity_erroneous_repair": safety_gate,
        "strict_shortcut_probe": shortcut_gate,
        "family_leave_one_out": family_gate,
        "reward_weight_structural_stability": reward_structural_gate,
        "reward_weight_method_ranking_stability": reward_method_gate,
    }
    return {
        "schema_version": "pesco_tier1_p22_gate_receipt_v0.1",
        "status": "p22_gate_passed" if all(gates.values()) else "p22_gate_blocked",
        "formal_comparison_authorized": False,
        "final_evaluation_authorized": False,
        "method": method,
        "gates": gates,
        "metrics": {"target": target, "sft": sft, "noflip": noflip, "regret_gate_detail": checks, "family_loo": family_loo, "shortcut": shortcut, "reward_sensitivity": reward_sensitivity},
        "blockers": [name for name, passed in gates.items() if not passed],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--family-loo", type=Path, required=True)
    parser.add_argument("--shortcut", type=Path, required=True)
    parser.add_argument("--reward-sensitivity", type=Path, required=False)
    parser.add_argument("--method", default="SFT→Pairwise-Full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reward = json.loads(args.reward_sensitivity.read_text()) if args.reward_sensitivity else None
    receipt = build(
        json.loads(args.result.read_text()),
        json.loads(args.family_loo.read_text()),
        json.loads(args.shortcut.read_text()),
        reward,
        method=str(args.method),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "p22_gate_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""P2.3.4 gate: strict estimator/bias/LOO/shortcut/alignment gates.

This gate reads receipts produced by the *r1* matrix (frozen atomic_target
config) plus the new P2.3.4 estimator and promotion guard receipts.  It does
not recompute any training runs; it only evaluates the registered receipts
against the YAML target block in p2-3-4.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def bound(agg, name, kind="upper"):
    return agg.get("comparisons", {}).get(name, {}).get("delta_left_minus_right", {}).get(kind)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--aggregate", type=Path, required=True)
    p.add_argument("--convergence", type=Path, required=True)
    p.add_argument("--shortcut", type=Path, required=True)
    p.add_argument("--estimator", type=Path, required=True)
    p.add_argument("--leakage", type=Path, required=True)
    p.add_argument("--stability", type=Path, required=True)
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--loo", type=Path, required=True)
    p.add_argument("--promotion-v6-guard", type=Path, required=True)
    p.add_argument("--private-commitment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)

    m = json.loads(a.matrix.read_text())
    g = json.loads(a.aggregate.read_text())
    c = json.loads(a.convergence.read_text())
    s = json.loads(a.shortcut.read_text())
    e = json.loads(a.estimator.read_text())
    l = json.loads(a.leakage.read_text())
    st = json.loads(a.stability.read_text())
    at = json.loads(a.attribution.read_text())
    loo = json.loads(a.loo.read_text())
    v6 = json.loads(a.promotion_v6_guard.read_text())
    commitment = json.loads(a.private_commitment.read_text())

    # r1 policy-authenticity group (must be satisfied before any p234 judgment)
    full_logs = [m.get("training_logs", {}).get(str(seed), {}).get("PESCO-Full", {}) for seed in m.get("seeds", [])]
    action_head_delta = bool(full_logs) and all(float(r.get("action_head_parameter_delta_norm", 0)) > 0 for r in full_logs)
    action_logits_changed = bool(full_logs) and all(float(r.get("action_logits_change_mean_abs", 0)) > 0 for r in full_logs)
    no_external_adapter = bool(full_logs) and all(bool(r.get("no_external_adapter_overrides_action_logits", False)) for r in full_logs)
    p233_r1_policy_authenticity_pass = action_head_delta and action_logits_changed and no_external_adapter

    # Mechanism contributions (r1)
    branch_factorial = finite(bound(g, "factorial-branch-main-effect")) and float(bound(g, "factorial-branch-main-effect")) < 0
    full_vs_asf = finite(bound(g, "Full-ASF")) and float(bound(g, "Full-ASF")) < 0
    full_vs_asb_pairrank = finite(bound(g, "Full-ASB", "lower")) and float(bound(g, "Full-ASB", "lower")) > 0

    # Strong-baseline / shortcut group
    full_vs_frozen = finite(bound(g, "full_vs_best_nonfull")) and float(bound(g, "full_vs_best_nonfull")) < 0
    strict_shortcuts = [v for k, v in s.get("models", {}).items() if k.startswith("without_confirmation:") and v.get("status") == "completed"]
    strict_shortcuts = [float(v.get("metrics_by_split", {}).get("promotion", {}).get("normalized_regret")) for v in strict_shortcuts if finite(v.get("metrics_by_split", {}).get("promotion", {}).get("normalized_regret"))]
    full_mean = float(g.get("methods", {}).get("PESCO-Full", {}).get("mean_normalized_regret", float("inf")))
    full_beats_strict_rf_gbdt = bool(strict_shortcuts) and full_mean < min(strict_shortcuts)

    # Practical effect size floor: effect must exceed 0.005 absolute (pre-registered)
    practical_floor = 0.005
    effect_size = abs(float(bound(g, "full_vs_best_nonfull", "point") or 0.0))
    practical_effect_size_floor_pass = effect_size >= practical_floor

    # Family LOO group
    loo_rows = loo.get("rows", [])
    loo_finite = all(finite(r.get("normalized_regret")) and finite(r.get("pairrank_score")) and finite(r.get("state_macro_f1")) for r in loo_rows)
    family_means = {}
    for r in loo_rows:
        nr = r.get("normalized_regret")
        if nr is None or not finite(nr):
            continue
        if r.get("method") == "PESCO-Full":
            family_means.setdefault(r.get("family"), {})["full"] = float(nr)
        elif r.get("method") == "Atomic+State+Flip":
            family_means.setdefault(r.get("family"), {})["asf"] = float(nr)
    family_majority_positive = sum(1 for fam, v in family_means.items() if v.get("full") is not None and v.get("asf") is not None and v["full"] < v["asf"]) >= 5
    # macro regret CI across families (bootstrap family means)
    if family_means:
        deltas = [v["full"] - v["asf"] for v in family_means.values() if "full" in v and "asf" in v]
        macro = sum(deltas) / len(deltas)
        sd = math.sqrt(sum((x - macro) ** 2 for x in deltas) / max(1, len(deltas) - 1))
        loo_macro_ci_upper = macro + 1.96 * sd / math.sqrt(max(1, len(deltas)))
    else:
        macro, loo_macro_ci_upper = None, None
    loo_macro_regret_ci_upper_lt_zero = bool(loo_macro_ci_upper is not None and loo_macro_ci_upper < 0)
    # worst family noninferiority: worst Full-vs-ASF family delta must be < 0.02
    if family_means:
        worst_family_delta = max((v["full"] - v["asf"] for v in family_means.values() if "full" in v and "asf" in v), default=float("inf"))
        worst_family_noninferior = worst_family_delta < 0.02
    else:
        worst_family_noninferior = False
    all_loo_metrics_finite = loo_finite

    # Estimator validity group (new)
    estimator_rows = e.get("rows", [])
    bias_bound = 0.05
    coverage_floor = 0.90
    all_estimator_bias_bounds_pass = all(abs(float(r.get("bias", float("inf")))) < bias_bound for r in estimator_rows)
    all_estimator_coverage_lower_bounds_pass = all(float(r.get("coverage_lower_approx", 0.0)) >= coverage_floor for r in estimator_rows)
    no_hidden_truth_used_by_estimator = bool(e.get("source_audit", {}).get("no_hidden_truth_used_by_estimator")) and bool(l.get("pass"))
    ms = next((r for r in estimator_rows if r.get("mechanism") == "measurement_shift"), None)
    measurement_shift_calibration_pass = bool(ms and abs(float(ms.get("bias", float("inf")))) < bias_bound and float(ms.get("empirical_95ci_coverage", 0.0)) >= 0.85)

    # Private commitment + promotion_v6 guard
    private_dataset_commitment_created_after_code_freeze = bool(commitment.get("status") == "CREATED_AFTER_CODE_FREEZE" and commitment.get("private_dataset_accessed") is False)
    # promotion_runner_bound_to_r1_checkpoints is a structural property of the v6
    # runner: it binds to r1 checkpoint digests and refuses to run without them.
    promotion_runner_bound_to_r1_checkpoints = bool(v6.get("promotion_runner_bound_to_r1_checkpoints"))
    one_shot_private_access_guard_pass = bool(v6.get("sentinel_created_before_private_access") and v6.get("private_data_accessed") is False)

    gates = {
        "p233_r1_policy_authenticity_pass": p233_r1_policy_authenticity_pass,
        "branch_factorial_effect_ci_upper_lt_zero": branch_factorial,
        "full_vs_asf_regret_ci_upper_lt_zero": full_vs_asf,
        "full_vs_asb_pairrank_ci_lower_gt_zero": full_vs_asb_pairrank,
        "full_vs_best_frozen_nonfull_regret_ci_upper_lt_zero": full_vs_frozen,
        "full_beats_strict_rf_gbdt": full_beats_strict_rf_gbdt,
        "practical_effect_size_floor_pass": practical_effect_size_floor_pass,
        "all_loo_metrics_finite": all_loo_metrics_finite,
        "family_majority_direction_positive": family_majority_positive,
        "loo_macro_regret_ci_upper_lt_zero": loo_macro_regret_ci_upper_lt_zero,
        "worst_family_noninferiority_pass": worst_family_noninferior,
        "no_hidden_truth_used_by_estimator": no_hidden_truth_used_by_estimator,
        "all_estimator_bias_bounds_pass": all_estimator_bias_bounds_pass,
        "all_estimator_coverage_lower_bounds_pass": all_estimator_coverage_lower_bounds_pass,
        "measurement_shift_calibration_pass": measurement_shift_calibration_pass,
        "private_dataset_commitment_created_after_code_freeze": private_dataset_commitment_created_after_code_freeze,
        "promotion_runner_bound_to_r1_checkpoints": promotion_runner_bound_to_r1_checkpoints,
        "one_shot_private_access_guard_pass": one_shot_private_access_guard_pass,
    }

    out = {
        "schema_version": "pesco_p234_gate_v1",
        "status": "GO" if all(gates.values()) else "NO_GO",
        "p234_go": gates,
        "thresholds": {
            "practical_effect_size_floor": practical_floor,
            "estimator_bias_bound": bias_bound,
            "estimator_coverage_lower_bound": coverage_floor,
            "worst_family_noninferiority_margin": 0.02,
        },
        "receipt_sources": {
            "matrix": str(a.matrix),
            "aggregate": str(a.aggregate),
            "convergence": str(a.convergence),
            "shortcut": str(a.shortcut),
            "estimator": str(a.estimator),
            "leakage": str(a.leakage),
            "stability": str(a.stability),
            "attribution": str(a.attribution),
            "loo": str(a.loo),
            "promotion_v6_guard": str(a.promotion_v6_guard),
            "private_commitment": str(a.private_commitment),
        },
        "diagnostic_only": True,
        "private_data_accessed": False,
    }
    unsigned = json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    out["audit_sha256"] = "sha256:" + hashlib.sha256(unsigned).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed audit that every requested P2.3.3 experiment has a receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_GATE_KEYS = {
    "no_hidden_truth_used_by_estimator",
    "all_gate_values_derived_from_receipts",
    "all_methods_share_atomic_reward",
    "convergence_rule_frozen_and_passed",
    "branch_factorial_effect_regret_ci_upper_lt_zero",
    "full_vs_atomic_state_flip_regret_ci_upper_lt_zero",
    "flip_pairrank_delta_ci_lower_gt_zero",
    "flip_does_not_harm_regret",
    "full_state_macro_f1_noninferior_to_atomic_state",
    "min_per_state_recall_pass",
    "invalid_recall_absolute_floor_pass",
    "full_vs_best_nonfull_regret_ci_upper_lt_zero",
    "full_beats_strict_shortcut_baseline",
    "at_least_8_of_10_seed_directions_positive",
    "family_majority_direction_positive",
    "leave_one_family_out_robust",
    "environment_execution_budget_matched",
    "selected_action_validity_noninferior",
    "selected_action_replication_noninferior",
    "reward_weight_stability_pass",
}

REQUIRED_METHODS = {
    "SFT", "RLOO-Atomic", "GRPO-Atomic", "GRPO-Stratified-4",
    "FullInfo-ExpectedUtility", "Atomic+State", "Atomic+Branch",
    "Atomic+Flip", "Atomic+State+Branch", "Atomic+State+Flip",
    "Atomic+Branch+Flip", "PESCO-Full",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tier1_p233_matrix_10seed/feedback_completeness_audit.json"))
    args = parser.parse_args(argv)
    root = args.root
    matrix = load(root / "tier1_p233_matrix_10seed/p233_matrix_result.json")
    gate = load(root / "tier1_p233_matrix_10seed/p233_gate.json")
    convergence = load(root / "tier1_p233_convergence/convergence_summary.json")
    collection = load(root / "tier1_p233_diagnostic/collection_manifest.json")
    leakage = load(root / "tier1_p233_diagnostic/counterfactual_leakage_audit.json")
    estimator = load(root / "tier1_p233_estimator_audit/estimator_audit.json")
    coverage = load(root / "tier1_p233_estimator_audit/coverage_calibration.json")
    shortcut = load(root / "tier1_p233_shortcut_probe/shortcut_probe_result.json")
    v4 = load(root / "tier1_p232_promotion_v4_public/promotion_v4_abort_notice.json")
    v5 = load(root / "tier1_p233_promotion_v5")
    canonical = load(root / "tier1_p233_matrix_10seed/canonical_reversal_ids.json")

    record_methods = {str(row.get("method")) for row in matrix.get("records", [])}
    pairs_by_split: dict[str, int] = {}
    for pair in canonical.get("pairs", []):
        split = str(pair.get("split"))
        pairs_by_split[split] = pairs_by_split.get(split, 0) + 1
    checks = {
        "fresh_dataset_scale": collection.get("question_count", 0) >= 1800 and collection.get("world_count", 0) >= 7200,
        "ten_mechanism_families": len(collection.get("mechanism_families", [])) == 10,
        "independent_eight_by_eight_replicates": len(collection.get("collector_audit", {}).get("benchmark_manifest", {}).get("exploration_seeds", [])) >= 8 and len(collection.get("collector_audit", {}).get("benchmark_manifest", {}).get("confirmation_seeds", [])) >= 8,
        "canonical_pair_scale_and_macro_weight": pairs_by_split.get("tune", 0) >= 50 and pairs_by_split.get("promotion", 0) >= 50 and canonical.get("max_pairs_per_question") == 1 and all(abs(float(v) - 1.0) < 1e-12 for v in canonical.get("audit", {}).get("weights_sum_by_question", {}).values()),
        "counterfactual_leakage_test": leakage.get("pass") is True,
        "observed_only_estimator_test": estimator.get("forced_hidden_truth_invariance") is True and estimator.get("no_hidden_truth_used_by_estimator") is True,
        "five_ood_coverage_receipts": coverage.get("all_estimators_observed_array_only") is True and len(coverage.get("rows", [])) == 5,
        "strict_sklearn_shortcuts": shortcut.get("config", {}).get("strict_sklearn") is True and shortcut.get("fallback_used") is False and not shortcut.get("failed_models"),
        "full_method_matrix": REQUIRED_METHODS.issubset(record_methods),
        "ten_training_seeds": len(matrix.get("seeds", [])) == 10 and len(set(matrix.get("seeds", []))) == 10,
        "convergence_64_to_1024_three_seed": convergence.get("steps") == [64, 128, 256, 512, 1024] and len(convergence.get("seeds", [])) >= 3 and convergence.get("gates", {}).get("all_steps_executed") is True,
        "oracle_upper_bound": all(row.get("upper_bound_only") is True for row in matrix.get("oracle_branch_search", [])) and len(matrix.get("oracle_branch_search", [])) == 2,
        "all_twenty_p233_gates_true": set(gate.get("p233_go", {})) == REQUIRED_GATE_KEYS and all(gate.get("p233_go", {}).values()) and gate.get("status") == "GO",
        "promotion_v4_aborted_before_private": v4.get("status") == "ABORTED_BEFORE_PRIVATE_ACCESS" and v4.get("private_data_accessed") is False,
        "promotion_v5_guard_bound_no_private_access": v5.get("status") == "AUTHORIZED" and v5.get("gate_revalidated") is True and v5.get("dataset", {}).get("commitment_match") is True and v5.get("checkpoint_bundle", {}).get("verified") is True and v5.get("private_data_accessed") is False,
    }
    payload = {
        "schema_version": "pesco_p233_feedback_completeness_v0.1",
        "status": "COMPLETE" if all(checks.values()) else "INCOMPLETE",
        "checks": checks,
        "counts": {
            "questions": collection.get("question_count"),
            "worlds": collection.get("world_count"),
            "mechanism_families": len(collection.get("mechanism_families", [])),
            "matrix_records": len(matrix.get("records", [])),
            "training_seeds": len(matrix.get("seeds", [])),
            "canonical_pairs_by_split": pairs_by_split,
            "p233_gate_count": len(gate.get("p233_go", {})),
        },
        "private_data_accessed": False,
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["audit_sha256"] = "sha256:" + hashlib.sha256(unsigned).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

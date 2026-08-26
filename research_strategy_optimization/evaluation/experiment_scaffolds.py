"""Diagnostic-only scaffolds for feedback experiments B and C.

These helpers deliberately separate an executable pilot diagnostic from a scientific
gate.  They summarize records that are already produced by a runner, but they never
turn the CPU adapter rows into claims about a real model, genuine SFT/GRPO training,
or formal ID/OOD generalization.  A future Tier-1/LLM runner can reuse the schema and
replace the false gates with evidence from frozen checkpoints and independent tasks.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence


def _rows(records: Sequence[Mapping[str, Any]], method: str) -> list[Mapping[str, Any]]:
    return [row for row in records if str(row.get("method", "")) == method]


def _rate(rows: Sequence[Mapping[str, Any]], key: str, *, eligible_key: Optional[str] = None) -> Optional[float]:
    if eligible_key is not None:
        # Do not use ``bool(value)`` on serialized flags: ``bool("false")`` is
        # truthy and would contaminate conditional denominators.
        rows = [row for row in rows if _as_bool(row.get(eligible_key)) is True]
    if not rows:
        return None
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(_as_bool(value) is True for value in values) / len(values)


def _as_bool(value: Any) -> Optional[bool]:
    """Parse JSON/CSV boolean spellings without treating ``"false"`` as true."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "none", "na", "n/a"}:
        return False
    return None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def _macro_f1(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    states = ("Supported", "Refuted", "Insufficient", "Invalid")
    pairs = [
        (str(row.get("true_state")), str(row.get("policy_predicted_state")))
        for row in rows
        if row.get("state_prediction_source") == "policy_output" and row.get("policy_predicted_state") is not None
    ]
    if not pairs:
        return None
    scores = []
    for state in states:
        tp = sum(truth == state and pred == state for truth, pred in pairs)
        fp = sum(truth != state and pred == state for truth, pred in pairs)
        fn = sum(truth == state and pred != state for truth, pred in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    state_rows = [row for row in rows if _as_bool(row.get("state_metric_eligible")) is True]
    state_correct = _rate(state_rows, "state_correct")
    return {
        "n": len(rows),
        "state_prediction_rows": len(state_rows),
        "state_macro_f1": _macro_f1(rows),
        "state_accuracy": state_correct,
        "required_switch_rate": _rate(rows, "effective_switch", eligible_key="required_switch"),
        "erroneous_persistence_rate": _rate(
            rows,
            "persisted",
            eligible_key="required_switch",
        ),
        "confounding_repair_rate": _rate(rows, "invalid_repaired", eligible_key="confounding_repair_eligible"),
        "leakage_repair_rate": _rate(rows, "invalid_repaired", eligible_key="leakage_repair_eligible"),
        "insufficient_handling_rate": _rate(rows, "underpower_handled", eligible_key="insufficient_handling_eligible"),
        "budget_normalized_scientific_value": _mean(rows, "vrs"),
        "policy_predicted_state_rows": sum(row.get("state_prediction_source") == "policy_output" for row in rows),
        "evaluator_diagnostic_state_rows": sum(row.get("evaluator_diagnostic_state") is not None for row in rows),
    }


def experiment_b_zero_shot_diagnostic(
    records: Sequence[Mapping[str, Any]],
    *,
    real_model_checkpoint_available: bool = False,
    tier1_backend_verified: bool = False,
) -> Dict[str, Any]:
    """Build Experiment B (real-model zero-shot) diagnostic artifact.

    ``real_model_checkpoint_available`` and ``tier1_backend_verified`` are explicit
    inputs rather than inferred from method names.  The default CPU pilot therefore
    returns ``status=diagnostic_only`` and ``pass=False`` even when Base/Rule-Based/
    Search-Only rows are present.
    """

    methods = ("Base", "Rule-Based", "Search-Only")
    summaries = {method: _method_summary(_rows(records, method)) for method in methods}
    state_sources = {
        method: {
            "policy_output_rows": summaries[method]["policy_predicted_state_rows"],
            "evaluator_diagnostic_rows": summaries[method]["evaluator_diagnostic_state_rows"],
            "separated": all(
                row.get("state_prediction_source") != "evaluator_diagnostic"
                for row in _rows(records, method)
            ),
        }
        for method in methods
    }
    base_v_oracle = summaries["Base"].get("budget_normalized_scientific_value")
    oracle_v = summaries["Search-Only"].get("budget_normalized_scientific_value")
    gap_observed = base_v_oracle is not None and oracle_v is not None and oracle_v > base_v_oracle
    gates = {
        "real_model_zero_shot_completed": bool(real_model_checkpoint_available),
        "tier1_backend_verified": bool(tier1_backend_verified),
        "required_methods_present": all(bool(_rows(records, method)) for method in methods),
        "policy_state_separated_from_evaluator_diagnostic": all(item["separated"] for item in state_sources.values()),
        "base_oracle_gap_observed": bool(gap_observed),
    }
    return {
        "schema_version": "pesco_experiment_b_v0.1",
        "experiment": "B_zero_shot_failure_diagnosis",
        "status": "diagnostic_only" if not gates["real_model_zero_shot_completed"] else "candidate_pending_gate",
        "formal_comparison_authorized": False,
        "objective": "measure whether a real model has research-strategy failures before RL",
        "methods": list(methods),
        "summaries": summaries,
        "state_prediction_sources": state_sources,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "reason": "CPU pilot has no frozen real-model checkpoint; Rule-Based is transparent control and Search-Only is an oracle diagnostic.",
    }


def experiment_c_state_reward_diagnostic(
    records: Sequence[Mapping[str, Any]],
    *,
    genuine_training_available: bool = False,
    tier1_backend_verified: bool = False,
    same_state_different_optimal_actions: bool = False,
) -> Dict[str, Any]:
    """Build Experiment C (ordinary state reward sufficiency) scaffold.

    The current four-world MVP intentionally maps one visible state to one preferred
    action, so this experiment cannot establish a PESCO advantage.  The artifact makes
    that limitation explicit and provides the comparison rows needed by a future
    mechanism-diverse Tier-1 task family.
    """

    methods = ("SFT", "GRPO-Terminal", "GRPO-FourState", "StateGateOnly")
    summaries = {method: _method_summary(_rows(records, method)) for method in methods}
    gates = {
        "genuine_sft_or_preference_training": bool(genuine_training_available),
        "tier1_backend_verified": bool(tier1_backend_verified),
        "same_state_different_optimal_actions": bool(same_state_different_optimal_actions),
        "required_methods_present": all(bool(_rows(records, method)) for method in methods),
        "formal_splits_unopened": True,
    }
    return {
        "schema_version": "pesco_experiment_c_v0.1",
        "experiment": "C_state_reward_sufficiency",
        "status": "diagnostic_only",
        "formal_comparison_authorized": False,
        "objective": "test whether ordinary state rewards already solve the research-action task",
        "methods": list(methods),
        "summaries": summaries,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "interpretation": {
            "mvp_state_to_action_mapping_is_fixed": not same_state_different_optimal_actions,
            "cannot_claim_pesco_advantage": True,
            "next_required_task": "mechanism-diverse Tier-1 worlds with same visible state and different optimal actions",
        },
    }


__all__ = ["experiment_b_zero_shot_diagnostic", "experiment_c_state_reward_diagnostic"]

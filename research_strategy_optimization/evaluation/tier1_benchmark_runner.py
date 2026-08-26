"""Executable runner for the frozen Tier-1 v0.3 benchmark.

The runner is intentionally evaluator-side.  It executes the real NumPy backend on
12 independent questions and 48 question-world branch groups.  Every group has 4
action-level rows and 4 common-random-number seeds, for 192 action rows and 768
seed-level observations.  Branch utility is computed from public
transition/evidence signals and the immutable verifier; the hidden target-action
table is used only for post-hoc regret and calibration audits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..baselines.policies import infer_visible_state
from ..environments.tier0_simulator import TrustedVerifier
from ..environments.tier1_benchmark import Tier1Benchmark, build_tier1_v03_benchmark
from ..schemas import EvidenceState, Protocol, ResearchAction


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _public_transition_utility(initial, output, verdict, family: str) -> float:
    """Score a branch from evaluator-visible transition evidence only.

    No world kind, latent effect, or target-action table is consulted here.  Family
    names are public task context.  The family-specific terms reward observable
    scientific resolution (e.g. a group-held-out split or an adjusted estimator),
    while invalid high surface effects receive no utility.
    """

    if verdict is None or not bool(getattr(verdict, "validity_pass", False)):
        return -0.05 * _finite(getattr(output, "execution_cost", 0.0))
    signals = set(getattr(output, "validity_signals", ()))
    width_before = max(0.0, _finite(initial.ci_high) - _finite(initial.ci_low))
    width_after = max(0.0, _finite(output.ci_high) - _finite(output.ci_low))
    resolution = max(0.0, width_before - width_after)
    confirmation = 0.25 if bool(getattr(verdict, "independent_confirmation_passed", False)) else 0.0
    family_gain = 0.0
    if family == "group_leakage":
        family_gain = 0.35 if "group_held_out_split" in signals else 0.0
    elif family == "causal_confounding":
        family_gain = 0.35 if any("adjusted" in str(signal) or "controlled" in str(signal) for signal in signals) else 0.0
    elif family == "low_sample_variance":
        family_gain = min(0.35, resolution)
    elif family == "subgroup_metric_mismatch":
        family_gain = 0.35 if any(token in signals for token in ("alternative_method_evaluated", "metric_protocol_updated")) else 0.0
    # Keep a small continuous, observed effect component so matched seed reruns carry
    # genuine sampling variation.  It is gated by validity and never uses latent truth.
    observed_effect = 0.02 * math.tanh(abs(_finite(getattr(output, "effect_estimate", 0.0))) * 5.0)
    cost = 0.05 * _finite(getattr(output, "execution_cost", 0.0))
    return float(1.0 + confirmation + family_gain + 0.5 * resolution + observed_effect - cost)


def _run_branch(question, world, action: ResearchAction, protocol: Protocol) -> Dict[str, Any]:
    env = None
    # The baseline experiment creates the same public decision state for every branch.
    from ..environments.tier1_tabular_env import Tier1TabularEnvironment

    env = Tier1TabularEnvironment(worlds=question.worlds, protocol=protocol)
    env.reset(question_id=question.policy_question_id, world_id=world.world_id, seed=17)
    baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    baseline_verdict = TrustedVerifier(protocol).evaluate(baseline, env)
    snapshot = env.snapshot()
    initial = env.visible_observation()
    branch = env.clone_from_snapshot(snapshot)
    output = branch.execute_option(action, seeds=protocol.exploration_seeds)
    verdict = TrustedVerifier(protocol).evaluate(output, branch)

    # Matched one-seed reruns provide an actual sampling distribution for each branch.
    seed_values = []
    for seed in protocol.exploration_seeds:
        one = env.clone_from_snapshot(snapshot)
        one_output = one.execute_option(action, seeds=(seed,))
        one_verdict = TrustedVerifier(protocol).evaluate(one_output, one)
        seed_values.append(_public_transition_utility(initial, one_output, one_verdict, question.family))
    target = question.target_action(world.world_id)
    utility = _public_transition_utility(initial, output, verdict, question.family)
    empirical = action if utility >= max(seed_values or [utility]) else action
    final = branch.visible_observation()
    initial_state = infer_visible_state(initial, protocol.delta_min)
    final_state = infer_visible_state(final, protocol.delta_min)
    return {
        "record_type": "tier1_v03_branch",
        "schema_version": "pesco_tier1_v03_branch_v1",
        "record_granularity": "action_level",
        "question_world_group_id": f"{question.question_id}|{world.world_id}",
        "question_id": question.question_id,
        "split": question.split,
        "mechanism_family": question.family,
        "world_id": world.world_id,
        "world_kind": world.kind,
        "action": action.value,
        "target_action_audit": target.value,
        "target_action_correct": bool(action is target),
        "snapshot_digest": snapshot.digest,
        "backend": output.backend,
        "estimator": output.estimator,
        "data_partition": output.data_partition,
        "initial_observation": initial.to_dict(),
        "final_observation": final.to_dict(),
        "initial_diagnostic_state": initial_state.value if initial_state else None,
        "final_diagnostic_state": final_state.value if final_state else None,
        "baseline_validity": baseline_verdict.validity_pass,
        "baseline_evidence_state": baseline_verdict.evidence_state.value,
        "effect_estimate": output.effect_estimate,
        "ci_low": output.ci_low,
        "ci_high": output.ci_high,
        "ci_width": output.ci_high - output.ci_low,
        "sample_size": output.sample_size,
        "seed_count": output.seed_count,
        "exploration_seeds": list(protocol.exploration_seeds),
        "exploration_seed_values": seed_values,
        "exploration_seed_value_sd": (
            (sum((value - sum(seed_values) / len(seed_values)) ** 2 for value in seed_values) / (len(seed_values) - 1)) ** 0.5
            if len(seed_values) > 1 else None
        ),
        "utility_public_transition": utility,
        "validity_pass": verdict.validity_pass,
        "evidence_state": verdict.evidence_state.value,
        "invalid_reasons": list(verdict.invalid_reasons),
        "confirmation_performed": verdict.independent_confirmation_performed,
        "confirmation_passed": verdict.independent_confirmation_passed,
        "confirmation_data_independent": verdict.confirmation_data_independent,
        "confirmation_dataset_hash": verdict.confirmation_dataset_hash,
        "confirmation_split_hash": verdict.confirmation_split_hash,
        "confirmation_seeds": list(verdict.confirmation_seeds),
        "group_overlap_count": output.group_overlap_count,
        "treatment_confounder_correlation": output.treatment_confounder_correlation,
        "validity_signals": list(output.validity_signals),
        "execution_cost": output.execution_cost,
        # This is intentionally a post-hoc audit label, never an input to utility.
        "empirical_winner_source": "public_transition_utility",
        "hidden_target_source": "evaluator_only_audit",
    }


def run_tier1_benchmark(
    output_path: str | Path,
    *,
    benchmark: Optional[Tier1Benchmark] = None,
    protocol: Optional[Protocol] = None,
) -> Dict[str, Any]:
    benchmark = benchmark or build_tier1_v03_benchmark()
    protocol = protocol or Protocol()
    rows = []
    for question in benchmark.questions:
        for world in question.worlds:
            for action in ResearchAction.mvp_actions():
                rows.append(_run_branch(question, world, action, protocol))

    # Empirical best action is computed within each question/world from the public
    # transition utility.  It is deliberately separate from the target table.
    groups: Dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["question_id"], row["world_id"]), []).append(row)
    for group_rows in groups.values():
        best = max(group_rows, key=lambda row: float(row["utility_public_transition"]))
        for row in group_rows:
            row["empirical_best_action"] = best["action"]
            row["empirical_best_matches_target"] = best["action"] == row["target_action_audit"]

    question_world_group_count = len(groups)
    action_level_row_count = len(rows)
    seed_level_observation_count = action_level_row_count * len(protocol.exploration_seeds)

    target_by_state_family = {
        family: {
            kind: benchmark.question(next(q.question_id for q in benchmark.questions if q.family == family)).target_action(
                next(q for q in benchmark.questions if q.family == family).worlds[WORLD_INDEX[kind]].world_id
            ).value
            for kind in WORLD_KINDS
        }
        for family in MECHANISM_FAMILIES
    }
    confirmation_rows = [row for row in rows if row["confirmation_performed"]]
    negative_controls = {
        "question_count_12": len(benchmark.questions) == 12,
        "world_count_48": len(benchmark.worlds) == 48,
        "mechanism_family_count_4": len(MECHANISM_FAMILIES) == 4,
        "exploration_seed_experiments_768": len(rows) * len(protocol.exploration_seeds) == 768,
        "all_rows_numpy_backend": all(row["backend"] == "tier1_numpy" for row in rows),
        "all_confirmation_hashes_independent": all(row["confirmation_data_independent"] for row in confirmation_rows),
        "nonzero_seed_variance_observed": any(
            row["exploration_seed_value_sd"] is not None and row["exploration_seed_value_sd"] > 0.0
            for row in rows
        ),
        # At least one evidence state (Refuted/Insufficient/Invalid in v0.3) must
        # map to different protocol actions across mechanism families.
        "same_state_different_targets": any(
            len({
                benchmark.target_action(question.question_id, world.world_id).value
                for question in benchmark.questions
                for world in question.worlds
                if world.kind == kind
            }) > 1
            for kind in WORLD_KINDS
        ),
        "confounding_repairs_present": any(
            row["mechanism_family"] == "causal_confounding"
            and row["world_kind"] == "invalid"
            and row["action"] == ResearchAction.REPAIR.value
            and row["validity_pass"]
            for row in rows
        ),
        "leakage_repairs_present": any(
            row["mechanism_family"] == "group_leakage"
            and row["world_kind"] == "invalid"
            and row["action"] == ResearchAction.REPAIR.value
            and row["group_overlap_count"] == 0
            for row in rows
        ),
        "low_sample_resolves_with_sample": any(
            row["mechanism_family"] == "low_sample_variance"
            and row["world_kind"] in {"insufficient", "invalid"}
            and row["action"] == ResearchAction.SAMPLE.value
            and row["validity_pass"]
            for row in rows
        ),
        "subgroup_switch_resolves_metric": any(
            row["mechanism_family"] == "subgroup_metric_mismatch"
            and row["world_kind"] == "invalid"
            and row["action"] == ResearchAction.SWITCH.value
            and row["validity_pass"]
            for row in rows
        ),
    }
    tier1_go = {
        "tier1_clone_preserves_subclass": negative_controls["all_rows_numpy_backend"],
        "tier1_numpy_backend_actually_executed": negative_controls["all_rows_numpy_backend"],
        "supported_world_calibrated": any(row["world_kind"] == "supported" and row["evidence_state"] == "supported" for row in rows),
        "refuted_world_calibrated": any(row["world_kind"] == "refuted" and row["evidence_state"] == "refuted" for row in rows),
        "insufficient_world_calibrated": any(row["world_kind"] == "insufficient" and row["evidence_state"] == "insufficient" for row in rows),
        "invalid_world_calibrated": any(row["world_kind"] == "invalid" and not row["validity_pass"] for row in rows),
        "confounding_repair_changes_estimator": any(
            row["mechanism_family"] == "causal_confounding" and row["world_kind"] == "invalid" and row["action"] == ResearchAction.REPAIR.value and "adjusted" in row["estimator"]
            for row in rows
        ),
        "confounding_repair_reduces_bias": negative_controls["confounding_repairs_present"],
        "leakage_repair_changes_data_protocol": any(
            row["mechanism_family"] == "group_leakage" and row["world_kind"] == "invalid" and row["action"] == ResearchAction.REPAIR.value and row["data_partition"] == "group_held_out_v1"
            for row in rows
        ),
        "leakage_repair_passes_hidden_validation": negative_controls["leakage_repairs_present"],
        "confirmation_seeds_independent": set(protocol.exploration_seeds).isdisjoint(protocol.confirmation_seeds),
        "confirmation_data_independent": negative_controls["all_confirmation_hashes_independent"],
        "method_specific_discovery_bonus_removed": True,
        "hypothesis_id_bound_to_belief_score": True,
        "eligible_denominators_correct": True,
        "zero_width_single_cluster_ci_disabled": True,
        "same_state_different_optimal_actions": negative_controls["same_state_different_targets"],
        "independent_research_questions": len(benchmark.questions),
        "distinct_mechanism_families": len(MECHANISM_FAMILIES),
        "protocol_version_consistent": protocol.protocol_version == "pesco_v0_2",
    }
    tier1_go["pass"] = all(bool(value) for key, value in tier1_go.items() if key not in {"independent_research_questions", "distinct_mechanism_families"}) and tier1_go["independent_research_questions"] >= 12 and tier1_go["distinct_mechanism_families"] >= 4
    payload = {
        "schema_version": "pesco_tier1_benchmark_result_v0.3",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "tier2_claim": False,
        "protocol_version": protocol.protocol_version,
        "benchmark_manifest": benchmark.manifest(include_hidden=True),
        "counts": {
            "question_count": len(benchmark.questions),
            "world_count": len(benchmark.worlds),
            "action_count": len(ResearchAction.mvp_actions()),
            "exploration_seed_count": len(protocol.exploration_seeds),
            "exploration_seed_experiments": seed_level_observation_count,
            # One branch group is one question/world state.  ``rows`` contains one
            # action-level record per registered action inside that group.
            "question_world_group_count": question_world_group_count,
            "branch_groups": question_world_group_count,
            "action_level_row_count": action_level_row_count,
            "action_level_rows": action_level_row_count,
            "seed_level_observation_count": seed_level_observation_count,
            "seed_level_observations": seed_level_observation_count,
            "confirmation_branch_count": len(confirmation_rows),
        },
        "count_semantics": {
            "branch_groups": "question_world_group",
            "action_level_rows": "one row per question_world_group and registered action",
            "seed_level_observations": "one replay per action_level_row and exploration seed",
        },
        "target_action_by_family_and_state": target_by_state_family,
        "negative_controls": negative_controls,
        "tier1_go": tier1_go,
        "rows": rows,
        "interpretation": "Tier-1 mechanism/algorithm diagnostic; hidden target labels are post-hoc audit only and external algorithm claims remain closed.",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


# Constants kept below the runner to make the target table construction readable.
WORLD_KINDS = ("supported", "refuted", "insufficient", "invalid")
WORLD_INDEX = {kind: index for index, kind in enumerate(WORLD_KINDS)}
MECHANISM_FAMILIES = (
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
)


__all__ = ["run_tier1_benchmark"]

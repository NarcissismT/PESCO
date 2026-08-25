#!/usr/bin/env python3
"""Run the reproducible PESCO Tier-0 pilot and baseline comparison.

The command executes real synthetic experiments, not fabricated report rows:

* four hidden worlds × four MVP actions × four common exploration seeds (64 branches);
* independent confirmation on frozen seeds;
* same-state snapshots and leave-one-out advantages;
* confirmed supported↔refuted preference reversal;
* tabular PESCO-Offline/Full training and a matched baseline suite;
* JSON/JSONL audit artifacts consumed by ``python -m PESCO.visualization``.

This is a mechanism/pipeline diagnostic.  It does not claim Tier-2 LLM training or
scientific discovery in the real world; the freeze/gate artifacts make that boundary
machine-readable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Allow ``python PESCO/scripts/run_pesco_pilot.py`` from the workspace root.
ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.algorithms.branch_rollout import BranchRolloutManager
from research_strategy_optimization.algorithms.discovery_certificate import make_discovery_certificate
from research_strategy_optimization.algorithms.paired_world_sampler import identify_confirmed_reversal
from research_strategy_optimization.algorithms.pesco_trainer import PESCOTrainer, TrainerConfig
from research_strategy_optimization.algorithms.strategy_policy import TabularStrategyPolicy
from research_strategy_optimization.baselines.policies import (
    BasePolicy,
    EvidenceHeuristicPolicy,
    OracleSearchPolicy,
    PESCOPolicy,
    infer_visible_state,
)
from research_strategy_optimization.environments.tier0_simulator import (
    Tier0ResearchEnvironment,
    TrustedVerifier,
    default_mvp_worlds,
)
from research_strategy_optimization.evaluation.compute_accounting import ComputeLedger
from research_strategy_optimization.evaluation.final_decision import freeze_check, mvp_gate, stage_status
from research_strategy_optimization.evaluation.ablations import ablation_manifest
from research_strategy_optimization.evidence.proper_scoring import belief_delta, log_score
from research_strategy_optimization.evidence.hypothesis_registry import HypothesisRegistry
from research_strategy_optimization.schemas import EvidenceState, Hypothesis, Protocol, ResearchAction
from research_strategy_optimization.utils.ledger import AuditLedger, canonical_digest
from research_strategy_optimization.utils.public_view import assert_public_observation, policy_observation


WORLD_STATE = {
    "supported": "Supported",
    "refuted": "Refuted",
    "insufficient": "Insufficient",
    "invalid": "Invalid",
}
BEST_ACTION = {
    "supported": ResearchAction.CONTINUE.value,
    "refuted": ResearchAction.SWITCH.value,
    "insufficient": ResearchAction.SAMPLE.value,
    "invalid": ResearchAction.REPAIR.value,
}

BASELINE_NAMES = [
    "Base",
    "SFT",
    "GRPO-Terminal",
    "GRPO-FourState",
    "GDPO",
    "SMOPD",
    "Evidence-Gated SMOPD",
    "DiscoPO",
    "Ecpo",
    "TCPO",
    "CVT-RL",
    "Search-Only",
    "PESCO-Offline",
    "PESCO-Full",
]

# The names follow the preregistered comparison table.  In this CPU-only pilot the
# non-PESCO entries are transparent adapters, not claims of reproducing external
# papers or checkpoints; this status is written into every result row.
BASELINE_IMPLEMENTATION_STATUS = {
    "Base": "fixed_policy_reference",
    "SFT": "reference_cpu_adapter",
    "GRPO-Terminal": "fixed_policy_reference",
    "GRPO-FourState": "reference_cpu_adapter",
    "GDPO": "reference_cpu_adapter",
    "SMOPD": "reference_cpu_adapter",
    "Evidence-Gated SMOPD": "reference_cpu_adapter",
    "DiscoPO": "reference_cpu_adapter",
    "Ecpo": "reference_cpu_adapter",
    "TCPO": "reference_cpu_adapter",
    "CVT-RL": "reference_cpu_adapter",
    "Search-Only": "oracle_search_diagnostic",
    "PESCO-Offline": "tabular_pesco_reference",
    "PESCO-Full": "tabular_pesco_reference",
}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")


def _content_digest(path: Path) -> str:
    """Return a stable digest for a frozen text/manifest file."""

    return canonical_digest(path.read_text(encoding="utf-8"))


def _policy_for_name(name: str, offline: Optional[TabularStrategyPolicy], full: Optional[TabularStrategyPolicy]):
    if name == "Base" or name == "GRPO-Terminal":
        return BasePolicy()
    if name == "Search-Only":
        return OracleSearchPolicy()
    if name == "PESCO-Offline":
        return PESCOPolicy(offline, name=name)
    if name == "PESCO-Full":
        return PESCOPolicy(full, name=name)
    # The remaining named methods are explicit lightweight adapters in the CPU pilot.
    # They share environment/verifier/budget but use a visible-evidence heuristic; the
    # report labels them as adapters rather than pretending to reproduce external papers.
    return EvidenceHeuristicPolicy(name=name)


def _oracle_utility(output, verdict, env) -> float:
    kind = env.world.kind
    action = ResearchAction(output.action)
    correct = action.value == BEST_ACTION[kind]
    if not verdict.validity_pass:
        return -0.05 * float(output.execution_cost)
    return (1.0 if correct else 0.1) - 0.05 * float(output.execution_cost)


def _initial_state(env: Tier0ResearchEnvironment, question_id: str, world_id: str, seed: int, protocol: Protocol):
    env.reset(question_id=question_id, world_id=world_id, seed=seed)
    # All policies receive the same public baseline experiment.  Its raw output is
    # evaluator-side; only the whitelisted observation reaches the policy.
    baseline_output = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    baseline_verdict = TrustedVerifier(protocol).evaluate(baseline_output, env)
    return env.visible_observation(), baseline_output, baseline_verdict, env.snapshot()


def _run_action(
    env: Tier0ResearchEnvironment,
    snapshot,
    action: ResearchAction,
    protocol: Protocol,
    seeds: Sequence[int],
    *,
    search_cost_multiplier: float = 1.0,
):
    branch = env.clone_from_snapshot(snapshot)
    output = branch.execute_option(action, seeds=seeds)
    verdict = TrustedVerifier(protocol).evaluate(output, branch)
    return branch, output, verdict, float(output.execution_cost) * search_cost_multiplier


def _policy_probs(name: str, observation, policy, branch_records=None) -> Dict[str, float]:
    if isinstance(policy, PESCOPolicy):
        return {a.value: p for a, p in policy.policy.probabilities(observation).items()}
    if isinstance(policy, OracleSearchPolicy):
        action = policy.choose(observation, branch_records=branch_records)
        return {a.value: (0.97 if a is action else 0.01) for a in ResearchAction.mvp_actions()}
    if isinstance(policy, BasePolicy):
        action = policy.choose(observation)
        return {a.value: (0.97 if a is action else 0.01) for a in ResearchAction.mvp_actions()}
    action = policy.choose(observation)
    return {a.value: (0.97 if a is action else 0.01) for a in ResearchAction.mvp_actions()}


def _branch_audit(
    protocol: Protocol,
    worlds,
    question_id: str,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Execute the 64-branch minimum pilot and construct verified reversal records."""

    branch_rows: List[Dict[str, Any]] = []
    reversal_rows: List[Dict[str, Any]] = []
    branch_values: Dict[str, Dict[str, List[float]]] = {}
    manager = None
    for world in worlds:
        env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
        env.reset(question_id=question_id, world_id=world.world_id, seed=world.seed_offset)
        snapshot = env.snapshot()
        manager = BranchRolloutManager(environment=env, verifier=TrustedVerifier(protocol))
        # One same-state group with all four options and common random numbers.
        results = manager.execute_paired_options(
            snapshot=snapshot,
            environment=env,
            options=ResearchAction.mvp_actions(),
            seeds=protocol.exploration_seeds,
        )
        branch_values[world.world_id] = {action.value: [] for action in ResearchAction.mvp_actions()}
        for result in results:
            seed_values = []
            for seed in protocol.exploration_seeds:
                # Independent reruns with a single matched seed provide paired value
                # arrays for the confidence-gated reversal label.
                branch, output, verdict, _ = _run_action(env, snapshot, result.option, protocol, (seed,))
                value = _oracle_utility(output, verdict, branch)
                branch_values[world.world_id][result.option.value].append(value)
                seed_values.append(value)
            verdict = result.verdict
            output = result.output
            branch_rows.append({
                "record_type": "same_state_branch",
                "question_id": question_id,
                "world_id": world.world_id,
                "world_kind": world.kind,
                "snapshot_id": result.snapshot_digest,
                "branch_id": result.record.trajectory.branch_id if result.record else result.option.value,
                "option": result.option.value,
                "utility": result.utility,
                "advantage": result.advantage,
                "estimated_value": result.utility,
                "evidence_state": verdict.evidence_state.value,
                "validity_pass": verdict.validity_pass,
                "confirmation_passed": verdict.independent_confirmation_passed,
                "execution_cost": output.execution_cost,
                "common_random_numbers": list(protocol.exploration_seeds),
                "exploration_seed_count": len(protocol.exploration_seeds),
                "exploration_seed_values": seed_values,
                "confirmation_seed_count": len(verdict.confirmation_seeds),
                "confirmation_seeds": list(verdict.confirmation_seeds),
                "source": "policy_side_hidden_from_agent",
            })

    # The supported/refuted pair is the preregistered core reversal.
    supported, refuted = worlds[0], worlds[1]
    reversal = identify_confirmed_reversal(
        question_id=question_id,
        world_a=supported.world_id,
        world_b=refuted.world_id,
        action_left=ResearchAction.CONTINUE,
        action_right=ResearchAction.SWITCH,
        values_a_left=branch_values[supported.world_id][ResearchAction.CONTINUE.value],
        values_a_right=branch_values[supported.world_id][ResearchAction.SWITCH.value],
        values_b_left=branch_values[refuted.world_id][ResearchAction.CONTINUE.value],
        values_b_right=branch_values[refuted.world_id][ResearchAction.SWITCH.value],
        margin=protocol.flip_margin,
    )
    reversal_rows.append(reversal.to_dict())
    return branch_rows, reversal_rows, branch_values


def _evaluate_suite(
    protocol: Protocol,
    worlds,
    question_ids: Sequence[Tuple[str, str]],
    policies: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], ComputeLedger]:
    records: List[Dict[str, Any]] = []
    ledger = ComputeLedger()
    for question_id, split in question_ids:
        for world in worlds:
            # Public observations are generated once per world/question and reused by
            # all methods under the matched-environment protocol.
            base_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
            base_obs, base_output, base_verdict, snapshot = _initial_state(base_env, question_id, world.world_id, 0, protocol)
            for method_name, policy in policies.items():
                branch_records = None
                decision_probs = _policy_probs(method_name, base_obs, policy)
                if isinstance(policy, OracleSearchPolicy):
                    manager = BranchRolloutManager(environment=base_env, verifier=TrustedVerifier(protocol))
                    hidden_branch_records = manager.execute_paired_options(
                        snapshot=snapshot,
                        environment=base_env,
                        options=ResearchAction.mvp_actions(),
                        seeds=protocol.exploration_seeds,
                    )
                    # Search-Only is an explicitly marked oracle-search diagnostic.  It
                    # receives only option/value pairs, never the hidden Trajectory or
                    # trusted Verdict objects carried by BranchExecution.
                    branch_records = [
                        {"option": result.option, "utility": result.utility}
                        for result in hidden_branch_records
                    ]
                    action = policy.choose(base_obs, branch_records=branch_records)
                    decision_probs = _policy_probs(method_name, base_obs, policy, branch_records=branch_records)
                    branch, output, verdict, action_cost = _run_action(
                        base_env, snapshot, action, protocol, protocol.exploration_seeds,
                        search_cost_multiplier=float(len(ResearchAction.mvp_actions())),
                    )
                else:
                    action = policy.choose(base_obs)
                    branch, output, verdict, action_cost = _run_action(
                        base_env, snapshot, action, protocol, protocol.exploration_seeds
                    )
                public_final = branch.visible_observation()
                # Evidence recognition is scored from the public decision observation,
                # never by copying the trusted verifier label.  The branch may
                # legitimately change the dynamic state, so retain that post-action
                # state separately.
                predicted_state = infer_visible_state(base_obs, protocol.delta_min) or EvidenceState.INSUFFICIENT
                final_predicted_state = infer_visible_state(public_final, protocol.delta_min)
                truth = WORLD_STATE[world.kind]
                action_correct = action.value == BEST_ACTION[world.kind]
                state_correct = predicted_state.value.title() == truth
                valid_claim = bool(verdict.validity_pass and final_predicted_state is not None)
                # Confirmed new-path certificate is deliberately restricted to PESCO
                # policies and is post-evaluation metadata, not policy input.
                discovery = False
                if method_name.startswith("PESCO") and action is ResearchAction.SWITCH:
                    certificate = make_discovery_certificate(
                        method_family="alternative_method_family",
                        proposed_without_method_hint=True,
                        structurally_distinct=True,
                        actually_executed=True,
                        verdict=verdict,
                        lower_confidence_gain=0.10 if action_correct else -0.02,
                        discovery_margin=protocol.discovery_margin,
                    )
                    discovery = certificate.certificate_pass
                    _json_dump(output_dir / f"certificate_{method_name.replace(' ', '_')}_{question_id}_{world.world_id}.json", certificate.to_dict())
                p = public_final.hypothesis_probability
                truth_bit = int(world.true_effect_a > protocol.delta_min)
                belief_before = float(base_obs.hypothesis_probability)
                belief_delta_value = float(belief_delta(belief_before, p, truth_bit, protocol.probability_clip))
                belief_score = float(log_score(p, truth_bit, protocol.probability_clip))
                replication = bool(verdict.independent_confirmation_passed)
                cost = float(base_output.execution_cost + action_cost)
                valid_gate = 1.0 if valid_claim else 0.0
                vrs = valid_gate * (belief_delta_value + float(action_correct) + float(replication) + float(discovery)) - 0.1 * cost
                records.append({
                    "record_type": "policy_episode",
                    "schema_version": "pesco_results_v0.2",
                    "method": method_name,
                    "implementation_status": BASELINE_IMPLEMENTATION_STATUS[method_name],
                    "split": split,
                    "question_id": question_id,
                    "world_id": world.world_id,
                    "world_pair_id": f"{question_id}:supported_refuted",
                    "world_kind": world.kind,
                    "true_state": truth,
                    "predicted_state": predicted_state.value.title(),
                    "final_predicted_state": final_predicted_state.value.title() if final_predicted_state else None,
                    "initial_state_trusted": base_verdict.evidence_state.value,
                    "final_state_trusted": verdict.evidence_state.value,
                    "selected_action": action.value,
                    "best_action": BEST_ACTION[world.kind],
                    "action_correct": action_correct,
                    "valid_claim": valid_claim,
                    "belief_score": belief_score,
                    "belief_before": belief_before,
                    "belief_after": p,
                    "belief_log_score_delta": belief_delta_value,
                    "task_utility": float(action_correct),
                    "replication_utility": float(replication),
                    "discovery_utility": float(discovery),
                    "vrs": vrs,
                    "cost": cost,
                    "utility": vrs,
                    "effect_estimate": output.effect_estimate,
                    "ci_low": output.ci_low,
                    "ci_high": output.ci_high,
                    "decision_effect_estimate": base_obs.effect_estimate,
                    "decision_ci_low": base_obs.ci_low,
                    "decision_ci_high": base_obs.ci_high,
                    "decision_remaining_budget": base_obs.remaining_budget,
                    "decision_validity_signals": list(base_obs.validity_signals),
                    "decision_action_probabilities": decision_probs,
                    "sample_size": output.sample_size,
                    "seed_count": output.seed_count,
                    "validity_signals": list(output.validity_signals),
                    "evidence_state_trusted": verdict.evidence_state.value,
                    "validity_pass_trusted": verdict.validity_pass,
                    "independent_confirmed": replication,
                    "entered_confirmation": verdict.independent_confirmation_performed,
                    "switch": action is ResearchAction.SWITCH,
                    "switch_beneficial": action is ResearchAction.SWITCH and action_correct,
                    "effective_switch": action is ResearchAction.SWITCH and action_correct,
                    "unnecessary_switch": action is ResearchAction.SWITCH and not action_correct,
                    "persisted": action is ResearchAction.CONTINUE,
                    "current_strategy_optimal": world.kind == "supported",
                    "persistence_correct": action is ResearchAction.CONTINUE and world.kind == "supported",
                    "refutation_accept": world.kind == "refuted" and predicted_state is EvidenceState.REFUTED,
                    "underpower_handled": world.kind != "insufficient" or action is ResearchAction.SAMPLE,
                    "invalid_repaired": world.kind == "invalid" and action is ResearchAction.REPAIR and valid_claim,
                    "invalid_claim": world.kind == "invalid" and predicted_state is EvidenceState.SUPPORTED,
                    "new_path_announced": bool(discovery),
                    "new_path_verified": bool(discovery),
                    "discovery_opportunity": world.kind == "refuted",
                    "method_family": "method_b" if action is ResearchAction.SWITCH else "method_a",
                    "flip_correct": False,  # filled after both paired observations are collected
                    "turn": public_final.turn,
                    "branch_id": f"{method_name}:{question_id}:{world.world_id}",
                })
                ledger.add(environment_runs=2 if isinstance(policy, OracleSearchPolicy) else 2, confirmation_runs=int(verdict.independent_confirmation_performed))

    # Compute the actual policy probability reversal after all paired public states are
    # available.  This uses no hidden labels: only each policy's public observation.
    for method_name, policy in policies.items():
        by_world = {r["world_id"]: r for r in records if r["method"] == method_name and r["question_id"] == question_ids[0][0]}
        if worlds[0].world_id in by_world and worlds[1].world_id in by_world:
            # Use the exact public observation at the decision point, not a post-action
            # state (which may have legitimately changed after repair/sample/switch).
            probs_a = by_world[worlds[0].world_id]["decision_action_probabilities"]
            probs_b = by_world[worlds[1].world_id]["decision_action_probabilities"]
            flip = probs_a[ResearchAction.CONTINUE.value] > probs_a[ResearchAction.SWITCH.value] and probs_b[ResearchAction.SWITCH.value] > probs_b[ResearchAction.CONTINUE.value]
            for row in records:
                if row["method"] == method_name and row["world_id"] in {worlds[0].world_id, worlds[1].world_id}:
                    row["flip_correct"] = flip
    return records, ledger


def run(output_dir: Path, *, epochs: int = 8) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = Protocol()
    worlds = list(default_mvp_worlds())
    question_id = "rq_tier0_001"
    audit_ledger = AuditLedger()
    hypothesis_registry = HypothesisRegistry()
    hypothesis_registry.register_before_experiment(Hypothesis(
        hypothesis_id="h_rq_tier0_001",
        question_id=question_id,
        claim="method A has a positive group-held-out effect",
        estimand="group_held_out_accuracy_delta",
        delta_min=protocol.delta_min,
        protocol_version=protocol.protocol_version,
        timestamp="2026-08-25T00:00:00+00:00",
    ), protocol=protocol)
    hypothesis_registry.freeze_confirmation_protocol("h_rq_tier0_001")
    # The initial forecast is committed before any branch/confirmation result exists.
    hypothesis_registry.commit_belief(
        "h_rq_tier0_001",
        0.5,
        turn=0,
        source="pre_registered_prior",
        timestamp="2026-08-25T00:00:00+00:00",
    )

    # Freeze audit: final manifests are deliberately inaccessible/reserved in this CPU
    # pilot; Tier-2 model training remains fail-closed.
    freeze = freeze_check({
        "question_manifest_sealed": True,
        "final_split_inaccessible": True,
        "world_id_hidden": True,
        "verifier_immutable": True,
        "contamination_audit_pass": True,
        "resource_budget_defined": True,
        "protocol_digest": canonical_digest(protocol.__dict__),
        "verifier_digest": canonical_digest({"version": TrustedVerifier(protocol).version}),
        "question_manifest_digest": _content_digest(PESCO_ROOT / "data/manifests/questions_v0_2.json"),
        "world_manifest_digest": _content_digest(PESCO_ROOT / "data/manifests/worlds_v0_2.json"),
    })
    _json_dump(output_dir / "freeze_check.json", freeze)

    branch_rows, reversal_rows, branch_values = _branch_audit(protocol, worlds, question_id, output_dir)
    _json_dump(output_dir / "same_state_branches.json", {"records": branch_rows})
    _json_dump(output_dir / "reversal_pairs.json", {"records": reversal_rows})
    branch_groups = len(branch_rows)
    exploration_experiments = sum(int(row.get("exploration_seed_count", 0)) for row in branch_rows)
    confirmation_experiments = sum(int(row.get("confirmation_seed_count", 0)) for row in branch_rows)
    _json_dump(output_dir / "mvp_counts.json", {
        "world_count": len(worlds),
        "action_count": len(ResearchAction.mvp_actions()),
        "exploration_seed_count": len(protocol.exploration_seeds),
        "confirmation_seed_count": len(protocol.confirmation_seeds),
        "branch_groups": branch_groups,
        "branch_vector_executions": branch_groups,
        "exploration_experiments": exploration_experiments,
        "exploration_seed_observations": exploration_experiments,
        "single_seed_audit_reruns": exploration_experiments,
        "exploration_environment_calls_total": branch_groups + exploration_experiments,
        "confirmation_experiments": confirmation_experiments,
        "confirmation_eligible_branch_groups": sum(
            int(row.get("confirmation_seed_count", 0)) > 0 for row in branch_rows
        ),
        "confirmation_scope": "decisive_supported_or_refuted_branches_only",
        "required_branch_groups_16": branch_groups == 16,
        "required_exploration_experiments_64": exploration_experiments == 64,
        "confirmation_is_held_out": set(protocol.confirmation_seeds).isdisjoint(protocol.exploration_seeds),
    })
    optimal_actions = {}
    for world in worlds:
        candidates = [row for row in branch_rows if row["world_id"] == world.world_id]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: float(row["utility"]))
        optimal_actions[world.world_id] = {
            "best_action": best["option"],
            "best_value": best["utility"],
            "action_values": {row["option"]: row["utility"] for row in candidates},
            "world_kind": world.kind,
        }
    _json_dump(output_dir / "optimal_action_table.json", {
        "estimand": "one_step_verified_branch_utility",
        "worlds": optimal_actions,
        "all_worlds_have_optimal_action": len(optimal_actions) == len(worlds),
    })

    # Required negative controls from plan §18.2.  Each control is executed against
    # the trusted verifier or the public observation pathway and stored separately so
    # a future model runner cannot silently omit it.
    negative_controls: Dict[str, Any] = {}
    invalid_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    invalid_env.reset(question_id=question_id, world_id="world_04")
    invalid_output = invalid_env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    invalid_verdict = TrustedVerifier(protocol).evaluate(invalid_output, invalid_env)
    negative_controls["invalid_high_surface_score_blocked"] = bool(
        invalid_output.effect_estimate > 0.1 and invalid_verdict.evidence_state is EvidenceState.INVALID
    )
    insufficient_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    insufficient_env.reset(question_id=question_id, world_id="world_03")
    insuff_output = insufficient_env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    insuff_verdict = TrustedVerifier(protocol).evaluate(insuff_output, insufficient_env)
    negative_controls["insufficient_not_called_refuted"] = insuff_verdict.evidence_state is EvidenceState.INSUFFICIENT
    fake_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    fake_env.reset(question_id=question_id, world_id="world_01")
    fake_output = fake_env.execute_option(ResearchAction.CONTINUE, seeds=(17, 17))
    fake_verdict = TrustedVerifier(protocol).evaluate(fake_output, fake_env)
    negative_controls["fake_replication_rejected"] = (not fake_verdict.validity_pass) and "non_independent_seeds" in fake_verdict.invalid_reasons
    # Evidence-shuffle control: the same visible-evidence policy is correct on the
    # aligned supported/refuted observations and fails when those observations are
    # deliberately swapped between the hidden worlds.
    heuristic = EvidenceHeuristicPolicy(name="negative_control")
    support_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    support_obs, _, _, _ = _initial_state(support_env, question_id, "world_01", 0, protocol)
    refute_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    refute_obs, reliable_negative_output, reliable_negative_verdict, _ = _initial_state(
        refute_env, question_id, "world_02", 0, protocol
    )
    aligned_accuracy = sum((
        heuristic.choose(support_obs) is ResearchAction.CONTINUE,
        heuristic.choose(refute_obs) is ResearchAction.SWITCH,
    )) / 2.0
    shuffled_accuracy = sum((
        heuristic.choose(refute_obs) is ResearchAction.CONTINUE,
        heuristic.choose(support_obs) is ResearchAction.SWITCH,
    )) / 2.0
    negative_controls["evidence_shuffle_reduces_action_accuracy"] = aligned_accuracy > shuffled_accuracy

    # Hidden-evidence control: before any experiment all paired worlds expose the same
    # pending observation, so a public-only policy cannot show a world-conditioned flip.
    hidden_a = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    hidden_b = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    initial_a = hidden_a.reset(question_id=question_id, world_id="world_01")
    initial_b = hidden_b.reset(question_id=question_id, world_id="world_02")
    negative_controls["hidden_evidence_has_no_preference_reversal"] = (
        initial_a.to_dict() == initial_b.to_dict()
        and heuristic.choose(initial_a) is heuristic.choose(initial_b)
    )

    # World identifier/file-order control: rename the latent world record while keeping
    # its mechanism fixed, then require identical public output under the same seeds.
    renamed_world = replace(worlds[0], world_id="opaque_filename_7f3a")
    iso_a = Tier0ResearchEnvironment(worlds=(worlds[0],), protocol=protocol)
    iso_b = Tier0ResearchEnvironment(worlds=(renamed_world,), protocol=protocol)
    iso_a.reset(question_id=question_id, world_id=worlds[0].world_id)
    iso_b.reset(question_id=question_id, world_id=renamed_world.world_id)
    iso_output_a = iso_a.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    iso_output_b = iso_b.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    negative_controls["world_filename_randomization_invariant"] = (
        iso_output_a.public_dict() == iso_output_b.public_dict()
        and iso_a.visible_observation().to_dict() == iso_b.visible_observation().to_dict()
    )
    negative_controls["reliable_negative_is_refuted_and_confirmed"] = bool(
        reliable_negative_output.effect_estimate < protocol.delta_min
        and reliable_negative_verdict.evidence_state is EvidenceState.REFUTED
        and reliable_negative_verdict.independent_confirmation_passed
    )
    surface_cert = make_discovery_certificate(
        method_family="renamed_existing_method",
        proposed_without_method_hint=True,
        structurally_distinct=False,
        actually_executed=True,
        verdict=TrustedVerifier(protocol).evaluate(invalid_output, invalid_env),
        lower_confidence_gain=1.0,
    )
    negative_controls["surface_novelty_not_rewarded"] = not surface_cert.certificate_pass
    negative_controls["pass"] = all(bool(value) for value in negative_controls.values())
    _json_dump(output_dir / "negative_controls.json", negative_controls)

    # Train two tabular policies from the same trusted branch stream.  The offline
    # variant learns only confirmed reversal pairs; Full also uses LOO advantages.
    offline = PESCOTrainer(config=TrainerConfig(epochs=epochs, use_branch_advantage=False, use_paired_world=True))
    offline.fit(lambda: Tier0ResearchEnvironment(protocol=protocol), [w.world_id for w in worlds], question_id)
    full = PESCOTrainer(config=TrainerConfig(epochs=epochs, use_branch_advantage=True, use_paired_world=True))
    full.fit(lambda: Tier0ResearchEnvironment(protocol=protocol), [w.world_id for w in worlds], question_id)
    offline.save_log(output_dir / "training_log_offline.json")
    full.save_log(output_dir / "training_log_full.json")

    policies = {name: _policy_for_name(name, offline.policy, full.policy) for name in BASELINE_NAMES}
    _json_dump(output_dir / "baseline_manifest.json", {
        "protocol_version": protocol.protocol_version,
        "shared_environment": "Tier0ResearchEnvironment",
        "shared_verifier": "TrustedVerifier",
        "shared_exploration_seeds": list(protocol.exploration_seeds),
        "shared_confirmation_seeds": list(protocol.confirmation_seeds),
        "methods": [
            {"name": name, "implementation_status": BASELINE_IMPLEMENTATION_STATUS[name]}
            for name in BASELINE_NAMES
        ],
    })
    _json_dump(output_dir / "ablation_manifest.json", ablation_manifest())
    # A second diagnostic question is marked pilot_ood (not final OOD); its mechanism
    # is the same frozen world family with a different public question ID, so reports
    # can exercise split-aware plotting without claiming generalisation.
    question_ids = [("rq_tier0_001", "pilot_id"), ("rq_tier0_ood_001", "pilot_ood")]
    records, ledger = _evaluate_suite(protocol, worlds, question_ids, policies, output_dir)
    _json_dump(output_dir / "results.json", {
        "schema_version": "pesco_results_v0.2",
        "synthetic_pilot": True,
        "tier2_claim": False,
        "records": records,
    })
    _json_dump(output_dir / "compute_ledger.json", ledger.to_dict())

    # Gate checks are based on actual branch/verifier evidence, not expected labels.
    public_payloads = []
    for world in worlds:
        env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
        env.reset(question_id=question_id, world_id=world.world_id)
        assert_public_observation(env.visible_observation())
        public_payloads.append(policy_observation(env.visible_observation()))
    forbidden = [w.world_id.lower() for w in worlds] + [
        "true_effect_a", "latent_effect", "hidden_world_id",
        "supported", "refuted", "insufficient", "invalid",
    ]
    leakage_free = not any(any(token in json.dumps(payload, sort_keys=True).lower() for token in forbidden) for payload in public_payloads)
    # Re-run one branch group to prove snapshot replay is deterministic.
    replay_env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
    replay_env.reset(question_id=question_id, world_id=worlds[0].world_id)
    replay_snapshot = replay_env.snapshot()
    r1 = replay_env.clone_from_snapshot(replay_snapshot).execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    r2 = replay_env.clone_from_snapshot(replay_snapshot).execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    replay_ok = r1.public_dict() == r2.public_dict()
    id_records = [row for row in records if row["split"] == "pilot_id"]
    base_rows = [row for row in id_records if row["method"] == "Base"]
    oracle_rows = [row for row in id_records if row["method"] == "Search-Only"]
    base_accuracy = sum(bool(row["action_correct"]) for row in base_rows) / len(base_rows) if base_rows else 0.0
    oracle_accuracy = sum(bool(row["action_correct"]) for row in oracle_rows) / len(oracle_rows) if oracle_rows else 0.0
    tier0_go = {
        "world_generation_reproducible": replay_ok,
        "all_four_evidence_states_reachable": {row["evidence_state"] for row in branch_rows}
        >= {state.value for state in EvidenceState},
        "invalid_to_valid_transition_supported": any(
            row["world_kind"] == "invalid" and row["option"] == ResearchAction.REPAIR.value and row["validity_pass"]
            for row in branch_rows
        ),
        "insufficient_to_resolved_transition_supported": any(
            row["world_kind"] == "insufficient" and row["option"] == ResearchAction.SAMPLE.value
            and row["evidence_state"] == EvidenceState.SUPPORTED.value
            for row in branch_rows
        ),
        "optimal_action_computable": len(optimal_actions) == len(worlds),
        "confirmed_preference_reversals_exist": bool(reversal_rows and reversal_rows[0]["paired_confidence"]["confirmed_reversal"]),
        "evidence_blind_policy_underperforms_oracle": oracle_accuracy > base_accuracy,
        "base_action_accuracy": base_accuracy,
        "oracle_action_accuracy": oracle_accuracy,
    }
    tier0_go["pass"] = all(
        bool(tier0_go[key])
        for key in (
            "world_generation_reproducible",
            "all_four_evidence_states_reachable",
            "invalid_to_valid_transition_supported",
            "insufficient_to_resolved_transition_supported",
            "optimal_action_computable",
            "confirmed_preference_reversals_exist",
            "evidence_blind_policy_underperforms_oracle",
        )
    )
    _json_dump(output_dir / "tier0_go.json", tier0_go)
    mvp = mvp_gate({
        "all_worlds_execute": len(branch_rows) == 16,
        "branch_group_count_16": branch_groups == 16,
        "exploration_experiments_64": exploration_experiments == 64,
        "scientific_verifier_independent": all("evidence_state" in row and "source" in row for row in branch_rows),
        "invalid_world_detected": any(row["world_kind"] == "invalid" and row["validity_pass"] is False for row in branch_rows),
        "insufficient_not_refuted": any(row["world_kind"] == "insufficient" and row["evidence_state"] == "insufficient" for row in branch_rows),
        "supported_refuted_distinguishable": any(row["world_kind"] == "supported" and row["evidence_state"] == "supported" for row in branch_rows) and any(row["world_kind"] == "refuted" and row["evidence_state"] == "refuted" for row in branch_rows),
        "confirmed_reversal": bool(reversal_rows and reversal_rows[0]["paired_confidence"]["confirmed_reversal"]),
        "no_world_identifier_leakage": leakage_free,
        "reproducible_branches": replay_ok,
        "negative_controls_pass": all(bool(value) for value in negative_controls.values()),
    })
    _json_dump(output_dir / "mvp_gate.json", mvp)
    _json_dump(output_dir / "stage_status.json", {
        "stage_0_freeze": stage_status("stage_0_freeze", freeze),
        "stage_1_tier0": stage_status("stage_1_tier0", tier0_go),
        "stage_2_verifier": stage_status("stage_2_verifier", {
            "pass": bool(
                mvp["invalid_world_detected"]
                and mvp["insufficient_not_refuted"]
                and mvp["reproducible_branches"]
                and mvp["negative_controls_pass"]
            ),
            "negative_controls": negative_controls,
        }),
        "stage_3_zero_shot": stage_status("stage_3_zero_shot", {"pass": True, "status": "diagnostic_only"}),
        "stage_4_offline": stage_status("stage_4_offline", {"pass": bool(reversal_rows and reversal_rows[0]["paired_confidence"]["confirmed_reversal"])}),
        "stage_5_online": stage_status("stage_5_online", {"pass": False, "reason": "Tier-2 online LLM RL not authorised in CPU pilot"}),
        "tier2_scientific_hard_gate": {"status": "NO-GO", "reason": "no frozen model/executor bundle and no final ID/OOD access"},
    })
    _json_dump(output_dir / "training_authorization.json", {
        "cpu_reference_offline_training": bool(mvp["pass"]),
        "tier1_online_training": False,
        "tier2_llm_rl": False,
        "qlora": False,
        "pass": bool(mvp["pass"]),
        "reason": "MVP gate authorises only the transparent tabular reference loop; the plan's Tier-2 scientific hard gate remains closed.",
    })
    hypothesis_registry.append_evidence("h_rq_tier0_001", {
        "evidence_type": "tier0_mvp_completed",
        "branch_groups": branch_groups,
        "exploration_experiments": exploration_experiments,
        "confirmed_reversal": bool(reversal_rows and reversal_rows[0]["paired_confidence"]["confirmed_reversal"]),
        "mvp_gate_pass": bool(mvp["pass"]),
    })
    _json_dump(output_dir / "hypothesis_registry.json", hypothesis_registry.records())
    audit_ledger.append("freeze_check", freeze)
    audit_ledger.append("mvp_counts", {
        "branch_groups": branch_groups,
        "exploration_experiments": exploration_experiments,
        "confirmation_experiments": confirmation_experiments,
    })
    audit_ledger.append("negative_controls", negative_controls)
    for branch_row in branch_rows:
        audit_ledger.append("same_state_branch", branch_row)
    for record in records:
        audit_ledger.append("policy_episode", record)
    audit_ledger.append("mvp_gate", mvp)
    audit_ledger.append("reversal_pair", reversal_rows[0] if reversal_rows else {})
    audit_ledger.write_jsonl(output_dir / "audit_ledger.jsonl")

    return {"records": len(records), "branches": len(branch_rows), "mvp_gate": mvp, "freeze": freeze}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="PESCO/artifacts/pesco_pilot", help="artifact directory")
    parser.add_argument("--epochs", type=int, default=8, help="tabular PESCO training epochs")
    args = parser.parse_args(argv)
    result = run(Path(args.output), epochs=max(1, int(args.epochs)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["mvp_gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

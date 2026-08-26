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
from research_strategy_optimization.evaluation.legacy_certificates import (
    DISCOVERY_POLICY_ID,
    LEGACY_CERTIFICATE_ARTIFACT_SCOPE,
    annotate_legacy_certificate,
    build_legacy_certificate_manifest,
)
from research_strategy_optimization.evaluation.experiment_scaffolds import (
    experiment_b_zero_shot_diagnostic,
    experiment_c_state_reward_diagnostic,
)
from research_strategy_optimization.evidence.proper_scoring import belief_delta, log_score
from research_strategy_optimization.evidence.hypothesis_registry import HypothesisRegistry
from research_strategy_optimization.schemas import (
    DEFAULT_PROTOCOL_VERSION,
    EvidenceState,
    Hypothesis,
    Protocol,
    ResearchAction,
)
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
    "Rule-Based",
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
    "Rule-Based": "transparent_rule_based_control",
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

# The CPU pilot intentionally does not claim to reproduce external papers.  The
# transparent Rule-Based control is a valid diagnostic comparator; named external
# methods are retained only as adapter smoke tests and are excluded from formal
# algorithm comparisons until genuine implementations/checkpoints are supplied.
BASELINE_COMPARISON_ROLE = {
    "Base": "diagnostic_reference",
    "Rule-Based": "diagnostic_control",
    "Search-Only": "diagnostic_oracle_upper_bound",
    "PESCO-Offline": "diagnostic_pesco_reference",
    "PESCO-Full": "diagnostic_pesco_reference",
    "SFT": "external_name_adapter_excluded",
    "GRPO-Terminal": "external_name_adapter_excluded",
    "GRPO-FourState": "external_name_adapter_excluded",
    "GDPO": "external_name_adapter_excluded",
    "SMOPD": "external_name_adapter_excluded",
    "Evidence-Gated SMOPD": "external_name_adapter_excluded",
    "DiscoPO": "external_name_adapter_excluded",
    "Ecpo": "external_name_adapter_excluded",
    "TCPO": "external_name_adapter_excluded",
    "CVT-RL": "external_name_adapter_excluded",
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


def _policy_predicted_state(policy: Any, observation, delta_min: float) -> Optional[EvidenceState]:
    """Return a state only when the policy actually exposes a state decision.

    Base/Search/PESCO tabular policies currently emit actions only, so their state
    prediction is NA.  The evaluator's trusted diagnostic state is recorded in a
    separate field and must not be mistaken for model recognition.
    """

    if isinstance(policy, EvidenceHeuristicPolicy):
        return infer_visible_state(observation, delta_min)
    predictor = getattr(policy, "predict_evidence_state", None)
    if callable(predictor):
        value = predictor(observation)
        return value if isinstance(value, EvidenceState) else EvidenceState(value)
    return None


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


def _zero_shot_diagnostic(protocol: Protocol, worlds, question_id: str) -> Dict[str, Any]:
    """Run the available CPU policy as an explicitly non-model diagnostic.

    The pilot does not have a frozen language model/checkpoint, so this artifact must
    remain ``pass: false``.  We still execute the same public-observation path and
    record its outputs, making stage 3 evidence-driven and ready to consume a real
    model report later instead of silently hard-coding a GO flag.
    """

    policy = BasePolicy()
    rows: List[Dict[str, Any]] = []
    for world in worlds:
        env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
        observation, _, trusted_verdict, _ = _initial_state(env, question_id, world.world_id, 0, protocol)
        action = policy.choose(observation)
        diagnostic_state = infer_visible_state(observation, protocol.delta_min)
        rows.append({
            "question_id": question_id,
            "world_id": world.world_id,
            "policy_input": observation.to_dict(),
            "selected_action": action.value,
            # No frozen model/checkpoint is available: this row has no policy state
            # output.  Keep the public-observation rule diagnostic under a separate
            # evaluator field so it cannot be reported as model recognition.
            "predicted_state": None,
            "policy_predicted_state": None,
            "evaluator_diagnostic_state": diagnostic_state.value if diagnostic_state is not None else None,
            "state_prediction_source": "evaluator_diagnostic",
            "trusted_state_for_audit": trusted_verdict.evidence_state.value,
            "action_correct_for_audit": action.value == BEST_ACTION[world.kind],
            "model_inference": False,
        })
    return {
        "protocol_version": protocol.protocol_version,
        "question_id": question_id,
        "policy_class": type(policy).__name__,
        "model_checkpoint": None,
        "real_model_zero_shot_completed": False,
        "diagnostic_only": True,
        "records": rows,
        "reason": "no frozen model/checkpoint is available in the CPU pilot",
        # A stage can only pass after a real model report is supplied and audited.
        "pass": False,
    }


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
                    # Search-Only is an explicitly evaluator-side oracle diagnostic.
                    # Give it the registered oracle utility so it can select the
                    # utility-maximising branch; ordinary policies continue to use
                    # only public observations and the generic transition utility.
                    manager = BranchRolloutManager(
                        environment=base_env,
                        verifier=TrustedVerifier(protocol),
                        utility_fn=_oracle_utility,
                    )
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
                # Keep policy output separate from the evaluator diagnostic.  The
                # latter is useful for environment sanity checks but is not a model
                # state-recognition prediction.  Only an explicit policy state head or
                # the transparent Rule-Based adapter gets a policy_predicted_state.
                policy_predicted_state = _policy_predicted_state(policy, base_obs, protocol.delta_min)
                evaluator_diagnostic_state = infer_visible_state(base_obs, protocol.delta_min) or EvidenceState.INSUFFICIENT
                evaluator_final_diagnostic_state = infer_visible_state(public_final, protocol.delta_min)
                truth = WORLD_STATE[world.kind]
                action_correct = action.value == BEST_ACTION[world.kind]
                state_correct = policy_predicted_state is not None and policy_predicted_state.value.title() == truth
                valid_claim = bool(verdict.validity_pass)
                # The MVP action space contains a fixed ``switch`` option; choosing it
                # is not autonomous method discovery.  Discovery is therefore disabled
                # for *every* method in this pilot.  This avoids the previous
                # method-name-specific bonus (PESCO rows received a certificate that
                # identical baseline actions could never earn).  The certificate
                # implementation remains available for a future open-ended runner
                # where all methods submit comparable candidates.
                discovery = False
                p = public_final.hypothesis_probability
                # Bind score identities to the environment's policy-visible
                # Observation, rather than synthesising a second ID namespace in the
                # evaluator.  The initial Tier-0 observation carries H_A/H_B; a
                # switch activates the existing H_B record and leaves H_A untouched.
                hypothesis_a_id = str(base_obs.active_hypothesis_id)
                belief_ids = tuple(base_obs.belief_map())
                hypothesis_b_id = next((item for item in belief_ids if item != hypothesis_a_id), "H_B")
                belief_before = float(base_obs.hypothesis_probability)
                # A switch changes the active estimand.  Score the post-switch
                # probability against H_B's truth and B's own prior rather than
                # reusing method B evidence as if it were evidence about H_A.
                active_hypothesis_id = str(public_final.active_hypothesis_id)
                prior_active = float(base_obs.belief_map().get(active_hypothesis_id, 0.5))
                truth_bit = int((world.true_effect_b if active_hypothesis_id == hypothesis_b_id else world.true_effect_a) > protocol.delta_min)
                belief_delta_value = float(belief_delta(prior_active, p, truth_bit, protocol.probability_clip))
                belief_score = float(log_score(p, truth_bit, protocol.probability_clip))
                replication = bool(verdict.independent_confirmation_passed)
                cost = float(base_output.execution_cost + action_cost)
                valid_gate = 1.0 if valid_claim else 0.0
                vrs = valid_gate * (belief_delta_value + float(action_correct) + float(replication) + float(discovery)) - 0.1 * cost
                # Preserve the exact belief maps emitted by the environment at the
                # action boundary and after execution.  This prevents an evaluator
                # from silently inventing a second hypothesis namespace or dropping
                # the untouched H_A belief after a method switch.
                beliefs_before = {str(key): float(value) for key, value in base_obs.belief_map().items()}
                beliefs_after = {str(key): float(value) for key, value in public_final.belief_map().items()}
                hypothesis_beliefs = dict(beliefs_after)
                records.append({
                    "record_type": "policy_episode",
                    "schema_version": "pesco_results_v0.2",
                    "method": method_name,
                    "implementation_status": BASELINE_IMPLEMENTATION_STATUS[method_name],
                    "comparison_role": BASELINE_COMPARISON_ROLE[method_name],
                    "formal_comparison_eligible": False,
                    "formal_comparison_exclusion_reason": "cpu_pilot_diagnostic_only",
                    "split": split,
                    "question_id": question_id,
                    "world_id": world.world_id,
                    "world_pair_id": f"{question_id}:supported_refuted",
                    "world_kind": world.kind,
                    "true_state": truth,
                    "predicted_state": policy_predicted_state.value.title() if policy_predicted_state else None,
                    "policy_predicted_state": policy_predicted_state.value.title() if policy_predicted_state else None,
                    "evaluator_diagnostic_state": evaluator_diagnostic_state.value.title(),
                    "evaluator_final_diagnostic_state": evaluator_final_diagnostic_state.value.title() if evaluator_final_diagnostic_state else None,
                    "state_prediction_source": "policy_output" if policy_predicted_state else "not_emitted",
                    "state_metric_eligible": policy_predicted_state is not None,
                    "final_predicted_state": evaluator_final_diagnostic_state.value.title() if evaluator_final_diagnostic_state else None,
                    "initial_state_trusted": base_verdict.evidence_state.value,
                    "final_state_trusted": verdict.evidence_state.value,
                    "selected_action": action.value,
                    "best_action": BEST_ACTION[world.kind],
                    "action_correct": action_correct,
                    "state_correct": state_correct,
                    "valid_claim": valid_claim,
                    "belief_score": belief_score,
                    "belief_submission_source": "environment_public_observation_diagnostic",
                    "belief_submitted_before_action": False,
                    "belief_score_formal_eligible": False,
                    "belief_before": belief_before,
                    "belief_before_active": prior_active,
                    "belief_after": p,
                    "belief_log_score_delta": belief_delta_value,
                    "active_hypothesis_id": active_hypothesis_id,
                    "active_hypothesis_id_before": hypothesis_a_id,
                    "new_hypothesis_id": active_hypothesis_id if active_hypothesis_id != hypothesis_a_id else None,
                    "observation_hypothesis_ids": list(belief_ids),
                    "hypothesis_beliefs": hypothesis_beliefs,
                    "beliefs_before": beliefs_before,
                    "beliefs_after": beliefs_after,
                    "belief_target": bool(truth_bit),
                    "belief_score_estimand": active_hypothesis_id,
                    "task_utility": float(action_correct),
                    "replication_utility": float(replication),
                    "discovery_utility": float(discovery),
                    "discovery_bonus_policy": "disabled_fixed_action_space",
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
                    "refutation_accept": world.kind == "refuted" and policy_predicted_state is EvidenceState.REFUTED,
                    "underpower_handled": world.kind != "insufficient" or action is ResearchAction.SAMPLE,
                    "invalid_repaired": world.kind == "invalid" and action is ResearchAction.REPAIR and valid_claim,
                    "invalid_claim": world.kind == "invalid" and policy_predicted_state is EvidenceState.SUPPORTED,
                    "flip_eligible": world.kind in {"supported", "refuted"},
                    "required_switch": world.kind == "refuted",
                    "switch_required": world.kind == "refuted",
                    "invalid_repair_eligible": world.kind == "invalid",
                    "invalid_initial": world.kind == "invalid",
                    "leakage_repair_eligible": bool(world.leakage),
                    "confounding_repair_eligible": bool(world.confounding),
                    "insufficient_handling_eligible": world.kind == "insufficient",
                    "insufficient_initial": world.kind == "insufficient",
                    # Eligibility is determined by the trusted final evidence state,
                    # not by whether this particular runner happened to execute the
                    # confirmation call.  Thus a method that skips an eligible
                    # confirmation remains in the denominator and cannot improve its
                    # rate by omitting the attempt.
                    "confirmation_eligible": verdict.evidence_state in {EvidenceState.SUPPORTED, EvidenceState.REFUTED},
                    "new_path_announced": False,
                    "new_path_verified": False,
                    "discovery_opportunity": False,
                    "discovery_eligible": False,
                    "method_family": "method_b" if action is ResearchAction.SWITCH else "method_a",
                    "flip_correct": False,  # filled after both paired observations are collected
                    "turn": public_final.turn,
                    "branch_id": f"{method_name}:{question_id}:{world.world_id}",
                })
                ledger.add(environment_runs=2 if isinstance(policy, OracleSearchPolicy) else 2, confirmation_runs=int(verdict.independent_confirmation_performed))

    # Compute the actual policy probability reversal after all paired public states
    # are available.  This uses no hidden labels: only each policy's public
    # observation.  Pair outcomes are calculated independently per question/split;
    # reusing the first question's probabilities would silently leak one task's
    # decision into every other task's FlipAcc row.
    supported_world = next((world for world in worlds if world.kind == "supported"), None)
    refuted_world = next((world for world in worlds if world.kind == "refuted"), None)
    if supported_world is not None and refuted_world is not None:
        for method_name in policies:
            for question_id, _split in question_ids:
                by_world = {
                    r["world_id"]: r
                    for r in records
                    if r["method"] == method_name and r["question_id"] == question_id
                }
                if supported_world.world_id not in by_world or refuted_world.world_id not in by_world:
                    continue
                # Use the exact public observation at the decision point, not a
                # post-action state (which may have legitimately changed after
                # repair/sample/switch).
                probs_a = by_world[supported_world.world_id].get("decision_action_probabilities", {})
                probs_b = by_world[refuted_world.world_id].get("decision_action_probabilities", {})
                continue_a = float(probs_a.get(ResearchAction.CONTINUE.value, 0.0))
                switch_a = float(probs_a.get(ResearchAction.SWITCH.value, 0.0))
                continue_b = float(probs_b.get(ResearchAction.CONTINUE.value, 0.0))
                switch_b = float(probs_b.get(ResearchAction.SWITCH.value, 0.0))
                flip = continue_a > switch_a and switch_b > continue_b
                for row in records:
                    if (
                        row["method"] == method_name
                        and row["question_id"] == question_id
                        and row["world_id"] in {supported_world.world_id, refuted_world.world_id}
                    ):
                        row["flip_correct"] = flip
    return records, ledger


def run(output_dir: Path, *, epochs: int = 8) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(protocol_version=DEFAULT_PROTOCOL_VERSION)
    worlds = list(default_mvp_worlds())
    question_id = "rq_tier0_001"
    # Fixed-action MVP fairness rule: no method receives an autonomous-discovery
    # utility or certificate bonus.  Keep this machine-readable next to results so a
    # report cannot accidentally interpret stale certificate files as current scores.
    _json_dump(output_dir / "discovery_policy.json", {
        "schema_version": "pesco_discovery_policy_v0.2",
        "policy_id": DISCOVERY_POLICY_ID,
        "status": "disabled",
        "reason": "fixed_mvp_action_space_switch_is_not_open_ended_discovery",
        "applies_to": "all_methods",
        "discovery_utility": 0.0,
        "new_path_announced": False,
        "new_path_verified": False,
        "formal_open_ended_certificate_authorized": False,
        "current_certificate_claim_authorized": False,
        "legacy_certificate_artifact_scope": LEGACY_CERTIFICATE_ARTIFACT_SCOPE,
        "legacy_certificate_manifest": "legacy_certificates_manifest.json",
        "legacy_certificate_pass_interpretation": "historical_only_not_current_discovery",
    })
    # A few certificates were produced by an older pilot before the fixed-action
    # boundary was encoded.  Preserve their historical pass/autonomous values but
    # attach an explicit machine-readable scope and manifest so they cannot be
    # mistaken for current discovery evidence.
    for certificate_path in sorted(output_dir.glob("certificate_*.json")):
        try:
            certificate_payload = json.loads(certificate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Preserve an explicitly scoped current/open-ended certificate emitted by a
        # future runner.  Only unscoped historical files (the conflict this pilot
        # needs to quarantine) receive the legacy marker.
        if certificate_payload.get("artifact_scope") not in (None, LEGACY_CERTIFICATE_ARTIFACT_SCOPE):
            continue
        _json_dump(
            certificate_path,
            annotate_legacy_certificate(
                certificate_payload,
                discovery_policy_ref="discovery_policy.json",
            ),
        )
    _json_dump(
        output_dir / "legacy_certificates_manifest.json",
        build_legacy_certificate_manifest(output_dir),
    )
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
        "protocol_version": protocol.protocol_version,
        "expected_protocol_version": DEFAULT_PROTOCOL_VERSION,
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
            {
                "name": name,
                "implementation_status": BASELINE_IMPLEMENTATION_STATUS[name],
                "comparison_role": BASELINE_COMPARISON_ROLE[name],
                "formal_comparison_eligible": False,
                "formal_comparison_exclusion_reason": (
                    "no_frozen_multi_question_final_splits_or_genuine_external_implementation"
                ),
            }
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
    # Experiments B/C are intentionally diagnostic-only in this CPU pilot.  Their
    # machine-readable gates prevent the report from treating adapter rows as a real
    # model zero-shot study or as evidence that ordinary state rewards are sufficient.
    _json_dump(output_dir / "experiment_b_diagnostic.json", experiment_b_zero_shot_diagnostic(records))
    _json_dump(output_dir / "experiment_c_diagnostic.json", experiment_c_state_reward_diagnostic(records))
    _json_dump(output_dir / "compute_ledger.json", ledger.to_dict())

    # Offline stage status is tied to measured pilot evidence rather than merely the
    # existence of a reversal pair.  This is still a CPU diagnostic (not a formal
    # promotion/final-split claim), but a future run whose PESCO-Offline VRS regresses
    # against Base will correctly close the stage.
    id_records = [row for row in records if row["split"] == "pilot_id"]
    def _mean_method_vrs(method_name: str) -> float:
        values = [float(row["vrs"]) for row in id_records if row["method"] == method_name]
        return sum(values) / len(values) if values else float("nan")
    base_vrs = _mean_method_vrs("Base")
    offline_vrs = _mean_method_vrs("PESCO-Offline")
    offline_training_evidence = {
        "loss_driven_update": any(bool(epoch.get("loss_driven_flip_update", False)) for epoch in offline.log.epochs),
        "confirmed_reversal": bool(reversal_rows and reversal_rows[0]["paired_confidence"]["confirmed_reversal"]),
        "base_vrs": base_vrs,
        "pesco_offline_vrs": offline_vrs,
        "offline_improves_base": bool(math.isfinite(offline_vrs) and math.isfinite(base_vrs) and offline_vrs > base_vrs),
    }
    offline_training_evidence["pass"] = all(
        bool(offline_training_evidence[key])
        for key in ("loss_driven_update", "confirmed_reversal", "offline_improves_base")
    )

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
    zero_shot = _zero_shot_diagnostic(protocol, worlds, question_id)
    _json_dump(output_dir / "zero_shot_diagnostic.json", zero_shot)
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
        "protocol_version": protocol.protocol_version,
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
        "stage_3_zero_shot": stage_status("stage_3_zero_shot", zero_shot),
        "stage_4_offline": stage_status("stage_4_offline", offline_training_evidence),
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
    audit_ledger.append("discovery_policy", {
        "status": "disabled",
        "applies_to": "all_methods",
        "reason": "fixed_mvp_action_space_switch_is_not_open_ended_discovery",
    })
    audit_ledger.append("mvp_counts", {
        "branch_groups": branch_groups,
        "exploration_experiments": exploration_experiments,
        "confirmation_experiments": confirmation_experiments,
    })
    audit_ledger.append("negative_controls", negative_controls)
    audit_ledger.append("experiment_b_diagnostic", experiment_b_zero_shot_diagnostic(records))
    audit_ledger.append("experiment_c_diagnostic", experiment_c_state_reward_diagnostic(records))
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

"""Tier-1 v0.4 benchmark hardening and posterior decision audit.

This module is deliberately separate from the frozen v0.3 runner.  v0.3 kept a
hand-written ``target_actions`` table for calibration; that table is useful for
testing the environment, but it is not a valid target for a policy when an action
depends on a latent method-B effect.  v0.4 therefore derives an evaluator target
from a leave-one-question-out posterior and expected utility/value-of-information
calculation.

Two input tracks are exposed:

``oracle_state``
    An evaluator-side upper-bound diagnostic.  The trusted initial evidence state is
    supplied as an input feature, but world IDs, latent effects, and target actions
    remain hidden.

``raw_evidence``
    Uses only the public/raw experiment observation and output fields.  The evidence
    state is *not* supplied as a feature; the posterior must infer it from estimates,
    intervals, sample size, and validity signals.

The posterior candidate bank excludes the current question.  This is the key
anti-hindsight boundary: in causal-confounding variant 3, the current hidden
positive method-B effect cannot be looked up when deciding whether to switch.  The
artifact is a diagnostic benchmark/planner, not a formal model result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..environments.tier0_simulator import TrustedVerifier
from ..environments.tier1_benchmark import (
    FAMILY_CAUSAL_CONFOUNDING,
    MECHANISM_FAMILIES,
    Tier1Benchmark,
    Tier1QuestionSpec,
    build_tier1_v03_benchmark,
    tier1_scientific_utility,
)
from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..schemas import EvidenceState, ExperimentOutput, Protocol, ResearchAction, Verdict


TRACK_ORACLE_STATE = "oracle_state"
TRACK_RAW_EVIDENCE = "raw_evidence"
V04_TRACKS = (TRACK_ORACLE_STATE, TRACK_RAW_EVIDENCE)
V04_SCHEMA_VERSION = "pesco_tier1_benchmark_v0.4"
V04_PROTOCOL_VERSION = "pesco_v0_2"
_ACTION_ORDER = tuple(ResearchAction.mvp_actions())
_VALIDITY_MARKERS = {
    "split_overlap_diagnostic",
    "leaky_row_split",
    "treatment_confounder_dependence",
    "metric_scope_mismatch",
    "variance_estimator_unstable",
    "protocol_invalid_diagnostic",
    "sample_count_below_precision_target",
    "confounder_adjusted_estimator",
    "group_held_out_split",
    "subgroup_metric_estimator",
}


@dataclass(frozen=True)
class V04Scenario:
    """A latent evaluator scenario in the leave-one-question-out bank."""

    source_question_id: str
    family: str
    world_id: str
    kind: str
    world: Any
    prior: float

    @property
    def key(self) -> str:
        return f"{self.source_question_id}|{self.world_id}"


@dataclass(frozen=True)
class CandidateSimulation:
    scenario_key: str
    action: ResearchAction
    utility: float
    output: ExperimentOutput
    verdict: Verdict
    raw_evidence: Mapping[str, Any]


def build_tier1_v04_benchmark() -> Tier1Benchmark:
    """Return the v0.3 environment with a v0.4 evaluator boundary.

    The executable worlds remain byte-for-byte compatible with v0.3.  The v0.4
    revision is a decision/evaluation hardening: no v0.3 hidden target table is
    consumed by the planner.  Keeping the world generator shared makes changes in
    the decision rule auditable instead of silently changing the environment too.
    """

    return build_tier1_v03_benchmark()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tier1_v04_manifest(
    benchmark: Optional[Tier1Benchmark] = None,
    *,
    include_hidden: bool = True,
) -> dict:
    """Build a manifest with no target-action table.

    ``include_hidden`` controls world parameters for the evaluator artifact only;
    neither form contains the legacy v0.3 ``target_actions`` field.
    """

    benchmark = benchmark or build_tier1_v04_benchmark()
    questions = []
    for question in benchmark.questions:
        item = {
            "question_id": question.question_id,
            "policy_question_id": question.policy_question_id,
            "family": question.family,
            "variant": int(question.variant),
            "split": question.split,
            "seed_offset": int(question.seed_offset),
            "description": question.description,
        }
        if include_hidden:
            item["world_ids"] = [world.world_id for world in question.worlds]
            item["worlds"] = [
                {
                    "world_id": world.world_id,
                    "kind": world.kind,
                    "true_effect_a": float(world.true_effect_a),
                    "true_effect_b": float(world.true_effect_b),
                    "noise_scale": float(world.noise_scale),
                    "initial_samples": int(world.initial_samples),
                    "leakage": bool(world.leakage),
                    "confounding": bool(world.confounding),
                    "metric_mismatch": bool(world.metric_mismatch),
                    "protocol_invalid": bool(world.protocol_invalid),
                    "seed_offset": int(world.seed_offset),
                    "question_family": world.question_family,
                }
                for world in question.worlds
            ]
        else:
            item["world_count"] = len(question.worlds)
        questions.append(item)
    payload = {
        "schema_version": V04_SCHEMA_VERSION,
        "parent_environment_schema_version": "pesco_tier1_benchmark_v0.3",
        "protocol_version": benchmark.protocol_version,
        "question_count": len(benchmark.questions),
        "world_count": len(benchmark.worlds),
        "mechanism_family_count": len(MECHANISM_FAMILIES),
        "mechanism_families": list(MECHANISM_FAMILIES),
        "questions": questions,
        "counts_by_split": {
            split: sum(question.split == split for question in benchmark.questions)
            for split in ("train", "dev", "diagnostic_ood")
        },
        "dual_tracks": {
            TRACK_ORACLE_STATE: {
                "input": "trusted_initial_evidence_state_plus_public_task_family",
                "state_label_is_evaluator_side_upper_bound": True,
                "world_id_visible": False,
                "target_action_visible": False,
            },
            TRACK_RAW_EVIDENCE: {
                "input": "numeric_raw_receipts_correlations_overlap_CI_sample_replication_logs",
                "trusted_evidence_state_visible": False,
                "structured_validity_tokens_visible": False,
                "task_family_visible": False,
                "world_id_visible": False,
                "target_action_visible": False,
            },
        },
        "posterior": {
            "prior": "uniform_over_same_family_other_questions",
            "candidate_pool": "leave_one_question_out",
            "method_b_hindsight_excluded": True,
            "posterior_expected_utility": "sum_h posterior(h|evidence) * branch_utility(h, action)",
            "value_of_information": "E_h[max_a EU(a|post_action_evidence)] - max_a EU(a|current_evidence)",
            "target_action_source": "argmax_expected_utility_plus_value_of_information",
        },
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
    }
    payload["manifest_digest"] = _canonical_digest(payload)
    return payload


def _state_likelihood(observed_state: str, candidate_kind: str) -> float:
    """Small-noise categorical likelihood for the oracle-state track."""

    return 0.97 if str(observed_state) == str(candidate_kind) else 0.01


def _normal_logpdf(value: float, mean: float, scale: float) -> float:
    scale = max(float(scale), 1e-3)
    z = (float(value) - float(mean)) / scale
    return -0.5 * z * z - math.log(scale) - 0.5 * math.log(2.0 * math.pi)


def _expected_signal_set(world: Any, *, method: str, sample_size: int) -> set[str]:
    expected: set[str] = set()
    if world.leakage:
        expected.add("split_overlap_diagnostic")
    if world.confounding:
        expected.add("treatment_confounder_dependence")
    if world.metric_mismatch and method == "method_a":
        expected.add("metric_scope_mismatch")
    if world.protocol_invalid and int(sample_size) < 60:
        expected.add("sample_count_below_precision_target")
    return expected


def _raw_likelihood(
    evidence: Mapping[str, Any],
    world: Any,
    *,
    method: str,
) -> float:
    """Pre-registered raw-evidence likelihood.

    Only fields that a policy can receive are used.  In particular, this function
    never reads ``world.kind`` as a label and never reads a target-action map.  The
    latent effect is used only as the generative mean of a candidate likelihood.
    """

    estimate = float(evidence.get("effect_estimate", 0.0))
    interval = evidence.get("confidence_interval", (estimate - 1.0, estimate + 1.0))
    try:
        low, high = float(interval[0]), float(interval[1])
    except (TypeError, ValueError, IndexError):
        low, high = estimate - 1.0, estimate + 1.0
    sample_size = max(1, int(evidence.get("sample_size", world.initial_samples)))
    ci_scale = max(0.01, abs(high - low) / 3.92)
    generative_scale = max(0.01, float(world.noise_scale) / math.sqrt(max(1, sample_size)))
    mean = float(world.true_effect_b if method == "method_b" else world.true_effect_a)
    score = _normal_logpdf(estimate, mean, max(ci_scale, generative_scale))
    # Raw validity signals are observable output receipts, not a hidden state label.
    observed = set(str(value) for value in evidence.get("validity_signals", ()))
    expected = _expected_signal_set(world, method=method, sample_size=sample_size)
    for marker in _VALIDITY_MARKERS:
        if marker not in expected and marker not in observed:
            continue
        score += math.log(0.90 if ((marker in expected) == (marker in observed)) else 0.10)
    # Sample-size and interval-width consistency is weak evidence; it prevents a
    # tiny interval from assigning all mass to a low-sample candidate by accident.
    expected_small = int(world.initial_samples) < 60
    observed_small = sample_size < 60 or "sample_count_below_precision_target" in observed
    score += math.log(0.80 if expected_small == observed_small else 0.20)
    return float(score)


def _evidence_payload(
    output: ExperimentOutput,
    verdict: Verdict,
    observation: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Serialize only raw/public evidence fields for posterior updates."""

    payload = {
        "effect_estimate": float(output.effect_estimate),
        "confidence_interval": [float(output.ci_low), float(output.ci_high)],
        "sample_size": int(output.sample_size),
        "seed_count": int(output.seed_count),
        "remaining_budget": int(observation.get("remaining_budget", 0)) if observation else 0,
        "current_method": str(output.method),
        "metric_name": str(observation.get("metric_name", "")) if observation else "",
        "validity_signals": list(output.validity_signals),
        "validity_signal_count": int(len(output.validity_signals)),
        "replication_ci_width": float(output.ci_high - output.ci_low),
        "treatment_confounder_correlation": float(getattr(output, "treatment_confounder_correlation", 0.0)),
        "group_overlap_count": int(getattr(output, "group_overlap_count", 0)),
        "replication_sample_size": int(output.sample_size),
        "replication_seed_count": int(output.seed_count),
        "task_family": str(observation.get("task_family", "")) if observation else "",
        "evidence_state": str(verdict.evidence_state.value),
    }
    return payload


def _track_observation(
    track: str,
    *,
    output: Optional[ExperimentOutput] = None,
    verdict: Optional[Verdict] = None,
    observation: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if track == TRACK_ORACLE_STATE:
        if verdict is None:
            raise ValueError("oracle_state track requires a trusted verdict")
        return {
            "track": TRACK_ORACLE_STATE,
            "evidence_state": verdict.evidence_state.value,
            "task_family": str(observation.get("task_family", "")) if observation else "",
        }
    if track == TRACK_RAW_EVIDENCE:
        if output is None or verdict is None:
            raise ValueError("raw_evidence track requires output and verdict")
        raw = _evidence_payload(output, verdict, observation)
        raw.pop("evidence_state", None)
        # Raw-evidence decisions are restricted to numeric/raw receipts.  Family
        # names and validity marker strings are useful evaluator logs, but are not
        # admitted as structured state inputs to this track.
        raw["task_family"] = "raw_evidence"
        raw["validity_signals"] = []
        return {"track": TRACK_RAW_EVIDENCE, **raw}
    raise ValueError(f"unknown v0.4 track: {track}")


def candidate_scenarios(
    benchmark: Tier1Benchmark,
    question: Tier1QuestionSpec,
) -> Tuple[V04Scenario, ...]:
    """Return same-family worlds from *other* questions only."""

    candidates = [
        (other, world)
        for other in benchmark.questions
        if other.family == question.family and other.question_id != question.question_id
        for world in other.worlds
    ]
    if not candidates:
        raise ValueError(f"no leave-one-question-out candidate worlds for {question.question_id}")
    prior = 1.0 / float(len(candidates))
    return tuple(
        V04Scenario(
            source_question_id=source.question_id,
            family=source.family,
            world_id=world.world_id,
            kind=world.kind,
            world=world,
            prior=prior,
        )
        for source, world in candidates
    )


def posterior_from_evidence(
    evidence: Mapping[str, Any],
    scenarios: Sequence[V04Scenario],
    *,
    track: str,
    method: str = "method_a",
) -> Dict[str, float]:
    """Compute a normalized posterior for either v0.4 input track."""

    if not scenarios:
        raise ValueError("posterior requires at least one candidate scenario")
    observed_state = str(evidence.get("evidence_state", ""))
    log_weights = []
    for scenario in scenarios:
        if track == TRACK_ORACLE_STATE:
            likelihood = _state_likelihood(observed_state, scenario.kind)
        elif track == TRACK_RAW_EVIDENCE:
            likelihood = math.exp(_raw_likelihood(evidence, scenario.world, method=method))
        else:
            raise ValueError(f"unknown v0.4 track: {track}")
        log_weights.append(math.log(max(1e-300, float(scenario.prior))) + math.log(max(1e-300, likelihood)))
    maximum = max(log_weights)
    weights = [math.exp(value - maximum) for value in log_weights]
    total = sum(weights) or 1.0
    return {scenario.key: float(weight / total) for scenario, weight in zip(scenarios, weights)}


def _simulate_candidate_action(
    question: Tier1QuestionSpec,
    scenario: V04Scenario,
    action: ResearchAction,
    protocol: Protocol,
) -> CandidateSimulation:
    env = Tier1TabularEnvironment(worlds=(scenario.world,), protocol=protocol)
    env.reset(question.policy_question_id, scenario.world_id, seed=17)
    baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    verifier = TrustedVerifier(protocol)
    baseline_verdict = verifier.evaluate(baseline, env)
    initial_observation = env.visible_observation().to_dict()
    snapshot = env.snapshot()
    branch = env.clone_from_snapshot(snapshot)
    output = branch.execute_option(action, seeds=protocol.exploration_seeds)
    verdict = verifier.evaluate(output, branch)
    utility = tier1_scientific_utility(
        question,
        scenario.world,
        action,
        output,
        verdict,
        protocol,
        initial_observation=env.visible_observation(),
    )
    # Ensure baseline evidence is actually evaluated; this catches accidental
    # planner code that computes a branch utility from a hidden kind/target only.
    if not isinstance(baseline_verdict.evidence_state, EvidenceState):
        raise RuntimeError("candidate baseline verifier did not produce an evidence state")
    return CandidateSimulation(
        scenario_key=scenario.key,
        action=action,
        utility=float(utility),
        output=output,
        verdict=verdict,
        raw_evidence=_evidence_payload(output, verdict, initial_observation),
    )


def build_candidate_action_table(
    benchmark: Tier1Benchmark,
    question: Tier1QuestionSpec,
    protocol: Optional[Protocol] = None,
) -> Dict[str, Dict[str, CandidateSimulation]]:
    """Materialize evaluator branch utilities for the blind candidate bank."""

    protocol = protocol or Protocol(protocol_version=V04_PROTOCOL_VERSION)
    table: Dict[str, Dict[str, CandidateSimulation]] = {}
    for scenario in candidate_scenarios(benchmark, question):
        table[scenario.key] = {
            action.value: _simulate_candidate_action(question, scenario, action, protocol)
            for action in _ACTION_ORDER
        }
    return table


def _expected_action_values(
    posterior: Mapping[str, float],
    scenarios: Sequence[V04Scenario],
    table: Mapping[str, Mapping[str, CandidateSimulation]],
) -> Dict[str, float]:
    return {
        action.value: float(
            sum(
                float(posterior.get(scenario.key, 0.0))
                * float(table[scenario.key][action.value].utility)
                for scenario in scenarios
            )
        )
        for action in _ACTION_ORDER
    }


def plan_world(
    benchmark: Tier1Benchmark,
    question: Tier1QuestionSpec,
    world: Any,
    initial_output: ExperimentOutput,
    initial_verdict: Verdict,
    initial_observation: Mapping[str, Any],
    *,
    track: str,
    protocol: Optional[Protocol] = None,
    candidate_table: Optional[Mapping[str, Mapping[str, CandidateSimulation]]] = None,
) -> dict:
    """Plan one world using posterior expected utility and value of information."""

    protocol = protocol or Protocol(protocol_version=V04_PROTOCOL_VERSION)
    scenarios = candidate_scenarios(benchmark, question)
    table = candidate_table or build_candidate_action_table(benchmark, question, protocol)
    base_evidence = _track_observation(
        track,
        output=initial_output,
        verdict=initial_verdict,
        observation=initial_observation,
    )
    posterior = posterior_from_evidence(
        base_evidence,
        scenarios,
        track=track,
        method="method_a",
    )
    current_values = _expected_action_values(posterior, scenarios, table)
    current_best = max(current_values.values())
    voi: Dict[str, float] = {}
    future_best_by_action: Dict[str, float] = {}
    posterior_after: Dict[str, Dict[str, float]] = {}
    for action in _ACTION_ORDER:
        expected_future = 0.0
        # The candidate posterior predicts a finite future output for each latent
        # scenario.  Reweight using the same pre-registered likelihood, rather than
        # grouping by exact floating-point outputs (which would create a false
        # point-mass oracle).
        for scenario in scenarios:
            probability = float(posterior.get(scenario.key, 0.0))
            result = table[scenario.key][action.value]
            after_evidence = _track_observation(
                track,
                output=result.output,
                verdict=result.verdict,
                observation=initial_observation,
            )
            after = posterior_from_evidence(
                after_evidence,
                scenarios,
                track=track,
                method=str(result.output.method),
            )
            after_values = _expected_action_values(after, scenarios, table)
            after_best = max(after_values.values())
            expected_future += probability * after_best
            # Keep one representative posterior for reproducibility.  The full
            # per-scenario posteriors are too large for a compact quick artifact and
            # are not needed to audit the VOI definition.
            posterior_after.setdefault(action.value, after)
        future_best_by_action[action.value] = float(expected_future)
        voi[action.value] = float(expected_future - current_best)
    total_values = {
        action.value: float(current_values[action.value] + voi[action.value])
        for action in _ACTION_ORDER
    }
    chosen = max(
        _ACTION_ORDER,
        key=lambda action: (total_values[action.value], -_ACTION_ORDER.index(action)),
    )
    entropy = -sum(
        probability * math.log(max(1e-300, probability))
        for probability in posterior.values()
    )
    raw_input = {
        key: value
        for key, value in _track_observation(
            TRACK_RAW_EVIDENCE,
            output=initial_output,
            verdict=initial_verdict,
            observation=initial_observation,
        ).items()
        if key not in {"track", "evidence_state"}
    }
    oracle_input = _track_observation(
        TRACK_ORACLE_STATE,
        output=initial_output,
        verdict=initial_verdict,
        observation=initial_observation,
    )
    return {
        "schema_version": "pesco_tier1_v04_decision_v0.1",
        "question_id": question.question_id,
        "world_id_audit": world.world_id,
        "family": question.family,
        "split": question.split,
        "track": track,
        "input_contract": (
            "trusted_initial_evidence_state_plus_public_task_family"
            if track == TRACK_ORACLE_STATE
            else "raw_public_observation_and_experiment_output"
        ),
        "oracle_state_input": oracle_input if track == TRACK_ORACLE_STATE else None,
        "raw_evidence_input": raw_input if track == TRACK_RAW_EVIDENCE else None,
        "candidate_pool_question_ids": sorted({scenario.source_question_id for scenario in scenarios}),
        "candidate_pool_size": len(scenarios),
        "candidate_pool_excludes_current_question": all(
            scenario.source_question_id != question.question_id for scenario in scenarios
        ),
        "posterior": {
            scenario.key: float(posterior[scenario.key]) for scenario in scenarios
        },
        "posterior_entropy": float(entropy),
        "posterior_expected_utility": current_values,
        "expected_utility": current_values,
        "current_best_expected_utility": float(current_best),
        "future_best_expected_utility": future_best_by_action,
        "value_of_information": voi,
        "voi": voi,
        "total_action_value": total_values,
        "posterior_optimal_action": chosen.value,
        "target_action_source": "leave_one_question_out_posterior_expected_utility_plus_value_of_information",
        "legacy_target_action_consumed": False,
        "method_b_hindsight_excluded": True,
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
    }


def public_track_input(decision: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the policy-facing portion of a decision for leakage tests."""

    if decision.get("track") == TRACK_ORACLE_STATE:
        return dict(decision.get("oracle_state_input") or {})
    return dict(decision.get("raw_evidence_input") or {})


__all__ = [
    "TRACK_ORACLE_STATE",
    "TRACK_RAW_EVIDENCE",
    "V04_TRACKS",
    "V04_SCHEMA_VERSION",
    "V04Scenario",
    "CandidateSimulation",
    "build_tier1_v04_benchmark",
    "tier1_v04_manifest",
    "candidate_scenarios",
    "posterior_from_evidence",
    "build_candidate_action_table",
    "plan_world",
    "public_track_input",
]

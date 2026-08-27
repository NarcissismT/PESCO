"""Tier-1 v0.3 collector, matched algorithm suite, and evaluator.

The collector executes one common public decision state and four same-state action
branches for every question/world in the 12-question benchmark: 48
question-world branch groups × 4 action-level rows = 192 rows × 4 exploration
seeds = 768 seed-level observations.  It then trains the small differentiable
policies from :mod:`algorithms.differentiable_strategy` on the exact same frozen
dataset and optimizer-step budget.

The action utility callback is evaluator-owned and receives trusted output/verdict
objects only.  It uses the public task family and evidence state, while world ID,
target action, latent truth, and verifier label are absent from policy features.
Artifacts explicitly label this as a CPU mechanism experiment, not an LLM result.
"""

from __future__ import annotations

import json
import hashlib
import math
import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..algorithms.branch_rollout import BranchRolloutManager
from ..algorithms.differentiable_strategy import (
    ACTION_SET,
    STATE_SET,
    DecisionDataset,
    DecisionExample,
    DifferentiableStrategyPolicy,
    DifferentiableStrategyTrainer,
    DifferentiableTrainingLog,
    DifferentiableTrainerConfig,
    ReversalExample,
    observation_to_features,
    policy_action,
    policy_probabilities,
)
from ..baselines.policies import infer_visible_state
from ..environments.tier0_simulator import TrustedVerifier
from ..environments.tier1_benchmark import (
    Tier1Benchmark,
    build_tier1_v03_benchmark,
    tier1_scientific_utility,
)
from ..algorithms.paired_world_sampler import identify_confirmed_reversal
from .cluster_bootstrap import clustered_bootstrap
from ..schemas import EvidenceState, Protocol, ResearchAction
from ..utils.run_manifest import build_run_manifest, write_run_manifest


DEFAULT_METHODS = (
    "SFT",
    "GRPO-Terminal",
    "GRPO-FourState",
    "StateGateOnly",
    "PESCO-NoBranch",
    "PESCO-BranchOnly",
    "PESCO-NoFlipLoss",
    "PESCO-Full",
    "Evidence-Gated SMOPD",
)


# ``invalid_local_optimization`` is deliberately a narrow diagnostic rather than a
# synonym for "any action other than REPAIR".  It flags persistence/switching while
# the trusted pre-action state is Invalid *only when that action loses to the
# evaluator-owned public branch-utility winner*.  This preserves legitimate
# Invalid→SWITCH cases (the subgroup-metric family) and does not read the hidden
# target-action audit field.
INVALID_LOCAL_OPTIMIZATION_ACTIONS = frozenset({
    ResearchAction.CONTINUE,
    ResearchAction.SWITCH,
})


def is_invalid_local_optimization(
    state_target: EvidenceState,
    selected_action: ResearchAction,
    public_best_action: ResearchAction,
) -> bool:
    """Return the evaluator-side Invalid-state local-optimization flag.

    The denominator is every example whose trusted pre-action state is Invalid.
    The numerator contains only CONTINUE/SWITCH selections that are *not* the
    public branch-utility winner.  In particular, an Invalid example whose
    correct action is SWITCH is not counted.  Hidden ``target_action`` metadata
    is intentionally not consulted.
    """

    return bool(
        state_target is EvidenceState.INVALID
        and selected_action in INVALID_LOCAL_OPTIMIZATION_ACTIONS
        and selected_action is not public_best_action
    )


@dataclass(frozen=True)
class Tier1SuiteConfig:
    exploration_seeds: Tuple[int, ...] = (17, 29, 41, 53)
    confirmation_seeds: Tuple[int, ...] = (103, 107, 109, 113)
    branch_margin: float = 0.05
    methods: Tuple[str, ...] = DEFAULT_METHODS
    trainer: DifferentiableTrainerConfig = field(default_factory=DifferentiableTrainerConfig)


def benchmark_action_utility(
    output: Any,
    verdict: Any,
    branch_env: Any,
    benchmark: Tier1Benchmark,
    *,
    question: Any = None,
    world: Any = None,
    initial_observation: Any = None,
) -> float:
    """Compute evaluator-side one-step scientific utility for one branch.

    The utility is derived from the public decision state, public task-family
    protocol, trusted transition validity, confirmation, and cost.  It intentionally
    does *not* consult the hidden target-action table; target actions remain post-hoc
    audit labels.
    """

    if question is not None and world is not None:
        return tier1_scientific_utility(
            question,
            world,
            ResearchAction(getattr(output, "action", ResearchAction.CONTINUE)),
            output,
            verdict,
            getattr(branch_env, "protocol", Protocol(protocol_version="pesco_v0_2")),
            initial_observation=initial_observation or getattr(branch_env, "_branch_initial_observation", None),
        )
    initial = getattr(branch_env, "_branch_initial_observation", None)
    before = infer_visible_state(initial) if initial is not None else None
    action = ResearchAction(getattr(output, "action", ResearchAction.CONTINUE))
    signals = set(getattr(output, "validity_signals", ()))
    valid = bool(getattr(verdict, "validity_pass", False))
    cost = float(getattr(output, "execution_cost", 0.0))
    if not valid:
        return float(-0.30 - 0.03 * cost)
    score = 0.25
    after = getattr(verdict, "evidence_state", EvidenceState.INSUFFICIENT)
    if bool(getattr(verdict, "independent_confirmation_passed", False)):
        score += 0.20
    if before is EvidenceState.INVALID and "split_protocol_updated" in signals:
        score += 0.45
    if "group_held_out_split" in signals and "split_overlap_diagnostic" not in signals:
        score += 0.80
    mechanism_transition = any(marker in signal for signal in signals for marker in ("adjusted", "controlled", "subgroup_metric_estimator"))
    if mechanism_transition and not (
        action is ResearchAction.SWITCH
        and getattr(output, "method", "") == "method_b"
        and float(getattr(output, "effect_estimate", 0.0)) <= max(0.08, 0.02)
    ):
        score += 0.65
    if action is ResearchAction.SWITCH and getattr(output, "method", "") == "method_b" and float(getattr(output, "effect_estimate", 0.0)) > max(0.08, 0.02):
        score += 0.55
    elif action is ResearchAction.SWITCH and getattr(output, "method", "") == "method_b":
        score -= 0.20
    if int(getattr(output, "sample_size", 0)) >= 60 and "sample_count_below_precision_target" not in signals:
        score += 0.35
    if before is EvidenceState.INSUFFICIENT and after is not EvidenceState.INSUFFICIENT:
        score += 0.30
    if action is ResearchAction.REPLICATE:
        score += 0.10
    return float(score - 0.03 * cost)


def collect_tier1_v03_dataset(
    benchmark: Optional[Tier1Benchmark] = None,
    protocol: Optional[Protocol] = None,
    *,
    include_seed_audit: bool = False,
) -> DecisionDataset:
    """Collect frozen public states and trusted same-state branch utilities."""

    benchmark = benchmark or build_tier1_v03_benchmark()
    protocol = protocol or Protocol(protocol_version="pesco_v0_2")
    examples: List[DecisionExample] = []
    branch_groups = 0
    seed_observations = 0
    confirmation_groups = 0
    for question in benchmark.questions:
        for world in question.worlds:
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            # Keep descriptive benchmark IDs on evaluator records only.  Policy
            # observations receive the neutral question token; the public task
            # family remains an allowed context feature and no target action is
            # included.
            env.reset(
                question_id=question.policy_question_id,
                world_id=world.world_id,
                seed=17,
            )
            # The baseline experiment is identical for all methods and only its public
            # observation is retained by the policy dataset.
            env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            observation = env.visible_observation()
            initial_verdict = TrustedVerifier(protocol).evaluate(env._last_output, env)
            snapshot = env.snapshot()

            def utility_fn(output, verdict, branch_env):
                return benchmark_action_utility(
                    output,
                    verdict,
                    branch_env,
                    benchmark,
                    question=question,
                    world=world,
                    initial_observation=observation,
                )

            manager = BranchRolloutManager(
                environment=env,
                verifier=TrustedVerifier(protocol),
                utility_fn=utility_fn,
            )
            branches = manager.execute_paired_options(
                snapshot=snapshot,
                environment=env,
                options=ACTION_SET,
                seeds=protocol.exploration_seeds,
            )
            utilities = tuple(float(branch.utility) for branch in branches)
            # Re-execute each action once per exploration seed from the identical
            # snapshot.  These are seed-level *utility* observations used for the
            # reversal confidence gate; the vector branch above remains the canonical
            # four-seed estimate used for ordinary action training.
            branch_seed_utilities: Dict[str, List[float]] = {}
            branch_seed_confirmations: Dict[str, List[dict]] = {}
            for action in ACTION_SET:
                values: List[float] = []
                confirmation_audits: List[dict] = []
                for seed in protocol.exploration_seeds:
                    seed_env = env.clone_from_snapshot(snapshot)
                    seed_initial = seed_env.visible_observation()
                    setattr(seed_env, "_branch_initial_observation", seed_initial)
                    seed_output = seed_env.execute_option(action, seeds=(int(seed),))
                    seed_verdict = TrustedVerifier(protocol).evaluate(seed_output, seed_env, confirm=False)
                    # Confirmation must be an independent receipt for this exact
                    # one-seed replicate.  Never copy the vector branch's
                    # confirmation bonus into all seed utilities: doing so makes
                    # the seed-level reversal variance spuriously optimistic.
                    confirmation_seed = int(protocol.confirmation_seeds[
                        list(protocol.exploration_seeds).index(seed) % len(protocol.confirmation_seeds)
                    ])
                    confirmation_passed = None
                    confirmation_performed = False
                    if (
                        seed_verdict.evidence_state in {EvidenceState.SUPPORTED, EvidenceState.REFUTED}
                        and bool(getattr(seed_verdict, "validity_pass", False))
                    ):
                        confirmation_performed = True
                        candidate = seed_env.clone_from_snapshot(seed_env.snapshot())
                        if hasattr(candidate, "_simulate"):
                            confirmation_output = candidate._simulate(  # noqa: SLF001 - verifier boundary
                                method=seed_output.method,
                                option=action,
                                seeds=(confirmation_seed,),
                                confirmation=True,
                            )
                        else:
                            confirmation_output = candidate.execute_option(
                                action, seeds=(confirmation_seed,), confirmation=True
                            )
                        confirmation_verdict = TrustedVerifier(protocol).evaluate(
                            confirmation_output, candidate, confirm=False
                        )
                        confirmation_passed = bool(
                            confirmation_verdict.validity_pass
                            and confirmation_verdict.evidence_state is seed_verdict.evidence_state
                            and confirmation_output.dataset_hash != seed_output.dataset_hash
                            and confirmation_output.split_hash != seed_output.split_hash
                        )
                    from dataclasses import replace
                    utility_verdict = replace(
                        seed_verdict,
                        independent_confirmation_performed=confirmation_performed,
                        independent_confirmation_passed=bool(confirmation_passed),
                    )
                    values.append(float(benchmark_action_utility(
                        seed_output,
                        utility_verdict,
                        seed_env,
                        benchmark,
                        question=question,
                        world=world,
                        initial_observation=observation,
                    )))
                    confirmation_audits.append({
                        "exploration_seed": int(seed),
                        "confirmation_seed": confirmation_seed,
                        "performed": confirmation_performed,
                        # A receipt exists for every executed seed, but only a
                        # state-valid seed on which an independent confirmation was
                        # attempted belongs to the replication-rate denominator.
                        # Failed attempted confirmations retain ``passed=False``;
                        # ineligible/unevaluable seeds remain explicit audit rows.
                        "confirmation_eligible": bool(confirmation_passed is not None),
                        "passed": confirmation_passed,
                    })
                branch_seed_utilities[action.value] = values
                branch_seed_confirmations[action.value] = confirmation_audits
            branch_states = tuple(branch.verdict.evidence_state for branch in branches)
            all_confirmation_receipts = [
                receipt
                for receipts in branch_seed_confirmations.values()
                for receipt in receipts
                if isinstance(receipt, Mapping)
            ]
            confirmation_observed_n = len(all_confirmation_receipts)
            confirmation_eligible_n = sum(
                _confirmation_receipt_eligible(receipt)
                for receipt in all_confirmation_receipts
            )
            confirmation_passed_n = sum(
                _confirmation_receipt_eligible(receipt)
                and receipt.get("passed") is True
                for receipt in all_confirmation_receipts
            )
            confirmation_passed = bool(confirmation_eligible_n) and (
                confirmation_passed_n == confirmation_eligible_n
            )
            branch_confirmation = {
                branch.option.value: {
                    "eligible": bool(getattr(branch.verdict, "independent_confirmation_performed", False)),
                    "confirmation_eligible": bool(
                        sum(
                            _confirmation_receipt_eligible(receipt)
                            for receipt in branch_seed_confirmations.get(branch.option.value, ())
                            if isinstance(receipt, Mapping)
                        )
                    ),
                    "passed": bool(getattr(branch.verdict, "independent_confirmation_passed", False)),
                    "replicate_receipt_n": len(branch_seed_confirmations.get(branch.option.value, ())),
                    "confirmation_eligible_n": sum(
                        _confirmation_receipt_eligible(receipt)
                        for receipt in branch_seed_confirmations.get(branch.option.value, ())
                        if isinstance(receipt, Mapping)
                    ),
                }
                for branch in branches
            }
            metadata = {
                "family": question.family,
                "variant": int(question.variant),
                "question_world_group_id": f"{question.question_id}|{world.world_id}",
                "record_granularity": "question_world_group",
                "target_action": question.target_action(world.world_id).value,
                "backend": getattr(branches[0].output, "backend", "unknown"),
                "initial_trusted_state": initial_verdict.evidence_state.value,
                "reward_source": "public_family_state_transition_utility",
                "policy_features_exclude_hidden_world": True,
                "exploration_seed_count": len(protocol.exploration_seeds),
                "confirmation_seed_count": len(protocol.confirmation_seeds),
                # Keep eligibility and success separate.  A group with no
                # independently performed confirmation is not a failed
                # confirmation; it is outside the conditional denominator.
                "confirmation_eligible": bool(confirmation_eligible_n),
                "confirmation_observed_n": confirmation_observed_n,
                "confirmation_receipt_n": confirmation_observed_n,
                "confirmation_eligible_n": confirmation_eligible_n,
                "confirmation_passed_n": confirmation_passed_n,
                "confirmation_passed": bool(confirmation_passed),
                "branch_confirmation": branch_confirmation,
                # Per-branch execution accounting is evaluator output, not a policy
                # feature.  Keeping cost and validity beside the frozen utility
                # vector lets the D ablation report utility per experimental cost
                # and distinguish an erroneous local optimisation from a valid
                # repair without consulting the hidden target-action table.
                "branch_costs": {
                    branch.option.value: float(getattr(branch.output, "execution_cost", 0.0))
                    for branch in branches
                },
                "branch_validity": {
                    branch.option.value: bool(getattr(branch.verdict, "validity_pass", False))
                    for branch in branches
                },
                "branch_evidence_states": {
                    branch.option.value: getattr(branch.verdict, "evidence_state", EvidenceState.INSUFFICIENT).value
                    for branch in branches
                },
                "branch_seed_utilities": branch_seed_utilities,
                "branch_seed_confirmations": branch_seed_confirmations,
                "seed_utility_observation_count": sum(len(values) for values in branch_seed_utilities.values()),
                "action_level_row_count": len(ACTION_SET),
                "seed_level_observation_count": len(ACTION_SET) * len(protocol.exploration_seeds),
                "seed_utility_confirmation_bonus_source": {
                    "type": "independent_per_seed_confirmation_receipts",
                    "not_independent_per_seed_confirmation": False,
                    "vector_confirmation_bonus_copied": False,
                    "bonus": 0.0,
                },
            }
            if include_seed_audit:
                metadata["branch_seed_values"] = [
                    list(getattr(branch, "seed_values", ())) for branch in branches
                ]
            examples.append(DecisionExample(
                observation=observation,
                branch_utilities=utilities,
                branch_states=branch_states,
                state_target=initial_verdict.evidence_state,
                split=question.split,
                question_id=question.question_id,
                world_id=world.world_id,
                world_pair_id=f"{question.question_id}:supported_refuted",
                confirmation_passed=confirmation_passed,
                metadata=metadata,
            ))
            branch_groups += 1
            seed_observations += len(protocol.exploration_seeds) * len(ACTION_SET)
            confirmation_groups += int(confirmation_passed)

    # Build only statistically decisive supported/refuted pair reversals.  The
    # evaluator labels are independent of policy features; uncertain pairs are not
    # admitted to the flip objective.
    reversals: List[ReversalExample] = []
    # Reversal endpoints must come from the same registered question.  Earlier
    # v0.3 code paired any Supported world with any Refuted world in a family,
    # creating cross-question constraints and invalid macro denominators.
    by_question_kind: Dict[str, Dict[str, List[int]]] = {}
    for index, example in enumerate(examples):
        question_id = str(example.question_id)
        kind = example.world_id.rsplit("__", 1)[-1]
        by_question_kind.setdefault(question_id, {}).setdefault(kind, []).append(index)
    reversal_statistics: List[dict] = []
    for question_id, by_kind in by_question_kind.items():
        if not {"supported", "refuted"}.issubset(by_kind):
            continue
        for left_index in by_kind["supported"]:
            for right_index in by_kind["refuted"]:
                left, right = examples[left_index], examples[right_index]
                for action_left in ACTION_SET:
                    for action_right in ACTION_SET:
                        if action_left is action_right:
                            continue
                        left_receipts = left.metadata.get("branch_seed_confirmations", {}).get(action_left.value, ())
                        right_receipts = right.metadata.get("branch_seed_confirmations", {}).get(action_right.value, ())
                        left_eligible_receipts = tuple(
                            receipt for receipt in left_receipts
                            if isinstance(receipt, Mapping) and _confirmation_receipt_eligible(receipt)
                        )
                        right_eligible_receipts = tuple(
                            receipt for receipt in right_receipts
                            if isinstance(receipt, Mapping) and _confirmation_receipt_eligible(receipt)
                        )
                        left_passed = sum(_confirmation_passed(receipt) for receipt in left_eligible_receipts)
                        right_passed = sum(_confirmation_passed(receipt) for receipt in right_eligible_receipts)
                        if (
                            not left_eligible_receipts
                            or not right_eligible_receipts
                            or left_passed < max(1, (len(left_eligible_receipts) + 1) // 2)
                            or right_passed < max(1, (len(right_eligible_receipts) + 1) // 2)
                        ):
                            continue
                        pair = identify_confirmed_reversal(
                            question_id=question_id,
                            world_a=left.world_id,
                            world_b=right.world_id,
                            action_left=action_left,
                            action_right=action_right,
                            values_a_left=left.metadata["branch_seed_utilities"][action_left.value],
                            values_a_right=left.metadata["branch_seed_utilities"][action_right.value],
                            values_b_left=right.metadata["branch_seed_utilities"][action_left.value],
                            values_b_right=right.metadata["branch_seed_utilities"][action_right.value],
                            margin=0.05,
                            confidence=0.95,
                        )
                        reversal_statistics.append({**pair.to_dict(), "question_id": question_id, "left_split": left.split, "right_split": right.split})
                        if pair.confirmed:
                            reversals.append(ReversalExample(
                                left_index,
                                right_index,
                                action_left,
                                action_right,
                                margin=0.05,
                                confirmed=True,
                                weight=max(1.0, abs(float(pair.double_difference))),
                                lcb_left=float(pair.lcb_a),
                                ucb_right=float(pair.ucb_b),
                                sample_count=len(left.metadata["branch_seed_utilities"][action_left.value]),
                            ))
                            # Keep every confirmed action pair.  The evaluator
                            # applies a question-macro normalization below instead
                            # of letting a question with many pairs dominate.

    # Normalize reversal-loss weights so each question contributes total weight one.
    # This is an evaluator/training annotation and never enters policy features.
    reversal_groups: Dict[str, List[int]] = {}
    for index, pair in enumerate(reversals):
        qid = str(examples[pair.left].question_id)
        reversal_groups.setdefault(qid, []).append(index)
    for qid, indices in reversal_groups.items():
        raw_weights = [max(0.0, float(reversals[index].weight)) for index in indices]
        total_weight = sum(raw_weights)
        if total_weight <= 0.0:
            raw_weights = [1.0 for _ in indices]
            total_weight = float(len(indices))
        for index, raw_weight in zip(indices, raw_weights):
            pair = reversals[index]
            reversals[index] = ReversalExample(
                left=pair.left,
                right=pair.right,
                action_left=pair.action_left,
                action_right=pair.action_right,
                margin=pair.margin,
                confirmed=pair.confirmed,
                weight=float(raw_weight / total_weight),
                lcb_left=pair.lcb_left,
                ucb_right=pair.ucb_right,
                sample_count=pair.sample_count,
            )

    # Keep the seed-level uncertainty audit explicit.  Most current utility terms
    # are protocol-transition/cost terms and therefore can be identical across the
    # four stochastic replays even though the underlying NumPy estimates vary.  This
    # is useful for debugging, but it is not enough to authorize a formal variance
    # claim for a method comparison.  Report both counts so downstream readers cannot
    # mistake a four-seed array with zero utility spread for a precise estimate.
    seed_utility_arrays = [
        tuple(float(value) for value in values)
        for example in examples
        for values in dict(example.metadata.get("branch_seed_utilities", {})).values()
        if values
    ]
    varying_seed_utility_arrays = sum(
        len({round(value, 12) for value in values}) > 1
        for values in seed_utility_arrays
    )
    constant_seed_utility_arrays = len(seed_utility_arrays) - varying_seed_utility_arrays
    provenance = {
        "benchmark_schema_version": benchmark.schema_version,
        "benchmark_protocol_version": benchmark.protocol_version,
        "question_count": len(benchmark.questions),
        "world_count": len(benchmark.worlds),
        # A branch group is one question/world public state.  Each group has one
        # vector utility for every registered action and one seed-level replay per
        # action/seed pair.  Keep explicit names alongside compatibility aliases so
        # artifact readers cannot confuse 48, 192, and 768.
        "question_world_group_count": branch_groups,
        "branch_groups": branch_groups,
        "action_level_row_count": branch_groups * len(ACTION_SET),
        "action_level_rows": branch_groups * len(ACTION_SET),
        "exploration_seed_count": len(protocol.exploration_seeds),
        "seed_level_observation_count": seed_observations,
        "seed_level_observations": seed_observations,
        "exploration_seed_observations": seed_observations,
        "seed_utility_array_count": len(seed_utility_arrays),
        "seed_utility_varying_array_count": varying_seed_utility_arrays,
        "seed_utility_constant_array_count": constant_seed_utility_arrays,
        "seed_utility_variability_observed": bool(varying_seed_utility_arrays),
        "formal_seed_variance_claim_authorized": False,
        "confirmation_groups": confirmation_groups,
        "independent_confirmation_seeds": list(protocol.confirmation_seeds),
        "reversal_count": len(reversals),
        "reversal_statistics": reversal_statistics,
        "reversal_confidence": "paired_seed_normal_radius_95pct",
        "tier2_claim": False,
        "llm_claim": False,
    }
    return DecisionDataset(examples, reversals, provenance=provenance)


def _macro_f1(targets: Sequence[int], predictions: Sequence[int], classes: int) -> float:
    scores = []
    for cls in range(classes):
        tp = sum(target == cls and pred == cls for target, pred in zip(targets, predictions))
        fp = sum(target != cls and pred == cls for target, pred in zip(targets, predictions))
        fn = sum(target == cls and pred != cls for target, pred in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / max(1, classes)


def _cluster_ci(
    records: Sequence[Mapping[str, Any]],
    statistic,
    *,
    cluster_key: str = "question_id",
    seed: int = 17,
) -> dict:
    """Return an explicit question-cluster bootstrap result.

    The diagnostic splits are intentionally small.  The existing bootstrap helper
    returns ``None`` bounds when fewer than two independent clusters are available;
    preserve that distinction in the JSON instead of emitting a misleading zero
    width interval.  This is evaluator-side uncertainty only, not a formal final
    split claim.
    """

    clusters = {str(record.get(cluster_key, "unknown")) for record in records}
    point, lower, upper = clustered_bootstrap(
        list(records), statistic, n_bootstrap=500, seed=seed, cluster_key=cluster_key
    )
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "cluster_key": cluster_key,
        "cluster_count": len(clusters),
        "status": "estimable" if lower is not None and upper is not None else "NA_single_cluster",
    }


def _selected_action_confirmation_receipts(
    example: DecisionExample,
    action: ResearchAction,
) -> tuple[tuple[Mapping[str, Any], ...], Optional[str]]:
    """Return every replicate receipt for the action actually selected.

    Confirmation is a property of an executed action, not of the question/world
    group as a whole.  In particular, falling back to ``example.confirmation_passed``
    silently drops failed attempted confirmations; the per-replicate eligibility bit
    keeps those failures in the denominator while distinguishing unattempted rows.
    The v0.4 collector calls the receipt field ``branch_replicate_confirmation``;
    the frozen v0.3 collector uses ``branch_seed_confirmations``.  Both schemas are
    accepted here, while a missing schema remains explicitly unestimable.
    """

    metadata = example.metadata if isinstance(example.metadata, Mapping) else {}
    for field_name in ("branch_replicate_confirmation", "branch_seed_confirmations"):
        mapping = metadata.get(field_name)
        if not isinstance(mapping, Mapping):
            continue
        raw = mapping.get(action.value)
        if raw is None:
            continue
        if isinstance(raw, Mapping):
            raw = (raw,)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            return (), field_name
        receipts = tuple(item for item in raw if isinstance(item, Mapping))
        return receipts, field_name
    return (), None


def _confirmation_passed(receipt: Mapping[str, Any]) -> bool:
    """Read the canonical success bit from either v0.3 or v0.4 receipts."""

    return receipt.get("passed") is True or receipt.get("confirmation_passed") is True


def _confirmation_receipt_eligible(receipt: Mapping[str, Any]) -> bool:
    """Return whether a replicate has an attempted confirmation receipt.

    Collectors emit one audit row for every exploration replicate.  Invalid or
    insufficient replicates therefore have a receipt with ``passed=None`` but no
    attempted confirmation and must not be treated as successful confirmations.
    The canonical ``confirmation_eligible`` flag is preferred; legacy v0.3/v0.4
    exports are accepted through ``eligible``/``performed`` fallbacks.
    """

    if "confirmation_eligible" in receipt:
        return bool(receipt.get("confirmation_eligible"))
    if "eligible" in receipt:
        return bool(receipt.get("eligible"))
    if "performed" in receipt:
        return bool(receipt.get("performed"))
    return receipt.get("passed") is not None or receipt.get("confirmation_passed") is not None


def _question_macro_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    seed: int,
    replicates: int = 500,
) -> dict:
    """Bootstrap a reversal statistic with one normalized contribution per question.

    Reversal annotations are often numerous for one question.  Treating each pair
    as an independent row lets a question with many admissible action pairs dominate
    the estimate.  This helper first computes a (possibly ``weight``-weighted) pair
    mean within each question, then averages those question means and resamples the
    question means.  Thus every question has total weight one before the
    across-question mean is taken.
    """

    grouped: Dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        question_id = str(row.get("question_id", "unknown"))
        try:
            weight = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not math.isfinite(weight) or weight < 0.0:
            weight = 0.0
        grouped.setdefault(question_id, []).append((value, weight))
    question_means = []
    for values in grouped.values():
        if not values:
            continue
        total_weight = sum(weight for _, weight in values)
        if total_weight > 0.0:
            question_means.append(
                sum(value * weight for value, weight in values) / total_weight
            )
        else:
            question_means.append(sum(value for value, _ in values) / len(values))
    if not question_means:
        return {
            "point": None,
            "lower": None,
            "upper": None,
            "cluster_key": "question_id",
            "cluster_count": 0,
            "aggregation": "question_macro",
            "status": "NA_no_eligible_questions",
        }
    point = sum(question_means) / len(question_means)
    if len(question_means) < 2:
        return {
            "point": point,
            "lower": None,
            "upper": None,
            "cluster_key": "question_id",
            "cluster_count": len(question_means),
            "aggregation": "question_macro",
            "status": "NA_single_question_cluster",
        }
    rng = random.Random(int(seed))
    draws = []
    for _ in range(max(100, int(replicates))):
        draws.append(sum(rng.choice(question_means) for _ in question_means) / len(question_means))
    draws.sort()
    lower_index = max(0, int(0.025 * len(draws)) - 1)
    upper_index = min(len(draws) - 1, int(0.975 * len(draws)))
    return {
        "point": point,
        "lower": float(draws[lower_index]),
        "upper": float(draws[upper_index]),
        "cluster_key": "question_id",
        "cluster_count": len(question_means),
        "aggregation": "question_macro",
        "status": "estimable",
    }


def _weighted_pair_mean(rows: Sequence[Mapping[str, Any]], value_key: str) -> float:
    """Average a pair outcome using its within-question normalized weight."""

    weighted_values: list[tuple[float, float]] = []
    for row in rows:
        try:
            value = float(row[value_key])
            weight = float(row.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if not math.isfinite(weight) or weight < 0.0:
            weight = 0.0
        weighted_values.append((value, weight))
    if not weighted_values:
        return float("nan")
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0.0:
        return sum(value for value, _ in weighted_values) / len(weighted_values)
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _pairwise_reversal_ranking_correct(
    left_probabilities: Mapping[str, Any],
    right_probabilities: Mapping[str, Any],
    action_left: ResearchAction,
    action_right: ResearchAction,
) -> bool:
    """Check the two strict pairwise preferences optimized by flip loss."""

    try:
        left_margin = float(left_probabilities.get(action_left.value, 0.0)) - float(
            left_probabilities.get(action_right.value, 0.0)
        )
        right_margin = float(right_probabilities.get(action_right.value, 0.0)) - float(
            right_probabilities.get(action_left.value, 0.0)
        )
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(left_margin) and math.isfinite(right_margin) and left_margin > 0.0 and right_margin > 0.0)


def evaluate_differentiable_policy(
    policy: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    split: str,
    *,
    state_gate: bool = False,
    retain_records: bool = True,
) -> dict:
    examples = [example for example in dataset.examples if example.split == split]
    if not examples:
        return {"split": split, "example_count": 0, "confidence_interval_status": "not_estimable"}
    # ``action_correct`` is the primary, evaluator-owned utility winner metric.  The
    # pre-registered target-action table is retained separately as an audit label;
    # it must never be the optimization target or be silently presented as policy
    # performance.
    action_correct: List[bool] = []
    audit_action_correct: List[Optional[bool]] = []
    regrets: List[float] = []
    state_targets: List[int] = []
    state_predictions: List[int] = []
    entropies: List[float] = []
    best_action_probabilities: List[float] = []
    belief_scores: List[float] = []
    selected_utilities: List[float] = []
    selected_costs: List[float] = []
    selected_utility_per_cost: List[float] = []
    selected_confirmation_receipt_counts: List[int] = []
    selected_confirmation_passed_counts: List[int] = []
    selected_confirmation_missing: List[bool] = []
    erroneous_repair_flags: List[bool] = []
    invalid_local_optimization_flags: List[bool] = []
    selected_invalid_branch_flags: List[bool] = []
    records: List[dict] = []
    # Keep the original dataset index so confirmed reversal pairs can be scored as
    # pairs rather than as two independent rows.  A pair is eligible only when both
    # members are in the requested split and the evaluator marked it confirmed.
    predicted_actions: Dict[int, ResearchAction] = {}
    predicted_probabilities: Dict[int, Mapping[str, float]] = {}
    split_indices = {
        index for index, example in enumerate(dataset.examples) if example.split == split
    }
    for dataset_index, example in enumerate(dataset.examples):
        if example.split != split:
            continue
        action = policy_action(policy, example.observation, state_gate=state_gate)
        probabilities = policy_probabilities(policy, example.observation)
        predicted_state = None
        # Infer state from the model's actual state head using the same public feature
        # vector used for action selection.
        import torch
        with torch.no_grad():
            outputs = policy(observation_to_features(example.observation))
            predicted_state = STATE_SET[int(outputs["state_logits"].argmax(dim=-1).item())]
            state_probability = float(torch.softmax(outputs["state_logits"], dim=-1).max().item())
            belief_map = {
                str(getattr(belief, "hypothesis_id", "")): float(getattr(belief, "probability", 0.5))
                for belief in getattr(example.observation, "hypothesis_beliefs", ())
            }
            belief_logits = outputs["belief_logits"]
            belief_probabilities = torch.sigmoid(belief_logits).squeeze(0)
            active_index = 1 if example.observation.active_hypothesis_id == "H_B" else 0
            active_target = 1.0 if example.state_target is EvidenceState.SUPPORTED else 0.0 if example.state_target is EvidenceState.REFUTED else 0.5
            active_probability = float(belief_probabilities[active_index].item())
            belief_scores.append(float(-(active_target * math.log(max(1e-7, active_probability)) + (1.0 - active_target) * math.log(max(1e-7, 1.0 - active_probability)))))
            entropy = float((-(torch.softmax(outputs["action_logits"], dim=-1) * torch.log_softmax(outputs["action_logits"], dim=-1)).sum()).item())
        selected = ACTION_SET.index(action)
        selected_utility = float(example.branch_utilities[selected])
        cost_map = example.metadata.get("branch_costs", {})
        selected_cost = float(cost_map.get(action.value, 0.0) or 0.0)
        selected_utilities.append(selected_utility)
        selected_costs.append(selected_cost)
        if selected_cost > 1e-12:
            selected_utility_per_cost.append(selected_utility / selected_cost)
        confirmation_receipts, confirmation_receipt_source = _selected_action_confirmation_receipts(
            example, action
        )
        # Every exploration replicate is retained in ``confirmation_receipts`` for
        # audit.  The replication-rate denominator, however, is the subset for
        # which an independent confirmation was actually attempted.  This preserves
        # failed attempted confirmations (passed=False) while excluding invalid or
        # insufficient rows that could not produce a confirmation receipt.
        eligible_confirmation_receipts = tuple(
            receipt
            for receipt in confirmation_receipts
            if _confirmation_receipt_eligible(receipt)
        )
        confirmation_observed_n = len(confirmation_receipts)
        confirmation_receipt_n = len(eligible_confirmation_receipts)
        confirmation_passed_n = sum(
            _confirmation_passed(receipt) for receipt in eligible_confirmation_receipts
        )
        selected_confirmation_receipt_counts.append(confirmation_receipt_n)
        selected_confirmation_passed_counts.append(confirmation_passed_n)
        selected_confirmation_missing.append(confirmation_receipt_source is None)
        # These are deliberately public-state conditional diagnostics.  An
        # erroneous repair is a repair selected while the trusted pre-action
        # state is not INVALID.  An invalid local optimisation is a CONTINUE or
        # SWITCH selected while that state is INVALID *and* the selected action
        # loses to the evaluator-owned public branch-utility winner.  Thus a
        # scientifically correct Invalid+SWITCH row is not mislabeled.  Neither
        # denominator uses the hidden target-action audit field.
        erroneous_repair_flags.append(
            example.state_target is not EvidenceState.INVALID
            and action is ResearchAction.REPAIR
        )
        invalid_local_optimization_flags.append(
            is_invalid_local_optimization(
                example.state_target,
                action,
                example.best_action,
            )
        )
        validity_map = example.metadata.get("branch_validity", {})
        selected_invalid_branch_flags.append(
            action.value in validity_map and not bool(validity_map.get(action.value, False))
        )
        audit_target_raw = example.metadata.get("target_action")
        try:
            audit_target = ResearchAction(audit_target_raw) if audit_target_raw is not None else None
        except (TypeError, ValueError):
            audit_target = None
        action_correct.append(action is example.best_action)
        audit_action_correct.append(action is audit_target if audit_target is not None else None)
        predicted_actions[dataset_index] = action
        predicted_probabilities[dataset_index] = probabilities
        regrets.append(float(max(example.branch_utilities) - example.branch_utilities[selected]))
        state_targets.append(STATE_SET.index(example.state_target))
        state_predictions.append(STATE_SET.index(predicted_state))
        entropies.append(entropy)
        best_action_probabilities.append(float(probabilities[example.best_action.value]))
        full_record = {
            "question_id": example.question_id,
            "world_id": example.world_id,
            "split": split,
            "selected_action": action.value,
            "best_action": example.best_action.value,
            # ``target_action_audit`` is never consumed by policy training or the
            # primary utility metric.  It is retained only for benchmark
            # calibration/audit comparisons.
            "target_action_audit": audit_target.value if audit_target is not None else None,
            # Backward-compatible alias for consumers of the original artifact
            # schema; callers must use the explicit audit field above for clarity.
            "target_action": audit_target.value if audit_target is not None else None,
            "action_correct": bool(action_correct[-1]),
            "utility_winner_correct": bool(action_correct[-1]),
            "audit_target_action_correct": audit_action_correct[-1],
            "regret": regrets[-1],
            "selected_utility": selected_utility,
            "selected_cost": selected_cost,
            "selected_utility_per_cost": (
                selected_utility / selected_cost if selected_cost > 1e-12 else None
            ),
            "selected_confirmation_receipt_source": confirmation_receipt_source,
            "selected_confirmation_eligible_n": confirmation_receipt_n,
            "selected_confirmation_receipt_n": confirmation_receipt_n,
            "selected_confirmation_observed_n": confirmation_observed_n,
            "selected_confirmation_ineligible_n": confirmation_observed_n - confirmation_receipt_n,
            "selected_confirmation_passed_n": confirmation_passed_n,
            "selected_confirmation_rate": (
                confirmation_passed_n / confirmation_receipt_n
                if confirmation_receipt_n else None
            ),
            "selected_confirmation_receipts": [dict(receipt) for receipt in confirmation_receipts],
            "selected_confirmation_eligible_receipts": [
                dict(receipt) for receipt in eligible_confirmation_receipts
            ],
            "erroneous_repair_action": bool(erroneous_repair_flags[-1]),
            "invalid_local_optimization": bool(invalid_local_optimization_flags[-1]),
            "invalid_local_optimization_definition": "invalid_state_and_continue_or_switch_not_public_best_action",
            "invalid_local_optimization_reference": "public_branch_utility_best_action",
            "selected_branch_invalid": bool(selected_invalid_branch_flags[-1]),
            "predicted_state": predicted_state.value,
            "true_state": example.state_target.value,
            "state_correct": predicted_state is example.state_target,
            "state_probability": state_probability,
            "belief_probability": active_probability,
            "belief_probabilities": {
                "H_A": float(belief_probabilities[0].item()),
                "H_B": float(belief_probabilities[1].item()),
            },
            "action_probabilities": probabilities,
            "entropy": entropy,
        }
        if retain_records:
            records.append(full_record)
        else:
            # The evaluator still needs scalar rows for cluster bootstrap and
            # confirmation denominators, but retaining every receipt/probability
            # dictionary can consume nearly a gigabyte across a seed matrix.
            records.append({
                key: full_record[key]
                for key in (
                    "question_id", "world_id", "split", "selected_action", "action_correct", "utility_winner_correct",
                    "audit_target_action_correct", "regret", "selected_confirmation_observed_n",
                    "selected_confirmation_ineligible_n", "predicted_state", "true_state",
                )
            })
    # Only same-question pairs can contribute to the primary reversal statistic.
    # Cross-question pairs from legacy artifacts are retained for audit but excluded
    # here because they cannot receive a normalized within-question macro weight.
    all_split_pairs = [
        pair for pair in dataset.reversals
        if bool(pair.confirmed) and pair.left in split_indices and pair.right in split_indices
    ]
    eligible_pairs = [
        pair for pair in all_split_pairs
        if dataset.examples[pair.left].question_id == dataset.examples[pair.right].question_id
    ]
    pair_rows: List[dict] = []
    for pair in eligible_pairs:
        left_probabilities = predicted_probabilities.get(pair.left, {})
        right_probabilities = predicted_probabilities.get(pair.right, {})
        try:
            pair_weight = float(pair.weight)
        except (TypeError, ValueError):
            pair_weight = 0.0
        if not math.isfinite(pair_weight) or pair_weight < 0.0:
            pair_weight = 0.0
        pair_rows.append({
            "question_id": str(dataset.examples[pair.left].question_id),
            "weight": pair_weight,
            "pairwise_ranking_correct": _pairwise_reversal_ranking_correct(
                left_probabilities,
                right_probabilities,
                pair.action_left,
                pair.action_right,
            ),
            "exact_top1_reversal_correct": bool(
                predicted_actions.get(pair.left) is pair.action_left
                and predicted_actions.get(pair.right) is pair.action_right
            ),
        })
    pairwise_ranking_correct = sum(
        bool(row["pairwise_ranking_correct"]) for row in pair_rows
    )
    exact_top1_reversal_correct = sum(
        bool(row["exact_top1_reversal_correct"]) for row in pair_rows
    )
    # Conditional action metrics use the public branch-utility winner and explicit
    # denominators.  Audit-target variants are reported separately below and are
    # never used by the trainer or primary score.
    required_switch_rows: List[ResearchAction] = []
    invalid_repair_rows: List[ResearchAction] = []
    insufficient_rows: List[bool] = []
    audit_switch_rows: List[ResearchAction] = []
    audit_repair_rows: List[ResearchAction] = []
    audit_insufficient_rows: List[bool] = []
    for index in sorted(split_indices):
        example = dataset.examples[index]
        predicted = predicted_actions[index]
        if example.best_action is ResearchAction.SWITCH:
            required_switch_rows.append(predicted)
        if example.best_action is ResearchAction.REPAIR:
            invalid_repair_rows.append(predicted)
        if example.state_target is EvidenceState.INSUFFICIENT:
            insufficient_rows.append(predicted is example.best_action)
        audit_target_raw = example.metadata.get("target_action")
        try:
            audit_target = ResearchAction(audit_target_raw) if audit_target_raw is not None else None
        except (TypeError, ValueError):
            audit_target = None
        if audit_target is ResearchAction.SWITCH:
            audit_switch_rows.append(predicted)
        if audit_target is ResearchAction.REPAIR:
            audit_repair_rows.append(predicted)
        if audit_target is not None and example.state_target is EvidenceState.INSUFFICIENT:
            audit_insufficient_rows.append(predicted is audit_target)
    # Confirmation is scored on the receipts for the selected action only.  The
    # denominator contains attempted/eligible confirmations, including failed
    # attempts (passed=False), while ineligible invalid/insufficient replicates are
    # retained in the observed/ineligible audit counts.  This prevents a policy from
    # obtaining a perfect rate by selecting branches whose failed confirmations are
    # silently dropped.
    confirmation_receipt_n = sum(selected_confirmation_receipt_counts)
    confirmation_observed_n = sum(
        int(record.get("selected_confirmation_observed_n", 0) or 0)
        for record in records
    )
    confirmation_ineligible_n = sum(
        int(record.get("selected_confirmation_ineligible_n", 0) or 0)
        for record in records
    )
    confirmation_passed_n = sum(selected_confirmation_passed_counts)
    confirmation_missing_n = sum(selected_confirmation_missing)
    required_switch_correct = sum(action is ResearchAction.SWITCH for action in required_switch_rows)
    invalid_repair_correct = sum(action is ResearchAction.REPAIR for action in invalid_repair_rows)
    insufficient_correct = sum(insufficient_rows)
    audit_switch_correct = sum(action is ResearchAction.SWITCH for action in audit_switch_rows)
    audit_repair_correct = sum(action is ResearchAction.REPAIR for action in audit_repair_rows)
    audit_action_observed = [value for value in audit_action_correct if value is not None]
    pair_margins: List[float] = []
    for pair in eligible_pairs:
        left_probs = predicted_probabilities.get(pair.left, {})
        right_probs = predicted_probabilities.get(pair.right, {})
        pair_margins.append(min(
            float(left_probs.get(pair.action_left.value, 0.0)) - float(left_probs.get(pair.action_right.value, 0.0)),
            float(right_probs.get(pair.action_right.value, 0.0)) - float(right_probs.get(pair.action_left.value, 0.0)),
        ))
    action_accuracy_ci = _cluster_ci(
        records,
        lambda rows: sum(bool(row.get("action_correct", False)) for row in rows) / max(1, len(rows)),
    )
    regret_ci = _cluster_ci(
        records,
        lambda rows: sum(float(row.get("regret", 0.0)) for row in rows) / max(1, len(rows)),
    )
    state_macro_f1_ci = _cluster_ci(
        records,
        lambda rows: _macro_f1(
            [STATE_SET.index(EvidenceState(row["true_state"])) for row in rows],
            [STATE_SET.index(EvidenceState(row["predicted_state"])) for row in rows],
            len(STATE_SET),
        ),
    )
    pairwise_ranking_accuracy_ci = _question_macro_bootstrap_ci(
        pair_rows,
        "pairwise_ranking_correct",
        seed=23,
        replicates=500,
    )
    exact_top1_reversal_accuracy_ci = _question_macro_bootstrap_ci(
        pair_rows,
        "exact_top1_reversal_correct",
        seed=29,
        replicates=500,
    )
    pairwise_ranking_accuracy = pairwise_ranking_accuracy_ci.get("point")
    exact_top1_reversal_accuracy = exact_top1_reversal_accuracy_ci.get("point")
    reversal_question_weights = {
        question_id: 1.0 / max(
            1,
            sum(row["question_id"] == question_id for row in pair_rows),
        )
        for question_id in sorted({row["question_id"] for row in pair_rows})
    }
    reversal_question_macro_weights = {
        question_id: 1.0 / max(1, len(reversal_question_weights))
        for question_id in reversal_question_weights
    }
    reversal_pair_normalized_weights: Dict[str, List[float]] = {}
    for question_id in reversal_question_weights:
        question_rows = [row for row in pair_rows if row["question_id"] == question_id]
        total_weight = sum(max(0.0, float(row.get("weight", 0.0))) for row in question_rows)
        if total_weight <= 0.0:
            total_weight = float(len(question_rows))
            normalized = [1.0 / total_weight for _ in question_rows]
        else:
            normalized = [
                max(0.0, float(row.get("weight", 0.0))) / total_weight
                for row in question_rows
            ]
        reversal_pair_normalized_weights[question_id] = normalized
    question_metric_rows: List[dict] = []
    for question_id in sorted({example.question_id for example in examples}):
        question_indices = [
            index for index in sorted(split_indices)
            if dataset.examples[index].question_id == question_id
        ]
        question_pair_rows = [
            row for row in pair_rows if row["question_id"] == question_id
        ]
        question_metric_rows.append({
            "question_id": str(question_id),
            "family": str(dataset.examples[question_indices[0]].metadata.get("family", ""))
            if question_indices else "",
            "example_count": len(question_indices),
            "action_accuracy": (
                sum(bool(action_correct[sorted(split_indices).index(index)]) for index in question_indices)
                / len(question_indices)
                if question_indices else None
            ),
            "mean_regret": (
                sum(float(regrets[sorted(split_indices).index(index)]) for index in question_indices)
                / len(question_indices)
                if question_indices else None
            ),
            "pair_count": len(question_pair_rows),
            "pairwise_reversal_ranking_accuracy": (
                _weighted_pair_mean(question_pair_rows, "pairwise_ranking_correct")
                if question_pair_rows else None
            ),
            "exact_top1_reversal_accuracy": (
                _weighted_pair_mean(question_pair_rows, "exact_top1_reversal_correct")
                if question_pair_rows else None
            ),
            "selected_confirmation_receipt_n": sum(
                selected_confirmation_receipt_counts[sorted(split_indices).index(index)]
                for index in question_indices
            ),
            "selected_confirmation_passed_n": sum(
                selected_confirmation_passed_counts[sorted(split_indices).index(index)]
                for index in question_indices
            ),
        })
    return {
        "split": split,
        "example_count": len(examples),
        "action_accuracy": sum(action_correct) / len(action_correct),
        "utility_winner_accuracy": sum(action_correct) / len(action_correct),
        "audit_target_action_n": len(audit_action_observed),
        "audit_target_action_accuracy": (
            sum(bool(value) for value in audit_action_observed) / len(audit_action_observed)
            if audit_action_observed else None
        ),
        "mean_regret": sum(regrets) / len(regrets),
        "research_regret": sum(regrets) / len(regrets),
        "selected_utility_sum": sum(selected_utilities),
        "selected_utility_mean": sum(selected_utilities) / len(selected_utilities),
        "selected_cost_sum": sum(selected_costs),
        "selected_cost_mean": sum(selected_costs) / len(selected_costs),
        "utility_per_cost": (
            sum(selected_utilities) / sum(selected_costs)
            if sum(selected_costs) > 1e-12 else None
        ),
        "mean_utility_per_cost": (
            sum(selected_utility_per_cost) / len(selected_utility_per_cost)
            if selected_utility_per_cost else None
        ),
        "utility_per_cost_n": len(selected_utility_per_cost),
        "state_macro_f1": _macro_f1(state_targets, state_predictions, len(STATE_SET)),
        "action_accuracy_ci": action_accuracy_ci,
        "mean_regret_ci": regret_ci,
        "state_macro_f1_ci": state_macro_f1_ci,
        "state_accuracy": sum(target == pred for target, pred in zip(state_targets, state_predictions)) / len(state_targets),
        "mean_entropy": sum(entropies) / len(entropies),
        "mean_best_action_probability": sum(best_action_probabilities) / len(best_action_probabilities),
        "mean_belief_log_loss": sum(belief_scores) / len(belief_scores),
        "independent_question_count": len({example.question_id for example in examples}),
        "question_metric_rows": question_metric_rows,
        # Keep a dedicated pairwise table for downstream two-level (seed x
        # question) aggregation.  ``question_metric_rows`` also carries these
        # fields, but the explicit alias makes it impossible for a consumer to
        # accidentally average ordinary action rows as reversal pairs.
        "pairwise_reversal_question_rows": [
            {
                "question_id": row["question_id"],
                "pair_count": row["pair_count"],
                "pairwise_reversal_ranking_accuracy": row["pairwise_reversal_ranking_accuracy"],
                "exact_top1_reversal_accuracy": row["exact_top1_reversal_accuracy"],
            }
            for row in question_metric_rows
            if int(row.get("pair_count", 0) or 0) > 0
        ],
        "confidence_interval_status": "estimable" if len({example.question_id for example in examples}) >= 2 else "not_estimable",
        # ``pairwise_reversal_ranking_accuracy`` is the metric aligned with the
        # two pairwise inequalities optimized by flip loss.  Exact two-endpoint
        # top-1 matching remains available as an ordinary action diagnostic and is
        # intentionally not called FlipAcc.
        "reversal_pair_eligible_n": len(eligible_pairs),
        "pairwise_reversal_ranking_correct_n": int(pairwise_ranking_correct),
        "pairwise_reversal_ranking_accuracy": pairwise_ranking_accuracy,
        "pairwise_reversal_ranking_accuracy_ci": pairwise_ranking_accuracy_ci,
        "exact_top1_reversal_correct_n": int(exact_top1_reversal_correct),
        "exact_top1_reversal_accuracy": exact_top1_reversal_accuracy,
        "exact_top1_reversal_accuracy_ci": exact_top1_reversal_accuracy_ci,
        # Backward-compatible aliases are explicitly marked as deprecated so old
        # artifact readers cannot silently mistake exact top-1 for the primary rank
        # metric.
        "flip_eligible_n": len(eligible_pairs),
        "flip_correct_n": int(exact_top1_reversal_correct),
        "flip_accuracy": (
            exact_top1_reversal_accuracy
        ),
        "pair_flip_accuracy": (
            pairwise_ranking_accuracy
        ),
        "pair_flip_accuracy_ci": pairwise_ranking_accuracy_ci,
        "flip_metric_alias": "deprecated_exact_top1_alias_and_pair_rank_alias",
        "reversal_pair_aggregation": "question_macro_equal_weight",
        # ``reversal_question_weights`` is a legacy pair-uniform audit alias.
        # Expose the actual question-macro and confidence-weighted pair weights
        # explicitly so consumers cannot mistake one for the other.
        "reversal_question_weights": reversal_question_weights,
        "reversal_question_weights_definition": "legacy pair-uniform 1/pair_count audit weights; not the across-question macro weights",
        "reversal_question_macro_weights": reversal_question_macro_weights,
        "reversal_pair_normalized_weights": reversal_pair_normalized_weights,
        "flip_mean_preference_margin": sum(pair_margins) / len(pair_margins) if pair_margins else None,
        "flip_margin_n": len(pair_margins),
        "required_switch_n": len(required_switch_rows),
        "required_switch_correct_n": required_switch_correct,
        "required_switch_rate": required_switch_correct / len(required_switch_rows) if required_switch_rows else None,
        "invalid_repair_n": len(invalid_repair_rows),
        "invalid_repair_correct_n": invalid_repair_correct,
        "invalid_repair_rate": invalid_repair_correct / len(invalid_repair_rows) if invalid_repair_rows else None,
        "erroneous_repair_n": sum(erroneous_repair_flags),
        "erroneous_repair_eligible_n": sum(
            example.state_target is not EvidenceState.INVALID for example in examples
        ),
        "erroneous_repair_rate": (
            sum(erroneous_repair_flags) / sum(
                example.state_target is not EvidenceState.INVALID for example in examples
            )
            if any(example.state_target is not EvidenceState.INVALID for example in examples)
            else None
        ),
        "invalid_local_optimization_n": sum(invalid_local_optimization_flags),
        "invalid_local_optimization_eligible_n": sum(
            example.state_target is EvidenceState.INVALID for example in examples
        ),
        "invalid_local_optimization_rate": (
            sum(invalid_local_optimization_flags) / sum(
                example.state_target is EvidenceState.INVALID for example in examples
            )
            if any(example.state_target is EvidenceState.INVALID for example in examples)
            else None
        ),
        "invalid_local_optimization_definition": {
            "eligible_state": EvidenceState.INVALID.value,
            "candidate_actions": [
                action.value for action in ACTION_SET
                if action in INVALID_LOCAL_OPTIMIZATION_ACTIONS
            ],
            "correct_action_reference": "public_branch_utility_best_action",
            "hidden_target_action_used": False,
            "correct_invalid_switch_excluded": True,
        },
        "invalid_correct_switch_n": sum(
            example.state_target is EvidenceState.INVALID
            and example.best_action is ResearchAction.SWITCH
            and predicted_actions.get(index) is ResearchAction.SWITCH
            for index, example in enumerate(dataset.examples)
            if index in split_indices
        ),
        "selected_invalid_branch_n": sum(selected_invalid_branch_flags),
        "selected_invalid_branch_rate": sum(selected_invalid_branch_flags) / len(examples),
        "insufficient_handling_correct_n": insufficient_correct,
        "audit_required_switch_n": len(audit_switch_rows),
        "audit_required_switch_correct_n": audit_switch_correct,
        "audit_required_switch_rate": (
            audit_switch_correct / len(audit_switch_rows)
            if audit_switch_rows else None
        ),
        "audit_invalid_repair_n": len(audit_repair_rows),
        "audit_invalid_repair_correct_n": audit_repair_correct,
        "audit_invalid_repair_rate": (
            audit_repair_correct / len(audit_repair_rows)
            if audit_repair_rows else None
        ),
        "insufficient_eligible_n": len(insufficient_rows),
        "audit_insufficient_eligible_n": len(audit_insufficient_rows),
        "audit_insufficient_handling_correct_n": sum(audit_insufficient_rows),
        "insufficient_handling_rate": (
            insufficient_correct / len(insufficient_rows) if insufficient_rows else None
        ),
        "confirmation_eligible_n": confirmation_receipt_n,
        # Explicit aliases make the two reported populations unambiguous for
        # downstream consumers.  ``confirmation_receipt_n`` is retained as the
        # historical observed-total field; the canonical denominator is the
        # attempted/eligible subset above.
        "confirmation_observed_n": confirmation_observed_n,
        "confirmation_receipt_n": confirmation_observed_n,
        "confirmation_ineligible_n": confirmation_ineligible_n,
        "confirmation_passed_n": confirmation_passed_n,
        "confirmation_rate": (
            confirmation_passed_n / confirmation_receipt_n
            if confirmation_receipt_n else None
        ),
        "confirmation_missing_receipt_n": confirmation_missing_n,
        "confirmation_metric_unit": "selected_action_replicate_receipt",
        "confirmation_denominator_unit": "eligible_selected_action_replicate_receipt",
        "confirmation_observed_definition": "all replicate receipts observed for the model-selected action, including ineligible/unevaluable rows",
        "confirmation_denominator_definition": "replicate receipts for the model-selected action where an independent confirmation was attempted",
        "records": records,
    }


def _cross_split_reversal_diagnostics(
    policy: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    split: str,
    *,
    state_gate: bool = False,
) -> dict:
    """Score confirmed reversals whose held-out side is an unseen world.

    ``diagnostic_ood`` is an internal question-level holdout, not the locked formal
    OOD partition.  This helper therefore reports the held-out evidence explicitly
    while carrying a boundary flag that prevents it from being interpreted as a
    final OOD result.
    """

    heldout_indices = {
        index for index, example in enumerate(dataset.examples) if example.split == split
    }
    train_worlds = {
        example.world_id for example in dataset.examples if example.split == "train"
    }
    unseen_indices = {
        index for index in heldout_indices if dataset.examples[index].world_id not in train_worlds
    }
    predicted = {
        index: policy_action(policy, example.observation, state_gate=state_gate)
        for index, example in enumerate(dataset.examples)
        if index in heldout_indices or index in {
            endpoint
            for pair in dataset.reversals
            for endpoint in (pair.left, pair.right)
            if pair.left in heldout_indices or pair.right in heldout_indices
        }
    }
    predicted_probabilities = {
        index: policy_probabilities(policy, dataset.examples[index].observation)
        for index in predicted
    }
    unseen_switch_examples = [
        index for index in sorted(unseen_indices)
        if dataset.examples[index].best_action is ResearchAction.SWITCH
    ]
    unseen_switch_correct = sum(
        predicted.get(index) is ResearchAction.SWITCH for index in unseen_switch_examples
    )
    unseen_action_correct = sum(
        predicted.get(index) is dataset.examples[index].best_action for index in sorted(unseen_indices)
    )
    cross_pairs = []
    for pair in dataset.reversals:
        if not pair.confirmed:
            continue
        left_heldout = pair.left in heldout_indices
        right_heldout = pair.right in heldout_indices
        # Exactly one endpoint held out gives a clean train -> unseen-world test.
        if left_heldout == right_heldout:
            continue
        if pair.left not in predicted or pair.right not in predicted:
            continue
        cross_pairs.append(pair)
    pairwise_pair_correct = sum(
        _pairwise_reversal_ranking_correct(
            predicted_probabilities[pair.left],
            predicted_probabilities[pair.right],
            pair.action_left,
            pair.action_right,
        )
        for pair in cross_pairs
    )
    exact_pair_correct = sum(
        predicted[pair.left] is pair.action_left
        and predicted[pair.right] is pair.action_right
        for pair in cross_pairs
    )
    reversal_switch_pairs = [
        pair for pair in cross_pairs
        if dataset.examples[pair.left].best_action is not dataset.examples[pair.right].best_action
    ]
    reversal_switch_correct = sum(
        _pairwise_reversal_ranking_correct(
            predicted_probabilities[pair.left],
            predicted_probabilities[pair.right],
            pair.action_left,
            pair.action_right,
        )
        for pair in reversal_switch_pairs
    )
    return {
        "heldout_split": split,
        "unseen_world_n": len(unseen_indices),
        "unseen_world_action_accuracy": (
            unseen_action_correct / len(unseen_indices) if unseen_indices else None
        ),
        "unseen_world_switch_n": len(unseen_switch_examples),
        "unseen_world_switch_correct_n": unseen_switch_correct,
        "unseen_world_switch_rate": (
            unseen_switch_correct / len(unseen_switch_examples)
            if unseen_switch_examples else None
        ),
        "cross_split_confirmed_pair_n": len(cross_pairs),
        "cross_split_pairwise_reversal_ranking_accuracy": (
            pairwise_pair_correct / len(cross_pairs) if cross_pairs else None
        ),
        "cross_split_exact_top1_reversal_accuracy": (
            exact_pair_correct / len(cross_pairs) if cross_pairs else None
        ),
        "cross_split_pair_flip_accuracy": (
            pairwise_pair_correct / len(cross_pairs) if cross_pairs else None
        ),
        "unseen_world_reversal_switch_pair_n": len(reversal_switch_pairs),
        "unseen_world_reversal_switch_pairwise_reversal_ranking_accuracy": (
            reversal_switch_correct / len(reversal_switch_pairs)
            if reversal_switch_pairs else None
        ),
        "unseen_world_reversal_switch_pair_flip_accuracy": (
            reversal_switch_correct / len(reversal_switch_pairs)
            if reversal_switch_pairs else None
        ),
        "formal_ood_access": False,
        "diagnostic_only": True,
    }


def _noise_condition_diagnostics(
    policy: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    split: str,
    *,
    state_gate: bool = False,
    noise_scales: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    replicates: int = 8,
) -> dict:
    """Evaluate controlled public-feature noise without changing labels.

    Noise is injected only into continuous public evidence features.  World IDs,
    target actions, and verifier outputs are never passed to the policy; they are
    used only to score the pre-registered utility winner after the perturbation.
    This is a robustness diagnostic, not a formal noisy-data benchmark.
    """

    import torch
    import torch.nn.functional as F

    examples = [example for example in dataset.examples if example.split == split]
    # Numeric evidence fields and named belief probabilities.  Categorical signal,
    # action-history, and task-family indicators remain fixed under this stress test.
    numeric_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12)
    result: Dict[str, dict] = {}
    for scale in noise_scales:
        correct_prob: List[float] = []
        greedy_correct: List[bool] = []
        switch_prob: List[float] = []
        switch_correct: List[bool] = []
        for example_index, example in enumerate(examples):
            base = observation_to_features(example.observation)
            for replicate in range(max(1, int(replicates))):
                token = f"{split}|{example.question_id}|{example.world_id}|{scale:.6f}|{replicate}"
                seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
                generator = torch.Generator().manual_seed(seed)
                noise = torch.zeros_like(base)
                if float(scale) > 0.0:
                    noise_values = torch.randn((len(numeric_indices),), generator=generator)
                    noise[list(numeric_indices)] = float(scale) * noise_values
                with torch.no_grad():
                    outputs = policy(base + noise)
                    action_probabilities = F.softmax(outputs["action_logits"], dim=-1).squeeze(0)
                    if state_gate:
                        predicted_state = STATE_SET[int(outputs["state_logits"].argmax(dim=-1).item())]
                        state_to_action = {
                            EvidenceState.SUPPORTED: ResearchAction.CONTINUE,
                            EvidenceState.REFUTED: ResearchAction.SWITCH,
                            EvidenceState.INSUFFICIENT: ResearchAction.SAMPLE,
                            EvidenceState.INVALID: ResearchAction.REPAIR,
                        }
                        predicted = state_to_action[predicted_state]
                    else:
                        predicted = ACTION_SET[int(action_probabilities.argmax().item())]
                target_index = ACTION_SET.index(example.best_action)
                correct_prob.append(float(action_probabilities[target_index].item()))
                greedy_correct.append(predicted is example.best_action)
                if example.best_action is ResearchAction.SWITCH:
                    switch_prob.append(float(action_probabilities[ACTION_SET.index(ResearchAction.SWITCH)].item()))
                    switch_correct.append(predicted is ResearchAction.SWITCH)
        scale_key = f"{float(scale):.3f}"
        result[scale_key] = {
            "noise_scale": float(scale),
            "replicates": max(1, int(replicates)),
            "example_count": len(examples),
            "correct_action_probability": (
                sum(correct_prob) / len(correct_prob) if correct_prob else None
            ),
            "greedy_action_accuracy": (
                sum(greedy_correct) / len(greedy_correct) if greedy_correct else None
            ),
            "switch_example_n": len(switch_prob),
            "switch_action_probability": (
                sum(switch_prob) / len(switch_prob) if switch_prob else None
            ),
            "switch_greedy_accuracy": (
                sum(switch_correct) / len(switch_correct) if switch_correct else None
            ),
            "diagnostic_only": True,
        }
    return {
        "split": split,
        "noise_definition": "iid Gaussian perturbation of continuous public evidence features",
        "feature_noise_indices": list(numeric_indices),
        "conditions": result,
        "formal_noisy_benchmark": False,
    }


def _sample_efficiency_diagnostics(
    dataset: DecisionDataset,
    config: Tier1SuiteConfig,
    trainer: DifferentiableStrategyTrainer,
    methods: Sequence[str],
    full_policies: Mapping[str, DifferentiableStrategyPolicy],
) -> dict:
    """Run a deterministic question-level data-ablation diagnostic.

    The selected train questions are a nested lexicographic prefix, so every
    fraction is reproducible and no row-level leakage is introduced.  Dev and
    diagnostic-OOD questions remain untouched.  Results are explicitly diagnostic
    because the formal promotion/final partitions are locked and unopened.
    """

    train_question_ids = sorted({
        example.question_id for example in dataset.examples if example.split == "train"
    })
    fractions = (0.25, 0.50, 0.75, 1.0)
    output: Dict[str, dict] = {
        "selection_rule": "nested_question_level_lexicographic_prefix",
        "train_question_count": len(train_question_ids),
        "fractions": {},
        "optimizer_step_cap": int(config.trainer.max_optimizer_steps),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    }
    for fraction in fractions:
        count = max(1, int(math.ceil(len(train_question_ids) * float(fraction))))
        selected_questions = set(train_question_ids[:count])
        fraction_key = f"{fraction:.2f}"
        output["fractions"][fraction_key] = {
            "fraction": float(fraction),
            "selected_train_question_ids": sorted(selected_questions),
            "methods": {},
        }
        for method in methods:
            if fraction == 1.0 and method in full_policies:
                policy = full_policies[method]
                log_steps = int(config.trainer.max_optimizer_steps)
            else:
                sampled = copy.deepcopy(dataset)
                for example in sampled.examples:
                    if example.split == "train" and example.question_id not in selected_questions:
                        # Keep original indices so confirmed reversal endpoints remain
                        # auditable; the trainer's train-index gate excludes them.
                        example.split = "sample_holdout"
                policy, sample_log = trainer.fit(sampled, method)
                log_steps = int(sample_log.optimizer_steps)
            heldout_metrics = evaluate_differentiable_policy(policy, dataset, "diagnostic_ood")
            dev_metrics = evaluate_differentiable_policy(policy, dataset, "dev")
            selected_train_n = sum(
                example.split == "train" and example.question_id in selected_questions
                for example in dataset.examples
            )
            selected_reversal_n = sum(
                pair.confirmed
                and dataset.examples[pair.left].question_id in selected_questions
                and dataset.examples[pair.right].question_id in selected_questions
                and dataset.examples[pair.left].split == "train"
                and dataset.examples[pair.right].split == "train"
                for pair in dataset.reversals
            )
            output["fractions"][fraction_key]["methods"][method] = {
                "train_question_n": count,
                "train_example_n": int(selected_train_n),
                "confirmed_train_pair_n": int(selected_reversal_n),
                "optimizer_steps": log_steps,
                "dev_action_accuracy": dev_metrics.get("action_accuracy"),
                "dev_mean_regret": dev_metrics.get("mean_regret"),
                "dev_utility_per_cost": dev_metrics.get("utility_per_cost"),
                "dev_pairwise_reversal_ranking_accuracy": dev_metrics.get("pairwise_reversal_ranking_accuracy"),
                "dev_pair_flip_accuracy": dev_metrics.get("pairwise_reversal_ranking_accuracy"),
                "diagnostic_ood_action_accuracy": heldout_metrics.get("action_accuracy"),
                "diagnostic_ood_mean_regret": heldout_metrics.get("mean_regret"),
                "diagnostic_ood_utility_per_cost": heldout_metrics.get("utility_per_cost"),
                "diagnostic_ood_pairwise_reversal_ranking_accuracy": heldout_metrics.get("pairwise_reversal_ranking_accuracy"),
                "diagnostic_ood_pair_flip_accuracy": heldout_metrics.get("pairwise_reversal_ranking_accuracy"),
            }
    return output


def run_tier1_differentiable_suite(
    output_dir: str | Path,
    *,
    benchmark: Optional[Tier1Benchmark] = None,
    protocol: Optional[Protocol] = None,
    config: Optional[Tier1SuiteConfig] = None,
) -> dict:
    config = config or Tier1SuiteConfig()
    benchmark = benchmark or build_tier1_v03_benchmark()
    protocol = protocol or Protocol(protocol_version="pesco_v0_2")
    if str(protocol.protocol_version) != str(benchmark.protocol_version):
        raise ValueError(
            f"protocol/benchmark mismatch: {protocol.protocol_version!r} != {benchmark.protocol_version!r}"
        )
    protocol_payload = {
        "protocol_version": protocol.protocol_version,
        "delta_min": protocol.delta_min,
        "confidence_level": protocol.confidence_level,
        "exploration_seeds": list(protocol.exploration_seeds),
        "confirmation_seeds": list(protocol.confirmation_seeds),
        # Keep the digest schema identical to the independent A runner.  A shared
        # protocol version with different omitted fields is otherwise ambiguous:
        # two artifacts could claim v0.2 while being bound to different verifier
        # rules.  These fields are all evaluator-side and never enter policy inputs.
        "invalid_precedence": protocol.invalid_precedence,
        "independent_confirmation_required": protocol.independent_confirmation_required,
        "max_budget": protocol.max_budget,
    }
    protocol_digest = "sha256:" + hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = collect_tier1_v03_dataset(benchmark, protocol)
    dataset.save_json(output / "dataset.json", include_audit=True)
    # Keep a separately materialized policy-facing export.  The training/evaluator
    # artifact retains audit labels for reproducibility, while this file demonstrates
    # that public observations, split IDs, and action utilities can be inspected
    # without exposing world IDs, target-action labels, or latent truth.
    dataset.save_json(output / "dataset_public.json", include_audit=False)
    logs: Dict[str, dict] = {}
    metrics: Dict[str, dict] = {}
    policies: Dict[str, DifferentiableStrategyPolicy] = {}
    trainer = DifferentiableStrategyTrainer(config.trainer)
    for method in config.methods:
        state_gate = method == "StateGateOnly"
        policy, log = trainer.fit(dataset, method)
        policies[method] = policy
        logs[method] = log.to_dict()
        metrics[method] = {
            split: evaluate_differentiable_policy(policy, dataset, split, state_gate=state_gate)
            for split in ("train", "dev", "diagnostic_ood")
        }
        torch_path = output / f"policy_{method.replace(' ', '_')}.pt"
        import torch
        torch.save(policy.state_dict(), torch_path)
    target_agreement = sum(
        example.best_action.value == str(example.metadata.get("target_action", example.best_action.value))
        for example in dataset.examples
    )
    ablations: Dict[str, Any] = {}
    for left, right in (("PESCO-Full", "PESCO-NoFlipLoss"), ("PESCO-Full", "PESCO-BranchOnly"), ("PESCO-Full", "PESCO-NoBranch"), ("PESCO-Full", "Evidence-Gated SMOPD")):
        if left in metrics and right in metrics:
            ablations[f"{left}_vs_{right}"] = {
                split: {
                    "action_accuracy_delta": metrics[left][split].get("action_accuracy", float("nan")) - metrics[right][split].get("action_accuracy", float("nan")),
                    "regret_delta": metrics[left][split].get("mean_regret", float("nan")) - metrics[right][split].get("mean_regret", float("nan")),
                    "state_macro_f1_delta": metrics[left][split].get("state_macro_f1", float("nan")) - metrics[right][split].get("state_macro_f1", float("nan")),
                }
                for split in ("train", "dev", "diagnostic_ood")
            }
    # The feedback asks for named C/D/E experiments rather than one opaque method
    # table.  Keep these machine-readable summaries next to ``suite.json``.  They
    # deliberately remain diagnostic-only: the benchmark is a frozen CPU MLP
    # mechanism reference, not an LLM or final ID/OOD evaluation.
    def compact(method_names: Sequence[str]) -> Dict[str, Any]:
        return {
            method: {
                split: {
                    key: metrics[method][split].get(key)
                    for key in (
                        "example_count", "action_accuracy", "audit_target_action_accuracy",
                        "mean_regret", "research_regret", "selected_utility_mean", "selected_cost_mean",
                        "utility_per_cost", "mean_utility_per_cost", "utility_per_cost_n",
                        "action_accuracy_ci", "mean_regret_ci", "state_macro_f1_ci",
                        "pairwise_reversal_ranking_accuracy_ci", "exact_top1_reversal_accuracy_ci",
                        "erroneous_repair_n", "erroneous_repair_eligible_n", "erroneous_repair_rate",
                        "invalid_local_optimization_n", "invalid_local_optimization_eligible_n",
                        "invalid_local_optimization_rate", "invalid_correct_switch_n",
                        "invalid_local_optimization_definition", "selected_invalid_branch_n",
                        "selected_invalid_branch_rate", "state_macro_f1", "state_accuracy",
                        "mean_belief_log_loss",
                        "mean_best_action_probability", "reversal_pair_eligible_n",
                        "pairwise_reversal_ranking_correct_n", "pairwise_reversal_ranking_accuracy",
                        "pairwise_reversal_ranking_accuracy_ci", "exact_top1_reversal_correct_n",
                        "exact_top1_reversal_accuracy", "exact_top1_reversal_accuracy_ci",
                        "flip_eligible_n", "flip_correct_n", "flip_accuracy", "pair_flip_accuracy",
                        "pair_flip_accuracy_ci", "flip_metric_alias", "reversal_pair_aggregation",
                        "reversal_question_weights", "flip_mean_preference_margin",
                        "required_switch_n", "required_switch_rate", "invalid_repair_n",
                        "invalid_repair_rate", "audit_required_switch_n",
                        "audit_required_switch_rate", "audit_invalid_repair_n",
                        "audit_invalid_repair_rate", "insufficient_eligible_n",
                        "insufficient_handling_rate", "confirmation_eligible_n",
                        "confirmation_passed_n", "confirmation_rate",
                    )
                }
                for split in ("train", "dev", "diagnostic_ood")
                if method in metrics
            }
            for method in method_names
            if method in metrics
        }

    state_action_map: Dict[str, set[str]] = {}
    public_observation_action_map: Dict[str, set[str]] = {}
    for example in dataset.examples:
        state_action_map.setdefault(example.state_target.value, set()).add(example.best_action.value)
        public_fingerprint = json.dumps(
            example.observation.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        # Use the evaluator-owned branch-utility winner for this hard-case audit.
        # ``metadata[target_action]`` is retained only as an external calibration
        # label and must never define the public-state diversity gate.
        public_observation_action_map.setdefault(public_fingerprint, set()).add(
            example.best_action.value
        )
    same_state_different_actions = any(len(actions) > 1 for actions in state_action_map.values())
    same_public_observation_different_actions = any(
        len(actions) > 1 for actions in public_observation_action_map.values()
    )
    dataset.provenance["public_observation_collision_count"] = sum(
        len(actions) > 1 for actions in public_observation_action_map.values()
    )
    dataset.provenance["same_public_observation_different_optimal_actions"] = bool(
        same_public_observation_different_actions
    )
    # Re-save after adding the exact public-state collision audit fields.
    dataset.save_json(output / "dataset.json", include_audit=True)
    dataset.save_json(output / "dataset_public.json", include_audit=False)
    genuine_updates = all(
        int(logs.get(method, {}).get("optimizer_steps", 0)) > 0
        for method in config.methods
    )
    full_log = logs.get("PESCO-Full", {})
    full_flip_updates = bool(
        float(full_log.get("flip_gradient_norm", 0.0)) > 0.0
        and int(full_log.get("flip_gradient_probe_count", 0)) > 0
        and any(float(epoch.get("flip_loss", 0.0)) > 0.0 for epoch in full_log.get("epochs", []))
    )
    e_methods = ("PESCO-BranchOnly", "PESCO-NoFlipLoss", "PESCO-Full", "Evidence-Gated SMOPD")
    unseen_world_diagnostics = {
        method: {
            split: _cross_split_reversal_diagnostics(
                policies[method], dataset, split, state_gate=(method == "StateGateOnly")
            )
            for split in ("dev", "diagnostic_ood")
        }
        for method in e_methods
        if method in policies
    }
    noise_diagnostics = {
        method: _noise_condition_diagnostics(
            policies[method], dataset, "diagnostic_ood", state_gate=(method == "StateGateOnly")
        )
        for method in e_methods
        if method in policies
    }
    # This is intentionally a nested, question-level diagnostic.  It does not
    # authorize promotion/final claims and never touches the locked formal splits.
    sample_efficiency = _sample_efficiency_diagnostics(
        dataset,
        config,
        trainer,
        e_methods,
        policies,
    )
    experiment_c = {
        "schema_version": "pesco_experiment_c_v0.2",
        "experiment": "C_state_reward_sufficiency",
        "status": "completed_cpu_diagnostic",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "objective": "test whether ordinary state rewards already solve the public research-action task",
        "methods": ["SFT", "GRPO-Terminal", "GRPO-FourState", "StateGateOnly"],
        "same_frozen_dataset": True,
        "same_state_different_optimal_actions": bool(same_state_different_actions),
        "same_public_observation_different_optimal_actions": bool(same_public_observation_different_actions),
        "same_public_observation_collision_count": sum(
            len(actions) > 1 for actions in public_observation_action_map.values()
        ),
        "genuine_training_updates": bool(genuine_updates),
        "results": compact(["SFT", "GRPO-Terminal", "GRPO-FourState", "StateGateOnly"]),
        "gates": {
            "same_state_different_optimal_actions": bool(same_state_different_actions),
            "same_public_observation_different_optimal_actions": bool(same_public_observation_different_actions),
            "genuine_training_updates": bool(genuine_updates),
            "formal_final_splits_unopened": True,
        },
        "interpretation": "CPU mechanism evidence only; hidden target actions are audit fields and are excluded from optimization.",
    }
    experiment_d = {
        "schema_version": "pesco_experiment_d_v0.2",
        "experiment": "D_branch_ablation",
        "status": "completed_cpu_diagnostic",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "objective": "isolate same-state branch credit from ordinary terminal/state objectives",
        "methods": ["GRPO-FourState", "PESCO-NoBranch", "PESCO-BranchOnly", "PESCO-Full"],
        "same_frozen_dataset": True,
        "results": compact(["GRPO-FourState", "PESCO-NoBranch", "PESCO-BranchOnly", "PESCO-Full"]),
        "primary_metrics": {
            method: {
                split: {
                    "research_regret": metrics[method][split].get("research_regret"),
                    "erroneous_repair_action_rate": metrics[method][split].get("erroneous_repair_rate"),
                    "invalid_local_optimization_n": metrics[method][split].get("invalid_local_optimization_n"),
                    "invalid_correct_switch_n": metrics[method][split].get("invalid_correct_switch_n"),
                    "invalid_local_optimization_rate": metrics[method][split].get("invalid_local_optimization_rate"),
                    "invalid_local_optimization_definition": metrics[method][split].get("invalid_local_optimization_definition"),
                    "utility_per_cost": metrics[method][split].get("utility_per_cost"),
                    "selected_invalid_branch_rate": metrics[method][split].get("selected_invalid_branch_rate"),
                }
                for split in ("train", "dev", "diagnostic_ood")
            }
            for method in ("GRPO-FourState", "PESCO-NoBranch", "PESCO-BranchOnly", "PESCO-Full")
            if method in metrics
        },
        "ablation_summary": {
            key: value for key, value in ablations.items()
            if any(name in key for name in ("NoBranch", "BranchOnly"))
        },
        "gates": {
            "genuine_training_updates": bool(genuine_updates),
            "matched_candidate_actions": True,
            "formal_final_splits_unopened": True,
        },
    }
    full_method_available = "PESCO-Full" in metrics and "PESCO-Full" in logs
    no_flip_method_available = "PESCO-NoFlipLoss" in metrics and "PESCO-NoFlipLoss" in logs
    smopd_method_available = "Evidence-Gated SMOPD" in metrics and "Evidence-Gated SMOPD" in logs
    if full_method_available and no_flip_method_available:
        full_vs_no_flip_action_improves = {
            split: metrics["PESCO-Full"][split]["action_accuracy"]
            > metrics["PESCO-NoFlipLoss"][split]["action_accuracy"]
            for split in ("train", "dev", "diagnostic_ood")
        }
        full_vs_no_flip_regret_improves = {
            split: metrics["PESCO-Full"][split]["mean_regret"]
            < metrics["PESCO-NoFlipLoss"][split]["mean_regret"]
            for split in ("train", "dev", "diagnostic_ood")
        }
        # A matched-budget claim is only considered positive when the full method
        # wins the same primary criterion on every reported diagnostic split.  A
        # single favorable split is retained as a descriptive delta, not a gate.
        full_beats_no_flip_all_reported_splits = bool(
            all(full_vs_no_flip_action_improves.values())
            or all(full_vs_no_flip_regret_improves.values())
        )
        matched_budget_comparison = {
            "full_vs_no_flip_action_accuracy_delta": {
                split: metrics["PESCO-Full"][split]["action_accuracy"] - metrics["PESCO-NoFlipLoss"][split]["action_accuracy"]
                for split in ("train", "dev", "diagnostic_ood")
            },
            "full_vs_no_flip_regret_delta": {
                split: metrics["PESCO-Full"][split]["mean_regret"] - metrics["PESCO-NoFlipLoss"][split]["mean_regret"]
                for split in ("train", "dev", "diagnostic_ood")
            },
            "full_vs_no_flip_pairwise_reversal_ranking_accuracy_delta": {
                split: (
                    metrics["PESCO-Full"][split]["pairwise_reversal_ranking_accuracy"] - metrics["PESCO-NoFlipLoss"][split]["pairwise_reversal_ranking_accuracy"]
                    if metrics["PESCO-Full"][split]["pairwise_reversal_ranking_accuracy"] is not None
                    and metrics["PESCO-NoFlipLoss"][split]["pairwise_reversal_ranking_accuracy"] is not None
                    else None
                )
                for split in ("train", "dev", "diagnostic_ood")
            },
            "full_vs_no_flip_action_improves": full_vs_no_flip_action_improves,
            "full_vs_no_flip_regret_improves": full_vs_no_flip_regret_improves,
            "full_improves_action_or_regret": any(
                full_vs_no_flip_action_improves.values()
            ) or any(full_vs_no_flip_regret_improves.values()),
            "full_beats_no_flip_on_all_reported_splits": full_beats_no_flip_all_reported_splits,
            "optimizer_steps_equal": all(
                int(logs[method].get("optimizer_steps", 0)) == int(config.trainer.max_optimizer_steps)
                for method in ("PESCO-Full", "PESCO-NoFlipLoss")
            ),
            "smopd_parameter_compute_equal": False,
            "formal_final_splits_unopened": True,
            "status": "estimable",
        }
        if smopd_method_available:
            full_vs_smopd_action_delta = {
                split: metrics["PESCO-Full"][split]["action_accuracy"]
                - metrics["Evidence-Gated SMOPD"][split]["action_accuracy"]
                for split in ("train", "dev", "diagnostic_ood")
            }
            full_vs_smopd_regret_delta = {
                split: metrics["PESCO-Full"][split]["mean_regret"]
                - metrics["Evidence-Gated SMOPD"][split]["mean_regret"]
                for split in ("train", "dev", "diagnostic_ood")
            }
            matched_budget_comparison["full_vs_smopd"] = {
                "action_accuracy_delta": full_vs_smopd_action_delta,
                "regret_delta": full_vs_smopd_regret_delta,
                "utility_per_cost_delta": {
                    split: metrics["PESCO-Full"][split]["utility_per_cost"]
                    - metrics["Evidence-Gated SMOPD"][split]["utility_per_cost"]
                    for split in ("train", "dev", "diagnostic_ood")
                },
                "full_improves_action_or_regret": any(
                    full_vs_smopd_action_delta[split] > 0.0
                    or full_vs_smopd_regret_delta[split] < 0.0
                    for split in ("train", "dev", "diagnostic_ood")
                ),
                "full_beats_smopd_on_all_reported_splits": bool(
                    all(value > 0.0 for value in full_vs_smopd_action_delta.values())
                    or all(value < 0.0 for value in full_vs_smopd_regret_delta.values())
                ),
                # SMOPD uses its own teacher update routine; its parameter/step
                # budget is intentionally recorded as unequal, so this remains a
                # descriptive diagnostic and cannot authorize a formal claim.
                "parameter_compute_equal": False,
                "formal_claim_blocked_by_compute": True,
            }
    else:
        matched_budget_comparison = {
            "status": "not_estimable_missing_methods",
            "missing_methods": [
                method for method, available in (
                    ("PESCO-Full", full_method_available),
                    ("PESCO-NoFlipLoss", no_flip_method_available),
                ) if not available
            ],
            "formal_claim_blocked": True,
        }
    experiment_e = {
        "schema_version": "pesco_experiment_e_v0.2",
        "experiment": "E_flip_ablation",
        "status": "completed_cpu_diagnostic",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "objective": "test whether confirmed cross-world preference reversal loss changes policy behavior",
        "methods": ["PESCO-BranchOnly", "PESCO-NoFlipLoss", "PESCO-Full", "Evidence-Gated SMOPD"],
        "confirmed_reversal_pairs": len(dataset.reversals),
        "same_frozen_dataset": True,
        "seed_utility_variability": {
            "array_count": int(dataset.provenance.get("seed_utility_array_count", 0)),
            "varying_array_count": int(dataset.provenance.get("seed_utility_varying_array_count", 0)),
            "constant_array_count": int(dataset.provenance.get("seed_utility_constant_array_count", 0)),
            "variability_observed": bool(dataset.provenance.get("seed_utility_variability_observed", False)),
            "formal_variance_claim_authorized": False,
            "interpretation": "utility spread is sparse because many protocol-transition terms are discrete; this suite remains diagnostic-only",
        },
        "flip_loss_is_differentiable_update": bool(full_flip_updates),
        "flip_gradient_norm": float(full_log.get("flip_gradient_norm", 0.0)),
        "flip_update_norm": float(full_log.get("flip_update_norm", 0.0)),
        "flip_updates_applied": int(full_log.get("flip_updates_applied", 0)),
        "flip_gradient_probe_count": int(full_log.get("flip_gradient_probe_count", 0)),
        "results": compact(["PESCO-BranchOnly", "PESCO-NoFlipLoss", "PESCO-Full", "Evidence-Gated SMOPD"]),
        "unseen_world_switch_diagnostics": unseen_world_diagnostics,
        "unseen_world_strategy_switching": unseen_world_diagnostics,
        "noise_condition_diagnostics": noise_diagnostics,
        "noise_conditions": noise_diagnostics,
        "sample_efficiency_diagnostics": sample_efficiency,
        "sample_efficiency": sample_efficiency,
        "ood_boundary": {
            "diagnostic_ood_split_is_internal_holdout": True,
            "formal_final_ood_access": False,
            "formal_ood_result_claim_allowed": False,
            "interpretation": "diagnostic_ood is a question-level mechanism holdout; it is not the locked formal OOD benchmark",
        },
        "ood_diagnostics": {
            method: {
                "diagnostic_ood": metrics[method].get("diagnostic_ood", {}),
                "unseen_world": unseen_world_diagnostics.get(method, {}).get("diagnostic_ood", {}),
                "formal_final_ood_access": False,
                "diagnostic_only": True,
            }
            for method in e_methods
            if method in metrics
        },
        "ablation_summary": {
            key: value for key, value in ablations.items()
            if any(name in key for name in ("NoFlipLoss", "Evidence-Gated SMOPD", "BranchOnly"))
        },
        "matched_budget_comparison": matched_budget_comparison,
        "gates": {
            "confirmed_pairs_present": len(dataset.reversals) > 0,
            "flip_loss_update_logged": bool(full_flip_updates),
            "genuine_training_updates": bool(genuine_updates),
            "formal_final_splits_unopened": True,
            "full_beats_no_flip_under_matched_optimizer_steps": bool(
                matched_budget_comparison.get("full_beats_no_flip_on_all_reported_splits", False)
            ),
            "formal_claim_blocked_by_compute_and_final_split_boundary": True,
            "smopd_comparison_reported": bool(
                matched_budget_comparison.get("full_vs_smopd")
            ),
            "smopd_formal_claim_blocked_by_compute": bool(
                matched_budget_comparison.get("full_vs_smopd", {}).get(
                    "formal_claim_blocked_by_compute", True
                )
            ),
        },
    }
    experiment_f = {
        "schema_version": "pesco_experiment_f_v0.1",
        "experiment": "F_open_ended_discovery",
        "status": "deferred_fixed_action_mvp_boundary",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "objective": "evaluate genuinely open-ended action/tool discovery after the fixed-action MVP",
        "fixed_action_mvp_discovery_utility": 0.0,
        "candidate_generation_executed": False,
        "reason": "The current benchmark has four registered actions; SWITCH is a registered action, not autonomous discovery. Open-ended candidate generation requires a separate tool sandbox, novelty certificate, and independent confirmation protocol.",
        "required_next_gates": [
            "held-out candidate-generation task family",
            "structure/execution/validity/confirmation/utility certificate",
            "discovery utility scored identically for all methods",
            "independent confirmation data and final split lock",
        ],
    }
    for experiment in (experiment_c, experiment_d, experiment_e, experiment_f):
        experiment.update({
            "protocol_version": protocol.protocol_version,
            "protocol_digest": protocol_digest,
            "benchmark_manifest_digest": benchmark.manifest(include_hidden=True)["manifest_digest"],
        })
    payload = {
        "schema_version": "pesco_tier1_differentiable_suite_v0.3",
        "implementation_status": "genuine_cpu_differentiable_reference_not_llm",
        "protocol_version": protocol.protocol_version,
        "protocol_digest": protocol_digest,
        "protocol_version_consistent": protocol.protocol_version == benchmark.protocol_version,
        "benchmark_manifest": benchmark.manifest(include_hidden=True),
        "dataset_provenance": dataset.provenance,
        "dataset_public_export": "dataset_public.json",
        "matched_compute": {
            "same_dataset": True,
            "same_candidate_actions": [action.value for action in ACTION_SET],
            "same_exploration_seeds": list(config.exploration_seeds),
            "same_confirmation_seeds": list(config.confirmation_seeds),
            "optimizer_step_cap": config.trainer.max_optimizer_steps,
            "same_hidden_final_splits_access": False,
        },
        "split_access_boundary": {
            "diagnostic_train_dev_ood": True,
            "formal_promotion_access": False,
            "formal_final_id_access": False,
            "formal_final_ood_access": False,
            "contamination_audit": "question_level_internal_diagnostic_only",
        },
        "training_logs": logs,
        "metrics": metrics,
        "target_action_agreement": {
            "matches": int(target_agreement),
            "groups": len(dataset.examples),
            "rate": target_agreement / max(1, len(dataset.examples)),
            "target_actions_audit_only": True,
        },
        "ablation_summary": ablations,
        "experiment_c": experiment_c,
        "experiment_d": experiment_d,
        "experiment_e": experiment_e,
        "experiment_f": experiment_f,
        "tier2_claim": False,
        "llm_claim": False,
    }
    (output / "suite.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "experiment_c_state_reward.json").write_text(json.dumps(experiment_c, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "experiment_d_branch_ablation.json").write_text(json.dumps(experiment_d, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "experiment_e_flip_ablation.json").write_text(json.dumps(experiment_e, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "experiment_f_discovery_boundary.json").write_text(json.dumps(experiment_f, indent=2, ensure_ascii=False), encoding="utf-8")

    # The suite is an executable C/D/E/F runner, so provenance belongs beside its
    # artifacts instead of being recoverable only through a later metadata pass.
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        repo_root / "scripts/run_tier1_differentiable_suite.py",
        repo_root / "research_strategy_optimization/algorithms/differentiable_strategy.py",
        repo_root / "research_strategy_optimization/algorithms/branch_rollout.py",
        repo_root / "research_strategy_optimization/environments/tier0_simulator.py",
        repo_root / "research_strategy_optimization/environments/tier1_benchmark.py",
        repo_root / "research_strategy_optimization/schemas.py",
        repo_root / "research_strategy_optimization/utils/run_manifest.py",
    ]
    data_paths = [
        output / "dataset.json",
        output / "dataset_public.json",
        output / "suite.json",
        output / "experiment_c_state_reward.json",
        output / "experiment_d_branch_ablation.json",
        output / "experiment_e_flip_ablation.json",
        output / "experiment_f_discovery_boundary.json",
        *sorted(output.glob("policy_*.pt")),
    ]
    manifest = build_run_manifest(
        experiment="C_D_E_F_differentiable_suite",
        repo_root=repo_root,
        # The library API may be called directly; the CLI wrapper supplies the
        # exact argv through its post-hoc companion when needed.
        command=None,
        runner_paths=source_paths,
        data_paths=data_paths,
        seeds={
            "training": [int(config.trainer.seed)],
            "inference": [int(config.trainer.seed)],
            "exploration": list(config.exploration_seeds),
            "confirmation": list(config.confirmation_seeds),
        },
        checkpoint=None,
        status="completed",
        diagnostics={
            "capture_mode": "in_run",
            "artifact_status": payload["implementation_status"],
            "protocol_version": protocol.protocol_version,
            "protocol_digest": protocol_digest,
            "experiments": ["C", "D", "E", "F"],
            "methods": list(config.methods),
            "optimizer_step_cap": config.trainer.max_optimizer_steps,
            "branch_groups": dataset.provenance.get("question_world_group_count"),
            "action_level_branch_rows": dataset.provenance.get("action_level_row_count"),
            "seed_level_executions": dataset.provenance.get("seed_level_observation_count"),
            "reversal_count": dataset.provenance.get("reversal_count"),
            "formal_final_splits_opened": False,
            "tier2_claim": payload["tier2_claim"],
            "llm_claim": payload["llm_claim"],
        },
    )
    write_run_manifest(output / "run_manifest.json", manifest)
    return payload


__all__ = [
    "DEFAULT_METHODS",
    "Tier1SuiteConfig",
    "benchmark_action_utility",
    "collect_tier1_v03_dataset",
    "evaluate_differentiable_policy",
    "is_invalid_local_optimization",
    "run_tier1_differentiable_suite",
]

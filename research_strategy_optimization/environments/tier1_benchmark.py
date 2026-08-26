"""Frozen Tier-1 v0.3 multi-question benchmark.

This module is an evaluator-side benchmark builder, not a policy oracle.  It creates
12 independent question instances (three variants in each of four mechanism
families), four hidden worlds per question, and exposes a factory for the concrete
NumPy Tier-1 environment.  The target-action table is retained only for external
audit/regret accounting; policies receive ordinary :class:`Observation` objects with
the public question family, but never receive world kind or target action.

The four families intentionally make the same evidence state require different
scientific actions:

* group leakage: Invalid -> REPAIR and Refuted -> SWITCH;
* causal confounding: Invalid -> REPAIR, but Insufficient -> SWITCH and Refuted ->
  CONTINUE (a reliable negative result is the goal);
* low-sample variance: Invalid/Insufficient -> SAMPLE;
* subgroup metric mismatch: Invalid/Insufficient/Refuted -> SWITCH to the registered
  subgroup estimator.

The table is not used to generate branch rewards.  ``tier1_v03_runner.py`` computes
action utility from the trusted verifier, observed transition, confirmation, and
pre-registered family protocol, then compares the empirical winner with this table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from ..schemas import EvidenceState, Protocol, ResearchAction, Verdict, ExperimentOutput, WorldSpec
from .tier1_tabular_env import Tier1TabularEnvironment


FAMILY_GROUP_LEAKAGE = "group_leakage"
FAMILY_CAUSAL_CONFOUNDING = "causal_confounding"
FAMILY_LOW_SAMPLE_VARIANCE = "low_sample_variance"
FAMILY_SUBGROUP_METRIC = "subgroup_metric_mismatch"
MECHANISM_FAMILIES = (
    FAMILY_GROUP_LEAKAGE,
    FAMILY_CAUSAL_CONFOUNDING,
    FAMILY_LOW_SAMPLE_VARIANCE,
    FAMILY_SUBGROUP_METRIC,
)


def _public_state(observation: object | None, delta_min: float) -> EvidenceState | None:
    """Dependency-free public evidence classifier for evaluator utilities."""

    if observation is None:
        return None
    signals = set(getattr(observation, "validity_signals", ()))
    if signals.intersection({
        "split_overlap_diagnostic", "metric_scope_mismatch", "variance_estimator_unstable",
        "treatment_confounder_dependence", "protocol_invalid_diagnostic",
    }):
        return EvidenceState.INVALID
    low = float(getattr(observation, "ci_low", 0.0))
    high = float(getattr(observation, "ci_high", 0.0))
    if high - low > 0.30 or "sample_count_below_precision_target" in signals:
        return EvidenceState.INSUFFICIENT
    if low > float(delta_min):
        return EvidenceState.SUPPORTED
    if high < float(delta_min):
        return EvidenceState.REFUTED
    return EvidenceState.INSUFFICIENT
WORLD_KINDS = ("supported", "refuted", "insufficient", "invalid")


@dataclass(frozen=True)
class Tier1QuestionSpec:
    """One independent question instance in the frozen benchmark."""

    question_id: str
    family: str
    variant: int
    split: str
    seed_offset: int
    description: str
    worlds: Tuple[WorldSpec, ...]
    target_actions: Mapping[str, ResearchAction]

    def __post_init__(self) -> None:
        if self.family not in MECHANISM_FAMILIES:
            raise ValueError(f"unknown Tier-1 mechanism family: {self.family}")
        if int(self.variant) not in (1, 2, 3):
            raise ValueError("variant must be one of 1, 2, 3")
        if self.split not in {"train", "dev", "diagnostic_ood"}:
            raise ValueError("split must be train, dev, or diagnostic_ood")
        if len(self.worlds) != 4:
            raise ValueError("each Tier-1 question must contain four hidden worlds")
        kinds = tuple(world.kind for world in self.worlds)
        if kinds != WORLD_KINDS:
            raise ValueError(f"world order must be {WORLD_KINDS}, got {kinds}")
        expected_ids = tuple(f"{self.question_id}__{kind}" for kind in WORLD_KINDS)
        if tuple(world.world_id for world in self.worlds) != expected_ids:
            raise ValueError("world IDs must be question-qualified and ordered by kind")
        if set(self.target_actions) != set(expected_ids):
            raise ValueError("target_actions must cover all four hidden worlds")
        if any(not isinstance(action, ResearchAction) for action in self.target_actions.values()):
            raise TypeError("target_actions must contain ResearchAction values")

    @property
    def world_map(self) -> Mapping[str, WorldSpec]:
        return MappingProxyType({world.world_id: world for world in self.worlds})

    @property
    def policy_question_id(self) -> str:
        """Neutral question token used in policy-visible observations.

        Evaluator manifests retain the descriptive ID, but exposing strings such as
        ``group_leakage`` or ``causal_confounding`` through ``Observation.question_id``
        would leak the mechanism family and make the benchmark unfair.
        """

        # All mechanism variants intentionally share one public question token.
        # The hard case is defined at the evidence-state/registered-protocol level:
        # the same evidence state can have different evaluator-optimal actions.
        # Descriptive IDs stay in audit manifests only.
        return "tier1_public_question"

    def target_action(self, world_id: str) -> ResearchAction:
        return self.target_actions[world_id]

    def to_dict(self, include_hidden: bool = True) -> dict:
        data = {
            "question_id": self.question_id,
            "policy_question_id": self.policy_question_id,
            "family": self.family,
            "variant": int(self.variant),
            "split": self.split,
            "seed_offset": int(self.seed_offset),
            "description": self.description,
        }
        if include_hidden:
            data["world_ids"] = [world.world_id for world in self.worlds]
            data["target_actions"] = {
                world_id: action.value for world_id, action in self.target_actions.items()
            }
            data["worlds"] = [asdict(world) for world in self.worlds]
        else:
            data["world_count"] = len(self.worlds)
        return data


@dataclass(frozen=True)
class Tier1Benchmark:
    """Immutable collection of the 12 benchmark questions."""

    questions: Tuple[Tier1QuestionSpec, ...]
    # Runtime decision rules stay bound to the frozen repository protocol.  The
    # benchmark's structural revision is carried separately by ``schema_version``.
    protocol_version: str = "pesco_v0_2"
    schema_version: str = "pesco_tier1_benchmark_v0.3"

    def __post_init__(self) -> None:
        if len(self.questions) != 12:
            raise ValueError("Tier1 v0.3 requires exactly 12 independent questions")
        question_ids = tuple(question.question_id for question in self.questions)
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question IDs must be unique")
        counts = {family: sum(q.family == family for q in self.questions) for family in MECHANISM_FAMILIES}
        if counts != {family: 3 for family in MECHANISM_FAMILIES}:
            raise ValueError(f"each family requires exactly three variants, got {counts}")

    @property
    def question_map(self) -> Mapping[str, Tier1QuestionSpec]:
        return MappingProxyType({question.question_id: question for question in self.questions})

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    def question(self, question_id: str) -> Tier1QuestionSpec:
        return self.question_map[question_id]

    def world(self, world_id: str) -> WorldSpec:
        for world in self.worlds:
            if world.world_id == world_id:
                return world
        raise KeyError(world_id)

    def target_action(self, question_id: str, world_id: str) -> ResearchAction:
        return self.question(question_id).target_action(world_id)

    def make_environment(
        self,
        question_id: str,
        protocol: Optional[Protocol] = None,
        budget: Optional[int] = None,
    ) -> Tier1TabularEnvironment:
        question = self.question(question_id)
        return Tier1TabularEnvironment(
            worlds=question.worlds,
            protocol=protocol,
            budget=budget,
        )

    def manifest(self, include_hidden: bool = True) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "question_count": len(self.questions),
            "world_count": len(self.worlds),
            "mechanism_family_count": len(MECHANISM_FAMILIES),
            "mechanism_families": list(MECHANISM_FAMILIES),
            "questions": [question.to_dict(include_hidden=include_hidden) for question in self.questions],
            "counts_by_split": {
                split: sum(question.split == split for question in self.questions)
                for split in ("train", "dev", "diagnostic_ood")
            },
        }
        # Digest the evaluator-side manifest so a runner can bind artifacts to the
        # exact frozen benchmark without exposing the digest as policy input.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        payload["manifest_digest"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
        return payload


def tier1_scientific_utility(
    question: Tier1QuestionSpec,
    world: WorldSpec,
    action: ResearchAction,
    output: ExperimentOutput,
    verdict: Verdict,
    protocol: Protocol,
    initial_observation: object | None = None,
) -> float:
    """Score one evaluator-side branch without consulting the target-action table.

    The score combines evaluator-visible validity, evidence transition, protocol
    signals, independent confirmation, and action cost.  It deliberately does not
    read ``world.kind`` or ``output.latent_effect``; those fields remain available
    only to the post-hoc calibration/audit layer.
    """

    cost = float(output.execution_cost)
    signals = set(output.validity_signals)
    # Invalid branches cannot win merely because their surface estimate is large.
    # Every positive term below is tied to an evaluator-observable protocol transition
    # or result, never to ``world.kind`` or the target-action table.
    if not verdict.validity_pass:
        return float(-0.30 - 0.03 * cost)

    score = 0.25
    before = _public_state(initial_observation, protocol.delta_min)
    after = verdict.evidence_state
    if verdict.independent_confirmation_passed:
        score += 0.20
    if before is EvidenceState.INVALID and "split_protocol_updated" in signals:
        score += 0.45
    if "group_held_out_split" in signals and "split_overlap_diagnostic" not in signals:
        score += 0.80
    mechanism_transition = any(
        marker in signal
        for signal in signals
        for marker in ("adjusted", "controlled", "subgroup_metric_estimator")
    )
    if mechanism_transition and not (
        action is ResearchAction.SWITCH
        and output.method == "method_b"
        and output.effect_estimate <= max(0.08, protocol.delta_min)
    ):
        score += 0.65
    # A method switch earns value only when the executed alternative produces a
    # materially positive, valid observed estimate.  This is a measured transition
    # criterion, not a family/state→action lookup, and creates genuine preference
    # reversals when the alternative estimand differs across worlds.
    if action is ResearchAction.SWITCH and output.method == "method_b" and output.effect_estimate > max(0.08, protocol.delta_min):
        score += 0.55
    elif action is ResearchAction.SWITCH and output.method == "method_b":
        # Switching to an alternative that produces no practically meaningful
        # observed effect incurs its execution cost without an evidence gain.
        score -= 0.20
    if output.sample_size >= 60 and "sample_count_below_precision_target" not in signals:
        # This is a property of the executed data protocol, not a reward for the
        # action name SAMPLE itself.
        score += 0.35
    if before is EvidenceState.INSUFFICIENT and after is not EvidenceState.INSUFFICIENT:
        score += 0.30
    if action is ResearchAction.REPLICATE:
        score += 0.10
    return float(score - 0.03 * cost)


def _world(
    question_id: str,
    family: str,
    kind: str,
    *,
    true_a: float,
    true_b: float,
    noise: float,
    samples: int,
    seed_offset: int,
    leakage: bool = False,
    confounding: bool = False,
    metric_mismatch: bool = False,
    protocol_invalid: bool = False,
) -> WorldSpec:
    return WorldSpec(
        world_id=f"{question_id}__{kind}",
        kind=kind,
        true_effect_a=true_a,
        true_effect_b=true_b,
        noise_scale=noise,
        initial_samples=samples,
        leakage=leakage,
        confounding=confounding,
        seed_offset=seed_offset,
        question_family=family,
        metric_mismatch=metric_mismatch,
        protocol_invalid=protocol_invalid,
    )


def _question(
    family: str,
    variant: int,
    split: str,
    seed_offset: int,
    description: str,
    params: Mapping[str, Mapping[str, float | int | bool]],
    target_by_kind: Mapping[str, ResearchAction],
) -> Tier1QuestionSpec:
    question_id = f"t1_{family}_{variant:02d}"
    worlds = tuple(
        _world(
            question_id,
            family,
            kind,
            true_a=float(params[kind]["true_a"]),
            true_b=float(params[kind]["true_b"]),
            noise=float(params[kind]["noise"]),
            samples=int(params[kind]["samples"]),
            seed_offset=seed_offset + int(params[kind].get("seed_delta", 0)),
            leakage=bool(params[kind].get("leakage", False)),
            confounding=bool(params[kind].get("confounding", False)),
            metric_mismatch=bool(params[kind].get("metric_mismatch", False)),
            protocol_invalid=bool(params[kind].get("protocol_invalid", False)),
        )
        for kind in WORLD_KINDS
    )
    target_actions = MappingProxyType(
        {
            world.world_id: target_by_kind[world.kind]
            for world in worlds
        }
    )
    return Tier1QuestionSpec(
        question_id=question_id,
        family=family,
        variant=variant,
        split=split,
        seed_offset=seed_offset,
        description=description,
        worlds=worlds,
        target_actions=target_actions,
    )


def _family_question(
    family: str,
    variant: int,
    split: str,
    seed_offset: int,
) -> Tier1QuestionSpec:
    """Build one parameterized question variant for a mechanism family."""

    # Small variant shifts create genuinely distinct data-generating instances while
    # preserving the family-level contrast and four evidence states.
    v = int(variant)
    if family == FAMILY_GROUP_LEAKAGE:
        scale = (1.0, 1.08, 0.90)[v - 1]
        noise = (0.035, 0.040, 0.032)[v - 1]
        params = {
            "supported": {"true_a": 0.18 * scale, "true_b": 0.025, "noise": noise, "samples": 1000, "seed_delta": 1},
            "refuted": {"true_a": -0.08, "true_b": 0.16 * scale, "noise": 0.035, "samples": 1000, "seed_delta": 2},
            "insufficient": {"true_a": 0.08 * scale, "true_b": 0.02, "noise": 0.42 + 0.01 * v, "samples": 18, "seed_delta": 3},
            "invalid": {"true_a": 0.0, "true_b": 0.14 * scale, "noise": noise, "samples": 1000, "leakage": True, "protocol_invalid": True, "seed_delta": 4},
        }
        targets = {
            "supported": ResearchAction.CONTINUE,
            "refuted": ResearchAction.SWITCH,
            "insufficient": ResearchAction.SAMPLE,
            "invalid": ResearchAction.REPAIR,
        }
        description = "Group-held-out evaluation with row-overlap leakage in the invalid world."
    elif family == FAMILY_CAUSAL_CONFOUNDING:
        scale = (1.0, 0.92, 1.12)[v - 1]
        noise = (0.035, 0.040, 0.032)[v - 1]
        params = {
            "supported": {"true_a": 0.18 * scale, "true_b": 0.025, "noise": noise, "samples": 1000, "seed_delta": 11},
            # Both registered estimands are negative: preserve a reliable negative
            # result rather than paying to switch methods.
            "refuted": {
                "true_a": -0.08,
                # Variant 3 keeps the public method-A evidence fixed but makes the
                # registered alternative estimand materially positive.
                "true_b": 0.18 if v == 3 else -0.08,
                "noise": 0.035,
                "samples": 1000,
                "seed_delta": 12,
            },
            # A randomized alternative (method B) is informative when observational
            # method A remains underpowered; hence Insufficient -> SWITCH.
            "insufficient": {"true_a": 0.015, "true_b": 0.18 * scale, "noise": 0.34 + 0.01 * v, "samples": 18, "seed_delta": 13},
            "invalid": {"true_a": 0.0, "true_b": 0.06, "noise": noise, "samples": 1000, "confounding": True, "protocol_invalid": True, "seed_delta": 14},
        }
        targets = {
            "supported": ResearchAction.CONTINUE,
            "refuted": ResearchAction.SWITCH if v == 3 else ResearchAction.CONTINUE,
            "insufficient": ResearchAction.SWITCH,
            "invalid": ResearchAction.REPAIR,
        }
        description = "Observational treatment assignment is confounded; adjusted OLS repair is required for validity."
    elif family == FAMILY_LOW_SAMPLE_VARIANCE:
        scale = (1.0, 1.15, 0.85)[v - 1]
        params = {
            "supported": {"true_a": 0.18 * scale, "true_b": 0.025, "noise": 0.035 + 0.005 * v, "samples": 1000, "seed_delta": 21},
            "refuted": {"true_a": -0.08, "true_b": -0.08, "noise": 0.035, "samples": 1000, "seed_delta": 22},
            "insufficient": {"true_a": 0.14 * scale, "true_b": 0.02, "noise": 0.42 + 0.01 * v, "samples": 18, "seed_delta": 23},
            # The effect is real but the measurement protocol itself is unstable at
            # the initial sample size.  SAMPLE resolves both precision and validity.
            "invalid": {"true_a": 0.14 * scale, "true_b": 0.02, "noise": 0.42 + 0.01 * v, "samples": 18, "protocol_invalid": True, "seed_delta": 24},
        }
        targets = {
            "supported": ResearchAction.CONTINUE,
            "refuted": ResearchAction.CONTINUE,
            "insufficient": ResearchAction.SAMPLE,
            "invalid": ResearchAction.SAMPLE,
        }
        description = "High-variance measurement protocol where added samples stabilize an otherwise invalid estimate."
    elif family == FAMILY_SUBGROUP_METRIC:
        scale = (1.0, 0.88, 1.15)[v - 1]
        params = {
            "supported": {"true_a": 0.18 * scale, "true_b": 0.03, "noise": 0.035 + 0.005 * v, "samples": 1000, "seed_delta": 31},
            # Aggregate metric refutes A while the registered subgroup estimator B is
            # positive, making SWITCH scientifically preferable.
            "refuted": {"true_a": -0.08, "true_b": 0.18 * scale, "noise": 0.035, "samples": 1000, "seed_delta": 32},
            "insufficient": {"true_a": 0.015, "true_b": 0.18 * scale, "noise": 0.34 + 0.01 * v, "samples": 18, "seed_delta": 33},
            # The aggregate metric is invalid for the subgroup estimand.  Switching
            # to the subgroup-aware method removes the metric-scope mismatch.
            "invalid": {"true_a": 0.0, "true_b": 0.18 * scale, "noise": 0.035 + 0.005 * v, "samples": 1000, "metric_mismatch": True, "seed_delta": 34},
        }
        targets = {
            "supported": ResearchAction.CONTINUE,
            "refuted": ResearchAction.SWITCH,
            "insufficient": ResearchAction.SWITCH,
            "invalid": ResearchAction.SWITCH,
        }
        description = "Overall metric is mismatched to a subgroup estimand; the alternative estimator is subgroup-aware."
    else:  # pragma: no cover - guarded by constants and question validation
        raise ValueError(family)
    return _question(family, v, split, seed_offset, description, params, targets)


def build_tier1_v03_questions() -> Tuple[Tier1QuestionSpec, ...]:
    questions = []
    families = MECHANISM_FAMILIES
    for family_index, family in enumerate(families):
        for variant in (1, 2, 3):
            if variant == 3:
                # Hold two variants for development and two for diagnostic OOD so the
                # 12 IDs are not merely cosmetic aliases of one split.
                split = "dev" if family_index < 2 else "diagnostic_ood"
            else:
                split = "train"
            questions.append(
                _family_question(
                    family,
                    variant,
                    split,
                    seed_offset=10_000 * (family_index + 1) + 1_000 * variant,
                )
            )
    return tuple(questions)


def build_tier1_v03_benchmark() -> Tier1Benchmark:
    return Tier1Benchmark(build_tier1_v03_questions())


__all__ = [
    "FAMILY_GROUP_LEAKAGE",
    "FAMILY_CAUSAL_CONFOUNDING",
    "FAMILY_LOW_SAMPLE_VARIANCE",
    "FAMILY_SUBGROUP_METRIC",
    "MECHANISM_FAMILIES",
    "Tier1QuestionSpec",
    "Tier1Benchmark",
    "build_tier1_v03_questions",
    "build_tier1_v03_benchmark",
    "tier1_scientific_utility",
]

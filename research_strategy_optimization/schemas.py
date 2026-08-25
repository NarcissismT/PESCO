"""Shared, JSON-serialisable schemas for the PESCO prototype.

Only :class:`Observation` is exposed to a policy.  Hidden world parameters, verifier
labels and confirmation-set information live in the environment/verifier records and
are deliberately absent from ``Observation.to_dict``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class EvidenceState(str, Enum):
    INVALID = "invalid"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient"


class ResearchAction(str, Enum):
    CONTINUE = "continue_current_method"
    REPLICATE = "replicate"
    SAMPLE = "add_samples_or_seeds"
    REPAIR = "repair_data_split"
    METRIC = "change_or_add_metric"
    REVISE = "revise_hypothesis"
    SWITCH = "switch_to_alternative_method"
    STOP = "stop_and_report"

    @classmethod
    def mvp_actions(cls) -> Tuple["ResearchAction", ...]:
        return (cls.CONTINUE, cls.SAMPLE, cls.REPAIR, cls.SWITCH)

    @classmethod
    def all_actions(cls) -> Tuple["ResearchAction", ...]:
        return tuple(cls)


@dataclass(frozen=True)
class Protocol:
    """Pre-registered decision rules.

    ``delta_min`` is the minimum practically meaningful positive effect.  A positive
    claim is refuted when a valid confidence interval lies below that threshold; an
    interval crossing the threshold is *insufficient*, never automatically refuted.
    """

    protocol_version: str = "pesco_v0_1"
    delta_min: float = 0.02
    confidence_level: float = 0.95
    invalid_precedence: bool = True
    independent_confirmation_required: bool = True
    confirmation_seeds: Tuple[int, ...] = (103, 107, 109, 113)
    exploration_seeds: Tuple[int, ...] = (17, 29, 41, 53)
    flip_margin: float = 0.05
    discovery_margin: float = 0.05
    probability_clip: float = 1e-3
    max_budget: int = 6

    def __post_init__(self) -> None:
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if not math.isfinite(float(self.delta_min)) or float(self.delta_min) < 0.0:
            raise ValueError("delta_min must be finite and non-negative")
        if not math.isfinite(float(self.flip_margin)) or float(self.flip_margin) < 0.0:
            raise ValueError("flip_margin must be finite and non-negative")
        if not math.isfinite(float(self.discovery_margin)) or float(self.discovery_margin) < 0.0:
            raise ValueError("discovery_margin must be finite and non-negative")
        if not 0.0 < float(self.probability_clip) < 0.5:
            raise ValueError("probability_clip must lie in (0, .5)")
        if isinstance(self.max_budget, bool) or int(self.max_budget) != self.max_budget or int(self.max_budget) < 0:
            raise ValueError("max_budget must be a non-negative integer")
        exploration = tuple(int(seed) for seed in self.exploration_seeds)
        confirmation = tuple(int(seed) for seed in self.confirmation_seeds)
        if not exploration or not confirmation:
            raise ValueError("exploration and confirmation seed sets must be non-empty")
        if len(set(exploration)) != len(exploration) or len(set(confirmation)) != len(confirmation):
            raise ValueError("exploration and confirmation seeds must be unique")
        if set(exploration) & set(confirmation):
            raise ValueError("exploration and confirmation seeds must be disjoint")


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    question_id: str
    claim: str
    estimand: str
    delta_min: float
    protocol_version: str
    timestamp: str
    registered_before_confirmation: bool = True


@dataclass(frozen=True)
class WorldSpec:
    """Latent mechanism used by the synthetic environment.

    This object is never handed to a policy.  ``world_id`` is useful for grouping and
    auditing only; visible observations contain no identifier derived from it.
    """

    world_id: str
    kind: str
    true_effect_a: float
    true_effect_b: float
    noise_scale: float
    initial_samples: int
    leakage: bool = False
    confounding: bool = False
    seed_offset: int = 0
    question_family: str = "group_generalization"

    def __post_init__(self) -> None:
        if not str(self.world_id):
            raise ValueError("world_id must be non-empty")
        if self.kind not in {"supported", "refuted", "insufficient", "invalid"}:
            raise ValueError("world kind must be supported, refuted, insufficient, or invalid")
        numeric = (self.true_effect_a, self.true_effect_b, self.noise_scale)
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("world effect/noise parameters must be finite")
        if isinstance(self.initial_samples, bool) or int(self.initial_samples) != self.initial_samples or int(self.initial_samples) <= 0:
            raise ValueError("initial_samples must be a positive integer")


@dataclass(frozen=True)
class Observation:
    """Policy-visible state.  Keep this schema free of hidden labels."""

    question_id: str
    turn: int
    current_method: str
    effect_estimate: float
    ci_low: float
    ci_high: float
    sample_size: int
    seed_count: int
    remaining_budget: int
    metric_name: str = "group_held_out_accuracy_delta"
    validity_signals: Tuple[str, ...] = ()
    history_summary: Tuple[str, ...] = ()
    hypothesis_probability: float = 0.5

    def __post_init__(self) -> None:
        values = (self.effect_estimate, self.ci_low, self.ci_high, self.hypothesis_probability)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("observation numeric fields must be finite")
        if float(self.ci_low) > float(self.ci_high):
            raise ValueError("observation confidence interval must be ordered")
        if not 0.0 <= float(self.hypothesis_probability) <= 1.0:
            raise ValueError("hypothesis_probability must lie in [0, 1]")
        counters = (self.turn, self.sample_size, self.seed_count, self.remaining_budget)
        if any(isinstance(raw, bool) or int(raw) != raw or int(raw) < 0 for raw in counters):
            raise ValueError("observation counters must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Return the exact public observation payload.

        The method is intentionally explicit rather than using ``asdict`` so a future
        hidden field cannot accidentally leak into the policy input.
        """

        return {
            "question_id": self.question_id,
            "turn": self.turn,
            "current_method": self.current_method,
            "effect_estimate": float(self.effect_estimate),
            "confidence_interval": [float(self.ci_low), float(self.ci_high)],
            "sample_size": int(self.sample_size),
            "seed_count": int(self.seed_count),
            "remaining_budget": int(self.remaining_budget),
            "metric_name": self.metric_name,
            "validity_signals": list(self.validity_signals),
            "history_summary": list(self.history_summary),
            "hypothesis_probability": float(self.hypothesis_probability),
        }


@dataclass(frozen=True)
class ExperimentOutput:
    """Raw result generated by an environment action.

    ``hidden_world_id`` and ``latent_effect`` are verifier-only fields and must never be
    serialised into a public observation.
    """

    action: str
    method: str
    effect_estimate: float
    ci_low: float
    ci_high: float
    sample_size: int
    seed_count: int
    execution_cost: float
    dataset_hash: str
    code_hash: str
    split_hash: str
    evaluator_hash: str
    seeds: Tuple[int, ...]
    validity_signals: Tuple[str, ...] = ()
    hidden_world_id: str = ""
    latent_effect: float = 0.0
    leakage: bool = False
    confounding: bool = False
    confirmation: bool = False

    def __post_init__(self) -> None:
        values = (self.effect_estimate, self.ci_low, self.ci_high, self.execution_cost, self.latent_effect)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("experiment output numeric fields must be finite")
        if float(self.ci_low) > float(self.ci_high):
            raise ValueError("experiment output confidence interval must be ordered")
        if float(self.execution_cost) < 0.0:
            raise ValueError("execution_cost must be non-negative")
        if int(self.sample_size) < 0 or int(self.seed_count) < 0:
            raise ValueError("sample_size and seed_count must be non-negative")
        if len(tuple(self.seeds)) != int(self.seed_count):
            raise ValueError("seed_count must match the seeds tuple length")

    def public_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "method": self.method,
            "effect_estimate": float(self.effect_estimate),
            "confidence_interval": [float(self.ci_low), float(self.ci_high)],
            "sample_size": int(self.sample_size),
            "seed_count": int(self.seed_count),
            "execution_cost": float(self.execution_cost),
            "validity_signals": list(self.validity_signals),
        }


@dataclass(frozen=True)
class Verdict:
    validity_pass: bool
    evidence_state: EvidenceState
    effect_estimate: float
    confidence_interval: Tuple[float, float]
    independent_confirmation_performed: bool
    independent_confirmation_passed: bool
    scientific_claim_consistency: bool
    audit_signature: str
    execution_cost: float
    confirmation_seeds: Tuple[int, ...] = ()
    invalid_reasons: Tuple[str, ...] = ()
    method_family: str = "method_a"
    autonomous_candidate: bool = False
    discovered_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity_pass": self.validity_pass,
            "evidence_state": self.evidence_state.value,
            "effect_estimate": self.effect_estimate,
            "confidence_interval": list(self.confidence_interval),
            "independent_confirmation": {
                "performed": self.independent_confirmation_performed,
                "passed": self.independent_confirmation_passed,
                "confirmation_seeds": list(self.confirmation_seeds),
            },
            "scientific_claim_consistency": self.scientific_claim_consistency,
            "audit_signature": self.audit_signature,
            "budget_cost": {"execution_units": self.execution_cost},
            "invalid_reasons": list(self.invalid_reasons),
            "method_family": self.method_family,
            "autonomous_candidate": self.autonomous_candidate,
            "discovered_gain": self.discovered_gain,
        }


@dataclass
class Trajectory:
    question_id: str
    world_id: str
    initial_observation: Observation
    final_observation: Observation
    outputs: List[ExperimentOutput] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)
    belief_before: float = 0.5
    belief_after: float = 0.5
    total_cost: float = 0.0
    branch_id: str = ""
    proposal_source: str = "policy_on_policy"

    def final_verdict(self) -> Optional[Verdict]:
        return self.verdicts[-1] if self.verdicts else None

    def to_dict(self, include_hidden: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "question_id": self.question_id,
            "world_id": self.world_id if include_hidden else "hidden_from_agent",
            "branch_id": self.branch_id,
            "initial_observation": self.initial_observation.to_dict(),
            "final_observation": self.final_observation.to_dict(),
            "outputs": [o.public_dict() for o in self.outputs],
            "belief_before": self.belief_before,
            "belief_after": self.belief_after,
            "total_cost": self.total_cost,
            "proposal_source": self.proposal_source,
        }
        if include_hidden:
            data["verdicts"] = [v.to_dict() for v in self.verdicts]
            data["hidden_outputs"] = [asdict(o) for o in self.outputs]
        return data

    def policy_view(self) -> Dict[str, Any]:
        """Strict whitelist used when serialising context for a policy."""

        return self.to_dict(include_hidden=False)


@dataclass
class BranchRecord:
    option: ResearchAction
    trajectory: Trajectory
    utility: float
    components: Dict[str, float]
    advantage: float = 0.0
    estimated_value: float = 0.0
    paired_seed_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option": self.option.value,
            "trajectory": self.trajectory.to_dict(include_hidden=True),
            "utility": self.utility,
            "components": self.components,
            "advantage": self.advantage,
            "estimated_value": self.estimated_value,
            "paired_seed_values": self.paired_seed_values,
        }

    def policy_view(self) -> Dict[str, Any]:
        """Whitelist a branch for policy-side contexts.

        Trusted verdicts and hidden trajectory fields are intentionally absent.  The
        scalar utility/advantage can be supplied by an evaluator as a training target,
        but the policy never receives the latent world or verifier object.
        """

        return {
            "option": self.option.value,
            "trajectory": self.trajectory.policy_view(),
            "utility": float(self.utility),
            "advantage": float(self.advantage),
            "estimated_value": float(self.estimated_value),
        }


@dataclass
class ReversalPair:
    question_id: str
    world_a: str
    world_b: str
    action_left: ResearchAction
    action_right: ResearchAction
    delta_a: float
    delta_b: float
    lcb_a: float
    ucb_b: float
    confirmed: bool
    double_difference: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "paired_worlds": [self.world_a, self.world_b],
            "candidate_options": [self.action_left.value, self.action_right.value],
            "estimated_values": {
                self.world_a: {self.action_left.value: self.delta_a, self.action_right.value: 0.0},
                self.world_b: {self.action_left.value: self.delta_b, self.action_right.value: 0.0},
            },
            "paired_confidence": {
                "delta_a_lcb": self.lcb_a,
                "delta_b_ucb": self.ucb_b,
                "confirmed_reversal": self.confirmed,
            },
            "double_difference": self.double_difference,
            "world_ids_visible_to_policy": False,
        }


@dataclass
class DiscoveryCertificate:
    method_family: str
    proposed_without_method_hint: bool
    structurally_distinct: bool
    actually_executed: bool
    valid_experiment: bool
    independently_confirmed: bool
    lower_confidence_gain: float
    autonomous: bool
    certificate_pass: bool
    reason: str = ""
    proposal_source: str = "policy_on_policy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditEvent:
    event_type: str
    event_index: int
    question_id: str
    hypothesis_id: str
    branch_id: str
    previous_event_hash: str
    experiment_code_hash: str
    dataset_hash: str
    visible_output_hash: str
    trusted_verdict_hash: str
    budget_before: int
    budget_after: int
    confirmation_data_accessed: bool = False
    event_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return as_jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return as_jsonable(asdict(value))
    return value

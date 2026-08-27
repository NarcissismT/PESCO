"""Executable Tier-1 v0.4 diagnostic benchmark (64 questions, eight families).

The original v0.3 benchmark is intentionally frozen.  This module adds the larger
diagnostic dataset requested by the review without changing v0.3 artifact semantics:
eight mechanism/protocol families, eight variants per family, eight exploration
seeds, independent per-replicate confirmation, same-question paired worlds, two
input tracks, and evaluator-only multi-step trajectories.

The benchmark remains a CPU/NumPy mechanism diagnostic.  Formal promotion is still
closed until the recorded pair/cluster gates pass; this module never turns a hidden
target-action table into a policy feature or a training label.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..algorithms.paired_world_sampler import identify_confirmed_reversal
from ..environments.tier0_simulator import TrustedVerifier
from ..environments.tier1_benchmark import (
    TIER1_REWARD_COMPONENT_NAMES,
    tier1_scientific_utility_components,
)
from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..schemas import EvidenceState, ExperimentOutput, Observation, Protocol, ResearchAction, Verdict, WorldSpec
ACTION_SET = ResearchAction.mvp_actions()


WORLD_KINDS: Tuple[str, ...] = ("supported", "refuted", "insufficient", "invalid")
V04_EXTENDED_FAMILIES: Tuple[str, ...] = (
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
    "confounding_underpower",
    "leakage_metric_mismatch",
    "protocol_drift",
    "replication_instability",
)
TRACK_ORACLE_STATE = "oracle_state"
TRACK_RAW_EVIDENCE = "raw_evidence"
V04_EXTENDED_SCHEMA = "pesco_tier1_benchmark_v0.4_extended"
V04_FORMAL_SCHEMA = "pesco_tier1_benchmark_v0.4_formal_final"
V04_EXTENDED_EXPLORATION_SEEDS: Tuple[int, ...] = (17, 29, 41, 53, 67, 71, 83, 97)
V04_EXTENDED_CONFIRMATION_SEEDS: Tuple[int, ...] = (103, 107, 109, 113, 127, 131, 137, 139)
# Compatibility aliases for early v0.4 scripts.  The canonical names above make
# the split semantics explicit; these aliases do not change the benchmark.
EXTENDED_MECHANISM_FAMILIES = V04_EXTENDED_FAMILIES
EXTENDED_SPLITS = ("train", "dev", "diagnostic_ood")
FORMAL_FINAL_ID_FAMILIES: Tuple[str, ...] = (
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
    "confounding_underpower",
    "replication_instability",
)
FORMAL_FINAL_OOD_FAMILIES: Tuple[str, ...] = (
    "leakage_metric_mismatch",
    "protocol_drift",
)
# Formal collection includes train/dev material for fitting and tuning, followed by
# locked final-ID/final-OOD clusters.  The final split counts are the preregistered
# promotion targets; train/dev are never counted as final evidence.
FORMAL_SPLITS: Tuple[str, ...] = ("train", "dev", "final_id", "final_ood")


@dataclass(frozen=True)
class V04ExtendedQuestion:
    question_id: str
    family: str
    variant: int
    split: str
    seed_offset: int
    description: str
    worlds: Tuple[WorldSpec, ...]
    target_actions: Mapping[str, ResearchAction]

    @property
    def policy_question_id(self) -> str:
        return "tier1_v04_extended_public_question"

    @property
    def world_map(self) -> Mapping[str, WorldSpec]:
        return MappingProxyType({world.world_id: world for world in self.worlds})

    @property
    def paired_world_ids(self) -> Tuple[str, str]:
        return (
            f"{self.question_id}__supported",
            f"{self.question_id}__refuted",
        )

    def target_action(self, world_id: str) -> ResearchAction:
        return self.target_actions[world_id]

    def to_dict(self, *, include_hidden: bool = True) -> dict:
        payload = {
            "question_id": self.question_id,
            "policy_question_id": self.policy_question_id,
            "family": self.family,
            "variant": int(self.variant),
            "split": self.split,
            "seed_offset": int(self.seed_offset),
            "description": self.description,
            "world_count": len(self.worlds),
        }
        if include_hidden:
            payload["worlds"] = [asdict(world) for world in self.worlds]
            # Legacy evaluator calibration only.  The posterior planner and all
            # policy-facing exports deliberately exclude this field.
            payload["legacy_target_actions_audit_only"] = {
                key: value.value for key, value in self.target_actions.items()
            }
        return payload


@dataclass(frozen=True)
class V04ExtendedBenchmark:
    questions: Tuple[V04ExtendedQuestion, ...]
    protocol_version: str = "pesco_v0_2"
    schema_version: str = V04_EXTENDED_SCHEMA

    def __post_init__(self) -> None:
        if len(self.questions) != 64:
            raise ValueError("v0.4 extended requires 64 independent questions")
        counts = {family: sum(question.family == family for question in self.questions) for family in V04_EXTENDED_FAMILIES}
        if counts != {family: 8 for family in V04_EXTENDED_FAMILIES}:
            raise ValueError(f"expected 8 variants per family, got {counts}")
        if sum(question.split == "train" for question in self.questions) != 24:
            raise ValueError("v0.4 extended train split must contain 24 clusters")
        if sum(question.split == "dev" for question in self.questions) != 20:
            raise ValueError("v0.4 extended promotion split must contain 20 clusters")
        if sum(question.split == "diagnostic_ood" for question in self.questions) != 20:
            raise ValueError("v0.4 extended OOD split must contain 20 clusters")

    @property
    def question_map(self) -> Mapping[str, V04ExtendedQuestion]:
        return MappingProxyType({question.question_id: question for question in self.questions})

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    @property
    def split_names(self) -> Tuple[str, ...]:
        """Registered diagnostic split names in stable manifest order."""

        return EXTENDED_SPLITS

    def question(self, question_id: str) -> V04ExtendedQuestion:
        return self.question_map[question_id]

    def make_environment(self, question_id: str, protocol: Optional[Protocol] = None, budget: Optional[int] = None) -> Tier1TabularEnvironment:
        question = self.question(question_id)
        return Tier1TabularEnvironment(
            worlds=question.worlds,
            protocol=protocol or Protocol(protocol_version=self.protocol_version),
            budget=budget,
        )

    def manifest(self, *, include_hidden: bool = True, exploration_seeds: Optional[Sequence[int]] = None) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "question_count": len(self.questions),
            "world_count": len(self.worlds),
            "mechanism_family_count": len(V04_EXTENDED_FAMILIES),
            "mechanism_families": list(V04_EXTENDED_FAMILIES),
            "exploration_seeds": list(exploration_seeds or V04_EXTENDED_EXPLORATION_SEEDS),
            "confirmation_seeds": list(V04_EXTENDED_CONFIRMATION_SEEDS),
            "counts_by_split": {
                split: sum(question.split == split for question in self.questions)
                for split in ("train", "dev", "diagnostic_ood")
            },
            "ood_protocol_holdout": {
                "whole_families": ["leakage_metric_mismatch", "protocol_drift"],
                "additional_clusters": "replication_instability variants 1-4",
            },
            "questions": [question.to_dict(include_hidden=include_hidden) for question in self.questions],
            "dual_tracks": {
                TRACK_ORACLE_STATE: "trusted initial state plus public evidence",
                TRACK_RAW_EVIDENCE: "numeric correlations/overlap/CI/sample/replication/log receipts only",
            },
            "formal_comparison_authorized": False,
            "diagnostic_only": True,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        payload["manifest_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return payload


def _split_for(family: str, variant: int) -> str:
    if family in {"leakage_metric_mismatch", "protocol_drift"}:
        return "diagnostic_ood"
    if family == "replication_instability":
        return "diagnostic_ood" if variant <= 4 else "dev"
    if family in {"group_leakage", "causal_confounding", "low_sample_variance", "subgroup_metric_mismatch"}:
        return "dev" if variant <= 4 else "train"
    return "train"


def _family_flags(family: str, kind: str, variant: int) -> dict:
    """Return latent flags/effects; the resulting fields stay evaluator-hidden."""

    odd = int(variant) % 2 == 1
    base = {
        "true_a": 0.18,
        "true_b": 0.03,
        "noise": 0.045 + 0.002 * (variant % 3),
        "samples": 96,
        "leakage": False,
        "confounding": False,
        "metric_mismatch": False,
        "protocol_invalid": False,
    }
    if kind == "supported":
        base["true_a"] = 0.16 + 0.01 * (variant % 4)
        base["true_b"] = 0.02
    elif kind == "refuted":
        base["true_a"] = -0.08 - 0.005 * (variant % 3)
        # Half the variants make method B visibly useful in the executed branch;
        # this creates same-question reversals without exposing B before action.
        base["true_b"] = 0.18 if odd else -0.08
    elif kind == "insufficient":
        base["true_a"] = 0.08
        base["true_b"] = 0.18 if odd else 0.02
        base["noise"] = 0.36 + 0.01 * (variant % 3)
        base["samples"] = 18
    else:  # invalid
        base["true_a"] = 0.0
        base["true_b"] = 0.18 if odd else 0.06

    if family in {"group_leakage", "leakage_metric_mismatch"}:
        base["leakage"] = kind == "invalid"
    if family in {"causal_confounding", "confounding_underpower"}:
        base["confounding"] = kind == "invalid"
    if family in {"subgroup_metric_mismatch", "leakage_metric_mismatch"}:
        base["metric_mismatch"] = kind == "invalid"
    if family in {"low_sample_variance", "protocol_drift", "replication_instability"}:
        # The Insufficient world is intentionally underpowered but not invalid; the
        # sample-count/CI rule must be observable before SAMPLE can repair it.
        base["protocol_invalid"] = kind == "invalid"
    if family == "confounding_underpower":
        base["protocol_invalid"] = kind == "invalid"
        if kind in {"invalid", "insufficient"}:
            base["samples"] = 18
            base["noise"] = 0.36 + 0.01 * (variant % 3)
    if family == "leakage_metric_mismatch" and kind == "invalid":
        base["true_b"] = 0.18
    if family == "protocol_drift" and kind == "invalid":
        base["true_b"] = 0.18 if odd else 0.06
    if family == "replication_instability":
        base["noise"] = 0.30 + 0.03 * (variant % 4) if kind in {"invalid", "insufficient"} else base["noise"]
        # Keep an underpowered replication attempt distinguishable from a broken
        # protocol: the former must surface as INSUFFICIENT via its wide CI/sample
        # receipt, while only the registered invalid world emits the protocol
        # failure signal and is classified INVALID.
        base["protocol_invalid"] = kind == "invalid"
    return base


def _target_for(family: str, kind: str, variant: int) -> ResearchAction:
    if kind == "supported":
        return ResearchAction.CONTINUE
    if family in {"group_leakage", "leakage_metric_mismatch"}:
        return {"refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.REPAIR}[kind]
    if family in {"causal_confounding", "confounding_underpower"}:
        return {"refuted": ResearchAction.SWITCH if variant % 2 else ResearchAction.CONTINUE, "insufficient": ResearchAction.SWITCH, "invalid": ResearchAction.REPAIR}[kind]
    if family in {"low_sample_variance", "replication_instability"}:
        return {"refuted": ResearchAction.CONTINUE, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.SAMPLE}[kind]
    if family == "subgroup_metric_mismatch":
        return {"refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SWITCH, "invalid": ResearchAction.SWITCH}[kind]
    if family == "protocol_drift":
        return {"refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.REPAIR}[kind]
    return ResearchAction.CONTINUE


def build_tier1_v04_extended_benchmark() -> V04ExtendedBenchmark:
    questions = []
    for family_index, family in enumerate(V04_EXTENDED_FAMILIES):
        for variant in range(1, 9):
            question_id = f"t1_v04_{family}_{variant:02d}"
            worlds = []
            targets = {}
            for kind_index, kind in enumerate(WORLD_KINDS):
                params = _family_flags(family, kind, variant)
                world = WorldSpec(
                    world_id=f"{question_id}__{kind}",
                    kind=kind,
                    true_effect_a=float(params["true_a"]),
                    true_effect_b=float(params["true_b"]),
                    noise_scale=float(params["noise"]),
                    initial_samples=int(params["samples"]),
                    leakage=bool(params["leakage"]),
                    confounding=bool(params["confounding"]),
                    seed_offset=10_000 * (family_index + 1) + 100 * variant + kind_index,
                    question_family=family,
                    metric_mismatch=bool(params["metric_mismatch"]),
                    protocol_invalid=bool(params["protocol_invalid"]),
                )
                worlds.append(world)
                targets[world.world_id] = _target_for(family, kind, variant)
            questions.append(V04ExtendedQuestion(
                question_id=question_id,
                family=family,
                variant=variant,
                split=_split_for(family, variant),
                seed_offset=10_000 * (family_index + 1) + 100 * variant,
                description=f"Tier-1 v0.4 {family} mechanism/protocol variant {variant}; paired worlds share the question.",
                worlds=tuple(worlds),
                target_actions=MappingProxyType(targets),
            ))
    return V04ExtendedBenchmark(tuple(questions))


@dataclass(frozen=True)
class V04FormalFinalBenchmark:
    """Locked final-ID/final-OOD benchmark definition.

    This profile is deliberately separate from :class:`V04ExtendedBenchmark`.
    It uses fresh opaque question/world IDs and whole mechanism families for OOD,
    so creating a formal manifest cannot mutate or silently relabel the diagnostic
    64-question artifact.  The builder defines the final split contract; a model
    evaluation remains fail-closed until an explicit final-access authorization is
    supplied to the formal runner.
    """

    questions: Tuple[V04ExtendedQuestion, ...]
    protocol_version: str = "pesco_v0_2"
    schema_version: str = V04_FORMAL_SCHEMA

    def __post_init__(self) -> None:
        ids = tuple(question.question_id for question in self.questions)
        if len(ids) != len(set(ids)):
            raise ValueError("formal final question IDs must be unique")
        if any(question.split not in FORMAL_SPLITS for question in self.questions):
            raise ValueError(f"formal final splits must be {FORMAL_SPLITS}")
        final_id = tuple(question for question in self.questions if question.split == "final_id")
        final_ood = tuple(question for question in self.questions if question.split == "final_ood")
        if len(final_id) < 20 or len(final_ood) < 20:
            raise ValueError("formal final requires at least 20 final_id and 20 final_ood questions")
        id_families = {question.family for question in final_id}
        ood_families = {question.family for question in final_ood}
        if not id_families or not ood_families or id_families.intersection(ood_families):
            raise ValueError("formal final OOD must be a whole-family holdout")
        all_id_families = {
            question.family for question in self.questions
            if question.split in {"train", "dev", "final_id"}
        }
        if all_id_families != set(FORMAL_FINAL_ID_FAMILIES):
            raise ValueError("final_id family registry does not match the formal profile")
        if ood_families != set(FORMAL_FINAL_OOD_FAMILIES):
            raise ValueError("final_ood family registry does not match the formal profile")
        for question in self.questions:
            if tuple(world.kind for world in question.worlds) != WORLD_KINDS:
                raise ValueError("formal final questions must contain the four ordered worlds")

    @property
    def question_map(self) -> Mapping[str, V04ExtendedQuestion]:
        return MappingProxyType({question.question_id: question for question in self.questions})

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    @property
    def split_names(self) -> Tuple[str, ...]:
        return FORMAL_SPLITS

    @property
    def final_id_questions(self) -> Tuple[V04ExtendedQuestion, ...]:
        return tuple(question for question in self.questions if question.split == "final_id")

    @property
    def final_ood_questions(self) -> Tuple[V04ExtendedQuestion, ...]:
        return tuple(question for question in self.questions if question.split == "final_ood")

    def question(self, question_id: str) -> V04ExtendedQuestion:
        return self.question_map[question_id]

    def make_environment(
        self,
        question_id: str,
        protocol: Optional[Protocol] = None,
        budget: Optional[int] = None,
    ) -> Tier1TabularEnvironment:
        question = self.question(question_id)
        return Tier1TabularEnvironment(
            worlds=question.worlds,
            protocol=protocol or Protocol(protocol_version=self.protocol_version),
            budget=budget,
        )

    def manifest(
        self,
        *,
        include_hidden: bool = False,
        exploration_seeds: Optional[Sequence[int]] = None,
    ) -> dict:
        """Return a locked formal manifest.

        The public form intentionally carries opaque question IDs and split counts
        only.  Family membership, world parameters, and legacy target actions remain
        evaluator-audit data in the hidden form and are never policy inputs.
        """

        questions = []
        for question in self.questions:
            item = {
                "question_id": question.question_id,
                "policy_question_id": question.policy_question_id,
                "split": question.split,
                "variant": int(question.variant),
                "world_count": len(question.worlds),
            }
            if include_hidden:
                item.update({
                    "family": question.family,
                    "seed_offset": int(question.seed_offset),
                    "description": question.description,
                    "worlds": [asdict(world) for world in question.worlds],
                    "legacy_target_actions_audit_only": {
                        key: value.value for key, value in question.target_actions.items()
                    },
                })
            questions.append(item)
        payload = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "profile": "formal_final",
            "question_count": len(self.questions),
            "world_count": len(self.worlds),
            "counts_by_split": {
                split: sum(question.split == split for question in self.questions)
                for split in FORMAL_SPLITS
            },
            "final_id_question_minimum": 20,
            "final_ood_question_minimum": 20,
            "mechanism_family_count": len(FORMAL_FINAL_ID_FAMILIES) + len(FORMAL_FINAL_OOD_FAMILIES),
            "mechanism_families": sorted(set(FORMAL_FINAL_ID_FAMILIES + FORMAL_FINAL_OOD_FAMILIES)) if include_hidden else [],
            "final_ood_family_count": len(FORMAL_FINAL_OOD_FAMILIES),
            "whole_family_holdout": {
                "enabled": True,
                "disjoint": True,
                "final_ood_families": list(FORMAL_FINAL_OOD_FAMILIES) if include_hidden else [],
                "final_id_families": list(FORMAL_FINAL_ID_FAMILIES) if include_hidden else [],
            },
            "exploration_seeds": list(exploration_seeds or V04_EXTENDED_EXPLORATION_SEEDS),
            "confirmation_seeds": list(V04_EXTENDED_CONFIRMATION_SEEDS),
            "questions": questions,
            "final_access": {
                "locked": True,
                "first_access_completed": False,
                "authorization_required": True,
                "formal_evaluation_completed": False,
                "formal_comparison_authorized": False,
            },
            "training_split_contract": {
                "train": "fit-only",
                "dev": "tuning-only",
                "final_id": "locked-promotion",
                "final_ood": "whole-family-heldout",
            },
            "diagnostic_only": False,
            "formal_comparison_authorized": False,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        payload["manifest_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return payload


def build_tier1_v04_formal_final_benchmark() -> V04FormalFinalBenchmark:
    """Build fresh final-ID/final-OOD questions without touching the diagnostic set."""

    questions: list[V04ExtendedQuestion] = []
    # Six ID families × four variants per split = 24 train, 24 dev, and 24
    # final-ID questions.  Two whole held-out families × ten variants = 20
    # final-OOD questions.  The OOD families never appear in train/dev.
    registry = (
        [(family, "train", variant) for family in FORMAL_FINAL_ID_FAMILIES for variant in range(5, 9)]
        + [(family, "dev", variant) for family in FORMAL_FINAL_ID_FAMILIES for variant in range(9, 13)]
        + [(family, "final_id", variant) for family in FORMAL_FINAL_ID_FAMILIES for variant in range(1, 5)]
        + [(family, "final_ood", variant) for family in FORMAL_FINAL_OOD_FAMILIES for variant in range(1, 11)]
    )
    diagnostic_ids = {
        question.question_id for question in build_tier1_v04_extended_benchmark().questions
    }
    for ordinal, (family, split, variant) in enumerate(registry, start=1):
        question_id = f"t1_v04_formal_{ordinal:03d}"
        if question_id in diagnostic_ids:
            raise RuntimeError("formal question ID unexpectedly overlaps diagnostic benchmark")
        worlds = []
        targets = {}
        family_index = V04_EXTENDED_FAMILIES.index(family)
        seed_offset = 300_000 + 10_000 * family_index + 100 * variant + ordinal
        for kind_index, kind in enumerate(WORLD_KINDS):
            params = _family_flags(family, kind, variant)
            world = WorldSpec(
                world_id=f"{question_id}__{kind}",
                kind=kind,
                true_effect_a=float(params["true_a"]),
                true_effect_b=float(params["true_b"]),
                noise_scale=float(params["noise"]),
                initial_samples=int(params["samples"]),
                leakage=bool(params["leakage"]),
                confounding=bool(params["confounding"]),
                seed_offset=seed_offset + kind_index,
                question_family=family,
                metric_mismatch=bool(params["metric_mismatch"]),
                protocol_invalid=bool(params["protocol_invalid"]),
            )
            worlds.append(world)
            targets[world.world_id] = _target_for(family, kind, variant)
        questions.append(V04ExtendedQuestion(
            question_id=question_id,
            family=family,
            variant=variant,
            split=split,
            seed_offset=seed_offset,
            description=f"Formal final {split} opaque question {ordinal:03d}; family held evaluator-side.",
            worlds=tuple(worlds),
            target_actions=MappingProxyType(targets),
        ))
    return V04FormalFinalBenchmark(tuple(questions))


def _raw_evidence(output: ExperimentOutput, observation: Observation, *, repeated_runs: int = 0) -> Tuple[Tuple[str, float], ...]:
    """Serialize only policy-observable raw experiment receipts.

    The Tier-1 environment also records ``hidden_validation_*`` fields for the
    trusted evaluator's leakage audit.  Those values are intentionally *not* raw
    evidence available to a policy and must never enter this track's feature vector.
    """

    return tuple(sorted({
        "treatment_confounder_correlation": float(getattr(output, "treatment_confounder_correlation", 0.0)),
        "group_overlap_count": float(getattr(output, "group_overlap_count", 0)),
        "replication_effect_delta": 0.0,
        "replication_ci_width": float(output.ci_high - output.ci_low),
        "replication_sample_size": float(output.sample_size),
        "replication_seed_count": float(output.seed_count),
        "log_validity_count": float(len(output.validity_signals)),
        "log_repeated_runs": float(repeated_runs),
        "log_protocol_change_count": float(sum(signal in {"split_protocol_updated", "confounder_adjusted_estimator", "group_held_out_split"} for signal in output.validity_signals)),
    }.items()))


def _track_observation(track: str, observation: Observation, output: ExperimentOutput, verdict: Verdict, *, repeated_runs: int = 0) -> Observation:
    if track == TRACK_ORACLE_STATE:
        return Observation(
            question_id=observation.question_id,
            turn=observation.turn,
            current_method=observation.current_method,
            effect_estimate=observation.effect_estimate,
            ci_low=observation.ci_low,
            ci_high=observation.ci_high,
            sample_size=observation.sample_size,
            seed_count=observation.seed_count,
            remaining_budget=observation.remaining_budget,
            metric_name=observation.metric_name,
            validity_signals=observation.validity_signals,
            history_summary=observation.history_summary,
            hypothesis_probability=observation.hypothesis_probability,
            active_hypothesis_id=observation.active_hypothesis_id,
            hypothesis_beliefs=observation.hypothesis_beliefs,
            task_family=observation.task_family,
            track=TRACK_ORACLE_STATE,
            raw_evidence=(),
        )
    return Observation(
        question_id=observation.question_id,
        turn=observation.turn,
        current_method=observation.current_method,
        effect_estimate=observation.effect_estimate,
        ci_low=observation.ci_low,
        ci_high=observation.ci_high,
        sample_size=observation.sample_size,
        seed_count=observation.seed_count,
        remaining_budget=observation.remaining_budget,
        metric_name=observation.metric_name,
        validity_signals=(),
        history_summary=(),
        hypothesis_probability=0.5,
        active_hypothesis_id="H_A",
        hypothesis_beliefs=(),
        task_family="raw_evidence",
        track=TRACK_RAW_EVIDENCE,
        raw_evidence=_raw_evidence(output, observation, repeated_runs=repeated_runs),
    )


def build_pre_action_raw_observation(
    question: Any,
    world: WorldSpec,
    protocol: Optional[Protocol] = None,
) -> Observation:
    """Construct the policy-visible raw observation before candidate branches.

    The helper intentionally executes only the registered baseline (method A) and
    returns immediately.  It is used by the counterfactual leakage audit: changing a
    hidden method-B effect must not alter this observation or its canonical hash.
    No confirmation or candidate-action receipt is consulted.
    """

    protocol = protocol or Protocol(protocol_version="pesco_v0_2")
    environment = Tier1TabularEnvironment(
        worlds=(world,),
        protocol=protocol,
        budget=protocol.max_budget,
    )
    environment.reset(
        getattr(question, "policy_question_id", "tier1_v04_extended_public_question"),
        world.world_id,
        seed=17,
    )
    baseline = environment.execute_option(
        ResearchAction.CONTINUE,
        seeds=protocol.exploration_seeds,
    )
    verifier = TrustedVerifier(protocol)
    verdict = verifier.evaluate(baseline, environment, confirm=False)
    return _track_observation(
        TRACK_RAW_EVIDENCE,
        environment.visible_observation(),
        baseline,
        verdict,
        repeated_runs=len(protocol.exploration_seeds),
    )


def pre_action_observation_hash(observation: Observation) -> str:
    """Return the canonical digest used by the counterfactual leakage audit."""

    encoded = json.dumps(
        observation.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def audit_counterfactual_raw_observation_leakage(
    benchmark: Any,
    protocol: Optional[Protocol] = None,
    *,
    world_limit: Optional[int] = None,
    effect_delta: float = 1.234567,
) -> dict:
    """Verify that an unselected method-B effect cannot change pre-action inputs.

    For each audited world, only ``true_effect_b`` is changed in a counterfactual
    copy.  The two baseline/raw observations are hashed before any candidate branch
    is executed.  A mismatch is a hard leakage finding rather than a statistical
    tolerance failure.
    """

    protocol = protocol or Protocol(protocol_version="pesco_v0_2")
    rows: list[dict] = []
    audited = 0
    for question in benchmark.questions:
        for world in question.worlds:
            if world_limit is not None and audited >= max(0, int(world_limit)):
                break
            counterfactual = replace(world, true_effect_b=float(world.true_effect_b) + float(effect_delta))
            original = build_pre_action_raw_observation(question, world, protocol)
            altered = build_pre_action_raw_observation(question, counterfactual, protocol)
            original_hash = pre_action_observation_hash(original)
            altered_hash = pre_action_observation_hash(altered)
            rows.append({
                "question_id": str(question.question_id),
                "world_id": str(world.world_id),
                "original_hash": original_hash,
                "counterfactual_hash": altered_hash,
                "unchanged": original_hash == altered_hash,
            })
            audited += 1
        if world_limit is not None and audited >= max(0, int(world_limit)):
            break
    return {
        "schema_version": "pesco_counterfactual_raw_leakage_audit_v0.1",
        "audited_world_count": len(rows),
        "effect_delta": float(effect_delta),
        "all_hashes_unchanged": bool(rows) and all(row["unchanged"] for row in rows),
        "rows": rows,
        "decision_before_candidate_branches": True,
        "feature_removed": "log_confirmation_pass_rate",
        "status": "pass" if rows and all(row["unchanged"] for row in rows) else "fail",
    }


def counterfactual_raw_observation_audit(
    question: Any,
    world: WorldSpec,
    protocol: Optional[Protocol] = None,
    *,
    method_b_delta: float = 1.234567,
) -> dict:
    """Verify that an unselected method-B latent change is invisible pre-decision.

    The two observations execute only the common method-A baseline.  The hidden
    method-B efficacy is changed in a counterfactual world with all public identity
    fields held fixed; equal canonical observations and hashes are therefore a
    direct information-boundary test rather than a post-hoc feature assertion.
    """

    from dataclasses import replace

    altered = replace(world, true_effect_b=float(world.true_effect_b) + float(method_b_delta))
    original_observation = build_pre_action_raw_observation(question, world, protocol)
    altered_observation = build_pre_action_raw_observation(question, altered, protocol)
    original_hash = pre_action_observation_hash(original_observation)
    altered_hash = pre_action_observation_hash(altered_observation)
    return {
        "question_id": str(getattr(question, "question_id", "")),
        "world_kind": str(world.kind),
        "method_b_delta": float(method_b_delta),
        "original_hash": original_hash,
        "counterfactual_hash": altered_hash,
        "observation_equal": original_observation.to_dict() == altered_observation.to_dict(),
        "pass": bool(original_hash == altered_hash and original_observation.to_dict() == altered_observation.to_dict()),
        "decision_boundary": "baseline_method_a_before_candidate_branches",
    }


def _independent_replicate_confirmations(
    env: Tier1TabularEnvironment,
    pre_action_snapshot: Any,
    action: ResearchAction,
    exploration_seeds: Sequence[int],
    protocol: Protocol,
    verifier: TrustedVerifier,
    question: V04ExtendedQuestion,
    world: WorldSpec,
    initial_observation: Observation,
) -> Tuple[list[float], list[dict]]:
    """Execute and independently confirm each one-seed replicate.

    The vector confirmation performed by ``TrustedVerifier`` is not copied into these
    rows.  Each eligible replicate gets its own held-out confirmation seed and its
    own receipt; this is the key v0.4 fix for seed-level uncertainty.
    """

    values: list[float] = []
    audits: list[dict] = []
    for index, seed in enumerate(exploration_seeds):
        branch = env.clone_from_snapshot(pre_action_snapshot)
        output = branch.execute_option(action, seeds=(int(seed),))
        verdict = verifier.evaluate(output, branch, confirm=False)
        passed: Optional[bool] = None
        confirmation_seed = int(protocol.confirmation_seeds[index % len(protocol.confirmation_seeds)])
        if verdict.evidence_state in {EvidenceState.SUPPORTED, EvidenceState.REFUTED} and verdict.validity_pass:
            candidate = branch.clone_from_snapshot(branch.snapshot())
            confirmation_output = candidate._simulate(  # noqa: SLF001 - verifier boundary
                method=output.method,
                option=action,
                seeds=(confirmation_seed,),
                confirmation=True,
            )
            confirmation_verdict = verifier.evaluate(confirmation_output, candidate, confirm=False)
            passed = bool(
                confirmation_verdict.validity_pass
                and confirmation_verdict.evidence_state is verdict.evidence_state
                and confirmation_output.dataset_hash != output.dataset_hash
                and confirmation_output.split_hash != output.split_hash
            )
        # Reuse the trusted utility function with an explicit per-replicate receipt.
        utility_verdict = verdict
        if passed is not None:
            from dataclasses import replace
            utility_verdict = replace(
                verdict,
                independent_confirmation_performed=True,
                independent_confirmation_passed=bool(passed),
            )
        reward_components = tier1_scientific_utility_components(
            question,
            world,
            action,
            output,
            utility_verdict,
            protocol,
            initial_observation=initial_observation,
        )
        # Keep the scalar and atomic receipt on the exact same evaluator path.
        utility = float(sum(reward_components.values()))
        values.append(float(utility))
        audits.append({
            "exploration_seed": int(seed),
            "confirmation_seed": confirmation_seed,
            # A receipt is emitted for every exploration replicate, while the
            # denominator for the replication-rate metric contains only receipts
            # for which an independent confirmation was actually attempted.  Keep
            # this canonical field separate from the receipt's ``passed`` result so
            # failed confirmations remain in the denominator rather than being
            # silently dropped.  ``eligible`` is retained as a legacy alias used by
            # reversal-label construction.
            "confirmation_eligible": bool(passed is not None),
            "eligible": passed is not None,
            "passed": bool(passed) if passed is not None else None,
            "state": verdict.evidence_state.value,
            "validity_pass": bool(verdict.validity_pass),
            "execution_cost": float(output.execution_cost),
            "method": str(output.method),
            "dataset_hash": output.dataset_hash,
            "confirmation_dataset_hash": getattr(confirmation_output, "dataset_hash", None) if passed is not None else None,
            "split_hash": output.split_hash,
            "confirmation_split_hash": getattr(confirmation_output, "split_hash", None) if passed is not None else None,
            "reward_components": {
                name: float(reward_components[name])
                for name in TIER1_REWARD_COMPONENT_NAMES
            },
        })
    return values, audits


def _confirmation_receipt_eligible(receipt: Mapping[str, Any]) -> bool:
    """Read the canonical attempted-confirmation bit from a receipt.

    ``confirmation_eligible`` is the current schema.  ``eligible`` remains a
    compatibility alias for older v0.4/v0.3 exports, but must not override the
    canonical field when both are present.  This distinction is important for
    audits that deliberately construct a receipt with inconsistent legacy and
    current fields.
    """

    if "confirmation_eligible" in receipt:
        return bool(receipt.get("confirmation_eligible"))
    return bool(receipt.get("eligible", False))


def _utility_for_branch(question: V04ExtendedQuestion, world: WorldSpec, action: ResearchAction, output: ExperimentOutput, verdict: Verdict, initial_observation: Observation, protocol: Protocol) -> float:
    from ..environments.tier1_benchmark import tier1_scientific_utility
    return float(tier1_scientific_utility(question, world, action, output, verdict, protocol, initial_observation=initial_observation))


def collect_tier1_v04_extended(
    benchmark: Optional[V04ExtendedBenchmark] = None,
    protocol: Optional[Protocol] = None,
    *,
    track: str = TRACK_ORACLE_STATE,
    question_limit: Optional[int] = None,
) -> Tuple[Any, dict]:
    """Collect a track-specific DecisionDataset and audit metadata.

    Importing ``DecisionDataset`` here avoids a module-level cycle with the v0.3
    suite's evaluator helpers.
    """

    from ..algorithms.differentiable_strategy import DecisionDataset, DecisionExample, ReversalExample
    benchmark = benchmark or build_tier1_v04_extended_benchmark()
    protocol = protocol or Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
    )
    if tuple(protocol.exploration_seeds) != V04_EXTENDED_EXPLORATION_SEEDS:
        raise ValueError("v0.4 extended requires eight pre-registered exploration seeds")
    examples: list[DecisionExample] = []
    trajectory_rows: list[dict] = []
    questions = benchmark.questions if question_limit is None else benchmark.questions[: max(0, int(question_limit))]
    for question in questions:
        for world in question.worlds:
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            env.reset(question.policy_question_id, world.world_id, seed=17)
            baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            verifier = TrustedVerifier(protocol)
            initial_verdict = verifier.evaluate(baseline, env, confirm=False)
            initial_observation = env.visible_observation()
            pre_action_snapshot = env.snapshot()
            # Materialize the raw-evidence track before *any* candidate branch or
            # confirmation is executed.  This makes the information boundary
            # auditable and prevents a future metadata field from accidentally
            # depending on an unselected action.
            track_observation = _track_observation(
                track,
                initial_observation,
                baseline,
                initial_verdict,
                repeated_runs=len(protocol.exploration_seeds),
            )
            branch_utilities: list[float] = []
            branch_states: list[EvidenceState] = []
            seed_utility_map: dict[str, list[float]] = {}
            replicate_confirmation_map: dict[str, list[dict]] = {}
            branch_costs: dict[str, float] = {}
            branch_validity: dict[str, bool] = {}
            branch_evidence: dict[str, str] = {}
            reward_components: dict[str, dict[str, float]] = {}
            for action in ACTION_SET:
                seed_values, confirmations = _independent_replicate_confirmations(
                    env,
                    pre_action_snapshot,
                    action,
                    protocol.exploration_seeds,
                    protocol,
                    verifier,
                    question,
                    world,
                    initial_observation,
                )
                # Canonical branch utility is the mean of independently confirmed
                # replicate utilities, never a vector confirmation bonus copied to
                # each seed.
                utility = sum(seed_values) / max(1, len(seed_values))
                branch_utilities.append(float(utility))
                state_counts: dict[str, int] = {}
                validity_count = 0
                costs = []
                for item in confirmations:
                    state_counts[item["state"]] = state_counts.get(item["state"], 0) + 1
                    validity_count += int(bool(item.get("validity_pass", False)))
                    costs.append(float(item.get("execution_cost", 0.0)))
                majority_state = max(state_counts, key=state_counts.get) if state_counts else EvidenceState.INSUFFICIENT.value
                branch_states.append(EvidenceState(majority_state))
                seed_utility_map[action.value] = seed_values
                replicate_confirmation_map[action.value] = confirmations
                # The branch utility is the mean over independently confirmed
                # replicate utilities.  Aggregate each atomic term with the same
                # mean so P2 can verify sum(reward_components[action].values())
                # against branch_utilities[action] without reconstructing terms.
                denominator = max(1, len(confirmations))
                reward_components[action.value] = {
                    name: float(
                        sum(
                            float(item["reward_components"][name])
                            for item in confirmations
                        ) / denominator
                    )
                    for name in TIER1_REWARD_COMPONENT_NAMES
                }
                branch_costs[action.value] = sum(costs) / max(1, len(costs))
                branch_validity[action.value] = validity_count >= max(1, len(confirmations) // 2)
                branch_evidence[action.value] = majority_state
            # Evaluator-only multi-step trajectory: diagnose -> repair/action -> retest
            # -> replicate.  The chosen corrective action is logged, never exposed as
            # a policy target.
            corrective = {
                EvidenceState.INVALID: ResearchAction.REPAIR,
                EvidenceState.INSUFFICIENT: ResearchAction.SAMPLE,
                EvidenceState.REFUTED: ResearchAction.SWITCH,
                EvidenceState.SUPPORTED: ResearchAction.REPLICATE,
            }.get(initial_verdict.evidence_state, ResearchAction.REPLICATE)
            trajectory_env = benchmark.make_environment(question.question_id, protocol=protocol, budget=6)
            trajectory_env.reset(question.policy_question_id, world.world_id, seed=17)
            trajectory: list[dict] = []
            first = trajectory_env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            first_v = verifier.evaluate(first, trajectory_env, confirm=False)
            trajectory.append({"phase": "diagnose", "action": ResearchAction.CONTINUE.value, "state": first_v.evidence_state.value, "valid": first_v.validity_pass})
            if corrective is not ResearchAction.REPLICATE:
                repaired = trajectory_env.execute_option(corrective, seeds=protocol.exploration_seeds)
                repaired_v = verifier.evaluate(repaired, trajectory_env, confirm=False)
                trajectory.append({"phase": "repair_or_change", "action": corrective.value, "state": repaired_v.evidence_state.value, "valid": repaired_v.validity_pass})
            retest = trajectory_env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            retest_v = verifier.evaluate(retest, trajectory_env, confirm=False)
            trajectory.append({"phase": "retest", "action": ResearchAction.CONTINUE.value, "state": retest_v.evidence_state.value, "valid": retest_v.validity_pass})
            if trajectory_env.remaining_budget() > 0:
                replicate = trajectory_env.execute_option(ResearchAction.REPLICATE, seeds=protocol.exploration_seeds)
                replicate_v = verifier.evaluate(replicate, trajectory_env, confirm=False)
                trajectory.append({"phase": "replicate", "action": ResearchAction.REPLICATE.value, "state": replicate_v.evidence_state.value, "valid": replicate_v.validity_pass})
            trajectory_rows.append({
                "question_id": question.question_id,
                "world_id_audit": world.world_id,
                "family": question.family,
                "split": question.split,
                "steps": trajectory,
                "diagnose_repair_retest_replicate_complete": [row["phase"] for row in trajectory] == ["diagnose", "repair_or_change", "retest", "replicate"] or [row["phase"] for row in trajectory] == ["diagnose", "retest", "replicate"],
            })
            eligible_confirmation_count = sum(
                _confirmation_receipt_eligible(item)
                for values in replicate_confirmation_map.values()
                for item in values
            )
            passed_confirmation_count = sum(
                _confirmation_receipt_eligible(item) and item.get("passed") is True
                for values in replicate_confirmation_map.values()
                for item in values
            )
            confirmation_passed = bool(
                eligible_confirmation_count > 0
                and passed_confirmation_count == eligible_confirmation_count
            )
            raw_source_output = env._last_output if env._last_output is not None else baseline
            metadata = {
                "family": question.family,
                "variant": int(question.variant),
                "record_granularity": "question_world_group",
                "question_world_group_id": f"{question.question_id}|{world.world_id}",
                "legacy_target_action_audit_only": question.target_action(world.world_id).value,
                "legacy_target_action_used_for_training": False,
                "reward_source": "independent_per_replicate_public_transition_utility",
                "policy_features_exclude_hidden_world": True,
                "track": track,
                "pre_action_observation_hash": pre_action_observation_hash(track_observation),
                "pre_action_observation_constructed_before_candidate_branches": True,
                "cross_candidate_confirmation_feature_excluded": True,
                "branch_costs": branch_costs,
                "branch_validity": branch_validity,
                "branch_evidence_states": branch_evidence,
                "branch_seed_utilities": seed_utility_map,
                "branch_replicate_confirmation": replicate_confirmation_map,
                "reward_components": reward_components,
                "reward_component_names": list(TIER1_REWARD_COMPONENT_NAMES),
                "reward_components_aggregation": "mean_over_independent_replicates",
                "confirmation_bonus_copied_to_seed": False,
                "exploration_seed_count": len(protocol.exploration_seeds),
                "confirmation_seed_count": len(protocol.confirmation_seeds),
                "independent_replicate_confirmation_count": sum(len(values) for values in replicate_confirmation_map.values()),
                "independent_replicate_confirmation_receipt_count": sum(len(values) for values in replicate_confirmation_map.values()),
                "independent_replicate_confirmation_eligible_count": sum(
                    _confirmation_receipt_eligible(item)
                    for values in replicate_confirmation_map.values()
                    for item in values
                ),
                "independent_replicate_confirmation_passed_count": sum(
                    _confirmation_receipt_eligible(item) and item.get("passed", False) is True
                    for values in replicate_confirmation_map.values()
                    for item in values
                ),
                "trajectory_id": f"trajectory|{question.question_id}|{world.world_id}",
                "trajectory_recorded": True,
            }
            examples.append(DecisionExample(
                observation=track_observation,
                branch_utilities=tuple(branch_utilities),
                branch_states=tuple(branch_states),
                state_target=initial_verdict.evidence_state,
                split=question.split,
                question_id=question.question_id,
                world_id=world.world_id,
                world_pair_id=f"{question.question_id}:supported_refuted",
                confirmation_passed=bool(confirmation_passed),
                metadata=metadata,
            ))
    # Same-question paired reversals only.  Add every statistically confirmed action
    # pair, then let the formal gate count how many independent pairs each split has.
    reversals: list[ReversalExample] = []
    seen_reversals: set[tuple[int, int, str, str]] = set()
    for question in questions:
        question_indices = [
            index for index, example in enumerate(examples)
            if example.question_id == question.question_id
        ]
        # Every endpoint pair remains inside one registered question.  Including
        # supported/refuted, supported/insufficient, and refuted/invalid pairs gives
        # the formal split a chance to reach 30 independent reversals without
        # manufacturing cross-question pairs.
        for left_index, right_index in __import__("itertools").combinations(question_indices, 2):
            left = examples[left_index]
            right = examples[right_index]
            for action_left in ACTION_SET:
                for action_right in ACTION_SET:
                    if action_left is action_right:
                        continue
                    left_audit = left.metadata["branch_replicate_confirmation"][action_left.value]
                    right_audit = right.metadata["branch_replicate_confirmation"][action_right.value]
                    # A paired reversal needs independent confirmations on both sides,
                    # but a single unlucky held-out seed must not erase an otherwise
                    # replicated pair.  Require at least half of the eight eligible
                    # replicate confirmations on each endpoint; failed/unevaluable
                    # rows remain in the audit denominator.
                    left_passed = sum(
                        _confirmation_receipt_eligible(item) and item.get("passed") is True
                        for item in left_audit
                    )
                    right_passed = sum(
                        _confirmation_receipt_eligible(item) and item.get("passed") is True
                        for item in right_audit
                    )
                    left_eligible = sum(_confirmation_receipt_eligible(item) for item in left_audit)
                    right_eligible = sum(_confirmation_receipt_eligible(item) for item in right_audit)
                    if left_eligible < 4 or right_eligible < 4 or left_passed < 4 or right_passed < 4:
                        continue
                    pair = identify_confirmed_reversal(
                        question_id=question.question_id,
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
                    if pair.confirmed:
                        key = (left_index, right_index, action_left.value, action_right.value)
                        if key in seen_reversals:
                            continue
                        seen_reversals.add(key)
                        reversals.append(ReversalExample(
                            left=left_index,
                            right=right_index,
                            action_left=action_left,
                            action_right=action_right,
                            margin=0.05,
                            confirmed=True,
                            weight=max(1.0, abs(float(pair.double_difference))),
                            lcb_left=float(pair.lcb_a),
                            ucb_right=float(pair.ucb_b),
                            sample_count=len(left.metadata["branch_seed_utilities"][action_left.value]),
                        ))
    # Normalize the reversal-loss weights within each question so that a question
    # with many admissible action pairs contributes exactly one unit of weight.
    # The confidence/effect magnitude remains available in the relative weights,
    # while the question macro boundary is explicit and deterministic.
    reversal_groups: dict[str, list[int]] = {}
    for index, pair in enumerate(reversals):
        question_id = str(examples[pair.left].question_id)
        reversal_groups.setdefault(question_id, []).append(index)
    for question_id, indices in reversal_groups.items():
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
    seed_arrays = [
        tuple(float(value) for value in values)
        for example in examples
        for values in example.metadata["branch_seed_utilities"].values()
    ]
    split_names = tuple(
        getattr(
            benchmark,
            "split_names",
            tuple(dict.fromkeys(str(question.split) for question in questions)),
        )
    )
    split_counts = {
        split: sum(example.split == split for example in examples)
        for split in split_names
    }
    # Diagnostic keeps its historical dev/OOD gate; the formal profile uses the
    # locked final_id/final_ood names without silently dropping them from audits.
    gate_splits = (
        ("final_id", "final_ood")
        if {"final_id", "final_ood"}.issubset(split_names)
        else ("tune", "promotion")
        if {"tune", "promotion"}.issubset(split_names)
        else ("dev", "diagnostic_ood")
    )
    mechanism_families = sorted({str(question.family) for question in questions})
    provenance = {
        "schema_version": benchmark.schema_version,
        "benchmark_schema_version": benchmark.schema_version,
        "question_count": len(questions),
        "world_count": len(benchmark.worlds),
        "mechanism_family_count": len(mechanism_families),
        "mechanism_families": mechanism_families,
        "split_names": list(split_names),
        "counts_by_split": split_counts,
        "question_world_group_count": len(examples),
        "snapshot_branch_group_count": len(examples),
        "action_level_branch_count": len(examples) * len(ACTION_SET),
        "action_level_row_count": len(examples) * len(ACTION_SET),
        "exploration_seed_count": len(protocol.exploration_seeds),
        "seed_level_execution_count": len(examples) * len(ACTION_SET) * len(protocol.exploration_seeds),
        "seed_level_observation_count": len(examples) * len(ACTION_SET) * len(protocol.exploration_seeds),
        "independent_replicate_confirmation_count": sum(int(example.metadata["independent_replicate_confirmation_count"]) for example in examples),
        "independent_replicate_confirmation_receipt_count": sum(int(example.metadata["independent_replicate_confirmation_receipt_count"]) for example in examples),
        "independent_replicate_confirmation_eligible_count": sum(int(example.metadata["independent_replicate_confirmation_eligible_count"]) for example in examples),
        "independent_replicate_confirmation_passed_count": sum(int(example.metadata["independent_replicate_confirmation_passed_count"]) for example in examples),
        "confirmation_bonus_copied_to_seed": False,
        "seed_utility_array_count": len(seed_arrays),
        "seed_utility_varying_array_count": sum(len(set(round(value, 12) for value in values)) > 1 for values in seed_arrays),
        "same_question_reversal_count": len(reversals),
        "reversal_count": len(reversals),
        "reversal_weighting": "question_macro_equal_total_weight_one",
        "reversal_weight_sum_by_question": {
            question_id: sum(float(reversals[index].weight) for index in indices)
            for question_id, indices in sorted(reversal_groups.items())
        },
        "reversal_pair_scope": "same_question_world_pairs_only",
        "raw_observation_pre_action": True,
        "cross_candidate_confirmation_feature_excluded": True,
        "trajectory_count": len(trajectory_rows),
        "trajectory_complete_count": sum(bool(row["diagnose_repair_retest_replicate_complete"]) for row in trajectory_rows),
        "formal_split_pair_minimum": 30,
        "formal_split_cluster_minimum": 20,
        "formal_gate_splits": list(gate_splits),
        "formal_comparison_authorized": False,
        "diagnostic_only": not {"final_id", "final_ood"}.issubset(split_names),
    }
    dataset = DecisionDataset(
        examples=examples,
        reversals=reversals,
        schema_version=(
            "pesco_decision_dataset_v0.4_formal_final"
            if {"final_id", "final_ood"}.issubset(split_names)
            else "pesco_decision_dataset_p21_fresh_diagnostic"
            if {"tune", "promotion"}.issubset(split_names)
            else "pesco_decision_dataset_v0.4_extended"
        ),
        provenance=provenance,
    )
    audit = {
        "benchmark_manifest": benchmark.manifest(include_hidden=True),
        "public_benchmark_manifest": benchmark.manifest(include_hidden=False),
        "trajectory_rows": trajectory_rows,
        "track": track,
        "same_question_reversal_count": len(reversals),
        "reversal_counts_by_split": {
            split: sum(
                bool(pair.confirmed)
                and examples[pair.left].split == split
                and examples[pair.right].split == split
                for pair in reversals
            )
            for split in split_names
        },
        "cluster_counts_by_split": {
            split: len({example.question_id for example in examples if example.split == split})
            for split in split_names
        },
        "formal_gate": {
            "pair_minimum_each_split": 30,
            "cluster_minimum_each_split": 20,
            "status": "OPEN" if all(
                sum(bool(pair.confirmed) and examples[pair.left].split == split and examples[pair.right].split == split for pair in reversals) >= 30
                and len({example.question_id for example in examples if example.split == split}) >= 20
                for split in gate_splits
            ) else "CLOSED_underpowered",
            "splits": list(gate_splits),
        },
    }
    return dataset, audit


__all__ = [
    "TRACK_ORACLE_STATE",
    "TRACK_RAW_EVIDENCE",
    "TIER1_REWARD_COMPONENT_NAMES",
    "EXTENDED_MECHANISM_FAMILIES",
    "EXTENDED_SPLITS",
    "V04_EXTENDED_FAMILIES",
    "V04_EXTENDED_EXPLORATION_SEEDS",
    "V04_EXTENDED_CONFIRMATION_SEEDS",
    "V04_FORMAL_SCHEMA",
    "FORMAL_FINAL_ID_FAMILIES",
    "FORMAL_FINAL_OOD_FAMILIES",
    "FORMAL_SPLITS",
    "V04ExtendedQuestion",
    "V04ExtendedBenchmark",
    "V04FormalFinalBenchmark",
    "build_tier1_v04_extended_benchmark",
    "build_tier1_v04_formal_final_benchmark",
    "build_pre_action_raw_observation",
    "pre_action_observation_hash",
    "audit_counterfactual_raw_observation_leakage",
    "counterfactual_raw_observation_audit",
    "collect_tier1_v04_extended",
]

"""Independent P2.3 promotion-v2 benchmark generator.

The P2.2 promotion is consumed and is never mutated in place.  This module
creates a larger diagnostic benchmark with a fresh arithmetic recipe, balanced
train/tune/promotion clusters, and the same evaluator/executor used by P2.1/2.2.
It deliberately exposes no new policy-visible labels; target actions remain
evaluator-audit metadata only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..schemas import Protocol, ResearchAction, WorldSpec
from .tier1_v04_extended import (
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_EXTENDED_FAMILIES,
    V04_EXTENDED_EXPLORATION_SEEDS,
    V04ExtendedQuestion,
    WORLD_KINDS,
)


P23_SCHEMA = "pesco_tier1_benchmark_p2.3_promotion_v2"
P23_GENERATOR_VERSION = "pesco_p23_promotion_v2_generator_20260827_r1"
P23_SPLITS: Tuple[str, ...] = ("train", "tune", "promotion")
P23_COUNTS = {"train": 160, "tune": 64, "promotion": 80}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _split_for(family_index: int, ordinal: int) -> str:
    # 20/8/10 questions per family gives 160/64/80 clusters and keeps every
    # mechanism represented in every split.
    local = int(ordinal) % 38
    if local < 20:
        return "train"
    if local < 28:
        return "tune"
    return "promotion"


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


def _world_parameters(family: str, family_index: int, kind: str, variant: int, kind_index: int) -> dict[str, Any]:
    """Fresh v2 recipe; no v0.4/P2.1 latent arithmetic is reused."""

    phase = (17 * (family_index + 1) + 29 * int(variant) + 11 * int(kind_index)) % 97
    jitter = (phase - 48) / 7000.0
    odd = int(variant) % 2 == 1
    if kind == "supported":
        true_a, true_b, noise, samples = 0.155 + 0.004 * ((variant + family_index) % 7), 0.018 + jitter, 0.040 + 0.001 * (variant % 5), 112 + (variant % 9)
    elif kind == "refuted":
        true_a, true_b, noise, samples = -0.105 - 0.004 * ((variant + family_index) % 5), 0.205 if odd else -0.095, 0.048 + 0.001 * ((variant + 2 * family_index) % 4), 104 + (variant % 11)
    elif kind == "insufficient":
        true_a, true_b, noise, samples = 0.070 + jitter, 0.205 if odd else 0.025, 0.315 + 0.012 * ((variant + family_index) % 4), 22 + (variant % 5)
    else:
        true_a, true_b, noise, samples = 0.005 + jitter, 0.205 if odd else 0.075, 0.060 + 0.002 * (variant % 4), 108 + (variant % 8)
    leakage = family in {"group_leakage", "leakage_metric_mismatch"} and kind == "invalid"
    confounding = family in {"causal_confounding", "confounding_underpower"} and kind == "invalid"
    metric_mismatch = family in {"subgroup_metric_mismatch", "leakage_metric_mismatch"} and kind == "invalid"
    protocol_invalid = family in {"low_sample_variance", "protocol_drift", "replication_instability", "confounding_underpower"} and kind == "invalid"
    if family == "replication_instability" and kind in {"insufficient", "invalid"}:
        noise = 0.28 + 0.025 * ((variant + family_index) % 5)
        samples = 20 + (variant % 5)
    if family == "confounding_underpower" and kind in {"insufficient", "invalid"}:
        noise = 0.30 + 0.01 * (variant % 4)
        samples = 20 + (variant % 5)
    if family == "protocol_drift" and kind == "invalid":
        true_b = 0.205 if odd else 0.080
    return {
        "true_effect_a": float(true_a),
        "true_effect_b": float(true_b),
        "noise_scale": float(max(0.012, noise)),
        "initial_samples": int(max(10, samples)),
        "leakage": bool(leakage),
        "confounding": bool(confounding),
        "metric_mismatch": bool(metric_mismatch),
        "protocol_invalid": bool(protocol_invalid),
    }


class P23PromotionV2Benchmark:
    """Balanced train/tune/promotion diagnostic benchmark for P2.3."""

    protocol_version = "pesco_v0_2"
    schema_version = P23_SCHEMA

    def __init__(self) -> None:
        questions: list[V04ExtendedQuestion] = []
        for family_index, family in enumerate(V04_EXTENDED_FAMILIES):
            for variant in range(1, 39):
                local = variant - 1
                split = _split_for(family_index, local)
                qid = f"p23_v2_{family}_{variant:02d}"
                worlds: list[WorldSpec] = []
                targets: dict[str, ResearchAction] = {}
                for kind_index, kind in enumerate(WORLD_KINDS):
                    params = _world_parameters(family, family_index, kind, variant, kind_index)
                    world_id = f"{qid}__{kind}"
                    world = WorldSpec(
                        world_id=world_id,
                        kind=kind,
                        true_effect_a=params["true_effect_a"],
                        true_effect_b=params["true_effect_b"],
                        noise_scale=params["noise_scale"],
                        initial_samples=params["initial_samples"],
                        leakage=params["leakage"],
                        confounding=params["confounding"],
                        metric_mismatch=params["metric_mismatch"],
                        protocol_invalid=params["protocol_invalid"],
                        seed_offset=900_000 + family_index * 50_000 + variant * 101 + kind_index,
                        question_family=family,
                    )
                    worlds.append(world)
                    targets[world_id] = _target_for(family, kind, variant)
                questions.append(V04ExtendedQuestion(
                    question_id=qid,
                    family=family,
                    variant=variant,
                    split=split,
                    seed_offset=900_000 + family_index * 50_000 + variant * 101,
                    description=f"P2.3 promotion-v2 independent recipe {P23_GENERATOR_VERSION}; family held evaluator-side.",
                    worlds=tuple(worlds),
                    target_actions=MappingProxyType(targets),
                ))
        self.questions = tuple(questions)
        counts = {split: sum(q.split == split for q in self.questions) for split in P23_SPLITS}
        if counts != P23_COUNTS:
            raise ValueError(f"unexpected P2.3 split counts: {counts}")

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    @property
    def split_names(self) -> Tuple[str, ...]:
        return P23_SPLITS

    @property
    def question_map(self) -> Mapping[str, V04ExtendedQuestion]:
        return MappingProxyType({question.question_id: question for question in self.questions})

    def question(self, question_id: str) -> V04ExtendedQuestion:
        return self.question_map[str(question_id)]

    def make_environment(self, question_id: str, protocol: Optional[Protocol] = None, budget: Optional[int] = None) -> Tier1TabularEnvironment:
        question = self.question(question_id)
        return Tier1TabularEnvironment(
            worlds=question.worlds,
            protocol=protocol or Protocol(protocol_version=self.protocol_version),
            budget=budget,
        )

    def manifest(self, *, include_hidden: bool = False, exploration_seeds: Optional[Sequence[int]] = None) -> dict[str, Any]:
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
                    "worlds": [asdict(world) for world in question.worlds],
                    "legacy_target_actions_audit_only": {key: value.value for key, value in question.target_actions.items()},
                })
            questions.append(item)
        payload = {
            "schema_version": self.schema_version,
            "generator_version": P23_GENERATOR_VERSION,
            "protocol_version": self.protocol_version,
            "question_count": len(self.questions),
            "world_count": len(self.worlds),
            "counts_by_split": {split: sum(q.split == split for q in self.questions) for split in P23_SPLITS},
            "mechanism_family_count": len(V04_EXTENDED_FAMILIES),
            "mechanism_families": list(V04_EXTENDED_FAMILIES) if include_hidden else [],
            "exploration_seeds": list(exploration_seeds or V04_EXTENDED_EXPLORATION_SEEDS),
            "confirmation_seeds": list(V04_EXTENDED_CONFIRMATION_SEEDS),
            "questions": questions,
            "dual_tracks": {
                TRACK_ORACLE_STATE: "structured state upper-bound diagnostic",
                TRACK_RAW_EVIDENCE: "public numeric evidence only; formal algorithm track",
            },
            "canonical_reversal_contract": {
                "top_candidate_only": True,
                "max_pairs_per_question": 1,
                "minimum_promotion_pairs": 30,
                "minimum_promotion_question_clusters": 20,
            },
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
        }
        payload["manifest_digest"] = _digest(payload)
        return payload


def build_tier1_p23_promotion_v2_benchmark() -> P23PromotionV2Benchmark:
    return P23PromotionV2Benchmark()


__all__ = [
    "P23_SCHEMA",
    "P23_GENERATOR_VERSION",
    "P23_SPLITS",
    "P23_COUNTS",
    "P23PromotionV2Benchmark",
    "build_tier1_p23_promotion_v2_benchmark",
]

"""Fresh, frozen P2.3 promotion-v3 benchmark definition.

The v3 profile is generated only after the P2.3.1 authenticity gate.  It has
40 final-ID clusters and 42 final-OOD clusters; the OOD side uses three whole
mechanism families that do not occur in the consumed P2.3 diagnostic data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..schemas import Protocol, ResearchAction, WorldSpec
from .tier1_v04_extended import V04ExtendedQuestion, WORLD_KINDS, V04_EXTENDED_FAMILIES


V3_SCHEMA = "pesco_tier1_benchmark_p2.3_promotion_v3"
V3_GENERATOR_VERSION = "pesco_p23_promotion_v3_generator_20260828_r3"
V3_ID_FAMILIES = ("group_leakage", "causal_confounding", "low_sample_variance", "subgroup_metric_mismatch")
V3_OOD_FAMILIES = ("heterogeneous_noise", "nonlinear_response", "measurement_shift")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _target(family: str, kind: str, variant: int) -> ResearchAction:
    if kind == "supported":
        return ResearchAction.CONTINUE
    if family in {"group_leakage", "measurement_shift"}:
        return {"refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.REPAIR}[kind]
    if family in {"causal_confounding"}:
        return {"refuted": ResearchAction.CONTINUE if variant % 2 == 0 else ResearchAction.SWITCH, "insufficient": ResearchAction.SWITCH, "invalid": ResearchAction.REPAIR}[kind]
    if family in {"low_sample_variance", "heterogeneous_noise"}:
        return {"refuted": ResearchAction.CONTINUE, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.SAMPLE}[kind]
    return {"refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SWITCH, "invalid": ResearchAction.SWITCH}[kind]


def _params(family: str, variant: int, kind: str) -> dict[str, Any]:
    phase = (31 * variant + 17 * (V3_ID_FAMILIES + V3_OOD_FAMILIES).index(family) + 7 * WORLD_KINDS.index(kind)) % 113
    jitter = (phase - 56) / 9000.0
    odd = variant % 2 == 1
    if kind == "supported":
        true_a, true_b, noise, samples = 0.16 + 0.003 * (variant % 9), 0.015 + jitter, 0.035 + 0.001 * (variant % 4), 128 + variant % 13
    elif kind == "refuted":
        true_a, true_b, noise, samples = -0.11 - 0.003 * (variant % 7), 0.21 if odd else -0.10, 0.043 + 0.001 * (variant % 5), 118 + variant % 11
    elif kind == "insufficient":
        true_a, true_b, noise, samples = 0.065 + jitter, 0.205 if odd else 0.028, 0.32 + 0.009 * (variant % 5), 20 + variant % 7
    else:
        true_a, true_b, noise, samples = 0.004 + jitter, 0.21 if odd else 0.075, 0.055 + 0.002 * (variant % 5), 112 + variant % 11
    leakage = family in {"group_leakage", "measurement_shift"} and kind == "invalid"
    confounding = family == "causal_confounding" and kind == "invalid"
    metric_mismatch = family in {"subgroup_metric_mismatch", "nonlinear_response"} and kind == "invalid"
    protocol_invalid = family in {"low_sample_variance", "heterogeneous_noise"} and kind == "invalid"
    if family == "heterogeneous_noise" and kind in {"insufficient", "invalid"}:
        noise, samples = 0.34 + 0.01 * (variant % 4), 20 + variant % 7
    if family == "nonlinear_response" and kind == "invalid":
        true_b = 0.21 if odd else 0.09
    if family == "measurement_shift" and kind == "invalid":
        true_b = 0.19 if odd else 0.07
    return {
        "true_effect_a": float(true_a), "true_effect_b": float(true_b), "noise_scale": float(max(0.012, noise)),
        "initial_samples": int(max(10, samples)), "leakage": leakage, "confounding": confounding,
        "metric_mismatch": metric_mismatch, "protocol_invalid": protocol_invalid,
    }


class P23PromotionV3Benchmark:
    protocol_version = "pesco_v0_2"
    schema_version = V3_SCHEMA
    split_names = ("final_id", "final_ood")

    def __init__(self) -> None:
        questions = []
        ordinal = 0
        registry = [(family, "final_id", variant) for family in V3_ID_FAMILIES for variant in range(1, 14)]
        registry += [(family, "final_ood", variant) for family in V3_OOD_FAMILIES for variant in range(1, 25)]
        for family, split, variant in registry:
            ordinal += 1
            qid = f"p23_v3_{split}_{ordinal:03d}"
            worlds = []
            targets = {}
            family_index = (V3_ID_FAMILIES + V3_OOD_FAMILIES).index(family)
            for kind_index, kind in enumerate(WORLD_KINDS):
                values = _params(family, variant, kind)
                world_id = f"{qid}__{kind}"
                worlds.append(WorldSpec(
                    world_id=world_id, kind=kind, true_effect_a=values["true_effect_a"], true_effect_b=values["true_effect_b"],
                    noise_scale=values["noise_scale"], initial_samples=values["initial_samples"], leakage=values["leakage"],
                    confounding=values["confounding"], metric_mismatch=values["metric_mismatch"], protocol_invalid=values["protocol_invalid"],
                    seed_offset=1_700_000 + 80_000 * family_index + 1000 * variant + kind_index, question_family=family,
                ))
                targets[world_id] = _target(family, kind, variant)
            questions.append(V04ExtendedQuestion(
                question_id=qid, family=family, variant=variant, split=split,
                seed_offset=1_700_000 + 80_000 * family_index + 1000 * variant,
                description=f"Frozen promotion-v3 {split}; generator={V3_GENERATOR_VERSION}; latent family held out by construction.",
                worlds=tuple(worlds), target_actions=MappingProxyType(targets),
            ))
        self.questions = tuple(questions)
        if sum(q.split == "final_id" for q in self.questions) != 52 or sum(q.split == "final_ood" for q in self.questions) != 72:
            raise ValueError("promotion-v3 split contract broken")

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    @property
    def question_map(self) -> Mapping[str, V04ExtendedQuestion]:
        return MappingProxyType({question.question_id: question for question in self.questions})

    def question(self, question_id: str) -> V04ExtendedQuestion:
        return self.question_map[str(question_id)]

    def make_environment(self, question_id: str, protocol: Optional[Protocol] = None, budget: Optional[int] = None) -> Tier1TabularEnvironment:
        question = self.question(question_id)
        return Tier1TabularEnvironment(question.worlds, protocol or Protocol(protocol_version=self.protocol_version), budget=budget)

    def manifest(self, *, include_hidden: bool = False, exploration_seeds=None) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version, "generator_version": V3_GENERATOR_VERSION,
            "profile": "promotion_v3", "question_count": len(self.questions), "world_count": len(self.worlds),
            "counts_by_split": {split: sum(q.split == split for q in self.questions) for split in self.split_names},
            "id_families": list(V3_ID_FAMILIES) if include_hidden else [],
            "ood_families": list(V3_OOD_FAMILIES) if include_hidden else [],
            "ood_family_count": len(V3_OOD_FAMILIES), "whole_family_holdout": True,
            "latent_generator_signature": _digest({"version": V3_GENERATOR_VERSION, "id": V3_ID_FAMILIES, "ood": V3_OOD_FAMILIES, "recipe": "phase31_jitter_v1"}),
            "exploration_seeds": list(exploration_seeds or (17, 29, 41, 53, 67, 71, 83, 97)),
            "confirmation_seeds": [103, 107, 109, 113, 127, 131, 137, 139],
            "questions": [q.to_dict(include_hidden=include_hidden) for q in self.questions],
            "final_access": {"locked": True, "first_access_completed": False, "authorization_required": True, "formal_comparison_authorized": False},
            "diagnostic_only": False, "formal_comparison_authorized": False,
        }
        payload["manifest_digest"] = _digest(payload)
        return payload


def build_tier1_p23_promotion_v3_benchmark() -> P23PromotionV3Benchmark:
    return P23PromotionV3Benchmark()


__all__ = ["V3_SCHEMA", "V3_GENERATOR_VERSION", "V3_ID_FAMILIES", "V3_OOD_FAMILIES", "P23PromotionV3Benchmark", "build_tier1_p23_promotion_v3_benchmark"]

"""Fresh train/tune/promotion diagnostic generator for P2.1.

The consumed v0.4 formal artifact must not be used to tune the repaired evaluator
or the constrained objective.  This module derives a new evaluator-side latent
recipe with opaque IDs, new seed offsets, and a distinct split contract while
retaining the registered mechanism-family semantics needed for diagnostics.
It is intentionally not a final benchmark and never authorizes model scaling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..schemas import Protocol, WorldSpec
from .tier1_v04_extended import (
    V04_EXTENDED_FAMILIES,
    V04_EXTENDED_EXPLORATION_SEEDS,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04ExtendedQuestion,
    WORLD_KINDS,
    build_tier1_v04_extended_benchmark,
)


P21_SCHEMA = "pesco_tier1_benchmark_p2.1_fresh_diagnostic"
P21_GENERATOR_VERSION = "pesco_p21_latent_generator_20260826_r1"
P21_SPLITS: Tuple[str, ...] = ("train", "tune", "promotion")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def p21_latent_signature(world: WorldSpec) -> str:
    """Signature excluding IDs and seed offsets, for overlap audits."""

    payload = {
        "kind": world.kind,
        "true_effect_a": round(float(world.true_effect_a), 12),
        "true_effect_b": round(float(world.true_effect_b), 12),
        "noise_scale": round(float(world.noise_scale), 12),
        "initial_samples": int(world.initial_samples),
        "leakage": bool(world.leakage),
        "confounding": bool(world.confounding),
        "metric_mismatch": bool(world.metric_mismatch),
        "protocol_invalid": bool(world.protocol_invalid),
        "question_family": str(world.question_family),
    }
    return _digest(payload)


def _split_for(old_split: str) -> str:
    return {"train": "train", "dev": "tune", "diagnostic_ood": "promotion"}[str(old_split)]


def _fresh_world(base: WorldSpec, family_index: int, variant: int, kind_index: int) -> WorldSpec:
    # The recipe is deliberately not the v0.4 arithmetic grid.  A deterministic
    # phase derived from family/variant/kind makes every latent output auditable
    # without consulting the old `_family_flags` generator.
    phase = ((family_index + 3) * 37 + int(variant) * 19 + int(kind_index) * 11) % 101
    signed_phase = (phase - 50) / 10000.0
    effect_a = 0.73 * float(base.true_effect_a) + signed_phase + 0.0021 * (int(variant) % 5)
    effect_b = 1.17 * float(base.true_effect_b) - signed_phase / 2.0 + 0.0017 * ((int(variant) + kind_index) % 4)
    noise = max(0.012, 0.86 * float(base.noise_scale) + 0.0013 * ((family_index + variant + kind_index) % 6))
    samples = max(10, int(base.initial_samples) + 4 + ((family_index * 5 + variant * 3 + kind_index) % 13))
    return replace(
        base,
        world_id=f"p21_{family_index:02d}_{variant:02d}_{base.kind}",
        true_effect_a=float(effect_a),
        true_effect_b=float(effect_b),
        noise_scale=float(noise),
        initial_samples=int(samples),
        seed_offset=700_000 + family_index * 10_000 + variant * 100 + kind_index,
        question_family=str(base.question_family),
    )


@dataclass(frozen=True)
class P21DiagnosticBenchmark:
    questions: Tuple[V04ExtendedQuestion, ...]
    protocol_version: str = "pesco_v0_2"
    schema_version: str = P21_SCHEMA

    def __post_init__(self) -> None:
        if len(self.questions) != 64:
            raise ValueError("P2.1 diagnostic requires 64 questions")
        if {question.split for question in self.questions} != set(P21_SPLITS):
            raise ValueError("P2.1 diagnostic must expose train/tune/promotion splits")
        counts = {split: sum(question.split == split for question in self.questions) for split in P21_SPLITS}
        if counts != {"train": 24, "tune": 20, "promotion": 20}:
            raise ValueError(f"unexpected P2.1 split counts: {counts}")

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    @property
    def split_names(self) -> Tuple[str, ...]:
        return P21_SPLITS

    @property
    def question_map(self) -> Mapping[str, V04ExtendedQuestion]:
        return MappingProxyType({question.question_id: question for question in self.questions})

    def question(self, question_id: str) -> V04ExtendedQuestion:
        return self.question_map[str(question_id)]

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

    def manifest(self, *, include_hidden: bool = False, exploration_seeds: Optional[Sequence[int]] = None) -> dict[str, Any]:
        questions = []
        for question in self.questions:
            item: dict[str, Any] = {
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
                    "latent_signatures": [p21_latent_signature(world) for world in question.worlds],
                })
            questions.append(item)
        payload = {
            "schema_version": self.schema_version,
            "generator_version": P21_GENERATOR_VERSION,
            "protocol_version": self.protocol_version,
            "question_count": len(self.questions),
            "world_count": len(self.worlds),
            "mechanism_families": list(V04_EXTENDED_FAMILIES),
            "counts_by_split": {split: sum(question.split == split for question in self.questions) for split in P21_SPLITS},
            "exploration_seeds": list(exploration_seeds or V04_EXTENDED_EXPLORATION_SEEDS),
            "confirmation_seeds": list(V04_EXTENDED_CONFIRMATION_SEEDS),
            "questions": questions,
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
        }
        payload["manifest_digest"] = _digest(payload)
        return payload


def build_tier1_p21_diagnostic_benchmark() -> P21DiagnosticBenchmark:
    base = build_tier1_v04_extended_benchmark()
    questions: list[V04ExtendedQuestion] = []
    for family_index, question in enumerate(base.questions):
        worlds: list[WorldSpec] = []
        targets: dict[str, Any] = {}
        for kind_index, world in enumerate(question.worlds):
            fresh = _fresh_world(world, family_index=V04_EXTENDED_FAMILIES.index(question.family), variant=question.variant, kind_index=kind_index)
            worlds.append(fresh)
            targets[fresh.world_id] = question.target_actions[world.world_id]
        fresh_question = replace(
            question,
            question_id=f"t1_p21_{question.family}_{int(question.variant):02d}",
            split=_split_for(question.split),
            seed_offset=800_000 + family_index * 100 + int(question.variant),
            description=f"P2.1 fresh diagnostic recipe {P21_GENERATOR_VERSION}; family {question.family} evaluator-only.",
            worlds=tuple(worlds),
            target_actions=MappingProxyType(targets),
        )
        questions.append(fresh_question)
    return P21DiagnosticBenchmark(tuple(questions))


__all__ = [
    "P21_SCHEMA",
    "P21_GENERATOR_VERSION",
    "P21_SPLITS",
    "P21DiagnosticBenchmark",
    "p21_latent_signature",
    "build_tier1_p21_diagnostic_benchmark",
]

"""Fresh P2.3.3 diagnostic benchmark with independent OOD mechanisms.

This dataset is diagnostic-only and is generated with a new arithmetic recipe and
generator signature.  It keeps all policy-visible observations numeric while the
mechanism flags remain evaluator-owned.  Ten families (five new OOD families) are
balanced across train/tune/promotion question clusters.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..schemas import Protocol, ResearchAction, WorldSpec
from .tier1_v04_extended import (
    V04ExtendedQuestion,
    WORLD_KINDS,
    V04_EXTENDED_EXPLORATION_SEEDS,
    V04_EXTENDED_CONFIRMATION_SEEDS,
)

P233_SCHEMA = "pesco_tier1_benchmark_p2.3.3_diagnostic"
P233_GENERATOR_VERSION = "pesco_p233_diagnostic_generator_20260828_r2"
P233_FAMILIES = (
    "group_leakage", "causal_confounding", "low_sample_variance",
    "subgroup_metric_mismatch", "protocol_drift", "heteroscedastic_noise",
    "nonlinear_response", "measurement_shift", "missing_not_at_random",
    "intervention_noncompliance",
)
P233_SPLITS = ("train", "tune", "promotion")
P233_VARIANTS_PER_FAMILY = 180
P233_COUNTS = {"train": 1000, "tune": 400, "promotion": 400}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _split(local: int) -> str:
    local = int(local) % P233_VARIANTS_PER_FAMILY
    return "train" if local < 100 else "tune" if local < 140 else "promotion"


def _target(family: str, kind: str, variant: int) -> ResearchAction:
    if kind == "supported":
        return ResearchAction.CONTINUE
    if kind == "refuted":
        return ResearchAction.SWITCH
    if kind == "insufficient":
        return ResearchAction.SAMPLE
    if family in {"group_leakage", "causal_confounding", "protocol_drift"}:
        return ResearchAction.REPAIR
    if family in {"low_sample_variance", "heteroscedastic_noise"}:
        return ResearchAction.SAMPLE
    return ResearchAction.REPAIR


def _params(family: str, fi: int, kind: str, variant: int, ki: int) -> dict[str, Any]:
    phase = (31 * (fi + 1) + 17 * variant + 7 * ki) % 113
    jitter = (phase - 56) / 9000.0
    odd = variant % 2 == 1
    if kind == "supported":
        a, b, noise, n = 0.17 + 0.003 * ((variant + fi) % 9), 0.015 + jitter, 0.042 + .001 * (variant % 4), 112 + variant % 7
    elif kind == "refuted":
        a, b, noise, n = -0.11 - 0.003 * ((variant + fi) % 7), 0.20 if odd else -0.09, 0.050 + .001 * (variant % 5), 108 + variant % 9
    elif kind == "insufficient":
        a, b, noise, n = 0.075 + jitter, 0.20 if odd else 0.025, 0.31 + .009 * ((variant + fi) % 5), 24 + variant % 6
    else:
        a, b, noise, n = 0.006 + jitter, 0.20 if odd else 0.07, 0.062 + .002 * (variant % 5), 110 + variant % 8
    flags = {
        "leakage": family == "group_leakage" and kind == "invalid",
        "confounding": family == "causal_confounding" and kind == "invalid",
        "metric_mismatch": family == "subgroup_metric_mismatch" and kind == "invalid",
        "protocol_invalid": family in {"low_sample_variance", "protocol_drift"} and kind == "invalid",
        "heteroscedastic_noise": family == "heteroscedastic_noise" and kind == "invalid",
        "nonlinear_response": family == "nonlinear_response" and kind == "invalid",
        "measurement_shift": family == "measurement_shift" and kind == "invalid",
        "missing_not_at_random": family == "missing_not_at_random" and kind == "invalid",
        "intervention_noncompliance": family == "intervention_noncompliance" and kind == "invalid",
    }
    if family in {"low_sample_variance", "heteroscedastic_noise", "missing_not_at_random"} and kind in {"insufficient", "invalid"}:
        n, noise = 22 + variant % 5, 0.30 + .01 * (variant % 4)
    return {"true_effect_a": float(a), "true_effect_b": float(b), "noise_scale": float(max(.014, noise)), "initial_samples": int(n), **flags}


class P233DiagnosticBenchmark:
    protocol_version = "pesco_v0_2"
    schema_version = P233_SCHEMA

    def __init__(self) -> None:
        questions = []
        for fi, family in enumerate(P233_FAMILIES):
            for variant in range(1, P233_VARIANTS_PER_FAMILY + 1):
                qid = f"p233_{family}_{variant:02d}"
                worlds = []
                targets = {}
                for ki, kind in enumerate(WORLD_KINDS):
                    p = _params(family, fi, kind, variant, ki)
                    wid = f"{qid}__{kind}"
                    worlds.append(WorldSpec(world_id=wid, kind=kind, question_family=family,
                        seed_offset=2_400_000 + fi * 70_000 + variant * 131 + ki,
                        true_effect_a=p.pop("true_effect_a"), true_effect_b=p.pop("true_effect_b"),
                        noise_scale=p.pop("noise_scale"), initial_samples=p.pop("initial_samples"), **p))
                    targets[wid] = _target(family, kind, variant)
                questions.append(V04ExtendedQuestion(
                    question_id=qid, family=family, variant=variant, split=_split(variant - 1),
                    seed_offset=2_400_000 + fi * 70_000 + variant * 131,
                    description=f"Fresh P2.3.3 diagnostic recipe {P233_GENERATOR_VERSION}",
                    worlds=tuple(worlds), target_actions=MappingProxyType(targets)))
        self.questions = tuple(questions)
        counts = {s: sum(q.split == s for q in self.questions) for s in P233_SPLITS}
        if counts != P233_COUNTS:
            raise ValueError(f"unexpected P2.3.3 split counts: {counts}")

    @property
    def worlds(self):
        return tuple(w for q in self.questions for w in q.worlds)

    @property
    def split_names(self):
        return P233_SPLITS

    @property
    def question_map(self) -> Mapping[str, V04ExtendedQuestion]:
        return MappingProxyType({q.question_id: q for q in self.questions})

    def question(self, question_id: str) -> V04ExtendedQuestion:
        return self.question_map[str(question_id)]

    def make_environment(self, question_id: str, protocol: Optional[Protocol] = None, budget: Optional[int] = None):
        return Tier1TabularEnvironment(self.question(question_id).worlds,
            protocol=protocol or Protocol(protocol_version=self.protocol_version), budget=budget)

    def manifest(self, *, include_hidden: bool = False, exploration_seeds=None) -> dict[str, Any]:
        rows = []
        for q in self.questions:
            row = {"question_id": q.question_id, "policy_question_id": q.policy_question_id,
                   "split": q.split, "variant": int(q.variant), "world_count": len(q.worlds)}
            if include_hidden:
                row.update({"family": q.family, "seed_offset": int(q.seed_offset),
                            "worlds": [asdict(w) for w in q.worlds],
                            "target_actions_audit_only": {k: v.value for k, v in q.target_actions.items()}})
            rows.append(row)
        payload = {"schema_version": self.schema_version, "generator_version": P233_GENERATOR_VERSION,
                   "protocol_version": self.protocol_version, "question_count": len(self.questions),
                   "world_count": len(self.worlds), "counts_by_split": P233_COUNTS,
                   "mechanism_family_count": len(P233_FAMILIES), "mechanism_families": list(P233_FAMILIES) if include_hidden else [],
                   "exploration_seeds": list(exploration_seeds or V04_EXTENDED_EXPLORATION_SEEDS),
                   "confirmation_seeds": list(V04_EXTENDED_CONFIRMATION_SEEDS), "questions": rows,
                   "canonical_reversal_contract": {"top_candidate_only": True, "max_pairs_per_question": 1,
                       "minimum_tune_pairs": 50, "minimum_promotion_pairs": 50},
                   "diagnostic_only": True, "formal_comparison_authorized": False}
        payload["manifest_digest"] = _digest(payload)
        return payload


def build_tier1_p233_diagnostic_benchmark() -> P233DiagnosticBenchmark:
    return P233DiagnosticBenchmark()


__all__ = ["P233_SCHEMA", "P233_GENERATOR_VERSION", "P233_FAMILIES", "P233_SPLITS", "P233_COUNTS", "P233DiagnosticBenchmark", "build_tier1_p233_diagnostic_benchmark"]

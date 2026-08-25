"""Sampling and labelling of semantically aligned paired worlds."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..schemas import ResearchAction, ReversalPair, WorldSpec
from ..environments.world_registry import WorldRegistry


def _seed(namespace: str, *parts: Any) -> int:
    payload = "|".join([namespace, *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    result = sum(float(value) for value in values) / len(values)
    if not math.isfinite(result):
        raise ValueError("values must be finite")
    return result


def _normal_radius(values: Sequence[float], confidence: float = 0.95) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    # NormalDist is intentionally imported lazily to keep this module CPU-only.
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return z * math.sqrt(max(0.0, variance / len(values)))


@dataclass(frozen=True)
class PairedWorldSample:
    """A training-side pair; IDs are omitted by :meth:`public_manifest`."""

    question_id: str
    world_a: WorldSpec
    world_b: WorldSpec
    pair_seed: int
    common_seeds: tuple[int, ...]

    @property
    def pair_id(self) -> str:
        return f"{self.question_id}:{self.world_a.world_id}:{self.world_b.world_id}"

    def public_manifest(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_family": self.world_a.question_family,
            "candidate_tools": ["run_experiment", "inspect_metrics", "register_hypothesis"],
            "common_seed_count": len(self.common_seeds),
            "world_ids_visible_to_policy": False,
        }


class PairedWorldSampler:
    """Build reproducible world pairs without exposing latent mechanisms."""

    def __init__(
        self,
        registry: WorldRegistry | Iterable[WorldSpec],
        *,
        seed: int = 0,
        question_family: Optional[str] = None,
    ) -> None:
        self.registry = registry if isinstance(registry, WorldRegistry) else WorldRegistry(registry)
        self.seed = int(seed)
        self.question_family = question_family

    def _eligible(self) -> list[WorldSpec]:
        worlds = [self.registry.get(world_id) for world_id in self.registry.ids()]
        if self.question_family is not None:
            worlds = [world for world in worlds if world.question_family == self.question_family]
        return worlds

    def enumerate_pairs(self) -> list[tuple[WorldSpec, WorldSpec]]:
        worlds = self._eligible()
        return [(left, right) for left, right in itertools.combinations(worlds, 2) if left.question_family == right.question_family]

    def pair_worlds(
        self,
        world_a: str | WorldSpec,
        world_b: str | WorldSpec,
        *,
        question_id: str = "rq_paired",
        common_seed_count: int = 4,
    ) -> PairedWorldSample:
        left = self.registry.get(world_a) if isinstance(world_a, str) else world_a
        right = self.registry.get(world_b) if isinstance(world_b, str) else world_b
        if left.world_id == right.world_id:
            raise ValueError("paired worlds must be distinct")
        if left.question_family != right.question_family:
            raise ValueError("paired worlds must share a question family")
        if common_seed_count < 1:
            raise ValueError("common_seed_count must be positive")
        pair_seed = _seed("pesco_pair", self.seed, question_id, left.world_id, right.world_id)
        rng = random.Random(pair_seed)
        seeds: list[int] = []
        while len(seeds) < common_seed_count:
            candidate = rng.randrange(1, 2**31 - 1)
            if candidate not in seeds:
                seeds.append(candidate)
        return PairedWorldSample(question_id, left, right, pair_seed, tuple(seeds))

    def sample(
        self,
        *,
        question_id: str = "rq_paired",
        count: int = 1,
        common_seed_count: int = 4,
        replace: bool = False,
    ) -> list[PairedWorldSample]:
        if count < 1:
            raise ValueError("count must be positive")
        pairs = self.enumerate_pairs()
        if not pairs:
            raise ValueError("no semantically aligned world pairs available")
        rng = random.Random(_seed("pesco_sample", self.seed, question_id))
        if not replace and count > len(pairs):
            raise ValueError("requested more pairs than available without replacement")
        selected = rng.choices(pairs, k=count) if replace else rng.sample(pairs, k=count)
        return [
            self.pair_worlds(left, right, question_id=question_id, common_seed_count=common_seed_count)
            for left, right in selected
        ]

    def paired_seed_grid(self, pair: PairedWorldSample, repeats: int = 1) -> tuple[tuple[int, ...], ...]:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        return tuple(
            tuple(_seed("pesco_repeat", pair.pair_seed, repeat, seed) for seed in pair.common_seeds)
            for repeat in range(repeats)
        )

    @staticmethod
    def align_candidate_options(
        options_a: Iterable[ResearchAction | str],
        options_b: Iterable[ResearchAction | str],
    ) -> tuple[ResearchAction, ...]:
        """Return the semantic intersection of candidate actions."""

        left = {ResearchAction(option) for option in options_a}
        right = {ResearchAction(option) for option in options_b}
        return tuple(sorted(left & right, key=lambda option: option.value))


def identify_confirmed_reversal(
    *,
    question_id: str,
    world_a: str,
    world_b: str,
    action_left: ResearchAction | str,
    action_right: ResearchAction | str,
    values_a_left: Sequence[float],
    values_a_right: Sequence[float],
    values_b_left: Sequence[float],
    values_b_right: Sequence[float],
    margin: float = 0.05,
    confidence: float = 0.95,
) -> ReversalPair:
    """Create a statistically gated two-action preference reversal label."""

    if margin < 0 or not math.isfinite(float(margin)):
        raise ValueError("margin must be finite and non-negative")
    left = ResearchAction(action_left)
    right = ResearchAction(action_right)
    if len(values_a_left) != len(values_a_right) or len(values_b_left) != len(values_b_right):
        raise ValueError("each world's paired action arrays must have equal lengths")
    if any(not math.isfinite(float(value)) for values in (values_a_left, values_a_right, values_b_left, values_b_right) for value in values):
        raise ValueError("paired action values must be finite")
    delta_a_values = [float(x) - float(y) for x, y in zip(values_a_left, values_a_right)]
    delta_b_values = [float(x) - float(y) for x, y in zip(values_b_left, values_b_right)]
    if not delta_a_values or not delta_b_values:
        raise ValueError("each world needs paired action values")
    if len(delta_a_values) != len(delta_b_values):
        raise ValueError("paired worlds need equal repeat counts")
    delta_a = _mean(delta_a_values)
    delta_b = _mean(delta_b_values)
    radius_a = _normal_radius(delta_a_values, confidence)
    radius_b = _normal_radius(delta_b_values, confidence)
    lcb_a = delta_a - radius_a
    ucb_b = delta_b + radius_b
    confirmed = lcb_a > float(margin) and ucb_b < -float(margin)
    return ReversalPair(
        question_id=question_id,
        world_a=world_a,
        world_b=world_b,
        action_left=left,
        action_right=right,
        delta_a=delta_a,
        delta_b=delta_b,
        lcb_a=lcb_a,
        ucb_b=ucb_b,
        confirmed=confirmed,
        double_difference=delta_a - delta_b,
    )


def construct_reversal_pairs(records: Iterable[Mapping[str, Any]], *, margin: float = 0.05) -> list[ReversalPair]:
    """Build reversal labels from compact record mappings.

    Each mapping must contain the fields accepted by
    :func:`identify_confirmed_reversal`; this adapter is useful when records are
    loaded from JSONL rather than in-memory rollout objects.
    """

    result: list[ReversalPair] = []
    for record in records:
        result.append(
            identify_confirmed_reversal(
                question_id=str(record["question_id"]),
                world_a=str(record["world_a"]),
                world_b=str(record["world_b"]),
                action_left=record["action_left"],
                action_right=record["action_right"],
                values_a_left=record["values_a_left"],
                values_a_right=record["values_a_right"],
                values_b_left=record["values_b_left"],
                values_b_right=record["values_b_right"],
                margin=margin,
                confidence=float(record.get("confidence", 0.95)),
            )
        )
    return result


def sample_paired_worlds(
    registry: WorldRegistry | Iterable[WorldSpec],
    *,
    seed: int = 0,
    question_id: str = "rq_paired",
    count: int = 1,
    common_seed_count: int = 4,
    replace: bool = False,
) -> list[PairedWorldSample]:
    """Functional wrapper around :class:`PairedWorldSampler.sample`."""

    return PairedWorldSampler(registry, seed=seed).sample(
        question_id=question_id,
        count=count,
        common_seed_count=common_seed_count,
        replace=replace,
    )


class PairedWorldPreferenceBuilder:
    """Plan-shaped façade for constructing aligned reversal training examples.

    The lower-level functions in this module remain the source of truth.  This class is
    intentionally a thin, JSON-friendly adapter so experiment runners can keep their
    world-record plumbing separate from the statistical gate.
    """

    def __init__(self, margin: float = 0.05, confidence: float = 0.95):
        self.margin = float(margin)
        self.confidence = float(confidence)
        if self.margin < 0.0 or not math.isfinite(self.margin):
            raise ValueError("margin must be finite and non-negative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")

    def align_candidate_options(
        self,
        world_records: Mapping[str, Iterable[ResearchAction | str]] | Sequence[Mapping[str, Any]],
    ) -> tuple[ResearchAction, ...]:
        if isinstance(world_records, Mapping):
            option_sets = list(world_records.values())
        else:
            option_sets = [record.get("options", ()) for record in world_records]
        if not option_sets:
            return ()
        aligned = {ResearchAction(option) for option in option_sets[0]}
        for options in option_sets[1:]:
            aligned &= {ResearchAction(option) for option in options}
        return tuple(sorted(aligned, key=lambda option: option.value))

    def identify_confirmed_reversals(self, records: Iterable[Mapping[str, Any]]) -> list[ReversalPair]:
        return construct_reversal_pairs(records, margin=self.margin)

    def build_training_pairs(self, reversals: Iterable[ReversalPair | Mapping[str, Any]]) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for reversal in reversals:
            record = reversal.to_dict() if isinstance(reversal, ReversalPair) else dict(reversal)
            confidence = record.get("paired_confidence", {})
            confirmed = bool(confidence.get("confirmed_reversal", record.get("confirmed", False)))
            if not confirmed:
                continue
            worlds = record.get("paired_worlds", [record.get("world_a"), record.get("world_b")])
            actions = record.get("candidate_options", [record.get("action_left"), record.get("action_right")])
            if len(worlds) != 2 or len(actions) != 2 or any(value is None for value in (*worlds, *actions)):
                raise ValueError("confirmed reversal must identify two worlds and two actions")
            pairs.append({
                "question_id": record.get("question_id", ""),
                "world_a": worlds[0],
                "world_b": worlds[1],
                "preferred_in_world_a": actions[0],
                "preferred_in_world_b": actions[1],
                "confirmed": True,
                "double_difference": record.get("double_difference"),
            })
        return pairs


build_reversal_pair = identify_confirmed_reversal


__all__ = [
    "PairedWorldSample",
    "PairedWorldSampler",
    "identify_confirmed_reversal",
    "construct_reversal_pairs",
    "sample_paired_worlds",
    "build_reversal_pair",
    "PairedWorldPreferenceBuilder",
]

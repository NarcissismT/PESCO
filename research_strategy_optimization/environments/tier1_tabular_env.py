"""Lightweight executable Tier-1 tabular/leakage environment.

It uses NumPy to generate grouped observations and evaluates a frozen, simple
method-A/method-B score.  The same public API as Tier 0 makes it possible to run the
verifier/branch/trainer stack on a less toy-like data-generation process.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ..schemas import ExperimentOutput, Protocol, ResearchAction, WorldSpec
from .tier0_simulator import Tier0ResearchEnvironment, TrustedVerifier, default_mvp_worlds


class Tier1TabularEnvironment(Tier0ResearchEnvironment):
    def __init__(self, worlds: Optional[Iterable[WorldSpec]] = None, protocol: Optional[Protocol] = None, budget: Optional[int] = None):
        super().__init__(worlds=worlds, protocol=protocol, budget=budget)

    def _simulate(self, method: str, option: ResearchAction, seeds: Sequence[int], confirmation: bool) -> ExperimentOutput:
        world = self.world
        latent = world.true_effect_a if method == "method_a" else world.true_effect_b
        observations = []
        for seed in seeds:
            rng = np.random.default_rng(int(seed) + world.seed_offset + (10000 if confirmation else 0))
            n = int(max(12, self._sample_size))
            groups = rng.integers(0, max(4, n // 8), size=n)
            confounder = rng.normal(size=n)
            treatment = (confounder + rng.normal(scale=0.75, size=n) > 0).astype(float)
            outcome = latent * treatment + 0.12 * confounder + rng.normal(scale=world.noise_scale, size=n)
            # A frozen difference-in-means estimator is intentionally transparent.
            estimate = float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
            if method == "method_b":
                # Alternative controls for the observed confounder.
                estimate = float(latent + rng.normal(scale=world.noise_scale / math.sqrt(n)))
            if world.leakage and not self._repaired and method == "method_a":
                # Leakage feature makes the surface estimate look excellent.
                estimate += 0.25
            observations.append(estimate)
        mean = float(np.mean(observations))
        if len(observations) > 1:
            se = float(np.std(observations, ddof=1) / math.sqrt(len(observations)))
        else:
            se = 0.0
        z = 1.96
        low, high = mean - z * se, mean + z * se
        signals = []
        if self._sample_size < 60:
            signals.append("sample_count_below_precision_target")
        if world.leakage and not self._repaired and method == "method_a":
            signals.append("split_overlap_diagnostic")
        if world.confounding and not self._repaired:
            signals.append("treatment_confounder_dependence")
        if self._repaired:
            signals.append("split_protocol_updated")
        if method == "method_b":
            signals.append("confounder_adjusted_alternative")
        return ExperimentOutput(
            action=option.value,
            method=method,
            effect_estimate=mean,
            ci_low=low,
            ci_high=high,
            sample_size=int(self._sample_size),
            seed_count=len(seeds),
            execution_cost=1.5 + 0.002 * self._sample_size,
            dataset_hash="sha256:" + hashlib.sha256(f"tier1:{world.world_id}:{self._sample_size}:{seeds}".encode()).hexdigest(),
            code_hash=self._hash(f"tier1-code:{method}"),
            split_hash=self._hash("split:group" if self._repaired else "split:random"),
            evaluator_hash=self._hash("trusted-evaluator-v1"),
            seeds=tuple(seeds),
            validity_signals=tuple(signals),
            hidden_world_id=world.world_id,
            latent_effect=latent,
            leakage=bool(world.leakage and not self._repaired and method == "method_a"),
            confounding=bool(world.confounding and not self._repaired),
            confirmation=confirmation,
        )


class Tier1ConfoundingEnvironment(Tier1TabularEnvironment):
    """Named adapter for the confounding task family in the plan."""


class Tier1LeakageEnvironment(Tier1TabularEnvironment):
    """Named adapter for the leakage task family in the plan."""

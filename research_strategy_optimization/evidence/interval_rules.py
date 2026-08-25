"""Confidence-interval and equivalence helpers used by the verifier."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Sequence, Tuple


def normal_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Return mean, two-sided normal CI and standard error.

    The synthetic environment uses this transparent approximation; Tier-1 adapters can
    replace it with a bootstrap or cluster-robust interval without changing the API.
    """

    if not values:
        raise ValueError("at least one value is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("values must be finite")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        se = 0.0
    else:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        se = math.sqrt(var / n)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return mean, mean - z * se, mean + z * se


def interval_inside_equivalence(ci: Sequence[float], delta: float) -> bool:
    return float(ci[0]) >= -abs(delta) and float(ci[1]) <= abs(delta)

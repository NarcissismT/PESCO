"""Preregistered statistical helpers for PESCO evaluations.

The research plan treats a question/mechanism family as the independent unit.  This
module therefore operates on already-aggregated paired observations and deliberately
does not pretend that seeds, worlds, or same-snapshot branches are independent.
It has no SciPy dependency so the Tier-0 reference loop remains runnable on a clean
CPU environment.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Optional, Sequence, Tuple

from .multiple_testing import holm_adjust as _canonical_holm_adjust


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def holm_adjust(p_values: Sequence[float]) -> Tuple[float, ...]:
    """Return Holm step-down adjusted p-values in the original order.

    NaNs, infinities, and values outside ``[0, 1]`` are rejected rather than silently
    changing the family-wise error decision.  The returned values are monotone in the
    sorted order and clipped to one.
    """

    # Keep one canonical implementation in ``multiple_testing``; the tuple return
    # here is convenient for immutable statistical result records.
    return tuple(_canonical_holm_adjust(p_values))


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[bool, ...]:
    """Return family-wise decisions after Holm correction."""

    alpha = _probability(alpha, "alpha")
    return tuple(value <= alpha for value in holm_adjust(p_values))


def _as_differences(
    differences: Optional[Sequence[float]] = None,
    *,
    left: Optional[Sequence[float]] = None,
    right: Optional[Sequence[float]] = None,
) -> Tuple[float, ...]:
    if differences is not None and (left is not None or right is not None):
        raise ValueError("provide differences or left/right, not both")
    if differences is None:
        if left is None or right is None:
            raise ValueError("provide differences or both left and right")
        if len(left) != len(right):
            raise ValueError("paired samples must have equal length")
        differences = [float(a) - float(b) for a, b in zip(left, right)]
    values = tuple(_finite(value, "difference") for value in differences)
    if not values:
        raise ValueError("at least one paired difference is required")
    return values


def paired_bootstrap_ci(
    differences: Optional[Sequence[float]] = None,
    *,
    left: Optional[Sequence[float]] = None,
    right: Optional[Sequence[float]] = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> Tuple[float, float, float]:
    """Estimate a paired mean difference and percentile bootstrap interval.

    ``left`` and ``right`` are matched at the question/mechanism level.  Passing a
    precomputed ``differences`` sequence is equivalent and useful after clustering.
    """

    values = _as_differences(differences, left=left, right=right)
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")
    confidence = _probability(confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    point = sum(values) / len(values)
    rng = random.Random(int(seed))
    boot = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(int(n_bootstrap))]
    boot.sort()
    tail = (1.0 - confidence) / 2.0
    low = boot[min(len(boot) - 1, max(0, int(math.floor(tail * (len(boot) - 1)))))]
    high_index = min(len(boot) - 1, max(0, int(math.ceil((1.0 - tail) * (len(boot) - 1)))))
    return float(point), float(low), float(boot[high_index])


def paired_sign_permutation_pvalue(
    differences: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 17,
    alternative: str = "two-sided",
) -> float:
    """Sign-flip permutation p-value for a paired mean difference.

    The observed absolute/one-sided mean is compared with random independent sign
    flips.  The observed arrangement is included in the Monte Carlo denominator via
    a ``+1`` correction, yielding a conservative finite-sample estimate.
    """

    values = _as_differences(differences)
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    alternative = str(alternative).lower()
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")
    observed = sum(values) / len(values)
    rng = random.Random(int(seed))
    exceed = 0
    for _ in range(int(n_resamples)):
        candidate = sum(value if rng.getrandbits(1) else -value for value in values) / len(values)
        if alternative == "two-sided":
            exceed += abs(candidate) >= abs(observed)
        elif alternative == "greater":
            exceed += candidate >= observed
        else:
            exceed += candidate <= observed
    return float((exceed + 1) / (int(n_resamples) + 1))


def paired_binary_required_n(
    p01: float,
    p10: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Approximate paired-binary sample size from plan §19.5.

    ``p01`` is the proportion where baseline fails and PESCO succeeds, while ``p10``
    is the reverse.  The approximation uses

    ``((z_(1-alpha/2) + z_power)^2 * (p01+p10)) / (p01-p10)^2``.

    The result is rounded up and at least one independent question is required.
    """

    p01 = _probability(p01, "p01")
    p10 = _probability(p10, "p10")
    if p01 + p10 > 1.0:
        raise ValueError("p01 + p10 cannot exceed one")
    alpha = _probability(alpha, "alpha")
    power = _probability(power, "power")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if power <= 0.0 or power >= 1.0:
        raise ValueError("power must lie strictly between zero and one")
    if p01 + p10 <= 0.0:
        return 1
    delta = abs(p01 - p10)
    if delta == 0.0:
        raise ValueError("p01 and p10 must differ for a finite sample-size estimate")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / (2.0 if two_sided else 1.0))
    z_power = NormalDist().inv_cdf(power)
    estimate = ((z_alpha + z_power) ** 2 * (p01 + p10)) / (delta**2)
    return max(1, int(math.ceil(estimate)))


def paired_binary_power(
    p01: float,
    p10: float,
    n: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Approximate power corresponding to :func:`paired_binary_required_n`."""

    p01 = _probability(p01, "p01")
    p10 = _probability(p10, "p10")
    if p01 + p10 > 1.0:
        raise ValueError("p01 + p10 cannot exceed one")
    alpha = _probability(alpha, "alpha")
    if int(n) < 1:
        raise ValueError("n must be positive")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    delta = abs(p01 - p10)
    if delta == 0.0 or p01 + p10 <= 0.0:
        return 0.0
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / (2.0 if two_sided else 1.0))
    z_effect = delta * math.sqrt(int(n) / (p01 + p10))
    # Under the normal approximation, power for a positive noncentrality is
    # P(Z > z_alpha-z_effect) plus the negligible opposite-tail contribution.
    normal = NormalDist()
    if two_sided:
        return float(normal.cdf(-z_alpha - z_effect) + 1.0 - normal.cdf(z_alpha - z_effect))
    return float(1.0 - normal.cdf(z_alpha - z_effect))


__all__ = [
    "holm_adjust",
    "holm_reject",
    "paired_bootstrap_ci",
    "paired_sign_permutation_pvalue",
    "paired_binary_required_n",
    "paired_binary_power",
]

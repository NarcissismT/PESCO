"""Small, preregistration-friendly practical-equivalence helpers.

These helpers do not replace a domain-specific TOST or cluster-robust analysis.  They
implement the conservative interval rule used by the CPU reference environment: an
effect is practically equivalent to zero only when the complete confidence interval is
inside ``[-margin, margin]``.  Intervals crossing either boundary remain unresolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


def _interval(confidence_interval: Sequence[float]) -> Tuple[float, float]:
    if len(confidence_interval) != 2:
        raise ValueError("confidence_interval must contain exactly two values")
    low, high = (float(confidence_interval[0]), float(confidence_interval[1]))
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("confidence interval must be finite")
    if low > high:
        raise ValueError("confidence interval must be ordered")
    return low, high


@dataclass(frozen=True)
class EquivalenceDecision:
    effect_estimate: float
    confidence_interval: Tuple[float, float]
    margin: float
    equivalent: bool
    reason: str


def interval_inside_equivalence(confidence_interval: Sequence[float], margin: float) -> bool:
    """Return whether the full interval lies in the preregistered equivalence band."""

    low, high = _interval(confidence_interval)
    margin = float(margin)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    return low >= -margin and high <= margin


def practical_equivalence(
    effect_estimate: float,
    confidence_interval: Sequence[float],
    margin: float,
) -> EquivalenceDecision:
    """Classify a result as equivalent, or explicitly unresolved."""

    effect = float(effect_estimate)
    if not math.isfinite(effect):
        raise ValueError("effect_estimate must be finite")
    ci = _interval(confidence_interval)
    margin = float(margin)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    equivalent = interval_inside_equivalence(ci, margin)
    reason = "interval_inside_equivalence_band" if equivalent else "interval_crosses_equivalence_boundary"
    return EquivalenceDecision(effect, ci, margin, equivalent, reason)


__all__ = ["EquivalenceDecision", "interval_inside_equivalence", "practical_equivalence"]

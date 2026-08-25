"""Strict proper scoring rules used for belief-shaping rewards."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _clip(probability: float, epsilon: float = 1e-3) -> float:
    if not math.isfinite(float(epsilon)) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, .5)")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and lie in [0, 1]")
    return min(1.0 - epsilon, max(epsilon, probability))


def log_score(probability: float, outcome: int, epsilon: float = 1e-3) -> float:
    """Binary logarithmic score; higher is better."""

    p = _clip(probability, epsilon)
    if int(outcome) not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    return math.log(p if outcome else 1.0 - p)


def belief_delta(before: float, after: float, outcome: int, epsilon: float = 1e-3) -> float:
    return log_score(after, outcome, epsilon) - log_score(before, outcome, epsilon)


def multiclass_log_score(probabilities: Mapping[str, float], truth: str, epsilon: float = 1e-3) -> float:
    """Multiclass logarithmic score for a validated probability distribution.

    Unlike the binary helper's scalar input, a multiclass mapping must be non-empty,
    finite, non-negative, and sum to one (within a small numerical tolerance).  We
    clip only the selected class probability to avoid ``log(0)``; silently clipping or
    renormalising an invalid distribution would make belief-improvement rewards
    manipulable.
    """

    _validate_multiclass_distribution(probabilities, epsilon)
    if truth not in probabilities:
        raise KeyError(truth)
    p = max(float(epsilon), float(probabilities[truth]))
    return math.log(p)


def multiclass_belief_delta(
    before: Mapping[str, float], after: Mapping[str, float], truth: str, epsilon: float = 1e-3
) -> float:
    return multiclass_log_score(after, truth, epsilon) - multiclass_log_score(before, truth, epsilon)


def _validate_multiclass_distribution(probabilities: Mapping[str, float], epsilon: float) -> None:
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a non-empty mapping")
    if not math.isfinite(float(epsilon)) or not 0.0 < float(epsilon) < 0.5:
        raise ValueError("epsilon must be finite and lie in (0, .5)")
    values = []
    for label, probability in probabilities.items():
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"probability for {label!r} must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"probability for {label!r} must lie in [0, 1]")
        values.append(value)
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"probabilities must sum to one (got {total:.12g})")

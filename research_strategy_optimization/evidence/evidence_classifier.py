"""Frozen evidence-state rules.

The implementation follows the plan's precedence exactly: invalid design is checked
before interval/direction logic; a confidence interval crossing the practical-effect
threshold is insufficient, not a refutation.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from ..schemas import EvidenceState, Protocol
from .evidence_schema import EvidenceDecision, EvidenceFactors


def classify_evidence(
    validity_pass: bool,
    effect_estimate: float,
    confidence_interval: Sequence[float],
    protocol: Protocol,
    replicated: bool = False,
    invalid_reasons: Optional[Iterable[str]] = None,
) -> EvidenceDecision:
    """Classify a positive-effect hypothesis using a pre-registered interval rule."""

    if len(confidence_interval) != 2:
        raise ValueError("confidence_interval must contain exactly two values")
    low, high = float(confidence_interval[0]), float(confidence_interval[1])
    if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(float(effect_estimate))):
        raise ValueError("effect estimate and confidence interval must be finite")
    if low > high:
        raise ValueError("confidence_interval must be ordered")
    reasons = tuple(invalid_reasons or ())
    if protocol.invalid_precedence and not validity_pass:
        return EvidenceDecision(
            EvidenceState.INVALID,
            EvidenceFactors(False, False, 0, replicated),
            "; ".join(reasons) or "trusted validity check failed",
        )

    # A valid interval is precise for the decision only when it lies wholly on one side
    # of the practically meaningful threshold.  Crossing delta_min is insufficient.
    if low > protocol.delta_min:
        return EvidenceDecision(
            EvidenceState.SUPPORTED,
            EvidenceFactors(True, True, 1, replicated),
            f"lower confidence bound {low:.4f} > delta_min {protocol.delta_min:.4f}",
        )
    if high < protocol.delta_min:
        return EvidenceDecision(
            EvidenceState.REFUTED,
            EvidenceFactors(True, True, -1, replicated),
            f"upper confidence bound {high:.4f} < delta_min {protocol.delta_min:.4f}",
        )
    return EvidenceDecision(
        EvidenceState.INSUFFICIENT,
        EvidenceFactors(True, False, 0, replicated),
        f"interval [{low:.4f}, {high:.4f}] crosses decision threshold",
    )


def evidence_factors(
    state: EvidenceState,
    validity_pass: bool = True,
    replicated: bool = False,
) -> EvidenceFactors:
    """Convert a frozen state label back to its factorised representation."""

    # Invalid precedence is a hard invariant, not merely a convention of the
    # interval classifier.  A caller cannot smuggle a Supported/Refuted factorisation
    # together with a failed validity check.
    if state is EvidenceState.INVALID or not validity_pass:
        return EvidenceFactors(False, False, 0, replicated)
    if state is EvidenceState.SUPPORTED:
        return EvidenceFactors(validity_pass, True, 1, replicated)
    if state is EvidenceState.REFUTED:
        return EvidenceFactors(validity_pass, True, -1, replicated)
    return EvidenceFactors(validity_pass, False, 0, replicated)

"""Validity-gated and budget-constrained objective helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ConstraintConfig:
    invalid_claim_tolerance: float = 0.0
    budget_limit: float = 6.0
    penalty_multiplier: float = 1.0


def validity_gate(validity_pass: bool, scientific_utility: float) -> float:
    return float(scientific_utility) if validity_pass else 0.0


def constrained_return(
    scientific_utility: float,
    invalid_claim: bool,
    cost: float,
    config: ConstraintConfig = ConstraintConfig(),
) -> float:
    violation = max(0.0, float(invalid_claim) - config.invalid_claim_tolerance)
    budget_violation = max(0.0, float(cost) - config.budget_limit)
    return float(scientific_utility) - config.penalty_multiplier * (violation + budget_violation)


def constraint_summary(records: Sequence[Mapping[str, object]]) -> dict:
    invalid = sum(bool(r.get("invalid_claim", False)) for r in records)
    total = len(records)
    cost = sum(float(r.get("cost", 0.0) or 0.0) for r in records)
    return {
        "invalid_claim_rate": invalid / total if total else 0.0,
        "total_cost": cost,
        "count": total,
    }


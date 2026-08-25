"""Fail-closed controls for preregistered sequential looks.

The Tier-0 pilot uses a fixed seed set and does not tune the stopping rule after seeing
the result.  This module provides a tiny reusable guard for Tier-1 runners: every
analysis look must be declared in advance and may not exceed the frozen horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class StoppingSchedule:
    max_looks: int
    allowed_looks: Tuple[int, ...]

    @classmethod
    def fixed_horizon(cls, max_looks: int) -> "StoppingSchedule":
        max_looks = int(max_looks)
        if max_looks < 1:
            raise ValueError("max_looks must be positive")
        return cls(max_looks=max_looks, allowed_looks=tuple(range(1, max_looks + 1)))

    @classmethod
    def preregistered(cls, looks: Iterable[int], max_looks: int | None = None) -> "StoppingSchedule":
        values = tuple(sorted({int(look) for look in looks}))
        if not values or any(look < 1 for look in values):
            raise ValueError("looks must contain positive integers")
        horizon = int(max_looks if max_looks is not None else max(values))
        if horizon < max(values):
            raise ValueError("max_looks cannot be smaller than a preregistered look")
        return cls(max_looks=horizon, allowed_looks=values)

    def allows(self, look: int) -> bool:
        return int(look) in self.allowed_looks and int(look) <= self.max_looks

    def require(self, look: int) -> None:
        if not self.allows(look):
            raise RuntimeError(
                f"analysis look {int(look)} is not in the preregistered stopping schedule "
                f"{self.allowed_looks}"
            )


def validate_fixed_horizon(look: int, max_looks: int) -> bool:
    """Convenience guard used by simple runners."""

    schedule = StoppingSchedule.fixed_horizon(max_looks)
    schedule.require(look)
    return True


__all__ = ["StoppingSchedule", "validate_fixed_horizon"]

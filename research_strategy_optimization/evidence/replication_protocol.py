"""Exploration/confirmation split and replication checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class ReplicationProtocol:
    exploration_seeds: Tuple[int, ...] = (17, 29, 41, 53)
    confirmation_seeds: Tuple[int, ...] = (103, 107, 109, 113)
    require_distinct_data_hash: bool = True

    def validate(self) -> None:
        if set(self.exploration_seeds) & set(self.confirmation_seeds):
            raise ValueError("exploration and confirmation seeds must be disjoint")
        if not self.exploration_seeds or not self.confirmation_seeds:
            raise ValueError("both seed sets must be non-empty")

    def is_independent(self, seeds: Sequence[int]) -> bool:
        return len(seeds) == len(set(seeds)) and not (set(seeds) & set(self.exploration_seeds))


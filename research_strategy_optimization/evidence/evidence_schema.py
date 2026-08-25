"""Evidence-related helper records."""

from dataclasses import dataclass
from typing import Tuple

from ..schemas import EvidenceState


@dataclass(frozen=True)
class EvidenceFactors:
    validity: bool
    precise: bool
    direction: int
    replicated: bool = False

    def as_tuple(self) -> Tuple[bool, bool, int, bool]:
        return self.validity, self.precise, self.direction, self.replicated


@dataclass(frozen=True)
class EvidenceDecision:
    state: EvidenceState
    factors: EvidenceFactors
    reason: str


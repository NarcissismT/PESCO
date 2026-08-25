"""Abstract environment/verifier interfaces from the research plan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

from ..schemas import Observation, ResearchAction, Verdict


class ResearchEnvironment(ABC):
    @abstractmethod
    def reset(self, question_id: str, world_id: str, seed: int = 0) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def visible_observation(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute_option(self, option: ResearchAction, seeds: Optional[Sequence[int]] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def remaining_budget(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def final_submission(self, claim: Mapping[str, Any]) -> Verdict:
        raise NotImplementedError


class TrustedScientificVerifier(ABC):
    @abstractmethod
    def assess_validity(self, trajectory: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def classify_evidence(self, trajectory: Any, protocol: Any) -> Verdict:
        raise NotImplementedError

    @abstractmethod
    def confirm_independently(self, candidate: Any) -> Verdict:
        raise NotImplementedError

    @abstractmethod
    def compute_scientific_utility(self, trajectory: Any) -> Mapping[str, float]:
        raise NotImplementedError

    @abstractmethod
    def produce_audit_record(self, trajectory: Any) -> Mapping[str, Any]:
        raise NotImplementedError


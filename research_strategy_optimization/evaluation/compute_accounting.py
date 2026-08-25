from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping, Sequence


@dataclass
class ComputeLedger:
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    environment_runs: int = 0
    confirmation_runs: int = 0
    tokens: int = 0
    teacher_calls: int = 0
    retries: int = 0

    def add(self, *, gpu_hours: float = 0.0, cpu_hours: float = 0.0, environment_runs: int = 0, confirmation_runs: int = 0, tokens: int = 0, teacher_calls: int = 0, retries: int = 0) -> None:
        floats = (gpu_hours, cpu_hours)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in floats):
            raise ValueError("compute hours must be finite and non-negative")
        counts = (environment_runs, confirmation_runs, tokens, teacher_calls, retries)
        if any(int(value) != value or int(value) < 0 for value in counts):
            raise ValueError("compute counts must be non-negative integers")
        self.gpu_hours += float(gpu_hours)
        self.cpu_hours += float(cpu_hours)
        self.environment_runs += int(environment_runs)
        self.confirmation_runs += int(confirmation_runs)
        self.tokens += int(tokens)
        self.teacher_calls += int(teacher_calls)
        self.retries += int(retries)

    def total_cost(
        self,
        cpu_weight: float = 1.0,
        env_weight: float = 0.01,
        token_weight: float = 1e-5,
        confirmation_weight: float = 0.01,
        teacher_weight: float = 0.1,
        retry_weight: float = 0.01,
    ) -> float:
        """Return the frozen pilot's normalized total cost.

        Confirmation, teacher, and retry calls are explicit terms rather than silently
        disappearing from the denominator.  The coefficients are configurable so a
        formal GPU/token budget can bind them to a preregistered accounting manifest.
        """

        return (
            self.gpu_hours
            + cpu_weight * self.cpu_hours
            + env_weight * self.environment_runs
            + token_weight * self.tokens
            + confirmation_weight * self.confirmation_runs
            + teacher_weight * self.teacher_calls
            + retry_weight * self.retries
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["total_cost"] = self.total_cost()
        result["cost_weights"] = {
            "cpu_hours": 1.0,
            "environment_runs": 0.01,
            "tokens": 1e-5,
            "confirmation_runs": 0.01,
            "teacher_calls": 0.1,
            "retries": 0.01,
        }
        return result

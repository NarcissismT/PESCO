"""Thin execution-layer adapter for high-level research options.

The plan separates a strategy policy from the tool/executor.  The CPU simulator already
implements the concrete tool call; this adapter records the boundary and makes it easy
to swap in a real executor later without changing branch or verifier code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from ..schemas import ResearchAction


@dataclass(frozen=True)
class ExecutionReceipt:
    action: ResearchAction
    output: Any
    execution_cost: float

    def to_dict(self) -> dict[str, Any]:
        output = self.output.public_dict() if hasattr(self.output, "public_dict") else self.output
        return {
            "action": self.action.value,
            "execution_cost": self.execution_cost,
            "output": output,
        }


class OptionExecutor:
    """Execute one or more registered high-level options against an environment."""

    def __init__(self, environment: Any):
        if not hasattr(environment, "execute_option"):
            raise TypeError("environment must expose execute_option")
        self.environment = environment

    def execute(
        self,
        option: ResearchAction | str,
        *,
        seeds: Optional[Sequence[int]] = None,
        confirmation: bool = False,
    ) -> ExecutionReceipt:
        action = ResearchAction(option)
        kwargs = {"seeds": seeds}
        if confirmation:
            kwargs["confirmation"] = True
        try:
            output = self.environment.execute_option(action, **kwargs)
        except TypeError as error:
            # Minimal third-party executors often implement only ``(option, seeds)``.
            # Retry without the optional confirmation keyword when that is the sole
            # incompatibility; do not mask arbitrary execution TypeErrors.
            if "confirmation" not in kwargs or "confirmation" not in str(error):
                raise
            output = self.environment.execute_option(action, seeds=seeds)
        cost = float(getattr(output, "execution_cost", 0.0))
        return ExecutionReceipt(action=action, output=output, execution_cost=cost)

    def execute_many(
        self,
        options: Iterable[ResearchAction | str],
        *,
        seeds: Optional[Sequence[int]] = None,
        confirmation: bool = False,
    ) -> list[ExecutionReceipt]:
        return [self.execute(option, seeds=seeds, confirmation=confirmation) for option in options]


__all__ = ["ExecutionReceipt", "OptionExecutor"]

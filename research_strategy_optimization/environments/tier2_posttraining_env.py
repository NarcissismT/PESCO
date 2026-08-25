"""Tier-2 integration seam.

The expensive LLM executor is intentionally not enabled by the CPU MVP.  This adapter
defines the contract and fails closed until a frozen model/executor/verifier bundle is
provided, matching the plan's scientific-hard-gate requirement.
"""

from __future__ import annotations

from typing import Any, Mapping

from .abstract_research_env import ResearchEnvironment


class Tier2PostTrainingEnvironment(ResearchEnvironment):
    world_id_hidden = True
    def __init__(self, bundle: Mapping[str, Any] | None = None):
        self.bundle = dict(bundle or {})
        self.authorized = bool(self.bundle.get("scientific_hard_gate_pass", False))

    def _require(self) -> None:
        if not self.authorized:
            raise RuntimeError("Tier-2 execution is NO-GO until scientific_hard_gate_pass is true")

    def _delegate(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call an injected frozen bundle method after the hard gate passes."""

        self._require()
        direct = self.bundle.get(name)
        executor = self.bundle.get("executor")
        target = direct if callable(direct) else getattr(executor, name, None)
        if not callable(target):
            raise RuntimeError(f"authorized Tier-2 bundle does not provide {name}()")
        return target(*args, **kwargs)

    def reset(self, question_id: str, world_id: str, seed: int = 0):
        return self._delegate("reset", question_id, world_id, seed)

    def visible_observation(self):
        return self._delegate("visible_observation")

    def snapshot(self):
        return self._delegate("snapshot")

    def restore(self, snapshot):
        return self._delegate("restore", snapshot)

    def execute_option(self, option, seeds=None):
        return self._delegate("execute_option", option, seeds=seeds)

    def remaining_budget(self):
        return self._delegate("remaining_budget")

    def final_submission(self, claim):
        return self._delegate("final_submission", claim)

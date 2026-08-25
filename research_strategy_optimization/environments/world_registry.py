"""World registry with deliberate policy-side identifier isolation."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Sequence

from ..schemas import WorldSpec


class WorldRegistry:
    def __init__(self, worlds: Iterable[WorldSpec]):
        self._worlds: Dict[str, WorldSpec] = {w.world_id: copy.deepcopy(w) for w in worlds}
        if len(self._worlds) == 0:
            raise ValueError("at least one world is required")

    def get(self, world_id: str) -> WorldSpec:
        if world_id not in self._worlds:
            raise KeyError(world_id)
        return copy.deepcopy(self._worlds[world_id])

    def ids(self) -> List[str]:
        return list(self._worlds)

    def public_manifest(self) -> List[dict]:
        # No latent parameters, labels, or IDs are exposed to a policy.  This manifest is
        # intended for a frozen audit file, not model input.
        return [
            {
                "question_family": w.question_family,
                "task_description": "Determine whether method A improves group-held-out performance.",
                "tools": ["run_experiment", "inspect_metrics", "register_hypothesis"],
            }
            for w in self._worlds.values()
        ]

    def mechanism_digest(self) -> str:
        payload = [w.__dict__ for w in self._worlds.values()]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


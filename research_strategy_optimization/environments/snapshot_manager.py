"""Deterministic, hashable environment snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class EnvironmentSnapshot:
    payload: Mapping[str, Any]
    digest: str

    def to_dict(self) -> dict:
        return {"payload": copy.deepcopy(dict(self.payload)), "digest": self.digest}


class SnapshotManager:
    @staticmethod
    def create(payload: Mapping[str, Any]) -> EnvironmentSnapshot:
        frozen = copy.deepcopy(dict(payload))
        return EnvironmentSnapshot(frozen, canonical_hash(frozen))

    @staticmethod
    def restore(snapshot: EnvironmentSnapshot) -> dict:
        if canonical_hash(snapshot.payload) != snapshot.digest:
            raise ValueError("snapshot digest mismatch")
        return copy.deepcopy(dict(snapshot.payload))

    @staticmethod
    def equivalent(first: EnvironmentSnapshot, second: EnvironmentSnapshot) -> bool:
        return first.digest == second.digest and first.payload == second.payload

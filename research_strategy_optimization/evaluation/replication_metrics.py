from __future__ import annotations

from typing import Mapping, Sequence


def replication_rate(records: Sequence[Mapping[str, object]]) -> float:
    entered = [r for r in records if bool(r.get("entered_confirmation", False))]
    return sum(bool(r.get("independent_confirmed", False)) for r in entered) / len(entered) if entered else 0.0


def false_discovery_rate(records: Sequence[Mapping[str, object]]) -> float:
    announced = [r for r in records if bool(r.get("new_path_verified", r.get("discovery_claim", False)))]
    failed = [r for r in announced if not bool(r.get("independent_confirmed", False))]
    return len(failed) / len(announced) if announced else 0.0


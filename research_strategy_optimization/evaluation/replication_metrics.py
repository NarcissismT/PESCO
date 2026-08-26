from __future__ import annotations

from typing import Mapping, Optional, Sequence


def _as_bool(value: object) -> Optional[bool]:
    """Parse serialized boolean values without treating ``"false"`` as true."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "none", "na", "n/a"}:
        return False
    return None


def _flag(record: Mapping[str, object], *names: str) -> Optional[bool]:
    for name in names:
        if name in record and record[name] is not None:
            return _as_bool(record[name])
    return None


def replication_rate(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    eligible = [
        r for r in records
        if _flag(r, "confirmation_eligible", "entered_confirmation") is True
    ]
    return sum(_flag(r, "independent_confirmed", "replication_passed", "independent_confirmation_passed") is True for r in eligible) / len(eligible) if eligible else None


def false_discovery_rate(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    announced = [r for r in records if _flag(r, "new_path_announced", "new_path_verified", "discovery_claim") is True]
    failed = [r for r in announced if _flag(r, "independent_confirmed", "replication_passed", "independent_confirmation_passed") is not True]
    return len(failed) / len(announced) if announced else None


def replication_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    eligible = [
        r for r in records
        if _flag(r, "confirmation_eligible", "entered_confirmation") is True
    ]
    return {
        "confirmation_eligible_n": len(eligible),
        "confirmation_passed_n": sum(_flag(r, "independent_confirmed", "replication_passed", "independent_confirmation_passed") is True for r in eligible),
    }

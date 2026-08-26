from __future__ import annotations

from typing import Mapping, Optional, Sequence


def _as_bool(value: object) -> Optional[bool]:
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


def _rate(records: Sequence[Mapping[str, object]], eligible_state: str, outcome_key: str) -> Optional[float]:
    candidates = [r for r in records if str(r.get("true_state", r.get("world_state", ""))).strip().lower() == eligible_state]
    if not candidates:
        return None
    return sum(_as_bool(r.get(outcome_key)) is True for r in candidates) / len(candidates)


def conditional_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Return auditable denominators for the four conditional strategy metrics."""

    states = [str(r.get("true_state", r.get("world_state", ""))).strip().lower() for r in records]
    return {
        "required_switch_n": sum(state == "refuted" for state in states),
        "invalid_repair_n": sum(state == "invalid" for state in states),
        "insufficient_handling_n": sum(state == "insufficient" for state in states),
        "confirmation_eligible_n": sum(_flag(r, "confirmation_eligible", "entered_confirmation") is True for r in records),
    }


def refutation_acceptance(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    return _rate(records, "refuted", "refutation_accept")


def underpower_handling(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    return _rate(records, "insufficient", "underpower_handled")


def invalid_repair_rate(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    return _rate(records, "invalid", "invalid_repaired")


def invalid_claim_rate(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    return _rate(records, "invalid", "invalid_claim")

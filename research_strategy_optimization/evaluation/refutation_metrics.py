from __future__ import annotations

from typing import Mapping, Sequence


def refutation_acceptance(records: Sequence[Mapping[str, object]]) -> float:
    candidates = [r for r in records if str(r.get("true_state", "")).lower() == "refuted"]
    return sum(bool(r.get("refutation_accept", False)) for r in candidates) / len(candidates) if candidates else 0.0


def underpower_handling(records: Sequence[Mapping[str, object]]) -> float:
    candidates = [r for r in records if str(r.get("true_state", "")).lower() == "insufficient"]
    return sum(bool(r.get("underpower_handled", False)) for r in candidates) / len(candidates) if candidates else 0.0


def invalid_repair_rate(records: Sequence[Mapping[str, object]]) -> float:
    candidates = [r for r in records if str(r.get("true_state", "")).lower() == "invalid"]
    return sum(bool(r.get("invalid_repaired", False)) for r in candidates) / len(candidates) if candidates else 0.0


def invalid_claim_rate(records: Sequence[Mapping[str, object]]) -> float:
    candidates = [r for r in records if str(r.get("true_state", "")).lower() == "invalid"]
    return sum(bool(r.get("invalid_claim", False)) for r in candidates) / len(candidates) if candidates else 0.0


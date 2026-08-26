from __future__ import annotations

from collections import defaultdict
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


def flip_accuracy(records: Sequence[Mapping[str, object]]) -> Optional[float]:
    """Pair-level FlipAcc over explicitly/structurally eligible pairs.

    A pair is eligible when marked by ``flip_eligible`` or when both Supported and
    Refuted members share a ``world_pair_id``.  Repeated annotations for one pair are
    collapsed to one outcome and must agree.
    """

    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for index, record in enumerate(records):
        explicit = _flag(record, "flip_eligible", "preference_reversal_eligible")
        truth = str(record.get("true_state", record.get("world_state", ""))).strip().lower()
        pair = record.get("world_pair_id", record.get("pair_id"))
        if explicit is False:
            continue
        if explicit is not True and truth not in {"supported", "refuted"}:
            continue
        if pair is None:
            key = f"__row_{index}"
        else:
            question = str(record.get("question_id", record.get("research_question_id", record.get("task_id", "all"))))
            split = str(record.get("split", record.get("evaluation_split", "all")))
            key = f"{question}::{split}::{pair}"
        groups[key].append(record)
    outcomes = []
    for rows in groups.values():
        states = {str(r.get("true_state", r.get("world_state", ""))).strip().lower() for r in rows}
        explicit = any(_flag(r, "flip_eligible", "preference_reversal_eligible") is True for r in rows)
        if not explicit and not {"supported", "refuted"}.issubset(states):
            continue
        values = [r.get("flip_correct") for r in rows if r.get("flip_correct") is not None]
        # Keep an eligible pair in the denominator even when an outcome is
        # omitted; omission cannot improve a conditional success rate.
        outcomes.append(bool(values) and all(_as_bool(value) is True for value in values))
    return sum(outcomes) / len(outcomes) if outcomes else None


def flip_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for index, record in enumerate(records):
        explicit = _flag(record, "flip_eligible", "preference_reversal_eligible")
        truth = str(record.get("true_state", record.get("world_state", ""))).strip().lower()
        if explicit is False or (explicit is not True and truth not in {"supported", "refuted"}):
            continue
        pair = record.get("world_pair_id", record.get("pair_id"))
        if pair is None:
            key = f"__row_{index}"
        else:
            question = str(record.get("question_id", record.get("research_question_id", record.get("task_id", "all"))))
            split = str(record.get("split", record.get("evaluation_split", "all")))
            key = f"{question}::{split}::{pair}"
        groups[key].append(record)

    eligible_groups: list[list[Mapping[str, object]]] = []
    for rows in groups.values():
        states = {str(r.get("true_state", r.get("world_state", ""))).strip().lower() for r in rows}
        explicit = any(_flag(r, "flip_eligible", "preference_reversal_eligible") is True for r in rows)
        if explicit or {"supported", "refuted"}.issubset(states):
            eligible_groups.append(rows)
    correct = 0
    for rows in eligible_groups:
        values = [r.get("flip_correct") for r in rows if r.get("flip_correct") is not None]
        if values and all(_as_bool(value) is True for value in values):
            correct += 1
    return {"flip_eligible_n": len(eligible_groups), "flip_correct_n": correct}


def paired_flip_accuracy(pairs: Sequence[Mapping[str, object]]) -> float:
    values = [p.get("confirmed_reversal", p.get("paired_confidence", {}).get("confirmed_reversal")) for p in pairs]
    return sum(bool(v) for v in values) / len(values) if values else 0.0

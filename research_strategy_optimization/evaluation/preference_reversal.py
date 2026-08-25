from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


def flip_accuracy(records: Sequence[Mapping[str, object]]) -> float:
    values = [r.get("flip_correct") for r in records if r.get("flip_correct") is not None]
    return sum(bool(v) for v in values) / len(values) if values else 0.0


def paired_flip_accuracy(pairs: Sequence[Mapping[str, object]]) -> float:
    values = [p.get("confirmed_reversal", p.get("paired_confidence", {}).get("confirmed_reversal")) for p in pairs]
    return sum(bool(v) for v in values) / len(values) if values else 0.0


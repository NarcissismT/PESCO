from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence


def validated_novel_path_rate(records: Sequence[Mapping[str, object]]) -> float:
    opportunities = [r for r in records if bool(r.get("discovery_opportunity", False))]
    return sum(bool(r.get("new_path_verified", False)) and bool(r.get("independent_confirmed", False)) for r in opportunities) / len(opportunities) if opportunities else 0.0


def method_family_entropy(records: Sequence[Mapping[str, object]]) -> float:
    families = [str(r.get("method_family", r.get("selected_action", "unknown"))) for r in records if bool(r.get("valid_claim", True))]
    if not families:
        return 0.0
    counts = Counter(families)
    n = len(families)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def effective_method_family_count(records: Sequence[Mapping[str, object]]) -> float:
    return math.exp(method_family_entropy(records))


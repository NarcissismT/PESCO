"""Question-cluster bootstrap utilities (no pandas dependency)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple


def clustered_bootstrap(
    records: Sequence[Mapping[str, object]],
    statistic: Callable[[Sequence[Mapping[str, object]]], float],
    n_bootstrap: int = 1000,
    seed: int = 17,
    cluster_key: str = "question_id",
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    clusters = defaultdict(list)
    for record in records:
        clusters[str(record.get(cluster_key, "unknown"))].append(record)
    if not clusters:
        return None, None, None
    groups = list(clusters.values())
    point = float(statistic(records))
    # A single independent cluster cannot support an uncertainty interval.  Returning
    # the point together with NA bounds is explicit and avoids the old misleading
    # zero-width [point, point] interval.
    if len(groups) < 2:
        return point, None, None
    rng = random.Random(seed)
    values = []
    for _ in range(max(1, n_bootstrap)):
        sample = []
        for _ in groups:
            sample.extend(rng.choice(groups))
        values.append(float(statistic(sample)))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return point, low, high


def paired_difference(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    statistic: Callable[[Sequence[Mapping[str, object]]], float],
) -> float:
    return statistic(left) - statistic(right)

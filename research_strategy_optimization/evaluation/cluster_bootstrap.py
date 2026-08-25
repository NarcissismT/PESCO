"""Question-cluster bootstrap utilities (no pandas dependency)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence, Tuple


def clustered_bootstrap(
    records: Sequence[Mapping[str, object]],
    statistic: Callable[[Sequence[Mapping[str, object]]], float],
    n_bootstrap: int = 1000,
    seed: int = 17,
    cluster_key: str = "question_id",
) -> Tuple[float, float, float]:
    clusters = defaultdict(list)
    for record in records:
        clusters[str(record.get(cluster_key, "unknown"))].append(record)
    if not clusters:
        return 0.0, 0.0, 0.0
    groups = list(clusters.values())
    rng = random.Random(seed)
    values = []
    for _ in range(max(1, n_bootstrap)):
        sample = []
        for _ in groups:
            sample.extend(rng.choice(groups))
        values.append(float(statistic(sample)))
    values.sort()
    point = float(statistic(records))
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return point, low, high


def paired_difference(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    statistic: Callable[[Sequence[Mapping[str, object]]], float],
) -> float:
    return statistic(left) - statistic(right)


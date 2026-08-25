"""Small, dependency-free Holm and Benjamini–Hochberg corrections."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = [float(p) for p in p_values]
    if any(not 0.0 <= p <= 1.0 for p in values):
        raise ValueError("p-values must lie in [0, 1]")
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    values = [float(p) for p in p_values]
    if any(not 0.0 <= p <= 1.0 for p in values):
        raise ValueError("p-values must lie in [0, 1]")
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [1.0] * n
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * n / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_permutation_pvalue(differences: Sequence[float], permutations: int = 2000, seed: int = 17) -> float:
    """Two-sided paired sign-flip permutation p-value."""

    import random

    values = [float(x) for x in differences]
    if not values:
        return 1.0
    observed = abs(sum(values) / len(values))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(max(1, int(permutations))):
        sampled = [x if rng.random() < 0.5 else -x for x in values]
        if abs(sum(sampled) / len(sampled)) >= observed:
            exceed += 1
    return (exceed + 1) / (max(1, int(permutations)) + 1)


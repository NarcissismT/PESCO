"""Same-state leave-one-out advantage estimation."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping, MutableSequence, Sequence


def leave_one_out_advantages(
    returns: Sequence[float], normalize: bool = False, detach_baseline: bool = True
) -> List[float]:
    """Return ``G_i - mean(G_j, j != i)`` for a same-state branch group.

    The baseline is detached conceptually: callers receive ordinary numbers and should
    not backpropagate through the baseline when using a tensor implementation.  For a
    one-branch group no counterfactual comparison exists, so the advantage is zero.
    """

    # Preserve autograd for a tensor sequence.  The baseline is detached by
    # default because it is a control variate, not a trainable return path.
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and returns and all(isinstance(value, torch.Tensor) for value in returns):
        values_tensor = torch.stack(tuple(returns))
        if values_tensor.ndim != 1:
            raise ValueError("tensor returns must be one-dimensional")
        if not bool(torch.isfinite(values_tensor).all()):
            raise ValueError("branch returns must be finite")
        n = int(values_tensor.numel())
        if n <= 1:
            return [value * 0.0 for value in values_tensor]
        total = values_tensor.sum()
        baselines = (total - values_tensor) / (n - 1)
        if detach_baseline:
            baselines = baselines.detach()
        advantages_tensor = values_tensor - baselines
        if normalize:
            mean = advantages_tensor.mean()
            scale = advantages_tensor.std(unbiased=False)
            advantages_tensor = torch.where(scale > 1e-12, (advantages_tensor - mean) / scale, torch.zeros_like(advantages_tensor))
        return list(advantages_tensor)

    values = []
    for value in returns:
        if isinstance(value, Mapping):
            if "utility" not in value:
                raise KeyError("branch mapping must contain utility")
            value = value["utility"]
        elif hasattr(value, "utility"):
            value = getattr(value, "utility")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("branch returns must be finite")
        values.append(value)
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    total = sum(values)
    advantages = [v - (total - v) / (n - 1) for v in values]
    if normalize:
        mean = sum(advantages) / n
        variance = sum((a - mean) ** 2 for a in advantages) / max(1, n - 1)
        scale = math.sqrt(variance)
        if scale > 1e-12:
            advantages = [(a - mean) / scale for a in advantages]
    return advantages


def leave_one_out_baselines(returns: Sequence[float]) -> List[float]:
    """Return the mean of all *other* branch returns for each branch."""

    values = [float(v) for v in returns]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("branch returns must be finite")
    if len(values) <= 1:
        return [0.0] * len(values)
    total = sum(values)
    return [(total - value) / (len(values) - 1) for value in values]


def assign_leave_one_out_advantages(
    branches: MutableSequence[Any], normalize: bool = False
) -> List[float]:
    """Compute LOO advantages and attach them to mutable branch records."""

    advantages = leave_one_out_advantages(branches, normalize=normalize)
    for branch, advantage in zip(branches, advantages):
        if isinstance(branch, dict):
            branch["advantage"] = advantage
        elif hasattr(branch, "advantage"):
            try:
                setattr(branch, "advantage", advantage)
            except (AttributeError, TypeError):
                pass
    return advantages


compute_leave_one_out_advantages = leave_one_out_advantages
loo_advantages = leave_one_out_advantages


def leave_one_out_from_records(records: Sequence[Mapping[str, float]], key: str = "utility") -> List[float]:
    return leave_one_out_advantages([float(record[key]) for record in records])

"""Differentiable loss for confirmed cross-world preference reversals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class PreferenceReversalExample:
    """Log-probability values for one aligned action pair."""

    world_a_left: Any
    world_a_right: Any
    world_b_left: Any
    world_b_right: Any
    confirmed: bool = True
    weight: float = 1.0


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device")


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only in minimal CPU installs
        return None
    return torch


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def log_probability_ratio(log_probability: Any, reference_log_probability: Any = 0.0) -> Any:
    """Compute ``log π - log π_ref`` without exponentiating probabilities."""

    if _is_tensor(log_probability) or _is_tensor(reference_log_probability):
        torch = _torch()
        result = log_probability - reference_log_probability
        if not bool(torch.isfinite(result).all()):
            raise ValueError("log probabilities must be finite")
        return result
    result = _finite_scalar(log_probability, "log_probability") - _finite_scalar(
        reference_log_probability, "reference_log_probability"
    )
    return result


def _softplus(value: Any) -> Any:
    if _is_tensor(value):
        torch = _torch()
        return torch.nn.functional.softplus(value)
    # Stable scalar softplus.
    value = _finite_scalar(value, "loss margin")
    return max(0.0, value) + math.log1p(math.exp(-abs(value)))


def reversal_margins(
    world_a_left: Any,
    world_a_right: Any,
    world_b_left: Any,
    world_b_right: Any,
    *,
    reference_a_left: Any = 0.0,
    reference_a_right: Any = 0.0,
    reference_b_left: Any = 0.0,
    reference_b_right: Any = 0.0,
) -> tuple[Any, Any]:
    """Return the desired margins for A(left>right) and B(right>left)."""

    a_left = log_probability_ratio(world_a_left, reference_a_left)
    a_right = log_probability_ratio(world_a_right, reference_a_right)
    b_left = log_probability_ratio(world_b_left, reference_b_left)
    b_right = log_probability_ratio(world_b_right, reference_b_right)
    return a_left - a_right, b_right - b_left


def preference_reversal_loss(
    world_a_left: Any,
    world_a_right: Any,
    world_b_left: Any,
    world_b_right: Any,
    *legacy: Any,
    beta: float = 1.0,
    reference_a_left: Any = 0.0,
    reference_a_right: Any = 0.0,
    reference_b_left: Any = 0.0,
    reference_b_right: Any = 0.0,
    confirmed: Any = True,
    weight: Any = 1.0,
    reduction: str = "mean",
) -> Any:
    """Compute the plan's two-term logistic reversal loss.

    ``world_a_left`` etc. are log probabilities (or log-ratios if reference
    values are omitted).  Unconfirmed pairs can be masked with ``confirmed``;
    this is preferable to silently training on statistically uncertain flips.
    """

    # Compatibility with the early tabular trainer API:
    # ``loss(logs_a, logs_b, left_label, right_label, beta)``.
    if isinstance(world_a_left, dict) and isinstance(world_a_right, dict) and isinstance(world_b_left, str) and isinstance(world_b_right, str):
        left_label, right_label = world_b_left, world_b_right
        if legacy:
            if len(legacy) > 1:
                raise TypeError("legacy reversal loss accepts at most one positional beta")
            beta = legacy[0]
        logs_a, logs_b = world_a_left, world_a_right
        try:
            world_a_left = logs_a[left_label]
            world_a_right = logs_a[right_label]
            world_b_left = logs_b[left_label]
            world_b_right = logs_b[right_label]
        except KeyError as error:
            raise KeyError(f"action label missing from tabular policy log ratios: {error}") from error
    elif legacy:
        raise TypeError("unexpected positional arguments after world log probabilities")
    beta = _finite_scalar(beta, "beta")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")
    margin_a, margin_b = reversal_margins(
        world_a_left,
        world_a_right,
        world_b_left,
        world_b_right,
        reference_a_left=reference_a_left,
        reference_a_right=reference_a_right,
        reference_b_left=reference_b_left,
        reference_b_right=reference_b_right,
    )
    torch = _torch()
    tensor_mode = _is_tensor(margin_a) or _is_tensor(margin_b)
    if tensor_mode:
        if not _is_tensor(margin_a):
            margin_a = torch.as_tensor(margin_a, device=margin_b.device, dtype=margin_b.dtype)
        if not _is_tensor(margin_b):
            margin_b = torch.as_tensor(margin_b, device=margin_a.device, dtype=margin_a.dtype)
        values = torch.stack((_softplus(-beta * margin_a), _softplus(-beta * margin_b)), dim=0)
        mask = torch.as_tensor(confirmed, device=values.device, dtype=values.dtype)
        if mask.ndim == 0:
            mask = mask.expand(values.shape[1:] if values.ndim > 1 else ())
        pair_weight = torch.as_tensor(weight, device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim - 1:
            mask = mask.unsqueeze(0)
        if values.ndim > 1:
            values = values * mask.unsqueeze(0) * pair_weight
        else:
            values = values * mask * pair_weight
        if reduction == "none":
            return values
        return values.sum() if reduction == "sum" else values.mean()
    values = [
        _softplus(-beta * margin_a),
        _softplus(-beta * margin_b),
    ]
    if isinstance(confirmed, (bool, int, float)) and not bool(confirmed):
        values = [0.0, 0.0]
    scalar_weight = _finite_scalar(weight, "weight")
    values = [scalar_weight * value for value in values]
    if reduction == "none":
        return values
    return sum(values) if reduction == "sum" else sum(values) / len(values)


def batched_preference_reversal_loss(
    examples: Sequence[PreferenceReversalExample], *, beta: float = 1.0, reduction: str = "mean"
) -> Any:
    """Aggregate a batch of examples while preserving autograd when tensors are used."""

    if not examples:
        raise ValueError("at least one reversal example is required")
    values = [
        preference_reversal_loss(
            example.world_a_left,
            example.world_a_right,
            example.world_b_left,
            example.world_b_right,
            beta=beta,
            confirmed=example.confirmed,
            weight=example.weight,
            reduction="mean",
        )
        for example in examples
    ]
    if any(_is_tensor(value) for value in values):
        torch = _torch()
        values = [value if _is_tensor(value) else torch.as_tensor(value, dtype=values[0].dtype, device=values[0].device) for value in values]
        stacked = torch.stack(values)
        if reduction == "none":
            return stacked
        if reduction == "sum":
            return stacked.sum()
        if reduction != "mean":
            raise ValueError("reduction must be mean, sum, or none")
        return stacked.mean()
    if reduction == "none":
        return values
    if reduction == "sum":
        return sum(values)
    if reduction != "mean":
        raise ValueError("reduction must be mean, sum, or none")
    return sum(values) / len(values)


def reversal_accuracy(
    world_a_left: Sequence[float],
    world_a_right: Sequence[float],
    world_b_left: Sequence[float],
    world_b_right: Sequence[float],
) -> float:
    """Diagnostic fraction satisfying left>right in A and right>left in B."""

    if not (len(world_a_left) == len(world_a_right) == len(world_b_left) == len(world_b_right)):
        raise ValueError("all preference arrays must have equal length")
    if not world_a_left:
        return 0.0
    hits = sum(
        (float(a_l) > float(a_r)) and (float(b_r) > float(b_l))
        for a_l, a_r, b_l, b_r in zip(world_a_left, world_a_right, world_b_left, world_b_right)
    )
    return hits / len(world_a_left)


# Names used in the plan and in downstream notebooks.
paired_preference_reversal_loss = preference_reversal_loss
flip_loss = preference_reversal_loss


__all__ = [
    "PreferenceReversalExample",
    "log_probability_ratio",
    "reversal_margins",
    "preference_reversal_loss",
    "paired_preference_reversal_loss",
    "batched_preference_reversal_loss",
    "reversal_accuracy",
    "flip_loss",
]

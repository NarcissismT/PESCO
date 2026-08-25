"""Reference losses for the complete PESCO objective.

The CPU trainer uses a tabular update, but these functions expose the exact objective
pieces from plan §12 for a future PPO/GRPO/LLM adapter.  They support Python scalars and
PyTorch tensors without requiring PyTorch for the CPU path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device")


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


@dataclass(frozen=True)
class ObjectiveWeights:
    flip: float = 0.5
    state: float = 0.2
    reference: float = 0.02
    constraint: float = 1.0


def clipped_option_loss(log_ratio: Any, advantage: Any, clip_epsilon: float = 0.2) -> Any:
    """PPO clipped surrogate loss using log-probability ratios."""

    if not 0.0 < float(clip_epsilon) < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    if _is_tensor(log_ratio) or _is_tensor(advantage):
        torch = _torch()
        log_ratio = log_ratio if _is_tensor(log_ratio) else torch.as_tensor(log_ratio, dtype=advantage.dtype, device=advantage.device)
        advantage = advantage if _is_tensor(advantage) else torch.as_tensor(advantage, dtype=log_ratio.dtype, device=log_ratio.device)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        return -torch.minimum(ratio * advantage, clipped * advantage).mean()
    lr = float(log_ratio)
    adv = float(advantage)
    ratio = math.exp(lr)
    clipped = min(1.0 + clip_epsilon, max(1.0 - clip_epsilon, ratio))
    return -min(ratio * adv, clipped * adv)


def categorical_cross_entropy(log_probabilities: Mapping[str, float], target: str) -> float:
    """Finite scalar CE for factorised evidence-state supervision."""

    if target not in log_probabilities:
        raise KeyError(target)
    value = float(log_probabilities[target])
    if not math.isfinite(value):
        raise ValueError("target log probability must be finite")
    return -value


def factorized_evidence_loss(
    validity_log_probs: Mapping[str, float],
    precision_log_probs: Mapping[str, float],
    direction_log_probs: Mapping[str, float],
    *,
    validity_target: str,
    precision_target: str,
    direction_target: str,
    weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    if len(weights) != 3:
        raise ValueError("three factor weights are required")
    return (
        float(weights[0]) * categorical_cross_entropy(validity_log_probs, validity_target)
        + float(weights[1]) * categorical_cross_entropy(precision_log_probs, precision_target)
        + float(weights[2]) * categorical_cross_entropy(direction_log_probs, direction_target)
    )


def reference_kl(policy_probabilities: Mapping[str, float], reference_probabilities: Mapping[str, float], epsilon: float = 1e-8) -> float:
    """Discrete KL(policy || reference), with explicit support checks."""

    if not math.isfinite(float(epsilon)) or not 0.0 < float(epsilon) < 0.5:
        raise ValueError("epsilon must be finite and lie in (0, .5)")
    if not policy_probabilities or not reference_probabilities:
        raise ValueError("policy and reference distributions must be non-empty")
    keys = set(policy_probabilities) | set(reference_probabilities)
    policy_total = 0.0
    reference_total = 0.0
    for key, value in policy_probabilities.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy probabilities must be finite and non-negative")
        policy_total += value
    for key, value in reference_probabilities.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("reference probabilities must be finite and non-negative")
        reference_total += value
    if not math.isclose(policy_total, 1.0, rel_tol=1e-7, abs_tol=1e-9) or not math.isclose(reference_total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("policy and reference probabilities must sum to one")
    total = 0.0
    for key in keys:
        p = max(float(epsilon), float(policy_probabilities.get(key, 0.0)))
        q = max(float(epsilon), float(reference_probabilities.get(key, 0.0)))
        if not (math.isfinite(p) and math.isfinite(q)):
            raise ValueError("probabilities must be finite")
        total += p * math.log(p / q)
    return total


def masked_token_advantages(advantages: Sequence[float], strategy_mask: Sequence[bool]) -> list[float]:
    """Apply the high-level strategy-token mask required by plan §12.6."""

    if len(advantages) != len(strategy_mask):
        raise ValueError("advantages and strategy_mask must have equal length")
    return [float(value) if bool(mask) else 0.0 for value, mask in zip(advantages, strategy_mask)]


def pesco_objective(
    option_loss: Any,
    flip_loss: Any = 0.0,
    state_loss: Any = 0.0,
    reference_loss: Any = 0.0,
    constraint_loss: Any = 0.0,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> Any:
    """Compose ``L_option + λ_flip L_flip + λ_state L_state + λ_ref L_ref``."""

    values = [option_loss, flip_loss, state_loss, reference_loss, constraint_loss]
    if any(_is_tensor(value) for value in values):
        torch = _torch()
        base = option_loss if _is_tensor(option_loss) else torch.as_tensor(option_loss, dtype=torch.float32)
        def as_like(value):
            return value if _is_tensor(value) else torch.as_tensor(value, dtype=base.dtype, device=base.device)
        return base + weights.flip * as_like(flip_loss) + weights.state * as_like(state_loss) + weights.reference * as_like(reference_loss) + weights.constraint * as_like(constraint_loss)
    return float(option_loss) + weights.flip * float(flip_loss) + weights.state * float(state_loss) + weights.reference * float(reference_loss) + weights.constraint * float(constraint_loss)

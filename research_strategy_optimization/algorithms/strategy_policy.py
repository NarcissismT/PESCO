"""Small tabular high-level policy used by the CPU MVP.

This is not intended to stand in for a language model.  It provides a transparent
policy object with the same high-level interface, allowing PESCO's branch/reversal
mechanisms and evidence gates to be tested independently of a particular backbone.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schemas import EvidenceState, Observation, ResearchAction


class TabularStrategyPolicy:
    def __init__(self, actions: Sequence[ResearchAction] = ResearchAction.mvp_actions(), seed: int = 17):
        self.actions = tuple(ResearchAction(a) for a in actions)
        self.rng = random.Random(seed)
        self.logits: Dict[Tuple[object, ...], Dict[ResearchAction, float]] = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )
        self.reference: Dict[Tuple[object, ...], Dict[ResearchAction, float]] = {}

    def state_key(self, observation: Observation) -> Tuple[object, ...]:
        # Binning keeps the table compact and prevents memorising hidden world IDs.
        width = observation.ci_high - observation.ci_low
        if width > 0.30:
            precision = "wide"
        elif width > 0.10:
            precision = "medium"
        else:
            precision = "narrow"
        if observation.effect_estimate > 0.05:
            direction = "positive"
        elif observation.effect_estimate < -0.02:
            direction = "negative"
        else:
            direction = "near_zero"
        signal_bucket = tuple(sorted(s for s in observation.validity_signals if s in {
            "split_overlap_diagnostic", "split_protocol_updated", "sample_count_below_precision_target"
        }))
        return (
            observation.task_family,
            observation.current_method,
            precision,
            direction,
            signal_bucket,
            min(6, observation.remaining_budget),
        )

    def probabilities(self, observation: Observation, temperature: float = 1.0) -> Dict[ResearchAction, float]:
        key = self.state_key(observation)
        values = self.logits[key]
        self.reference.setdefault(key, {action: 1.0 / len(self.actions) for action in self.actions})
        temperature = max(1e-6, float(temperature))
        maximum = max(values.values()) if values else 0.0
        weights = {a: math.exp((v - maximum) / temperature) for a, v in values.items()}
        total = sum(weights.values()) or 1.0
        return {a: w / total for a, w in weights.items()}

    def sample_research_options(self, observation: Observation, k: int = 4) -> List[ResearchAction]:
        # Candidate options are unique and deterministic under the policy RNG.
        probs = self.probabilities(observation)
        ordered = sorted(self.actions, key=lambda a: (-probs[a], a.value))
        if k >= len(ordered):
            return ordered
        chosen = ordered[: max(1, k - 1)]
        remaining = [a for a in self.actions if a not in chosen]
        chosen.append(self.rng.choice(remaining))
        return chosen

    def choose_action(self, observation: Observation, greedy: bool = True) -> ResearchAction:
        probs = self.probabilities(observation)
        if greedy:
            return max(probs, key=probs.get)
        actions, weights = zip(*probs.items())
        return self.rng.choices(actions, weights=weights, k=1)[0]

    def report_belief(self, observation: Observation) -> float:
        return float(observation.hypothesis_probability)

    def update_from_advantages(self, observation: Observation, actions: Sequence[ResearchAction], advantages: Sequence[float], lr: float = 0.1) -> None:
        key = self.state_key(observation)
        for action, advantage in zip(actions, advantages):
            self.logits[key][ResearchAction(action)] += float(lr) * float(advantage)

    def update_preference(self, obs_a: Observation, obs_b: Observation, preferred_a: ResearchAction, preferred_b: ResearchAction, lr: float = 0.2) -> None:
        """Apply a legacy direct preference nudge.

        This method is retained for small compatibility callers.  The PESCO trainer
        uses :meth:`apply_preference_reversal_loss` instead, so its flip update is
        explicitly derived from the differentiable objective rather than being a
        hand-written update followed by loss logging.
        """
        key_a, key_b = self.state_key(obs_a), self.state_key(obs_b)
        self.logits[key_a][preferred_a] += lr
        self.logits[key_a][preferred_b] -= lr
        self.logits[key_b][preferred_b] += lr
        self.logits[key_b][preferred_a] -= lr

    def parameter_checksum(self) -> str:
        """Return a deterministic digest of the trainable tabular parameters."""

        rows = []
        for key in sorted(self.logits, key=repr):
            values = self.logits[key]
            rows.append((repr(key), tuple((action.value, float(values[action])) for action in self.actions)))
        payload = repr(rows).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def apply_preference_reversal_loss(
        self,
        obs_a: Observation,
        obs_b: Observation,
        preferred_a: ResearchAction,
        preferred_b: ResearchAction,
        *,
        beta: float = 1.0,
        learning_rate: float = 0.2,
        confirmed: bool = True,
        weight: float = 1.0,
        gradient_clip_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Update tabular logits by the analytic gradient of the flip loss.

        The tabular CPU policy is intentionally dependency-free, so this is the
        transparent analogue of ``loss.backward(); optimizer.step()``.  For each
        state, softmax log-probability differences reduce to a pair of logit
        differences.  We compute the exact logistic-gradient of
        ``preference_reversal_loss`` and apply one gradient-descent step.  The
        returned diagnostics make it possible to verify that a non-zero gradient
        actually changed parameters and reduced the loss.
        """

        beta = float(beta)
        learning_rate = float(learning_rate)
        weight = float(weight)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be positive and finite")
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative")
        if gradient_clip_norm is not None:
            gradient_clip_norm = float(gradient_clip_norm)
            if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
                raise ValueError("gradient_clip_norm must be positive and finite")

        left = ResearchAction(preferred_a)
        right = ResearchAction(preferred_b)
        if left == right:
            raise ValueError("preference reversal requires two distinct actions")
        key_a, key_b = self.state_key(obs_a), self.state_key(obs_b)
        # Ensure both parameter rows and frozen reference rows exist before taking
        # the pre-update measurement.
        self.probabilities(obs_a)
        self.probabilities(obs_b)
        logs_a = self.action_log_ratios(obs_a)
        logs_b = self.action_log_ratios(obs_b)

        def _softplus(value: float) -> float:
            return max(0.0, value) + math.log1p(math.exp(-abs(value)))

        def _sigmoid_negative(value: float) -> float:
            # ``sigmoid(-value)`` without overflowing for large positive margins.
            if value >= 0.0:
                exp_neg = math.exp(-value)
                return exp_neg / (1.0 + exp_neg)
            exp_pos = math.exp(value)
            return 1.0 / (1.0 + exp_pos)

        margin_a = (logs_a[left.value] - logs_a[right.value])
        margin_b = (logs_b[right.value] - logs_b[left.value])
        loss_a = _softplus(-beta * margin_a)
        loss_b = _softplus(-beta * margin_b)
        # ``preference_reversal_loss`` uses mean reduction over its two terms.
        pre_loss = 0.5 * weight * (loss_a + loss_b)
        if not confirmed or weight == 0.0 or learning_rate == 0.0:
            return {
                "loss_before": float(pre_loss),
                "loss_after": float(pre_loss),
                "gradient_norm": 0.0,
                "update_norm": 0.0,
                "margin_a_before": float(margin_a),
                "margin_b_before": float(margin_b),
                "margin_a_after": float(margin_a),
                "margin_b_after": float(margin_b),
                "applied": 0.0,
                "parameter_checksum_before": self.parameter_checksum(),
                "parameter_checksum_after": self.parameter_checksum(),
            }

        # d softplus(-beta*m) / d m = -beta * sigmoid(-beta*m).  Each margin is a
        # difference of two logits, so its gradient has only two non-zero entries.
        sigmoid_a = _sigmoid_negative(beta * margin_a)
        sigmoid_b = _sigmoid_negative(beta * margin_b)
        coefficient_a = -0.5 * weight * beta * sigmoid_a
        coefficient_b = -0.5 * weight * beta * sigmoid_b
        checksum_before = self.parameter_checksum()
        gradients: Dict[Tuple[str, ResearchAction], float] = {}
        gradients[("a", left)] = gradients.get(("a", left), 0.0) + coefficient_a
        gradients[("a", right)] = gradients.get(("a", right), 0.0) - coefficient_a
        gradients[("b", right)] = gradients.get(("b", right), 0.0) + coefficient_b
        gradients[("b", left)] = gradients.get(("b", left), 0.0) - coefficient_b
        norm = math.sqrt(sum(value * value for value in gradients.values()))
        scale = 1.0
        if gradient_clip_norm is not None and norm > gradient_clip_norm:
            scale = gradient_clip_norm / norm
        update_norm_sq = 0.0
        for (side, action), gradient in gradients.items():
            delta = -learning_rate * gradient * scale
            key = key_a if side == "a" else key_b
            self.logits[key][action] += delta
            update_norm_sq += delta * delta

        post_logs_a = self.action_log_ratios(obs_a)
        post_logs_b = self.action_log_ratios(obs_b)
        margin_a_after = post_logs_a[left.value] - post_logs_a[right.value]
        margin_b_after = post_logs_b[right.value] - post_logs_b[left.value]
        post_loss = 0.5 * weight * (_softplus(-beta * margin_a_after) + _softplus(-beta * margin_b_after))
        return {
            "loss_before": float(pre_loss),
            "loss_after": float(post_loss),
            "gradient_norm": float(norm),
            "update_norm": float(math.sqrt(update_norm_sq)),
            "margin_a_before": float(margin_a),
            "margin_b_before": float(margin_b),
            "margin_a_after": float(margin_a_after),
            "margin_b_after": float(margin_b_after),
            "applied": 1.0,
            "gradient_clip_scale": float(scale),
            "parameter_checksum_before": checksum_before,
            "parameter_checksum_after": self.parameter_checksum(),
        }

    def action_log_ratios(self, observation: Observation) -> Dict[str, float]:
        probs = self.probabilities(observation)
        reference = self.reference.get(self.state_key(observation), {})
        return {
            a.value: math.log(max(1e-8, p)) - math.log(max(1e-8, reference.get(a, 1.0 / len(self.actions))))
            for a, p in probs.items()
        }

    def freeze_reference(self) -> None:
        """Freeze the current tabular policy as the preference-loss reference."""

        for key, logits in self.logits.items():
            maximum = max(logits.values()) if logits else 0.0
            weights = {action: math.exp(value - maximum) for action, value in logits.items()}
            total = sum(weights.values()) or 1.0
            self.reference[key] = {action: weight / total for action, weight in weights.items()}

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
        return (observation.current_method, precision, direction, signal_bucket, min(6, observation.remaining_budget))

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
        key_a, key_b = self.state_key(obs_a), self.state_key(obs_b)
        self.logits[key_a][preferred_a] += lr
        self.logits[key_a][preferred_b] -= lr
        self.logits[key_b][preferred_b] += lr
        self.logits[key_b][preferred_a] -= lr

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

"""Policy adapters used by the reproducible MVP runner."""

from __future__ import annotations

import random
from typing import Iterable, List, Mapping, Optional, Sequence

from ..schemas import EvidenceState, Observation, ResearchAction
from ..algorithms.strategy_policy import TabularStrategyPolicy


def infer_visible_state(observation: Observation, delta_min: float = 0.02) -> Optional[EvidenceState]:
    """Infer a state from public evidence only (never reads world/verifier labels)."""

    signals = set(observation.validity_signals)
    if signals.intersection({
        "split_overlap_diagnostic",
        "metric_scope_mismatch",
        "variance_estimator_unstable",
        "treatment_confounder_dependence",
        "protocol_invalid_diagnostic",
    }):
        return EvidenceState.INVALID
    width = observation.ci_high - observation.ci_low
    if width > 0.30 or "sample_count_below_precision_target" in signals:
        return EvidenceState.INSUFFICIENT
    if observation.ci_low > delta_min:
        return EvidenceState.SUPPORTED
    if observation.ci_high < delta_min:
        return EvidenceState.REFUTED
    return EvidenceState.INSUFFICIENT


class BasePolicy:
    name = "Base"

    def choose(self, observation: Observation, **kwargs) -> ResearchAction:
        return ResearchAction.CONTINUE


class RandomPolicy(BasePolicy):
    name = "Random"

    def __init__(self, seed: int = 17):
        self.rng = random.Random(seed)

    def choose(self, observation: Observation, **kwargs) -> ResearchAction:
        return self.rng.choice(ResearchAction.mvp_actions())


class EvidenceHeuristicPolicy(BasePolicy):
    """Four-state/SFT-style policy using only visible evidence."""

    def __init__(self, name: str = "GRPO-FourState", delta_min: float = 0.02):
        self.name = name
        self.delta_min = delta_min

    def choose(self, observation: Observation, **kwargs) -> ResearchAction:
        state = infer_visible_state(observation, self.delta_min)
        if state is EvidenceState.SUPPORTED:
            return ResearchAction.CONTINUE
        if state is EvidenceState.REFUTED:
            return ResearchAction.SWITCH
        if state is EvidenceState.INSUFFICIENT:
            return ResearchAction.SAMPLE
        if state is EvidenceState.INVALID:
            return ResearchAction.REPAIR
        return ResearchAction.CONTINUE


class OracleSearchPolicy(BasePolicy):
    name = "Search-Only"

    def choose(self, observation: Observation, branch_records=None, **kwargs) -> ResearchAction:
        if not branch_records:
            return ResearchAction.CONTINUE
        def value(record):
            if isinstance(record, Mapping):
                return float(record["utility"]), ResearchAction(record["option"])
            return float(record.utility), ResearchAction(record.option)
        best = max(branch_records, key=lambda r: value(r)[0])
        return value(best)[1]


class PESCOPolicy(BasePolicy):
    name = "PESCO-Full"

    def __init__(self, tabular_policy: Optional[TabularStrategyPolicy] = None, name: str = "PESCO-Full"):
        self.policy = tabular_policy or TabularStrategyPolicy()
        self.name = name

    def choose(self, observation: Observation, **kwargs) -> ResearchAction:
        return self.policy.choose_action(observation, greedy=True)

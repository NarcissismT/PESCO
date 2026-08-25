"""CPU reference trainer implementing PESCO's stage-C/D mechanisms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schemas import BranchRecord, ReversalPair, ResearchAction
from .branch_rollout import BranchRolloutManager
from .preference_reversal_loss import preference_reversal_loss
from .strategy_policy import TabularStrategyPolicy


@dataclass
class TrainerConfig:
    epochs: int = 8
    branch_options: int = 4
    branch_seeds: Tuple[int, ...] = (17, 29, 41, 53)
    option_learning_rate: float = 0.15
    reversal_learning_rate: float = 0.25
    reversal_beta: float = 1.0
    use_paired_world: bool = True
    use_flip_loss: bool = True
    use_branch_advantage: bool = True
    use_validity_gate: bool = True


@dataclass
class TrainingLog:
    epochs: List[Dict[str, float]] = field(default_factory=list)
    reversal_pairs: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"epochs": self.epochs, "reversal_pairs": self.reversal_pairs}


class PESCOTrainer:
    def __init__(self, policy: Optional[TabularStrategyPolicy] = None, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.policy = policy or TabularStrategyPolicy(seed=17)
        self.log = TrainingLog()

    def fit(self, environment_factory, world_ids: Sequence[str], question_id: str = "rq_mvp_001") -> TabularStrategyPolicy:
        if not self.policy.reference:
            # Preference reversal ratios are measured against a frozen reference policy,
            # not against a moving zero baseline.
            self.policy.freeze_reference()
        for epoch in range(self.config.epochs):
            world_records: Dict[str, List[BranchRecord]] = {}
            epoch_advantages: List[float] = []
            for world_id in world_ids:
                env = environment_factory()
                env.reset(question_id=question_id, world_id=world_id, seed=epoch)
                # A common, frozen pilot produces the public evidence state on which
                # strategy choices are conditioned.  Its trusted verdict remains
                # evaluator-side; only ``visible_observation`` reaches the policy.
                env.execute_option(ResearchAction.CONTINUE, seeds=self.config.branch_seeds)
                initial = env.visible_observation()
                options = self.policy.sample_research_options(initial, self.config.branch_options)
                utility_fn = None
                if not self.config.use_validity_gate:
                    # Explicit No-ValidityGate ablation: expose the raw surface effect
                    # to the branch utility while keeping the trusted verdict in the
                    # audit record.  The default path remains hard-gated.
                    utility_fn = lambda output, verdict, branch_env: float(output.effect_estimate) - 0.05 * float(output.execution_cost)
                manager = BranchRolloutManager(env.build_verifier(), self.config.branch_seeds, utility_fn=utility_fn)
                records = manager.execute_paired_options(env, options, snapshot=env.snapshot(), seeds=self.config.branch_seeds)
                world_records[world_id] = records
                if self.config.use_branch_advantage:
                    self.policy.update_from_advantages(initial, [r.option for r in records], [r.advantage for r in records], self.config.option_learning_rate)
                    epoch_advantages.extend(r.advantage for r in records)

            reversal_count = 0
            reversal_losses: List[float] = []
            if self.config.use_paired_world and self.config.use_flip_loss and len(world_ids) >= 2:
                # The frozen MVP pairing uses the first two worlds (supported/refuted).
                for left, right in zip(world_ids[::2], world_ids[1::2]):
                    a_records, b_records = world_records[left], world_records[right]
                    by_a = {r.option: r for r in a_records}
                    by_b = {r.option: r for r in b_records}
                    if ResearchAction.CONTINUE not in by_a or ResearchAction.SWITCH not in by_a:
                        continue
                    if ResearchAction.CONTINUE not in by_b or ResearchAction.SWITCH not in by_b:
                        continue
                    delta_a = by_a[ResearchAction.CONTINUE].utility - by_a[ResearchAction.SWITCH].utility
                    delta_b = by_b[ResearchAction.CONTINUE].utility - by_b[ResearchAction.SWITCH].utility
                    if delta_a > 0.05 and delta_b < -0.05:
                        reversal_count += 1
                        obs_a = a_records[0].trajectory.initial_observation
                        obs_b = b_records[0].trajectory.initial_observation
                        self.policy.update_preference(obs_a, obs_b, ResearchAction.CONTINUE, ResearchAction.SWITCH, self.config.reversal_learning_rate)
                        loss = preference_reversal_loss(
                            self.policy.action_log_ratios(obs_a),
                            self.policy.action_log_ratios(obs_b),
                            ResearchAction.CONTINUE.value,
                            ResearchAction.SWITCH.value,
                            self.config.reversal_beta,
                        )
                        reversal_losses.append(loss)
                        self.log.reversal_pairs.append({
                            "epoch": epoch,
                            "world_a": left,
                            "world_b": right,
                            "delta_a": delta_a,
                            "delta_b": delta_b,
                            "confirmed": True,
                        })
            self.log.epochs.append({
                "epoch": float(epoch),
                "mean_advantage": sum(epoch_advantages) / len(epoch_advantages) if epoch_advantages else 0.0,
                "reversal_count": float(reversal_count),
                "reversal_loss": sum(reversal_losses) / len(reversal_losses) if reversal_losses else 0.0,
            })
        return self.policy

    def save_log(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.log.to_dict(), indent=2), encoding="utf-8")

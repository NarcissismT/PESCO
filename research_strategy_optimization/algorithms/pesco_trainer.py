"""CPU reference trainer implementing PESCO's stage-C/D mechanisms."""

from __future__ import annotations

import json
import math
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
    # The tabular CPU reference applies the exact analytic gradient of the flip
    # objective.  A future torch/LLM adapter can replace this with autograd while
    # preserving the same logged components and gate semantics.
    use_differentiable_flip_update: bool = True
    gradient_clip_norm: Optional[float] = 5.0


@dataclass
class TrainingLog:
    epochs: List[Dict[str, object]] = field(default_factory=list)
    reversal_pairs: List[Dict[str, object]] = field(default_factory=list)
    schema_version: str = "pesco_training_log_v0.3"
    update_rule: str = "analytic_tabular_gradient_of_flip_loss"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "update_rule": self.update_rule,
            "epochs": self.epochs,
            "reversal_pairs": self.reversal_pairs,
        }


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
                else:
                    utility_fn = self._public_transition_utility
                manager = BranchRolloutManager(env.build_verifier(), self.config.branch_seeds, utility_fn=utility_fn)
                records = manager.execute_paired_options(env, options, snapshot=env.snapshot(), seeds=self.config.branch_seeds)
                world_records[world_id] = records
                if self.config.use_branch_advantage:
                    self.policy.update_from_advantages(initial, [r.option for r in records], [r.advantage for r in records], self.config.option_learning_rate)
                    epoch_advantages.extend(r.advantage for r in records)

            reversal_count = 0
            reversal_losses: List[float] = []
            reversal_post_losses: List[float] = []
            flip_gradient_norms: List[float] = []
            flip_update_norms: List[float] = []
            flip_updates_applied = 0
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
                        logs_a_before = self.policy.action_log_ratios(obs_a)
                        logs_b_before = self.policy.action_log_ratios(obs_b)
                        loss_before = preference_reversal_loss(
                            logs_a_before,
                            logs_b_before,
                            ResearchAction.CONTINUE.value,
                            ResearchAction.SWITCH.value,
                            self.config.reversal_beta,
                        )
                        if self.config.use_differentiable_flip_update:
                            update = self.policy.apply_preference_reversal_loss(
                                obs_a,
                                obs_b,
                                ResearchAction.CONTINUE,
                                ResearchAction.SWITCH,
                                beta=self.config.reversal_beta,
                                learning_rate=self.config.reversal_learning_rate,
                                gradient_clip_norm=self.config.gradient_clip_norm,
                            )
                            loss_after = float(update["loss_after"])
                            flip_gradient_norms.append(float(update["gradient_norm"]))
                            flip_update_norms.append(float(update["update_norm"]))
                            flip_updates_applied += int(bool(update["applied"]))
                        else:
                            # Explicit diagnostic mode for ablations/adapters that do
                            # not yet expose an optimizer.  It never masquerades as an
                            # applied flip update in the log.
                            loss_after = float(loss_before)
                            update = {
                                "loss_before": float(loss_before),
                                "loss_after": float(loss_after),
                                "gradient_norm": 0.0,
                                "update_norm": 0.0,
                                "applied": 0.0,
                            }
                        reversal_losses.append(float(loss_before))
                        reversal_post_losses.append(float(loss_after))
                        self.log.reversal_pairs.append({
                            "epoch": epoch,
                            "world_a": left,
                            "world_b": right,
                            "delta_a": delta_a,
                            "delta_b": delta_b,
                            "confirmed": True,
                            "loss_before": float(loss_before),
                            "loss_after": float(loss_after),
                            "gradient_norm": float(update["gradient_norm"]),
                            "parameter_update_norm": float(update["update_norm"]),
                            "loss_driven_update": bool(update["applied"]),
                            "parameter_checksum_before": update.get("parameter_checksum_before"),
                            "parameter_checksum_after": update.get("parameter_checksum_after"),
                        })
            self.log.epochs.append({
                "epoch": float(epoch),
                "mean_advantage": sum(epoch_advantages) / len(epoch_advantages) if epoch_advantages else 0.0,
                "reversal_count": float(reversal_count),
                "reversal_loss": sum(reversal_losses) / len(reversal_losses) if reversal_losses else 0.0,
                "flip_loss_before": sum(reversal_losses) / len(reversal_losses) if reversal_losses else 0.0,
                "flip_loss_after": sum(reversal_post_losses) / len(reversal_post_losses) if reversal_post_losses else 0.0,
                "flip_loss_reduction": (
                    (sum(reversal_losses) - sum(reversal_post_losses)) / len(reversal_losses)
                    if reversal_losses else 0.0
                ),
                "flip_gradient_norm": sum(flip_gradient_norms) / len(flip_gradient_norms) if flip_gradient_norms else 0.0,
                "flip_parameter_update_norm": sum(flip_update_norms) / len(flip_update_norms) if flip_update_norms else 0.0,
                "flip_updates_applied": float(flip_updates_applied),
                "loss_driven_flip_update": bool(flip_updates_applied > 0),
                "policy_parameter_checksum": self.policy.parameter_checksum(),
            })
        return self.policy

    @staticmethod
    def _public_transition_utility(output, verdict, branch_env) -> float:
        """Score a branch using the public state *before* its selected action.

        The verifier's final state can legitimately change after ``switch`` or
        ``sample``.  Scoring against only that final state makes a correct switch in
        a refuted world look identical to an unnecessary switch in a supported world.
        This evaluator-side helper uses the branch's captured public initial
        observation and trusted validity gate; no latent world kind or oracle action
        table is consulted.
        """

        # This legacy tabular trainer has no registered question/world utility
        # callback.  Keep its fallback action-agnostic: a generic state→action table
        # would turn the trainer into an evaluator oracle.  The Tier-1 suite supplies
        # a richer public-transition callback for family-specific estimands.
        signals = set(getattr(output, "validity_signals", ()))
        if verdict is not None and getattr(verdict, "validity_pass", True) is False:
            task = -0.30
        else:
            task = 0.20
            if "split_protocol_updated" in signals:
                task += 0.25
            if "group_held_out_split" in signals and "split_overlap_diagnostic" not in signals:
                task += 0.35
            if any(token in signal for signal in signals for token in ("adjusted", "controlled", "subgroup_metric_estimator")):
                task += 0.30
            if int(getattr(output, "sample_size", 0)) >= 60 and "sample_count_below_precision_target" not in signals:
                task += 0.10
            # Use the executed, validity-gated effect as a continuous scientific
            # signal.  This keeps the legacy Tier-0 trainer action-agnostic (there
            # is no EvidenceState->action lookup), while still allowing the frozen
            # supported/refuted pair to produce a real preference reversal: method A
            # is observed positive in one world and method B is observed positive in
            # the other.  Invalid surface effects never reach this term.
            task += 0.50 * math.tanh(5.0 * float(getattr(output, "effect_estimate", 0.0)))
        confirmation = 0.1 if bool(getattr(verdict, "independent_confirmation_passed", False)) else 0.0
        cost = float(getattr(output, "execution_cost", 0.0))
        return float(task + confirmation - 0.05 * cost)

    def save_log(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.log.to_dict(), indent=2), encoding="utf-8")

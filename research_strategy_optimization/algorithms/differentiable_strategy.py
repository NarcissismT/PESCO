"""Small CPU-differentiable strategy policies and matched training objectives.

This module is deliberately backbone-free: a two-layer PyTorch MLP consumes only
the public :class:`~research_strategy_optimization.schemas.Observation` payload and
emits action, evidence-state, and active-hypothesis belief heads.  It is a genuine
optimization reference for the C/D/E experiments in the research plan, not an LLM
or external-paper reimplementation.

The trainer accepts an evaluator-produced ``DecisionDataset``.  Hidden world labels
are retained only in the dataset's audit fields; feature construction never reads
them.  All methods share the same examples, optimizer-step budget, candidate action
set, and evaluation code.  Their differences are explicit objective switches:
SFT, terminal/four-state policy gradients, state gating, same-state branch LOO
advantages, cross-world flip loss, and evidence-gated multi-teacher distillation.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..schemas import EvidenceState, HypothesisBelief, Observation, ResearchAction
from .preference_reversal_loss import preference_reversal_loss


ACTION_SET: Tuple[ResearchAction, ...] = ResearchAction.mvp_actions()
STATE_SET: Tuple[EvidenceState, ...] = tuple(EvidenceState)
_SIGNAL_VOCAB = (
    "initial_experiment_pending",
    "sample_count_below_precision_target",
    "split_overlap_diagnostic",
    "leaky_row_split",
    "split_protocol_updated",
    "group_held_out_split",
    "treatment_confounder_dependence",
    "confounder_adjusted_estimator",
    "confounding_controlled",
    "treatment_assignment_independent",
    "alternative_method_evaluated",
    "independent_confirmation_partition",
)
_HISTORY_ACTIONS = tuple(action.value for action in ACTION_SET)
_TASK_FAMILIES = (
    "group_generalization",
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
)


def observation_to_features(observation: Observation) -> Tensor:
    """Encode a public observation without world IDs, latent truth, or verifier labels."""

    width = float(observation.ci_high - observation.ci_low)
    belief_map = {
        str(getattr(belief, "hypothesis_id", "")): float(getattr(belief, "probability", 0.5))
        for belief in getattr(observation, "hypothesis_beliefs", ())
    }
    values: List[float] = [
        float(observation.effect_estimate),
        float(observation.ci_low),
        float(observation.ci_high),
        width,
        math.log1p(max(0, int(observation.sample_size))) / 8.0,
        math.log1p(max(0, int(observation.seed_count))) / 4.0,
        float(observation.remaining_budget) / 6.0,
        float(observation.turn) / 6.0,
        float(observation.hypothesis_probability),
        1.0 if observation.current_method == "method_b" else 0.0,
        1.0 if observation.active_hypothesis_id == "H_B" else 0.0,
        float(belief_map.get("H_A", 0.5)),
        float(belief_map.get("H_B", 0.5)),
    ]
    signals = set(observation.validity_signals)
    values.extend(1.0 if signal in signals else 0.0 for signal in _SIGNAL_VOCAB)
    history = set(observation.history_summary)
    values.extend(
        1.0 if any(action in item for item in history) else 0.0
        for action in _HISTORY_ACTIONS
    )
    values.extend(1.0 if observation.task_family == family else 0.0 for family in _TASK_FAMILIES)
    return torch.tensor(values, dtype=torch.float32)


FEATURE_DIM = int(observation_to_features(
    Observation("q", 0, "method_a", 0.0, -0.1, 0.1, 1, 1, 0)
).numel())


@dataclass
class DecisionExample:
    """One evaluator-produced state and all matched same-state branch outcomes."""

    observation: Observation
    branch_utilities: Tuple[float, ...]
    state_target: EvidenceState
    split: str = "train"
    question_id: str = ""
    world_id: str = "audit_hidden"
    world_pair_id: str = ""
    branch_states: Tuple[EvidenceState, ...] = ()
    confirmation_passed: bool = False
    branch_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.branch_utilities = tuple(float(value) for value in self.branch_utilities)
        if not self.branch_utilities:
            raise ValueError("branch_utilities must be non-empty")
        if any(not math.isfinite(value) for value in self.branch_utilities):
            raise ValueError("branch_utilities must be finite")
        self.state_target = EvidenceState(self.state_target)
        self.question_id = self.question_id or self.observation.question_id
        self.branch_count = int(self.branch_count or len(self.branch_utilities))
        if self.branch_count != len(self.branch_utilities):
            raise ValueError("branch_count must match branch_utilities")
        if self.branch_states and len(self.branch_states) != self.branch_count:
            raise ValueError("branch_states must match branch_utilities")

    @property
    def best_action_index(self) -> int:
        return int(max(range(self.branch_count), key=lambda index: self.branch_utilities[index]))

    @property
    def best_action(self) -> ResearchAction:
        return ACTION_SET[self.best_action_index]

    @property
    def regret_if(self) -> Tuple[float, ...]:
        best = max(self.branch_utilities)
        return tuple(float(best - value) for value in self.branch_utilities)

    def to_dict(self, include_audit: bool = True) -> dict:
        if not include_audit:
            # A public replay export contains observations and the split boundary
            # only.  Branch utilities, state labels, best actions, confirmation
            # outcomes, and reversal endpoints are evaluator/training labels and are
            # intentionally absent; the in-memory audit dataset remains available to
            # the controlled trainer.
            return {
                "observation": self.observation.to_dict(),
                "branch_count": self.branch_count,
                "split": self.split,
                "question_id": self.observation.question_id,
                "world_pair_id": "",
                "metadata": {},
            }
        payload = {
            "observation": self.observation.to_dict(),
            "branch_utilities": list(self.branch_utilities),
            "branch_count": self.branch_count,
            "state_target": self.state_target.value,
            "split": self.split,
            "question_id": self.question_id,
            "world_pair_id": self.world_pair_id,
            "branch_states": [state.value for state in self.branch_states],
            "confirmation_passed": bool(self.confirmation_passed),
            "best_action": self.best_action.value,
            "metadata": dict(self.metadata),
        }
        if include_audit:
            payload["world_id"] = self.world_id
        return payload


@dataclass(frozen=True)
class ReversalExample:
    """Confirmed paired-world preference reversal used only by the flip objective."""

    left: int
    right: int
    action_left: ResearchAction
    action_right: ResearchAction
    margin: float = 0.05
    confirmed: bool = True
    weight: float = 1.0
    # Seed-level uncertainty used to admit the pair to flip training.  These are
    # evaluator statistics, not policy inputs, and remain optional for legacy data.
    lcb_left: float = 0.0
    ucb_right: float = 0.0
    sample_count: int = 0


@dataclass
class DecisionDataset:
    examples: List[DecisionExample]
    reversals: List[ReversalExample] = field(default_factory=list)
    schema_version: str = "pesco_decision_dataset_v0.3"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def split(self, name: str) -> "DecisionDataset":
        indices = [index for index, example in enumerate(self.examples) if example.split == name]
        remap = {old: new for new, old in enumerate(indices)}
        examples = [self.examples[index] for index in indices]
        reversals = [
            ReversalExample(
                left=remap[pair.left],
                right=remap[pair.right],
                action_left=pair.action_left,
                action_right=pair.action_right,
                margin=pair.margin,
                confirmed=pair.confirmed,
                weight=pair.weight,
                lcb_left=pair.lcb_left,
                ucb_right=pair.ucb_right,
                sample_count=pair.sample_count,
            )
            for pair in self.reversals
            if pair.left in remap and pair.right in remap
        ]
        return DecisionDataset(examples, reversals, self.schema_version, dict(self.provenance))

    def to_dict(self, include_audit: bool = True) -> dict:
        if not include_audit:
            split_counts = {
                split: sum(example.split == split for example in self.examples)
                for split in ("train", "dev", "diagnostic_ood")
            }
            public_provenance = {
                "schema_version": self.schema_version,
                "example_count": len(self.examples),
                "split_counts": split_counts,
                "branch_groups": self.provenance.get("branch_groups"),
                "exploration_seed_observations": self.provenance.get("exploration_seed_observations"),
                "public_policy_inputs_only": True,
                "audit_labels_excluded": True,
            }
            return {
                "schema_version": "pesco_public_observation_dataset_v0.1",
                "provenance": public_provenance,
                "examples": [example.to_dict(include_audit=False) for example in self.examples],
                "reversals": [],
            }
        return {
            "schema_version": self.schema_version,
            "provenance": dict(self.provenance),
            "examples": [example.to_dict(include_audit=include_audit) for example in self.examples],
            "reversals": [
                {
                    "left": pair.left,
                    "right": pair.right,
                    "action_left": pair.action_left.value,
                    "action_right": pair.action_right.value,
                    "margin": pair.margin,
                    "confirmed": pair.confirmed,
                    "weight": pair.weight,
                    "lcb_left": pair.lcb_left,
                    "ucb_right": pair.ucb_right,
                    "sample_count": pair.sample_count,
                }
                for pair in self.reversals
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionDataset":
        examples: List[DecisionExample] = []
        for raw in payload.get("examples", []):
            obs_payload = dict(raw["observation"])
            interval = obs_payload.pop("confidence_interval", [-0.1, 0.1])
            # ``Observation.to_dict`` writes both a compact probability mapping and
            # full identity/turn records.  Prefer the latter so a save/reload cycle
            # cannot silently collapse H_A/H_B into the active scalar belief.
            belief_records = obs_payload.pop("belief_records", None)
            belief_map = obs_payload.pop("hypothesis_beliefs", None)
            if belief_records:
                hypothesis_beliefs = tuple(
                    HypothesisBelief(
                        hypothesis_id=str(record["hypothesis_id"]),
                        probability=float(record["probability"]),
                        turn=int(record.get("turn", obs_payload.get("turn", 0))),
                        committed_before_action=bool(record.get("committed_before_action", True)),
                    )
                    for record in belief_records
                )
            elif isinstance(belief_map, Mapping):
                hypothesis_beliefs = tuple(
                    HypothesisBelief(
                        hypothesis_id=str(hypothesis_id),
                        probability=float(probability),
                        turn=int(obs_payload.get("turn", 0)),
                    )
                    for hypothesis_id, probability in sorted(belief_map.items())
                )
            else:
                hypothesis_beliefs = ()
            observation = Observation(
                question_id=obs_payload.pop("question_id"),
                turn=int(obs_payload.pop("turn", 0)),
                current_method=obs_payload.pop("current_method", "method_a"),
                effect_estimate=float(obs_payload.pop("effect_estimate", 0.0)),
                ci_low=float(interval[0]),
                ci_high=float(interval[1]),
                sample_size=int(obs_payload.pop("sample_size", 0)),
                seed_count=int(obs_payload.pop("seed_count", 0)),
                remaining_budget=int(obs_payload.pop("remaining_budget", 0)),
                metric_name=obs_payload.pop("metric_name", "group_held_out_accuracy_delta"),
                validity_signals=tuple(obs_payload.pop("validity_signals", ())),
                history_summary=tuple(obs_payload.pop("history_summary", ())),
                hypothesis_probability=float(obs_payload.pop("hypothesis_probability", 0.5)),
                active_hypothesis_id=obs_payload.pop("active_hypothesis_id", "H_A"),
                hypothesis_beliefs=hypothesis_beliefs,
                task_family=obs_payload.pop("task_family", "group_generalization"),
            )
            examples.append(DecisionExample(
                observation=observation,
                branch_utilities=tuple(raw["branch_utilities"]),
                branch_states=tuple(EvidenceState(value) for value in raw.get("branch_states", ())),
                state_target=EvidenceState(raw.get("state_target", "insufficient")),
                split=str(raw.get("split", "train")),
                question_id=str(raw.get("question_id", observation.question_id)),
                world_id=str(raw.get("world_id", "audit_hidden")),
                world_pair_id=str(raw.get("world_pair_id", "")),
                confirmation_passed=bool(raw.get("confirmation_passed", False)),
                branch_count=int(raw.get("branch_count", len(raw["branch_utilities"]))),
                metadata=dict(raw.get("metadata", {})),
            ))
        reversals = [
            ReversalExample(
                left=int(raw["left"]),
                right=int(raw["right"]),
                action_left=ResearchAction(raw["action_left"]),
                action_right=ResearchAction(raw["action_right"]),
                margin=float(raw.get("margin", 0.05)),
                confirmed=bool(raw.get("confirmed", True)),
                weight=float(raw.get("weight", 1.0)),
                lcb_left=float(raw.get("lcb_left", 0.0)),
                ucb_right=float(raw.get("ucb_right", 0.0)),
                sample_count=int(raw.get("sample_count", 0)),
            )
            for raw in payload.get("reversals", [])
        ]
        return cls(examples, reversals, str(payload.get("schema_version", "pesco_decision_dataset_v0.3")), dict(payload.get("provenance", {})))

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionDataset":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_json(self, path: str | Path, include_audit: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(include_audit=include_audit), indent=2, ensure_ascii=False), encoding="utf-8")


class DifferentiableStrategyPolicy(nn.Module):
    """Compact public-observation policy used by the CPU algorithm experiments."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 48, seed: int = 17):
        super().__init__()
        torch.manual_seed(int(seed))
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, len(ACTION_SET))
        self.state_head = nn.Linear(hidden_dim, len(STATE_SET))
        # One logit per named hypothesis prevents H_B evidence from being scored as
        # if it belonged to H_A after a method switch.
        self.belief_head = nn.Linear(hidden_dim, 2)

    def forward(self, features: Tensor | Sequence[Tensor]) -> Dict[str, Tensor]:
        if not torch.is_tensor(features):
            features = torch.stack(list(features))
        if features.ndim == 1:
            features = features.unsqueeze(0)
        hidden = self.trunk(features)
        return {
            "action_logits": self.action_head(hidden),
            "state_logits": self.state_head(hidden),
            "belief_logits": self.belief_head(hidden),
        }

    def action_distribution(self, features: Tensor) -> Tensor:
        return F.softmax(self.forward(features)["action_logits"], dim=-1)

    def clone_frozen(self) -> "DifferentiableStrategyPolicy":
        clone = copy.deepcopy(self)
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        return clone


@dataclass
class DifferentiableTrainerConfig:
    epochs: int = 32
    batch_size: int = 16
    learning_rate: float = 3e-3
    hidden_dim: int = 48
    seed: int = 17
    max_optimizer_steps: int = 256
    state_loss_weight: float = 0.25
    belief_loss_weight: float = 0.10
    flip_loss_weight: float = 0.5
    kl_weight: float = 0.02
    entropy_weight: float = 0.0
    constraint_loss_weight: float = 0.10
    gradient_clip_norm: float = 5.0
    temperature: float = 1.0


@dataclass
class DifferentiableTrainingLog:
    method: str
    epochs: List[Dict[str, float]] = field(default_factory=list)
    optimizer_steps: int = 0
    environment_branch_groups: int = 0
    reversal_count: int = 0
    parameter_count: int = 0
    teacher_parameter_count: int = 0
    teacher_optimizer_steps: int = 0
    matched_budget_steps: int = 0
    # ``flip_gradient_norm`` is a periodic live-autograd probe; ``flip_update_norm``
    # is the mean combined-objective parameter step while the flip term is active.
    flip_gradient_norm: float = 0.0
    flip_update_norm: float = 0.0
    flip_updates_applied: int = 0
    flip_gradient_probe_count: int = 0
    constraint_loss_mean: float = 0.0
    # SFT provenance is explicit so audit artifacts cannot be mistaken for
    # privileged benchmark supervision.
    supervised_target_source: str = "not_applicable"
    # Machine-readable ablation provenance.  A component can be logged for
    # diagnostics while its weight is zero; this map records what actually entered
    # the optimized objective for the method.
    objective_components: Dict[str, bool] = field(default_factory=dict)
    implementation_status: str = "genuine_cpu_differentiable_reference"

    def to_dict(self) -> dict:
        return {
            "schema_version": "pesco_differentiable_training_v0.3",
            "method": self.method,
            "implementation_status": self.implementation_status,
            "optimizer_steps": self.optimizer_steps,
            "environment_branch_groups": self.environment_branch_groups,
            "reversal_count": self.reversal_count,
            "parameter_count": self.parameter_count,
            "teacher_parameter_count": self.teacher_parameter_count,
            "teacher_optimizer_steps": self.teacher_optimizer_steps,
            "matched_budget_steps": self.matched_budget_steps,
            "flip_gradient_norm": self.flip_gradient_norm,
            "flip_update_norm": self.flip_update_norm,
            "flip_updates_applied": self.flip_updates_applied,
            "flip_gradient_probe_count": self.flip_gradient_probe_count,
            "constraint_loss_mean": self.constraint_loss_mean,
            "supervised_target_source": self.supervised_target_source,
            "objective_components": dict(self.objective_components),
            "epochs": self.epochs,
        }


def _batch_indices(size: int, batch_size: int, generator: torch.Generator) -> Iterable[Tensor]:
    order = torch.randperm(size, generator=generator)
    for start in range(0, size, max(1, batch_size)):
        yield order[start:start + max(1, batch_size)]


def _stack_examples(examples: Sequence[DecisionExample], indices: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    selected = [examples[int(index)] for index in indices.tolist()]
    features = torch.stack([observation_to_features(example.observation) for example in selected])
    utilities = torch.tensor([example.branch_utilities for example in selected], dtype=torch.float32)
    states = torch.tensor([STATE_SET.index(example.state_target) for example in selected], dtype=torch.long)
    best = torch.tensor([example.best_action_index for example in selected], dtype=torch.long)
    return features, utilities, states, best


def _entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()


def _mean_kl(policy_logits: Tensor, reference_logits: Tensor) -> Tensor:
    p = F.log_softmax(policy_logits, dim=-1)
    q = F.softmax(reference_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean", log_target=False)


def _belief_target_matrix(selected: Sequence[DecisionExample]) -> Tensor:
    """Build a target for each named hypothesis, preserving identity after SWITCH."""

    targets: List[List[float]] = []
    for example in selected:
        state = example.state_target
        active_target = (
            1.0 if state is EvidenceState.SUPPORTED else
            0.0 if state is EvidenceState.REFUTED else
            0.5
        )
        observed = {
            str(getattr(belief, "hypothesis_id", "")): float(getattr(belief, "probability", 0.5))
            for belief in getattr(example.observation, "hypothesis_beliefs", ())
        }
        values = [float(observed.get("H_A", 0.5)), float(observed.get("H_B", 0.5))]
        active_index = 1 if example.observation.active_hypothesis_id == "H_B" else 0
        values[active_index] = active_target
        # Invalid/insufficient observations must not manufacture confidence in either
        # hypothesis.  Neutral targets are explicit rather than a hidden reward.
        if state in {EvidenceState.INVALID, EvidenceState.INSUFFICIENT}:
            values[active_index] = 0.5
        targets.append(values)
    return torch.tensor(targets, dtype=torch.float32)


def _constraint_loss(outputs: Mapping[str, Tensor], state_targets: Tensor, observations: Sequence[DecisionExample]) -> Tensor:
    """Penalize overconfident beliefs/actions on invalid public states."""

    invalid = state_targets.eq(STATE_SET.index(EvidenceState.INVALID)).float()
    if not bool(torch.any(invalid)):
        return outputs["action_logits"].sum() * 0.0
    belief = torch.sigmoid(outputs["belief_logits"])
    active_indices = torch.tensor(
        [1 if example.observation.active_hypothesis_id == "H_B" else 0 for example in observations],
        dtype=torch.long,
        device=belief.device,
    )
    active_belief = belief.gather(1, active_indices.unsqueeze(1)).squeeze(1)
    action_max = F.softmax(outputs["action_logits"], dim=-1).max(dim=-1).values
    neutral = (active_belief - 0.5).square()
    overconfident_action = F.relu(action_max - 0.70).square()
    return ((neutral + overconfident_action) * invalid).sum() / invalid.sum().clamp_min(1.0)


class DifferentiableStrategyTrainer:
    """Train one named method with a shared, bounded CPU optimizer budget."""

    def __init__(self, config: Optional[DifferentiableTrainerConfig] = None):
        self.config = config or DifferentiableTrainerConfig()

    def _new_policy(self) -> DifferentiableStrategyPolicy:
        return DifferentiableStrategyPolicy(hidden_dim=self.config.hidden_dim, seed=self.config.seed)

    def _log_epoch(
        self,
        log: DifferentiableTrainingLog,
        losses: Sequence[Tensor | float],
        *,
        option_loss: Tensor | float = 0.0,
        flip_loss: Tensor | float = 0.0,
        state_loss: Tensor | float = 0.0,
        belief_loss: Tensor | float = 0.0,
        kl_loss: Tensor | float = 0.0,
        entropy: Tensor | float = 0.0,
        gradient_norm: float = 0.0,
        effective_sample_size: float = 0.0,
        flip_gradient_norm: float = 0.0,
        parameter_update_norm: float = 0.0,
        flip_updates_applied: int = 0,
        constraint_loss: Tensor | float = 0.0,
    ) -> None:
        def scalar(value: Tensor | float) -> float:
            return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        log.epochs.append({
            "epoch": float(len(log.epochs)),
            "total_loss": sum(scalar(value) for value in losses) / max(1, len(losses)),
            "option_loss": scalar(option_loss),
            "flip_loss": scalar(flip_loss),
            "state_loss": scalar(state_loss),
            "belief_loss": scalar(belief_loss),
            "reference_kl": scalar(kl_loss),
            "policy_entropy": scalar(entropy),
            "gradient_norm": float(gradient_norm),
            "effective_sample_size": float(effective_sample_size),
            "flip_gradient_norm": float(flip_gradient_norm),
            "parameter_update_norm": float(parameter_update_norm),
            "flip_updates_applied": float(flip_updates_applied),
            "constraint_loss": scalar(constraint_loss),
        })

    def _step(self, optimizer: torch.optim.Optimizer, loss: Tensor) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], self.config.gradient_clip_norm))
        optimizer.step()
        return gradient_norm

    def fit(self, dataset: DecisionDataset, method: str) -> Tuple[DifferentiableStrategyPolicy, DifferentiableTrainingLog]:
        method = str(method)
        if not dataset.examples:
            raise ValueError("cannot train on an empty decision dataset")
        train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
        train_indices = {
            index for index, example in enumerate(dataset.examples)
            if example.split == "train"
        }
        # Never train on reversal pairs whose observations belong to dev or
        # diagnostic-OOD.  Pair supervision is frozen evaluator data, but it is
        # still split-specific and must obey the same question-level holdout as
        # ordinary action examples.
        train_reversals = [
            pair for pair in dataset.reversals
            if pair.confirmed and pair.left in train_indices and pair.right in train_indices
        ]
        policy = self._new_policy()
        reference = policy.clone_frozen()
        # A branch group is one frozen public state with its candidate-action
        # vector; ``branch_count`` is the number of actions inside that group and
        # must not be reported as additional groups.
        log = DifferentiableTrainingLog(method=method, environment_branch_groups=len(train))
        log.parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        generator = torch.Generator().manual_seed(self.config.seed)
        state_only = method == "StateGateOnly"
        supervised = method in {"SFT", "StateGateOnly"}
        use_branch = method in {"PESCO-BranchOnly", "PESCO-NoFlipLoss", "PESCO-Full"}
        use_flip = method == "PESCO-Full"
        # Ablations are semantically isolated: NoBranch has no counterfactual branch
        # advantage, BranchOnly has no state auxiliary, and NoFlip retains branch +
        # state terms but omits the reversal loss.
        four_state = method in {"GRPO-FourState", "PESCO-NoBranch", "PESCO-NoFlipLoss", "PESCO-Full"}
        # Keep each ablation semantically isolated.  SFT is action supervision only;
        # GRPO-Terminal uses terminal utility plus the reference/entropy terms;
        # StateGateOnly receives only the state gate; and the four-state controls
        # (GRPO-FourState/PESCO-NoBranch/NoFlip/Full) receive the evidence-state,
        # belief, and constraint auxiliaries.  None of these terms reads the hidden
        # target-action table.
        use_belief_auxiliary = four_state
        use_constraint_auxiliary = four_state
        log.objective_components = {
            "option": not state_only,
            "flip": bool(use_flip),
            "state": bool(state_only or four_state),
            "belief": bool(use_belief_auxiliary),
            "kl": True,
            "constraint": bool(use_constraint_auxiliary),
        }
        if supervised:
            # The hidden benchmark target-action table is an audit label only.  The
            # supervised baseline gets the same evaluator-owned public branch winner
            # as every other method, avoiding privileged target-table supervision.
            log.supervised_target_source = "public_branch_utility_best_action"
        # State-gated SMOPD is trained by a separate teacher routine below.
        if method == "Evidence-Gated SMOPD":
            log.objective_components = {
                "option": True,
                "flip": False,
                "state": True,
                "belief": True,
                "kl": True,
                "constraint": True,
                "teacher_distillation": True,
            }
            return self._fit_smopd(dataset, policy, optimizer, reference, log, generator)

        # ``epochs`` is a minimum diagnostic horizon.  To make matched compute
        # literal, continue cycling the fixed train batches until the optimizer-step
        # cap is consumed by every method.
        target_steps = max(1, int(self.config.max_optimizer_steps))
        epoch = 0
        all_flip_gradients: List[float] = []
        all_flip_updates: List[float] = []
        while log.optimizer_steps < target_steps:
            epoch_losses: List[Tensor] = []
            epoch_option: List[Tensor] = []
            epoch_state: List[Tensor] = []
            epoch_belief: List[Tensor] = []
            epoch_kl: List[Tensor] = []
            epoch_entropy: List[Tensor] = []
            epoch_grad: List[float] = []
            epoch_ess: List[float] = []
            epoch_flip: List[Tensor] = []
            epoch_flip_gradients: List[float] = []
            epoch_flip_updates: List[float] = []
            epoch_constraint: List[Tensor] = []
            for indices in _batch_indices(len(train), self.config.batch_size, generator):
                features, utilities, state_targets, best_targets = _stack_examples(train, indices)
                supervised_targets = torch.tensor(
                    [train[int(index)].best_action_index for index in indices.tolist()],
                    dtype=torch.long,
                )
                outputs = policy(features)
                action_logits = outputs["action_logits"]
                state_loss = F.cross_entropy(outputs["state_logits"], state_targets)
                selected_examples = [train[int(index)] for index in indices.tolist()]
                belief_targets_matrix = _belief_target_matrix(selected_examples).to(outputs["belief_logits"].device)
                belief_loss = F.binary_cross_entropy_with_logits(outputs["belief_logits"], belief_targets_matrix)
                constraint_loss = _constraint_loss(outputs, state_targets, selected_examples)
                probabilities = F.softmax(action_logits / max(1e-6, self.config.temperature), dim=-1)
                entropy = _entropy(probabilities)
                kl = _mean_kl(action_logits, reference(features)["action_logits"])
                if supervised:
                    # SFT consumes the public branch-utility winner.  It does not
                    # read metadata[target_action], which is retained only for audit.
                    option_loss = F.cross_entropy(action_logits, supervised_targets)
                    if state_only:
                        option_loss = option_loss * 0.0
                elif use_branch:
                    centered = utilities - (utilities.sum(dim=-1, keepdim=True) - utilities) / max(1, utilities.shape[-1] - 1)
                    scale = centered.std(unbiased=False).clamp_min(1e-4)
                    advantages = centered / scale
                    option_loss = -(probabilities * advantages.detach()).sum(dim=-1).mean()
                    ess = float((advantages.abs().sum() ** 2 / advantages.square().sum().clamp_min(1e-6)).detach())
                    epoch_ess.append(ess)
                else:
                    # Terminal GRPO uses only the sampled terminal reward and a
                    # batch baseline; it does not inspect the other counterfactual
                    # branches when constructing its policy gradient.
                    sampled = torch.multinomial(probabilities.detach(), 1, generator=generator).squeeze(-1)
                    terminal = utilities.gather(1, sampled.unsqueeze(1)).squeeze(1)
                    advantages = (terminal - terminal.mean()) / terminal.std(unbiased=False).clamp_min(1e-4)
                    option_loss = -(F.log_softmax(action_logits, dim=-1).gather(1, sampled.unsqueeze(1)).squeeze(1) * advantages.detach()).mean()
                    epoch_ess.append(float((advantages.abs().sum() ** 2 / advantages.square().sum().clamp_min(1e-6)).detach()))
                if state_only or four_state:
                    # Evidence-conditioned reward is a real auxiliary CE objective,
                    # not a post-hoc state label used only for reporting.
                    state_weight = self.config.state_loss_weight
                else:
                    state_weight = 0.0
                flip_loss = torch.tensor(0.0)
                if use_flip and train_reversals:
                    pair_values: List[Tensor] = []
                    for pair in train_reversals:
                        if pair.left >= len(dataset.examples) or pair.right >= len(dataset.examples):
                            continue
                        pair_features = torch.stack([
                            observation_to_features(dataset.examples[pair.left].observation),
                            observation_to_features(dataset.examples[pair.right].observation),
                        ])
                        current = F.log_softmax(policy(pair_features)["action_logits"], dim=-1)
                        frozen = F.log_softmax(reference(pair_features)["action_logits"], dim=-1)
                        pair_values.append(preference_reversal_loss(
                            current[0, ACTION_SET.index(pair.action_left)],
                            current[0, ACTION_SET.index(pair.action_right)],
                            current[1, ACTION_SET.index(pair.action_left)],
                            current[1, ACTION_SET.index(pair.action_right)],
                            beta=1.0,
                            reference_a_left=frozen[0, ACTION_SET.index(pair.action_left)],
                            reference_a_right=frozen[0, ACTION_SET.index(pair.action_right)],
                            reference_b_left=frozen[1, ACTION_SET.index(pair.action_left)],
                            reference_b_right=frozen[1, ACTION_SET.index(pair.action_right)],
                            confirmed=pair.confirmed,
                            weight=pair.weight,
                        ))
                    if pair_values:
                        flip_loss = torch.stack(pair_values).mean()
                # Measure the gradient contributed by the reversal term itself,
                # before the combined objective update.  ``retain_graph`` keeps the
                # graph available for the real backward/optimizer step below.
                flip_gradient_norm = 0.0
                # A periodic gradient probe keeps the CPU reference inexpensive
                # while still proving that the flip term has a live autograd path.
                # The actual combined objective is optimized on every step.
                measure_flip_gradient = bool(
                    use_flip
                    and flip_loss.requires_grad
                    and (log.optimizer_steps % 16 == 0)
                )
                if measure_flip_gradient:
                    flip_gradients = torch.autograd.grad(
                        flip_loss,
                        tuple(policy.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    flip_gradient_norm = math.sqrt(sum(
                        float(gradient.detach().square().sum())
                        for gradient in flip_gradients
                        if gradient is not None
                    ))
                epoch_flip.append(flip_loss.detach())
                belief_weight = self.config.belief_loss_weight if use_belief_auxiliary else 0.0
                constraint_weight = (
                    self.config.constraint_loss_weight if use_constraint_auxiliary else 0.0
                )
                total = option_loss + state_weight * state_loss + belief_weight * belief_loss + self.config.flip_loss_weight * flip_loss + self.config.kl_weight * kl + constraint_weight * constraint_loss - self.config.entropy_weight * entropy
                before_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
                grad = self._step(optimizer, total)
                update_norm = math.sqrt(sum(
                    float((parameter.detach() - before).square().sum())
                    for parameter, before in zip(policy.parameters(), before_parameters)
                ))
                epoch_losses.append(total.detach())
                epoch_option.append(option_loss.detach())
                epoch_state.append(state_loss.detach())
                epoch_belief.append(belief_loss.detach())
                epoch_constraint.append(constraint_loss.detach())
                epoch_kl.append(kl.detach())
                epoch_entropy.append(entropy.detach())
                epoch_grad.append(grad)
                if measure_flip_gradient:
                    epoch_flip_gradients.append(flip_gradient_norm)
                    all_flip_gradients.append(flip_gradient_norm)
                epoch_flip_updates.append(update_norm)
                all_flip_updates.append(update_norm)
                log.optimizer_steps += 1
                if log.optimizer_steps >= target_steps:
                    break
            self._log_epoch(
                log,
                epoch_losses,
                option_loss=torch.stack(epoch_option).mean() if epoch_option else 0.0,
                flip_loss=torch.stack(epoch_flip).mean() if epoch_flip else 0.0,
                state_loss=torch.stack(epoch_state).mean() if epoch_state else 0.0,
                belief_loss=torch.stack(epoch_belief).mean() if epoch_belief else 0.0,
                kl_loss=torch.stack(epoch_kl).mean() if epoch_kl else 0.0,
                entropy=torch.stack(epoch_entropy).mean() if epoch_entropy else 0.0,
                gradient_norm=sum(epoch_grad) / max(1, len(epoch_grad)),
                effective_sample_size=sum(epoch_ess) / max(1, len(epoch_ess)),
                flip_gradient_norm=sum(epoch_flip_gradients) / max(1, len(epoch_flip_gradients)),
                parameter_update_norm=sum(epoch_flip_updates) / max(1, len(epoch_flip_updates)),
                flip_updates_applied=sum(float(value) > 0.0 for value in epoch_flip),
                constraint_loss=torch.stack(epoch_constraint).mean() if epoch_constraint else 0.0,
            )
            epoch += 1
            if log.optimizer_steps >= target_steps:
                break
        log.reversal_count = len(train_reversals) if use_flip else 0
        if use_flip and train_reversals:
            log.flip_gradient_norm = sum(all_flip_gradients) / max(1, len(all_flip_gradients))
            log.flip_update_norm = sum(all_flip_updates) / max(1, len(all_flip_updates))
            log.flip_updates_applied = len(all_flip_updates)
            log.flip_gradient_probe_count = len(all_flip_gradients)
        log.matched_budget_steps = target_steps
        log.constraint_loss_mean = sum(float(epoch.get("constraint_loss", 0.0)) for epoch in log.epochs) / max(1, len(log.epochs))
        return policy, log

    def _fit_smopd(
        self,
        dataset: DecisionDataset,
        policy: DifferentiableStrategyPolicy,
        optimizer: torch.optim.Optimizer,
        reference: DifferentiableStrategyPolicy,
        log: DifferentiableTrainingLog,
        generator: torch.Generator,
    ) -> Tuple[DifferentiableStrategyPolicy, DifferentiableTrainingLog]:
        """Genuine evidence-gated multi-teacher distillation.

        Four specialist CPU teachers are fitted on the same train split, each with a
        state-conditioned action target.  The student receives a differentiable
        mixture weighted by its evidence-state probabilities; this is the mechanism
        boundary of the SMOPD adapter and is not claimed to reproduce a paper's LLM.
        """

        train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
        teachers: Dict[EvidenceState, DifferentiableStrategyPolicy] = {}
        # Count teacher and student optimizer steps against one shared compute cap.
        # Reserve at most half for specialists while always leaving at least one
        # student step.  Distribute a non-divisible teacher budget across states so
        # tiny test budgets cannot silently overshoot the cap.
        target_steps = max(1, int(self.config.max_optimizer_steps))
        teacher_budget = min(target_steps // 2, max(0, target_steps - 1))
        teacher_base, teacher_remainder = divmod(teacher_budget, len(STATE_SET))
        teacher_step_counts = [
            teacher_base + int(index < teacher_remainder)
            for index in range(len(STATE_SET))
        ]
        for state in STATE_SET:
            state_index = STATE_SET.index(state)
            teacher = self._new_policy()
            teachers[state] = teacher
            teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=self.config.learning_rate)
            teacher_examples = [example for example in train if example.state_target is state] or train
            for step in range(teacher_step_counts[state_index]):
                indices = torch.randint(len(teacher_examples), (min(self.config.batch_size, len(teacher_examples)),), generator=generator)
                features, _, states, best = _stack_examples(teacher_examples, indices)
                outputs = teacher(features)
                loss = F.cross_entropy(outputs["action_logits"], best) + self.config.state_loss_weight * F.cross_entropy(outputs["state_logits"], states)
                self._step(teacher_optimizer, loss)
                log.teacher_optimizer_steps += 1
                log.optimizer_steps += 1
        log.teacher_parameter_count = sum(
            sum(parameter.numel() for parameter in teacher.parameters())
            for teacher in teachers.values()
        )
        # Spend the remaining shared budget on the gated student.  As with the
        # other methods, cycle the frozen train batches so the logged optimizer
        # count reaches the configured cap rather than silently stopping after
        # ``epochs * ceil(n/batch_size)`` updates.
        while log.optimizer_steps < target_steps:
            losses: List[Tensor] = []
            states_losses: List[Tensor] = []
            beliefs_losses: List[Tensor] = []
            constraints_losses: List[Tensor] = []
            kls: List[Tensor] = []
            ents: List[Tensor] = []
            grads: List[float] = []
            for indices in _batch_indices(len(train), self.config.batch_size, generator):
                features, _, state_targets, _ = _stack_examples(train, indices)
                outputs = policy(features)
                state_probs = F.softmax(outputs["state_logits"], dim=-1)
                teacher_probs = torch.stack([
                    F.softmax(teachers[state](features)["action_logits"].detach(), dim=-1)
                    for state in STATE_SET
                ], dim=1)
                mixture = (state_probs.unsqueeze(-1) * teacher_probs).sum(dim=1).clamp_min(1e-7)
                student_log = F.log_softmax(outputs["action_logits"], dim=-1)
                distill = F.kl_div(student_log, mixture.detach(), reduction="batchmean")
                state_loss = F.cross_entropy(outputs["state_logits"], state_targets)
                selected_examples = [train[int(index)] for index in indices.tolist()]
                belief_targets = _belief_target_matrix(selected_examples).to(outputs["belief_logits"].device)
                belief_loss = F.binary_cross_entropy_with_logits(outputs["belief_logits"], belief_targets)
                constraint_loss = _constraint_loss(outputs, state_targets, selected_examples)
                kl = _mean_kl(outputs["action_logits"], reference(features)["action_logits"])
                entropy = _entropy(F.softmax(outputs["action_logits"], dim=-1))
                total = distill + self.config.state_loss_weight * state_loss + self.config.belief_loss_weight * belief_loss + self.config.constraint_loss_weight * constraint_loss + self.config.kl_weight * kl - self.config.entropy_weight * entropy
                grads.append(self._step(optimizer, total))
                losses.append(total.detach()); states_losses.append(state_loss.detach()); beliefs_losses.append(belief_loss.detach()); constraints_losses.append(constraint_loss.detach()); kls.append(kl.detach()); ents.append(entropy.detach())
                log.optimizer_steps += 1
                if log.optimizer_steps >= target_steps:
                    break
            self._log_epoch(log, losses, option_loss=torch.stack(losses).mean() if losses else 0.0, state_loss=torch.stack(states_losses).mean() if states_losses else 0.0, belief_loss=torch.stack(beliefs_losses).mean() if beliefs_losses else 0.0, constraint_loss=torch.stack(constraints_losses).mean() if constraints_losses else 0.0, kl_loss=torch.stack(kls).mean() if kls else 0.0, entropy=torch.stack(ents).mean() if ents else 0.0, gradient_norm=sum(grads) / max(1, len(grads)))
            log.constraint_loss_mean = sum(float(value.detach()) for value in constraints_losses) / max(1, len(constraints_losses))
            if log.optimizer_steps >= target_steps:
                break
        log.reversal_count = 0
        log.matched_budget_steps = target_steps
        log.constraint_loss_mean = sum(float(epoch.get("constraint_loss", 0.0)) for epoch in log.epochs) / max(1, len(log.epochs))
        return policy, log


def policy_action(policy: DifferentiableStrategyPolicy, observation: Observation, *, state_gate: bool = False) -> ResearchAction:
    """Greedy public action used by matched evaluation."""

    with torch.no_grad():
        features = observation_to_features(observation)
        outputs = policy(features)
        if state_gate:
            state = STATE_SET[int(outputs["state_logits"].argmax(dim=-1).item())]
            mapping = {
                EvidenceState.SUPPORTED: ResearchAction.CONTINUE,
                EvidenceState.REFUTED: ResearchAction.SWITCH,
                EvidenceState.INSUFFICIENT: ResearchAction.SAMPLE,
                EvidenceState.INVALID: ResearchAction.REPAIR,
            }
            return mapping[state]
        return ACTION_SET[int(outputs["action_logits"].argmax(dim=-1).item())]


def policy_probabilities(policy: DifferentiableStrategyPolicy, observation: Observation) -> Dict[str, float]:
    with torch.no_grad():
        probabilities = F.softmax(policy(observation_to_features(observation))["action_logits"], dim=-1).squeeze(0)
    return {action.value: float(probabilities[index]) for index, action in enumerate(ACTION_SET)}


__all__ = [
    "ACTION_SET",
    "STATE_SET",
    "FEATURE_DIM",
    "DecisionExample",
    "DecisionDataset",
    "ReversalExample",
    "DifferentiableStrategyPolicy",
    "DifferentiableTrainerConfig",
    "DifferentiableTrainingLog",
    "DifferentiableStrategyTrainer",
    "observation_to_features",
    "policy_action",
    "policy_probabilities",
]

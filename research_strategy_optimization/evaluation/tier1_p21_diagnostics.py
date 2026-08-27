"""P2.1 algorithm diagnostics and constrained CPU reference training.

This module is deliberately separate from the preregistered P2 runner.  It is used
to diagnose the conflict between branch utility and reversal ranking after the
evaluator fixes: reversal labels are reduced to decision-relevant candidates,
weights are normalized within each question, component gradient cosines are
recorded, and a constrained PESCO variant can be compared on a fresh diagnostic
split.  Nothing here opens formal-model or Tier-2 authorization.
"""

from __future__ import annotations

import json
import math
import random
import gc
import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from ..algorithms.differentiable_strategy import (
    ACTION_SET,
    STATE_SET,
    DecisionDataset,
    DecisionExample,
    DifferentiableStrategyPolicy,
    DifferentiableStrategyTrainer,
    DifferentiableTrainerConfig,
    _batch_indices,
    _belief_target_matrix,
    _constraint_loss,
    _mean_kl,
    _stack_examples,
    observation_to_features,
    policy_action,
)
from ..algorithms.preference_reversal_loss import preference_reversal_loss
from ..schemas import EvidenceState, ResearchAction
from .tier1_differentiable_suite import evaluate_differentiable_policy


@dataclass(frozen=True)
class P21Config:
    """Bounded diagnostic settings; these are not formal-promotion settings."""

    top_k: int = 2
    minimum_confidence_margin: float = 0.0
    epsilon: float = 0.0
    lagrange_learning_rate: float = 0.05
    initial_lagrange: float = 0.0
    flip_weight: float = 0.5
    batch_size: int = 16
    max_optimizer_steps: int = 64
    epochs: int = 8
    seed: int = 17
    pcgrad: bool = True
    max_pairs_per_question: int = 2
    repair_safety_weight: float = 0.8


def _rank_indices(values: Sequence[float]) -> List[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))


def _pair_confidence_weight(pair: Any, *, minimum: float = 0.0) -> float:
    """Convert evaluator confidence margin into a positive training weight."""

    # lcb_left and ucb_right are the two one-sided confidence bounds stored with a
    # confirmed pair.  If a legacy artifact lacks them, retain a small finite weight
    # rather than silently deleting all reversals.
    left_margin = float(getattr(pair, "lcb_left", 0.0)) - float(getattr(pair, "margin", 0.05))
    right_margin = -float(getattr(pair, "ucb_right", 0.0)) - float(getattr(pair, "margin", 0.05))
    confidence = max(0.0, min(left_margin, right_margin))
    if confidence < float(minimum):
        return 0.0
    return max(1e-6, confidence)


def select_top_candidate_reversals(
    dataset: DecisionDataset,
    *,
    top_k: int = 2,
    minimum_confidence_margin: float = 0.0,
    max_pairs_per_question: int | None = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Keep only reversals whose requested actions are top-k at both endpoints.

    The returned weights sum to one independently within every question cluster.
    This prevents a question with many correlated endpoint/action combinations from
    dominating the reversal objective.
    """

    top_k = max(1, int(top_k))
    selected: List[Any] = []
    by_question: Dict[str, List[Any]] = {}
    dropped = {"unconfirmed": 0, "non_top_candidate": 0, "low_confidence": 0}
    for pair in dataset.reversals:
        if not bool(pair.confirmed):
            dropped["unconfirmed"] += 1
            continue
        if not (0 <= int(pair.left) < len(dataset.examples) and 0 <= int(pair.right) < len(dataset.examples)):
            dropped["non_top_candidate"] += 1
            continue
        left = dataset.examples[int(pair.left)]
        right = dataset.examples[int(pair.right)]
        if left.question_id != right.question_id:
            # A cross-question pair cannot be a within-question macro unit.
            dropped["non_top_candidate"] += 1
            continue
        left_rank = _rank_indices(left.branch_utilities)
        right_rank = _rank_indices(right.branch_utilities)
        a_left = ACTION_SET.index(pair.action_left)
        a_right = ACTION_SET.index(pair.action_right)
        if a_left not in left_rank[:top_k] or a_right not in right_rank[:top_k]:
            dropped["non_top_candidate"] += 1
            continue
        weight = _pair_confidence_weight(pair, minimum=minimum_confidence_margin)
        if weight <= 0.0:
            dropped["low_confidence"] += 1
            continue
        # ReversalExample is frozen; use a replacement with the diagnostic weight so
        # the source dataset remains untouched.
        clone = replace(pair, weight=float(weight))
        selected.append(clone)
        by_question.setdefault(str(left.question_id), []).append(clone)
    # Retain only the most confident decision-relevant candidates per question.  A
    # cap keeps the pairwise autograd graph bounded while preserving every question
    # as a macro unit; it is not a post-hoc performance filter.
    cap = None if max_pairs_per_question is None else max(1, int(max_pairs_per_question))
    if cap is not None:
        for question_id, pairs in list(by_question.items()):
            by_question[question_id] = sorted(
                pairs,
                key=lambda pair: float(getattr(pair, "weight", 0.0)),
                reverse=True,
            )[:cap]
    normalized: List[Any] = []
    for question_id, pairs in sorted(by_question.items()):
        # ``replace`` stores the confidence-derived weight in the public dataclass
        # field itself.  Do not look for a private side-channel attribute here:
        # doing so would divide by the pair count rather than by the confidence
        # mass and would violate the per-question normalization contract whenever
        # weights differ.
        total = sum(max(0.0, float(getattr(pair, "weight", 0.0))) for pair in pairs)
        if total <= 0.0:
            total = float(len(pairs))
        normalized_pair_map: Dict[int, Any] = {}
        for pair in pairs:
            normalized_pair = replace(
                pair,
                weight=max(0.0, float(getattr(pair, "weight", 0.0))) / max(total, 1e-12),
            )
            normalized_pair_map[id(pair)] = normalized_pair
        for index, pair in enumerate(pairs):
            normalized_pair = normalized_pair_map[id(pair)]
            pairs[index] = normalized_pair
            normalized.append(normalized_pair)
    audit = {
        "input_reversal_count": len(dataset.reversals),
        "selected_reversal_count": len(normalized),
        "question_count_with_reversals": len(by_question),
        "weights_sum_by_question": {
            question_id: sum(float(getattr(pair, "weight", 0.0)) for pair in pairs)
            for question_id, pairs in sorted(by_question.items())
        },
        "top_k": top_k,
        "minimum_confidence_margin": float(minimum_confidence_margin),
        "max_pairs_per_question": cap,
        "dropped": dropped,
        "weight_definition": "positive confidence margin min(lcb_left-margin, -ucb_right-margin), normalized within question",
    }
    return normalized, audit


def _flat_gradients(loss: Tensor, parameters: Sequence[Tensor], *, retain_graph: bool = True) -> Tensor:
    grads = torch.autograd.grad(loss, tuple(parameters), retain_graph=retain_graph, allow_unused=True)
    # Preserve one zero block per parameter.  Besides making cosine diagnostics
    # well-defined, this lets PCGrad write the combined flat vector back to the
    # original parameter shapes without losing positions for unused heads.
    pieces = [
        (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter, gradient in zip(parameters, grads)
    ]
    return torch.cat(pieces) if pieces else torch.zeros(1, dtype=loss.dtype, device=loss.device)


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = float(left.norm().detach()) * float(right.norm().detach())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).detach()) / denominator


def _flip_loss_for_pairs(
    policy: DifferentiableStrategyPolicy,
    reference: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    pairs: Sequence[Any],
) -> Tensor:
    values: List[Tensor] = []
    for pair in pairs:
        if not (0 <= int(pair.left) < len(dataset.examples) and 0 <= int(pair.right) < len(dataset.examples)):
            continue
        features = torch.stack([
            observation_to_features(dataset.examples[int(pair.left)].observation),
            observation_to_features(dataset.examples[int(pair.right)].observation),
        ])
        current = F.log_softmax(policy(features)["action_logits"], dim=-1)
        frozen = F.log_softmax(reference(features)["action_logits"], dim=-1)
        weight = float(getattr(pair, "weight", 1.0))
        values.append(weight * preference_reversal_loss(
            current[0, ACTION_SET.index(pair.action_left)],
            current[0, ACTION_SET.index(pair.action_right)],
            current[1, ACTION_SET.index(pair.action_left)],
            current[1, ACTION_SET.index(pair.action_right)],
            beta=1.0,
            reference_a_left=frozen[0, ACTION_SET.index(pair.action_left)],
            reference_a_right=frozen[0, ACTION_SET.index(pair.action_right)],
            reference_b_left=frozen[1, ACTION_SET.index(pair.action_left)],
            reference_b_right=frozen[1, ACTION_SET.index(pair.action_right)],
            confirmed=bool(pair.confirmed),
            weight=1.0,
        ))
    if not values:
        return next(policy.parameters()).sum() * 0.0
    return torch.stack(values).sum()


def _diagnostic_losses(
    policy: DifferentiableStrategyPolicy,
    reference: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    examples: Sequence[DecisionExample],
    pairs: Sequence[Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    features, utilities, state_targets, _ = _stack_examples(
        list(examples), torch.arange(len(examples), dtype=torch.long)
    )
    outputs = policy(features)
    probabilities = F.softmax(outputs["action_logits"], dim=-1)
    centered = utilities - (utilities.sum(dim=-1, keepdim=True) - utilities) / max(1, utilities.shape[-1] - 1)
    advantages = centered / centered.std(unbiased=False).clamp_min(1e-4)
    branch_loss = -(probabilities * advantages.detach()).sum(dim=-1).mean()
    state_loss = F.cross_entropy(outputs["state_logits"], state_targets)
    flip_loss = _flip_loss_for_pairs(policy, reference, dataset, pairs)
    return branch_loss, state_loss, flip_loss


def _normalized_regret(utilities: Sequence[float], selected_index: int) -> float:
    if not utilities:
        return 0.0
    best = max(float(value) for value in utilities)
    worst = min(float(value) for value in utilities)
    return max(0.0, (best - float(utilities[int(selected_index)])) / max(best - worst, 1e-8))


def _add_normalized_metrics(row: Dict[str, Any], policy: DifferentiableStrategyPolicy, dataset: DecisionDataset, split: str) -> Dict[str, Any]:
    examples = [example for example in dataset.examples if example.split == split]
    values: list[float] = []
    by_question: dict[str, list[float]] = {}
    for example in examples:
        action = policy_action(policy, example.observation)
        value = _normalized_regret(example.branch_utilities, ACTION_SET.index(action))
        values.append(value)
        by_question.setdefault(str(example.question_id), []).append(value)
    row["normalized_regret"] = sum(values) / len(values) if values else None
    row["normalized_regret_n"] = len(values)
    for question_row in row.get("question_metric_rows", []):
        qid = str(question_row.get("question_id"))
        qvalues = by_question.get(qid, [])
        question_row["normalized_regret"] = sum(qvalues) / len(qvalues) if qvalues else None
    return row


def measure_gradient_cosines(
    dataset: DecisionDataset,
    *,
    config: Optional[P21Config] = None,
) -> Dict[str, Any]:
    """Measure branch/flip and state/flip gradient alignment over train batches."""

    config = config or P21Config()
    pairs, reversal_audit = select_top_candidate_reversals(
        dataset,
        top_k=config.top_k,
        minimum_confidence_margin=config.minimum_confidence_margin,
        max_pairs_per_question=config.max_pairs_per_question,
    )
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    train_ids = {id(example) for example in train}
    pairs = [pair for pair in pairs if id(dataset.examples[int(pair.left)]) in train_ids and id(dataset.examples[int(pair.right)]) in train_ids]
    policy = DifferentiableStrategyPolicy(seed=config.seed)
    reference = policy.clone_frozen()
    generator = torch.Generator().manual_seed(config.seed)
    rows: List[Dict[str, float]] = []
    parameters = tuple(policy.parameters())
    for batch_indices in _batch_indices(len(train), config.batch_size, generator):
        batch = [train[int(index)] for index in batch_indices.tolist()]
        batch_ids = {id(example) for example in batch}
        local_pairs = [pair for pair in pairs if id(dataset.examples[int(pair.left)]) in batch_ids and id(dataset.examples[int(pair.right)]) in batch_ids]
        # A global flip gradient remains useful when a batch has no endpoint pair;
        # use the first normalized pair so every batch has a comparable probe.
        probe_pairs = local_pairs or pairs[:1]
        branch_loss, state_loss, flip_loss = _diagnostic_losses(policy, reference, dataset, batch, probe_pairs)
        branch_grad = _flat_gradients(branch_loss, parameters)
        state_grad = _flat_gradients(state_loss, parameters)
        flip_grad = _flat_gradients(flip_loss, parameters)
        rows.append({
            "batch_size": float(len(batch)),
            "pair_count": float(len(local_pairs)),
            "branch_loss": float(branch_loss.detach()),
            "state_loss": float(state_loss.detach()),
            "flip_loss": float(flip_loss.detach()),
            "cos_branch_flip": _cosine(branch_grad, flip_grad),
            "cos_state_flip": _cosine(state_grad, flip_grad),
        })
    def mean(key: str) -> Optional[float]:
        values = [row[key] for row in rows if math.isfinite(row[key])]
        return sum(values) / len(values) if values else None
    return {
        "schema_version": "pesco_tier1_p21_gradient_diagnostics_v0.1",
        "config": asdict(config),
        "batch_count": len(rows),
        "rows": rows,
        "mean_cos_branch_flip": mean("cos_branch_flip"),
        "mean_cos_state_flip": mean("cos_state_flip"),
        "negative_branch_flip_fraction": sum(row["cos_branch_flip"] < 0 for row in rows) / max(1, len(rows)),
        "negative_state_flip_fraction": sum(row["cos_state_flip"] < 0 for row in rows) / max(1, len(rows)),
        "reversal_audit": reversal_audit,
        "train_reversal_count": len(pairs),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    }


def train_constrained_pesco(
    dataset: DecisionDataset,
    *,
    config: Optional[P21Config] = None,
) -> Tuple[DifferentiableStrategyPolicy, Dict[str, Any]]:
    """Train a branch+rank policy with a NoFlip utility floor.

    A frozen NoFlip policy supplies the reference expected utility.  A dynamic
    non-negative multiplier penalizes batches where the current policy falls below
    that reference by more than ``epsilon``.  If ``pcgrad`` is enabled, the flip
    gradient is projected away from a conflicting branch gradient before combining
    the objectives.  This is a bounded mechanism diagnostic, not a formal claim.
    """

    config = config or P21Config()
    trainer_config = DifferentiableTrainerConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        max_optimizer_steps=config.max_optimizer_steps,
        seed=config.seed,
    )
    # P2.2 feedback requires the constrained candidate to start from the same
    # public-winner SFT checkpoint as every other continuation method.  The prior
    # diagnostic used a NoFlip reference and randomly reinitialized the constrained
    # policy, which made its utility floor incomparable and allowed early collapse.
    baseline, baseline_log = DifferentiableStrategyTrainer(trainer_config).fit(dataset, "SFT")
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    policy = copy.deepcopy(baseline)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    reference = baseline.clone_frozen()
    optimizer = torch.optim.Adam(policy.parameters(), lr=trainer_config.learning_rate)
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    train_indices = {index for index, example in enumerate(dataset.examples) if example.split == "train"}
    pairs, reversal_audit = select_top_candidate_reversals(
        dataset,
        top_k=config.top_k,
        minimum_confidence_margin=config.minimum_confidence_margin,
        max_pairs_per_question=config.max_pairs_per_question,
    )
    pairs = [pair for pair in pairs if int(pair.left) in train_indices and int(pair.right) in train_indices]
    generator = torch.Generator().manual_seed(config.seed)
    lagrange = max(0.0, float(config.initial_lagrange))
    rows: List[Dict[str, float]] = []
    cosine_rows: List[Dict[str, float]] = []
    parameters = tuple(policy.parameters())
    steps = 0
    while steps < max(1, int(config.max_optimizer_steps)):
        for batch_indices in _batch_indices(len(train), config.batch_size, generator):
            batch = [train[int(index)] for index in batch_indices.tolist()]
            branch_loss, state_loss, flip_loss = _diagnostic_losses(policy, reference, dataset, batch, pairs)
            features, utilities, _, _ = _stack_examples(
                batch, torch.arange(len(batch), dtype=torch.long)
            )
            policy_outputs = policy(features)
            current_prob = F.softmax(policy_outputs["action_logits"], dim=-1)
            non_invalid = (torch.tensor([example.state_target is not EvidenceState.INVALID for example in batch], dtype=current_prob.dtype) > 0).to(current_prob.dtype)
            safety_loss = (current_prob[:, ACTION_SET.index(ResearchAction.REPAIR)] * non_invalid).mean()
            with torch.no_grad():
                baseline_prob = F.softmax(baseline(features)["action_logits"], dim=-1)
                baseline_expected = (baseline_prob * utilities).sum(dim=-1).mean()
            current_expected = (current_prob * utilities).sum(dim=-1).mean()
            violation = F.relu(baseline_expected - current_expected - float(config.epsilon))
            branch_grad = _flat_gradients(branch_loss, parameters)
            flip_grad = _flat_gradients(flip_loss, parameters)
            state_grad = _flat_gradients(state_loss, parameters)
            cosine_rows.append({
                "step": float(steps),
                "cos_branch_flip": _cosine(branch_grad, flip_grad),
                "cos_state_flip": _cosine(state_grad, flip_grad),
                "constraint_violation": float(violation.detach()),
            })
            if config.pcgrad:
                dot = torch.dot(flip_grad, branch_grad)
                denominator = torch.dot(branch_grad, branch_grad).clamp_min(1e-12)
                projected_flip = flip_grad - torch.minimum(dot, torch.zeros_like(dot)) / denominator * branch_grad
                combined = branch_grad + float(config.flip_weight) * projected_flip + float(trainer_config.state_loss_weight) * state_grad
                safety_grad = _flat_gradients(safety_loss, parameters)
                combined = combined + float(config.repair_safety_weight) * safety_grad
                if float(lagrange) > 0.0:
                    constraint_grad = _flat_gradients(violation, parameters)
                    combined = combined + float(lagrange) * constraint_grad
                optimizer.zero_grad(set_to_none=True)
                cursor = 0
                for parameter in parameters:
                    count = parameter.numel()
                    parameter.grad = combined[cursor:cursor + count].reshape_as(parameter).clone()
                    cursor += count
                torch.nn.utils.clip_grad_norm_(parameters, trainer_config.gradient_clip_norm)
                optimizer.step()
            else:
                total = branch_loss + float(config.flip_weight) * flip_loss + float(trainer_config.state_loss_weight) * state_loss + float(config.repair_safety_weight) * safety_loss + float(lagrange) * violation
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, trainer_config.gradient_clip_norm)
                optimizer.step()
            lagrange = max(0.0, lagrange + float(config.lagrange_learning_rate) * float(violation.detach()))
            rows.append({
                "step": float(steps),
                "branch_loss": float(branch_loss.detach()),
                "state_loss": float(state_loss.detach()),
                "flip_loss": float(flip_loss.detach()),
                "current_expected_utility": float(current_expected.detach()),
                "baseline_expected_utility": float(baseline_expected.detach()),
                "constraint_violation": float(violation.detach()),
                "repair_safety_loss": float(safety_loss.detach()),
                "lagrange": float(lagrange),
            })
            steps += 1
            if steps >= max(1, int(config.max_optimizer_steps)):
                break
        if steps >= max(1, int(config.max_optimizer_steps)):
            break
    log = {
        "schema_version": "pesco_tier1_p21_constrained_training_v0.1",
        "method": "PESCO-Constrained-PCGrad" if config.pcgrad else "PESCO-Constrained-Lagrangian",
        "config": asdict(config),
        "optimizer_steps": steps,
        "rows": rows,
        "gradient_cosines": cosine_rows,
        "mean_cos_branch_flip": sum(row["cos_branch_flip"] for row in cosine_rows) / max(1, len(cosine_rows)),
        "mean_cos_state_flip": sum(row["cos_state_flip"] for row in cosine_rows) / max(1, len(cosine_rows)),
        "final_lagrange": lagrange,
        "baseline_method": "SFT",
        "baseline_optimizer_steps": baseline_log.optimizer_steps,
        "reversal_audit": reversal_audit,
        "train_reversal_count": len(pairs),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    }
    return policy, log


def run_p21_diagnostic(
    output_dir: str | Path,
    dataset: DecisionDataset,
    *,
    config: Optional[P21Config] = None,
    methods: Sequence[str] = ("SFT", "GRPO-Terminal", "GRPO-FourState", "PESCO-BranchOnly", "PESCO-NoFlipLoss", "PESCO-Full", "Evidence-Gated SMOPD"),
    retain_example_records: bool = False,
    include_gradient: bool = True,
    include_constrained: bool = True,
) -> Dict[str, Any]:
    """Run bounded gradient diagnostics and method comparison on all present splits.

    Example-level evaluator rows are intentionally omitted by default.  Keeping all
    action-probability dictionaries for every method can exceed the small CPU
    runner's memory limit after several PyTorch policies; aggregate and
    question-macro rows remain sufficient for the P2.1 diagnosis.  Callers that
    need a forensic dump may opt in explicitly.
    """

    config = config or P21Config()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gradient = measure_gradient_cosines(dataset, config=config) if include_gradient else {
        "status": "not_run",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    }
    (output / "gradient_diagnostics.json").write_text(json.dumps(gradient, indent=2, ensure_ascii=False), encoding="utf-8")
    # Use the same decision-relevant reversal construction for the actual Full
    # training run: top-candidate pairs, confidence-margin weights, and per-question
    # normalization.  Evaluation still receives the untouched audit dataset.
    selected_pairs, reversal_audit = select_top_candidate_reversals(
        dataset,
        top_k=config.top_k,
        minimum_confidence_margin=config.minimum_confidence_margin,
        max_pairs_per_question=config.max_pairs_per_question,
    )
    training_dataset = DecisionDataset(
        examples=list(dataset.examples),
        reversals=selected_pairs,
        schema_version=dataset.schema_version,
        provenance=dict(dataset.provenance),
    )
    records: List[Dict[str, Any]] = []
    logs: Dict[str, Any] = {}
    for method in methods:
        trainer_config = DifferentiableTrainerConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            max_optimizer_steps=config.max_optimizer_steps,
            seed=config.seed,
        )
        policy, log = DifferentiableStrategyTrainer(trainer_config).fit(training_dataset, method)
        logs[method] = log.to_dict()
        for split in sorted({str(example.split) for example in dataset.examples}):
            row = evaluate_differentiable_policy(policy, dataset, split)
            row = _add_normalized_metrics(row, policy, dataset, split)
            if not retain_example_records:
                row = dict(row)
                row["record_count"] = len(row.get("records", ()))
                row.pop("records", None)
            row.update({"method": method, "seed": config.seed, "diagnostic_only": True})
            records.append(row)
        # PyTorch's CPU allocator retains freed blocks.  The diagnostic runs many
        # independently initialized policies; explicitly release each one so a
        # clean bounded run does not get killed by cumulative allocator pressure.
        del policy
        gc.collect()
    constrained_log = None
    if include_constrained:
        constrained, constrained_log = train_constrained_pesco(dataset, config=config)
        logs[constrained_log["method"]] = constrained_log
        for split in sorted({str(example.split) for example in dataset.examples}):
            row = evaluate_differentiable_policy(constrained, dataset, split)
            row = _add_normalized_metrics(row, constrained, dataset, split)
            if not retain_example_records:
                row = dict(row)
                row["record_count"] = len(row.get("records", ()))
                row.pop("records", None)
            row.update({"method": constrained_log["method"], "seed": config.seed, "diagnostic_only": True})
            records.append(row)
        del constrained
        gc.collect()
    result = {
        "schema_version": "pesco_tier1_p21_diagnostic_v0.1",
        "config": asdict(config),
        "methods": list(methods) + ([constrained_log["method"]] if constrained_log is not None else []),
        "records": records,
        "training_logs": logs,
        "gradient_diagnostics": gradient,
        "reversal_training_audit": reversal_audit,
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
    }
    (output / "p21_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


__all__ = [
    "P21Config",
    "select_top_candidate_reversals",
    "measure_gradient_cosines",
    "train_constrained_pesco",
    "run_p21_diagnostic",
]

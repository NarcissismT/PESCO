"""P2.2 common-SFT initialization and objective-alignment diagnostics.

This module is intentionally a diagnostic runner.  Every method is initialized
from the same, seed-specific SFT checkpoint and then receives an equal fine-tuning
budget.  Reversal supervision is restricted to canonical top-1 pairs with a
minimum top1--top2 gap and unique endpoint winners.  The utility floor is defined
relative to the frozen SFT policy, while KL and entropy safeguards are applied to
all repaired objectives.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

# The reference runtime is intentionally single-threaded.  This keeps isolated
# seed/method workers below the container's memory ceiling and makes the measured
# optimizer budget independent of OpenMP allocator behavior.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

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
from .tier1_p21_diagnostics import select_top_candidate_reversals


P22_METHODS: Tuple[str, ...] = (
    "SFT-Continued",
    "SFT→BranchOnly",
    "SFT→NoFlip",
    "SFT→Pairwise-Full",
    "SFT→Listwise-Full",
    "SFT→Listwise+PCGrad",
    "SFT→Pairwise+Listwise",
)


@dataclass(frozen=True)
class P22Config:
    seed: int = 17
    sft_steps: int = 32
    finetune_steps: int = 32
    batch_size: int = 16
    sft_learning_rate: float = 3e-3
    learning_rate: float = 2e-3
    hidden_dim: int = 48
    utility_temperature: float = 0.25
    utility_target_weight: float = 1.0
    utility_hard_weight: float = 0.0
    top1_margin_weight: float = 0.0
    top1_margin: float = 0.05
    branch_loss_weight: float = 0.25
    top1_gap_threshold: float = 0.05
    max_pairs_per_question: int = 2
    pairwise_weight: float = 0.35
    listwise_weight: float = 0.65
    state_weight: float = 0.20
    belief_weight: float = 0.08
    base_kl_weight: float = 0.02
    target_kl: float = 0.05
    adaptive_kl_gain: float = 0.50
    entropy_floor: float = 0.55
    entropy_floor_weight: float = 0.15
    constraint_weight: float = 0.10
    repair_safety_weight: float = 0.0
    utility_floor_epsilon: float = 0.0
    lagrange_learning_rate: float = 0.10
    gradient_clip_norm: float = 5.0
    bootstrap_replicates: int = 2000
    # Public evidence-state class weights.  The default is neutral for backwards
    # compatibility; P2.3.3 can register an INVALID-upweighted vector.
    state_class_weights: Tuple[float, ...] | None = None


def _utility_soft_targets(utilities: Tensor, temperature: float) -> Tensor:
    """q(a|x) ∝ exp((U(a)-max U)/tau), with a numerically stable shift."""

    tau = max(1e-6, float(temperature))
    return F.softmax((utilities - utilities.max(dim=-1, keepdim=True).values) / tau, dim=-1)


def _canonical_reversals(
    dataset: DecisionDataset,
    config: P22Config,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Select unique-winner, top-1 canonical pairs and normalize per question."""

    candidates, audit = select_top_candidate_reversals(
        dataset,
        top_k=1,
        max_pairs_per_question=config.max_pairs_per_question,
    )
    by_question: Dict[str, List[Any]] = {}
    dropped_gap = 0
    dropped_winner = 0
    for pair in candidates:
        left = dataset.examples[int(pair.left)]
        right = dataset.examples[int(pair.right)]
        def unique_top1(example: DecisionExample) -> Tuple[int, float, bool]:
            values = sorted((float(v) for v in example.branch_utilities), reverse=True)
            winner = int(max(range(len(example.branch_utilities)), key=lambda i: example.branch_utilities[i]))
            gap = values[0] - values[1] if len(values) > 1 else float("inf")
            unique = len(values) < 2 or values[0] > values[1]
            return winner, gap, unique
        left_winner, left_gap, left_unique = unique_top1(left)
        right_winner, right_gap, right_unique = unique_top1(right)
        if not (left_unique and right_unique):
            dropped_winner += 1
            continue
        if left_gap < float(config.top1_gap_threshold) or right_gap < float(config.top1_gap_threshold):
            dropped_gap += 1
            continue
        if ACTION_SET.index(pair.action_left) != left_winner or ACTION_SET.index(pair.action_right) != right_winner:
            dropped_winner += 1
            continue
        if left_winner == right_winner:
            dropped_winner += 1
            continue
        by_question.setdefault(str(left.question_id), []).append(pair)
    selected: List[Any] = []
    for question_id, pairs in sorted(by_question.items()):
        total = sum(max(0.0, float(getattr(pair, "weight", 1.0))) for pair in pairs) or float(len(pairs))
        for pair in pairs:
            from dataclasses import replace
            selected.append(replace(pair, weight=max(0.0, float(getattr(pair, "weight", 1.0))) / total))
    audit = dict(audit)
    audit.update({
        "canonical_top1": True,
        "top1_gap_threshold": float(config.top1_gap_threshold),
        "selected_reversal_count": len(selected),
        "question_count_with_reversals": len(by_question),
        "dropped_gap": dropped_gap,
        "dropped_nonunique_or_same_winner": dropped_winner,
        "reversal_weighting": "question_macro_equal_weight_after_canonical_filter",
    })
    return selected, audit


def _pairwise_loss(
    policy: DifferentiableStrategyPolicy,
    reference: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    pairs: Sequence[Any],
) -> Tensor:
    values: List[Tensor] = []
    for pair in pairs:
        left = dataset.examples[int(pair.left)]
        right = dataset.examples[int(pair.right)]
        features = torch.stack([observation_to_features(left.observation), observation_to_features(right.observation)])
        logits = F.log_softmax(policy(features)["action_logits"], dim=-1)
        ref_logits = F.log_softmax(reference(features)["action_logits"], dim=-1)
        values.append(float(getattr(pair, "weight", 1.0)) * preference_reversal_loss(
            logits[0, ACTION_SET.index(pair.action_left)], logits[0, ACTION_SET.index(pair.action_right)],
            logits[1, ACTION_SET.index(pair.action_left)], logits[1, ACTION_SET.index(pair.action_right)],
            beta=1.0,
            reference_a_left=ref_logits[0, ACTION_SET.index(pair.action_left)],
            reference_a_right=ref_logits[0, ACTION_SET.index(pair.action_right)],
            reference_b_left=ref_logits[1, ACTION_SET.index(pair.action_left)],
            reference_b_right=ref_logits[1, ACTION_SET.index(pair.action_right)],
            confirmed=True,
            weight=1.0,
        ))
    return torch.stack(values).sum() if values else next(policy.parameters()).sum() * 0.0


def _listwise_loss(
    policy: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    pairs: Sequence[Any],
) -> Tensor:
    """Top-1 reversal loss: the endpoint winner beats every other action."""

    values: List[Tensor] = []
    for pair in pairs:
        for endpoint, action in ((int(pair.left), pair.action_left), (int(pair.right), pair.action_right)):
            logits = policy(observation_to_features(dataset.examples[endpoint].observation))["action_logits"]
            values.append(F.cross_entropy(logits, torch.tensor([ACTION_SET.index(action)], dtype=torch.long)))
    return torch.stack(values).mean() if values else next(policy.parameters()).sum() * 0.0


def _top1_margin_loss(
    policy: DifferentiableStrategyPolicy,
    dataset: DecisionDataset,
    pairs: Sequence[Any],
    margin: float,
) -> Tensor:
    """Push each canonical endpoint winner above all competing actions."""

    values: List[Tensor] = []
    for pair in pairs:
        for endpoint, action in ((int(pair.left), pair.action_left), (int(pair.right), pair.action_right)):
            logits = policy(observation_to_features(dataset.examples[endpoint].observation))["action_logits"].squeeze(0)
            winner_index = ACTION_SET.index(action)
            winner = logits[winner_index]
            competitors = torch.cat([logits[:winner_index], logits[winner_index + 1:]])
            values.append(float(getattr(pair, "weight", 1.0)) * F.relu(float(margin) - winner + competitors).mean())
    return torch.stack(values).mean() if values else next(policy.parameters()).sum() * 0.0


def _flat_grad(loss: Tensor, parameters: Sequence[Tensor]) -> Tensor:
    grads = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(parameters, grads)])


def _project_conflict(main: Tensor, rank: Tensor) -> Tensor:
    dot = torch.dot(main, rank)
    denom = torch.dot(main, main).clamp_min(1e-12)
    return rank - torch.minimum(dot, torch.zeros_like(dot)) / denom * main


def _cosine(a: Tensor, b: Tensor) -> float | None:
    if a.numel() == 0 or b.numel() == 0:
        return None
    denom = a.norm() * b.norm()
    if float(denom.detach()) <= 1e-12:
        return None
    return float((torch.dot(a, b) / denom).detach())


def _fit_from_sft(
    dataset: DecisionDataset,
    sft_policy: DifferentiableStrategyPolicy,
    method: str,
    config: P22Config,
    pairs: Sequence[Any],
) -> Tuple[DifferentiableStrategyPolicy, Dict[str, Any]]:
    policy = copy.deepcopy(sft_policy)
    reference = sft_policy.clone_frozen()
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    train_indices = {
        index for index, example in enumerate(dataset.examples) if example.split == "train"
    }
    pairs = [pair for pair in pairs if int(pair.left) in train_indices and int(pair.right) in train_indices]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.learning_rate))
    generator = torch.Generator().manual_seed(int(config.seed) + 1009)
    parameters = tuple(policy.parameters())
    lagrange = 0.0
    rows: List[Dict[str, float]] = []
    steps = 0
    use_pair = method in {"SFT→Pairwise-Full", "SFT→Pairwise+Listwise"}
    use_list = method in {"SFT→Listwise-Full", "SFT→Listwise+PCGrad"}
    use_rank = use_pair or use_list
    use_pcgrad = method == "SFT→Listwise+PCGrad"
    use_branch = method in {"SFT→BranchOnly", "SFT→NoFlip", "SFT→Pairwise-Full", "SFT→Listwise-Full", "SFT→Listwise+PCGrad"}
    use_full_aux = method in {"SFT→NoFlip", "SFT→Pairwise-Full", "SFT→Listwise-Full", "SFT→Listwise+PCGrad"}
    while steps < max(1, int(config.finetune_steps)):
        for batch_indices in _batch_indices(len(train), config.batch_size, generator):
            batch = [train[int(i)] for i in batch_indices.tolist()]
            features, utilities, state_targets, best_targets = _stack_examples(batch, torch.arange(len(batch), dtype=torch.long))
            outputs = policy(features)
            reference_outputs = reference(features)
            probabilities = F.softmax(outputs["action_logits"], dim=-1)
            q_targets = _utility_soft_targets(utilities, config.utility_temperature)
            utility_ce = -(q_targets * F.log_softmax(outputs["action_logits"], dim=-1)).sum(dim=-1).mean()
            centered = utilities - utilities.mean(dim=-1, keepdim=True)
            branch_loss = -(probabilities * centered.detach()).sum(dim=-1).mean()
            sft_ce = F.cross_entropy(outputs["action_logits"], best_targets)
            state_loss = F.cross_entropy(outputs["state_logits"], state_targets)
            belief_loss = F.binary_cross_entropy_with_logits(outputs["belief_logits"], _belief_target_matrix(batch))
            constraint_loss = _constraint_loss(outputs, state_targets, batch)
            # Penalize selecting REPAIR when the trusted pre-action state is not
            # INVALID.  This is an explicit safety regularizer for the
            # evaluator-owned erroneous-repair gate; it uses only the public
            # state target and cannot inspect hidden target-action labels.
            non_invalid = (state_targets != STATE_SET.index(EvidenceState.INVALID)).to(outputs["action_logits"].dtype)
            repair_probability = probabilities[:, ACTION_SET.index(ResearchAction.REPAIR)]
            erroneous_repair_penalty = (repair_probability * non_invalid).mean()
            entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()
            kl = _mean_kl(outputs["action_logits"], reference_outputs["action_logits"])
            expected = (probabilities * utilities).sum(dim=-1).mean()
            with torch.no_grad():
                # The floor is tied to the action actually selected by SFT, not
                # merely its softer expected utility.  This better matches the
                # downstream argmax regret gate while retaining a differentiable
                # current-policy side.
                reference_argmax = reference_outputs["action_logits"].argmax(dim=-1)
                reference_expected = utilities.gather(1, reference_argmax.unsqueeze(1)).squeeze(1).mean()
            violation = F.relu(reference_expected - expected - float(config.utility_floor_epsilon))
            lagrange = max(0.0, lagrange + float(config.lagrange_learning_rate) * float(violation.detach()))
            adaptive_kl = float(config.base_kl_weight) * (1.0 + float(config.adaptive_kl_gain) * float(F.relu(kl.detach() - config.target_kl)))
            entropy_penalty = F.relu(float(config.entropy_floor) - entropy).square()
            main = (
                (sft_ce if method == "SFT-Continued" else 0.0)
                + (float(config.branch_loss_weight) * branch_loss if use_branch else 0.0)
                + (float(config.utility_target_weight) * utility_ce if use_branch else 0.0)
                + (float(config.utility_hard_weight) * sft_ce if use_branch else 0.0)
                + (float(config.state_weight) * state_loss if use_full_aux else 0.0)
                + (float(config.belief_weight) * belief_loss if use_full_aux else 0.0)
                + (float(config.constraint_weight) * constraint_loss if use_full_aux else 0.0)
                + float(config.repair_safety_weight) * erroneous_repair_penalty
                + adaptive_kl * kl
                + float(config.entropy_floor_weight) * entropy_penalty
                + lagrange * violation
            )
            rank = torch.tensor(0.0)
            if use_pair:
                rank = _pairwise_loss(policy, reference, dataset, pairs)
                main = main + float(config.pairwise_weight) * rank
            if use_list or method == "SFT→Pairwise+Listwise":
                list_rank = _listwise_loss(policy, dataset, pairs)
                if method == "SFT→Pairwise+Listwise":
                    rank = rank + float(config.listwise_weight) * list_rank
                    main = main + float(config.listwise_weight) * list_rank
                elif use_list:
                    rank = list_rank
                    if not use_pcgrad:
                        main = main + float(config.listwise_weight) * rank
            top1_rank = (
                _top1_margin_loss(policy, dataset, pairs, config.top1_margin)
                if use_rank and float(config.top1_margin_weight) > 0.0
                else torch.tensor(0.0)
            )
            if float(config.top1_margin_weight) > 0.0:
                main = main + float(config.top1_margin_weight) * top1_rank
            branch_flip_cosine = state_flip_cosine = None
            if use_rank:
                # Batch-level objective-conflict diagnostics required by P2.2.
                # These are measured before the optimizer step and are retained in
                # the diagnostic log, never used as a tuning signal.
                branch_grad = _flat_grad(branch_loss, parameters)
                state_grad = _flat_grad(state_loss, parameters)
                rank_grad = _flat_grad(rank, parameters)
                branch_flip_cosine = _cosine(branch_grad, rank_grad)
                state_flip_cosine = _cosine(state_grad, rank_grad)
            optimizer.zero_grad(set_to_none=True)
            if use_pcgrad and use_rank:
                main_grad = _flat_grad(main, parameters)
                rank_grad = _flat_grad(rank, parameters)
                combined = main_grad + float(config.listwise_weight) * _project_conflict(main_grad, rank_grad)
                cursor = 0
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.grad = combined[cursor:cursor + size].reshape_as(parameter).clone()
                    cursor += size
                torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip_norm))
                optimizer.step()
            else:
                main.backward()
                torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip_norm))
                optimizer.step()
            rows.append({
                "step": float(steps), "loss": float(main.detach()), "branch_loss": float(branch_loss.detach()),
                "utility_soft_ce": float(utility_ce.detach()), "rank_loss": float(rank.detach()),
                "top1_margin_loss": float(top1_rank.detach()),
                "kl": float(kl.detach()), "adaptive_kl_weight": adaptive_kl,
                "entropy": float(entropy.detach()), "entropy_floor_penalty": float(entropy_penalty.detach()),
                "utility_floor_violation": float(violation.detach()), "lagrange": lagrange,
                "erroneous_repair_penalty": float(erroneous_repair_penalty.detach()),
                "cos_grad_branch_flip": branch_flip_cosine,
                "cos_grad_state_flip": state_flip_cosine,
                "reference_argmax_utility": float(reference_expected.detach()),
            })
            steps += 1
            if steps >= max(1, int(config.finetune_steps)):
                break
        if steps >= max(1, int(config.finetune_steps)):
            break
    return policy, {
        "method": method,
        "optimizer_steps": steps,
        "initialized_from_sft": True,
        "reference_policy": "SFT",
        "utility_floor_reference": "SFT",
        "canonical_reversal_count": len(pairs) if use_rank else 0,
        "pcgrad": bool(use_pcgrad),
        "rows": rows,
    }


def _fit_sft(dataset: DecisionDataset, config: P22Config) -> Tuple[DifferentiableStrategyPolicy, Dict[str, Any]]:
    trainer = DifferentiableStrategyTrainer(DifferentiableTrainerConfig(
        seed=int(config.seed), hidden_dim=int(config.hidden_dim), batch_size=int(config.batch_size),
        learning_rate=float(config.sft_learning_rate), max_optimizer_steps=max(1, int(config.sft_steps)),
        epochs=max(1, int(config.sft_steps)),
        state_class_weights=tuple(config.state_class_weights) if config.state_class_weights is not None else None,
        state_loss_weight=1.0,
    ))
    policy, log = trainer.fit(dataset, "SFT")
    return policy, {"optimizer_steps": int(log.optimizer_steps), "objective": "public_branch_utility_winner_cross_entropy"}


def _two_layer_ci(matrix: Mapping[int, Mapping[str, float]], *, seed: int, replicates: int) -> Dict[str, Any]:
    seeds = sorted(int(s) for s in matrix)
    questions = sorted({str(q) for row in matrix.values() for q in row})
    values = [float(matrix[s][q]) for s in seeds for q in questions if q in matrix[s]]
    point = float(sum(values) / len(values)) if values else None
    if point is None or len(seeds) <= 1 or len(questions) <= 1:
        return {"point": point, "lower": point, "upper": point, "method": "degenerate_seed_question_bootstrap", "seed_count": len(seeds), "question_count": len(questions)}
    rng = torch.Generator().manual_seed(int(seed))
    draws: List[float] = []
    for _ in range(max(1, int(replicates))):
        sampled_seeds = [seeds[int(torch.randint(len(seeds), (1,), generator=rng))] for _ in seeds]
        sampled_values: List[float] = []
        for sampled_seed in sampled_seeds:
            sampled_questions = [questions[int(torch.randint(len(questions), (1,), generator=rng))] for _ in questions]
            sampled_values.extend(float(matrix[sampled_seed][q]) for q in sampled_questions if q in matrix[sampled_seed])
        if sampled_values:
            draws.append(sum(sampled_values) / len(sampled_values))
    return {"point": point, "lower": float(torch.tensor(draws).quantile(0.025)), "upper": float(torch.tensor(draws).quantile(0.975)), "method": "two_layer_seed_then_question_bootstrap_percentile_95", "seed_count": len(seeds), "question_count": len(questions), "replicates": len(draws)}


def _aggregate(records: Sequence[Mapping[str, Any]], methods: Sequence[str], seeds: Sequence[int], *, bootstrap_replicates: int = 2000) -> Dict[str, Any]:
    aggregate_methods = list(dict.fromkeys([*methods, "SFT"]))
    by_method: Dict[str, List[Mapping[str, Any]]] = {method: [] for method in aggregate_methods}
    for row in records:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary: Dict[str, Any] = {}
    for method in aggregate_methods:
        rows = [row for row in by_method.get(method, []) if row.get("split") == "promotion"]
        summary[method] = {
            "seed_count": len({int(row["seed"]) for row in rows}),
            "mean_regret": float(sum(float(row["normalized_regret"]) for row in rows) / len(rows)) if rows else None,
            "pairrank_acc": float(sum(float(row.get("pairwise_reversal_ranking_accuracy") or 0.0) for row in rows) / len(rows)) if rows else None,
            "required_switch_rate": float(sum(float(row.get("required_switch_rate") or 0.0) for row in rows) / len(rows)) if rows else None,
            "confirmation_rate": float(sum(float(row.get("confirmation_rate") or 0.0) for row in rows) / len(rows)) if rows else None,
            "erroneous_repair_rate": float(sum(float(row.get("erroneous_repair_rate") or 0.0) for row in rows) / len(rows)) if rows else None,
            "validity_rate": 1.0 - float(sum(float(row.get("selected_invalid_branch_rate") or 0.0) for row in rows) / len(rows)) if rows else None,
        }
    # Question-level normalized-regret matrices for the preregistered gate.
    matrices: Dict[str, Dict[int, Dict[str, float]]] = {method: {} for method in aggregate_methods}
    for method in aggregate_methods:
        for row in by_method.get(method, []):
            if row.get("split") != "promotion":
                continue
            matrices[method][int(row["seed"])] = {
                str(q["question_id"]): float(q.get("normalized_regret", q.get("regret", 0.0)))
                for q in row.get("question_metric_rows", [])
            }
    baseline = "SFT"
    noflip = "SFT→NoFlip"
    gates: Dict[str, Any] = {}
    for method in methods:
        if method == baseline:
            continue
        delta_matrix: Dict[int, Dict[str, float]] = {}
        for seed in sorted(set(matrices.get(method, {})).intersection(matrices.get(baseline, {}))):
            common = set(matrices[method][seed]).intersection(matrices[baseline][seed])
            delta_matrix[seed] = {q: matrices[method][seed][q] - matrices[baseline][seed][q] for q in common}
        rank_delta_matrix: Dict[int, Dict[str, float]] = {}
        top1_delta_matrix: Dict[int, Dict[str, float]] = {}
        method_rows = {int(row["seed"]): row for row in by_method.get(method, []) if row.get("split") == "promotion"}
        noflip_rows = {int(row["seed"]): row for row in by_method.get(noflip, []) if row.get("split") == "promotion"}
        for seed in sorted(set(method_rows).intersection(noflip_rows)):
            # Use the per-question reversal rows directly.  Replicating a
            # question-macro scalar across all questions would erase the
            # question-level component of the preregistered seed×question
            # bootstrap and make the CI spuriously narrow.
            def question_metric(row: Mapping[str, Any], key: str) -> Dict[str, float]:
                out: Dict[str, float] = {}
                for question in row.get("question_metric_rows", []):
                    value = question.get(key)
                    if value is not None:
                        out[str(question.get("question_id"))] = float(value)
                return out
            method_rank = question_metric(method_rows[seed], "pairwise_reversal_ranking_accuracy")
            noflip_rank = question_metric(noflip_rows[seed], "pairwise_reversal_ranking_accuracy")
            common_rank = set(method_rank) & set(noflip_rank)
            rank_delta_matrix[seed] = {q: method_rank[q] - noflip_rank[q] for q in common_rank}
            method_top1 = question_metric(method_rows[seed], "exact_top1_reversal_accuracy")
            noflip_top1 = question_metric(noflip_rows[seed], "exact_top1_reversal_accuracy")
            common_top1 = set(method_top1) & set(noflip_top1)
            top1_delta_matrix[seed] = {q: method_top1[q] - noflip_top1[q] for q in common_top1}
        regret_ci = _two_layer_ci(delta_matrix, seed=71, replicates=bootstrap_replicates)
        pairrank_ci = _two_layer_ci(rank_delta_matrix, seed=73, replicates=bootstrap_replicates)
        top1_ci = _two_layer_ci(top1_delta_matrix, seed=79, replicates=bootstrap_replicates)
        positive_seeds = sum(
            (sum(values.values()) / len(values) if values else 0.0) < 0.0
            for values in delta_matrix.values()
        )
        gates[method] = {
            "regret_delta_vs_sft_ci": regret_ci,
            "regret_positive_seed_n_vs_sft": int(positive_seeds),
            "pairrank_delta_vs_noflip_ci": pairrank_ci,
            "canonical_top1_delta_vs_noflip_ci": top1_ci,
            "regret_gate": bool(regret_ci.get("upper") is not None and regret_ci["upper"] < 0.0 and positive_seeds >= 8),
            "pairrank_gate": bool(pairrank_ci.get("lower") is not None and pairrank_ci["lower"] > 0.0),
            "canonical_top1_gate": bool(top1_ci.get("lower") is not None and top1_ci["lower"] > 0.0),
        }
    return {"promotion_summary": summary, "gate_checks": gates, "baseline_method": baseline, "pairrank_reference": noflip}


def _attach_normalized_regret(row: Dict[str, Any], dataset: DecisionDataset) -> Dict[str, Any]:
    """Add utility-range-normalized regret while retaining evaluator metrics verbatim."""

    by_world = {str(example.world_id): example for example in dataset.examples if example.split == row.get("split")}
    values: List[float] = []
    by_question: Dict[str, List[float]] = {}
    for record in row.get("records", []):
        example = by_world.get(str(record.get("world_id")))
        if example is None:
            continue
        try:
            selected = ACTION_SET.index(ResearchAction(str(record.get("selected_action"))))
        except (TypeError, ValueError):
            continue
        utilities = [float(value) for value in example.branch_utilities]
        value = (max(utilities) - utilities[selected]) / max(max(utilities) - min(utilities), 1e-8)
        values.append(float(value))
        by_question.setdefault(str(example.question_id), []).append(float(value))
    row["normalized_regret"] = float(sum(values) / len(values)) if values else None
    for question_row in row.get("question_metric_rows", []):
        question_id = str(question_row.get("question_id"))
        qvalues = by_question.get(question_id, [])
        question_row["normalized_regret"] = float(sum(qvalues) / len(qvalues)) if qvalues else None
    return row


def run_p22_diagnostic(
    output_dir: str | Path,
    dataset: DecisionDataset,
    *,
    seeds: Sequence[int] = (17, 23, 29),
    config: Optional[P22Config] = None,
    methods: Sequence[str] = P22_METHODS,
    stage: str = "screening",
    eval_splits: Sequence[str] | None = None,
) -> Dict[str, Any]:
    config = config or P22Config()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs, reversal_audit = _canonical_reversals(dataset, config)
    records: List[Dict[str, Any]] = []
    logs: Dict[str, Any] = {}
    selected_splits = tuple(eval_splits) if eval_splits is not None else tuple(sorted({str(example.split) for example in dataset.examples}))
    for seed in [int(s) for s in seeds]:
        seed_config = P22Config(**{**asdict(config), "seed": seed})
        random.seed(seed)
        torch.manual_seed(seed)
        sft_policy, sft_log = _fit_sft(dataset, seed_config)
        logs[str(seed)] = {"SFT": sft_log}
        for method in methods:
            policy, log = _fit_from_sft(dataset, sft_policy, str(method), seed_config, pairs)
            logs[str(seed)][str(method)] = log
            for split in selected_splits:
                row = evaluate_differentiable_policy(policy, dataset, split, retain_records=False)
                row = _attach_normalized_regret(row, dataset)
                row["selected_action_rows"] = [
                    {key: record.get(key) for key in ("question_id", "world_id", "selected_action", "regret", "selected_utility")}
                    for record in row.get("records", [])
                ]
                row["record_count"] = len(row.get("records", ()))
                row.pop("records", None)
                row["method"] = str(method)
                row["seed"] = seed
                row["stage"] = str(stage)
                row["initialized_from_sft"] = True
                row["reference_policy"] = "SFT"
                row["utility_floor_reference"] = "SFT"
                records.append(row)
            del policy
            gc.collect()
        # SFT itself is recorded as the frozen reference; continuation is the
        # separately fine-tuned method above.
        for split in selected_splits:
            row = evaluate_differentiable_policy(sft_policy, dataset, split, retain_records=False)
            row = _attach_normalized_regret(row, dataset)
            row["selected_action_rows"] = [
                {key: record.get(key) for key in ("question_id", "world_id", "selected_action", "regret", "selected_utility")}
                for record in row.get("records", [])
            ]
            row["record_count"] = len(row.get("records", ()))
            row.pop("records", None)
            row["method"] = "SFT"
            row["seed"] = seed
            row["stage"] = str(stage)
            row["sft_checkpoint"] = True
            records.append(row)
        del sft_policy
        gc.collect()
    result = {
        "schema_version": "pesco_tier1_p22_common_sft_diagnostic_v0.1",
        "stage": str(stage),
        "config": asdict(config),
        "seeds": [int(s) for s in seeds],
        "methods": list(methods),
        "eval_splits": list(selected_splits),
        "canonical_reversal_audit": reversal_audit,
        "records": records,
        "training_logs": logs,
        "aggregation": _aggregate(records, methods, [int(s) for s in seeds]),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "final_evaluation_authorized": False,
    }
    (output / "p22_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return result


__all__ = [
    "P22Config",
    "P22_METHODS",
    "run_p22_diagnostic",
    "_utility_soft_targets",
    "_canonical_reversals",
]

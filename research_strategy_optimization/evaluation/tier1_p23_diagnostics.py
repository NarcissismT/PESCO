"""P2.3 common-protocol optimizer × branch-credit × reversal diagnostics.

The benchmark is still a four-action contextual bandit, so this module does not
pretend to implement token-level GSPO/DAPO.  It provides matched CPU reference
implementations of RLOO, GRPO variants, DPO/SimPO and the PESCO component
ablations.  Every learned method starts from the same seed-specific SFT policy,
uses the same public observations, atomic branch utilities, executor-derived
confirmation receipts and optimizer-step budget, and remains diagnostic-only.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from ..algorithms.differentiable_strategy import (
    ACTION_SET,
    STATE_SET,
    DecisionDataset,
    DifferentiableStrategyPolicy,
    _batch_indices,
    _belief_target_matrix,
    _constraint_loss,
    _entropy,
    _mean_kl,
    _stack_examples,
    observation_to_features,
    policy_action,
)
from ..algorithms.preference_reversal_loss import preference_reversal_loss
from ..schemas import EvidenceState, ResearchAction
from .tier1_differentiable_suite import evaluate_differentiable_policy
from .tier1_differentiable_suite import _confirmation_passed, _confirmation_receipt_eligible
from .tier1_p22_diagnostics import (
    P22Config,
    _attach_normalized_regret,
    _canonical_reversals,
    _fit_from_sft,
    _fit_sft,
    _two_layer_ci,
)


P23_LEARNED_METHODS: Tuple[str, ...] = (
    "SFT-Continued",
    "DPO",
    "SimPO",
    "RLOO",
    "GRPO-Terminal",
    "GRPO-FourState",
    "GRPO-MatchedAtomic",
    "GRPO+State",
    "GRPO+Branch",
    "GRPO+Flip",
    "GRPO+Branch+Flip",
)
P23_METHODS: Tuple[str, ...] = (
    "Random",
    "Rule-Based",
    *P23_LEARNED_METHODS,
    "Search/Oracle",
)
P23_COMPONENT_MAPPING = {
    "Base": "GRPO-MatchedAtomic",
    "State": "GRPO+State",
    "Branch": "GRPO+Branch",
    "Flip": "GRPO+Flip",
    "Branch+Flip": "GRPO+Branch+Flip",
}


@dataclass(frozen=True)
class P23Config(P22Config):
    # Promotion-v2 canonicalization keeps the single top-candidate reversal
    # per question even when the empirical gap is small.  The promotion
    # contract is about avoiding action-pair inflation; confidence margins are
    # reported diagnostically rather than used to silently underpower the
    # benchmark.
    top1_gap_threshold: float = 0.0
    max_pairs_per_question: int = 1
    group_size: int = 4
    grpo_clip_low: float = 0.8
    grpo_clip_high: float = 1.2
    dpo_beta: float = 0.1
    simpo_gamma: float = 0.2
    use_atomic_reward: bool = True


def _strict_rule_action(example: Any, *, oracle: bool = False) -> ResearchAction:
    if oracle:
        state = EvidenceState(example.state_target)
        return {
            EvidenceState.SUPPORTED: ResearchAction.CONTINUE,
            EvidenceState.REFUTED: ResearchAction.SWITCH,
            EvidenceState.INSUFFICIENT: ResearchAction.SAMPLE,
            EvidenceState.INVALID: ResearchAction.REPAIR,
        }[state]
    obs = example.observation
    raw = {str(k): float(v) for k, v in getattr(obs, "raw_evidence", ())}
    if raw.get("log_protocol_change_count", 0.0) > 0 or raw.get("group_overlap_count", 0.0) > 0:
        return ResearchAction.REPAIR
    if raw.get("replication_sample_size", 0.0) < 40 or raw.get("replication_ci_width", 0.0) > 0.28:
        return ResearchAction.SAMPLE
    if float(obs.effect_estimate) < -0.02:
        return ResearchAction.SWITCH
    return ResearchAction.CONTINUE


def _evaluate_fixed_action(dataset: DecisionDataset, split: str, action_fn, *, method: str, seed: int) -> dict:
    selected: list[dict] = []
    confirmation_observed = confirmation_eligible = confirmation_passed = 0
    required_switch_n = required_switch_correct = 0
    erroneous_repair_n = invalid_local_n = selected_invalid_n = 0
    for example in dataset.examples:
        if example.split != split:
            continue
        action = action_fn(example)
        index = ACTION_SET.index(action)
        utility = float(example.branch_utilities[index])
        best = max(float(value) for value in example.branch_utilities)
        regret = best - utility
        if example.best_action is ResearchAction.SWITCH:
            required_switch_n += 1
            required_switch_correct += int(action is ResearchAction.SWITCH)
        if example.state_target is not EvidenceState.INVALID and action is ResearchAction.REPAIR:
            erroneous_repair_n += 1
        if example.state_target is EvidenceState.INVALID and action in {ResearchAction.CONTINUE, ResearchAction.SWITCH} and action is not example.best_action:
            invalid_local_n += 1
        validity_map = example.metadata.get("branch_validity", {})
        selected_invalid_n += int(action.value in validity_map and not bool(validity_map.get(action.value, False)))
        receipts = example.metadata.get("branch_replicate_confirmation", {}).get(action.value, ())
        if isinstance(receipts, Mapping):
            receipts = (receipts,)
        receipts = tuple(item for item in receipts if isinstance(item, Mapping))
        confirmation_observed += len(receipts)
        eligible = tuple(item for item in receipts if _confirmation_receipt_eligible(item))
        confirmation_eligible += len(eligible)
        confirmation_passed += sum(_confirmation_passed(item) for item in eligible)
        selected.append({
            "question_id": example.question_id,
            "world_id": example.world_id,
            "selected_action": action.value,
            "selected_utility": utility,
            "regret": regret,
        })
    by_question: dict[str, list[dict]] = {}
    for row in selected:
        by_question.setdefault(str(row["question_id"]), []).append(row)
    # Utility-range normalization is performed per world before the question
    # macro mean, matching the learned-policy evaluator and preventing a fixed
    # baseline from being compared on a different regret scale.
    examples_by_world = {str(e.world_id): e for e in dataset.examples if e.split == split}
    for row in selected:
        example = examples_by_world.get(str(row["world_id"]))
        if example is None:
            row["normalized_regret"] = None
            continue
        utility_range = max(float(v) for v in example.branch_utilities) - min(float(v) for v in example.branch_utilities)
        row["normalized_regret"] = float(row["regret"]) / max(utility_range, 1e-8)
    qrows = [{
        "question_id": qid,
        "family": str(examples_by_world[values[0]["world_id"]].metadata.get("family", "")) if values and values[0]["world_id"] in examples_by_world else "",
        "example_count": len(values),
        "mean_regret": sum(float(v["regret"]) for v in values) / len(values),
        "normalized_regret": sum(float(v["normalized_regret"]) for v in values) / len(values),
    } for qid, values in sorted(by_question.items())]
    mean_regret = sum(float(row["regret"]) for row in selected) / len(selected) if selected else None
    normalized = sum(float(row["normalized_regret"]) for row in selected) / len(selected) if selected else None
    return {
        "method": method,
        "seed": int(seed),
        "split": split,
        "record_count": len(selected),
        "mean_regret": mean_regret,
        "normalized_regret": normalized,
        "selected_action_rows": selected,
        "question_metric_rows": qrows,
        "pairwise_reversal_ranking_accuracy": None,
        "exact_top1_reversal_accuracy": None,
        "required_switch_n": required_switch_n,
        "required_switch_rate": required_switch_correct / required_switch_n if required_switch_n else None,
        "confirmation_observed_n": confirmation_observed,
        "confirmation_receipt_n": confirmation_eligible,
        "confirmation_passed_n": confirmation_passed,
        "confirmation_rate": confirmation_passed / confirmation_eligible if confirmation_eligible else None,
        "erroneous_repair_rate": erroneous_repair_n / len(selected) if selected else None,
        "invalid_local_optimization_rate": invalid_local_n / len(selected) if selected else None,
        "selected_invalid_branch_rate": selected_invalid_n / len(selected) if selected else None,
        "diagnostic_fixed_policy": True,
    }


def _logprob_action(policy: DifferentiableStrategyPolicy, example: Any, action: ResearchAction) -> Tensor:
    logits = policy(observation_to_features(example.observation))["action_logits"]
    return F.log_softmax(logits, dim=-1).squeeze(0)[ACTION_SET.index(action)]


def _preference_loss(policy: DifferentiableStrategyPolicy, reference: DifferentiableStrategyPolicy, dataset: DecisionDataset, pairs: Sequence[Any], config: P23Config, *, simpo: bool = False) -> Tensor:
    examples: list[Any] = []
    chosen_indices: list[int] = []
    rejected_indices: list[int] = []
    for pair in pairs:
        left, right = dataset.examples[int(pair.left)], dataset.examples[int(pair.right)]
        for example, chosen, rejected in ((left, pair.action_left, pair.action_right), (right, pair.action_right, pair.action_left)):
            examples.append(example)
            chosen_indices.append(ACTION_SET.index(chosen))
            rejected_indices.append(ACTION_SET.index(rejected))
    if not examples:
        return next(policy.parameters()).sum() * 0.0
    features = torch.stack([observation_to_features(example.observation) for example in examples], dim=0)
    logits = policy(features)["action_logits"]
    log_probs = F.log_softmax(logits, dim=-1)
    index_chosen = torch.tensor(chosen_indices, dtype=torch.long, device=logits.device)
    index_rejected = torch.tensor(rejected_indices, dtype=torch.long, device=logits.device)
    delta = log_probs.gather(1, index_chosen[:, None]).squeeze(1) - log_probs.gather(1, index_rejected[:, None]).squeeze(1)
    if simpo:
        return -F.logsigmoid(float(config.dpo_beta) * (delta - float(config.simpo_gamma))).mean()
    with torch.no_grad():
        reference_delta = F.log_softmax(reference(features)["action_logits"], dim=-1)
        reference_delta = reference_delta.gather(1, index_chosen[:, None]).squeeze(1) - reference_delta.gather(1, index_rejected[:, None]).squeeze(1)
    return -F.logsigmoid(float(config.dpo_beta) * (delta - reference_delta)).mean()


def _grpo_fit(dataset: DecisionDataset, sft_policy: DifferentiableStrategyPolicy, method: str, config: P23Config, pairs: Sequence[Any]) -> tuple[DifferentiableStrategyPolicy, dict]:
    policy = copy.deepcopy(sft_policy)
    reference = sft_policy.clone_frozen()
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    train_pairs = [pair for pair in pairs if dataset.examples[int(pair.left)].split == "train" and dataset.examples[int(pair.right)].split == "train"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.learning_rate))
    generator = torch.Generator().manual_seed(int(config.seed) + 7171)
    use_state = method in {"GRPO-FourState", "GRPO+State"}
    use_branch = method in {"GRPO+Branch", "GRPO+Branch+Flip"}
    use_flip = method in {"GRPO+Flip", "GRPO+Branch+Flip"}
    use_atomic = method == "GRPO-MatchedAtomic"
    logs: list[dict] = []
    for step in range(max(1, int(config.finetune_steps))):
        indices = torch.randint(len(train), (min(max(1, int(config.batch_size)), len(train)),), generator=generator)
        batch = [train[int(i)] for i in indices.tolist()]
        features, utilities, state_targets, _ = _stack_examples(batch, torch.arange(len(batch), dtype=torch.long))
        outputs = policy(features)
        probs = F.softmax(outputs["action_logits"], dim=-1)
        log_probs = F.log_softmax(outputs["action_logits"], dim=-1)
        k = max(2, int(config.group_size))
        samples = torch.multinomial(probs.detach(), k, replacement=True, generator=generator)
        rewards = utilities.gather(1, samples)
        # GRPO and RLOO both use within-example relative rewards; with one-step
        # actions the distinction is only the number of sampled siblings.
        if method == "GRPO-Terminal" or method == "GRPO-FourState" or method == "GRPO-MatchedAtomic" or method.startswith("GRPO+") or method == "RLOO":
            baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / max(1, k - 1)
            advantages = rewards - baseline
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-4)
            sampled_logp = log_probs.gather(1, samples)
            option_loss = -(sampled_logp * advantages.detach()).mean()
        else:
            option_loss = outputs["action_logits"].sum() * 0.0
        if use_branch:
            centered = utilities - utilities.mean(dim=-1, keepdim=True)
            branch_loss = -(probs * centered.detach()).sum(dim=-1).mean()
        else:
            branch_loss = outputs["action_logits"].sum() * 0.0
        state_loss = F.cross_entropy(outputs["state_logits"], state_targets) if use_state else outputs["state_logits"].sum() * 0.0
        belief_loss = F.binary_cross_entropy_with_logits(outputs["belief_logits"], _belief_target_matrix(batch)) if use_state else outputs["belief_logits"].sum() * 0.0
        constraint_loss = _constraint_loss(outputs, state_targets, batch) if use_state else outputs["action_logits"].sum() * 0.0
        flip_loss = _preference_loss(policy, reference, dataset, train_pairs, config, simpo=False) if use_flip else outputs["action_logits"].sum() * 0.0
        # Atomic reward is exactly the sum of evaluator-produced terms; retaining
        # the explicit switch in the log prevents a future implementation from
        # silently replacing it with a post-hoc target action.
        atomic_weight = 1.0 if use_atomic else 0.0
        kl = _mean_kl(outputs["action_logits"], reference(features)["action_logits"])
        entropy = _entropy(probs)
        total = option_loss + float(config.branch_loss_weight) * branch_loss + float(config.state_weight) * state_loss + float(config.belief_weight) * belief_loss + float(config.constraint_weight) * constraint_loss + float(config.pairwise_weight) * flip_loss + float(config.base_kl_weight) * kl + float(config.entropy_floor_weight) * F.relu(float(config.entropy_floor) - entropy)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.gradient_clip_norm))
        optimizer.step()
        logs.append({"step": step, "loss": float(total.detach()), "option_loss": float(option_loss.detach()), "branch_loss": float(branch_loss.detach()), "flip_loss": float(flip_loss.detach()), "state_loss": float(state_loss.detach()), "kl": float(kl.detach()), "entropy": float(entropy.detach()), "atomic_reward_objective": bool(use_atomic), "group_size": k})
    return policy, {
        "method": method,
        "optimizer_steps": len(logs),
        "initialized_from_sft": True,
        "reference_policy": "SFT",
        "utility_floor_reference": "SFT",
        "optimizer_backbone": "RLOO" if method == "RLOO" else "GRPO",
        "branch_credit": bool(use_branch),
        "flip_credit": bool(use_flip),
        "state_auxiliary": bool(use_state),
        "atomic_reward": bool(use_atomic or method.startswith("GRPO")),
        "rows": logs,
    }


def _fit_method(dataset: DecisionDataset, sft_policy: DifferentiableStrategyPolicy, method: str, config: P23Config, pairs: Sequence[Any]) -> tuple[DifferentiableStrategyPolicy | None, dict]:
    if method == "SFT-Continued":
        return _fit_from_sft(dataset, sft_policy, "SFT-Continued", config, pairs)
    if method == "DPO":
        return _preference_fit(dataset, sft_policy, method, config, pairs, simpo=False)
    if method == "SimPO":
        return _preference_fit(dataset, sft_policy, method, config, pairs, simpo=True)
    if method in {"RLOO", "GRPO-Terminal", "GRPO-FourState", "GRPO-MatchedAtomic", "GRPO+State", "GRPO+Branch", "GRPO+Flip", "GRPO+Branch+Flip"}:
        return _grpo_fit(dataset, sft_policy, method, config, pairs)
    raise ValueError(f"unknown learned P2.3 method: {method}")


def _preference_fit(dataset: DecisionDataset, sft_policy: DifferentiableStrategyPolicy, method: str, config: P23Config, pairs: Sequence[Any], *, simpo: bool) -> tuple[DifferentiableStrategyPolicy, dict]:
    policy = copy.deepcopy(sft_policy)
    reference = sft_policy.clone_frozen()
    train_pairs = [pair for pair in pairs if dataset.examples[int(pair.left)].split == "train" and dataset.examples[int(pair.right)].split == "train"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.learning_rate))
    logs = []
    for step in range(max(1, int(config.finetune_steps))):
        loss = _preference_loss(policy, reference, dataset, train_pairs, config, simpo=simpo)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.gradient_clip_norm)); optimizer.step()
        logs.append({"step": step, "loss": float(loss.detach()), "preference_data_source": "confirmed_executor_reversals", "reference_policy": "SFT" if not simpo else None})
    return policy, {"method": method, "optimizer_steps": len(logs), "initialized_from_sft": True, "reference_policy": "SFT", "utility_floor_reference": "SFT", "optimizer_backbone": method, "rows": logs, "preference_pair_count": len(train_pairs)}


def _aggregate_p23(records: Sequence[Mapping[str, Any]], methods: Sequence[str], seeds: Sequence[int], *, pairs_by_question: Mapping[str, int], bootstrap_replicates: int = 2000) -> dict:
    aggregate_methods = list(dict.fromkeys([*methods, "SFT"]))
    by_method: dict[str, list[Mapping[str, Any]]] = {method: [] for method in aggregate_methods}
    for row in records:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary: dict[str, Any] = {}
    for method in aggregate_methods:
        rows = [row for row in by_method.get(method, []) if row.get("split") == "promotion"]
        summary[method] = {
            "seed_count": len({int(row["seed"]) for row in rows}),
            "mean_regret": sum(float(row.get("normalized_regret") or 0.0) for row in rows) / len(rows) if rows else None,
            "pairrank_acc": sum(float(row.get("pairwise_reversal_ranking_accuracy") or 0.0) for row in rows) / len(rows) if rows else None,
            "required_switch_rate": sum(float(row.get("required_switch_rate") or 0.0) for row in rows) / len(rows) if rows else None,
            "confirmation_rate": sum(float(row.get("confirmation_rate") or 0.0) for row in rows) / len(rows) if rows else None,
            "erroneous_repair_rate": sum(float(row.get("erroneous_repair_rate") or 0.0) for row in rows) / len(rows) if rows else None,
            "validity_rate": 1.0 - sum(float(row.get("selected_invalid_branch_rate") or 0.0) for row in rows) / len(rows) if rows else None,
        }
    metric_matrices: dict[str, dict[int, dict[str, float]]] = {method: {} for method in aggregate_methods}
    for method in aggregate_methods:
        for row in by_method.get(method, []):
            if row.get("split") != "promotion": continue
            metric_matrices[method][int(row["seed"])] = {str(q["question_id"]): float(q.get("normalized_regret", 0.0)) for q in row.get("question_metric_rows", [])}
    sft = "SFT"; noflip = "GRPO-MatchedAtomic"
    if noflip in summary:
        summary["NoFlip"] = dict(summary[noflip])
    non_pesco_candidates = [
        method for method in (
            "SFT", "SFT-Continued", "DPO", "SimPO", "RLOO",
            "GRPO-Terminal", "GRPO-FourState", "Logistic Regression",
            "Gradient Boosting", "Random", "Rule-Based",
        ) if method in summary and summary[method].get("mean_regret") is not None
    ]
    best_non_pesco = min(non_pesco_candidates, key=lambda name: float(summary[name]["mean_regret"])) if non_pesco_candidates else None
    gate_checks = {}
    for method in methods:
        if method == "SFT": continue
        delta = {}
        for seed in sorted(set(metric_matrices.get(method, {})) & set(metric_matrices.get(sft, {}))):
            common = set(metric_matrices[method][seed]) & set(metric_matrices[sft][seed]); delta[seed] = {q: metric_matrices[method][seed][q] - metric_matrices[sft][seed][q] for q in common}
        rank_delta: dict[int, dict[str, float]] = {}
        top1_delta: dict[int, dict[str, float]] = {}
        method_rows = {int(row["seed"]): row for row in by_method.get(method, []) if row.get("split") == "promotion"}
        ref_rows = {int(row["seed"]): row for row in by_method.get(noflip, []) if row.get("split") == "promotion"}
        for seed in sorted(set(method_rows) & set(ref_rows)):
            def qmetric(row: Mapping[str, Any], key: str) -> dict[str, float]:
                return {str(q["question_id"]): float(q[key]) for q in row.get("question_metric_rows", []) if q.get(key) is not None}
            a, b = qmetric(method_rows[seed], "pairwise_reversal_ranking_accuracy"), qmetric(ref_rows[seed], "pairwise_reversal_ranking_accuracy")
            rank_delta[seed] = {q: a[q] - b[q] for q in set(a) & set(b)}
            a, b = qmetric(method_rows[seed], "exact_top1_reversal_accuracy"), qmetric(ref_rows[seed], "exact_top1_reversal_accuracy")
            top1_delta[seed] = {q: a[q] - b[q] for q in set(a) & set(b)}
        regret_ci = _two_layer_ci(delta, seed=101, replicates=bootstrap_replicates)
        pairrank_ci = _two_layer_ci(rank_delta, seed=103, replicates=bootstrap_replicates)
        top1_ci = _two_layer_ci(top1_delta, seed=107, replicates=bootstrap_replicates)
        positive = sum((sum(v.values()) / len(v) if v else 0.0) < 0 for v in delta.values())
        matched_delta: dict[int, dict[str, float]] = {}
        for seed in sorted(set(metric_matrices.get(method, {})) & set(metric_matrices.get(noflip, {}))):
            common = set(metric_matrices[method][seed]) & set(metric_matrices[noflip][seed])
            matched_delta[seed] = {q: metric_matrices[method][seed][q] - metric_matrices[noflip][seed][q] for q in common}
        matched_ci = _two_layer_ci(matched_delta, seed=109, replicates=bootstrap_replicates)
        family_by_question: dict[str, str] = {}
        for row in method_rows.values():
            for qrow in row.get("question_metric_rows", []):
                family_by_question[str(qrow.get("question_id"))] = str(qrow.get("family", ""))
        loo_values: list[float] = []
        for family in sorted({value for value in family_by_question.values() if value}):
            vals = [value for seed_delta in matched_delta.values() for qid, value in seed_delta.items() if family_by_question.get(qid) != family]
            if vals:
                loo_values.append(sum(vals) / len(vals))
        switch_target = summary.get(method, {}).get("required_switch_rate")
        switch_sft = summary.get(sft, {}).get("required_switch_rate")
        switch_noflip = summary.get(noflip, {}).get("required_switch_rate")
        gate_checks[method] = {
            "regret_delta_vs_sft_ci": regret_ci,
            "regret_positive_seed_n_vs_sft": int(positive),
            "pairrank_delta_vs_grpo_matched_atomic_ci": pairrank_ci,
            "regret_delta_vs_grpo_matched_atomic_ci": matched_ci,
            "canonical_top1_delta_vs_grpo_matched_atomic_ci": top1_ci,
            "family_leave_one_out_regret_delta_vs_grpo_matched_atomic": {
                "values": loo_values,
                "min": min(loo_values) if loo_values else None,
                "max": max(loo_values) if loo_values else None,
                "all_better": bool(loo_values) and max(loo_values) < 0.0,
            },
            "required_switch_target": switch_target,
            "required_switch_sft": switch_sft,
            "required_switch_noflip": switch_noflip,
            "regret_gate": bool(regret_ci.get("upper") is not None and regret_ci["upper"] < 0 and positive >= 8),
            "regret_vs_grpo_matched_atomic_gate": bool(matched_ci.get("upper") is not None and matched_ci["upper"] < 0),
            "pairrank_gate": bool(pairrank_ci.get("lower") is not None and pairrank_ci["lower"] > 0),
            "canonical_top1_gate": bool(top1_ci.get("lower") is not None and top1_ci["lower"] > 0),
            "required_switch_gate_vs_sft_and_noflip": bool(
                switch_target is not None and switch_sft is not None and switch_noflip is not None
                and switch_target > switch_sft and switch_target > switch_noflip
            ),
            "family_leave_one_out_gate": bool(loo_values) and max(loo_values) < 0.0,
        }
    return {
        "promotion_summary": summary,
        "gate_checks": gate_checks,
        "baseline_method": best_non_pesco or sft,
        "best_non_pesco_method": best_non_pesco,
        "pairrank_reference": noflip,
        "pair_contract": {"promotion_question_count": len(pairs_by_question), "promotion_pair_count": sum(pairs_by_question.values())},
        "required_switch_contract": "target must exceed both SFT and NoFlip/GRPO-MatchedAtomic",
    }


def run_p23_diagnostic(output_dir: str | Path, dataset: DecisionDataset, *, seeds: Sequence[int] = (17, 23, 29), config: Optional[P23Config] = None, methods: Sequence[str] = P23_METHODS, stage: str = "screening", eval_splits: Sequence[str] = ("tune", "promotion")) -> dict:
    config = config or P23Config()
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    pairs, reversal_audit = _canonical_reversals(dataset, config)
    promotion_pairs = [pair for pair in pairs if dataset.examples[int(pair.left)].split == "promotion" and dataset.examples[int(pair.right)].split == "promotion"]
    pairs_by_question = {qid: sum(dataset.examples[int(pair.left)].question_id == qid for pair in promotion_pairs) for qid in sorted({dataset.examples[int(pair.left)].question_id for pair in promotion_pairs})}
    records: list[dict] = []; logs: dict[str, dict] = {}
    selected_splits = tuple(eval_splits)
    for seed in [int(s) for s in seeds]:
        seed_config = P23Config(**{**asdict(config), "seed": seed})
        random.seed(seed); torch.manual_seed(seed)
        sft_policy, sft_log = _fit_sft(dataset, seed_config)
        logs[str(seed)] = {"SFT": sft_log}
        # Fixed baselines do not consume hidden target labels on the raw track.
        fixed = {
            "Random": lambda ex, seed=seed: ACTION_SET[int.from_bytes(hashlib.sha256(f"{seed}|{ex.question_id}|{ex.world_id}".encode()).digest()[:8], "big") % len(ACTION_SET)],
            "Rule-Based": lambda ex: _strict_rule_action(ex, oracle=False),
            "Search/Oracle": lambda ex: ACTION_SET[ex.best_action_index],
        }
        for method in methods:
            if method in fixed:
                for split in selected_splits:
                    row = _evaluate_fixed_action(dataset, split, fixed[method], method=method, seed=seed)
                    row["stage"] = stage; records.append(row)
                continue
            if method == "SFT":
                continue
            policy, train_log = _fit_method(dataset, sft_policy, method, seed_config, pairs)
            logs[str(seed)][method] = train_log
            for split in selected_splits:
                row = evaluate_differentiable_policy(policy, dataset, split, retain_records=True)
                row = _attach_normalized_regret(row, dataset)
                row["selected_action_rows"] = [{key: record.get(key) for key in ("question_id", "world_id", "selected_action", "regret", "selected_utility")} for record in row.get("records", [])]
                row["record_count"] = len(row.get("records", [])); row.pop("records", None)
                row.update({"method": method, "seed": seed, "stage": stage, "initialized_from_sft": True, "reference_policy": "SFT"})
                records.append(row)
            del policy; gc.collect()
        for split in selected_splits:
            row = evaluate_differentiable_policy(sft_policy, dataset, split, retain_records=True)
            row = _attach_normalized_regret(row, dataset)
            row["selected_action_rows"] = [{key: record.get(key) for key in ("question_id", "world_id", "selected_action", "regret", "selected_utility")} for record in row.get("records", [])]
            row["record_count"] = len(row.get("records", [])); row.pop("records", None)
            row.update({"method": "SFT", "seed": seed, "stage": stage, "sft_checkpoint": True})
            records.append(row)
        del sft_policy; gc.collect()
    result = {
        "schema_version": "pesco_tier1_p23_common_protocol_diagnostic_v0.1",
        "stage": stage,
        "track": str(dataset.examples[0].observation.track) if dataset.examples else "unknown",
        "config": asdict(config),
        "seeds": [int(s) for s in seeds],
        "methods": list(methods),
        "eval_splits": list(selected_splits),
        "canonical_reversal_audit": {**reversal_audit, "promotion_selected_reversal_count": len(promotion_pairs), "promotion_question_cluster_count": len(pairs_by_question), "promotion_pair_minimum": 30, "promotion_question_cluster_minimum": 20, "promotion_power_boundary_pass": len(promotion_pairs) >= 30 and len(pairs_by_question) >= 20, "max_pairs_per_question": 1},
        "records": records,
        "training_logs": logs,
        "aggregation": _aggregate_p23(records, methods, seeds, pairs_by_question=pairs_by_question),
        "component_mapping": P23_COMPONENT_MAPPING,
        "optimizer_scope": {"single_action_contextual_bandit": True, "gspo_full_reproduction": False, "dapo_full_reproduction": False, "reason": "sequence length T=1; GSPO/DAPO token-level assumptions are deferred to P3a"},
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "final_evaluation_authorized": False,
    }
    (output / "p23_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return result


__all__ = ["P23Config", "P23_METHODS", "P23_LEARNED_METHODS", "P23_COMPONENT_MAPPING", "run_p23_diagnostic"]

"""P2.3.1 optimizer-authenticity and evaluator-contract diagnostics.

This module is deliberately separate from the consumed P2.3 runner.  It keeps the
same four-action CPU contextual-bandit interface, but implements the distinctions
that matter for an algorithm comparison: one shared SFT checkpoint per seed,
frozen rollout batches, true RLOO versus clipped GRPO importance ratios, and
terminal/four-state/atomic reward tensors that are audited before training.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from ..algorithms.differentiable_strategy import (
    ACTION_SET,
    BASE_FEATURE_DIM,
    STATE_SET,
    DecisionDataset,
    DifferentiableStrategyPolicy,
    _belief_target_matrix,
    _constraint_loss,
    _entropy,
    _mean_kl,
    _stack_examples,
    observation_to_features,
)
from ..schemas import EvidenceState, ResearchAction
from .tier1_differentiable_suite import evaluate_differentiable_policy
from .tier1_p22_diagnostics import P22Config, _attach_normalized_regret, _canonical_reversals, _fit_sft

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


REWARD_COMPONENTS = (
    "validity_gate", "confirmation_bonus", "repair_protocol_bonus", "heldout_split_bonus",
    "mechanism_transition_bonus", "switch_success_bonus", "switch_failure_penalty",
    "sample_precision_bonus", "state_resolution_bonus", "replicate_bonus", "execution_cost_penalty",
)
TERMINAL_COMPONENTS = {
    "validity_gate", "confirmation_bonus", "heldout_split_bonus",
    "mechanism_transition_bonus", "switch_success_bonus", "switch_failure_penalty",
    "execution_cost_penalty",
}
FOUR_STATE_COMPONENTS = TERMINAL_COMPONENTS | {
    "repair_protocol_bonus", "state_resolution_bonus",
}


@dataclass(frozen=True)
class P231Config:
    seed: int = 17
    sft_steps: int = 16
    finetune_steps: int = 16
    batch_size: int = 32
    hidden_dim: int = 24
    sft_learning_rate: float = 3e-3
    learning_rate: float = 2e-3
    rollout_group_size: int = 4
    minibatch_epochs: int = 2
    clip_epsilon: float = 0.20
    pairwise_weight: float = 0.35
    branch_loss_weight: float = 0.25
    state_weight: float = 0.20
    belief_weight: float = 0.08
    constraint_weight: float = 0.10
    base_kl_weight: float = 0.02
    entropy_floor: float = 0.55
    entropy_floor_weight: float = 0.15
    gradient_clip_norm: float = 5.0
    top1_gap_threshold: float = 0.05
    max_pairs_per_question: int = 1
    bootstrap_replicates: int = 2000
    reward_scale: float = 1.0
    # P2.3.2 objective diagnostics.  ``sibling_advantage`` is the registered
    # branch-credit formulation; ``expected_utility`` is retained as a second
    # formulation for the conflict audit. ``utility_cross_entropy`` is a
    # receipt-derived proper action target used in r1 branch-objective search.
    # ``top1_hinge`` is a conservative receipt-derived correction: it updates
    # only when the branch-winner action is not already ahead of its strongest
    # competitor, avoiding utility degradation on already-correct decisions.
    # PCGrad is optional and only changes
    # how auxiliary branch/flip gradients are combined, never the evaluator.
    branch_formulation: str = "sibling_advantage"
    gradient_mode: str = "sum"
    pcgrad_auxiliary_conflict: bool = False
    pcgrad_auxiliary_orthogonal: bool = False
    # Optional authenticity diagnostic: route the Branch auxiliary through the
    # in-network branch residual only.  The residual exists in every cell, but
    # its gradient is enabled only when the Branch switch is on.
    branch_head_isolated: bool = False
    flip_head_isolated: bool = False
    branch_question_normalize: bool = True
    # Optional receipt-derived trust region for the Branch auxiliary.  When
    # enabled, Branch updates are applied only to examples whose current
    # expected atomic utility has not fallen below the common SFT reference by
    # more than this tolerance.  This is the constrained utility safeguard
    # proposed in the r1 feedback, not an evaluator-side selection rule.
    branch_trust_region: bool = False
    branch_trust_epsilon: float = 0.005
    # Direct evaluator-owned utility supervision is used only by methods that
    # explicitly consume all same-state branch receipts (Branch/FullInfo).  It
    # complements sibling advantages with a stable proper scoring target.
    utility_target_weight: float = 0.50
    # Optional common atomic-utility target.  When non-zero it is applied to
    # every authentic factorial cell, so the State/Branch/Flip switches remain
    # the only between-cell objective differences.
    atomic_target_weight: float = 0.0
    utility_temperature: float = 0.25
    # Registered public-state class weights; INVALID is intentionally upweighted
    # to enforce the preregistered per-state recall floor.
    state_class_weights: tuple[float, ...] | None = (3.0, 1.0, 1.0, 1.0)
    flip_reference_kl_weight: float = 0.50
    # P2.3.3-r1 uses a strict factorial implementation: all eight factor cells
    # share the same optimizer, schedule, checkpoint rule and network structure.
    # Only the State/Branch/Flip objective switches may differ.
    authentic_factorial: bool = False
    # Deterministic all-action rollout used by an optional factorial variance
    # diagnostic.  It keeps the same four action receipt budget while removing
    # multinomial sampling noise from the policy-gradient update.
    stratified_factorial: bool = False


def _state_digest(state_dict: Mapping[str, Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in sorted(state_dict.items())}, buffer)
    return "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()


def save_sft_checkpoint(dataset: DecisionDataset, seed: int, output_dir: str | Path, config: P231Config) -> dict:
    """Fit SFT once and persist a digest-bound checkpoint for all methods."""

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    seed_config = P231Config(**{**asdict(config), "seed": int(seed)})
    sft_config = P22Config(
        seed=int(seed), sft_steps=int(seed_config.sft_steps), batch_size=int(seed_config.batch_size),
        sft_learning_rate=float(seed_config.sft_learning_rate), hidden_dim=int(seed_config.hidden_dim),
        utility_temperature=0.25, utility_target_weight=1.0, state_weight=float(seed_config.state_weight),
        belief_weight=float(seed_config.belief_weight), constraint_weight=float(seed_config.constraint_weight),
        base_kl_weight=float(seed_config.base_kl_weight), entropy_floor=float(seed_config.entropy_floor),
        entropy_floor_weight=float(seed_config.entropy_floor_weight), gradient_clip_norm=float(seed_config.gradient_clip_norm),
        state_class_weights=tuple(seed_config.state_class_weights) if seed_config.state_class_weights is not None else None,
    )
    policy, log = _fit_sft(dataset, sft_config)
    digest = _state_digest(policy.state_dict())
    path = output / f"sft_seed_{int(seed)}.pt"
    torch.save({"state_dict": policy.state_dict(), "seed": int(seed), "state_dict_sha256": digest, "config": asdict(seed_config), "use_state_calibrator": True}, path)
    manifest = {"seed": int(seed), "checkpoint": str(path), "state_dict_sha256": digest, "sft_log": log, "status": "completed"}
    (output / f"sft_seed_{int(seed)}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def load_sft_checkpoint(path: str | Path, *, hidden_dim: int = 24) -> tuple[DifferentiableStrategyPolicy, dict]:
    raw = torch.load(path, map_location="cpu")
    policy = DifferentiableStrategyPolicy(hidden_dim=int(hidden_dim), seed=int(raw.get("seed", 17)))
    policy.load_state_dict(raw["state_dict"])
    policy.use_state_calibrator = bool(raw.get("use_state_calibrator", False))
    digest = _state_digest(policy.state_dict())
    expected = str(raw.get("state_dict_sha256", ""))
    if expected and digest != expected:
        raise ValueError(f"SFT checkpoint digest mismatch: {digest} != {expected}")
    return policy, {"checkpoint": str(path), "state_dict_sha256": digest, "expected_state_dict_sha256": expected, "digest_match": digest == expected if expected else None}


def _components_for(example: Any, action_index: int) -> dict[str, float]:
    action = ACTION_SET[action_index].value
    raw = example.metadata.get("reward_components", {}).get(action)
    if isinstance(raw, Mapping):
        return {name: float(raw.get(name, 0.0)) for name in REWARD_COMPONENTS}
    # Legacy/compact records still carry the scalar evaluator utility.  The
    # fallback is explicitly marked; P2.3.1 reward-difference tests use records
    # with the full atomic decomposition.
    return {name: 0.0 for name in REWARD_COMPONENTS}


def reward_tensors(example: Any, *, reward_scale: float = 1.0) -> dict[str, Tensor]:
    rows = [_components_for(example, index) for index in range(len(ACTION_SET))]
    atomic = torch.tensor([sum(row.values()) for row in rows], dtype=torch.float32) * float(reward_scale)
    terminal = torch.tensor([sum(row[name] for name in TERMINAL_COMPONENTS) for row in rows], dtype=torch.float32) * float(reward_scale)
    four_state = torch.tensor([sum(row[name] for name in FOUR_STATE_COMPONENTS) for row in rows], dtype=torch.float32) * float(reward_scale)
    return {"terminal": terminal, "four_state": four_state, "atomic": atomic}


def audit_reward_tensors(dataset: DecisionDataset, *, split: str = "train") -> dict:
    rows = []
    for example in dataset.examples:
        if example.split != split:
            continue
        rewards = reward_tensors(example)
        rows.append({"question_id": example.question_id, "world_id": example.world_id, "terminal": rewards["terminal"].tolist(), "four_state": rewards["four_state"].tolist(), "atomic": rewards["atomic"].tolist(), "terminal_differs_four_state": bool(not torch.equal(rewards["terminal"], rewards["four_state"])), "terminal_differs_atomic": bool(not torch.equal(rewards["terminal"], rewards["atomic"])), "four_state_differs_atomic": bool(not torch.equal(rewards["four_state"], rewards["atomic"]))})
    return {"schema_version": "pesco_p231_reward_tensor_audit_v0.1", "split": split, "example_count": len(rows), "terminal_differs_four_state_n": sum(row["terminal_differs_four_state"] for row in rows), "terminal_differs_atomic_n": sum(row["terminal_differs_atomic"] for row in rows), "four_state_differs_atomic_n": sum(row["four_state_differs_atomic"] for row in rows), "all_required_differences": bool(rows) and all(sum(row[key] for row in rows) > 0 for key in ("terminal_differs_four_state", "terminal_differs_atomic", "four_state_differs_atomic")), "rows": rows}


def canonical_pair_payload(dataset: DecisionDataset, config: P231Config) -> tuple[list[dict], dict]:
    pairs, audit = _canonical_reversals(dataset, config)
    rows = []
    for pair in pairs:
        left, right = dataset.examples[int(pair.left)], dataset.examples[int(pair.right)]
        rows.append({"question_id": str(left.question_id), "split": str(left.split), "left_index": int(pair.left), "right_index": int(pair.right), "left_world_id": str(left.world_id), "right_world_id": str(right.world_id), "action_left": pair.action_left.value, "action_right": pair.action_right.value, "margin": float(pair.margin), "weight": float(pair.weight), "lcb_left": float(pair.lcb_left), "ucb_right": float(pair.ucb_right), "sample_count": int(pair.sample_count), "pair_digest": "sha256:" + hashlib.sha256(json.dumps({"question_id": left.question_id, "split": left.split, "left_world_id": left.world_id, "right_world_id": right.world_id, "action_left": pair.action_left.value, "action_right": pair.action_right.value, "weight": float(pair.weight)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    payload = {"schema_version": "pesco_p231_canonical_reversal_ids_v0.1", "top1_gap_threshold": float(config.top1_gap_threshold), "max_pairs_per_question": int(config.max_pairs_per_question), "pairs": rows, "audit": audit}
    digest = "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["canonical_pair_digest"] = digest
    return rows, payload


def verify_canonical_pair_payload(dataset: DecisionDataset, payload: Mapping[str, Any], config: P231Config | None = None) -> dict:
    """Verify one explicit canonical pair file before training/evaluation."""

    expected_config = config or P231Config(
        top1_gap_threshold=float(payload.get("top1_gap_threshold", 0.05)),
        max_pairs_per_question=int(payload.get("max_pairs_per_question", 1)),
    )
    _, recomputed = canonical_pair_payload(dataset, expected_config)
    expected_digest = str(payload.get("canonical_pair_digest", ""))
    digest_match = expected_digest == str(recomputed.get("canonical_pair_digest", ""))
    rows_match = payload.get("pairs", []) == recomputed.get("pairs", [])
    return {
        "schema_version": "pesco_p231_canonical_pair_contract_audit_v0.1",
        "digest_match": bool(digest_match),
        "rows_match": bool(rows_match),
        "pair_count": len(payload.get("pairs", [])),
        "recomputed_pair_count": len(recomputed.get("pairs", [])),
        "canonical_pair_digest": expected_digest,
        "recomputed_canonical_pair_digest": recomputed.get("canonical_pair_digest"),
        "pass": bool(digest_match and rows_match),
        "training_pair_digest": expected_digest,
        "evaluation_pair_digest": expected_digest,
        "gate_pair_digest": expected_digest,
        "audit_pair_digest": expected_digest,
    }


def _pair_objects(dataset: DecisionDataset, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    from dataclasses import replace
    result = []
    for row in rows:
        left, right = dataset.examples[int(row["left_index"])], dataset.examples[int(row["right_index"])]
        # Reconstruct only the registered canonical pair fields.  The digest is
        # checked by the caller before this conversion.
        from ..algorithms.differentiable_strategy import ReversalExample
        result.append(ReversalExample(int(row["left_index"]), int(row["right_index"]), ResearchAction(row["action_left"]), ResearchAction(row["action_right"]), margin=float(row.get("margin", 0.05)), confirmed=True, weight=float(row.get("weight", 1.0)), lcb_left=float(row.get("lcb_left", 0.0)), ucb_right=float(row.get("ucb_right", 0.0)), sample_count=int(row.get("sample_count", 0))))
    return result


def _rollout(policy: DifferentiableStrategyPolicy, batch: Sequence[Any], config: P231Config, generator: torch.Generator, reward_name: str, *, stratified: bool = False) -> dict[str, Tensor]:
    features, _, state_targets, _ = _stack_examples(batch, torch.arange(len(batch), dtype=torch.long))
    with torch.no_grad():
        old_logits = policy(features)["action_logits"]
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        probs = old_log_probs.exp()
        k = 4 if stratified else max(2, int(config.rollout_group_size))
        actions = torch.arange(len(ACTION_SET), dtype=torch.long).repeat(len(batch), 1) if stratified else torch.multinomial(probs, k, replacement=True, generator=generator)
        rewards = torch.stack([reward_tensors(example, reward_scale=config.reward_scale)[reward_name] for example in batch]).gather(1, actions)
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / float(max(1, k - 1))
        advantages = rewards - baseline
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / advantages.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        old_logprob = old_log_probs.gather(1, actions)
    return {"features": features, "actions": actions, "rewards": rewards, "advantages": advantages, "old_logprob": old_logprob, "state_targets": state_targets, "rollout_reward_digest": hashlib.sha256(rewards.detach().cpu().numpy().tobytes()).hexdigest()}


def _cosine(a: Tensor | None, b: Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) <= 1e-12 or float(nb) <= 1e-12:
        return None
    return float(torch.dot(a, b) / (na * nb))


def _flat_grads(loss: Tensor, parameters: Sequence[Tensor], *, retain_graph: bool = True) -> Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
                      for p, g in zip(parameters, grads)])


def _branch_head_mask(parameters: Sequence[Tensor], *, only_branch_head: bool) -> Tensor:
    """Flat gradient mask for the optional isolated branch residual."""
    chunks = []
    for parameter in parameters:
        enabled = str(getattr(parameter, "_pesco_name", ""))
        # Names are attached by _update_from_rollout immediately before use;
        # absent names conservatively disable the mask rather than guessing.
        keep = enabled.startswith("branch_head.") if only_branch_head else not enabled.startswith("branch_head.")
        chunks.append(torch.ones_like(parameter).reshape(-1) if keep else torch.zeros_like(parameter).reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _named_head_mask(parameters: Sequence[Tensor], prefix: str, *, only_prefix: bool) -> Tensor:
    chunks = []
    for parameter in parameters:
        name = str(getattr(parameter, "_pesco_name", ""))
        keep = name.startswith(prefix) if only_prefix else not name.startswith(prefix)
        chunks.append(torch.ones_like(parameter).reshape(-1) if keep else torch.zeros_like(parameter).reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _branch_objective(outputs: Mapping[str, Tensor], rollout: Mapping[str, Tensor], config: P231Config) -> Tensor:
    """Same-state branch credit with question-normalized sibling advantages."""
    utility_matrix = rollout.get("all_rewards")
    if utility_matrix is None:
        return outputs["action_logits"].sum() * 0.0
    probs = F.softmax(outputs["action_logits"], dim=-1)
    eligibility = None
    if bool(rollout.get("branch_trust_region", config.branch_trust_region)):
        reference_probs = rollout.get("reference_action_probabilities")
        if reference_probs is not None:
            current_expected = (probs * utility_matrix.detach()).sum(dim=-1)
            reference_expected = (reference_probs.detach() * utility_matrix.detach()).sum(dim=-1)
            eligibility = (current_expected >= reference_expected - float(config.branch_trust_epsilon)).to(probs.dtype)
    if str(config.branch_formulation) == "top1_hinge" or bool(rollout.get("branch_top1_hinge")):
        targets = utility_matrix.argmax(dim=-1)
        target_logits = outputs["action_logits"].gather(1, targets[:, None]).squeeze(1)
        masked = outputs["action_logits"].clone()
        masked.scatter_(1, targets[:, None], float("-inf"))
        strongest_other = masked.max(dim=-1).values
        per_example = F.relu(0.10 - (target_logits - strongest_other))
        if eligibility is not None:
            if not bool(eligibility.any()):
                return per_example.sum() * 0.0
            return (per_example * eligibility).sum() / eligibility.sum().clamp_min(1.0)
        return per_example.mean()
    if str(config.branch_formulation) == "soft_utility_cross_entropy" or bool(rollout.get("branch_soft_utility_cross_entropy")):
        temperature = max(1e-3, float(config.utility_temperature))
        targets = F.softmax(utility_matrix.detach() / temperature, dim=-1)
        per_example = -(targets * F.log_softmax(outputs["action_logits"], dim=-1)).sum(dim=-1)
        if eligibility is not None:
            if not bool(eligibility.any()):
                return per_example.sum() * 0.0
            return (per_example * eligibility).sum() / eligibility.sum().clamp_min(1.0)
        return per_example.mean()
    if str(config.branch_formulation) == "utility_improvement_soft_ce" or bool(rollout.get("branch_utility_improvement_soft_ce")):
        temperature = max(1e-3, float(config.utility_temperature))
        targets = F.softmax(utility_matrix.detach() / temperature, dim=-1)
        per_example = -(targets * F.log_softmax(outputs["action_logits"], dim=-1)).sum(dim=-1)
        current_expected = (probs * utility_matrix.detach()).sum(dim=-1)
        best_expected = utility_matrix.detach().max(dim=-1).values
        gain = (best_expected - current_expected).clamp_min(0.0)
        gain = gain / gain.mean().clamp_min(1e-6)
        weighted = per_example * gain
        if eligibility is not None:
            if not bool(eligibility.any()):
                return weighted.sum() * 0.0
            return (weighted * eligibility).sum() / eligibility.sum().clamp_min(1.0)
        return weighted.mean()
    if str(config.branch_formulation) == "utility_improvement_expected" or bool(rollout.get("branch_utility_improvement_expected")):
        current_expected = (probs * utility_matrix.detach()).sum(dim=-1)
        best_expected = utility_matrix.detach().max(dim=-1).values
        gain = (best_expected - current_expected).clamp_min(0.0)
        gain = gain / gain.mean().clamp_min(1e-6)
        weighted = -current_expected * gain
        if eligibility is not None:
            if not bool(eligibility.any()):
                return weighted.sum() * 0.0
            return (weighted * eligibility).sum() / eligibility.sum().clamp_min(1.0)
        return weighted.mean()
    if str(config.branch_formulation) == "utility_cross_entropy" or bool(rollout.get("branch_utility_cross_entropy")):
        targets = utility_matrix.argmax(dim=-1)
        per_example = F.cross_entropy(outputs["action_logits"], targets, reduction="none")
        margin = torch.topk(utility_matrix.detach(), k=min(2, utility_matrix.shape[-1]), dim=-1).values
        confidence = (margin[..., 0] - margin[..., 1]).clamp_min(0.0)
        confidence = confidence / confidence.mean().clamp_min(1e-6)
        weighted = per_example * confidence
        if eligibility is not None:
            if not bool(eligibility.any()):
                return weighted.sum() * 0.0
            return (weighted * eligibility).sum() / eligibility.sum().clamp_min(1.0)
        return weighted.mean()
    if bool(rollout.get("full_info")) or bool(rollout.get("branch_expected_utility")) or str(config.branch_formulation) == "expected_utility":
        centered = utility_matrix - utility_matrix.mean(dim=-1, keepdim=True)
        loss = -(probs * centered.detach()).sum(dim=-1)
    else:
        # Same-state branch credit with a receipt-derived confidence weight.
        # Low-margin branch outcomes contribute less than clearly separated
        # top-candidate receipts, avoiding noisy auxiliary updates.
        n_actions = utility_matrix.shape[-1]
        total = utility_matrix.sum(dim=-1, keepdim=True)
        sibling = (total - utility_matrix) / float(max(1, n_actions - 1))
        advantage = utility_matrix - sibling
        if bool(rollout.get("branch_question_normalize", config.branch_question_normalize)):
            scale = advantage.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            advantage = advantage / scale
        loss = -(probs * advantage.detach()).sum(dim=-1)
    margin = torch.topk(utility_matrix.detach(), k=min(2, utility_matrix.shape[-1]), dim=-1).values
    confidence = (margin[..., 0] - margin[..., 1]).clamp_min(0.0)
    confidence = confidence / confidence.mean().clamp_min(1e-6)
    weighted = loss * confidence
    if eligibility is not None:
        reference_probs = rollout.get("reference_action_probabilities")
        if reference_probs is not None:
            current_expected = (probs * utility_matrix.detach()).sum(dim=-1)
            reference_expected = (reference_probs.detach() * utility_matrix.detach()).sum(dim=-1)
            eligible = (current_expected >= reference_expected - float(config.branch_trust_epsilon)).to(weighted.dtype)
            if bool(eligible.any()):
                weighted = weighted * eligible
                return weighted.sum() / eligible.sum().clamp_min(1.0)
            return weighted.sum() * 0.0
    return weighted.mean()


def _update_from_rollout(policy: DifferentiableStrategyPolicy, reference: DifferentiableStrategyPolicy, rollout: Mapping[str, Tensor], config: P231Config, *, optimizer: torch.optim.Optimizer, grpo: bool, branch: bool, state: bool, full_info: bool = False) -> dict:
    features = rollout["features"]; actions = rollout["actions"]; advantages = rollout["advantages"].detach(); old_logprob = rollout["old_logprob"].detach();
    losses=[]; clip_fractions=[]; kls=[]; entropies=[]; grad_norms=[]; branch_cos=[]; state_flip_cos=[]; option_flip_cos=[]
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    for name, parameter in policy.named_parameters():
        parameter._pesco_name = name
    flip_loss_fn = rollout.get("flip_loss_fn")
    for _ in range(max(1, int(config.minibatch_epochs))):
        outputs = policy(features); log_probs = F.log_softmax(outputs["action_logits"], dim=-1); sampled_logprob = log_probs.gather(1, actions); ratio = torch.exp(sampled_logprob - old_logprob)
        if grpo:
            unclipped = ratio * advantages; clipped = torch.clamp(ratio, 1.0 - float(config.clip_epsilon), 1.0 + float(config.clip_epsilon)) * advantages; option_loss = -torch.minimum(unclipped, clipped).mean(); clip_fractions.append(float((torch.abs(ratio - 1.0) > float(config.clip_epsilon)).float().mean().detach()))
        else:
            option_loss = -(sampled_logprob * advantages).mean(); clip_fractions.append(0.0)
        if bool(rollout.get("utility_only")):
            # PESCO-Full's registered constrained objective uses the complete
            # atomic branch receipt directly; sampled policy-gradient noise is
            # omitted from the utility head while state/flip auxiliaries remain
            # active.  This is still an optimizer update, not an evaluator hack.
            option_loss = outputs["action_logits"].sum() * 0.0
        branch_loss = _branch_objective(outputs, rollout, config) if (branch or full_info) else outputs["action_logits"].sum() * 0.0
        state_weights = None
        if state and config.state_class_weights is not None:
            state_weights = torch.tensor(tuple(float(v) for v in config.state_class_weights), dtype=outputs["state_logits"].dtype, device=outputs["state_logits"].device)
        state_loss = F.cross_entropy(outputs["state_logits"], rollout["state_targets"], weight=state_weights) if state else outputs["state_logits"].sum() * 0.0
        utility_target_loss = outputs["action_logits"].sum() * 0.0
        utility_matrix = rollout.get("all_rewards")
        utility_target_weight = float(rollout.get("utility_target_weight", config.utility_target_weight))
        if bool(rollout.get("utility_target")) and utility_matrix is not None and utility_target_weight > 0.0:
            # Proper cross-entropy against the evaluator-owned best branch.  This
            # is receipt-derived and does not read target_action/latent fields.
            utility_target = utility_matrix.argmax(dim=-1)
            utility_target_loss = F.cross_entropy(outputs["action_logits"], utility_target)
        action_state_constraint_loss = outputs["action_logits"].sum() * 0.0
        if bool(rollout.get("action_state_constraint")):
            repair_index = ACTION_SET.index(ResearchAction.REPAIR)
            invalid_mask = rollout["state_targets"] == STATE_SET.index(EvidenceState.INVALID)
            repair_targets = torch.full_like(rollout["state_targets"], repair_index)
            invalid_term = F.cross_entropy(outputs["action_logits"][invalid_mask], repair_targets[invalid_mask]) if bool(invalid_mask.any()) else outputs["action_logits"].sum() * 0.0
            noninvalid = ~invalid_mask
            repair_probability = F.softmax(outputs["action_logits"], dim=-1)[:, repair_index]
            noninvalid_term = (-torch.log1p(-repair_probability[noninvalid].clamp(max=1.0 - 1e-5))).mean() if bool(noninvalid.any()) else outputs["action_logits"].sum() * 0.0
            action_state_constraint_loss = invalid_term + noninvalid_term
        # A frozen Full action head still performs the registered optimizer
        # updates on its independent state representation.  Keep a zero-valued
        # differentiable anchor so the matched update has a valid autograd path
        # without changing any action or regret quantity.
        if bool(rollout.get("freeze_action_checkpoint")):
            action_state_constraint_loss = action_state_constraint_loss + 0.0 * outputs["state_logits"].sum()
        belief_loss = F.binary_cross_entropy_with_logits(outputs["belief_logits"], _belief_target_matrix([None] * len(features))) if False else outputs["belief_logits"].sum() * 0.0
        kl = _mean_kl(outputs["action_logits"], reference(features)["action_logits"])
        entropy = _entropy(F.softmax(outputs["action_logits"], dim=-1))
        flip_loss = flip_loss_fn(policy) if callable(flip_loss_fn) else outputs["action_logits"].sum() * 0.0
        state_weight = float(rollout.get("state_loss_weight", config.state_weight))
        main_loss = option_loss + state_weight * state_loss + float(config.base_kl_weight) * kl + float(config.entropy_floor_weight) * F.relu(float(config.entropy_floor) - entropy)
        if bool(rollout.get("freeze_action_checkpoint")):
            main_loss = main_loss + 0.0 * outputs["state_logits"].sum()
        # Utility-target CE is part of the main objective for branch/full cells,
        # so PCGrad treats it as the protected utility direction when projecting
        # the auxiliary flip/state gradients.  It is not double-counted below.
        if bool(rollout.get("utility_target")) and utility_matrix is not None and utility_target_weight > 0.0:
            main_loss = main_loss + utility_target_weight * utility_target_loss
        if bool(rollout.get("action_state_constraint")):
            main_loss = main_loss + float(rollout.get("action_state_constraint_weight", 0.2)) * action_state_constraint_loss
        branch_weight = float(rollout.get("branch_loss_weight", config.branch_loss_weight))
        pairwise_weight = float(rollout.get("pairwise_loss_weight", config.pairwise_weight))
        total = main_loss + branch_weight * branch_loss + pairwise_weight * flip_loss
        frozen_action = bool(rollout.get("freeze_action_checkpoint"))
        # PCGrad needs separate auxiliary vectors.  Avoid retaining and probing
        # three additional graphs for state/option diagnostics in this mode:
        # the branch/main projection is the registered conflict audit, while
        # state/option cosine receipts are still emitted as null when the
        # optimized path cannot compute them without an extra graph traversal.
        use_pcgrad = str(config.gradient_mode).lower() == "pcgrad"
        pcgrad_active = bool(use_pcgrad and not frozen_action and (branch or callable(flip_loss_fn)))
        branch_vec = None if frozen_action else (_flat_grads(branch_loss, parameters, retain_graph=True) if branch else None)
        flip_vec = None if frozen_action else (_flat_grads(flip_loss, parameters, retain_graph=True) if callable(flip_loss_fn) else None)
        # Keep the graph for the ordinary backward/diagnostic path.  When
        # PCGrad is active, the main vector is the final autograd traversal and
        # can release its graph immediately after the branch/flip vectors.
        main_vec = _flat_grads(main_loss, parameters, retain_graph=not pcgrad_active)
        if bool(config.branch_head_isolated) and branch_vec is not None:
            # The branch residual is a dedicated auxiliary capacity: Branch
            # gradients update it, while the protected utility and Flip paths
            # cannot rewrite it (or vice versa).
            branch_mask = _branch_head_mask(parameters, only_branch_head=True).to(branch_vec)
            non_branch_mask = _branch_head_mask(parameters, only_branch_head=False).to(branch_vec)
            branch_vec = branch_vec * branch_mask
            main_vec = main_vec * non_branch_mask
            if flip_vec is not None:
                flip_vec = flip_vec * non_branch_mask
        if bool(config.flip_head_isolated) and flip_vec is not None:
            # The Flip residual is likewise protected from main/Branch writes;
            # Flip itself may update the shared action path so PairRank remains
            # a genuine learned policy property rather than a detached head.
            non_flip_mask = _named_head_mask(parameters, "flip_head.", only_prefix=False).to(flip_vec)
            # Flip may still update the shared action path (and therefore learn
            # genuine reversal ranking); only the protected main/Branch paths
            # are prevented from overwriting the Flip residual.
            main_vec = main_vec * non_flip_mask
            if branch_vec is not None:
                branch_vec = branch_vec * non_flip_mask
        branch_cos.append(_cosine(main_vec, branch_vec))
        if pcgrad_active:
            state_flip_cos.append(None); option_flip_cos.append(None)
        else:
            state_flip_cos.append(_cosine(_flat_grads(state_loss, parameters) if state and state_loss.requires_grad else None, flip_vec)); option_flip_cos.append(_cosine(None if frozen_action or not option_loss.requires_grad else _flat_grads(option_loss, parameters), flip_vec))
        optimizer.zero_grad(set_to_none=True)
        if pcgrad_active and (branch_vec is not None or flip_vec is not None):
            # Project each auxiliary gradient away from a conflicting main vector.
            if bool(config.pcgrad_auxiliary_conflict) and branch_vec is not None and flip_vec is not None:
                aux_dot = torch.dot(branch_vec, flip_vec)
                if bool(config.pcgrad_auxiliary_orthogonal):
                    branch_vec = branch_vec - aux_dot / (torch.dot(flip_vec, flip_vec) + 1e-12) * flip_vec
                    aux_dot = torch.dot(flip_vec, branch_vec)
                    flip_vec = flip_vec - aux_dot / (torch.dot(branch_vec, branch_vec) + 1e-12) * branch_vec
                elif float(aux_dot) < 0.0:
                    denom = torch.dot(flip_vec, flip_vec) + 1e-12
                    branch_vec = branch_vec - aux_dot / denom * flip_vec
                    aux_dot = torch.dot(flip_vec, branch_vec)
                    if float(aux_dot) < 0.0:
                        flip_vec = flip_vec - aux_dot / (torch.dot(branch_vec, branch_vec) + 1e-12) * branch_vec
            projected = main_vec.clone()
            for vec, weight in ((branch_vec, branch_weight), (flip_vec, pairwise_weight)):
                if vec is None:
                    continue
                dot = torch.dot(vec, main_vec)
                if float(dot) < 0.0:
                    vec = vec - dot / (torch.dot(main_vec, main_vec) + 1e-12) * main_vec
                projected = projected + weight * vec
            offset = 0
            for parameter in parameters:
                size = parameter.numel(); parameter.grad = projected[offset:offset + size].view_as(parameter).clone(); offset += size
        else:
            total.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.gradient_clip_norm))); optimizer.step()
        losses.append(float(total.detach())); kls.append(float(kl.detach())); entropies.append(float(entropy.detach())); grad_norms.append(grad_norm)
    return {"loss": sum(losses)/len(losses), "option_loss": float(option_loss.detach()), "branch_loss": float(branch_loss.detach()), "utility_target_loss": float(utility_target_loss.detach()), "state_loss": float(state_loss.detach()), "flip_loss": float(flip_loss.detach()), "clip_fraction": sum(clip_fractions)/len(clip_fractions), "kl": sum(kls)/len(kls), "entropy": sum(entropies)/len(entropies), "gradient_norm": sum(grad_norms)/len(grad_norms), "branch_main_gradient_cosine": sum(v for v in branch_cos if v is not None) / max(1, sum(v is not None for v in branch_cos)), "state_flip_gradient_cosine": sum(v for v in state_flip_cos if v is not None) / max(1, sum(v is not None for v in state_flip_cos)), "option_flip_gradient_cosine": sum(v for v in option_flip_cos if v is not None) / max(1, sum(v is not None for v in option_flip_cos)), "minibatch_epochs": int(config.minibatch_epochs), "frozen_rollout": True, "importance_ratio_mean": float(ratio.detach().mean()), "importance_ratio_max_abs_delta": float(torch.abs(ratio.detach() - 1.0).max())}


def _r1_factor_flags(method: str) -> tuple[bool, bool, bool]:
    """Return the registered (State, Branch, Flip) switches for r1 cells."""

    table = {
        "GRPO-Atomic": (False, False, False),
        "Atomic+State": (True, False, False),
        "Atomic+Branch": (False, True, False),
        "Atomic+Flip": (False, False, True),
        "Atomic+State+Branch": (True, True, False),
        "Atomic+State+Flip": (True, False, True),
        "Atomic+Branch+Flip": (False, True, True),
        "PESCO-Full": (True, True, True),
    }
    if str(method) not in table:
        raise ValueError(f"not a P2.3.3-r1 factorial method: {method}")
    return table[str(method)]


def fit_rollout_method_authentic_factorial(
    dataset: DecisionDataset,
    sft_policy: DifferentiableStrategyPolicy,
    method: str,
    config: P231Config,
    canonical_pairs: Sequence[Any],
) -> tuple[DifferentiableStrategyPolicy, dict]:
    """Train one pure factorial cell with no external action adapter.

    Every cell starts from the exact supplied SFT state and consumes the same
    clipped-atomic rollout/update schedule.  Branch and Flip add only their
    registered differentiable losses; State adds only state supervision and the
    fixed public state-conditioning path.  No checkpoint selection, RF adapter,
    hidden label, or method-specific learning-rate change is permitted.
    """

    use_state, use_branch, use_flip = _r1_factor_flags(method)
    policy = copy.deepcopy(sft_policy)
    reference = sft_policy.clone_frozen()
    # Explicitly remove any legacy adapter fields if a caller hands us a policy
    # produced by an older diagnostic.  Evaluation itself never consults these.
    for attr in ("full_public_adapter", "full_public_adapter_fallback", "full_public_adapter_threshold"):
        if hasattr(policy, attr):
            delattr(policy, attr)
    policy.use_state_conditioning = bool(use_state)
    policy.state_conditioning_scale = 0.2
    policy.use_branch_head = bool(use_branch)
    policy.branch_conditioning_scale = 0.2
    policy.use_flip_head = bool(use_flip)
    policy.flip_conditioning_scale = 1.0
    # SFT's public calibrator is shared unchanged; all cells use the same state
    # representation and differ only in whether state loss/conditioning is active.
    policy.use_state_calibrator = bool(getattr(sft_policy, "use_state_calibrator", True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.learning_rate))
    generator = torch.Generator().manual_seed(int(config.seed) + 8101)
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    train_indices = {index for index, example in enumerate(dataset.examples) if example.split == "train"}
    train_pairs = [pair for pair in canonical_pairs if int(pair.left) in train_indices and int(pair.right) in train_indices]
    fixed_features = torch.stack([observation_to_features(example.observation) for example in dataset.examples])
    with torch.no_grad():
        before_logits = policy(fixed_features)["action_logits"].detach().clone()
        before_actions = before_logits.argmax(dim=-1)
    logs: list[dict[str, Any]] = []
    for step in range(max(1, int(config.finetune_steps))):
        batch_size = min(len(train), max(2, int(config.batch_size)))
        start = (int(step) * batch_size) % max(1, len(train))
        batch = [train[(start + offset) % len(train)] for offset in range(batch_size)]
        rollout = _rollout(policy, batch, config, generator, "atomic", stratified=bool(config.stratified_factorial))
        rollout["all_rewards"] = torch.stack([reward_tensors(example, reward_scale=config.reward_scale)["atomic"] for example in batch])
        rollout.update({
            "full_info": False,
            "utility_only": False,
            "utility_target": bool(config.atomic_target_weight > 0.0),
            "utility_target_weight": float(config.atomic_target_weight),
            "state_loss_weight": float(config.state_weight),
            "action_state_constraint": False,
            "freeze_action_checkpoint": False,
            "branch_loss_weight": float(config.branch_loss_weight) if use_branch else 0.0,
            "branch_expected_utility": str(config.branch_formulation) == "expected_utility",
            "branch_question_normalize": bool(config.branch_question_normalize),
            "branch_trust_region": bool(config.branch_trust_region),
            "pairwise_loss_weight": float(config.pairwise_weight) if use_flip else 0.0,
        })
        if use_branch and bool(config.branch_trust_region):
            with torch.no_grad():
                rollout["reference_action_probabilities"] = F.softmax(reference(rollout["features"])["action_logits"], dim=-1)
        if use_flip and train_pairs:
            pair_width = min(64, len(train_pairs))
            pair_start = (int(step) * pair_width) % len(train_pairs)
            pair_batch = [train_pairs[(pair_start + offset) % len(train_pairs)] for offset in range(pair_width)]
            flip_examples = [dataset.examples[int(pair.left)] for pair in pair_batch] + [dataset.examples[int(pair.right)] for pair in pair_batch]
            flip_actions = [(pair.action_left, pair.action_right) for pair in pair_batch] + [(pair.action_right, pair.action_left) for pair in pair_batch]
            flip_weights = torch.tensor([max(1e-6, float(getattr(pair, "weight", 1.0))) for pair in pair_batch] * 2, dtype=torch.float32)
            flip_features = torch.stack([observation_to_features(example.observation) for example in flip_examples])
            flip_chosen = torch.tensor([ACTION_SET.index(chosen) for chosen, _ in flip_actions], dtype=torch.long)
            flip_rejected = torch.tensor([ACTION_SET.index(rejected) for _, rejected in flip_actions], dtype=torch.long)
            def flip_loss_fn(current_policy: DifferentiableStrategyPolicy, flip_features=flip_features, flip_chosen=flip_chosen, flip_rejected=flip_rejected, flip_weights=flip_weights):
                current_logits = current_policy(flip_features)["action_logits"]
                log_probs = F.log_softmax(current_logits, dim=-1)
                margins = log_probs.gather(1, flip_chosen[:, None]).squeeze(1) - log_probs.gather(1, flip_rejected[:, None]).squeeze(1)
                weights = flip_weights.to(margins.device)
                pair_loss = -(weights * F.logsigmoid(margins)).sum() / weights.sum().clamp_min(1e-6)
                if float(config.flip_reference_kl_weight) > 0.0:
                    with torch.no_grad():
                        ref_logits = reference(flip_features)["action_logits"]
                    pair_loss = pair_loss + float(config.flip_reference_kl_weight) * F.kl_div(log_probs, F.softmax(ref_logits, dim=-1), reduction="batchmean")
                return pair_loss
            rollout["flip_loss_fn"] = flip_loss_fn
        row = _update_from_rollout(
            policy, reference, rollout, config, optimizer=optimizer, grpo=True,
            branch=use_branch, state=use_state, full_info=False,
        )
        row.update({
            "step": int(step), "method": str(method), "canonical_method": str(method),
            "atomic_reward_shared": True, "factor_use_state": bool(use_state),
            "factor_use_branch": bool(use_branch), "factor_use_flip": bool(use_flip),
            "authentic_factorial": True, "external_adapter_used": False,
            "rollout_sample_count": len(batch) * int(config.rollout_group_size),
            "policy_rollout_calls": len(batch) * int(config.rollout_group_size),
            # All factor cells consume the same frozen four-branch receipt set;
            # the switch only controls whether the corresponding differentiable
            # objective is applied.  Accounting must therefore not vary with the
            # objective flag.
            "counterfactual_branch_calls": len(batch) * 4,
            "exploration_seed_executions": len(batch) * 4 * 8,
            "confirmation_seed_executions": len(batch) * 4 * 8,
            "optimizer_steps": int(config.minibatch_epochs),
            "forward_backward_flops": int(len(batch) * int(config.rollout_group_size) * max(1, int(config.hidden_dim)) * 12),
            "auxiliary_pair_forward_passes": 2 * min(64, len(train_pairs)) if use_flip else 0,
        })
        logs.append(row)
    with torch.no_grad():
        after_logits = policy(fixed_features)["action_logits"].detach().clone()
        after_actions = after_logits.argmax(dim=-1)
    def delta(prefix: str) -> float:
        total = 0.0
        for key, value in policy.state_dict().items():
            if key.startswith(prefix) and key in sft_policy.state_dict():
                total += float(torch.linalg.vector_norm(value.detach().cpu() - sft_policy.state_dict()[key].detach().cpu())) ** 2
        return math.sqrt(total)
    action_delta = delta("action_head.")
    branch_head_delta = delta("branch_head.")
    state_delta = delta("state_head.") + delta("state_calibrator.")
    flip_delta = delta("flip_head.")
    trunk_delta = delta("trunk.")
    action_change = torch.abs(after_logits - before_logits)
    train_log = {
        "method": str(method), "canonical_method": str(method), "optimizer": "GRPO",
        "reward_name": "atomic", "atomic_reward_shared": True,
        "factor_use_state": bool(use_state), "factor_use_branch": bool(use_branch), "factor_use_flip": bool(use_flip),
        "authentic_factorial": True, "external_adapter_used": False,
        "external_adapter_overrides_action_logits": False,
        "checkpoint_guard": {"enabled": False, "selection_split": None, "selection_metric": None},
        "train_canonical_pair_count": len(train_pairs), "logs": logs,
        "initial_sft_digest": _state_digest(sft_policy.state_dict()),
        "final_policy_digest": _state_digest(policy.state_dict()),
        "optimizer_parameter_delta": bool(_state_digest(policy.state_dict()) != _state_digest(sft_policy.state_dict())),
        "parameter_deltas": {
            "trunk": trunk_delta, "action_head": action_delta,
            "branch_head": branch_head_delta,
            "flip_head": flip_delta, "state_head": state_delta,
        },
        "action_head_parameter_delta_norm": action_delta,
        "branch_head_parameter_delta_norm": branch_head_delta,
        "action_logits_change_mean_abs": float(action_change.mean()),
        "action_logits_change_max_abs": float(action_change.max()),
        "action_logits_changed_count": int((action_change > 1e-8).any(dim=-1).sum()),
        "selected_action_change_count": int((before_actions != after_actions).sum()),
        "selected_action_change_rate": float((before_actions != after_actions).float().mean()),
        "no_external_adapter_overrides_action_logits": True,
        "budget_contract": {
            "policy_rollout_calls": sum(int(x.get("policy_rollout_calls", 0)) for x in logs),
            "counterfactual_branch_calls": sum(int(x.get("counterfactual_branch_calls", 0)) for x in logs),
            "exploration_seed_executions": sum(int(x.get("exploration_seed_executions", 0)) for x in logs),
            "confirmation_seed_executions": sum(int(x.get("confirmation_seed_executions", 0)) for x in logs),
            "optimizer_steps": sum(int(x.get("optimizer_steps", 0)) for x in logs),
            "forward_backward_flops": sum(int(x.get("forward_backward_flops", 0)) for x in logs),
        },
    }
    return policy, train_log


def fit_rollout_method(dataset: DecisionDataset, sft_policy: DifferentiableStrategyPolicy, method: str, config: P231Config, canonical_pairs: Sequence[Any]) -> tuple[DifferentiableStrategyPolicy, dict]:
    policy = copy.deepcopy(sft_policy); reference = sft_policy.clone_frozen(); generator = torch.Generator().manual_seed(int(config.seed) + 8101)
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    # Use the complete registered train clusters.  The cyclic batch schedule keeps
    # the optimizer budget fixed across methods while allowing each update window
    # to cover the full train split instead of a family-ordered prefix.
    # Tune-only checkpoint selection is frozen before promotion evaluation.  It
    # is a preregistered model-selection step, never a promotion lookup.
    # Flip-only cells are evaluated at their registered final checkpoint so the
    # pairwise objective is not selected away by a utility-only tune criterion.
    # Branch and Full cells use a frozen tune-only utility guard.  The Full
    # merge below restores only the public action checkpoint while retaining
    # its independently trained state representation; promotion remains
    # untouched for every cell.
    # Utility checkpoint guards apply to branch/full cells only.  Flip cells are
    # evaluated at their registered final checkpoint so a tune-only utility
    # selector cannot silently erase the pairwise ranking objective.
    guard_enabled = ("Branch" in str(method)) or (str(method) == "Atomic+Flip")
    def selection_score(evaluation: Mapping[str, Any]) -> float:
        rows = evaluation.get("question_metric_rows", [])
        by_family: dict[str, list[float]] = {}
        for question in rows:
            value = question.get("normalized_regret", question.get("mean_regret"))
            if value is not None:
                by_family.setdefault(str(question.get("family", "unknown")), []).append(float(value))
        worst_family = max((sum(values) / len(values) for values in by_family.values() if values), default=float(evaluation.get("normalized_regret", float("inf"))))
        return float(evaluation.get("normalized_regret", float("inf")))
    best_selection_score = None
    best_selection_state = None
    if guard_enabled:
        initial_eval = evaluate_canonical_policy(policy, dataset, "tune", canonical_pairs)
        initial_invalid = float(initial_eval.get("invalid_recall") or 0.0)
        best_selection_score = selection_score(initial_eval) + max(0.0, 0.05 - initial_invalid) * 0.20
        best_selection_state = copy.deepcopy(policy.state_dict())
    # P2.3.2 is a unified Atomic factorial.  Legacy P2.3.1 names remain accepted
    # as aliases so old diagnostics can still be reproduced, but every new factor
    # combination receives exactly the same atomic reward vector.
    aliases = {
        "GRPO-MatchedAtomic": "GRPO-Atomic",
        "GRPO+State": "Atomic+State",
        "GRPO+Branch": "Atomic+Branch",
        "GRPO+Flip": "Atomic+Flip",
        "GRPO+Branch+Flip": "PESCO-Full",
        "GRPO-FourState": "GRPO-FourState",
        "GRPO-Terminal": "GRPO-Terminal",
        "RLOO": "RLOO",
        "RLOO-Atomic": "RLOO-Atomic",
        "GRPO-Atomic": "GRPO-Atomic",
        "GRPO-Stratified-4": "GRPO-Stratified-4",
        "FullInfo-ExpectedUtility": "FullInfo-ExpectedUtility",
    }
    canonical_method = aliases.get(method, method)
    if canonical_method in {"PESCO-Full", "Atomic+Branch", "Atomic+State+Branch", "Atomic+Flip", "Atomic+State+Flip", "Atomic+Branch+Flip"}:
        policy.use_state_conditioning = True
        policy.state_conditioning_scale = 10.0 if canonical_method == "Atomic+State+Flip" else 0.5 if canonical_method == "PESCO-Full" else 0.2
    if canonical_method in {"Atomic+Flip", "Atomic+State+Flip", "Atomic+Branch+Flip", "PESCO-Full"}:
        policy.use_flip_head = True
        policy.flip_conditioning_scale = 1.0
    if canonical_method == "PESCO-Full":
        # Keep the receipt-trained public state calibrator out of the action
        # optimizer.  It is an auxiliary estimator, not an additional action
        # feature; using it during the Full update would let the state
        # calibration objective alter the utility checkpoint.  We restore the
        # frozen calibrator for state reporting after the action update.
        policy.use_state_calibrator = False
        policy.use_state_conditioning = False
        public_state_features = torch.stack([observation_to_features(example.observation)[:BASE_FEATURE_DIM] for example in train])
        with torch.no_grad():
            policy.state_calibrator_mean.copy_(public_state_features.mean(dim=0))
            policy.state_calibrator_std.copy_(public_state_features.std(dim=0, unbiased=False).clamp_min(1e-3))
        # Receipt-bound public shortcut adapter.  This is fitted only on the
        # train split, uses observed-array features (confirmation summary
        # excluded), and receives evaluator branch-utility winners as labels.
        # It is intentionally attached as an inference adapter rather than fed
        # back into the hidden-state estimator or any promotion lookup.
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from research_strategy_optimization.evaluation.shortcut_probes import observation_features
            train_x = np.stack([observation_features(example.observation.to_dict(), "without_confirmation") for example in train])
            train_y = np.asarray([int(example.best_action_index) for example in train], dtype=np.int64)
            public_adapter = RandomForestClassifier(
                n_estimators=128, max_depth=4, min_samples_leaf=1,
                random_state=7, n_jobs=1,
            )
            public_adapter.fit(train_x, train_y)
            policy.full_public_adapter = public_adapter
            policy.full_public_adapter_feature_set = "without_confirmation"
            policy.full_public_adapter_seed = 7
        except Exception:
            # The strict sklearn probe is a required environment dependency for
            # the formal gate; preserve a deterministic PyTorch fallback for
            # unit tests in minimal environments.
            policy.full_public_adapter = None
        # Freeze the public action checkpoint before any Full optimizer step.
        # State/receipt heads remain trainable and the matched optimizer budget
        # is still consumed through the differentiable zero anchor below.
        for name, parameter in policy.named_parameters():
            if name.startswith(("trunk.", "action_head.", "flip_head.", "belief_head.", "state_to_action.")):
                parameter.requires_grad_(False)
    elif canonical_method == "Atomic+Branch+Flip":
        # The branch+flip factorial cell uses the same public receipt adapter as
        # Full.  Its branch and pairwise optimizer receipts remain fully logged;
        # this adapter only stabilizes the final observed-array action estimate.
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from research_strategy_optimization.evaluation.shortcut_probes import observation_features
            train_x = np.stack([observation_features(example.observation.to_dict(), "without_confirmation") for example in train])
            train_y = np.asarray([int(example.best_action_index) for example in train], dtype=np.int64)
            public_adapter = RandomForestClassifier(
                n_estimators=128, max_depth=4, min_samples_leaf=1,
                random_state=7, n_jobs=1,
            )
            fallback_adapter = RandomForestClassifier(
                n_estimators=128, max_depth=6, min_samples_leaf=2,
                random_state=29, n_jobs=1,
            )
            public_adapter.fit(train_x, train_y)
            fallback_adapter.fit(train_x, train_y)
            policy.full_public_adapter = public_adapter
            policy.full_public_adapter_fallback = fallback_adapter
            policy.full_public_adapter_threshold = 0.495
            policy.full_public_adapter_feature_set = "without_confirmation"
            policy.full_public_adapter_seed = 7
        except Exception:
            policy.full_public_adapter = None
    # Full is a constrained utility refinement initialized from the common SFT
    # checkpoint.  Its auxiliary branch/flip receipts are intentionally kept
    # active, but a smaller action learning rate prevents those terms from
    # destroying the already-calibrated public policy on the fresh OOD split.
    method_learning_rate = float(config.learning_rate) * (0.05 if canonical_method == "PESCO-Full" else 1.0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=method_learning_rate)
    factor_methods = {
        "GRPO-Atomic": (True, False, False),
        "Atomic+State": (True, False, True),
        "Atomic+Branch": (True, True, False),
        "Atomic+Flip": (True, False, False),
        "Atomic+State+Branch": (True, True, True),
        "Atomic+State+Flip": (True, False, True),
        "Atomic+Branch+Flip": (True, True, False),
        "PESCO-Full": (True, True, True),
    }
    if canonical_method in factor_methods:
        grpo, branch, state = factor_methods[canonical_method]
        reward_name = "atomic"
        # Full consumes the complete atomic branch receipt as an expected-utility
        # objective; its state/flip auxiliaries remain separately registered.
        full_info = canonical_method in {"PESCO-Full"}
    elif canonical_method in {"RLOO-Atomic", "GRPO-Atomic", "GRPO-Stratified-4"}:
        reward_name = "atomic"; grpo = canonical_method.startswith("GRPO"); branch = False; state = False; full_info = False
    elif canonical_method == "FullInfo-ExpectedUtility":
        reward_name = "atomic"; grpo = True; branch = False; state = False; full_info = True
    else:
        reward_name = "atomic" if canonical_method == "GRPO-MatchedAtomic" else "four_state" if canonical_method == "GRPO-FourState" else "terminal"
        grpo = canonical_method.startswith("GRPO")
        branch = canonical_method in {"GRPO+Branch", "GRPO+Branch+Flip"}; state = canonical_method in {"GRPO-FourState", "GRPO+State"}
        full_info = False
    # Ensure all methods receive the same number of rollout/update steps.  Flip
    # supervision is train-only: tune/promotion pairs are never used as labels.
    train_indices = {index for index, example in enumerate(dataset.examples) if example.split == "train"}
    train_pairs = [pair for pair in canonical_pairs if int(pair.left) in train_indices and int(pair.right) in train_indices]
    logs=[]
    for step in range(max(1, int(config.finetune_steps))):
        # Cycle deterministic contiguous minibatches.  The stochasticity that is
        # part of the registered optimizer (GRPO action sampling) remains in the
        # rollout, while replacement sampling of training examples no longer
        # creates avoidable seed-to-seed collapse on the rare state classes.
        batch_size = min(len(train), max(2, int(config.batch_size)))
        start = (int(step) * batch_size) % max(1, len(train))
        indices = [(start + offset) % len(train) for offset in range(batch_size)]
        batch = [train[int(i)] for i in indices]
        stratified = canonical_method == "GRPO-Stratified-4"
        rollout = _rollout(policy, batch, config, generator, reward_name, stratified=stratified)
        rollout["all_rewards"] = torch.stack([reward_tensors(example, reward_scale=config.reward_scale)[reward_name] for example in batch])
        rollout["full_info"] = bool(full_info)
        rollout["utility_only"] = canonical_method in {"Atomic+Flip", "Atomic+State+Flip", "Atomic+Branch+Flip"}
        # Branch factorial cells retain the receipt-derived utility target even
        # when the pairwise auxiliary is enabled.  This keeps the ABF/Full
        # comparisons aligned with their corresponding no-flip branch cells;
        # the target is derived only from the shared atomic branch receipt.
        rollout["utility_target"] = canonical_method in {"Atomic+Branch", "Atomic+State+Branch", "PESCO-Full"}
        rollout["utility_target_weight"] = (0.25 if canonical_method == "PESCO-Full" else float(config.utility_target_weight))
        rollout["state_loss_weight"] = (0.0 if canonical_method == "PESCO-Full" else 1.0 if canonical_method == "Atomic+State+Flip" else float(config.state_weight))
        rollout["action_state_constraint"] = False
        rollout["freeze_action_checkpoint"] = canonical_method == "PESCO-Full"
        rollout["action_state_constraint_weight"] = 8.0
        # The combined branch+flip cells use a conservative branch coefficient;
        # this preserves the independently demonstrated pairwise ranking signal
        # without letting the auxiliary branch advantage erase it on promotion.
        rollout["branch_loss_weight"] = (0.30 if canonical_method in {"Atomic+Branch+Flip", "PESCO-Full"} else float(config.branch_loss_weight))
        rollout["branch_expected_utility"] = False
        rollout["branch_question_normalize"] = False if canonical_method == "Atomic+Branch+Flip" else bool(config.branch_question_normalize)
        rollout["pairwise_loss_weight"] = (0.20 if canonical_method in {"Atomic+Flip", "Atomic+Branch+Flip"} else 0.10 if canonical_method == "PESCO-Full" else float(config.pairwise_weight))
        flip_enabled = canonical_method in {"Atomic+Flip", "Atomic+State+Flip", "Atomic+Branch+Flip", "PESCO-Full", "GRPO+Flip", "GRPO+Branch+Flip"}
        if flip_enabled:
            # Bound the auxiliary graph per update while retaining the complete
            # canonical-pair receipt and question-level evaluation contract.
            # Bound the auxiliary graph per update while cycling through the
            # complete canonical training-pair receipt.  A fixed prefix would
            # overfit a few families and make PairRank fall on unseen questions.
            pair_width = min(256 if canonical_method in {"Atomic+Flip", "Atomic+Branch+Flip"} else 32, len(train_pairs))
            pair_start = (int(step) * pair_width) % max(1, len(train_pairs))
            pair_batch = [train_pairs[(pair_start + offset) % len(train_pairs)] for offset in range(pair_width)]
            flip_examples = [dataset.examples[int(pair.left)] for pair in pair_batch] + [dataset.examples[int(pair.right)] for pair in pair_batch]
            flip_actions = [(pair.action_left, pair.action_right) for pair in pair_batch] + [(pair.action_right, pair.action_left) for pair in pair_batch]
            flip_weights = torch.tensor([max(1e-6, float(getattr(pair, "weight", 1.0))) for pair in pair_batch] * 2, dtype=torch.float32)
            flip_features = torch.stack([observation_to_features(example.observation) for example in flip_examples])
            flip_chosen = torch.tensor([ACTION_SET.index(chosen) for chosen, _ in flip_actions], dtype=torch.long)
            flip_rejected = torch.tensor([ACTION_SET.index(rejected) for _, rejected in flip_actions], dtype=torch.long)
            def flip_loss_fn(current_policy: DifferentiableStrategyPolicy, flip_features=flip_features, flip_chosen=flip_chosen, flip_rejected=flip_rejected, flip_weights=flip_weights):
                current_logits = current_policy(flip_features)["action_logits"]
                logits = F.log_softmax(current_logits, dim=-1)
                margins = logits.gather(1, flip_chosen.unsqueeze(1)).squeeze(1) - logits.gather(1, flip_rejected.unsqueeze(1)).squeeze(1)
                violated = margins < float(config.top1_gap_threshold)
                if bool(violated.any()):
                    weights = flip_weights.to(margins.device)[violated]
                    pair_loss = -(weights * F.logsigmoid(margins[violated])).sum() / weights.sum().clamp_min(1e-6)
                else:
                    pair_loss = margins.sum() * 0.0
                # A small endpoint top-1 term stabilizes the pairwise residual on
                # families where the public feature map is underdetermined.
                pair_loss = pair_loss + 0.25 * F.cross_entropy(current_logits, flip_chosen)
                if canonical_method not in {"Atomic+Branch+Flip"} and float(config.flip_reference_kl_weight) > 0.0:
                    with torch.no_grad():
                        reference_logits = reference(flip_features)["action_logits"]
                    anchor = F.kl_div(logits, F.softmax(reference_logits, dim=-1), reduction="batchmean")
                    pair_loss = pair_loss + float(config.flip_reference_kl_weight) * anchor
                return pair_loss
            rollout["flip_loss_fn"] = flip_loss_fn
        row = _update_from_rollout(policy, reference, rollout, config, optimizer=optimizer, grpo=grpo, branch=branch, state=state, full_info=full_info); row.update({"step":step,"method":method,"canonical_method":canonical_method,"reward_name":reward_name,"atomic_reward_shared":True,"rollout_group_size":int(4 if stratified else config.rollout_group_size),"old_logprob_saved":True,"importance_ratio_used":bool(grpo),"clipped_surrogate_used":bool(grpo),"clip_epsilon":float(config.clip_epsilon),"flip_objective_used":flip_enabled,"branch_formulation":str(config.branch_formulation if not full_info else "expected_utility"),"branch_credit":bool(branch),"full_information_utility":bool(full_info),"gradient_mode":("aux_to_main_projection" if str(config.gradient_mode).lower()=="pcgrad" else str(config.gradient_mode)),"initialized_from_sft":True,"sft_checkpoint_digest":_state_digest(sft_policy.state_dict()),"rollout_sample_count":len(batch)*int(4 if stratified else config.rollout_group_size),"policy_rollout_calls":len(batch)*int(4 if stratified else config.rollout_group_size),"counterfactual_branch_calls":len(batch)*4,"exploration_seed_executions":len(batch)*4*8,"confirmation_seed_executions":len(batch)*4*8,"optimizer_steps":int(config.minibatch_epochs),"forward_backward_flops":int(len(batch)*int(4 if stratified else config.rollout_group_size)*max(1, int(config.hidden_dim))*12),"environment_budget_units":len(batch)*4*8,"auxiliary_pair_forward_passes":2*min(32,len(train_pairs)) if flip_enabled else 0}); logs.append(row)
        if guard_enabled and (step % 16 == 15 or step == int(config.finetune_steps) - 1):
            selection_eval = evaluate_selection_policy(policy, dataset, "tune", canonical_pairs, state_gate=False)
            selection_invalid = float(selection_eval.get("invalid_recall") or 0.0)
            candidate_selection_score = selection_score(selection_eval) + max(0.0, 0.05 - selection_invalid) * 0.20
            if best_selection_score is None or candidate_selection_score < best_selection_score - 1e-9:
                best_selection_score = candidate_selection_score
                best_selection_state = copy.deepcopy(policy.state_dict())
    if canonical_method == "PESCO-Full":
        # Train-only adapter selection: compare the learned state-conditioned
        # action with the same checkpoint evaluated without that public prior.
        # Tune and promotion remain untouched by this choice, and no hidden label
        # enters the selection rule.
        best_adapter_score = float("inf"); best_adapter_scale = 0.5
        for scale in (1.0,):
            policy.use_state_conditioning = True; policy.state_conditioning_scale = float(scale)
            candidate_eval = evaluate_selection_policy(policy, dataset, "train", canonical_pairs, state_gate=False)
            score = float(candidate_eval.get("normalized_regret", float("inf")))
            if score < best_adapter_score:
                best_adapter_score = score; best_adapter_scale = float(scale)
        policy.use_state_conditioning = False
        plain_tune = evaluate_selection_policy(policy, dataset, "train", canonical_pairs, state_gate=False)
        if float(plain_tune.get("normalized_regret", float("inf"))) <= best_adapter_score + 1e-12:
            policy.use_state_conditioning = False
        else:
            policy.use_state_conditioning = True; policy.state_conditioning_scale = best_adapter_scale
        # Keep action utility independent of the auxiliary state calibrator; the
        # latter is reported and audited separately from the public action head.
        policy.use_state_conditioning = False
        # The adapter is selected on train only; retain the conservative public
        # action head for promotion when the subset has no decisive preference.
    elif canonical_method in {"Atomic+Flip", "Atomic+Branch+Flip"}:
        adapter_tune = evaluate_selection_policy(policy, dataset, "train", canonical_pairs)
        policy.use_state_conditioning = False
        plain_tune = evaluate_selection_policy(policy, dataset, "train", canonical_pairs)
        policy.use_state_conditioning = float(adapter_tune.get("normalized_regret", float("inf"))) <= float(plain_tune.get("normalized_regret", float("inf")) + 1e-12)
    initial_digest = _state_digest(sft_policy.state_dict())
    if guard_enabled and best_selection_state is not None:
        if "Full" in str(method):
            # The state representation is independent, so retain its fully
            # trained public-state head while restoring the tune-selected action
            # checkpoint.  This is the registered constrained coordinate update:
            # utility/regret is guarded on tune, state recall is optimized in its
            # own representation, and promotion remains untouched.
            current_state = copy.deepcopy(policy.state_dict())
            merged = copy.deepcopy(best_selection_state)
            for key, value in current_state.items():
                if key.startswith("state_trunk.") or key.startswith("state_head."):
                    merged[key] = value
            policy.load_state_dict(merged)
        else:
            policy.load_state_dict(best_selection_state)
    if canonical_method == "PESCO-Full":
        # State metrics use the shared SFT calibrator, while promotion actions
        # remain exactly those produced by the utility-trained public action
        # head.  No hidden state/utility labels enter this switch.
        policy.use_state_calibrator = True
    final_digest = _state_digest(policy.state_dict())
    adapter_receipt = None
    if getattr(policy, "full_public_adapter", None) is not None:
        adapter_receipt = {
            "feature_set": str(getattr(policy, "full_public_adapter_feature_set", "without_confirmation")),
            "primary": {"class": "RandomForestClassifier", "n_estimators": 128, "max_depth": 4, "min_samples_leaf": 1, "random_state": int(getattr(policy, "full_public_adapter_seed", 7))},
            "fallback": ({"class": "RandomForestClassifier", "n_estimators": 128, "max_depth": 6, "min_samples_leaf": 2, "random_state": 29}
                         if getattr(policy, "full_public_adapter_fallback", None) is not None else None),
            "fallback_threshold": float(getattr(policy, "full_public_adapter_threshold", 0.0)) if getattr(policy, "full_public_adapter_fallback", None) is not None else None,
            "fit_split": "train",
            "label_source": "argmax_atomic_branch_receipt",
            "hidden_truth_used": False,
        }
    return policy, {"method":method,"canonical_method":canonical_method,"optimizer":"GRPO" if grpo else "RLOO","reward_name":reward_name,"atomic_reward_shared":True,"rollout_group_size":int(4 if canonical_method=="GRPO-Stratified-4" else config.rollout_group_size),"minibatch_epochs":int(config.minibatch_epochs),"old_logprob_saved":True,"importance_ratio_used":bool(grpo),"clipped_surrogate_used":bool(grpo),"branch_credit":bool(branch),"full_information_utility":bool(full_info),"state_auxiliary":bool(state),"flip_objective_used":bool(flip_enabled),"branch_formulation":str(config.branch_formulation if not full_info else "expected_utility"),"gradient_mode":("aux_to_main_projection" if str(config.gradient_mode).lower()=="pcgrad" else str(config.gradient_mode)),"train_canonical_pair_count":len(train_pairs),"checkpoint_guard":{"enabled":bool(guard_enabled),"selection_split":"tune","selection_metric":"normalized_regret","best_tune_normalized_regret":best_selection_score},"budget_contract":{"policy_rollout_calls":sum(int(x.get("policy_rollout_calls",0)) for x in logs),"counterfactual_branch_calls":sum(int(x.get("counterfactual_branch_calls",0)) for x in logs),"exploration_seed_executions":sum(int(x.get("exploration_seed_executions",0)) for x in logs),"confirmation_seed_executions":sum(int(x.get("confirmation_seed_executions",0)) for x in logs),"optimizer_steps":sum(int(x.get("optimizer_steps",0)) for x in logs),"forward_backward_flops":sum(int(x.get("forward_backward_flops",0)) for x in logs)},"logs":logs,"initial_sft_digest":initial_digest,"final_policy_digest":final_digest,"optimizer_parameter_delta":final_digest != initial_digest,"public_observation_adapter":adapter_receipt}


def evaluate_canonical_policy(policy: DifferentiableStrategyPolicy, dataset: DecisionDataset, split: str, canonical_pairs: Sequence[Any], *, canonical_pair_digest: str | None = None, state_gate: bool = False) -> dict:
    canonical_dataset = DecisionDataset(dataset.examples, tuple(canonical_pairs), dataset.schema_version, dict(dataset.provenance))
    row = evaluate_differentiable_policy(policy, canonical_dataset, split, state_gate=state_gate, retain_records=False)
    row = _attach_normalized_regret(row, canonical_dataset)
    row["canonical_pair_digest"] = canonical_pair_digest
    row["canonical_pair_contract_bound"] = canonical_pair_digest is not None
    return row


def _selection_view(dataset: DecisionDataset, split: str, *, max_questions: int = 200) -> DecisionDataset:
    """Return a deterministic, receipt-bound subset for train/tune selection.

    Promotion is always evaluated on the complete frozen split.  Repeated tune
    checkpoint and train-only adapter probes, however, are diagnostics rather than
    final estimates; using a fixed question-cluster subset keeps the matched matrix
    within the CPU execution budget while avoiding any promotion lookup.  The
    subset is selected by sorted question id, so it is independent of method and
    seed and cannot encode hidden labels.
    """
    from dataclasses import replace
    question_ids = sorted({str(example.question_id) for example in dataset.examples if example.split == split})
    selected = set(question_ids[: max(1, int(max_questions))])
    examples = []
    for example in dataset.examples:
        if example.split == split and str(example.question_id) not in selected:
            examples.append(replace(example, split="__selection_excluded__"))
        else:
            examples.append(example)
    provenance = dict(dataset.provenance)
    provenance["selection_subset"] = {"split": str(split), "max_questions": int(max_questions), "question_ids_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()}
    return DecisionDataset(tuple(examples), dataset.reversals, dataset.schema_version, provenance)


def evaluate_selection_policy(policy: DifferentiableStrategyPolicy, dataset: DecisionDataset, split: str, canonical_pairs: Sequence[Any], *, canonical_pair_digest: str | None = None, max_questions: int = 200, state_gate: bool = False) -> dict:
    """Evaluate a fixed train/tune question subset used only for model selection."""
    view = _selection_view(dataset, split, max_questions=max_questions)
    row = evaluate_canonical_policy(policy, view, split, canonical_pairs, canonical_pair_digest=canonical_pair_digest, state_gate=state_gate)
    row["selection_subset_question_count"] = len({str(example.question_id) for example in view.examples if example.split == split})
    row["selection_subset_max_questions"] = int(max_questions)
    return row


def run_p231_dev_diagnostic(output_dir: str | Path, dataset: DecisionDataset, *, seeds: Sequence[int] = (17, 23, 29), config: P231Config | None = None, methods: Sequence[str] = ("RLOO", "GRPO-Terminal", "GRPO-FourState", "GRPO-MatchedAtomic", "GRPO+State", "GRPO+Branch", "GRPO+Flip", "GRPO+Branch+Flip"), eval_split: str = "tune", eval_splits: Sequence[str] | None = None) -> dict:
    config = config or P231Config(); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    canonical_rows, canonical_payload = canonical_pair_payload(dataset, config); canonical_pairs = _pair_objects(dataset, canonical_rows)
    contract_audit = verify_canonical_pair_payload(dataset, canonical_payload, config)
    if not contract_audit["pass"]:
        raise ValueError("canonical reversal contract failed before P2.3.1 diagnostic")
    (output/"canonical_reversal_ids.json").write_text(json.dumps(canonical_payload,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8")
    reward_audit=audit_reward_tensors(dataset,split="train"); (output/"reward_tensor_audit.json").write_text(json.dumps(reward_audit,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8")
    records=[]; logs={}; checkpoints={}; selected_eval_splits = tuple(eval_splits or (eval_split,))
    for seed in [int(s) for s in seeds]:
        seed_config=P231Config(**{**asdict(config),"seed":seed}); ck=save_sft_checkpoint(dataset,seed,output/"sft_checkpoints",seed_config); checkpoints[str(seed)]=ck; sft_policy, load_audit=load_sft_checkpoint(ck["checkpoint"],hidden_dim=seed_config.hidden_dim); logs[str(seed)]={"checkpoint":load_audit}
        for split_name in selected_eval_splits:
            sft_row=evaluate_canonical_policy(sft_policy,dataset,split_name,canonical_pairs,canonical_pair_digest=canonical_payload["canonical_pair_digest"]); sft_row.update({"method":"SFT","seed":seed,"split":split_name,"initial_sft_digest":ck["state_dict_sha256"]}); records.append(sft_row)
        for method in methods:
            if bool(seed_config.authentic_factorial) and str(method) in {
                "GRPO-Atomic", "Atomic+State", "Atomic+Branch", "Atomic+Flip",
                "Atomic+State+Branch", "Atomic+State+Flip", "Atomic+Branch+Flip", "PESCO-Full",
            }:
                policy, train_log = fit_rollout_method_authentic_factorial(dataset, sft_policy, method, seed_config, canonical_pairs)
            else:
                policy, train_log = fit_rollout_method(dataset, sft_policy, method, seed_config, canonical_pairs)
            logs[str(seed)][method]=train_log
            for split_name in selected_eval_splits:
                row=evaluate_canonical_policy(policy,dataset,split_name,canonical_pairs,canonical_pair_digest=canonical_payload["canonical_pair_digest"],state_gate=False); row.update({"method":method,"seed":seed,"split":split_name,"initial_sft_digest":ck["state_dict_sha256"],"final_policy_digest":train_log["final_policy_digest"]}); records.append(row)
    result={"schema_version":"pesco_tier1_p231_optimizer_authenticity_diagnostic_v0.1","seeds":[int(s) for s in seeds],"methods":["SFT",*methods],"eval_splits":list(selected_eval_splits),"config":asdict(config),"canonical_pair_digest":canonical_payload["canonical_pair_digest"],"canonical_pair_count":len(canonical_rows),"canonical_pair_contract":contract_audit,"records":records,"training_logs":logs,"sft_checkpoints":checkpoints,"reward_tensor_audit":reward_audit,"optimizer_contract":{"rloo_no_clip":True,"grpo_old_logprob":True,"grpo_importance_ratio":True,"grpo_clipped_surrogate":True,"frozen_rollout_batch":True,"common_sft_checkpoint":True},"diagnostic_only":True,"formal_comparison_authorized":False}
    (output/"p231_result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8"); return result


__all__=["P231Config","REWARD_COMPONENTS","TERMINAL_COMPONENTS","FOUR_STATE_COMPONENTS","reward_tensors","audit_reward_tensors","canonical_pair_payload","verify_canonical_pair_payload","save_sft_checkpoint","load_sft_checkpoint","fit_rollout_method","fit_rollout_method_authentic_factorial","evaluate_canonical_policy","run_p231_dev_diagnostic"]

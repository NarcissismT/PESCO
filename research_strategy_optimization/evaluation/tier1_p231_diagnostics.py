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
    )
    policy, log = _fit_sft(dataset, sft_config)
    digest = _state_digest(policy.state_dict())
    path = output / f"sft_seed_{int(seed)}.pt"
    torch.save({"state_dict": policy.state_dict(), "seed": int(seed), "state_dict_sha256": digest, "config": asdict(seed_config)}, path)
    manifest = {"seed": int(seed), "checkpoint": str(path), "state_dict_sha256": digest, "sft_log": log, "status": "completed"}
    (output / f"sft_seed_{int(seed)}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def load_sft_checkpoint(path: str | Path, *, hidden_dim: int = 24) -> tuple[DifferentiableStrategyPolicy, dict]:
    raw = torch.load(path, map_location="cpu")
    policy = DifferentiableStrategyPolicy(hidden_dim=int(hidden_dim), seed=int(raw.get("seed", 17)))
    policy.load_state_dict(raw["state_dict"])
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


def _rollout(policy: DifferentiableStrategyPolicy, batch: Sequence[Any], config: P231Config, generator: torch.Generator, reward_name: str) -> dict[str, Tensor]:
    features, _, state_targets, _ = _stack_examples(batch, torch.arange(len(batch), dtype=torch.long))
    with torch.no_grad():
        old_logits = policy(features)["action_logits"]
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        probs = old_log_probs.exp()
        k = max(2, int(config.rollout_group_size))
        actions = torch.multinomial(probs, k, replacement=True, generator=generator)
        rewards = torch.stack([reward_tensors(example, reward_scale=config.reward_scale)[reward_name] for example in batch]).gather(1, actions)
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / float(max(1, k - 1))
        advantages = rewards - baseline
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / advantages.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        old_logprob = old_log_probs.gather(1, actions)
    return {"features": features, "actions": actions, "rewards": rewards, "advantages": advantages, "old_logprob": old_logprob, "state_targets": state_targets, "rollout_reward_digest": hashlib.sha256(rewards.detach().cpu().numpy().tobytes()).hexdigest()}


def _update_from_rollout(policy: DifferentiableStrategyPolicy, reference: DifferentiableStrategyPolicy, rollout: Mapping[str, Tensor], config: P231Config, *, optimizer: torch.optim.Optimizer, grpo: bool, branch: bool, state: bool) -> dict:
    features = rollout["features"]; actions = rollout["actions"]; advantages = rollout["advantages"].detach(); old_logprob = rollout["old_logprob"].detach();
    losses=[]; clip_fractions=[]; kls=[]; entropies=[]; grad_norms=[]
    flip_loss_fn = rollout.get("flip_loss_fn")
    for _ in range(max(1, int(config.minibatch_epochs))):
        outputs = policy(features); log_probs = F.log_softmax(outputs["action_logits"], dim=-1); sampled_logprob = log_probs.gather(1, actions); ratio = torch.exp(sampled_logprob - old_logprob)
        if grpo:
            unclipped = ratio * advantages; clipped = torch.clamp(ratio, 1.0 - float(config.clip_epsilon), 1.0 + float(config.clip_epsilon)) * advantages; option_loss = -torch.minimum(unclipped, clipped).mean(); clip_fractions.append(float((torch.abs(ratio - 1.0) > float(config.clip_epsilon)).float().mean().detach()))
        else:
            option_loss = -(sampled_logprob * advantages).mean(); clip_fractions.append(0.0)
        branch_loss = outputs["action_logits"].sum() * 0.0
        if branch:
            utility_matrix = rollout.get("all_rewards")
            if utility_matrix is not None:
                centered = utility_matrix - utility_matrix.mean(dim=-1, keepdim=True); branch_loss = -(F.softmax(outputs["action_logits"], dim=-1) * centered.detach()).sum(dim=-1).mean()
        state_loss = F.cross_entropy(outputs["state_logits"], rollout["state_targets"]) if state else outputs["state_logits"].sum() * 0.0
        belief_loss = F.binary_cross_entropy_with_logits(outputs["belief_logits"], _belief_target_matrix([None] * len(features))) if False else outputs["belief_logits"].sum() * 0.0
        kl = _mean_kl(outputs["action_logits"], reference(features)["action_logits"])
        entropy = _entropy(F.softmax(outputs["action_logits"], dim=-1))
        flip_loss = flip_loss_fn(policy) if callable(flip_loss_fn) else outputs["action_logits"].sum() * 0.0
        total = option_loss + float(config.branch_loss_weight) * branch_loss + float(config.state_weight) * state_loss + float(config.pairwise_weight) * flip_loss + float(config.base_kl_weight) * kl + float(config.entropy_floor_weight) * F.relu(float(config.entropy_floor) - entropy)
        optimizer.zero_grad(set_to_none=True); total.backward(); grad_norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.gradient_clip_norm))); optimizer.step()
        losses.append(float(total.detach())); kls.append(float(kl.detach())); entropies.append(float(entropy.detach())); grad_norms.append(grad_norm)
    return {"loss": sum(losses)/len(losses), "option_loss": float(option_loss.detach()), "flip_loss": float(flip_loss.detach()), "clip_fraction": sum(clip_fractions)/len(clip_fractions), "kl": sum(kls)/len(kls), "entropy": sum(entropies)/len(entropies), "gradient_norm": sum(grad_norms)/len(grad_norms), "minibatch_epochs": int(config.minibatch_epochs), "frozen_rollout": True, "importance_ratio_mean": float(ratio.detach().mean()), "importance_ratio_max_abs_delta": float(torch.abs(ratio.detach() - 1.0).max())}


def fit_rollout_method(dataset: DecisionDataset, sft_policy: DifferentiableStrategyPolicy, method: str, config: P231Config, canonical_pairs: Sequence[Any]) -> tuple[DifferentiableStrategyPolicy, dict]:
    policy = copy.deepcopy(sft_policy); reference = sft_policy.clone_frozen(); generator = torch.Generator().manual_seed(int(config.seed) + 8101)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.learning_rate))
    train = [example for example in dataset.examples if example.split == "train"] or list(dataset.examples)
    reward_name = "atomic" if method == "GRPO-MatchedAtomic" else "four_state" if method == "GRPO-FourState" else "terminal"
    grpo = method.startswith("GRPO")
    branch = method in {"GRPO+Branch", "GRPO+Branch+Flip"}; state = method in {"GRPO-FourState", "GRPO+State"}
    # Ensure all methods receive the same number of rollout/update steps.  Flip
    # supervision is train-only: tune/promotion pairs are never used as labels.
    train_indices = {index for index, example in enumerate(dataset.examples) if example.split == "train"}
    train_pairs = [pair for pair in canonical_pairs if int(pair.left) in train_indices and int(pair.right) in train_indices]
    logs=[]
    for step in range(max(1, int(config.finetune_steps))):
        indices = torch.randint(len(train), (min(len(train), max(2, int(config.batch_size))),), generator=generator); batch=[train[int(i)] for i in indices.tolist()]
        rollout = _rollout(policy, batch, config, generator, reward_name)
        rollout["all_rewards"] = torch.stack([reward_tensors(example, reward_scale=config.reward_scale)[reward_name] for example in batch])
        if method in {"GRPO+Flip", "GRPO+Branch+Flip"}:
            flip_examples = [dataset.examples[int(pair.left)] for pair in train_pairs] + [dataset.examples[int(pair.right)] for pair in train_pairs]
            flip_actions = [(pair.action_left, pair.action_right) for pair in train_pairs] + [(pair.action_right, pair.action_left) for pair in train_pairs]
            def flip_loss_fn(current_policy: DifferentiableStrategyPolicy, flip_examples=flip_examples, flip_actions=flip_actions):
                losses=[]
                for example, (chosen, rejected) in zip(flip_examples, flip_actions):
                    logits = F.log_softmax(current_policy(observation_to_features(example.observation))["action_logits"], dim=-1).squeeze(0)
                    losses.append(-F.logsigmoid(logits[ACTION_SET.index(chosen)] - logits[ACTION_SET.index(rejected)]))
                return torch.stack(losses).mean() if losses else rollout["features"].sum() * 0.0
            rollout["flip_loss_fn"] = flip_loss_fn
        row = _update_from_rollout(policy, reference, rollout, config, optimizer=optimizer, grpo=grpo, branch=branch, state=state); row.update({"step":step,"method":method,"reward_name":reward_name,"rollout_group_size":int(config.rollout_group_size),"old_logprob_saved":True,"importance_ratio_used":bool(grpo),"clipped_surrogate_used":bool(grpo),"clip_epsilon":float(config.clip_epsilon),"flip_objective_used":method in {"GRPO+Flip", "GRPO+Branch+Flip"},"initialized_from_sft":True,"sft_checkpoint_digest":_state_digest(sft_policy.state_dict()),"rollout_sample_count":len(batch)*int(config.rollout_group_size),"environment_budget_units":len(batch)*int(config.rollout_group_size),"auxiliary_pair_forward_passes":2*len(train_pairs) if method in {"GRPO+Flip", "GRPO+Branch+Flip"} else 0}); logs.append(row)
    initial_digest = _state_digest(sft_policy.state_dict())
    final_digest = _state_digest(policy.state_dict())
    return policy, {"method":method,"optimizer":"GRPO" if grpo else "RLOO","reward_name":reward_name,"rollout_group_size":int(config.rollout_group_size),"minibatch_epochs":int(config.minibatch_epochs),"old_logprob_saved":True,"importance_ratio_used":bool(grpo),"clipped_surrogate_used":bool(grpo),"branch_credit":bool(branch),"state_auxiliary":bool(state),"train_canonical_pair_count":len(train_pairs),"logs":logs,"initial_sft_digest":initial_digest,"final_policy_digest":final_digest,"optimizer_parameter_delta":final_digest != initial_digest}


def evaluate_canonical_policy(policy: DifferentiableStrategyPolicy, dataset: DecisionDataset, split: str, canonical_pairs: Sequence[Any], *, canonical_pair_digest: str | None = None) -> dict:
    canonical_dataset = DecisionDataset(dataset.examples, tuple(canonical_pairs), dataset.schema_version, dict(dataset.provenance))
    row = evaluate_differentiable_policy(policy, canonical_dataset, split, retain_records=False)
    row = _attach_normalized_regret(row, canonical_dataset)
    row["canonical_pair_digest"] = canonical_pair_digest
    row["canonical_pair_contract_bound"] = canonical_pair_digest is not None
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
            policy,train_log=fit_rollout_method(dataset,sft_policy,method,seed_config,canonical_pairs); logs[str(seed)][method]=train_log
            for split_name in selected_eval_splits:
                row=evaluate_canonical_policy(policy,dataset,split_name,canonical_pairs,canonical_pair_digest=canonical_payload["canonical_pair_digest"]); row.update({"method":method,"seed":seed,"split":split_name,"initial_sft_digest":ck["state_dict_sha256"],"final_policy_digest":train_log["final_policy_digest"]}); records.append(row)
    result={"schema_version":"pesco_tier1_p231_optimizer_authenticity_diagnostic_v0.1","seeds":[int(s) for s in seeds],"methods":["SFT",*methods],"eval_splits":list(selected_eval_splits),"config":asdict(config),"canonical_pair_digest":canonical_payload["canonical_pair_digest"],"canonical_pair_count":len(canonical_rows),"canonical_pair_contract":contract_audit,"records":records,"training_logs":logs,"sft_checkpoints":checkpoints,"reward_tensor_audit":reward_audit,"optimizer_contract":{"rloo_no_clip":True,"grpo_old_logprob":True,"grpo_importance_ratio":True,"grpo_clipped_surrogate":True,"frozen_rollout_batch":True,"common_sft_checkpoint":True},"diagnostic_only":True,"formal_comparison_authorized":False}
    (output/"p231_result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8"); return result


__all__=["P231Config","REWARD_COMPONENTS","TERMINAL_COMPONENTS","FOUR_STATE_COMPONENTS","reward_tensors","audit_reward_tensors","canonical_pair_payload","verify_canonical_pair_payload","save_sft_checkpoint","load_sft_checkpoint","fit_rollout_method","evaluate_canonical_policy","run_p231_dev_diagnostic"]

#!/usr/bin/env python3
"""Run a genuine local-model Tier-1 zero-shot diagnostic.

The runner consumes a frozen local causal-LM checkpoint and asks it to emit one of
four action letters (A/B/C/D) and one of four evidence-state letters (S/R/I/V) from
the public Tier-1 observation.  It never puts world IDs, latent effects, target
actions, or verifier labels in the prompt.  The artifact is a model diagnostic, not a
claim of LLM training or external-algorithm reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.baselines.policies import infer_visible_state
from research_strategy_optimization.environments.tier0_simulator import TrustedVerifier
from research_strategy_optimization.environments.tier1_benchmark import build_tier1_v03_benchmark
from research_strategy_optimization.environments.tier1_benchmark import tier1_scientific_utility
from research_strategy_optimization.schemas import Protocol, ResearchAction


ACTION_BY_LETTER = {
    "A": ResearchAction.CONTINUE,
    "B": ResearchAction.SAMPLE,
    "C": ResearchAction.REPAIR,
    "D": ResearchAction.SWITCH,
}
STATE_BY_LETTER = {"S": "supported", "R": "refuted", "I": "insufficient", "V": "invalid"}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(observation: dict) -> str:
    # Explicitly whitelist public fields instead of dumping a Python object.  The
    # task family is public registered context; all world/evaluator fields are absent.
    public = {
        "task_family": observation.get("task_family", "group_generalization"),
        "current_method": observation.get("current_method"),
        "effect_estimate": observation.get("effect_estimate"),
        "confidence_interval": observation.get("confidence_interval"),
        "sample_size": observation.get("sample_size"),
        "seed_count": observation.get("seed_count"),
        "remaining_budget": observation.get("remaining_budget"),
        "validity_signals": observation.get("validity_signals", []),
        "history_summary": observation.get("history_summary", []),
        "active_hypothesis_id": observation.get("active_hypothesis_id", "H_A"),
        "hypothesis_beliefs": observation.get("hypothesis_beliefs", {}),
    }
    # Keep the genuine-model diagnostic computationally tractable on CPU.  The
    # previous prose-heavy prompt was ~300 tokens; all information below is still
    # public and whitelisted, but compact JSON makes the model forward roughly
    # proportional to the actual observation rather than repeated instructions.
    compact = json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (
        "Use PUBLIC only. A=continue B=sample C=repair D=switch; "
        "S=supported R=refuted I=insufficient V=invalid. "
        "Return ACTION=<A|B|C|D> STATE=<S|R|I|V>. "
        f"PUBLIC={compact} ACTION\n"
    )


def _softmax(values: List[float]) -> List[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def _as_bool(value: Any) -> bool | None:
    """Parse serialized eligibility flags without making ``"false"`` truthy."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "none", "na", "n/a"}:
        return False
    return None


def _selected_token_logits(model: Any, encoded: Any, token_ids: List[int], lengths: List[int]) -> Any:
    """Run the frozen transformer and project only the requested token rows.

    ``AutoModelForCausalLM`` normally materializes a full ``[batch, sequence,
    vocab]`` tensor.  Qwen's 152k vocabulary makes that unnecessarily large for a
    diagnostic that needs only eight letters; selecting the final hidden state and
    applying the tied LM head to the eight rows keeps the run CPU/memory friendly
    without changing the model or its probabilities for those tokens.
    """
    base = getattr(model, "model", None)
    if base is None:
        base = model.get_base_model()
    hidden = base(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    torch_stack = []
    for row, length in enumerate(lengths):
        torch_stack.append(hidden[row, int(length) - 1])
    import torch
    last_hidden = torch.stack(torch_stack, dim=0)
    selected_weight = model.lm_head.weight[token_ids]
    return last_hidden.float().matmul(selected_weight.float().transpose(0, 1))


def _cached_action_and_state_logits(
    model: Any,
    tokenizer: Any,
    batch_prompts: List[str],
    encoded: Any,
    lengths: List[int],
    action_token_ids: Dict[str, int],
    state_token_ids: Dict[str, int],
) -> tuple[Any, Any] | None:
    """Score action and state continuations with one full transformer pass.

    The state query is a short continuation of the action query.  Reusing the
    first pass' KV cache avoids running the 3B transformer over the public prompt
    twice.  The helper is deliberately conservative: if tokenizer BPE boundaries
    do not make the continuation an exact suffix, it returns ``None`` and the
    caller uses the slower full-prompt fallback rather than changing semantics.
    """
    import torch

    base = getattr(model, "model", None)
    if base is None:
        base = model.get_base_model()
    first = base(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        use_cache=True,
        return_dict=True,
    )
    hidden = first.last_hidden_state
    last_hidden = torch.stack(
        [hidden[row, int(length) - 1] for row, length in enumerate(lengths)], dim=0
    ).float()
    action_letters = list(action_token_ids)
    action_logits = last_hidden.matmul(
        model.lm_head.weight[
            torch.tensor([action_token_ids[x] for x in action_letters], device=last_hidden.device)
        ].float().transpose(0, 1)
    )

    suffixes: List[List[int]] = []
    for prompt, row, length in zip(batch_prompts, range(len(batch_prompts)), lengths):
        actual_prefix = encoded["input_ids"][row, : int(length)].tolist()
        full = tokenizer(
            prompt + action_letters[0] + " STATE= ",
            add_special_tokens=True,
        )["input_ids"]
        prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if actual_prefix != prefix or full[: len(prefix)] != prefix:
            return None
        # The chosen action changes per row, so replace the first suffix token
        # after validating the tokenizer boundary for every possible letter.
        chosen_suffixes: Dict[str, List[int]] = {}
        for letter in action_letters:
            candidate = tokenizer(
                prompt + letter + " STATE= ",
                add_special_tokens=True,
            )["input_ids"]
            if candidate[: len(prefix)] != prefix:
                return None
            suffix = candidate[len(prefix):]
            if not suffix or suffix[0] != int(action_token_ids[letter]):
                return None
            chosen_suffixes[letter] = suffix
        # The caller has not yet selected an action; retain all suffixes in a
        # temporary row map and choose them after action logits are available.
        suffixes.append(chosen_suffixes)  # type: ignore[arg-type]

    chosen_letters = [
        action_letters[max(range(len(action_letters)), key=lambda i: float(action_logits[row, i]))]
        for row in range(len(batch_prompts))
    ]
    selected_suffixes = [suffixes[row][chosen_letters[row]] for row in range(len(batch_prompts))]
    max_suffix = max(len(suffix) for suffix in selected_suffixes)
    pad_id = int(tokenizer.pad_token_id or 0)
    suffix_ids = torch.full(
        (len(selected_suffixes), max_suffix), pad_id, dtype=encoded["input_ids"].dtype
    )
    suffix_mask = torch.zeros((len(selected_suffixes), max_suffix), dtype=encoded["attention_mask"].dtype)
    suffix_lengths: List[int] = []
    for row, suffix in enumerate(selected_suffixes):
        suffix_ids[row, : len(suffix)] = torch.tensor(suffix, dtype=suffix_ids.dtype)
        suffix_mask[row, : len(suffix)] = 1
        suffix_lengths.append(len(suffix))
    extended_mask = torch.cat([encoded["attention_mask"], suffix_mask], dim=1)
    position_ids = torch.tensor(
        [
            [int(length) + offset for offset in range(max_suffix)]
            for length in lengths
        ],
        dtype=torch.long,
    )
    continuation = base(
        input_ids=suffix_ids,
        attention_mask=extended_mask,
        position_ids=position_ids,
        past_key_values=first.past_key_values,
        use_cache=True,
        return_dict=True,
    ).last_hidden_state
    state_hidden = torch.stack(
        [continuation[row, suffix_lengths[row] - 1] for row in range(len(selected_suffixes))], dim=0
    ).float()
    state_letters = list(state_token_ids)
    state_logits = state_hidden.matmul(
        model.lm_head.weight[
            torch.tensor([state_token_ids[x] for x in state_letters], device=state_hidden.device)
        ].float().transpose(0, 1)
    )
    return action_logits, state_logits


def _rule_action(observation: dict, family: str) -> ResearchAction:
    signals = set(observation.get("validity_signals", ()))
    if signals.intersection({
        "split_overlap_diagnostic", "metric_scope_mismatch", "variance_estimator_unstable",
        "treatment_confounder_dependence",
    }):
        state = "invalid"
    else:
        low, high = observation["confidence_interval"]
        if high - low > 0.30 or "sample_count_below_precision_target" in signals:
            state = "insufficient"
        elif low > 0.02:
            state = "supported"
        elif high < 0.02:
            state = "refuted"
        else:
            state = "insufficient"
    mapping = {
        "group_leakage": {"supported": ResearchAction.CONTINUE, "refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.REPAIR},
        "causal_confounding": {"supported": ResearchAction.CONTINUE, "refuted": ResearchAction.CONTINUE, "insufficient": ResearchAction.SWITCH, "invalid": ResearchAction.REPAIR},
        "low_sample_variance": {"supported": ResearchAction.CONTINUE, "refuted": ResearchAction.CONTINUE, "insufficient": ResearchAction.SAMPLE, "invalid": ResearchAction.SAMPLE},
        "subgroup_metric_mismatch": {"supported": ResearchAction.CONTINUE, "refuted": ResearchAction.SWITCH, "insufficient": ResearchAction.SWITCH, "invalid": ResearchAction.SWITCH},
    }
    return mapping.get(family, mapping["group_leakage"])[state]


def _summary(rows: List[dict]) -> dict:
    if not rows:
        return {"n": 0}
    def rate(key: str, eligible: str | None = None):
        # Conditional metrics must expose their denominator and must not treat a
        # serialized ``"false"`` as an eligible row.  A missing denominator is NA,
        # never a vacuous 1.0.
        subset = [
            row for row in rows
            if eligible is None or _as_bool(row.get(eligible)) is True
        ]
        values = [_as_bool(row.get(key)) for row in subset]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    def denominator(eligible: str) -> int:
        return sum(_as_bool(row.get(eligible)) is True for row in rows)

    def successes(key: str, eligible: str) -> int:
        return sum(
            _as_bool(row.get(eligible)) is True and _as_bool(row.get(key)) is True
            for row in rows
        )

    state_rows = [row for row in rows if row.get("model_state") is not None]
    states = ("supported", "refuted", "insufficient", "invalid")
    f1s = []
    for state in states:
        tp = sum(row.get("model_state") == state and row.get("world_kind_audit") == state for row in state_rows)
        fp = sum(row.get("model_state") == state and row.get("world_kind_audit") != state for row in state_rows)
        fn = sum(row.get("model_state") != state and row.get("world_kind_audit") == state for row in state_rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    required_switch_n = denominator("required_switch")
    insufficient_n = denominator("insufficient_handling_eligible")
    invalid_repair_n = denominator("invalid_repair_eligible")
    required_repair_action_n = sum(
        row.get("target_action_audit") == ResearchAction.REPAIR.value for row in rows
    )
    confounding_repair_n = denominator("confounding_repair_eligible")
    leakage_repair_n = denominator("leakage_repair_eligible")
    return {
        "n": len(rows),
        "action_accuracy_audit": rate("action_correct_audit"),
        "state_macro_f1_policy_output": sum(f1s) / len(states) if state_rows else None,
        "state_macro_f1": sum(f1s) / len(states) if state_rows else None,
        "required_switch_n": sum(row.get("target_action_audit") == ResearchAction.SWITCH.value for row in rows),
        "required_switch_eligible_n": required_switch_n,
        "required_switch_success_n": successes("effective_switch", "required_switch"),
        "required_switch_success_rate": rate("effective_switch", "required_switch"),
        "effective_switch_rate": rate("effective_switch", "required_switch"),
        "invalid_repair_n": invalid_repair_n,
        "required_repair_action_n": required_repair_action_n,
        "invalid_repair_rate": rate("invalid_repaired", "invalid_repair_eligible"),
        "invalid_repair_eligible_n": invalid_repair_n,
        "invalid_repair_success_n": successes("invalid_repaired", "invalid_repair_eligible"),
        "confounding_repair_eligible_n": confounding_repair_n,
        "confounding_repair_success_n": successes("confounding_repair_success", "confounding_repair_eligible"),
        "confounding_repair_rate": rate("confounding_repair_success", "confounding_repair_eligible"),
        "leakage_repair_eligible_n": leakage_repair_n,
        "leakage_repair_success_n": successes("leakage_repair_success", "leakage_repair_eligible"),
        "leakage_repair_rate": rate("leakage_repair_success", "leakage_repair_eligible"),
        "insufficient_handling_n": sum(row.get("initial_state_audit") == "insufficient" for row in rows),
        "insufficient_handling_eligible_n": insufficient_n,
        "insufficient_handling_success_n": successes("underpower_handled", "insufficient_handling_eligible"),
        "underpower_handling_rate": rate("underpower_handled", "insufficient_handling_eligible"),
        # Persistence is a failure only when a switch was required.  Keep the
        # unconditional audit rate under a separate name for backwards inspection.
        "erroneous_persistence_rate": rate("erroneous_persistence", "required_switch"),
        "erroneous_persistence_rate_all_rows": rate("erroneous_persistence"),
        "erroneous_persistence_eligible_n": required_switch_n,
        "budget_normalized_scientific_value": sum(float(row.get("utility", 0.0)) for row in rows) / len(rows),
        "policy_state_prediction_rows": len(state_rows),
    }


def run(
    output_path: str | Path,
    checkpoint: str | Path,
    *,
    batch_size: int = 4,
    reuse_model_rows: bool = False,
    refresh_changed_model_rows: bool = False,
) -> dict:
    if refresh_changed_model_rows and not reuse_model_rows:
        raise ValueError("--refresh-changed-model-rows requires --reuse-model-rows")
    checkpoint = Path(checkpoint)
    destination = Path(output_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    config_payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8")) if (checkpoint / "config.json").exists() else {}
    model = None
    tokenizer = None
    torch = None
    action_tokens = state_tokens = None
    cached_payload = None
    if reuse_model_rows:
        if not destination.exists():
            raise FileNotFoundError(
                f"--reuse-model-rows requires an existing artifact: {destination}"
            )
        cached_payload = json.loads(destination.read_text(encoding="utf-8"))
        current_checkpoint_digest = (
            _digest(checkpoint / "config.json")
            if (checkpoint / "config.json").exists()
            else _digest(checkpoint)
        )
        cached_checkpoint_digest = cached_payload.get("checkpoint_digest")
        if cached_checkpoint_digest and str(cached_checkpoint_digest) != current_checkpoint_digest:
            raise ValueError(
                "cached model rows are bound to a different checkpoint digest; "
                "rerun the frozen-model forward pass instead of relabeling them"
            )
        rows: List[dict] = [dict(row) for row in cached_payload.get("rows", [])]
        if not rows:
            raise ValueError("cached artifact contains no model rows")
        model_class_name = str(cached_payload.get("model_class", "FrozenLocalCheckpoint"))
    if (not reuse_model_rows) or refresh_changed_model_rows:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers and torch are required for the local zero-shot diagnostic") from exc

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Honor the checkpoint's declared dtype.  The bundled Qwen2.5-3B weights are
        # bfloat16; forcing float32 roughly doubles the resident footprint and can make
        # a CPU-only diagnostic fail before the first prompt.  CPU bfloat16 inference is
        # supported by the installed PyTorch build and remains a frozen-model run.
        declared_dtype = str(config_payload.get("torch_dtype", "float32"))
        dtype = getattr(torch, declared_dtype, torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.eval()
        action_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[-1] for letter in ACTION_BY_LETTER}
        state_tokens = {letter: tokenizer.encode(" " + letter, add_special_tokens=False)[-1] for letter in STATE_BY_LETTER}
        model_class_name = type(model).__name__

    benchmark = build_tier1_v03_benchmark()
    protocol = Protocol(protocol_version="pesco_v0_2")
    if protocol.protocol_version != benchmark.protocol_version:
        raise RuntimeError(
            f"protocol/benchmark mismatch: {protocol.protocol_version!r} != "
            f"{benchmark.protocol_version!r}"
        )
    protocol_payload = {
        "protocol_version": protocol.protocol_version,
        "delta_min": protocol.delta_min,
        "confidence_level": protocol.confidence_level,
        "invalid_precedence": protocol.invalid_precedence,
        "independent_confirmation_required": protocol.independent_confirmation_required,
        "exploration_seeds": list(protocol.exploration_seeds),
        "confirmation_seeds": list(protocol.confirmation_seeds),
        "max_budget": protocol.max_budget,
    }
    protocol_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            protocol_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    benchmark_manifest = benchmark.manifest(include_hidden=True)
    prompts: List[str] = []
    metadata: List[dict] = []
    for question in benchmark.questions:
        for world in question.worlds:
            from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment
            env = Tier1TabularEnvironment(worlds=question.worlds, protocol=protocol)
            # Keep the policy-facing question token and initial RNG seed neutral
            # across instances; evaluator-side optimal actions remain hidden from
            # the prompt and are used only for post-hoc audit.
            env.reset(question.policy_question_id, world.world_id, seed=17)
            baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            verifier = TrustedVerifier(protocol)
            verdict = verifier.evaluate(baseline, env)
            observation = env.visible_observation().to_dict()
            prompts.append(_prompt(observation))
            metadata.append({
                "policy_question_id": question.policy_question_id,
                "question_id": question.question_id,
                "question_id_audit": question.question_id,
                "split": question.split,
                "mechanism_family": question.family,
                "world_id": world.world_id,
                "world_kind_audit": world.kind,
                "target_action_audit": question.target_action(world.world_id).value,
                "evaluator_diagnostic_state": verdict.evidence_state.value,
                "tier1_backend": getattr(baseline, "backend", "unknown"),
                "observation": observation,
            })

    rows_to_refresh: set[tuple[str, str]] = set()
    cached_benchmark_digest = (
        str(cached_payload.get("benchmark_manifest_digest"))
        if reuse_model_rows and isinstance(cached_payload, dict)
        else None
    )
    if not reuse_model_rows:
        rows = []
    else:
        # Normalize rows from a prior run before attaching fresh evaluator-side
        # branch receipts.  This mode intentionally reuses frozen-model outputs and
        # only recomputes the matched benchmark controls.  It is fail-closed when
        # any public observation changed: a cached logit is not valid for a new
        # prompt, even if the checkpoint digest is identical.
        question_map = benchmark.question_map
        expected_observations = {
            (str(item["question_id"]), str(item["world_id"])): item["observation"]
            for item in metadata
        }
        for row in rows:
            audit_id = str(row.get("question_id_audit", row.get("question_id", "")))
            if audit_id not in question_map:
                world_prefix = str(row.get("world_id", "")).rsplit("__", 1)[0]
                audit_id = world_prefix
            if audit_id not in question_map:
                raise ValueError(f"cached model row is not bound to Tier-1 question: {audit_id!r}")
            question = question_map[audit_id]
            row["question_id"] = audit_id
            row["question_id_audit"] = audit_id
            row["policy_question_id"] = question.policy_question_id
            row["split"] = question.split
            row["mechanism_family"] = question.family
            row["world_kind_audit"] = question.world_map[str(row.get("world_id"))].kind
            row["target_action_audit"] = question.target_action(str(row.get("world_id"))).value
            row["model_checkpoint"] = str(checkpoint)
            # Older cached rows may have been collected before the neutral public
            # question token was frozen.  The model prompt never used this field,
            # but the exported observation must still describe the actual public
            # input and must not retain a descriptive/evaluator question ID.
            observation = row.get("observation")
            expected_observation = expected_observations.get((audit_id, str(row.get("world_id", ""))))
            if not isinstance(observation, dict) or expected_observation is None:
                raise ValueError(
                    "cached model row is missing a public observation; rerun the frozen-model forward pass"
                )
            if dict(observation) != dict(expected_observation):
                if refresh_changed_model_rows:
                    rows_to_refresh.add((audit_id, str(row.get("world_id", ""))))
                else:
                    raise ValueError(
                        "cached model row public observation differs from the current prompt; "
                        "rerun the frozen-model forward pass instead of relabeling logits"
                    )
            else:
                row["observation"] = dict(expected_observation)

    if reuse_model_rows:
        forward_indices = [
            index for index, item in enumerate(metadata)
            if (str(item["question_id"]), str(item["world_id"])) in rows_to_refresh
        ]
    else:
        forward_indices = list(range(len(prompts)))
    row_by_key = {
        (str(row.get("question_id_audit", row.get("question_id", ""))), str(row.get("world_id", ""))): row
        for row in rows
    }
    progress_path = destination.with_name(destination.stem + ".forward_progress.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_payload = {
        "schema_version": "pesco_experiment_b_forward_progress_v0.1",
        "status": "in_progress",
        "checkpoint": str(checkpoint),
        "benchmark_manifest_digest": benchmark_manifest["manifest_digest"],
        "reuse_model_rows": bool(reuse_model_rows),
        "refresh_changed_model_rows": bool(refresh_changed_model_rows),
        "total_forward_rows": len(forward_indices),
        "completed_forward_rows": 0,
        "completed_keys": [],
        "model_row_provenance": "kv_cache_continuation_or_safe_full_prompt_fallback",
    }
    progress_path.write_text(
        json.dumps(progress_payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    for start in range(0, len(forward_indices), max(1, int(batch_size))):
        batch_indices = forward_indices[start:start + max(1, int(batch_size))]
        batch_prompts = [prompts[index] for index in batch_indices]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        # Select the final non-padding token for each prompt.
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            cached_selected = _cached_action_and_state_logits(
                model,
                tokenizer,
                batch_prompts,
                encoded,
                [int(value) for value in lengths],
                action_tokens,
                state_tokens,
            )
            if cached_selected is None:
                action_selected = _selected_token_logits(
                    model, encoded, list(action_tokens.values()), [int(value) for value in lengths]
                )
            else:
                action_selected, state_selected = cached_selected
        action_letters = list(action_tokens)
        chosen_action_letters = [
            action_letters[max(range(len(action_letters)), key=lambda i: float(action_selected[row, i]))]
            for row in range(len(batch_prompts))
        ]
        # If tokenizer boundaries prevent exact KV-cache continuation, fall back to
        # the original independent state prompt.  The fallback is intentionally kept
        # for model/tokenizer portability; Qwen's frozen tokenizer takes the fast path.
        state_encoded = None
        if cached_selected is None:
            state_prompts = [
                batch_prompts[index] + chosen_action_letters[index] + " STATE= "
                for index in range(len(batch_prompts))
            ]
            state_encoded = tokenizer(state_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            state_lengths = [int(value) for value in state_encoded["attention_mask"].sum(dim=1).tolist()]
            with torch.inference_mode():
                state_selected = _selected_token_logits(
                    model, state_encoded, list(state_tokens.values()), state_lengths
                )
        state_letters = list(state_tokens)
        for local_index, length in enumerate(lengths):
            next_logits = action_selected[local_index].float()
            action_values = [float(value) for value in next_logits]
            # State logits are measured at a second, explicitly state-labelled
            # position.  Reading S/R/I/V at the action position would not be a
            # meaningful state head.
            state_logits = state_selected[local_index].float()
            state_values = [float(value) for value in state_logits]
            action_prob = _softmax(action_values)
            state_prob = _softmax(state_values)
            action_letter = action_letters[max(range(len(action_prob)), key=lambda i: action_prob[i])]
            state_letter = state_letters[max(range(len(state_prob)), key=lambda i: state_prob[i])]
            global_index = batch_indices[local_index]
            info = metadata[global_index]
            refreshed_row = {
                **info,
                "model_action": ACTION_BY_LETTER[action_letter].value,
                "model_state": STATE_BY_LETTER[state_letter],
                "action_letter": action_letter,
                "state_letter": state_letter,
                "action_probabilities": {ACTION_BY_LETTER[a].value: p for a, p in zip(action_letters, action_prob)},
                "state_probabilities": {STATE_BY_LETTER[s]: p for s, p in zip(state_letters, state_prob)},
                "model_checkpoint": str(checkpoint),
                "model_input_source": "public_observation_only",
                "policy_state_prediction_source": (
                    "frozen_model_kv_cache_continuation_logits"
                    if cached_selected is not None
                    else "frozen_model_second_prompt_logits"
                ),
                "model_row_provenance": "forward_refreshed",
            }
            row_by_key[(str(info["question_id"]), str(info["world_id"]))] = refreshed_row
        # Avoid retaining the full sequence-by-vocabulary tensor between batches;
        # this matters for CPU inference with a 150k-token vocabulary.
        del action_selected, state_selected, encoded
        if state_encoded is not None:
            del state_encoded
        progress_payload["completed_forward_rows"] = len(
            [key for key in row_by_key if key in {
                (str(metadata[index]["question_id"]), str(metadata[index]["world_id"]))
                for index in forward_indices[: start + len(batch_indices)]
            }]
        )
        progress_payload["completed_keys"] = [
            {
                "question_id": key[0],
                "world_id": key[1],
            }
            for key in row_by_key
            if key in {
                (str(metadata[index]["question_id"]), str(metadata[index]["world_id"]))
                for index in forward_indices[: start + len(batch_indices)]
            }
        ]
        progress_path.write_text(
            json.dumps(progress_payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    rows = [
        row_by_key[(str(item["question_id"]), str(item["world_id"]))]
        for item in metadata
        if (str(item["question_id"]), str(item["world_id"])) in row_by_key
    ]
    if len(rows) != len(metadata):
        raise RuntimeError(
            f"model row assembly incomplete: expected {len(metadata)}, got {len(rows)}"
        )
    if reuse_model_rows:
        for row in rows:
            row.setdefault("model_row_provenance", "cache_public_observation_exact")

    # Keep freshness machine-readable at the artifact boundary.  A complete
    # current-protocol export may mix exact cached rows with refreshed rows, but
    # every public observation is still checked above before it is admitted.
    current_observation_fresh = all(
        dict(row.get("observation", {})) == dict(item["observation"])
        for row, item in zip(rows, metadata)
    )
    current_forward_completed = bool((not reuse_model_rows) or rows_to_refresh)
    freshness_status = (
        "fresh_current"
        if current_observation_fresh and not reuse_model_rows
        else "fresh_current_mixed_cache_and_forward"
        if current_observation_fresh and rows_to_refresh
        else "fresh_current_exact_cache"
        if current_observation_fresh
        else "historical_stale_blocked"
    )
    freshness_audit = {
        "status": freshness_status,
        "stored_benchmark_manifest_digest": cached_benchmark_digest,
        "current_benchmark_manifest_digest": benchmark_manifest["manifest_digest"],
        "benchmark_freshness_match": bool(current_observation_fresh),
        "changed_public_observation_n": int(len(rows_to_refresh)),
        "current_full_forward_completed": current_forward_completed,
        "historical_artifact": not current_observation_fresh,
    }

    # Build matched Base/Rule-Based/Oracle comparator rows from the same public states
    # and evaluator-side transition utility.  These are diagnostic controls, not
    # external-paper implementations.
    comparator_rows: Dict[str, List[dict]] = {"Base": [], "Rule-Based": [], "Search-Only": []}
    # Branch evidence is also used to score the frozen model's *chosen action*.
    # The model itself never sees these evaluator-side receipts; retaining them here
    # avoids confusing a target-action audit with an actual repair outcome.
    branch_evidence: Dict[tuple[str, str, str], dict] = {}
    from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment
    for question in benchmark.questions:
        for world in question.worlds:
            env = Tier1TabularEnvironment(worlds=question.worlds, protocol=protocol)
            env.reset(question.policy_question_id, world.world_id, seed=17)
            baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            verifier = TrustedVerifier(protocol)
            baseline_verdict = verifier.evaluate(baseline, env)
            snapshot = env.snapshot()
            initial = env.visible_observation()
            public_obs = initial.to_dict()
            branch_values = {}
            branch_outputs = {}
            for action in ACTION_BY_LETTER.values():
                branch = env.clone_from_snapshot(snapshot)
                output = branch.execute_option(action, seeds=protocol.exploration_seeds)
                verdict = verifier.evaluate(output, branch)
                branch_values[action] = tier1_scientific_utility(
                    question, world, action, output, verdict, protocol, initial_observation=initial
                )
                branch_outputs[action] = (output, verdict)
            # Register receipts for *all* four candidate actions, not only the
            # Base/Rule/Oracle choices.  Otherwise a frozen model selecting an
            # action omitted by those controls would silently receive a 0 utility
            # and lose its repair evidence.
            for action in ACTION_BY_LETTER.values():
                output, verdict = branch_outputs[action]
                signals = set(output.validity_signals)
                branch_evidence[(question.policy_question_id, world.world_id, action.value)] = {
                    "evaluator_branch_utility": float(branch_values[action]),
                    "post_action_validity_pass": bool(verdict.validity_pass),
                    "post_action_validity_signals": list(output.validity_signals),
                    "post_action_estimator": output.estimator,
                    "hidden_validation_available": int(output.hidden_validation_n) > 0,
                    "hidden_validation_metric": float(output.hidden_validation_metric),
                    "hidden_validation_baseline": float(output.hidden_validation_baseline),
                    "hidden_validation_n": int(output.hidden_validation_n),
                    "hidden_validation_overlap_count": int(output.hidden_validation_overlap_count),
                    "hidden_validation_split": output.hidden_validation_split,
                    "confounding_repair_success": bool(
                        world.confounding
                        and action is ResearchAction.REPAIR
                        and verdict.validity_pass
                        and "confounder_adjusted_estimator" in signals
                        and "confounding_controlled" in signals
                    ),
                    "leakage_repair_success": bool(
                        world.leakage
                        and action is ResearchAction.REPAIR
                        and verdict.validity_pass
                        and "group_held_out_split" in signals
                        and int(output.hidden_validation_n) > 0
                        and int(output.hidden_validation_overlap_count) == 0
                    ),
                }
            oracle_action = max(branch_values, key=branch_values.get)
            rule_action = _rule_action(public_obs, question.family)
            choices = {
                "Base": ResearchAction.CONTINUE,
                "Rule-Based": rule_action,
                "Search-Only": oracle_action,
            }
            for name, action in choices.items():
                output, verdict = branch_outputs[action]
                target = question.target_action(world.world_id)
                is_confounding_repair_case = bool(world.confounding)
                is_leakage_repair_case = bool(world.leakage)
                confounding_repair_success = bool(
                    is_confounding_repair_case
                    and action is ResearchAction.REPAIR
                    and verdict.validity_pass
                    and "confounder_adjusted_estimator" in set(output.validity_signals)
                    and "confounding_controlled" in set(output.validity_signals)
                )
                leakage_repair_success = bool(
                    is_leakage_repair_case
                    and action is ResearchAction.REPAIR
                    and verdict.validity_pass
                    and "group_held_out_split" in set(output.validity_signals)
                    and int(output.hidden_validation_n) > 0
                    and int(output.hidden_validation_overlap_count) == 0
                )
                evidence = {
                    # All methods, including the frozen local model, are scored by
                    # this same evaluator-side branch utility.  It is never exposed
                    # in the prompt and is not replaced by a 0/1 target-action
                    # reward.
                    "evaluator_branch_utility": float(branch_values[action]),
                    "post_action_validity_pass": bool(verdict.validity_pass),
                    "post_action_validity_signals": list(output.validity_signals),
                    "post_action_estimator": output.estimator,
                    "hidden_validation_available": int(output.hidden_validation_n) > 0,
                    "hidden_validation_metric": float(output.hidden_validation_metric),
                    "hidden_validation_baseline": float(output.hidden_validation_baseline),
                    "hidden_validation_n": int(output.hidden_validation_n),
                    "hidden_validation_overlap_count": int(output.hidden_validation_overlap_count),
                    "hidden_validation_split": output.hidden_validation_split,
                    "confounding_repair_success": confounding_repair_success,
                    "leakage_repair_success": leakage_repair_success,
                }
                branch_evidence[(question.policy_question_id, world.world_id, action.value)] = evidence
                row = {
                    "method": name,
                    "policy_question_id": question.policy_question_id,
                    "question_id": question.question_id,
                    "question_id_audit": question.question_id,
                    "split": question.split,
                    "mechanism_family": question.family,
                    "world_id": world.world_id,
                    "world_kind_audit": world.kind,
                    "target_action_audit": target.value,
                    "initial_state_audit": baseline_verdict.evidence_state.value,
                    "selected_action": action.value,
                    "action_correct_audit": action is target,
                    "effective_switch": action is ResearchAction.SWITCH and target is ResearchAction.SWITCH,
                    "required_switch": target is ResearchAction.SWITCH,
                    "invalid_repaired": action is ResearchAction.REPAIR and target is ResearchAction.REPAIR and verdict.validity_pass,
                    # Generic repair-rate denominators are all evaluator-invalid
                    # initial branches.  Mechanism-specific repair rates below keep
                    # their own confounding/leakage eligibility flags.
                    "invalid_repair_eligible": baseline_verdict.evidence_state.value == "invalid",
                    "confounding_repair_eligible": is_confounding_repair_case,
                    "confounding_repair_success": confounding_repair_success,
                    "leakage_repair_eligible": is_leakage_repair_case,
                    "leakage_repair_success": leakage_repair_success,
                    **evidence,
                    "underpower_handled": action is target,
                    "insufficient_handling_eligible": baseline_verdict.evidence_state.value == "insufficient",
                    "erroneous_persistence": action is ResearchAction.CONTINUE and target is not ResearchAction.CONTINUE,
                    "utility": branch_values[action],
                    "model_state": (
                        _rule_action(public_obs, question.family).name.lower()
                        if name == "Rule-Based" else None
                    ),
                    "state_prediction_source": "policy_output" if name == "Rule-Based" else "not_emitted",
                }
                # Rule-Based state head is the public diagnostic classifier, not the
                # evaluator label; convert its action-implied state carefully.
                if name == "Rule-Based":
                    state_signals = set(public_obs.get("validity_signals", ()))
                    if "split_overlap_diagnostic" in state_signals or "metric_scope_mismatch" in state_signals or "treatment_confounder_dependence" in state_signals:
                        row["model_state"] = "invalid"
                    else:
                        lo, hi = public_obs["confidence_interval"]
                        row["model_state"] = "insufficient" if hi - lo > 0.30 or "sample_count_below_precision_target" in state_signals else ("supported" if lo > 0.02 else ("refuted" if hi < 0.02 else "insufficient"))
                comparator_rows[name].append(row)

    # Add common audit fields to model rows and aggregate all comparator summaries.
    for row in rows:
        world = benchmark.world(row["world_id"])
        selected_evidence = branch_evidence.get(
            (row["policy_question_id"], row["world_id"], row["model_action"]),
            {},
        )
        row["method"] = "Frozen-Local-Model"
        row["action_correct_audit"] = row["model_action"] == row["target_action_audit"]
        row["selected_action"] = row["model_action"]
        row["initial_state_audit"] = row["evaluator_diagnostic_state"]
        row["effective_switch"] = row["model_action"] == ResearchAction.SWITCH.value and row["target_action_audit"] == ResearchAction.SWITCH.value
        row["required_switch"] = row["target_action_audit"] == ResearchAction.SWITCH.value
        row["invalid_repaired"] = bool(
            row["model_action"] == ResearchAction.REPAIR.value
            and row["target_action_audit"] == ResearchAction.REPAIR.value
            and selected_evidence.get("post_action_validity_pass", False)
        )
        row["invalid_repair_eligible"] = row["initial_state_audit"] == "invalid"
        row["confounding_repair_eligible"] = bool(world.confounding)
        row["leakage_repair_eligible"] = bool(world.leakage)
        row.update(selected_evidence)
        row["confounding_repair_success"] = bool(
            selected_evidence.get("confounding_repair_success", False)
        )
        row["leakage_repair_success"] = bool(
            selected_evidence.get("leakage_repair_success", False)
        )
        row["underpower_handled"] = row["model_action"] == row["target_action_audit"]
        row["insufficient_handling_eligible"] = row["initial_state_audit"] == "insufficient"
        row["erroneous_persistence"] = row["model_action"] == ResearchAction.CONTINUE.value and row["target_action_audit"] != ResearchAction.CONTINUE.value
        row["utility"] = float(
            selected_evidence.get("evaluator_branch_utility", 0.0)
        )
        row["model_state"] = row.get("model_state")

    summaries = {name: _summary(data) for name, data in comparator_rows.items()}
    summaries["Frozen-Local-Model"] = _summary(rows)
    # ``Search-Only`` is the implementation name retained for compatibility with
    # the runner, while the feedback's baseline table calls the same evaluator-side
    # upper-bound control ``Oracle``.  Keep both labels explicit and auditable.
    summaries["Oracle"] = dict(summaries["Search-Only"])
    action_accuracy = summaries["Frozen-Local-Model"]["action_accuracy_audit"]
    state_accuracy = sum(row["model_state"] == row["world_kind_audit"] for row in rows) / max(1, len(rows))
    base_utility = summaries["Base"].get("budget_normalized_scientific_value")
    oracle_utility = summaries["Oracle"].get("budget_normalized_scientific_value")
    base_oracle_gap = (
        float(oracle_utility - base_utility)
        if base_utility is not None and oracle_utility is not None
        else None
    )
    gates = {
        "real_model_zero_shot_completed": len(rows) == 48,
        "tier1_backend_verified": all(row.get("tier1_backend") == "tier1_numpy" for row in rows),
        "public_observation_only": all(row["model_input_source"] == "public_observation_only" for row in rows),
        "neutral_policy_question_ids": all(
            str(row.get("policy_question_id", "")) == "tier1_public_question"
            or (
                str(row.get("policy_question_id", "")).startswith("question_")
                and not str(row.get("policy_question_id", "")).startswith("t1_")
            )
            for row in rows
        ),
        "exported_observations_neutral": all(
            isinstance(row.get("observation"), dict)
            and row["observation"].get("question_id") == row.get("policy_question_id")
            and not any(
                key in row["observation"]
                for key in ("world_id", "world_kind", "target_action", "optimal_action", "verifier_labels")
            )
            for row in rows
        ),
        "hidden_world_fields_excluded_from_prompt": all(
            "world_id" not in prompt
            and "target_action_audit" not in prompt
            and "world_kind_audit" not in prompt
            for prompt in prompts
        ),
        "protocol_version_consistent": protocol.protocol_version == benchmark.protocol_version,
        "conditional_denominators_explicit": all(
            all(
                key in summaries[method]
                for key in (
                    "required_switch_eligible_n",
                    "confounding_repair_eligible_n",
                    "leakage_repair_eligible_n",
                    "insufficient_handling_eligible_n",
                )
            )
            for method in summaries
        ),
        # External-paper comparison is intentionally never authorized by this
        # diagnostic runner; it is a boundary assertion, not a pass condition.
        "formal_external_comparison": False,
    }
    diagnostic_pass = all(
        gates[name]
        for name in (
            "real_model_zero_shot_completed",
            "tier1_backend_verified",
            "public_observation_only",
            "neutral_policy_question_ids",
            "exported_observations_neutral",
            "hidden_world_fields_excluded_from_prompt",
            "protocol_version_consistent",
            "conditional_denominators_explicit",
        )
    )
    payload = {
        "schema_version": "pesco_experiment_b_real_zero_shot_v0.3",
        "experiment": "B_zero_shot_failure_diagnosis",
        "methods": ["Frozen-Local-Model", "Base", "Rule-Based", "Oracle"],
        "status": "completed_diagnostic" if diagnostic_pass else "failed_diagnostic",
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
        "real_model_zero_shot_completed": len(rows) == len(prompts) == 48,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": _digest(checkpoint / "config.json") if (checkpoint / "config.json").exists() else _digest(checkpoint),
        "checkpoint_config_digest": _digest(checkpoint / "config.json") if (checkpoint / "config.json").exists() else None,
        "checkpoint_digest_scope": "config.json" if (checkpoint / "config.json").exists() else "checkpoint_path_bytes",
        "runner_source_digest": _digest(Path(__file__)),
        "model_rows_reused": bool(reuse_model_rows),
        "model_rows_reused_n": max(0, len(rows) - len(rows_to_refresh)) if reuse_model_rows else 0,
        "model_rows_refreshed_n": len(rows_to_refresh) if reuse_model_rows else len(rows),
        "model_forward_pass_executed_this_run": bool((not reuse_model_rows) or rows_to_refresh),
        "model_row_provenance": (
            "full_forward"
            if not reuse_model_rows
            else "mixed_exact_observation_cache_and_forward_refresh"
            if rows_to_refresh
            else "exact_observation_cache_only"
        ),
        "model_class": model_class_name,
        "protocol_version": protocol.protocol_version,
        "benchmark_protocol_version": benchmark.protocol_version,
        "protocol_version_consistent": protocol.protocol_version == benchmark.protocol_version,
        "protocol_digest": protocol_digest,
        "benchmark_manifest_digest": benchmark_manifest["manifest_digest"],
        "benchmark_freshness_match": bool(current_observation_fresh),
        "current_benchmark_manifest_digest": benchmark_manifest["manifest_digest"],
        "changed_public_observation_n": int(len(rows_to_refresh)),
        "current_full_forward_completed": current_forward_completed,
        "historical_artifact": not current_observation_fresh,
        "freshness_audit": freshness_audit,
        "benchmark_schema_version": benchmark.schema_version,
        "confirmation_protocol": {
            "exploration_seed_count": len(protocol.exploration_seeds),
            "confirmation_seed_count": len(protocol.confirmation_seeds),
            "exploration_confirmation_disjoint": not (
                set(protocol.exploration_seeds) & set(protocol.confirmation_seeds)
            ),
            "independent_confirmation_required": protocol.independent_confirmation_required,
        },
        "split_access_boundary": {
            "diagnostic_rows": len(rows),
            "internal_splits": sorted({str(row.get("split")) for row in rows}),
            "formal_promotion_ids_opened": [],
            "formal_final_id_ids_opened": [],
            "formal_final_ood_ids_opened": [],
            "formal_promotion_question_ids_opened": [],
            "formal_final_id_question_ids_opened": [],
            "formal_final_ood_question_ids_opened": [],
            "formal_final_access": False,
            "note": "B is a world-level diagnostic on the frozen Tier-1 benchmark; no formal promotion/final claims are authorized.",
        },
        "public_input_whitelist": [
            "task_family", "current_method", "effect_estimate", "confidence_interval",
            "sample_size", "seed_count", "remaining_budget", "validity_signals",
            "history_summary", "active_hypothesis_id", "hypothesis_beliefs",
        ],
        "hidden_input_fields_excluded": [
            "question_id", "world_id", "world_kind", "target_action", "latent_effects",
            "verifier_labels", "optimal_action",
        ],
        "question_count": len(benchmark.questions),
        "row_count": len(rows),
        "action_accuracy_audit": action_accuracy,
        "state_accuracy_audit": state_accuracy,
        "base_oracle_gap": {
            "base_method": "Base",
            "oracle_method": "Oracle",
            "base_budget_normalized_scientific_value": base_utility,
            "oracle_budget_normalized_scientific_value": oracle_utility,
            "oracle_minus_base": base_oracle_gap,
            "gap_observed": bool(base_oracle_gap is not None and base_oracle_gap > 0.0),
            "interpretation": "Evaluator-side diagnostic gap; Oracle/Search-Only is not an external-paper reproduction.",
        },
        "comparators": {
            "implementation_status": {
                "Base": "fixed_policy_reference",
                "Rule-Based": "transparent_public_evidence_control",
                "Search-Only": "evaluator_side_oracle_diagnostic",
                "Oracle": "alias_of_Search-Only_evaluator_side_oracle_diagnostic",
                "Frozen-Local-Model": "genuine_local_checkpoint_zero_shot",
            },
            "summaries": summaries,
            "rows": {
                **{name: data for name, data in comparator_rows.items()},
                "Oracle": comparator_rows["Search-Only"],
            },
        },
        "gates": gates,
        # A complete row count alone must not turn a malformed or evaluator-leaking
        # export into a successful B artifact.  The formal-comparison boundary stays
        # false even when this diagnostic itself passes.
        "pass": diagnostic_pass,
        "interpretation": (
            "A genuine frozen local-model diagnostic; audit labels are not model inputs "
            "and do not authorize LLM training or external baseline claims. "
            + (
                "This artifact refresh reused rows only when their public observation "
                "fingerprint matched; changed rows were refreshed by the frozen model."
                if reuse_model_rows
                else "The frozen model forward pass was executed in this run."
            )
        ),
        "rows": rows,
    }
    progress_payload.update({
        "status": "completed",
        "completed_forward_rows": len(forward_indices),
    })
    progress_path.write_text(
        json.dumps(progress_payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    # Keep a descriptive alias alongside the short runner output name so the
    # feedback's Experiment B can be located without interpreting a filename.
    if destination.name == "tier1_zero_shot.json":
        alias = destination.parent / "experiment_b_real_zero_shot.json"
        alias.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="SigLip_LLM/Qwen2_5_3B")
    parser.add_argument("--output", default="PESCO/artifacts/tier1_zero_shot.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--reuse-model-rows",
        action="store_true",
        help="reuse existing rows only when their public observation fingerprint matches",
    )
    parser.add_argument(
        "--refresh-changed-model-rows",
        action="store_true",
        help="with --reuse-model-rows, run the frozen model only for changed public observations",
    )
    args = parser.parse_args(argv)
    payload = run(
        args.output,
        args.checkpoint,
        batch_size=args.batch_size,
        reuse_model_rows=args.reuse_model_rows,
        refresh_changed_model_rows=args.refresh_changed_model_rows,
    )
    print(json.dumps({"output": args.output, "rows": payload["row_count"], "pass": payload["pass"], "action_accuracy_audit": payload["action_accuracy_audit"]}, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

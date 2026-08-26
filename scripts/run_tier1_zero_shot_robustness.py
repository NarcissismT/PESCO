#!/usr/bin/env python3
"""Run Experiment-B format/prior robustness controls on local checkpoints.

Examples
--------
The default command uses the one checkpoint available in the working tree and
records a diagnostic (it intentionally fails the three-checkpoint requirement):

    python scripts/run_tier1_zero_shot_robustness.py \
      --checkpoint ../SigLip_LLM/Qwen2_5_3B \
      --output artifacts/tier1_zero_shot_robustness.json \
      --question-limit 4 --max-configs 8

For a formal multi-checkpoint diagnostic, repeat ``--checkpoint`` at least three
times with distinct full weight digests.  Duplicate paths/digests are rejected
from the denominator rather than counted as independent checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_zero_shot_robustness import (
    ACTION_LETTER_ROTATIONS,
    PROMPT_TEMPLATES,
    action_mapping_payload,
    canonical_action_letter,
    entropy,
    mapping_checksum,
    null_prior_summary,
    public_rows,
    render_public_prompt,
    resolve_single_token_ids,
    softmax,
)
from research_strategy_optimization.utils.run_manifest import (
    build_run_manifest,
    checkpoint_inventory,
    write_run_manifest,
)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _checkpoint_args(values: Sequence[str], list_path: str | None) -> list[Path]:
    paths = [Path(value) for value in values]
    if list_path:
        source = Path(list_path)
        payload = json.loads(source.read_text(encoding="utf-8")) if source.suffix == ".json" else None
        if isinstance(payload, dict):
            payload = payload.get("checkpoints", [])
        if payload is None:
            payload = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not isinstance(payload, list):
            raise ValueError("checkpoint list must be a JSON list or newline-delimited paths")
        paths.extend(Path(str(item)) for item in payload)
    if not paths:
        paths = [Path("../SigLip_LLM/Qwen2_5_3B")]
    return paths


_INSTRUCTION_ROLES = {"instruction", "instruct", "chat", "sft"}


def _available_checkpoints(
    paths: Sequence[Path], roles: Sequence[str] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Inventory and deduplicate checkpoints by complete digest."""

    inventories: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_paths: list[str] = []
    role_values = list(roles or ())
    if role_values and len(role_values) != len(paths):
        raise ValueError("--checkpoint-role must be supplied once per --checkpoint")
    for index, raw in enumerate(paths):
        path = raw if raw.is_absolute() else (ROOT / raw)
        inventory = checkpoint_inventory(path, root=ROOT)
        inventory["requested_path"] = str(raw)
        inventory["checkpoint_role"] = (
            str(role_values[index]).strip().lower() if role_values else "unknown"
        )
        if not inventory.get("available"):
            inventories.append(inventory)
            continue
        digest = str(inventory.get("full_checkpoint_digest"))
        if digest in seen:
            duplicate_paths.append(str(raw))
            inventory["duplicate_of_full_checkpoint_digest"] = digest
            inventories.append(inventory)
            continue
        seen.add(digest)
        inventory["unique_checkpoint_index"] = len(seen) - 1
        inventories.append(inventory)
    return inventories, duplicate_paths


def _unique_available(inventories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in inventories:
        if not item.get("available"):
            continue
        digest = str(item.get("full_checkpoint_digest"))
        if digest in seen:
            continue
        seen.add(digest)
        result.append(dict(item))
    return result


def _chat_template_inventory(inventories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report declared template availability separately from runtime support."""

    declared: list[dict[str, Any]] = []
    for item in inventories:
        if not item.get("available"):
            continue
        checkpoint_path = Path(str(item.get("path")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        tokenizer_config = checkpoint_path / "tokenizer_config.json"
        has_template = False
        if tokenizer_config.exists():
            try:
                config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
                has_template = bool(config.get("chat_template"))
            except (OSError, ValueError, TypeError):
                has_template = False
        declared.append(
            {
                "checkpoint_digest": item.get("full_checkpoint_digest"),
                "declared_in_tokenizer_config": has_template,
            }
        )
    try:
        jinja_version = importlib.metadata.version("jinja2")
    except importlib.metadata.PackageNotFoundError:
        jinja_version = None
    except Exception:
        jinja_version = None
    native_runtime = False
    if jinja_version:
        try:
            native_runtime = tuple(int(part) for part in jinja_version.split(".")[:2]) >= (3, 1)
        except ValueError:
            native_runtime = False
    return {
        "checkpoint_declarations": declared,
        "jinja2_version": jinja_version,
        "native_apply_chat_template_runtime_available": native_runtime,
        "native_runtime_requirement": "jinja2>=3.1",
        "manual_fallback_is_not_native": True,
    }


def _selected_token_logits(model: Any, encoded: Any, token_ids: Sequence[int]) -> list[float]:
    """Project the final hidden state onto a small allowed-token set."""

    import torch

    base = getattr(model, "model", None)
    if base is None:
        base = model.get_base_model()
    outputs = base(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        use_cache=False,
        return_dict=True,
    )
    length = int(encoded["attention_mask"].sum(dim=1)[0])
    hidden = outputs.last_hidden_state[0, length - 1].float()
    weight = model.lm_head.weight[list(token_ids)].float()
    return [float(value) for value in hidden.matmul(weight.transpose(0, 1)).detach().cpu()]


def _load_model(checkpoint: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch and transformers are required for B robustness") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = {}
    config_path = checkpoint / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    dtype_name = str(config.get("torch_dtype", "float32"))
    dtype = getattr(torch, dtype_name, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    return torch, tokenizer, model


def _score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    action_mapping: Mapping[str, Any],
    *,
    run_generation_control: bool = False,
) -> dict[str, Any]:
    import torch

    action_letters = tuple(action_mapping)
    action_ids = resolve_single_token_ids(tokenizer, action_letters)
    state_ids = resolve_single_token_ids(tokenizer, ("S", "R", "I", "V"), leading_space=True)
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    with torch.inference_mode():
        action_logits = _selected_token_logits(model, encoded, [action_ids[x] for x in action_letters])
    action_probabilities = softmax(action_logits)
    action_letter = action_letters[max(range(len(action_probabilities)), key=action_probabilities.__getitem__)]

    # State is deliberately scored after the chosen action marker, matching the
    # original B output grammar rather than reading unrelated letters at ACTION=.
    state_prompt = prompt + action_letter + " STATE= "
    state_encoded = tokenizer(state_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    with torch.inference_mode():
        state_logits = _selected_token_logits(model, state_encoded, [state_ids[x] for x in ("S", "R", "I", "V")])
    state_probabilities = softmax(state_logits)
    state_letters = ("S", "R", "I", "V")
    state_letter = state_letters[max(range(len(state_probabilities)), key=state_probabilities.__getitem__)]
    result: dict[str, Any] = {
        "action_letter": action_letter,
        "action": action_mapping[action_letter].value,
        "action_probabilities_by_letter": {
            letter: float(probability) for letter, probability in zip(action_letters, action_probabilities)
        },
        "action_probabilities": {
            action_mapping[letter].value: float(probability)
            for letter, probability in zip(action_letters, action_probabilities)
        },
        "state_letter": state_letter,
        "state": {"S": "supported", "R": "refuted", "I": "insufficient", "V": "invalid"}[state_letter],
        "state_probabilities": {
            state: float(probability)
            for state, probability in zip(("supported", "refuted", "insufficient", "invalid"), state_probabilities)
        },
        "action_entropy": entropy(action_probabilities),
        "state_entropy": entropy(state_probabilities),
        "constrained_logits_equivalent": True,
        "generation_control": None,
    }
    if run_generation_control:
        # This is intentionally a separate control from the equivalent masked
        # logits above.  ``generate`` is only run for a bounded subset because a
        # 3B CPU forward for every prompt/configuration is expensive.
        try:
            allowed = tuple(action_ids.values())
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

            def allowed_tokens(_batch: int, _input: Any) -> list[int]:
                return list(allowed)

            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    prefix_allowed_tokens_fn=allowed_tokens,
                    pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id),
                )
            generated_id = int(generated[0, -1])
            generated_letter = next(
                (letter for letter, token_id in action_ids.items() if token_id == generated_id), None
            )
            result["generation_control"] = {
                "status": "completed" if generated_letter is not None else "unexpected_token",
                "generated_token_id": generated_id,
                "generated_letter": generated_letter,
                "generated_action": action_mapping[generated_letter].value if generated_letter else None,
                "matches_masked_argmax": bool(generated_letter == action_letter),
            }
        except Exception as exc:  # pragma: no cover - transformers-version dependent
            result["generation_control"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def _null_prompt(
    model: Any,
    tokenizer: Any,
    *,
    template: str,
    mapping: Mapping[str, Any],
    chat: bool,
) -> dict[str, Any]:
    prompt, rendering = render_public_prompt(
        None, template=template, mapping=mapping, chat=chat, tokenizer=tokenizer
    )
    score = _score_prompt(model, tokenizer, prompt, mapping, run_generation_control=False)
    return {
        "prompt": prompt,
        "rendering": rendering,
        "mapping": action_mapping_payload(mapping),
        "mapping_checksum": mapping_checksum(mapping),
        **null_prior_summary(score["action_probabilities_by_letter"], mapping),
        "state_prior": score["state_probabilities"],
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "action_accuracy": None}
    correct = sum(bool(row.get("action_correct")) for row in rows)
    letter_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    for row in rows:
        letter = str(row.get("action_letter"))
        action = str(row.get("predicted_action"))
        letter_counts[letter] = letter_counts.get(letter, 0) + 1
        semantic_counts[action] = semantic_counts.get(action, 0) + 1
    return {
        "n": len(rows),
        "action_accuracy": correct / len(rows),
        "letter_counts": letter_counts,
        "semantic_action_counts": semantic_counts,
        "semantic_switch_rate": semantic_counts.get(ResearchAction.SWITCH.value, 0) / len(rows),
        "mean_action_entropy": sum(float(row["action_entropy"]) for row in rows) / len(rows),
        "mean_state_entropy": sum(float(row["state_entropy"]) for row in rows) / len(rows),
        "generation_control_completed_n": sum(
            row.get("generation_control", {}).get("status") == "completed" for row in rows
        ),
    }


def run(
    output: str | Path,
    checkpoints: Sequence[str | Path],
    *,
    question_limit: int | None = None,
    max_configs: int | None = None,
    generation_control_limit: int = 4,
    checkpoint_roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    destination = Path(output)
    inventories, duplicate_paths = _available_checkpoints(
        [Path(path) for path in checkpoints], checkpoint_roles
    )
    unique = _unique_available(inventories)
    rows = public_rows(question_limit)
    configs: list[tuple[str, str, bool]] = []
    for template in PROMPT_TEMPLATES:
        for rotation in ACTION_LETTER_ROTATIONS:
            configs.append((template, rotation, False))
            configs.append((template, rotation, True))
    if max_configs is not None:
        configs = configs[: max(0, int(max_configs))]

    model_rows: list[dict[str, Any]] = []
    null_priors: list[dict[str, Any]] = []
    runtime_errors: list[dict[str, Any]] = []
    for checkpoint_index, inventory in enumerate(unique):
        checkpoint = Path(str(inventory["path"]))
        # inventory path is root-relative only when root was supplied; recover the
        # actual path from the user request by matching the digest.
        requested = next(
            (Path(item.get("requested_path")) for item in inventories if item.get("full_checkpoint_digest") == inventory.get("full_checkpoint_digest")),
            checkpoint,
        )
        if not requested.is_absolute():
            requested = (ROOT / requested).resolve()
        try:
            torch, tokenizer, model = _load_model(requested)
            generation_count = 0
            for template, rotation_name, chat in configs:
                mapping = ACTION_LETTER_ROTATIONS[rotation_name]
                prior = _null_prompt(
                    model, tokenizer, template=template, mapping=mapping, chat=chat
                )
                null_priors.append(
                    {
                        "checkpoint_index": checkpoint_index,
                        "checkpoint_digest": inventory.get("full_checkpoint_digest"),
                        "template": template,
                        "rotation": rotation_name,
                        "chat": chat,
                        **prior,
                    }
                )
                for row in rows:
                    prompt, rendering = render_public_prompt(
                        row["observation"],
                        template=template,
                        mapping=mapping,
                        chat=chat,
                        tokenizer=tokenizer,
                    )
                    generation = generation_count < int(generation_control_limit)
                    score = _score_prompt(
                        model,
                        tokenizer,
                        prompt,
                        mapping,
                        run_generation_control=generation,
                    )
                    generation_count += 1 if generation else 0
                    hidden_tokens = ("target_action_audit", "world_kind_audit", row["world_id"])
                    prompt_leak = any(token and token in prompt for token in hidden_tokens)
                    model_rows.append(
                        {
                            "checkpoint_index": checkpoint_index,
                            "checkpoint_digest": inventory.get("full_checkpoint_digest"),
                            "checkpoint_path": inventory.get("path"),
                            "question_id": row["question_id"],
                            "policy_question_id": row["policy_question_id"],
                            "split": row["split"],
                            "mechanism_family_audit": row["family"],
                            "world_id_audit": row["world_id"],
                            "target_action_audit": row["target_action_audit"],
                            "world_kind_audit": row["world_kind_audit"],
                            "prompt_template": template,
                            "rotation": rotation_name,
                            "action_mapping": action_mapping_payload(mapping),
                            "mapping_checksum": mapping_checksum(mapping),
                            "chat_requested": bool(chat),
                            "chat_rendering": rendering,
                            "prompt": prompt,
                            "prompt_leak_audit": prompt_leak,
                            "predicted_action": score["action"],
                            "action_letter": score["action_letter"],
                            "state": score["state"],
                            "state_letter": score["state_letter"],
                            "action_probabilities": score["action_probabilities"],
                            "action_probabilities_by_letter": score["action_probabilities_by_letter"],
                            "state_probabilities": score["state_probabilities"],
                            "action_entropy": score["action_entropy"],
                            "state_entropy": score["state_entropy"],
                            "action_correct": score["action"] == row["target_action_audit"],
                            "generation_control": score["generation_control"],
                            "model_input_source": "public_observation_only",
                        }
                    )
        except Exception as exc:
            runtime_errors.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_digest": inventory.get("full_checkpoint_digest"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            try:
                del model, tokenizer, torch
            except UnboundLocalError:
                pass
            gc.collect()

    checkpoint_requirement = {
        "minimum_unique_checkpoints": 3,
        "unique_available_checkpoints": len(unique),
        "instruction_checkpoint_count": sum(
            str(item.get("checkpoint_role", "unknown")).lower() in _INSTRUCTION_ROLES
            for item in unique
        ),
        "instruction_role_labels_complete": bool(unique)
        and all(str(item.get("checkpoint_role", "unknown")).lower() in _INSTRUCTION_ROLES for item in unique),
        "duplicate_requested_paths": duplicate_paths,
        "passed": bool(
            len(unique) >= 3
            and sum(
                str(item.get("checkpoint_role", "unknown")).lower() in _INSTRUCTION_ROLES
                for item in unique
            )
            >= 3
        ),
        "status": "passed"
        if len(unique) >= 3
        and sum(
            str(item.get("checkpoint_role", "unknown")).lower() in _INSTRUCTION_ROLES
            for item in unique
        )
        >= 3
        else "insufficient_checkpoints_or_instruction_roles",
        "note": "Only distinct full_checkpoint_digest values count; every formal checkpoint must be explicitly labelled instruction/instruct/chat/sft.",
    }
    full_coverage = question_limit is None or int(question_limit) >= 12
    expected_configuration_count = len(PROMPT_TEMPLATES) * len(ACTION_LETTER_ROTATIONS) * 2
    full_configuration_coverage = len(configs) == expected_configuration_count
    prompt_leak_free = all(not bool(row.get("prompt_leak_audit")) for row in model_rows)
    chat_template_inventory = _chat_template_inventory(inventories)
    # A null-prompt-only invocation intentionally has no benchmark rows but is a
    # completed model forward if its prior calibration was emitted.  Keep that
    # case distinct from a failed model load/forward.
    model_completed = bool(model_rows or null_priors) and not runtime_errors
    status = (
        "completed_multicheckpoint_robustness"
        if model_completed and checkpoint_requirement["passed"] and full_coverage and full_configuration_coverage
        else "completed_single_checkpoint_robustness"
        if model_completed
        else "blocked_model_forward"
        if unique
        else "blocked_no_checkpoint"
    )
    payload: dict[str, Any] = {
        "schema_version": "pesco_experiment_b_zero_shot_robustness_v0.1",
        "experiment": "B_zero_shot_format_prior_robustness",
        "status": status,
        "pass": bool(status == "completed_multicheckpoint_robustness" and prompt_leak_free),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "checkpoint_requirement": checkpoint_requirement,
        "checkpoint_inventories": inventories,
        "checkpoint_full_digests": [item.get("full_checkpoint_digest") for item in unique],
        "checkpoint_weights_digests": [item.get("full_weights_digest") for item in unique],
        "checkpoint_tokenizer_digests": [item.get("full_tokenizer_digest") for item in unique],
        "chat_template_inventory": chat_template_inventory,
        "requirements_audit": {
            "action_letter_rotation": {
                "implemented": True,
                "executed": bool(
                    {str(row.get("rotation")) for row in model_rows}
                    == set(ACTION_LETTER_ROTATIONS)
                ),
            },
            "prompt_templates": {
                "implemented": True,
                "executed": bool(
                    {str(row.get("prompt_template")) for row in model_rows}
                    == set(PROMPT_TEMPLATES)
                ),
            },
            "chat_template_control": {
                "implemented": True,
                "executed": bool(any(bool(row.get("chat_requested")) for row in model_rows)),
                "native_runtime_available": bool(
                    chat_template_inventory["native_apply_chat_template_runtime_available"]
                ),
            },
            "constrained_generation_control": {
                "implemented": True,
                "executed": bool(
                    any(row.get("generation_control", {}).get("status") == "completed" for row in model_rows)
                ),
            },
            "null_prompt_prior_calibration": {
                "implemented": True,
                "executed": bool(null_priors),
            },
            "full_checkpoint_tokenizer_code_dependency_hashes": {
                "implemented": True,
                "executed": bool(unique),
            },
            "instruction_checkpoint_panel": {
                "implemented": True,
                "executed": bool(checkpoint_requirement["passed"]),
            },
        },
        "row_count_per_configuration": len(rows),
        "question_limit": question_limit,
        "full_benchmark_coverage": full_coverage,
        "expected_configuration_count": expected_configuration_count,
        "full_configuration_coverage": full_configuration_coverage,
        "configuration_count": len(configs),
        "configuration_protocol": {
            "action_letter_rotations": {
                name: action_mapping_payload(mapping)
                for name, mapping in ACTION_LETTER_ROTATIONS.items()
            },
            "prompt_templates": list(PROMPT_TEMPLATES),
            "renderings": ["plain_text", "chat_template"],
            "constrained_generation": {
                "masked_next_token_scoring": True,
                "generate_prefix_allowed_tokens_control": True,
                "generation_control_limit_per_checkpoint": int(generation_control_limit),
            },
            "null_prompt_prior_calibration": True,
        },
        "runtime_errors": runtime_errors,
        "prompt_leak_free": prompt_leak_free,
        "summaries": {
            "by_checkpoint_template_rotation_chat": {},
            "null_prior_count": len(null_priors),
        },
        "null_prior_calibration": null_priors,
        "rows": model_rows,
    }
    summaries: dict[str, list[dict[str, Any]]] = {}
    for row in model_rows:
        key = ":".join(
            [str(row["checkpoint_index"]), str(row["prompt_template"]), str(row["rotation"]), str(row["chat_requested"])]
        )
        summaries.setdefault(key, []).append(row)
    payload["summaries"]["by_checkpoint_template_rotation_chat"] = {
        key: _aggregate(values) for key, values in sorted(summaries.items())
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    # The manifest is written after the result so the command/data/code hashes
    # bind the observed artifact without hashing the manifest itself recursively.
    manifest = build_run_manifest(
        experiment="B_zero_shot_format_prior_robustness",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            Path(__file__),
            ROOT / "research_strategy_optimization/evaluation/tier1_zero_shot_robustness.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_benchmark.py",
            ROOT / "research_strategy_optimization/environments/tier1_tabular_env.py",
        ],
        data_paths=[ROOT / "artifacts/tier1_v03/benchmark_manifest.json"],
        seeds={"inference": [17], "exploration": [17, 29, 41, 53], "confirmation": [103, 107, 109, 113]},
        checkpoint=Path(str(unique[0]["path"])) if unique else None,
        status=status,
        diagnostics={
            "checkpoint_requirement": checkpoint_requirement,
            "configuration_count": len(configs),
            "row_count_per_configuration": len(rows),
            "prompt_leak_free": prompt_leak_free,
            "runtime_error_count": len(runtime_errors),
            "full_benchmark_coverage": full_coverage,
            "full_configuration_coverage": full_configuration_coverage,
            "native_chat_template_rows": sum(
                bool(row.get("chat_rendering", {}).get("native_chat_template")) for row in model_rows
            ),
            "chat_template_inventory": chat_template_inventory,
            "null_prior_count": len(null_priors),
        },
    )
    # Keep a per-artifact sidecar.  A generic ``artifacts/run_manifest.json``
    # would collide with unrelated experiments when several bounded B controls
    # are run in the same directory.
    manifest_path = destination.with_name(destination.stem + ".run_manifest.json")
    write_run_manifest(manifest_path, manifest)
    payload["run_manifest"] = str(manifest_path)
    payload["provenance"] = {
        "git_sha": manifest.get("git_sha"),
        "source_digest": manifest.get("source_digest"),
        "dependency_spec_digest": manifest.get("dependency_spec_digest"),
        "dependency_versions_digest": manifest.get("dependency_versions_digest"),
        "checkpoint_full_digest": manifest.get("checkpoint_full_digest"),
        "checkpoint_weights_digest": manifest.get("checkpoint_weights_digest"),
        "checkpoint_tokenizer_digest": manifest.get("checkpoint_tokenizer_digest"),
        "manifest_digest": manifest.get("manifest_digest"),
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="local checkpoint path; repeatable")
    parser.add_argument("--checkpoint-list", default=None, help="JSON list or newline-delimited checkpoint paths")
    parser.add_argument("--output", default="artifacts/tier1_zero_shot_robustness.json")
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--generation-control-limit", type=int, default=4)
    parser.add_argument(
        "--checkpoint-role",
        action="append",
        default=[],
        help="checkpoint role (instruction/instruct/chat/sft); repeat once per --checkpoint",
    )
    args = parser.parse_args(argv)
    payload = run(
        args.output,
        _checkpoint_args(args.checkpoint, args.checkpoint_list),
        question_limit=args.question_limit,
        max_configs=args.max_configs,
        generation_control_limit=args.generation_control_limit,
        checkpoint_roles=args.checkpoint_role,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "status": payload["status"],
                "pass": payload["pass"],
                "unique_checkpoints": payload["checkpoint_requirement"]["unique_available_checkpoints"],
                "rows": len(payload["rows"]),
                "null_priors": len(payload["null_prior_calibration"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

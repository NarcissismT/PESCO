"""Experiment-B robustness controls for a frozen local language model.

The original B diagnostic used one prompt and one A/B/C/D encoding.  This module
keeps the evaluator boundary unchanged while making the *format* controls
explicit:

* four fixed action-letter permutations (semantic actions are scored after
  decoding, so a letter prior cannot be mistaken for scientific competence);
* three public-observation prompt templates;
* plain text versus a tokenizer chat-template rendering;
* constrained next-token scoring as a generation control; and
* a null-prompt action prior calibration.

This is intentionally a diagnostic harness, not a new training method.  A
formal multi-checkpoint result requires at least three distinct full checkpoint
digests.  The runner therefore reports ``insufficient_checkpoints`` rather than
silently treating repeated copies of one checkpoint as independent models.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..environments.tier1_benchmark import build_tier1_v03_benchmark
from ..environments.tier1_tabular_env import Tier1TabularEnvironment
from ..environments.tier0_simulator import TrustedVerifier
from ..schemas import Protocol, ResearchAction


ACTION_ORDER: tuple[ResearchAction, ...] = (
    ResearchAction.CONTINUE,
    ResearchAction.SAMPLE,
    ResearchAction.REPAIR,
    ResearchAction.SWITCH,
)
ACTION_LABELS: Mapping[ResearchAction, str] = {
    ResearchAction.CONTINUE: "continue_current_method",
    ResearchAction.SAMPLE: "add_samples_or_seeds",
    ResearchAction.REPAIR: "repair_data_split",
    ResearchAction.SWITCH: "switch_to_alternative_method",
}
STATE_ORDER: tuple[str, ...] = ("supported", "refuted", "insufficient", "invalid")
STATE_LETTERS: tuple[str, ...] = ("S", "R", "I", "V")

# Every permutation is fixed in the protocol.  The semantic action order is
# never inferred from the letters and is always recovered through this table.
ACTION_LETTER_ROTATIONS: Mapping[str, Mapping[str, ResearchAction]] = {
    "canonical": {
        "A": ResearchAction.CONTINUE,
        "B": ResearchAction.SAMPLE,
        "C": ResearchAction.REPAIR,
        "D": ResearchAction.SWITCH,
    },
    "cyclic_1": {
        "A": ResearchAction.SAMPLE,
        "B": ResearchAction.REPAIR,
        "C": ResearchAction.SWITCH,
        "D": ResearchAction.CONTINUE,
    },
    "cyclic_2": {
        "A": ResearchAction.REPAIR,
        "B": ResearchAction.SWITCH,
        "C": ResearchAction.CONTINUE,
        "D": ResearchAction.SAMPLE,
    },
    "reverse": {
        "A": ResearchAction.SWITCH,
        "B": ResearchAction.REPAIR,
        "C": ResearchAction.SAMPLE,
        "D": ResearchAction.CONTINUE,
    },
}

PROMPT_TEMPLATES: tuple[str, ...] = ("compact_json", "prose", "schema")


@dataclass(frozen=True)
class PublicRow:
    """A policy-visible observation plus evaluator-only audit metadata."""

    question_id: str
    policy_question_id: str
    split: str
    family: str
    world_id: str
    observation: Mapping[str, Any]
    target_action: str
    world_kind: str


def _public_rows(question_limit: int | None = None) -> list[PublicRow]:
    benchmark = build_tier1_v03_benchmark()
    protocol = Protocol(protocol_version="pesco_v0_2")
    questions = list(benchmark.questions)
    if question_limit is not None:
        questions = questions[: max(0, int(question_limit))]
    rows: list[PublicRow] = []
    for question in questions:
        env = Tier1TabularEnvironment(worlds=question.worlds, protocol=protocol)
        for world in question.worlds:
            env.reset(question.policy_question_id, world.world_id, seed=17)
            baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            verdict = TrustedVerifier(protocol).evaluate(baseline, env)
            observation = env.visible_observation().to_dict()
            rows.append(
                PublicRow(
                    question_id=question.question_id,
                    policy_question_id=question.policy_question_id,
                    split=question.split,
                    family=question.family,
                    world_id=world.world_id,
                    observation=observation,
                    target_action=question.target_action(world.world_id).value,
                    world_kind=world.kind,
                )
            )
    return rows


def public_rows(question_limit: int | None = None) -> list[dict[str, Any]]:
    """Return serializable rows; hidden fields are kept outside prompt rendering."""

    return [
        {
            "question_id": row.question_id,
            "policy_question_id": row.policy_question_id,
            "split": row.split,
            "family": row.family,
            "world_id": row.world_id,
            "observation": dict(row.observation),
            "target_action_audit": row.target_action,
            "world_kind_audit": row.world_kind,
        }
        for row in _public_rows(question_limit)
    ]


def _compact_public(observation: Mapping[str, Any]) -> str:
    # Explicit whitelist: no question/world/target/evaluator fields can enter the
    # model input even if the source Observation grows new fields later.
    keys = (
        "task_family",
        "current_method",
        "effect_estimate",
        "confidence_interval",
        "sample_size",
        "seed_count",
        "remaining_budget",
        "validity_signals",
        "history_summary",
        "active_hypothesis_id",
        "hypothesis_beliefs",
    )
    payload = {key: observation.get(key) for key in keys}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _mapping_description(mapping: Mapping[str, ResearchAction]) -> str:
    return "; ".join(
        f"{letter}={ACTION_LABELS[action]}" for letter, action in mapping.items()
    )


def render_public_prompt(
    observation: Mapping[str, Any] | None,
    *,
    template: str,
    mapping: Mapping[str, ResearchAction],
    chat: bool = False,
    tokenizer: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render one public prompt and return rendering provenance.

    ``chat=True`` first calls the checkpoint's native ``apply_chat_template``.
    If that optional dependency is unavailable, a deterministic Qwen-style
    rendering is returned with ``native_chat_template=False``; the artifact must
    retain that distinction.
    """

    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt template: {template}")
    public = _compact_public(observation or {})
    mapping_text = _mapping_description(mapping)
    output_instruction = (
        f"Choose one action letter using this mapping: {mapping_text}. "
        "Choose one state letter S=supported R=refuted I=insufficient V=invalid. "
        "Return exactly ACTION=<letter> STATE=<letter>."
    )
    if template == "compact_json":
        text = (
            "Use PUBLIC evidence only. "
            + output_instruction
            + " PUBLIC="
            + public
            + " ACTION="
        )
    elif template == "prose":
        text = (
            "You are auditing a research decision. Do not infer hidden labels or "
            "world identity. Read only the public record below.\n"
            + output_instruction
            + "\nPUBLIC RECORD:\n"
            + public
            + "\nACTION="
        )
    else:
        text = (
            "<research_decision>\n"
            + "<public_observation>"
            + public
            + "</public_observation>\n"
            + "<action_mapping>"
            + mapping_text
            + "</action_mapping>\n"
            + output_instruction
            + "\nACTION="
        )

    provenance: dict[str, Any] = {
        "prompt_template": template,
        "chat_requested": bool(chat),
        "native_chat_template": False,
        "chat_rendering": "plain_text",
    }
    if not chat:
        return text, provenance
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "Return only the requested research decision fields."},
            {"role": "user", "content": text},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Put the marker after the assistant generation prompt so the scored
            # continuation is still a single action token.
            provenance.update({"native_chat_template": True, "chat_rendering": "tokenizer_apply_chat_template"})
            return rendered + "ACTION=", provenance
        except (ImportError, RuntimeError, TypeError) as exc:
            provenance.update(
                {
                    "chat_rendering": "manual_qwen_fallback",
                    "native_chat_template_error": f"{type(exc).__name__}: {exc}",
                }
            )
            # Qwen2's checked-in template has a deterministic no-tools path.  This
            # fallback is useful for a diagnostic, but is never relabeled as native.
            system = "Return only the requested research decision fields."
            return (
                "<|im_start|>system\n"
                + system
                + "<|im_end|>\n<|im_start|>user\n"
                + text
                + "<|im_end|>\n<|im_start|>assistant\nACTION=",
                provenance,
            )
    provenance.update({"chat_rendering": "unavailable_no_template"})
    return text, provenance


def action_mapping_payload(mapping: Mapping[str, ResearchAction]) -> dict[str, str]:
    return {letter: action.value for letter, action in mapping.items()}


def canonical_action_letter(mapping: Mapping[str, ResearchAction], action: ResearchAction) -> str:
    for letter, mapped_action in mapping.items():
        if mapped_action is action:
            return letter
    raise KeyError(action)


def resolve_single_token_ids(tokenizer: Any, letters: Iterable[str], *, leading_space: bool = False) -> dict[str, int]:
    """Resolve letters to one-token IDs, failing closed on tokenization changes."""

    result: dict[str, int] = {}
    for letter in letters:
        candidates = [(" " + letter) if leading_space else letter, letter if leading_space else (" " + letter)]
        chosen: list[int] | None = None
        for candidate in candidates:
            ids = list(tokenizer.encode(candidate, add_special_tokens=False))
            if len(ids) == 1:
                chosen = ids
                break
        if chosen is None:
            raise ValueError(f"letter {letter!r} is not a single tokenizer token")
        result[letter] = int(chosen[0])
    return result


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    weights = [math.exp(float(value) - maximum) for value in values]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def entropy(probabilities: Sequence[float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-300)) for p in probabilities))


def null_prior_summary(
    action_probabilities: Mapping[str, float],
    mapping: Mapping[str, ResearchAction],
) -> dict[str, Any]:
    semantic = {
        mapping[letter].value: float(action_probabilities.get(letter, 0.0))
        for letter in mapping
    }
    winner = max(semantic, key=semantic.get) if semantic else None
    values = list(semantic.values())
    return {
        "letter_probabilities": {str(k): float(v) for k, v in action_probabilities.items()},
        "semantic_action_probabilities": semantic,
        "semantic_prior_winner": winner,
        "semantic_entropy": entropy(values),
        "semantic_max_probability": max(values) if values else None,
    }


def mapping_checksum(mapping: Mapping[str, ResearchAction]) -> str:
    payload = action_mapping_payload(mapping)
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ACTION_LETTER_ROTATIONS",
    "ACTION_LABELS",
    "ACTION_ORDER",
    "PROMPT_TEMPLATES",
    "PublicRow",
    "action_mapping_payload",
    "canonical_action_letter",
    "entropy",
    "mapping_checksum",
    "null_prior_summary",
    "public_rows",
    "render_public_prompt",
    "resolve_single_token_ids",
    "softmax",
]

"""Independent v0.5 final-freeze benchmark and evaluator contract.

The v0.4 formal artifact is deliberately *not* promoted by this module.  It reused
the diagnostic latent generator and therefore cannot serve as a paper final after
the evaluator fixes in the review.  v0.5 creates a new, opaque final profile with
fresh generator recipes, at least forty final-ID and forty final-OOD question
clusters, and a whole-family holdout containing mechanisms absent from development.

Only the evaluator process should load the hidden manifest.  The public manifest is
an opaque commitment (question commitments, split counts, and digests) and contains
no family labels, world parameters, target actions, or world IDs.  The freeze
receipt is fail-closed: a receipt is ``frozen`` only when an explicit baseline
selection, a clean Git commit/tag, and all signature audits are present.  This file
does not modify the P2 evaluator; it is intentionally a separate final-boundary
module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from ..environments.tier0_simulator import TrustedVerifier
from ..environments.tier1_benchmark import (
    TIER1_REWARD_COMPONENT_NAMES,
    tier1_scientific_utility_components,
)
from ..schemas import EvidenceState, Observation, Protocol, ResearchAction, WorldSpec


V05_SCHEMA = "pesco_tier1_benchmark_v0.5_frozen_final"
V05_GENERATOR_VERSION = "pesco_v0.5_latent_generator_20260826_r1"
V05_EVALUATOR_VERSION = "pesco_v0.5_independent_evaluator_v1"
V05_SPLITS: Tuple[str, ...] = ("final_id", "final_ood")
V05_EXPLORATION_SEEDS: Tuple[int, ...] = (17, 29, 41, 53, 67, 71, 83, 97)
V05_CONFIRMATION_SEEDS: Tuple[int, ...] = (103, 107, 109, 113, 127, 131, 137, 139)

# The ID profile intentionally retains familiar mechanism *names* while changing
# the latent recipe.  This tests interpolation under a new generator.  OOD is a
# whole-family holdout and all four families are new relative to v0.4 development.
V05_FINAL_ID_FAMILIES: Tuple[str, ...] = (
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
    "confounding_underpower",
    "replication_instability",
)
V05_FINAL_OOD_FAMILIES: Tuple[str, ...] = (
    "selection_collider_v05",
    "measurement_delay_v05",
    "interference_spillover_v05",
    "adaptive_stopping_v05",
)
V05_ALL_FAMILIES = V05_FINAL_ID_FAMILIES + V05_FINAL_OOD_FAMILIES
# Naming aliases make the final profile discoverable alongside the v0.4 formal
# builder without changing the underlying registry.
V05_FORMAL_FINAL_ID_FAMILIES = V05_FINAL_ID_FAMILIES
V05_FORMAL_FINAL_OOD_FAMILIES = V05_FINAL_OOD_FAMILIES
V04_DEVELOPMENT_FAMILIES = {
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
    "confounding_underpower",
    "leakage_metric_mismatch",
    "protocol_drift",
    "replication_instability",
}
WORLD_KINDS: Tuple[str, ...] = ("supported", "refuted", "insufficient", "invalid")
ACTION_SET = ResearchAction.mvp_actions()


@dataclass(frozen=True)
class V05GeneratorRecipe:
    """Evaluator-only recipe descriptor used in generator-signature audits."""

    recipe_id: str
    mechanism_family: str
    version: str
    coefficients: Tuple[float, ...]
    flag_profile: Tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "mechanism_family": self.mechanism_family,
            "version": self.version,
            "coefficients": [float(value) for value in self.coefficients],
            "flag_profile": list(self.flag_profile),
        }


# Coefficients are deliberately unlike the v0.4 arithmetic grid.  They are part of
# the signed latent generator contract, not a policy feature.
V05_GENERATOR_RECIPES: Mapping[str, V05GeneratorRecipe] = MappingProxyType({
    "group_leakage": V05GeneratorRecipe(
        "gk_resample_241", "group_leakage", V05_GENERATOR_VERSION,
        (0.137, 0.041, 0.0071, 1.13), ("leakage",),
    ),
    "causal_confounding": V05GeneratorRecipe(
        "cc_shift_517", "causal_confounding", V05_GENERATOR_VERSION,
        (0.119, 0.063, 0.0047, 0.91), ("confounding",),
    ),
    "low_sample_variance": V05GeneratorRecipe(
        "lv_hetero_733", "low_sample_variance", V05_GENERATOR_VERSION,
        (0.151, 0.029, 0.0093, 1.27), ("sample_variance",),
    ),
    "subgroup_metric_mismatch": V05GeneratorRecipe(
        "sm_metric_829", "subgroup_metric_mismatch", V05_GENERATOR_VERSION,
        (0.131, 0.057, 0.0062, 1.07), ("metric_mismatch",),
    ),
    "confounding_underpower": V05GeneratorRecipe(
        "cu_underpower_941", "confounding_underpower", V05_GENERATOR_VERSION,
        (0.109, 0.071, 0.0056, 0.83), ("confounding", "underpower"),
    ),
    "replication_instability": V05GeneratorRecipe(
        "ri_replica_1103", "replication_instability", V05_GENERATOR_VERSION,
        (0.143, 0.035, 0.0081, 1.19), ("replication_instability",),
    ),
    "selection_collider_v05": V05GeneratorRecipe(
        "sc_collider_1229", "selection_collider_v05", V05_GENERATOR_VERSION,
        (0.173, 0.083, 0.0117, 1.41), ("confounding", "metric_mismatch", "selection"),
    ),
    "measurement_delay_v05": V05GeneratorRecipe(
        "md_delay_1373", "measurement_delay_v05", V05_GENERATOR_VERSION,
        (0.181, 0.047, 0.0109, 1.33), ("metric_mismatch", "protocol_delay"),
    ),
    "interference_spillover_v05": V05GeneratorRecipe(
        "is_spill_1499", "interference_spillover_v05", V05_GENERATOR_VERSION,
        (0.167, 0.079, 0.0123, 1.47), ("leakage", "confounding", "interference"),
    ),
    "adaptive_stopping_v05": V05GeneratorRecipe(
        "as_adaptive_1601", "adaptive_stopping_v05", V05_GENERATOR_VERSION,
        (0.191, 0.053, 0.0131, 1.57), ("protocol_invalid", "optional_stopping"),
    ),
})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_evidence_path(value: Any, *, repo_root: Path) -> Path | None:
    """Resolve a baseline-evidence path without silently accepting a missing file."""

    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if candidate.is_file() else None


def _development_manifest_has_observed_dev_data(path: Path) -> bool:
    """Require an actual, non-empty development split for a freeze receipt.

    Reserved placeholder manifests (for example ``question_ids: []``) are useful
    scaffolding but cannot prove that a baseline was selected before final access.
    A valid receipt must bind at least one dev question to one concrete world (or a
    split-level world count), while remaining agnostic to the benchmark generator.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    questions = payload.get("questions")
    if isinstance(questions, list):
        dev_questions = [
            item for item in questions
            if isinstance(item, Mapping) and str(item.get("split", "")) == "dev"
        ]
        if not dev_questions:
            # A split-specific manifest may use a top-level split marker.
            if str(payload.get("split", "")) != "dev":
                return False
            dev_questions = questions
        for question in dev_questions:
            if not str(question.get("question_id", "")).strip():
                continue
            worlds = question.get("worlds")
            if isinstance(worlds, list) and worlds:
                return True
            try:
                if int(question.get("world_count", 0)) > 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                pass
        return False
    # Support a compact split manifest, but reject the old empty reservation.
    if str(payload.get("split", "")) == "dev":
        ids = payload.get("question_ids")
        return isinstance(ids, list) and bool(ids) and all(str(item).strip() for item in ids)
    return False


def _baseline_receipt_digest_is_valid(baseline: Mapping[str, Any]) -> bool:
    supplied = baseline.get("selection_receipt_digest")
    if not isinstance(supplied, str) or not supplied.startswith("sha256:"):
        return False
    unsigned = {
        key: value for key, value in baseline.items()
        if key != "selection_receipt_digest"
    }
    return supplied == _digest(unsigned)


# Baseline selection is intentionally a small, deterministic contract.  The
# historical receipt format did not carry these fields, so their absence is
# interpreted as the canonical rule below; newly produced receipts always copy
# the resolved values into the signed payload.
_BASELINE_SELECTION_RULE = "min_normalized_regret"
_BASELINE_TIE_POLICIES = frozenset({"lexicographic", "first_in_file"})
_BASELINE_TIE_TOLERANCE_MAX = 1.0
_BASELINE_COUNT_FIELDS = frozenset({
    "n", "count", "sample_count", "record_count", "question_count",
    "n_questions", "seed_count", "normalized_regret_n",
})
_BASELINE_NUMERIC_FIELDS = frozenset({
    "normalized_regret", "raw_regret", "mean_regret", "regret",
    "utility", "mean_utility", "pairrank", "pair_rank", "pairwise_rank",
})
_BUDGET_INTEGER_FIELDS = frozenset({
    "budget", "max_budget", "budget_limit", "max_optimizer_steps",
    "optimizer_step_cap", "max_steps", "training_steps", "train_steps",
    "max_high_level_turns", "compute_units",
    "token_budget", "flop_budget",
})


def _finite_number(value: Any) -> bool:
    """Return true only for a JSON/YAML numeric scalar with finite value."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _positive_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _field_value(payload: Mapping[str, Any], nested: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a duplicated top-level/nested field, rejecting contradictory copies."""

    in_payload = key in payload
    in_nested = key in nested
    if in_payload and in_nested and payload[key] != nested[key]:
        return _INVALID_FIELD
    if in_payload:
        return payload[key]
    if in_nested:
        return nested[key]
    return default


_INVALID_FIELD = object()


def _candidate_entries(candidates: Any) -> list[tuple[str, Mapping[str, Any]]] | None:
    """Normalize mapping/list candidate formats and validate metric structure."""

    if isinstance(candidates, Mapping):
        raw_entries = list(candidates.items())
    elif isinstance(candidates, list):
        raw_entries = []
        for item in candidates:
            if not isinstance(item, Mapping):
                return None
            name = item.get("method", item.get("name"))
            metric = item.get("metrics", item)
            raw_entries.append((name, metric))
    else:
        return None
    if not raw_entries:
        return None
    entries: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for raw_name, metric in raw_entries:
        if not isinstance(raw_name, str) or not raw_name.strip() or raw_name != raw_name.strip():
            return None
        name = raw_name
        if name in seen or not isinstance(metric, Mapping):
            return None
        normalized_regret = metric.get("normalized_regret")
        if not _finite_number(normalized_regret):
            return None
        # Optional reported metrics must not contain NaN/Inf or malformed values.
        for field in _BASELINE_NUMERIC_FIELDS:
            if field in metric and not _finite_number(metric[field]):
                return None
        # Counts are optional for compatibility with the original compact format,
        # but if present they must be positive integer denominators.
        for field in _BASELINE_COUNT_FIELDS:
            if field in metric and not _positive_integer(metric[field]):
                return None
        if "split" in metric and metric["split"] != "dev":
            return None
        seen.add(name)
        entries.append((name, metric))
    return entries


def _selection_results_validation(
    payload: Mapping[str, Any],
    *,
    selected_baseline: str,
    selection_split: str,
) -> dict[str, Any] | None:
    """Validate candidate metrics and recompute the declared deterministic winner."""

    nested_raw = payload.get("baseline_selection")
    nested = nested_raw if isinstance(nested_raw, Mapping) else {}
    observed_split = _field_value(payload, nested, "selection_split")
    observed_baseline = _field_value(payload, nested, "selected_baseline")
    if observed_split is _INVALID_FIELD or observed_baseline is _INVALID_FIELD:
        return None
    if str(observed_split) != str(selection_split) or str(observed_baseline) != str(selected_baseline):
        return None
    if "candidate_metrics" in payload:
        candidates = payload.get("candidate_metrics")
    elif "records" in payload:
        candidates = payload.get("records")
    else:
        candidates = payload.get("metrics")
    entries = _candidate_entries(candidates)
    if entries is None:
        return None

    rule = _field_value(payload, nested, "selection_rule", _BASELINE_SELECTION_RULE)
    tie_policy = _field_value(payload, nested, "tie_policy", "lexicographic")
    tolerance = _field_value(payload, nested, "tie_tolerance", 0.0)
    if (
        rule is _INVALID_FIELD or tie_policy is _INVALID_FIELD or tolerance is _INVALID_FIELD
        or not isinstance(rule, str) or rule != _BASELINE_SELECTION_RULE
        or not isinstance(tie_policy, str) or tie_policy not in _BASELINE_TIE_POLICIES
        or not _finite_number(tolerance) or float(tolerance) < 0.0
        or float(tolerance) > _BASELINE_TIE_TOLERANCE_MAX
    ):
        return None
    values = [(name, float(metric["normalized_regret"])) for name, metric in entries]
    minimum = min(value for _, value in values)
    tied = [(name, value) for name, value in values if value <= minimum + float(tolerance)]
    if tie_policy == "lexicographic":
        expected = min(name for name, _ in tied)
    else:  # first_in_file; list/dict insertion order is part of the committed file.
        expected = tied[0][0]
    selected = str(selected_baseline)
    if selected not in {name for name, _ in values} or selected != expected:
        return None
    return {
        "selection_rule": rule,
        "tie_policy": tie_policy,
        "tie_tolerance": float(tolerance),
        "expected_baseline": expected,
        "candidate_count": len(entries),
        "minimum_normalized_regret": minimum,
    }


def _load_structured_mapping(path: Path) -> Mapping[str, Any] | None:
    """Load a finite JSON/YAML object without making PyYAML a hard dependency."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value: Any = None
    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (ValueError, TypeError):
        try:
            import yaml  # type: ignore
            value = yaml.safe_load(text)
        except Exception:
            return None
    if not isinstance(value, Mapping):
        return None

    def finite_tree(node: Any) -> bool:
        if isinstance(node, float):
            return math.isfinite(node)
        if isinstance(node, Mapping):
            return all(finite_tree(key) and finite_tree(item) for key, item in node.items())
        if isinstance(node, (list, tuple)):
            return all(finite_tree(item) for item in node)
        return True

    try:
        return value if finite_tree(value) else None
    except RecursionError:
        return None


def _has_valid_budget_field(node: Any) -> bool:
    if isinstance(node, Mapping):
        for key, value in node.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in _BUDGET_INTEGER_FIELDS:
                if isinstance(value, Mapping):
                    if _has_valid_budget_field(value):
                        return True
                elif _positive_integer(value):
                    return True
            if _has_valid_budget_field(value):
                return True
    elif isinstance(node, (list, tuple)):
        return any(_has_valid_budget_field(item) for item in node)
    return False


def _structured_config_has_schema(path: Path, *, kind: str) -> bool:
    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        return False
    payload = _load_structured_mapping(path)
    if payload is None:
        return False
    if "schema_version" in payload and (
        not isinstance(payload["schema_version"], str) or not payload["schema_version"].strip()
    ):
        return False
    try:
        if not _has_valid_budget_field(payload):
            return False
    except RecursionError:
        return False
    if kind == "algorithm":
        sections = ("algorithm", "objective", "branching", "horizon", "methods", "trainer")
    else:
        sections = ("hyperparameters", "trainer", "training", "optimizer", "model")
    # A schema marker does not by itself establish whether this is an algorithm
    # config or a hyperparameter document; require a kind-specific section too.
    try:
        return any(isinstance(payload.get(section), Mapping) for section in sections)
    except RecursionError:
        return False


def _selection_results_have_dev_evidence(
    path: Path,
    *,
    selected_baseline: str,
    selection_split: str,
) -> bool:
    """Validate dev metrics and recompute the selected baseline under the rule."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    return _selection_results_validation(
        payload,
        selected_baseline=selected_baseline,
        selection_split=selection_split,
    ) is not None


def _selection_receipt_semantics_match(path: Path, baseline: Mapping[str, Any]) -> bool:
    """Ensure optional receipt metadata agrees with the bound selection file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    details = _selection_results_validation(
        payload,
        selected_baseline=str(baseline.get("selected_baseline", "")),
        selection_split=str(baseline.get("selection_split", "")),
    )
    if details is None:
        return False
    for field in ("selection_rule", "tie_policy", "tie_tolerance"):
        if field in baseline and baseline[field] != details[field]:
            return False
    return True


def _round(value: float) -> float:
    # Stable decimal representation keeps signatures independent of platform float
    # formatting while preserving the intentionally distinct recipe grid.
    return float(f"{float(value):.12f}")


def latent_signature_payload(world: WorldSpec) -> dict[str, Any]:
    """Return latent parameters without IDs, seeds, or family labels.

    Omitting ``question_family`` is deliberate: a copied latent template must still
    be detected when a developer renames the mechanism family.  Family/recipe
    identity is audited separately through ``generator_signature``.
    """

    return {
        "kind": str(world.kind),
        "true_effect_a": _round(world.true_effect_a),
        "true_effect_b": _round(world.true_effect_b),
        "noise_scale": _round(world.noise_scale),
        "initial_samples": int(world.initial_samples),
        "leakage": bool(world.leakage),
        "confounding": bool(world.confounding),
        "metric_mismatch": bool(world.metric_mismatch),
        "protocol_invalid": bool(world.protocol_invalid),
    }


def latent_signature(world: WorldSpec) -> str:
    """Hash the latent generator output while excluding question/world IDs."""

    return _digest(latent_signature_payload(world))


def recipe_signature(recipe: V05GeneratorRecipe) -> str:
    return _digest(recipe.canonical())


def _recipe_for(family: str) -> V05GeneratorRecipe:
    try:
        return V05_GENERATOR_RECIPES[family]
    except KeyError as exc:
        raise ValueError(f"unregistered v0.5 generator family: {family}") from exc


def _latent_parameters(family: str, variant: int, kind: str) -> dict[str, Any]:
    """Generate one hidden world from the v0.5 recipe.

    The arithmetic intentionally uses a recipe-specific irrational-looking phase
    and non-v0.4 coefficient grid.  No development manifest is consulted while
    generating final worlds.
    """

    recipe = _recipe_for(family)
    family_index = V05_ALL_FAMILIES.index(family)
    c0, c1, c2, c3 = recipe.coefficients
    v = int(variant)
    phase = ((v * 17 + family_index * 29 + (v % 3) * 7) % 97) / 10000.0
    skew = ((v * 11 + family_index * 13) % 23) / 1000.0
    odd = bool(v % 2)
    if kind == "supported":
        true_a = c0 + 0.0041 * (v % 5) + phase
        true_b = c1 / 2.0 + 0.0013 * (v % 4)
        noise, samples = 0.031 + 0.0007 * (v % 4), 112 + 3 * (v % 5)
    elif kind == "refuted":
        true_a = -c0 / 2.0 - 0.0037 * (v % 4) - phase
        true_b = (c1 + 0.037 if odd else -c1 / 2.0) + skew
        noise, samples = 0.037 + 0.0009 * (v % 3), 108 + 2 * (v % 7)
    elif kind == "insufficient":
        true_a = c0 / 3.0 + 0.002 * (v % 3)
        true_b = (c1 + 0.043 if odd else c1 / 3.0) - phase
        noise, samples = 0.247 + 0.009 * (v % 5) + c2 / 10.0, 21 + (v % 4)
    else:
        true_a = 0.004 + 0.001 * (v % 3) + phase / 3.0
        true_b = (c1 + 0.051 if odd else c1 / 4.0) + skew
        noise, samples = 0.043 + 0.0011 * (v % 4), 104 + (v % 6)

    flags = set(recipe.flag_profile)
    # Every invalid world has a protocol-level invalid marker.  Family-specific
    # flags make OOD worlds structurally distinct while preserving the four-state
    # evaluator contract.
    protocol_invalid = kind == "invalid" or "protocol_invalid" in flags
    leakage = "leakage" in flags or (
        kind == "invalid"
        and family in {"selection_collider_v05", "interference_spillover_v05"}
    )
    confounding = "confounding" in flags or (kind == "invalid" and family == "selection_collider_v05")
    metric_mismatch = "metric_mismatch" in flags or (kind == "invalid" and family == "measurement_delay_v05")
    if family == "adaptive_stopping_v05" and kind == "insufficient":
        samples = max(12, samples - 5)
        noise += 0.021
    if family == "interference_spillover_v05" and kind == "refuted":
        true_b += 0.026
    if family == "measurement_delay_v05" and kind == "supported":
        true_a += 0.009
    if family == "selection_collider_v05" and kind == "insufficient":
        noise += 0.017
    return {
        "true_a": _round(true_a),
        "true_b": _round(true_b),
        "noise": _round(noise),
        "samples": int(samples),
        "leakage": bool(leakage),
        "confounding": bool(confounding),
        "metric_mismatch": bool(metric_mismatch),
        "protocol_invalid": bool(protocol_invalid),
    }


def _target_for(family: str, kind: str, variant: int) -> ResearchAction:
    """Evaluator-only target action used for post-hoc regret accounting."""

    if kind == "supported":
        return ResearchAction.CONTINUE
    if kind == "invalid":
        if family in {"subgroup_metric_mismatch", "measurement_delay_v05"}:
            return ResearchAction.SWITCH
        return ResearchAction.REPAIR
    if kind == "insufficient":
        if family in {"causal_confounding", "selection_collider_v05"}:
            return ResearchAction.SWITCH
        return ResearchAction.SAMPLE
    if family in {"causal_confounding", "low_sample_variance", "adaptive_stopping_v05"}:
        return ResearchAction.CONTINUE
    return ResearchAction.SWITCH


@dataclass(frozen=True)
class V05FinalQuestion:
    question_id: str
    family: str
    variant: int
    split: str
    seed_offset: int
    description: str
    worlds: Tuple[WorldSpec, ...]
    target_actions: Mapping[str, ResearchAction]
    generator_signature: str
    recipe_id: str

    @property
    def policy_question_id(self) -> str:
        return "pesco_v05_final_public_question"

    def target_action(self, world_id: str) -> ResearchAction:
        return self.target_actions[world_id]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "policy_question_id": self.policy_question_id,
            "split": self.split,
            "world_count": len(self.worlds),
            "question_commitment": _digest({
                "generator_signature": self.generator_signature,
                "latent_signatures": [latent_signature(world) for world in self.worlds],
            }),
        }

    def to_hidden_dict(self) -> dict[str, Any]:
        return {
            **self.to_public_dict(),
            "family": self.family,
            "variant": int(self.variant),
            "seed_offset": int(self.seed_offset),
            "description": self.description,
            "recipe_id": self.recipe_id,
            "generator_signature": self.generator_signature,
            "latent_signatures": [latent_signature(world) for world in self.worlds],
            "worlds": [asdict(world) for world in self.worlds],
            "legacy_target_actions_audit_only": {
                key: action.value for key, action in self.target_actions.items()
            },
        }


@dataclass(frozen=True)
class V05FrozenFinalBenchmark:
    questions: Tuple[V05FinalQuestion, ...]
    protocol_version: str = "pesco_v0_2"
    schema_version: str = V05_SCHEMA

    def __post_init__(self) -> None:
        if len(self.questions) != 96:
            raise ValueError("v0.5 frozen final requires exactly 96 question clusters")
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("v0.5 question IDs must be unique")
        split_counts = {
            split: sum(question.split == split for question in self.questions)
            for split in V05_SPLITS
        }
        if split_counts != {"final_id": 48, "final_ood": 48}:
            raise ValueError(f"v0.5 split counts must be 48/48, got {split_counts}")
        id_families = {q.family for q in self.questions if q.split == "final_id"}
        ood_families = {q.family for q in self.questions if q.split == "final_ood"}
        if id_families != set(V05_FINAL_ID_FAMILIES):
            raise ValueError("v0.5 final-ID family registry mismatch")
        if ood_families != set(V05_FINAL_OOD_FAMILIES):
            raise ValueError("v0.5 final-OOD family registry mismatch")
        if id_families.intersection(ood_families):
            raise ValueError("final-ID and final-OOD families must be disjoint")
        for question in self.questions:
            if tuple(world.kind for world in question.worlds) != WORLD_KINDS:
                raise ValueError("world order must remain the four-state contract")
            if len({world.world_id for world in question.worlds}) != 4:
                raise ValueError("world IDs must be unique within each question")

    @property
    def worlds(self) -> Tuple[WorldSpec, ...]:
        return tuple(world for question in self.questions for world in question.worlds)

    @property
    def split_names(self) -> Tuple[str, ...]:
        return V05_SPLITS

    @property
    def final_id_questions(self) -> Tuple[V05FinalQuestion, ...]:
        return tuple(q for q in self.questions if q.split == "final_id")

    @property
    def final_ood_questions(self) -> Tuple[V05FinalQuestion, ...]:
        return tuple(q for q in self.questions if q.split == "final_ood")

    def question(self, question_id: str) -> V05FinalQuestion:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        raise KeyError(question_id)

    def make_environment(self, question_id: str, protocol: Optional[Protocol] = None, budget: Optional[int] = None):
        from ..environments.tier1_tabular_env import Tier1TabularEnvironment
        question = self.question(question_id)
        return Tier1TabularEnvironment(
            worlds=question.worlds,
            protocol=protocol or Protocol(protocol_version=self.protocol_version),
            budget=budget,
        )

    def manifest(self, *, include_hidden: bool = False) -> dict[str, Any]:
        questions = [
            question.to_hidden_dict() if include_hidden else question.to_public_dict()
            for question in self.questions
        ]
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "profile": "frozen_final_v0.5",
            "question_count": len(self.questions),
            "world_count": len(self.worlds),
            "counts_by_split": {
                split: sum(question.split == split for question in self.questions)
                for split in V05_SPLITS
            },
            "final_id_cluster_minimum": 40,
            "final_ood_cluster_minimum": 40,
            "generator_version": V05_GENERATOR_VERSION,
            "evaluator_version": V05_EVALUATOR_VERSION,
            "questions": questions,
            "final_access": {
                "locked": True,
                "authorization_required": True,
                "model_evaluation_completed": False,
                "formal_comparison_authorized": False,
            },
            "training_boundary": {
                "baseline_selection": (
                    "must be supplied from a pre-final dev receipt with "
                    "manifest/selection/algorithm/hyperparameter digests"
                ),
                "final_id": "locked_promotion_only",
                "final_ood": "whole_family_holdout_only",
            },
            "hidden_fields_in_public_manifest": False,
        }
        if include_hidden:
            payload.update({
                "mechanism_families": list(V05_ALL_FAMILIES),
                "final_id_families": list(V05_FINAL_ID_FAMILIES),
                "final_ood_families": list(V05_FINAL_OOD_FAMILIES),
                "generator_recipes": {
                    family: recipe.canonical()
                    for family, recipe in V05_GENERATOR_RECIPES.items()
                },
                "latent_signature_definition": list(latent_signature_payload(self.worlds[0]).keys()),
            })
        payload["manifest_digest"] = _digest(payload)
        return payload


def build_tier1_v05_frozen_final_benchmark() -> V05FrozenFinalBenchmark:
    """Create 48 ID + 48 whole-family OOD clusters from the new generator."""

    questions: list[V05FinalQuestion] = []
    ordinal = 0
    registry = [
        (family, "final_id", variant)
        for family in V05_FINAL_ID_FAMILIES
        for variant in range(1, 9)
    ] + [
        (family, "final_ood", variant)
        for family in V05_FINAL_OOD_FAMILIES
        for variant in range(1, 13)
    ]
    for family, split, variant in registry:
        ordinal += 1
        question_id = f"v05f_q_{ordinal:04d}"
        recipe = _recipe_for(family)
        seed_offset = 700_000 + ordinal * 131 + V05_ALL_FAMILIES.index(family) * 17
        worlds: list[WorldSpec] = []
        targets: dict[str, ResearchAction] = {}
        for kind_index, kind in enumerate(WORLD_KINDS):
            params = _latent_parameters(family, variant, kind)
            world = WorldSpec(
                world_id=f"{question_id}__w{kind_index + 1}",
                kind=kind,
                true_effect_a=params["true_a"],
                true_effect_b=params["true_b"],
                noise_scale=params["noise"],
                initial_samples=params["samples"],
                leakage=params["leakage"],
                confounding=params["confounding"],
                metric_mismatch=params["metric_mismatch"],
                protocol_invalid=params["protocol_invalid"],
                seed_offset=seed_offset + kind_index,
                question_family=family,
                # Mechanism identity remains evaluator-owned.  In particular, the
                # final OOD family names must not reach ``Observation.task_family``
                # when an independently packaged model evaluator runs this profile.
                public_task_family="v05_final_public_task",
            )
            worlds.append(world)
            targets[world.world_id] = _target_for(family, kind, variant)
        generator_signature = _digest({
            "generator_version": V05_GENERATOR_VERSION,
            "recipe": recipe.canonical(),
            "variant": int(variant),
            "split": split,
            "latent_signatures": [latent_signature(world) for world in worlds],
        })
        questions.append(V05FinalQuestion(
            question_id=question_id,
            family=family,
            variant=variant,
            split=split,
            seed_offset=seed_offset,
            description=f"v0.5 frozen {split} cluster {ordinal:04d}; mechanism held by evaluator.",
            worlds=tuple(worlds),
            target_actions=MappingProxyType(targets),
            generator_signature=generator_signature,
            recipe_id=recipe.recipe_id,
        ))
    return V05FrozenFinalBenchmark(tuple(questions))


build_tier1_v05_formal_final_benchmark = build_tier1_v05_frozen_final_benchmark


def _manifest_worlds(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for item in manifest.get("questions", []):
        for world in item.get("worlds", []):
            if isinstance(world, Mapping):
                yield world


def audit_latent_generator_signatures(
    benchmark: V05FrozenFinalBenchmark,
    development_manifests: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Audit IDs *and* latent outputs against all supplied development manifests."""

    final_question_ids = {question.question_id for question in benchmark.questions}
    final_world_ids = {world.world_id for world in benchmark.worlds}
    final_latent = {latent_signature(world) for world in benchmark.worlds}
    final_generator = {question.generator_signature for question in benchmark.questions}
    final_ood_families = set(V05_FINAL_OOD_FAMILIES)

    development_question_ids: set[str] = set()
    development_world_ids: set[str] = set()
    development_latent: set[str] = set()
    development_generator: set[str] = set()
    development_families: set[str] = set()
    for manifest in development_manifests:
        for item in manifest.get("questions", []):
            if not isinstance(item, Mapping):
                continue
            if item.get("question_id") is not None:
                development_question_ids.add(str(item["question_id"]))
            if item.get("family") is not None:
                development_families.add(str(item["family"]))
            if item.get("generator_signature") is not None:
                development_generator.add(str(item["generator_signature"]))
            # Legacy v0.4 manifests have no generator signature.  Derive a stable
            # legacy marker from their recipe identity so the audit remains explicit.
            elif item.get("family") is not None:
                development_generator.add(_digest({
                    "legacy_generator": "v04_family_flags",
                    "family": str(item["family"]),
                    "variant": item.get("variant"),
                }))
            for world in item.get("worlds", []):
                if not isinstance(world, Mapping):
                    continue
                if world.get("world_id") is not None:
                    development_world_ids.add(str(world["world_id"]))
                latent_payload = {
                    key: world[key]
                    for key in (
                        "kind", "true_effect_a", "true_effect_b", "noise_scale",
                        "initial_samples", "leakage", "confounding",
                        "metric_mismatch", "protocol_invalid",
                    )
                    if key in world
                }
                if len(latent_payload) == 9:
                    development_latent.add(_digest(latent_payload))

    id_families = {q.family for q in benchmark.final_id_questions}
    ood_families = {q.family for q in benchmark.final_ood_questions}
    known_development_families = set(development_families).union(V04_DEVELOPMENT_FAMILIES)
    gates = {
        # A zero-count audit is not evidence of non-overlap.  Final signing must
        # receive at least one hidden development manifest containing latent worlds;
        # the known v0.4 family registry is only a conservative fallback for the OOD
        # family-name check.
        "development_manifest_evidence_available": bool(development_manifests),
        "development_latent_evidence_available": bool(development_latent),
        "development_generator_evidence_available": bool(development_generator),
        "question_ids_disjoint": not final_question_ids.intersection(development_question_ids),
        "world_ids_disjoint": not final_world_ids.intersection(development_world_ids),
        "latent_outputs_disjoint": not final_latent.intersection(development_latent),
        "generator_signatures_disjoint": not final_generator.intersection(development_generator),
        "ood_family_names_unseen": not ood_families.intersection(known_development_families),
        "ood_has_at_least_three_new_families": len(ood_families - known_development_families) >= 3,
        "id_ood_families_disjoint": not id_families.intersection(ood_families),
        "all_final_clusters_have_generator_signature": all(bool(q.generator_signature) for q in benchmark.questions),
        "all_worlds_have_latent_signature": len(final_latent) == len(final_world_ids),
    }
    return {
        "schema_version": "pesco_v05_latent_signature_audit_v0.1",
        "generator_version": V05_GENERATOR_VERSION,
        "development_manifest_count": len(development_manifests),
        "final_question_count": len(final_question_ids),
        "final_world_count": len(final_world_ids),
        "final_latent_signature_count": len(final_latent),
        "final_generator_signature_count": len(final_generator),
        "development_question_count": len(development_question_ids),
        "development_world_count": len(development_world_ids),
        "development_latent_signature_count": len(development_latent),
        "development_generator_signature_count": len(development_generator),
        "question_id_overlap_count": len(final_question_ids.intersection(development_question_ids)),
        "world_id_overlap_count": len(final_world_ids.intersection(development_world_ids)),
        "latent_overlap_count": len(final_latent.intersection(development_latent)),
        "generator_signature_overlap_count": len(final_generator.intersection(development_generator)),
        "development_families": sorted(known_development_families),
        "final_ood_families": sorted(ood_families),
        "new_ood_family_count": len(ood_families - known_development_families),
        "gates": gates,
        "pass": bool(all(gates.values())),
    }


def audit_public_manifest(public_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that the public side does not contain evaluator-owned fields."""

    forbidden_tokens = (
        "true_effect_a", "true_effect_b", "noise_scale", "initial_samples",
        "world_id", "legacy_target_actions_audit_only", "target_actions",
        "mechanism_family",
    )
    encoded = json.dumps(public_manifest, sort_keys=True, ensure_ascii=False)
    # Do not reject ordinary prose such as ``whole_family_holdout_only``.  Audit
    # evaluator leakage by inspecting actual JSON keys and explicit hidden payload
    # markers, rather than a substring search for the word ``family``.
    def _keys(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key)
                yield from _keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from _keys(child)

    key_set = set(_keys(public_manifest))
    forbidden_key_names = {
        "family", "mechanism_families", "final_id_families", "final_ood_families",
        "worlds", "latent_signatures", "generator_recipes",
        "legacy_target_actions_audit_only", "target_actions",
    }
    question_fields_ok = all(
        set(item).issubset({"question_id", "policy_question_id", "split", "world_count", "question_commitment"})
        for item in public_manifest.get("questions", [])
        if isinstance(item, Mapping)
    )
    gates = {
        "locked": public_manifest.get("final_access", {}).get("locked") is True,
        "no_forbidden_tokens": not any(token in encoded for token in forbidden_tokens)
        and not key_set.intersection(forbidden_key_names),
        "question_fields_opaque": question_fields_ok,
        "manifest_digest_present": str(public_manifest.get("manifest_digest", "")).startswith("sha256:"),
    }
    return {"schema_version": "pesco_v05_public_manifest_audit_v0.1", "gates": gates, "pass": bool(all(gates.values()))}


def _git_tag(repo_root: Path) -> Optional[str]:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "describe", "--exact-match", "--tags", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_freeze_receipt(
    *,
    repo_root: str | Path,
    benchmark: V05FrozenFinalBenchmark,
    signature_audit: Mapping[str, Any],
    public_audit: Mapping[str, Any],
    baseline_selection: Optional[Mapping[str, Any]] = None,
    explicit_sign: bool = False,
) -> dict[str, Any]:
    """Create a fail-closed freeze receipt.

    The repository must have a clean, tagged HEAD *before* the receipt is signed.
    The baseline receipt must bind the development manifest, candidate-metrics
    selection results, algorithm configuration, and hyperparameter digests; its
    canonical receipt digest and file hashes are recomputed here.  A bare method
    name or invented ``sha256:`` strings are not evidence that selection happened
    before final access.
    Writing generated artifacts necessarily dirties a tracked worktree in some
    installations, so callers should generate into an external evaluator volume or
    sign from a clean commit in CI.  We never infer cleanliness from a missing status.
    """

    root = Path(repo_root).resolve()
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10, env=env,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True, capture_output=True, text=True, timeout=5, env=env,
        ).stdout
        status_available = True
    except (OSError, subprocess.SubprocessError):
        sha, status, status_available = None, None, False
    clean = bool(status_available and status == "")
    tag = _git_tag(root)
    baseline = dict(baseline_selection or {})
    # A receipt is only meaningful when its canonical digest and the files it
    # claims to bind are still present.  Merely checking for ``sha256:`` prefixes
    # would allow a caller to sign with invented digests or an empty placeholder
    # development manifest.
    baseline_evidence = baseline.get("selection_evidence")
    evidence_paths = (
        baseline_evidence if isinstance(baseline_evidence, Mapping) else {}
    )
    development_path = _resolve_evidence_path(
        evidence_paths.get("development_manifest_path"), repo_root=root,
    )
    algorithm_path = _resolve_evidence_path(
        evidence_paths.get("algorithm_config_path"), repo_root=root,
    )
    hyperparameters_path = _resolve_evidence_path(
        evidence_paths.get("hyperparameters_path"), repo_root=root,
    )
    selection_results_path = _resolve_evidence_path(
        evidence_paths.get("selection_results_path"), repo_root=root,
    )
    evidence_files_ok = bool(
        development_path
        and algorithm_path
        and hyperparameters_path
        and selection_results_path
        and _development_manifest_has_observed_dev_data(development_path)
        and _structured_config_has_schema(algorithm_path, kind="algorithm")
        and _structured_config_has_schema(hyperparameters_path, kind="hyperparameters")
        and _selection_results_have_dev_evidence(
            selection_results_path,
            selected_baseline=str(baseline.get("selected_baseline", "")),
            selection_split=str(baseline.get("selection_split", "")),
        )
        and _selection_receipt_semantics_match(selection_results_path, baseline)
        and _digest_file(development_path) == baseline.get("development_manifest_digest")
        and _digest_file(algorithm_path) == baseline.get("algorithm_config_digest")
        and _digest_file(hyperparameters_path) == baseline.get("hyperparameters_digest")
        and _digest_file(selection_results_path) == baseline.get("selection_results_digest")
    )
    baseline_ok = bool(
        baseline.get("selected_baseline")
        # v0.5 final selection is allowed only on the untouched development
        # split.  Accepting promotion here would let a caller choose the
        # baseline after opening final data, defeating the freeze contract.
        and baseline.get("selection_split") == "dev"
        and baseline.get("selection_locked_before_final") is True
        and str(baseline.get("development_manifest_digest", "")).startswith("sha256:")
        and str(baseline.get("selection_receipt_digest", "")).startswith("sha256:")
        and str(baseline.get("algorithm_config_digest", "")).startswith("sha256:")
        and str(baseline.get("hyperparameters_digest", "")).startswith("sha256:")
        and baseline.get("algorithm_hyperparameters_frozen") is True
        and _baseline_receipt_digest_is_valid(baseline)
        and evidence_files_ok
    )
    audit_ok = bool(signature_audit.get("pass") is True and public_audit.get("pass") is True)
    signable = bool(clean and tag and baseline_ok and audit_ok and explicit_sign)
    payload: dict[str, Any] = {
        "schema_version": "pesco_v05_clean_freeze_receipt_v0.1",
        "generator_version": V05_GENERATOR_VERSION,
        "evaluator_version": V05_EVALUATOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen" if signable else "pending_clean_commit_tag",
        "signed": signable,
        "signature_algorithm": "sha256-canonical-json",
        "git": {
            "head": sha,
            "tag_at_head": tag,
            "status_available": status_available,
            "clean_before_receipt": clean,
        },
        "baseline_selection": baseline,
        "baseline_selection_valid": baseline_ok,
        "baseline_evidence_files_valid": evidence_files_ok,
        # Do not echo an unverified claim from a malformed receipt.  The frozen
        # flag is true only after the complete baseline receipt has passed all
        # digest/file/content checks above.
        "algorithm_hyperparameters_frozen": baseline_ok,
        "signature_audit_pass": bool(signature_audit.get("pass") is True),
        "public_manifest_audit_pass": bool(public_audit.get("pass") is True),
        "final_cluster_counts": {
            "final_id": len(benchmark.final_id_questions),
            "final_ood": len(benchmark.final_ood_questions),
        },
        "model_evaluation_completed": False,
        "formal_comparison_authorized": False,
    }
    payload["receipt_digest"] = _digest(payload)
    return payload


def build_baseline_selection_receipt(
    *,
    selected_baseline: str,
    selection_split: str,
    development_manifest: str | Path,
    algorithm_config: str | Path,
    hyperparameters: str | Path,
    selection_results: str | Path | None = None,
    selection_evidence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create the strict pre-final baseline receipt consumed by ``build_freeze_receipt``.

    The helper binds baseline choice to the exact development manifest, candidate
    metrics/selection-results file, and frozen algorithm/hyperparameter files.  It
    intentionally does not inspect final data, so a caller cannot select a baseline
    after opening the final evaluator.
    """

    if not str(selected_baseline).strip():
        raise ValueError("selected_baseline must be non-empty")
    if selection_split != "dev":
        raise ValueError("selection_split must be dev (promotion selection is not a valid v0.5 freeze receipt)")
    if selection_results is None:
        raise ValueError(
            "selection_results is required so the chosen baseline is bound to actual dev metrics"
        )
    paths = {
        "development_manifest_digest": Path(development_manifest),
        "algorithm_config_digest": Path(algorithm_config),
        "hyperparameters_digest": Path(hyperparameters),
        "selection_results_digest": Path(selection_results),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("baseline receipt input missing: " + ", ".join(missing))
    development_path = Path(development_manifest)
    if not _development_manifest_has_observed_dev_data(development_path):
        raise ValueError(
            "development_manifest must contain non-empty observed dev question/world data"
        )
    algorithm_path = Path(algorithm_config)
    hyperparameters_path = Path(hyperparameters)
    if not _structured_config_has_schema(algorithm_path, kind="algorithm"):
        raise ValueError(
            "algorithm_config must be a JSON/YAML object with a recognizable schema and positive budget field"
        )
    if not _structured_config_has_schema(hyperparameters_path, kind="hyperparameters"):
        raise ValueError(
            "hyperparameters must be a JSON/YAML object with a recognizable schema and positive budget field"
        )
    selection_path = Path(selection_results)
    try:
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selection_results must be valid JSON") from exc
    if not isinstance(selection_payload, Mapping):
        raise ValueError("selection_results must be a JSON object")
    selection_validation = _selection_results_validation(
        selection_payload,
        selected_baseline=str(selected_baseline),
        selection_split=str(selection_split),
    )
    if selection_validation is None:
        raise ValueError(
            "selection_results must contain well-formed dev candidate metrics and select the declared baseline under min_normalized_regret"
        )
    evidence = dict(selection_evidence or {})
    # The paths are part of the signed evidence.  Without them a later freeze
    # cannot recompute the file digests and would have to trust bare hash strings.
    evidence.setdefault("development_manifest_path", str(development_path))
    evidence.setdefault("algorithm_config_path", str(Path(algorithm_config)))
    evidence.setdefault("hyperparameters_path", str(Path(hyperparameters)))
    evidence.setdefault("selection_results_path", str(Path(selection_results)))
    payload: dict[str, Any] = {
        "schema_version": "pesco_v05_baseline_selection_receipt_v0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_baseline": str(selected_baseline),
        "selection_split": str(selection_split),
        # Resolve legacy selection files to explicit signed semantics.  A caller
        # cannot later reinterpret a tie or swap the optimization direction.
        "selection_rule": selection_validation["selection_rule"],
        "tie_policy": selection_validation["tie_policy"],
        "tie_tolerance": selection_validation["tie_tolerance"],
        "selection_locked_before_final": True,
        "algorithm_hyperparameters_frozen": True,
        **{name: _digest_file(path) for name, path in paths.items()},
        "selection_evidence": evidence,
    }
    payload["selection_receipt_digest"] = _digest(payload)
    return payload


def collect_v05_environment_receipts(
    benchmark: V05FrozenFinalBenchmark,
    protocol: Optional[Protocol] = None,
    *,
    question_limit: Optional[int] = None,
) -> dict[str, Any]:
    """Run evaluator-only CPU receipts for a bounded or complete final profile.

    This is not a model comparison and does not open final access.  It records one
    independent transition/confirmation receipt per action and exploration seed in a
    caller-controlled evaluator directory.
    """

    protocol = protocol or Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V05_EXPLORATION_SEEDS,
        confirmation_seeds=V05_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    questions = benchmark.questions if question_limit is None else benchmark.questions[: max(0, int(question_limit))]
    verifier = TrustedVerifier(protocol)
    rows: list[dict[str, Any]] = []
    for question in questions:
        for world in question.worlds:
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            env.reset(question.policy_question_id, world.world_id, seed=17)
            baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            initial_observation = env.visible_observation()
            snapshot = env.snapshot()
            for action in ACTION_SET:
                for index, seed in enumerate(protocol.exploration_seeds):
                    branch = env.clone_from_snapshot(snapshot)
                    output = branch.execute_option(action, seeds=(int(seed),))
                    verdict = verifier.evaluate(output, branch, confirm=False)
                    confirmation = None
                    confirmation_verdict = None
                    passed = None
                    if (
                        verdict.evidence_state in {EvidenceState.SUPPORTED, EvidenceState.REFUTED}
                        and verdict.validity_pass
                    ):
                        confirmation_seed = int(protocol.confirmation_seeds[index % len(protocol.confirmation_seeds)])
                        candidate = branch.clone_from_snapshot(branch.snapshot())
                        confirmation = candidate._simulate(  # noqa: SLF001 - evaluator boundary
                            method=output.method,
                            option=action,
                            seeds=(confirmation_seed,),
                            confirmation=True,
                        )
                        confirmation_verdict = verifier.evaluate(confirmation, candidate, confirm=False)
                        passed = bool(
                            confirmation_verdict.validity_pass
                            and confirmation_verdict.evidence_state is verdict.evidence_state
                            and confirmation.dataset_hash != output.dataset_hash
                            and confirmation.split_hash != output.split_hash
                        )
                    if passed is not None:
                        from dataclasses import replace
                        verdict_for_utility = replace(
                            verdict,
                            independent_confirmation_performed=True,
                            independent_confirmation_passed=bool(passed),
                        )
                    else:
                        verdict_for_utility = verdict
                    components = tier1_scientific_utility_components(
                        question, world, action, output, verdict_for_utility,
                        protocol, initial_observation=initial_observation,
                    )
                    rows.append({
                        "question_id_audit": question.question_id,
                        "world_id_audit": world.world_id,
                        "split": question.split,
                        "action": action.value,
                        "exploration_seed": int(seed),
                        "confirmation_seed": int(protocol.confirmation_seeds[index % len(protocol.confirmation_seeds)]),
                        "confirmation_eligible": passed is not None,
                        "confirmation_passed": passed,
                        "state": verdict.evidence_state.value,
                        "validity_pass": bool(verdict.validity_pass),
                        "utility": float(sum(components.values())),
                        "reward_components": {name: float(components[name]) for name in TIER1_REWARD_COMPONENT_NAMES},
                        "dataset_hash": output.dataset_hash,
                        "split_hash": output.split_hash,
                        "confirmation_dataset_hash": getattr(confirmation, "dataset_hash", None),
                        "confirmation_split_hash": getattr(confirmation, "split_hash", None),
                    })
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in V05_SPLITS
    }
    payload = {
        "schema_version": "pesco_v05_environment_receipts_v0.1",
        "generator_version": V05_GENERATOR_VERSION,
        "question_limit": question_limit,
        "question_count_collected": len(questions),
        "world_count_collected": len(questions) * 4,
        "action_seed_receipt_count": len(rows),
        "receipt_counts_by_split": counts,
        "independent_confirmation_not_copied": True,
        "model_evaluation_completed": False,
        "formal_comparison_authorized": False,
        "rows": rows,
    }
    payload["receipt_digest"] = _digest({key: value for key, value in payload.items() if key != "receipt_digest"})
    return payload


__all__ = [
    "V05_SCHEMA", "V05_GENERATOR_VERSION", "V05_EVALUATOR_VERSION",
    "V05_SPLITS", "V05_EXPLORATION_SEEDS", "V05_CONFIRMATION_SEEDS",
    "V05_FINAL_ID_FAMILIES", "V05_FINAL_OOD_FAMILIES", "V05_ALL_FAMILIES",
    "V05_FORMAL_FINAL_ID_FAMILIES", "V05_FORMAL_FINAL_OOD_FAMILIES",
    "V05GeneratorRecipe", "V05FinalQuestion", "V05FrozenFinalBenchmark",
    "build_tier1_v05_frozen_final_benchmark", "build_tier1_v05_formal_final_benchmark",
    "latent_signature_payload",
    "latent_signature", "recipe_signature", "audit_latent_generator_signatures",
    "audit_public_manifest", "build_freeze_receipt",
    "build_baseline_selection_receipt",
    "collect_v05_environment_receipts",
]

#!/usr/bin/env python3
"""Audit an already-prepared v0.5 final without regenerating it.

This is intentionally a second, dependency-light evaluator boundary.  It reads the
private hidden bundle and public commitments, verifies canonical digests, recomputes
question/world commitments from the hidden data, independently rechecks the
whole-family OOD novelty gate, and checks that no evaluator-owned fields crossed into
the public bundle.  It never calls the v0.5 generator, so a generator bug cannot make
its own audit pass by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import math
from typing import Any, Mapping


# Keep the independent evaluator free of the generator implementation.  These are
# the development-family names registered before the v0.5 final was generated;
# family novelty is recomputed here instead of trusting a producer-side summary.
KNOWN_DEVELOPMENT_FAMILIES = frozenset({
    "group_leakage",
    "causal_confounding",
    "low_sample_variance",
    "subgroup_metric_mismatch",
    "confounding_underpower",
    "leakage_metric_mismatch",
    "protocol_drift",
    "replication_instability",
})
EXPECTED_ACTIONS = frozenset({
    "continue_current_method",
    "add_samples_or_seeds",
    "repair_data_split",
    "switch_to_alternative_method",
})
EXPECTED_EXPLORATION_SEEDS = frozenset({17, 29, 41, 53, 67, 71, 83, 97})

# Keep this small selection contract independent from the producer-side module.
# Missing fields retain the legacy interpretation (minimum normalized regret,
# deterministic lexicographic tie break), while malformed explicit fields fail
# closed.
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
EXPECTED_CONFIRMATION_SEEDS = frozenset({103, 107, 109, 113, 127, 131, 137, 139})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _question_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return only well-formed question objects; malformed entries audit as absent."""

    result: dict[str, Mapping[str, Any]] = {}
    questions = payload.get("questions", [])
    if not isinstance(questions, list):
        return result
    for item in questions:
        if not isinstance(item, Mapping) or item.get("question_id") is None:
            continue
        result[str(item["question_id"])] = item
    return result


def _question_ids_are_unique(payload: Mapping[str, Any], question_map: Mapping[str, Any]) -> bool:
    questions = payload.get("questions", [])
    return (
        isinstance(questions, list)
        and all(isinstance(item, Mapping) and item.get("question_id") is not None for item in questions)
        and len(question_map) == len(questions)
    )


def _valid_digest(payload: Mapping[str, Any], field: str) -> bool:
    supplied = payload.get(field)
    if not isinstance(supplied, str) or not supplied.startswith("sha256:"):
        return False
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return supplied == _digest(unsigned)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_receipt_path(value: Any, *, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    try:
        path = path.resolve()
    except OSError:
        return None
    return path if path.is_file() else None


def _development_manifest_has_data(path: Path) -> bool:
    """Reject reserved/empty dev placeholders in a purported freeze receipt."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    questions = payload.get("questions")
    if isinstance(questions, list):
        candidates = [
            item for item in questions
            if isinstance(item, Mapping) and str(item.get("split", "")) == "dev"
        ]
        if not candidates and str(payload.get("split", "")) == "dev":
            candidates = questions
        for item in candidates:
            if not str(item.get("question_id", "")).strip():
                continue
            worlds = item.get("worlds")
            if isinstance(worlds, list) and worlds:
                return True
            try:
                if int(item.get("world_count", 0)) > 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                pass
        return False
    ids = payload.get("question_ids")
    return (
        str(payload.get("split", "")) == "dev"
        and isinstance(ids, list)
        and bool(ids)
        and all(str(item).strip() for item in ids)
    )


_INVALID_FIELD = object()


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _field_value(payload: Mapping[str, Any], nested: Mapping[str, Any], key: str, default: Any = None) -> Any:
    in_payload = key in payload
    in_nested = key in nested
    if in_payload and in_nested and payload[key] != nested[key]:
        return _INVALID_FIELD
    if in_payload:
        return payload[key]
    if in_nested:
        return nested[key]
    return default


def _candidate_entries(candidates: Any) -> list[tuple[str, Mapping[str, Any]]] | None:
    if isinstance(candidates, Mapping):
        raw_entries = list(candidates.items())
    elif isinstance(candidates, list):
        raw_entries = []
        for item in candidates:
            if not isinstance(item, Mapping):
                return None
            raw_entries.append((item.get("method", item.get("name")), item.get("metrics", item)))
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
        if not _finite_number(metric.get("normalized_regret")):
            return None
        for field in _BASELINE_NUMERIC_FIELDS:
            if field in metric and not _finite_number(metric[field]):
                return None
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
    expected = (
        min(name for name, _ in tied)
        if tie_policy == "lexicographic"
        else tied[0][0]
    )
    if str(selected_baseline) not in {name for name, _ in values} or str(selected_baseline) != expected:
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
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value: Any = json.loads(text, parse_constant=reject_constant)
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
    sections = (
        ("algorithm", "objective", "branching", "horizon", "methods", "trainer")
        if kind == "algorithm"
        else ("hyperparameters", "trainer", "training", "optimizer", "model")
    )
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
    """Require concrete dev candidate metrics for an independently checked receipt.

    This mirrors the producer-side freeze check without importing the generator
    module.  A file that merely names a baseline, or contains metrics for another
    split/candidate, must not make a signed receipt appear valid.
    """

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
    return all(
        field not in baseline or baseline[field] == details[field]
        for field in ("selection_rule", "tie_policy", "tie_tolerance")
    )


def _baseline_receipt_valid(receipt: Mapping[str, Any], *, repo_root: Path) -> bool:
    baseline = receipt.get("baseline_selection")
    if not isinstance(baseline, Mapping) or not baseline:
        # An empty baseline is allowed only for the explicitly pending structural
        # receipt.  A tampered ``frozen``/signed-looking receipt cannot hide behind
        # the unsigned-placeholder exception.
        return (
            receipt.get("signed") is not True
            and receipt.get("status") in {None, "pending_clean_commit_tag"}
        )
    if not str(baseline.get("selected_baseline", "")).strip():
        return False
    supplied = baseline.get("selection_receipt_digest")
    if not isinstance(supplied, str) or not supplied.startswith("sha256:"):
        return False
    unsigned = {key: value for key, value in baseline.items() if key != "selection_receipt_digest"}
    if supplied != _digest(unsigned):
        return False
    if baseline.get("selection_split") != "dev" or baseline.get("selection_locked_before_final") is not True:
        return False
    if baseline.get("algorithm_hyperparameters_frozen") is not True:
        return False
    evidence = baseline.get("selection_evidence")
    if not isinstance(evidence, Mapping):
        return False
    paths = {
        "development_manifest_digest": _resolve_receipt_path(
            evidence.get("development_manifest_path"), repo_root=repo_root,
        ),
        "algorithm_config_digest": _resolve_receipt_path(
            evidence.get("algorithm_config_path"), repo_root=repo_root,
        ),
        "hyperparameters_digest": _resolve_receipt_path(
            evidence.get("hyperparameters_path"), repo_root=repo_root,
        ),
        "selection_results_digest": _resolve_receipt_path(
            evidence.get("selection_results_path"), repo_root=repo_root,
        ),
    }
    if any(path is None for path in paths.values()):
        return False
    dev_path = paths["development_manifest_digest"]
    assert dev_path is not None
    if not _development_manifest_has_data(dev_path):
        return False
    algorithm_path = paths["algorithm_config_digest"]
    hyperparameters_path = paths["hyperparameters_digest"]
    assert algorithm_path is not None and hyperparameters_path is not None
    if not _structured_config_has_schema(algorithm_path, kind="algorithm"):
        return False
    if not _structured_config_has_schema(hyperparameters_path, kind="hyperparameters"):
        return False
    selection_path = paths["selection_results_digest"]
    assert selection_path is not None
    if not _selection_results_have_dev_evidence(
        selection_path,
        selected_baseline=str(baseline.get("selected_baseline", "")),
        selection_split=str(baseline.get("selection_split", "")),
    ):
        return False
    if not _selection_receipt_semantics_match(selection_path, baseline):
        return False
    return all(
        _file_digest(path) == baseline.get(field)
        for field, path in paths.items()
        if path is not None
    )


def _inventory_matches(entries: Any, root: Path) -> bool:
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        path = root / str(entry.get("path", ""))
        if not path.is_file():
            return False
        if int(entry.get("size_bytes", -1)) != path.stat().st_size:
            return False
        if entry.get("sha256") != _file_digest(path):
            return False
    return True


def _world_latent_payload(world: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(world["kind"]),
        "true_effect_a": float(world["true_effect_a"]),
        "true_effect_b": float(world["true_effect_b"]),
        "noise_scale": float(world["noise_scale"]),
        "initial_samples": int(world["initial_samples"]),
        "leakage": bool(world["leakage"]),
        "confounding": bool(world["confounding"]),
        "metric_mismatch": bool(world["metric_mismatch"]),
        "protocol_invalid": bool(world["protocol_invalid"]),
    }


def _safe_int(value: Any) -> int | None:
    """Parse an integer audit field without turning malformed input into a crash."""

    try:
        # Reject booleans and lossy floating-point values in identifier/seed fields.
        if isinstance(value, bool):
            return None
        parsed = int(value)
        if isinstance(value, float) and float(parsed) != value:
            return None
        return parsed
    except (TypeError, ValueError, OverflowError):
        return None


def _field_is_int(value: Any, expected: int) -> bool:
    parsed = _safe_int(value)
    return parsed is not None and parsed == int(expected)


def _environment_rows_are_well_bound(
    rows: list[Any],
    hidden_questions: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Check that every private receipt row binds to a hidden world/action/seed.

    The older count-only check could be satisfied by replacing rows with unrelated
    IDs while retaining the expected 12,288-row shape.  This independent check
    validates the row identity and rejects duplicate receipt keys fail-closed.
    """

    expected_worlds: dict[tuple[str, str], str] = {}
    for question_id, question in hidden_questions.items():
        split = str(question.get("split", ""))
        worlds = question.get("worlds", [])
        if not isinstance(worlds, list):
            return False
        for world in worlds:
            if not isinstance(world, Mapping):
                return False
            world_id = str(world.get("world_id", ""))
            if not world_id:
                return False
            expected_worlds[(question_id, world_id)] = split

    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        question_id = str(row.get("question_id_audit", ""))
        world_id = str(row.get("world_id_audit", ""))
        action = str(row.get("action", ""))
        seed = _safe_int(row.get("exploration_seed"))
        if (question_id, world_id) not in expected_worlds:
            return False
        if str(row.get("split", "")) != expected_worlds[(question_id, world_id)]:
            return False
        if action not in EXPECTED_ACTIONS or seed not in EXPECTED_EXPLORATION_SEEDS:
            return False
        key = (question_id, world_id, action, seed)
        if key in seen:
            return False
        seen.add(key)

    expected_keys = {
        (question_id, world_id, action, seed)
        for (question_id, world_id), _split in expected_worlds.items()
        for action in EXPECTED_ACTIONS
        for seed in EXPECTED_EXPLORATION_SEEDS
    }
    return seen == expected_keys


def _environment_rows_have_atomic_utilities(rows: list[Any]) -> bool:
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                return False
            utility = float(row["utility"])
            components = row["reward_components"]
            if not isinstance(components, Mapping):
                return False
            component_sum = sum(float(value) for value in components.values())
            if not math.isclose(utility, component_sum, rel_tol=0.0, abs_tol=1e-12):
                return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _is_sha256_digest(value: Any) -> bool:
    """Return whether ``value`` is a canonical sha256 receipt identifier."""

    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    encoded = value.removeprefix("sha256:")
    if len(encoded) != 64 or any(char not in "0123456789abcdef" for char in encoded):
        return False
    return True


def _environment_confirmation_receipt_checks(rows: list[Any]) -> dict[str, bool]:
    """Independently validate confirmation partition semantics for private rows.

    Producer-side ``confirmation_eligible`` and
    ``independent_confirmation_not_copied`` flags are claims, not evidence.  The
    independent auditor therefore reconstructs eligibility from the exploration
    verdict and checks the underlying confirmation receipts directly.
    """

    checks = {
        "seeds_preregistered": bool(rows),
        "eligible_hashes_independent": bool(rows),
        "eligibility_semantics_consistent": bool(rows),
    }
    required_fields = {
        "confirmation_seed",
        "confirmation_eligible",
        "confirmation_passed",
        "confirmation_dataset_hash",
        "confirmation_split_hash",
        "dataset_hash",
        "split_hash",
        "state",
        "validity_pass",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            return {name: False for name in checks}

        confirmation_seed = _safe_int(row.get("confirmation_seed"))
        if confirmation_seed not in EXPECTED_CONFIRMATION_SEEDS:
            checks["seeds_preregistered"] = False

        if not required_fields.issubset(row):
            checks["eligibility_semantics_consistent"] = False

        eligible_value = row.get("confirmation_eligible")
        eligible_is_bool = isinstance(eligible_value, bool)
        state = str(row.get("state", ""))
        validity_value = row.get("validity_pass")
        expected_eligible = (
            validity_value is True
            and state in {"supported", "refuted"}
        )
        if (
            not eligible_is_bool
            or not isinstance(validity_value, bool)
            or eligible_value is not expected_eligible
        ):
            checks["eligibility_semantics_consistent"] = False

        if eligible_value is True:
            # An attempted confirmation has a Boolean outcome even when it
            # fails.  Both confirmation hashes must be real receipts and must
            # identify partitions distinct from the exploration output.
            if not isinstance(row.get("confirmation_passed"), bool):
                checks["eligibility_semantics_consistent"] = False
            exploration_dataset = row.get("dataset_hash")
            exploration_split = row.get("split_hash")
            confirmation_dataset = row.get("confirmation_dataset_hash")
            confirmation_split = row.get("confirmation_split_hash")
            if not (
                _is_sha256_digest(exploration_dataset)
                and _is_sha256_digest(exploration_split)
                and _is_sha256_digest(confirmation_dataset)
                and _is_sha256_digest(confirmation_split)
                and confirmation_dataset != exploration_dataset
                and confirmation_split != exploration_split
            ):
                checks["eligible_hashes_independent"] = False
        else:
            # Ineligible means no independent confirmation was attempted.  A
            # false pass value or a retained confirmation hash would instead
            # describe an attempted-and-failed confirmation and must not be
            # silently excluded from the eligible denominator.
            if (
                row.get("confirmation_passed") is not None
                or row.get("confirmation_dataset_hash") is not None
                or row.get("confirmation_split_hash") is not None
            ):
                checks["eligibility_semantics_consistent"] = False

    return checks


def audit(public_dir: Path, evaluator_dir: Path) -> dict[str, Any]:
    public = _load(public_dir / "benchmark_public_manifest.json")
    hidden = _load(evaluator_dir / "benchmark_hidden_manifest.json")
    receipt = _load(public_dir / "freeze_receipt.json")
    hidden_receipt = _load(evaluator_dir / "freeze_receipt.json")
    latent_audit_path = evaluator_dir / "latent_signature_audit.json"
    latent_audit = _load(latent_audit_path) if latent_audit_path.exists() else {}
    public_contract_path = public_dir / "evaluator_contract.json"
    private_contract_path = evaluator_dir / "evaluator_contract.json"
    public_contract = _load(public_contract_path) if public_contract_path.exists() else {}
    private_contract = _load(private_contract_path) if private_contract_path.exists() else {}
    environment_path = evaluator_dir / "environment_receipts.json"
    environment = _load(environment_path) if environment_path.exists() else None
    run_manifest_path = public_dir / "run_manifest.json"
    run_manifest = _load(run_manifest_path) if run_manifest_path.exists() else {}
    public_questions = _question_map(public)
    hidden_questions = _question_map(hidden)
    public_question_ids_unique = _question_ids_are_unique(public, public_questions)
    hidden_question_ids_unique = _question_ids_are_unique(hidden, hidden_questions)
    latent_count = 0
    all_latent_signatures: set[str] = set()
    world_ids: set[str] = set()
    question_commitments_ok = True
    generator_signatures_ok = True
    world_order_ok = True
    hidden_latent_fields_ok = True
    for question_id, item in hidden_questions.items():
        worlds = item.get("worlds", [])
        if not isinstance(worlds, list):
            world_order_ok = False
            hidden_latent_fields_ok = False
            generator_signatures_ok = False
            continue
        if not all(isinstance(world, Mapping) for world in worlds):
            world_order_ok = False
            hidden_latent_fields_ok = False
            generator_signatures_ok = False
            continue
        world_order_ok = world_order_ok and [world.get("kind") for world in worlds] == [
            "supported", "refuted", "insufficient", "invalid",
        ]
        latent_signatures = []
        for world in worlds:
            world_id = world.get("world_id")
            if world_id is None:
                world_order_ok = False
            world_ids.add(str(world_id))
            try:
                latent_signatures.append(_digest(_world_latent_payload(world)))
            except (KeyError, TypeError, ValueError, OverflowError):
                hidden_latent_fields_ok = False
                generator_signatures_ok = False
                continue
        all_latent_signatures.update(latent_signatures)
        latent_count += len(set(latent_signatures))
        hidden_latent_fields_ok = hidden_latent_fields_ok and item.get("latent_signatures") == latent_signatures
        public_item = public_questions.get(question_id, {})
        expected_commitment = _digest({
            "generator_signature": item.get("generator_signature"),
            "latent_signatures": latent_signatures,
        })
        question_commitments_ok = (
            question_commitments_ok
            and public_item.get("question_commitment") == expected_commitment
        )
        recipe = (hidden.get("generator_recipes") or {}).get(item.get("family"), {})
        expected_generator = _digest({
            "generator_version": hidden.get("generator_version"),
            "recipe": recipe,
            "variant": int(item.get("variant")),
            "split": item.get("split"),
            "latent_signatures": latent_signatures,
        })
        generator_signatures_ok = generator_signatures_ok and item.get("generator_signature") == expected_generator

    hidden_id_families = {
        str(item.get("family")) for item in hidden_questions.values()
        if item.get("split") == "final_id"
    }
    hidden_ood_families = {
        str(item.get("family")) for item in hidden_questions.values()
        if item.get("split") == "final_ood"
    }
    declared_id_families = {str(value) for value in hidden.get("final_id_families", [])}
    declared_ood_families = {str(value) for value in hidden.get("final_ood_families", [])}
    new_ood_families = hidden_ood_families - KNOWN_DEVELOPMENT_FAMILIES
    hidden_family_registry_ok = (
        hidden_id_families == declared_id_families
        and hidden_ood_families == declared_ood_families
        and hidden_id_families.isdisjoint(hidden_ood_families)
    )
    # Reconcile the producer-side audit file with independently recomputed facts;
    # do not let a stale/edited summary authorize a final profile.
    latent_audit_reconciles = bool(
        latent_audit
        and latent_audit.get("pass") is True
        and _field_is_int(latent_audit.get("final_question_count"), len(hidden_questions))
        and _field_is_int(latent_audit.get("final_world_count"), len(world_ids))
        and _field_is_int(latent_audit.get("final_latent_signature_count"), len(all_latent_signatures))
        and _field_is_int(latent_audit.get("new_ood_family_count"), len(new_ood_families))
        and _field_is_int(latent_audit.get("latent_overlap_count"), 0)
        and bool((latent_audit.get("gates") or {}).get("ood_family_names_unseen") is True)
        and bool((latent_audit.get("gates") or {}).get("ood_has_at_least_three_new_families") is True)
    )

    # Check hidden/public manifest digests independently of the generator module.
    public_manifest_digest_ok = _valid_digest(public, "manifest_digest")
    hidden_manifest_digest_ok = _valid_digest(hidden, "manifest_digest")
    receipt_digest_ok = _valid_digest(receipt, "receipt_digest") and _valid_digest(hidden_receipt, "receipt_digest")
    forbidden_keys = {
        "family", "mechanism_families", "final_id_families", "final_ood_families",
        "worlds", "latent_signatures", "generator_recipes", "target_actions",
        "legacy_target_actions_audit_only",
    }

    def keys(value: Any):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key)
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    public_keys = set(keys(public))
    public_encoded = json.dumps(public, sort_keys=True, ensure_ascii=False)
    public_boundary_ok = not public_keys.intersection(forbidden_keys) and all(
        set(item) == {"question_id", "policy_question_id", "split", "world_count", "question_commitment"}
        for item in public.get("questions", [])
    ) and "true_effect_a" not in public_encoded and "true_effect_b" not in public_encoded
    split_counts = {
        split: sum(item.get("split") == split for item in hidden_questions.values())
        for split in ("final_id", "final_ood")
    }
    repo_root = public_dir.resolve().parent.parent
    baseline_receipt_valid = _baseline_receipt_valid(receipt, repo_root=repo_root)
    hidden_baseline_receipt_valid = _baseline_receipt_valid(hidden_receipt, repo_root=repo_root)
    expected_source_paths = [
        repo_root / "research_strategy_optimization/evaluation/tier1_v05_frozen_final.py",
        repo_root / "scripts/prepare_tier1_v05_frozen_final.py",
        repo_root / "scripts/audit_tier1_v05_frozen_final.py",
        repo_root / "scripts/record_v05_baseline_selection.py",
    ]
    source_entries = (
        (run_manifest.get("source") or {}).get("files", [])
        if isinstance(run_manifest, Mapping) else []
    )
    source_hashes_current = bool(
        len(source_entries) == len(expected_source_paths)
        and all(
            any(
                str(entry.get("path")) == path.relative_to(repo_root).as_posix()
                and entry.get("sha256") == _file_digest(path)
                for entry in source_entries
            )
            for path in expected_source_paths
        )
    )
    data_entries = (run_manifest.get("data") or {}).get("files", []) if isinstance(run_manifest, Mapping) else []
    data_hashes_current = _inventory_matches(data_entries, repo_root)
    raw_environment_rows = environment.get("rows", []) if environment else []
    # Treat a malformed/non-list rows field as an empty collection so every
    # receipt gate fails closed instead of allowing a tampered private bundle to
    # crash the independent auditor.
    environment_rows = raw_environment_rows if isinstance(raw_environment_rows, list) else []
    environment_digest_ok = bool(
        environment is not None and _valid_digest(environment, "receipt_digest")
    )
    environment_atomic_ok = bool(
        environment is not None and _environment_rows_have_atomic_utilities(environment_rows)
    )
    grouped_seeds: dict[tuple[str, str, str], set[int]] = {}
    grouped_parse_ok = True
    for row in environment_rows:
        if not isinstance(row, Mapping):
            grouped_parse_ok = False
            continue
        seed = _safe_int(row.get("exploration_seed"))
        if seed is None:
            grouped_parse_ok = False
            continue
        key = (
            str(row.get("question_id_audit")),
            str(row.get("world_id_audit")),
            str(row.get("action")),
        )
        grouped_seeds.setdefault(key, set()).add(seed)
    environment_coverage_ok = bool(
        environment is not None
        and grouped_parse_ok
        and _field_is_int(environment.get("question_count_collected"), 96)
        and len(environment_rows) == 96 * 4 * 4 * 8
        and len(grouped_seeds) == 96 * 4 * 4
        and all(len(seeds) == 8 for seeds in grouped_seeds.values())
    )
    environment_binding_ok = bool(
        environment is not None
        and _environment_rows_are_well_bound(environment_rows, hidden_questions)
    )
    environment_confirmation_checks = _environment_confirmation_receipt_checks(
        environment_rows
    )
    contract_boundary_ok = bool(
        public_contract
        and private_contract
        and public_contract.get("hidden_manifest_not_in_public_bundle") is True
        and public_contract.get("independent_process_required_for_claim") is True
        and public_contract.get("formal_comparison_authorized") is False
        and "private_bundle_path" not in public_contract
        and private_contract.get("private_bundle_path") == str(evaluator_dir.resolve())
        and private_contract.get("public_bundle_path") == str(public_dir.resolve())
    )
    gates = {
        "public_manifest_digest": public_manifest_digest_ok,
        "hidden_manifest_digest": hidden_manifest_digest_ok,
        "receipt_digest": receipt_digest_ok,
        "run_manifest_digest": _valid_digest(run_manifest, "manifest_digest") if run_manifest else False,
        "run_manifest_source_current": source_hashes_current,
        "run_manifest_data_current": data_hashes_current,
        "environment_receipt_digest": environment_digest_ok,
        "environment_atomic_reward_sums": environment_atomic_ok,
        "environment_full_seed_coverage": environment_coverage_ok,
        "environment_rows_bound_to_hidden_worlds": environment_binding_ok,
        "environment_confirmation_seeds_preregistered": bool(
            environment is not None
            and environment_confirmation_checks["seeds_preregistered"]
        ),
        "environment_eligible_confirmation_hashes_independent": bool(
            environment is not None
            and environment_confirmation_checks["eligible_hashes_independent"]
        ),
        "environment_confirmation_eligibility_semantics_consistent": bool(
            environment is not None
            and environment_confirmation_checks["eligibility_semantics_consistent"]
        ),
        "question_sets_match": set(public_questions) == set(hidden_questions),
        "public_question_ids_unique": public_question_ids_unique,
        "hidden_question_ids_unique": hidden_question_ids_unique,
        "public_question_count_matches": _field_is_int(public.get("question_count"), len(public_questions)),
        "public_world_count_matches": _field_is_int(public.get("world_count"), len(world_ids)),
        "hidden_question_count_matches": _field_is_int(hidden.get("question_count"), len(hidden_questions)),
        "hidden_world_count_matches": _field_is_int(hidden.get("world_count"), len(world_ids)),
        "question_commitments_match": question_commitments_ok,
        "generator_signatures_match": generator_signatures_ok,
        "hidden_latent_signature_fields_match": hidden_latent_fields_ok,
        "latent_signature_audit_reconciles": latent_audit_reconciles,
        "ood_family_names_unseen": bool(hidden_ood_families) and not hidden_ood_families.intersection(KNOWN_DEVELOPMENT_FAMILIES),
        "ood_has_at_least_three_new_families": len(new_ood_families) >= 3,
        "hidden_family_registry_valid": hidden_family_registry_ok,
        "world_order_valid": world_order_ok,
        "world_ids_unique": len(world_ids) == sum(len(item.get("worlds", [])) for item in hidden_questions.values()),
        "latent_outputs_unique": len(all_latent_signatures) == len(world_ids),
        "public_boundary_clean": public_boundary_ok,
        "final_id_minimum": split_counts["final_id"] >= 40,
        "final_ood_minimum": split_counts["final_ood"] >= 40,
        "private_bundle_separate": public_dir.resolve() != evaluator_dir.resolve(),
        "private_bundle_not_nested": (
            public_dir.resolve() not in evaluator_dir.resolve().parents
            and evaluator_dir.resolve() not in public_dir.resolve().parents
        ),
        "final_access_locked": public.get("final_access", {}).get("locked") is True,
        "hidden_final_access_locked": hidden.get("final_access", {}).get("locked") is True,
        "freeze_receipts_match": receipt == hidden_receipt,
        # Empty baseline is valid for an unsigned/pending structural bundle; once
        # a baseline is present (and especially when signed), its digest and bound
        # files must independently validate.  This prevents fake sha256 prefixes
        # or reserved empty manifests from satisfying the freeze contract.
        "baseline_receipt_integrity": bool(
            baseline_receipt_valid
            and hidden_baseline_receipt_valid
            and (receipt.get("signed") is not True or baseline_receipt_valid)
        ),
        "evaluator_contract_boundary": contract_boundary_ok,
    }
    result = {
        "schema_version": "pesco_v05_independent_audit_result_v0.1",
        "evaluator_version": "pesco_v05_independent_evaluator_v1",
        "public_directory": str(public_dir.resolve()),
        "evaluator_directory": str(evaluator_dir.resolve()),
        "question_count": len(hidden_questions),
        "world_count": len(world_ids),
        "latent_signature_count": latent_count,
        "new_ood_family_count": len(new_ood_families),
        "final_ood_families": sorted(hidden_ood_families),
        "counts_by_split": split_counts,
        "freeze_status": receipt.get("status"),
        "freeze_signed": receipt.get("signed") is True,
        "baseline_receipt_valid": baseline_receipt_valid,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", default="artifacts/tier1_v05_frozen_final")
    parser.add_argument("--evaluator", default="artifacts/tier1_v05_evaluator_private")
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args(argv)
    result = audit(Path(args.public), Path(args.evaluator))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if not result["pass"]:
        return 1
    if args.require_frozen and not result["freeze_signed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

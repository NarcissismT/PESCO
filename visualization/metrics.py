"""Input loading and plan-aligned PESCO metrics.

The evaluator is intentionally tolerant about field names so that early
simulators and later training runners can share the reporting code.  The
canonical names are documented in ``README.md``.  Missing metrics are reported
as ``None`` rather than silently interpreted as zero.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


STATES: Tuple[str, ...] = ("Supported", "Refuted", "Insufficient", "Invalid")

_STATE_ALIASES = {
    "supported": "Supported",
    "support": "Supported",
    "positive": "Supported",
    "refuted": "Refuted",
    "refute": "Refuted",
    "negative": "Refuted",
    "insufficient": "Insufficient",
    "underpowered": "Insufficient",
    "uncertain": "Insufficient",
    "invalid": "Invalid",
    "invalid_experiment": "Invalid",
    "leakage": "Invalid",
}


def _first(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "none"}:
        return False
    return None


def normalize_state(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text in STATES:
        return text
    return _STATE_ALIASES.get(text.lower().replace(" ", "_").replace("-", "_"))


def record_method(record: Mapping[str, Any]) -> str:
    value = _first(record, "method", "policy", "algorithm", "agent", default="Unknown")
    return str(value)


def record_split(record: Mapping[str, Any]) -> str:
    value = _first(record, "split", "evaluation_split", "partition", default="all")
    return str(value)


def _unwrap_records(value: Any) -> List[Dict[str, Any]]:
    """Extract records from the common JSON containers used by runners."""
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    for key in ("records", "runs", "results", "events", "trajectories", "branches", "data"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return [dict(item) for item in candidate if isinstance(item, Mapping)]
    # A single run is also a valid input.
    if any(k in value for k in ("method", "policy", "question_id", "world_state", "true_state")):
        return [dict(value)]
    return []


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON or JSONL result file and return flat dictionaries.

    If a record contains a ``branches`` list, each branch is flattened while
    inheriting the parent method/question/world metadata.  This lets branch
    runners retain a compact trajectory file without a second conversion step.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"result file not found: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: List[Dict[str, Any]] = []
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_no}: {exc}") from exc
            records.extend(_unwrap_records(parsed))
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {source}: {exc}") from exc
        records = _unwrap_records(parsed)

    flattened: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record.get("metrics"), Mapping):
            # Runners commonly keep measurements under a ``metrics`` object;
            # copy them to a flat reporting row without overwriting explicit
            # audit fields.
            record = {**dict(record["metrics"]), **record}
        for ledger_key in ("compute_ledger", "ledger", "compute"):
            ledger = record.get(ledger_key)
            if isinstance(ledger, Mapping):
                record = {**dict(ledger), **record}
        # Core PESCO trajectories are intentionally nested to preserve replay
        # information.  A reporting row is made from the terminal output and
        # terminal trusted verdict while retaining all audit identifiers.
        if isinstance(record.get("trajectory"), Mapping):
            parent = {k: v for k, v in record.items() if k != "trajectory"}
            nested = dict(record["trajectory"])
            nested.update(parent)
            record = nested
        if isinstance(record.get("outputs"), list):
            output = next((item for item in reversed(record["outputs"]) if isinstance(item, Mapping)), {})
            verdicts = record.get("verdicts")
            verdict = next((item for item in reversed(verdicts) if isinstance(item, Mapping)), {}) if isinstance(verdicts, list) else {}
            confirmation = verdict.get("independent_confirmation") if isinstance(verdict, Mapping) else None
            confirmation_passed = confirmation.get("passed") if isinstance(confirmation, Mapping) else None
            hidden_outputs = record.get("hidden_outputs")
            hidden_output = next((item for item in reversed(hidden_outputs) if isinstance(item, Mapping)), {}) if isinstance(hidden_outputs, list) else {}
            flattened_row = {k: v for k, v in record.items() if k not in {"outputs", "verdicts", "hidden_outputs", "initial_observation", "final_observation"}}
            flattened_row.update({
                "action": _first(output, "action", "selected_action"),
                "selected_action": _first(output, "selected_action", "action"),
                "method": _first(output, "method", default=flattened_row.get("method", "Unknown")),
                "effect_estimate": _first(output, "effect_estimate"),
                "execution_cost": _first(output, "execution_cost", "cost"),
                "cost": _first(output, "cost", "execution_cost", default=flattened_row.get("total_cost")),
                "validity_signals": _first(output, "validity_signals", default=()),
                "predicted_state": _first(verdict, "evidence_state", "state", "predicted_state"),
                "valid_claim": _first(verdict, "validity_pass", "valid_claim"),
                "independent_confirmed": _first(verdict, "independent_confirmation_passed", "replication_passed", default=confirmation_passed),
                "new_path_verified": _first(verdict, "autonomous", "certificate_pass", "new_path_verified"),
                "discovery_gain": _first(verdict, "discovered_gain", "lower_confidence_gain"),
                "latent_effect": _first(hidden_output, "latent_effect"),
                "leakage": _first(hidden_output, "leakage"),
                "confounding": _first(hidden_output, "confounding"),
            })
            # The public trajectory may only have a ``world_id`` placeholder;
            # use an explicitly exported true/evaluation state when present,
            # never treat the policy's verdict as ground truth implicitly.
            if "true_state" not in flattened_row:
                explicit_truth = _first(record, "ground_truth_state", "verifier_state", "true_evidence_state")
                if explicit_truth is not None:
                    flattened_row["true_state"] = explicit_truth
            flattened.append(flattened_row)
            continue
        branches = record.get("branches")
        if isinstance(branches, list):
            parent = {k: v for k, v in record.items() if k != "branches"}
            for branch_index, branch in enumerate(branches):
                if not isinstance(branch, Mapping):
                    continue
                merged = dict(parent)
                merged.update(dict(branch))
                merged.setdefault("branch_index", branch_index)
                flattened.append(merged)
        else:
            flattened.append(record)
    return flattened


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


# These metrics are only defined on a scientifically relevant subset of rows.  A
# previous version averaged their boolean values over *all* worlds, which made a
# method look worse simply because the experiment contained worlds where the
# corresponding action was not required.  Keep the eligibility rules in one place so
# the CSV, plots, and bootstrap code cannot silently drift apart.
CONDITIONAL_METRICS: Tuple[str, ...] = (
    "flip_accuracy",
    "effective_switch_rate",
    "invalid_repair_rate",
    "underpower_handling",
    "replication_rate",
)

# Public denominator aliases.  The ``*_n`` field on the metric itself remains for
# backward compatibility; these explicit names make the conditional denominator
# auditable in reports and are intentionally stable for downstream consumers.
CONDITIONAL_DENOMINATOR_FIELDS = {
    "flip_accuracy": "flip_eligible_n",
    "effective_switch_rate": "required_switch_n",
    "invalid_repair_rate": "invalid_repair_n",
    "underpower_handling": "insufficient_handling_n",
    "replication_rate": "confirmation_eligible_n",
}


def _truth_state(record: Mapping[str, Any]) -> Optional[str]:
    """Read an evaluator-side state without treating a policy prediction as truth."""

    return normalize_state(_first(
        record,
        "true_state",
        "ground_truth_state",
        "verifier_state",
        "world_kind",
        "world_state",
    ))


def _explicit_bool(record: Mapping[str, Any], *names: str) -> Optional[bool]:
    """Return the first explicitly supplied boolean, preserving explicit False."""

    for name in names:
        if name in record and record[name] is not None:
            return _as_bool(record[name])
    return None


def metric_eligibility(record: Mapping[str, Any], metric: str) -> bool:
    """Whether one record belongs in a conditional metric's denominator.

    Explicit eligibility flags always win.  Fallbacks use evaluator-side truth and
    protocol metadata, never the model's predicted state.  This function is public so
    runners can emit the same flags and tests can audit the denominator independently
    of the aggregate implementation.
    """

    metric = str(metric)
    if metric == "flip_accuracy":
        explicit = _explicit_bool(record, "flip_eligible", "preference_reversal_eligible", "paired_flip_eligible")
        if explicit is not None:
            return explicit
        truth = _truth_state(record)
        pair = _first(record, "world_pair_id", "pair_id", "preference_pair_id")
        # A hand-labelled flip outcome is itself an eligibility declaration for
        # legacy records that did not carry an explicit pair field.
        labelled = _first(record, "flip_correct", "preference_reversal_correct")
        return truth in {"Supported", "Refuted"} and (pair is not None or labelled is not None)
    if metric == "effective_switch_rate":
        explicit = _explicit_bool(record, "required_switch", "switch_required", "required_action_is_switch")
        if explicit is not None:
            return explicit
        best_action = str(_first(record, "best_action", "optimal_action", default="")).lower()
        return _truth_state(record) == "Refuted" or best_action == "switch_to_alternative_method"
    if metric == "invalid_repair_rate":
        explicit = _explicit_bool(record, "invalid_repair_eligible", "invalid_initial", "initial_invalid")
        if explicit is not None:
            return explicit
        return _truth_state(record) == "Invalid"
    if metric == "underpower_handling":
        explicit = _explicit_bool(record, "insufficient_handling_eligible", "insufficient_initial", "initial_insufficient")
        if explicit is not None:
            return explicit
        return _truth_state(record) == "Insufficient"
    if metric == "replication_rate":
        explicit = _explicit_bool(record, "confirmation_eligible", "eligible_for_confirmation")
        if explicit is not None:
            return explicit
        # ``entered_confirmation`` is the strongest legacy signal.  Do not default to
        # True: invalid/insufficient branches are intentionally not confirmation
        # eligible and must not dilute the denominator.
        inferred = _explicit_bool(
            record,
            "entered_confirmation",
            "confirmation_attempted",
            "independent_confirmation_performed",
            "independent_confirmed",
        )
        return bool(inferred) if inferred is not None else False
    return True


def _raw_metric_value(record: Mapping[str, Any], name: str, weights: Mapping[str, float]) -> Optional[float]:
    """Record-level metric value before applying conditional eligibility."""

    if name == "flip_accuracy":
        value = _as_bool(_first(record, "flip_correct", "preference_reversal_correct"))
        return float(value) if value is not None else None
    if name == "effective_switch_rate":
        value = _as_bool(_first(record, "effective_switch", "switch_effective"))
        if value is not None:
            return float(value)
        switched = _as_bool(_first(record, "switch", "switched"))
        beneficial = _as_bool(_first(record, "switch_beneficial", "switch_confirmed_better"))
        return float(switched and beneficial) if switched is not None and beneficial is not None else None
    if name == "invalid_repair_rate":
        value = _as_bool(_first(record, "invalid_repaired", "experiment_repaired", "repair_success"))
        if value is not None:
            return float(value)
        truth = _truth_state(record)
        action = str(_first(record, "selected_action", "action", default="")).lower()
        valid = _as_bool(_first(record, "valid_claim", "claim_valid", "validity_pass"))
        return float("repair" in action and valid) if truth == "Invalid" and valid is not None else None
    if name == "underpower_handling":
        value = _as_bool(_first(record, "underpower_handled", "insufficient_handled", "evidence_gap_handled"))
        if value is not None:
            return float(value)
        action = str(_first(record, "selected_action", "action", default="")).lower()
        reasonable = any(token in action for token in ("sample", "repeat", "measure", "stop", "insufficient"))
        return float(reasonable) if _truth_state(record) == "Insufficient" else None
    if name == "replication_rate":
        confirmed = _as_bool(_first(record, "independent_confirmed", "replication_passed", "independent_confirmation_passed", "confirmed"))
        return float(confirmed) if confirmed is not None else None
    return _metric_value_unconditional(record, name, weights)


def _metric_value_unconditional(record: Mapping[str, Any], name: str, weights: Mapping[str, float]) -> Optional[float]:
    """Implementation for metrics whose denominator is the full group."""

    if name == "vrs":
        explicit = _as_float(_first(record, "vrs", "credible_scientific_value"))
        if explicit is not None:
            return explicit
        valid = _as_bool(_first(record, "valid_claim", "claim_valid", "validity_pass"))
        belief = _as_float(_first(record, "belief_score", "s_belief"))
        task = _as_float(_first(record, "task_utility", "u_task"))
        replication = _as_float(_first(record, "replication_utility", "u_replication"))
        discovery = _as_float(_first(record, "discovery_utility", "u_discovery"))
        cost = _as_float(_first(record, "cost", "total_cost", "compute_cost"))
        if valid is None or any(value is None for value in (belief, task, replication, discovery, cost)):
            return None
        value = (
            weights["alpha"] * float(belief)
            + weights["beta"] * float(task)
            + weights["gamma"] * float(replication)
            + weights["eta"] * float(discovery)
        )
        return (value if valid else 0.0) - weights["lambda"] * float(cost)
    if name == "state_correct":
        truth = _truth_state(record)
        pred = normalize_state(_first(record, "predicted_state", "reported_state", "state_prediction"))
        return float(truth == pred) if truth is not None and pred is not None else None
    if name == "unnecessary_switch_rate":
        value = _as_bool(_first(record, "unnecessary_switch", "switch_unnecessary"))
        if value is not None:
            return float(value)
        switched = _as_bool(_first(record, "switch", "switched"))
        beneficial = _as_bool(_first(record, "switch_beneficial", "switch_confirmed_better"))
        return float(switched and not beneficial) if switched is not None and beneficial is not None else None
    if name == "appropriate_persistence":
        value = _as_bool(_first(record, "persistence_correct", "appropriate_persistence"))
        if value is not None:
            return float(value)
        persisted = _as_bool(_first(record, "persisted", "continued_current_method"))
        optimal = _as_bool(_first(record, "current_strategy_optimal", "current_optimal"))
        return float(persisted and optimal) if persisted is not None and optimal is not None else None
    if name == "refutation_acceptance":
        value = _as_bool(_first(record, "refutation_accept", "accepted_refutation", "reliable_negative_result"))
        if value is not None:
            return float(value)
        truth = _truth_state(record)
        pred = normalize_state(_first(record, "predicted_state", "reported_state"))
        return float(truth == "Refuted" and pred == "Refuted") if truth and pred else None
    if name == "invalid_claim_rate":
        value = _as_bool(_first(record, "invalid_claim", "claim_on_invalid"))
        if value is not None:
            return float(value)
        truth = _truth_state(record)
        valid_claim = _as_bool(_first(record, "valid_claim", "claim_valid"))
        return float(valid_claim is True) if truth == "Invalid" and valid_claim is not None else None
    if name == "vnpr":
        verified = _as_bool(_first(record, "new_path_verified", "verified_new_path", "discovery_confirmed"))
        opportunity = _as_bool(_first(record, "discovery_opportunity", "new_path_opportunity", default=True))
        return float(verified) if verified is not None and opportunity else None
    if name == "fdr":
        announced = _as_bool(_first(record, "new_path_announced", "discovery_announced", "announced_valid", "method_announced_valid"))
        if not announced:
            return None
        confirmed = _as_bool(_first(record, "independent_confirmed", "replication_passed", "independent_confirmation_passed", "confirmed"))
        return float(confirmed is not True)
    if name == "cost":
        explicit = _as_float(_first(record, "cost", "total_cost", "compute_cost", "budget_used", "normalized_cost"))
        if explicit is not None:
            return explicit
        gpu = _as_float(_first(record, "gpu_hours", "gpu_cost", default=0.0)) or 0.0
        cpu = _as_float(_first(record, "cpu_hours", "cpu_cost", default=0.0)) or 0.0
        env_runs = _as_float(_first(record, "environment_runs", "rollout_runs", default=0.0)) or 0.0
        tokens = _as_float(_first(record, "tokens", "token_count", default=0.0)) or 0.0
        derived = gpu + cpu + 0.01 * env_runs + 1e-5 * tokens
        return derived if derived else None
    if name == "utility":
        return _as_float(_first(record, "utility", "verified_scientific_utility", "task_utility"))
    return _as_float(record.get(name))


def _metric_value(record: Mapping[str, Any], name: str, weights: Mapping[str, float]) -> Optional[float]:
    """Compute one scalar metric at record level.

    Conditional metrics return ``None`` outside their declared denominator.  The
    aggregate layer then computes the ratio using the corresponding eligible rows;
    callers should not average this function over an unfiltered record list.
    """

    if name in CONDITIONAL_METRICS and not metric_eligibility(record, name):
        return None
    return _raw_metric_value(record, name, weights)


METRIC_NAMES: Tuple[str, ...] = (
    "vrs",
    "state_macro_f1",
    "flip_accuracy",
    "effective_switch_rate",
    "unnecessary_switch_rate",
    "appropriate_persistence",
    "refutation_acceptance",
    "underpower_handling",
    "invalid_repair_rate",
    "invalid_claim_rate",
    "vnpr",
    "replication_rate",
    "fdr",
    "cost",
    "utility",
)


def _macro_f1(records: Sequence[Mapping[str, Any]]) -> Optional[float]:
    pairs = []
    for record in records:
        # A trusted evaluator diagnostic is not a policy state prediction.  When a
        # runner emits the explicit eligibility/source fields, honor them so an
        # action-only method cannot receive perfect Macro-F1 by copying the verifier.
        eligible = _as_bool(record.get("state_metric_eligible"))
        if eligible is False or record.get("state_prediction_source") == "evaluator_diagnostic":
            continue
        truth = normalize_state(_first(record, "true_state", "world_state", "evidence_state"))
        pred = normalize_state(_first(record, "predicted_state", "reported_state", "state_prediction"))
        if truth is not None and pred is not None:
            pairs.append((truth, pred))
    if not pairs:
        return None
    f1_values = []
    for state in STATES:
        tp = sum(t == state and p == state for t, p in pairs)
        fp = sum(t != state and p == state for t, p in pairs)
        fn = sum(t == state and p != state for t, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1_values) / len(STATES)


def confusion_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str, str]] = Counter()
    for record in records:
        method = record_method(record)
        truth = normalize_state(_first(record, "true_state", "world_state", "evidence_state")) or "Unknown"
        pred = normalize_state(_first(record, "predicted_state", "reported_state", "state_prediction")) or "Unknown"
        counts[(method, truth, pred)] += 1
    rows = []
    for (method, truth, pred), count in sorted(counts.items()):
        rows.append({"method": method, "true_state": truth, "predicted_state": pred, "count": count})
    return rows


def _group_records(records: Sequence[Mapping[str, Any]], by_split: bool = True) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record_method(record), record_split(record) if by_split else "all")].append(record)
    return groups


def _flip_pair_groups(records: Sequence[Mapping[str, Any]]) -> List[List[Mapping[str, Any]]]:
    """Return eligible supported/refuted world-pair groups for FlipAcc.

    FlipAcc is a pair-level estimand, not a four-world row-level average.  A pair is
    eligible when the runner explicitly marks it eligible, or when both Supported and
    Refuted members of the same ``world_pair_id`` are present.  Rows without a pair
    identifier remain useful legacy observations when they carry an explicit
    ``flip_eligible`` flag, but are never inferred eligible from a prediction alone.
    """

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not metric_eligibility(record, "flip_accuracy"):
            continue
        pair = _first(record, "world_pair_id", "pair_id", "preference_pair_id")
        explicit = _explicit_bool(record, "flip_eligible", "preference_reversal_eligible", "paired_flip_eligible")
        if pair is None:
            # Explicitly labelled rows can be scored independently, but get distinct
            # keys so two unrelated records are not accidentally paired.
            pair = f"__explicit_flip_row_{index}"
        else:
            # Pair identifiers are often only unique within a question/split.  Keep
            # that scope in the key so a direct aggregate over multiple splits cannot
            # combine unrelated pairs that happen to share a short ID.
            scope_question = str(_first(record, "question_id", "research_question_id", "task_id", default="all"))
            scope_split = record_split(record)
            pair = f"{scope_question}::{scope_split}::{pair}"
        grouped[str(pair)].append(record)

    eligible: List[List[Mapping[str, Any]]] = []
    for rows in grouped.values():
        explicit_flags = [
            _explicit_bool(row, "flip_eligible", "preference_reversal_eligible", "paired_flip_eligible")
            for row in rows
        ]
        states = {_truth_state(row) for row in rows}
        explicit_yes = any(flag is True for flag in explicit_flags)
        complete_pair = {"Supported", "Refuted"}.issubset(states)
        if explicit_yes or complete_pair:
            eligible.append(rows)
    return eligible


def _conditional_stats(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    weights: Mapping[str, float],
) -> Tuple[Optional[float], int, int]:
    """Return ``(rate, numerator, denominator)`` for a conditional metric."""

    if metric == "flip_accuracy":
        # One outcome per eligible pair.  If a legacy export repeats the pair outcome
        # on both worlds, all repeated annotations must agree; disagreement is kept as
        # a failed pair rather than silently selecting the first row.
        numerator = 0
        denominator = 0
        for pair_rows in _flip_pair_groups(records):
            values = [_raw_metric_value(row, metric, weights) for row in pair_rows]
            denominator += 1
            # An eligible pair with a missing/invalid outcome is retained in the
            # denominator and fails closed.  Otherwise a policy could improve
            # FlipAcc by simply omitting one side of a required pair.
            numerator += int(bool(values) and all(value is not None and float(value) > 0.5 for value in values))
        return _ratio(numerator, denominator), numerator, denominator

    eligible_rows = [row for row in records if metric_eligibility(row, metric)]
    values = [_raw_metric_value(row, metric, weights) for row in eligible_rows]
    numerator = sum(
        1 for value in values
        if value is not None and math.isfinite(float(value)) and float(value) > 0.5
    )
    # Keep all eligible observations in the denominator, including rows whose
    # outcome was omitted or malformed.  Missing outcomes are failures, not NA
    # opportunities, for a conditional success-rate estimand.
    denominator = len(eligible_rows)
    return _ratio(numerator, denominator), numerator, denominator


def conditional_metric_stats(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Optional[float] | int]:
    """Public auditable conditional-rate helper.

    The returned mapping contains ``value``, ``numerator``, and ``denominator``.  A
    metric with no eligible observations is ``None`` rather than zero.  This helper is
    intentionally shared by report tests and downstream evaluators.
    """

    if metric not in CONDITIONAL_METRICS:
        raise ValueError(f"{metric!r} is not a conditional metric")
    metric_weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0, "lambda": 0.1}
    if weights:
        metric_weights.update({key: float(value) for key, value in weights.items() if key in metric_weights})
    value, numerator, denominator = _conditional_stats(records, metric, metric_weights)
    return {"value": value, "numerator": numerator, "denominator": denominator}


def _aggregate_metric_value(records: Sequence[Mapping[str, Any]], metric: str, weights: Mapping[str, float]) -> Optional[float]:
    if metric in CONDITIONAL_METRICS:
        return _conditional_stats(records, metric, weights)[0]
    return _mean(_raw_metric_value(record, metric, weights) for record in records)


def aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    by_split: bool = True,
    weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate plan metrics by method and (optionally) evaluation split."""
    metric_weights = {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "eta": 1.0,
        "lambda": 0.1,
    }
    if weights:
        metric_weights.update({key: float(value) for key, value in weights.items() if key in metric_weights})

    rows: List[Dict[str, Any]] = []
    for (method, split), group in sorted(_group_records(records, by_split).items()):
        roles = sorted({str(record.get("comparison_role")) for record in group if record.get("comparison_role") is not None})
        row: Dict[str, Any] = {
            "method": method,
            "split": split,
            "n": len(group),
            "comparison_role": "; ".join(roles) if roles else None,
            # Parse serialized booleans explicitly.  ``bool("false")`` is True and
            # would turn a diagnostic-only export into a formally eligible method.
            "formal_comparison_eligible": any(
                _as_bool(record.get("formal_comparison_eligible")) is True for record in group
            ),
        }
        row["state_macro_f1"] = _macro_f1(group)
        for metric in METRIC_NAMES:
            if metric == "state_macro_f1":
                continue
            if metric == "fdr":
                announced = [record for record in group if _as_bool(_first(record, "new_path_announced", "discovery_announced", "announced_valid", "method_announced_valid"))]
                values = [_raw_metric_value(record, metric, metric_weights) for record in announced]
                row[metric] = _mean(values)
                row["fdr_n"] = sum(value is not None for value in values)
                continue
            if metric in CONDITIONAL_METRICS:
                value, numerator, denominator = _conditional_stats(group, metric, metric_weights)
                row[metric] = value
                row[f"{metric}_n"] = denominator
                row[f"{metric}_numerator"] = numerator
                row[f"{metric}_denominator"] = denominator
                # Add an explicit semantic alias (for example
                # ``confirmation_eligible_n``) alongside the generic metric count.
                row[CONDITIONAL_DENOMINATOR_FIELDS[metric]] = denominator
                row[f"{metric}_eligible_n"] = denominator
                row[f"{metric}_success_n"] = numerator
                continue
            values = [_raw_metric_value(record, metric, metric_weights) for record in group]
            row[metric] = _mean(values)
        # Useful denominators make sparse metrics auditable in CSV/report output.
        for metric in METRIC_NAMES:
            if metric == "state_macro_f1":
                continue
            if metric == "fdr" or metric in CONDITIONAL_METRICS:
                continue
            row[f"{metric}_n"] = sum(_raw_metric_value(record, metric, metric_weights) is not None for record in group)
        rows.append(row)
    return rows


def _cluster_key(record: Mapping[str, Any]) -> str:
    return str(_first(record, "question_id", "research_question_id", "task_id", "world_pair_id", "run_id", default="row"))


def bootstrap_ci(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_boot: int = 300,
    seed: int = 7,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Question-cluster bootstrap percentile intervals for one metric.

    The output maps ``(method, split)`` to a 95% interval.  It is intentionally
    small and deterministic for pilot reports; final papers may replace it
    with the preregistered statistical package without changing the schema.
    """
    if not records or n_boot <= 0:
        return {}
    metric_weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0, "lambda": 0.1}
    if weights:
        metric_weights.update({key: float(value) for key, value in weights.items() if key in metric_weights})
    groups: Dict[Tuple[str, str], Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record_method(record), record_split(record))
        groups[key][_cluster_key(record)].append(record)
    rng = random.Random(seed)
    intervals: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    for key, clusters in groups.items():
        cluster_values = list(clusters.values())
        # A percentile interval needs at least two independent question/task
        # clusters.  Resampling one cluster only reproduces the point estimate and
        # falsely suggests certainty, so the scientifically correct result is NA.
        if len(cluster_values) < 2:
            intervals[key] = (None, None)
            continue
        samples: List[float] = []
        for _ in range(n_boot):
            sampled = [cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values]
            flat = [item for cluster in sampled for item in cluster]
            if metric == "state_macro_f1":
                value = _macro_f1(flat)
            else:
                value = _aggregate_metric_value(flat, metric, metric_weights)
            if value is not None:
                samples.append(value)
        if samples:
            samples.sort()
            lo = samples[max(0, int(0.025 * (len(samples) - 1)))]
            hi = samples[min(len(samples) - 1, int(0.975 * (len(samples) - 1)))]
            intervals[key] = (lo, hi)
        else:
            intervals[key] = (None, None)
    return intervals


def cluster_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Return the number of independent question/task clusters in ``records``."""

    return len({_cluster_key(record) for record in records})


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("\n", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)

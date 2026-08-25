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


def _metric_value(record: Mapping[str, Any], name: str, weights: Mapping[str, float]) -> Optional[float]:
    """Compute one scalar metric at record level."""
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
        # A fallback VRS must be auditable.  If a runner has not supplied all
        # frozen components, leave it NA rather than silently treating missing
        # scientific value as zero (plan §17.1).
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
        truth = normalize_state(_first(record, "true_state", "world_state", "evidence_state"))
        pred = normalize_state(_first(record, "predicted_state", "reported_state", "state_prediction"))
        return float(truth == pred) if truth is not None and pred is not None else None
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
        truth = normalize_state(_first(record, "true_state", "world_state"))
        pred = normalize_state(_first(record, "predicted_state", "reported_state"))
        return float(truth == "Refuted" and pred == "Refuted") if truth and pred else None
    if name == "underpower_handling":
        value = _as_bool(_first(record, "underpower_handled", "insufficient_handled", "evidence_gap_handled"))
        if value is not None:
            return float(value)
        truth = normalize_state(_first(record, "true_state", "world_state"))
        action = str(_first(record, "selected_action", "action", default="")).lower()
        reasonable = any(token in action for token in ("sample", "repeat", "measure", "stop", "insufficient"))
        return float(reasonable) if truth == "Insufficient" else None
    if name == "invalid_repair_rate":
        value = _as_bool(_first(record, "invalid_repaired", "experiment_repaired", "repair_success"))
        return float(value) if value is not None else None
    if name == "invalid_claim_rate":
        value = _as_bool(_first(record, "invalid_claim", "claim_on_invalid"))
        if value is not None:
            return float(value)
        truth = normalize_state(_first(record, "true_state", "world_state"))
        valid_claim = _as_bool(_first(record, "valid_claim", "claim_valid"))
        return float(valid_claim is True) if truth == "Invalid" and valid_claim is not None else None
    if name == "vnpr":
        verified = _as_bool(_first(record, "new_path_verified", "verified_new_path", "discovery_confirmed"))
        opportunity = _as_bool(_first(record, "discovery_opportunity", "new_path_opportunity", default=True))
        return float(verified) if verified is not None and opportunity else None
    if name == "replication_rate":
        confirmed = _as_bool(_first(record, "independent_confirmed", "replication_passed", "independent_confirmation_passed", "confirmed"))
        entered = _as_bool(_first(record, "entered_confirmation", "confirmation_attempted", default=True))
        return float(confirmed) if confirmed is not None and entered else None
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
        row: Dict[str, Any] = {"method": method, "split": split, "n": len(group)}
        row["state_macro_f1"] = _macro_f1(group)
        for metric in METRIC_NAMES:
            if metric == "state_macro_f1":
                continue
            if metric == "fdr":
                announced = [record for record in group if _as_bool(_first(record, "new_path_announced", "discovery_announced", "announced_valid", "method_announced_valid"))]
                values = [_metric_value(record, metric, metric_weights) for record in announced]
                row[metric] = _mean(values)
                row["fdr_n"] = len(announced)
                continue
            values = [_metric_value(record, metric, metric_weights) for record in group]
            row[metric] = _mean(values)
        # Useful denominators make sparse metrics auditable in CSV/report output.
        for metric in METRIC_NAMES:
            if metric == "state_macro_f1":
                continue
            if metric == "fdr":
                continue
            row[f"{metric}_n"] = sum(_metric_value(record, metric, metric_weights) is not None for record in group)
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
        samples: List[float] = []
        for _ in range(n_boot):
            sampled = [cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values]
            flat = [item for cluster in sampled for item in cluster]
            if metric == "state_macro_f1":
                value = _macro_f1(flat)
            else:
                value = _mean(_metric_value(item, metric, metric_weights) for item in flat)
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

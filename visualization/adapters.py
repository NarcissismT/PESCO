"""Adapters from the prototype's nested trajectory objects to report rows.

This module is intentionally an offline boundary.  It can consume dictionaries
or ``Trajectory``-like objects after a trusted verifier has run, but it is not
imported by environments or policy prompts.  Hidden labels are only copied
when the caller explicitly supplies ``ground_truth_by_question``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _dict(value: Any, **kwargs: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
        if kwargs.get("include_hidden") is False:
            # A caller may accidentally pass ``Trajectory.to_dict(True)`` as a
            # plain mapping.  Keep this adapter safe at the serialization
            # boundary by scrubbing verifier-only payloads.
            if "world_id" in result:
                result["world_id"] = "hidden_from_agent"
            for key in ("hidden_world_id", "hidden_outputs", "latent_effect", "leakage", "confounding", "ground_truth_state"):
                result.pop(key, None)
        return result
    method = getattr(value, "to_dict", None)
    if callable(method):
        try:
            return dict(method(**kwargs))
        except TypeError:
            return dict(method())
    return {}


def trajectory_to_record(
    trajectory: Any,
    *,
    ground_truth: Optional[str] = None,
    split: str = "all",
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Flatten a core ``Trajectory`` into one terminal episode row.

    ``ground_truth`` is intentionally an explicit argument.  Without it, the
    row contains only the policy-visible terminal verdict and state metrics stay
    unavailable rather than being spuriously perfect.
    """
    raw = _dict(trajectory, include_hidden=False)
    outputs = raw.get("outputs") if isinstance(raw.get("outputs"), list) else []
    verdicts = raw.get("verdicts") if isinstance(raw.get("verdicts"), list) else []
    output = next((dict(item) for item in reversed(outputs) if isinstance(item, Mapping)), {})
    verdict = next((dict(item) for item in reversed(verdicts) if isinstance(item, Mapping)), {})
    confirmation = verdict.get("independent_confirmation")
    if isinstance(confirmation, Mapping):
        confirmation_passed = confirmation.get("passed")
    else:
        confirmation_passed = verdict.get("independent_confirmation_passed")
    question_id = raw.get("question_id", "unknown_question")
    record: Dict[str, Any] = {
        "record_type": "trajectory_terminal",
        "method": method or output.get("method", raw.get("method", "Unknown")),
        "split": split,
        "question_id": question_id,
        "world_id": raw.get("world_id", "hidden_from_agent"),
        "branch_id": raw.get("branch_id", ""),
        "selected_action": output.get("action", raw.get("selected_action")),
        "predicted_state": verdict.get("evidence_state", verdict.get("state")),
        "valid_claim": verdict.get("validity_pass"),
        "independent_confirmed": confirmation_passed,
        "cost": raw.get("total_cost", verdict.get("execution_cost", output.get("execution_cost"))),
        "utility": raw.get("utility", raw.get("verified_scientific_utility")),
        "turn": len(outputs),
        "validity_signals": output.get("validity_signals", []),
        "belief_score": raw.get("belief_after", raw.get("belief_before")),
        "discovered_gain": verdict.get("discovered_gain", 0.0),
        "new_path_verified": bool(verdict.get("autonomous", False) and verdict.get("certificate_pass", False)),
    }
    if ground_truth is not None:
        record["true_state"] = ground_truth
    return record


def trajectories_to_records(
    trajectories: Iterable[Any],
    *,
    ground_truth_by_question: Optional[Mapping[str, str]] = None,
    split: str = "all",
) -> list[Dict[str, Any]]:
    labels = ground_truth_by_question or {}
    rows = []
    for trajectory in trajectories:
        raw = _dict(trajectory, include_hidden=False)
        question_id = str(raw.get("question_id", "unknown_question"))
        rows.append(trajectory_to_record(trajectory, ground_truth=labels.get(question_id), split=split))
    return rows


def write_trajectory_records(
    trajectories: Iterable[Any],
    destination: str | Path,
    *,
    ground_truth_by_question: Optional[Mapping[str, str]] = None,
    split: str = "all",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a canonical JSON result file for :mod:`PESCO.visualization.cli`."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "pesco_results_v0.1",
        "metadata": dict(metadata or {}),
        "records": trajectories_to_records(trajectories, ground_truth_by_question=ground_truth_by_question, split=split),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

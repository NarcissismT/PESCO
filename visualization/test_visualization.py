"""Fast smoke tests for the standalone report pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:  # package import (pytest / ``python -m PESCO.visualization...``)
    from .demo import generate_demo_records
    from .adapters import trajectory_to_record
    from .metrics import aggregate_metrics, confusion_rows, load_records
    from .plots import plot_branch_trajectory
    from .report import write_report
except ImportError:  # stdlib unittest discovery with ``-s PESCO``
    from visualization.demo import generate_demo_records
    from visualization.adapters import trajectory_to_record
    from visualization.metrics import aggregate_metrics, confusion_rows, load_records
    from visualization.plots import plot_branch_trajectory
    from visualization.report import write_report


def test_demo_has_four_states_and_methods():
    records = generate_demo_records(17, questions=4)
    assert len(records) == 4 * 4 * 4
    assert {row["true_state"] for row in records} == {"Supported", "Refuted", "Insufficient", "Invalid"}
    assert {row["method"] for row in records} == {"Base", "GRPO-FourState", "PESCO-Offline", "PESCO-Full"}


def test_aggregation_and_confusion():
    records = generate_demo_records(17, questions=4)
    summary = aggregate_metrics(records)
    assert summary
    assert all("vrs" in row and "state_macro_f1" in row for row in summary)
    confusion = confusion_rows(records)
    assert confusion
    assert sum(row["count"] for row in confusion) == len(records)


def test_json_and_jsonl_loading(tmp_path: Path):
    records = generate_demo_records(3, questions=1)
    json_path = tmp_path / "results.json"
    json_path.write_text(json.dumps({"records": records}), encoding="utf-8")
    assert len(load_records(json_path)) == len(records)
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    assert len(load_records(jsonl_path)) == len(records)


def test_report_writes_artifacts(tmp_path: Path):
    records = generate_demo_records(17, questions=2)
    result = write_report(records, tmp_path, bootstrap=10, formats=("png", "svg"))
    assert Path(result["report"]).exists()
    assert (tmp_path / "metrics_overall.csv").exists()
    assert (tmp_path / "overview_metrics.png").exists()
    assert (tmp_path / "overview_metrics.svg").exists()


def test_trajectory_adapter_does_not_invent_ground_truth():
    trajectory = {
        "question_id": "q1",
        "outputs": [{"action": "continue_current_method", "method": "method_a", "execution_cost": 1.0}],
        "verdicts": [{"evidence_state": "supported", "validity_pass": True}],
        "total_cost": 1.0,
    }
    row = trajectory_to_record(trajectory)
    assert "true_state" not in row
    assert row["predicted_state"] == "supported"
    assert trajectory_to_record(trajectory, ground_truth="supported")["true_state"] == "supported"
    hidden = dict(trajectory, world_id="world_01", latent_effect=0.2, ground_truth_state="Supported")
    scrubbed = trajectory_to_record(hidden)
    assert scrubbed["world_id"] == "hidden_from_agent"
    assert "true_state" not in scrubbed


def test_branch_plot_aggregates_large_branch_logs(tmp_path: Path):
    records = []
    methods = ("Base", "GRPO-FourState", "PESCO-Offline", "PESCO-Full")
    for branch in range(40):
        for turn in (1, 2, 3):
            records.append({
                "method": methods[branch % len(methods)],
                "world_state": ("Supported", "Refuted", "Insufficient", "Invalid")[branch % 4],
                "branch_id": f"branch_{branch:03d}",
                "turn": turn,
                "utility": (branch % 7) / 7 + 0.05 * turn,
            })
    paths = plot_branch_trajectory(records, tmp_path, ("png", "svg"))
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.exists() for path in paths)


class VisualizationSmokeTests(unittest.TestCase):
    """stdlib-unittest mirror so CI does not require pytest."""

    def test_demo_has_four_states_and_methods_unittest(self):
        test_demo_has_four_states_and_methods()

    def test_aggregation_and_confusion_unittest(self):
        test_aggregation_and_confusion()

    def test_json_and_jsonl_loading_unittest(self):
        with tempfile.TemporaryDirectory() as directory:
            test_json_and_jsonl_loading(Path(directory))

    def test_report_writes_artifacts_unittest(self):
        with tempfile.TemporaryDirectory() as directory:
            test_report_writes_artifacts(Path(directory))

    def test_trajectory_adapter_does_not_invent_ground_truth_unittest(self):
        test_trajectory_adapter_does_not_invent_ground_truth()

    def test_branch_plot_aggregates_large_branch_logs_unittest(self):
        with tempfile.TemporaryDirectory() as directory:
            test_branch_plot_aggregates_large_branch_logs(Path(directory))

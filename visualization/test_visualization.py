"""Fast smoke tests for the standalone report pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:  # package import (pytest / ``python -m PESCO.visualization...``)
    from .demo import generate_demo_records
    from .adapters import trajectory_to_record
    from .metrics import aggregate_metrics, bootstrap_ci, conditional_metric_stats, confusion_rows, load_records
    from .plots import plot_branch_trajectory
    from .report import write_report
    from research_strategy_optimization.evaluation.experiment_scaffolds import (
        experiment_b_zero_shot_diagnostic,
        experiment_c_state_reward_diagnostic,
    )
except ImportError:  # stdlib unittest discovery with ``-s PESCO``
    from visualization.demo import generate_demo_records
    from visualization.adapters import trajectory_to_record
    from visualization.metrics import aggregate_metrics, bootstrap_ci, conditional_metric_stats, confusion_rows, load_records
    from visualization.plots import plot_branch_trajectory
    from visualization.report import write_report
    from research_strategy_optimization.evaluation.experiment_scaffolds import (
        experiment_b_zero_shot_diagnostic,
        experiment_c_state_reward_diagnostic,
    )


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


def test_conditional_denominators_exclude_irrelevant_worlds():
    records = [
        {
            "method": "M",
            "split": "id",
            "question_id": "q1",
            "world_pair_id": "pair1",
            "true_state": "Supported",
            "flip_eligible": True,
            "flip_correct": True,
            "required_switch": False,
            "invalid_repair_eligible": False,
            "insufficient_handling_eligible": False,
            "confirmation_eligible": True,
            "independent_confirmed": True,
        },
        {
            "method": "M",
            "split": "id",
            "question_id": "q1",
            "world_pair_id": "pair1",
            "true_state": "Refuted",
            "flip_eligible": True,
            "flip_correct": True,
            "required_switch": True,
            "effective_switch": True,
            "invalid_repair_eligible": False,
            "insufficient_handling_eligible": False,
            "confirmation_eligible": True,
            "independent_confirmed": False,
        },
        {
            "method": "M",
            "split": "id",
            "question_id": "q1",
            "world_pair_id": "pair2",
            "true_state": "Invalid",
            "flip_eligible": False,
            "required_switch": False,
            "invalid_repair_eligible": True,
            "invalid_repaired": True,
            "insufficient_handling_eligible": False,
            "confirmation_eligible": False,
            "independent_confirmed": False,
        },
        {
            "method": "M",
            "split": "id",
            "question_id": "q1",
            "world_pair_id": "pair3",
            "true_state": "Insufficient",
            "flip_eligible": False,
            "required_switch": False,
            "invalid_repair_eligible": False,
            "insufficient_handling_eligible": True,
            "underpower_handled": True,
            "confirmation_eligible": False,
            "independent_confirmed": False,
        },
    ]
    row = aggregate_metrics(records, by_split=True)[0]
    assert row["flip_accuracy"] == 1.0
    assert row["flip_eligible_n"] == 1
    assert row["effective_switch_rate"] == 1.0
    assert row["required_switch_n"] == 1
    assert row["invalid_repair_rate"] == 1.0
    assert row["invalid_repair_n"] == 1
    assert row["underpower_handling"] == 1.0
    assert row["insufficient_handling_n"] == 1
    assert row["replication_rate"] == 0.5
    assert row["confirmation_eligible_n"] == 2
    assert conditional_metric_stats(records, "flip_accuracy")["denominator"] == 1


def test_single_cluster_bootstrap_is_na():
    records = [
        {"method": "M", "split": "id", "question_id": "only", "true_state": "Supported", "vrs": 1.0},
        {"method": "M", "split": "id", "question_id": "only", "true_state": "Refuted", "vrs": 0.5},
    ]
    intervals = bootstrap_ci(records, "vrs", n_boot=100)
    assert intervals[("M", "id")] == (None, None)


def test_missing_conditional_outcomes_stay_in_fail_closed_denominator():
    records = [
        {
            "method": "M",
            "split": "id",
            "question_id": "q1",
            "world_pair_id": "pair1",
            "true_state": "Supported",
            "flip_eligible": True,
            # flip_correct intentionally omitted
            "required_switch": True,
            # effective_switch intentionally omitted
            "confirmation_eligible": True,
            # independent_confirmed intentionally omitted
        },
        {
            "method": "M",
            "split": "id",
            "question_id": "q1",
            "world_pair_id": "pair1",
            "true_state": "Refuted",
            "flip_eligible": True,
            "required_switch": False,
            # The second side is also missing an outcome.
        },
    ]
    row = aggregate_metrics(records, by_split=True)[0]
    assert row["flip_accuracy"] == 0.0
    assert row["flip_eligible_n"] == 1
    assert row["effective_switch_rate"] == 0.0
    assert row["required_switch_n"] == 1
    assert row["replication_rate"] == 0.0
    assert row["confirmation_eligible_n"] == 1


def test_evaluator_diagnostic_is_not_state_macro_f1():
    records = [{
        "method": "M",
        "split": "id",
        "question_id": "q1",
        "true_state": "Supported",
        "predicted_state": "Supported",
        "policy_predicted_state": None,
        "evaluator_diagnostic_state": "Supported",
        "state_prediction_source": "evaluator_diagnostic",
        "state_metric_eligible": False,
    }]
    row = aggregate_metrics(records, by_split=True)[0]
    assert row["state_macro_f1"] is None


def test_demo_discovery_bonus_is_uniformly_disabled():
    records = generate_demo_records(17, questions=4)
    assert {row["discovery_utility"] for row in records} == {0.0}
    assert {row["discovery_bonus_policy"] for row in records} == {"disabled_fixed_action_space"}
    assert not any(row["new_path_verified"] for row in records)


def test_experiment_b_c_are_fail_closed_diagnostic_scaffolds():
    records = generate_demo_records(17, questions=2)
    experiment_b = experiment_b_zero_shot_diagnostic(records)
    experiment_c = experiment_c_state_reward_diagnostic(records)
    assert experiment_b["status"] == "diagnostic_only"
    assert experiment_b["formal_comparison_authorized"] is False
    assert experiment_b["pass"] is False
    assert experiment_b["gates"]["policy_state_separated_from_evaluator_diagnostic"] is True
    assert experiment_c["status"] == "diagnostic_only"
    assert experiment_c["formal_comparison_authorized"] is False
    assert experiment_c["pass"] is False
    assert experiment_c["interpretation"]["cannot_claim_pesco_advantage"] is True


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

    def test_conditional_denominators_exclude_irrelevant_worlds_unittest(self):
        test_conditional_denominators_exclude_irrelevant_worlds()

    def test_single_cluster_bootstrap_is_na_unittest(self):
        test_single_cluster_bootstrap_is_na()

    def test_missing_conditional_outcomes_stay_in_fail_closed_denominator_unittest(self):
        test_missing_conditional_outcomes_stay_in_fail_closed_denominator()

    def test_evaluator_diagnostic_is_not_state_macro_f1_unittest(self):
        test_evaluator_diagnostic_is_not_state_macro_f1()

    def test_demo_discovery_bonus_is_uniformly_disabled_unittest(self):
        test_demo_discovery_bonus_is_uniformly_disabled()

    def test_experiment_b_c_are_fail_closed_diagnostic_scaffolds_unittest(self):
        test_experiment_b_c_are_fail_closed_diagnostic_scaffolds()

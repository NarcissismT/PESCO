from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import research_strategy_optimization.evaluation.shortcut_probes as shortcut_module

from research_strategy_optimization.evaluation.shortcut_probes import (
    ACTION_NAMES,
    FEATURE_SETS,
    MODEL_NAMES,
    feature_names,
    observation_features,
    run_shortcut_probe,
    sklearn_status,
)


def _observation(*, effect: float, overlap: float = 0.0, confirmation: float = 0.0) -> dict:
    return {
        "question_id": "public_question",
        "turn": 1,
        "current_method": "method_a",
        "effect_estimate": effect,
        "confidence_interval": [effect - 0.02, effect + 0.02],
        "sample_size": 96,
        "seed_count": 8,
        "remaining_budget": 5,
        "hypothesis_probability": 0.5,
        "active_hypothesis_id": "H_A",
        "raw_evidence": {
            "treatment_confounder_correlation": effect,
            "group_overlap_count": overlap,
            "replication_effect_delta": 0.0,
            "replication_ci_width": 0.04,
            "replication_sample_size": 96.0,
            "replication_seed_count": 8.0,
            "log_confirmation_pass_rate": confirmation,
            "log_validity_count": 2.0,
            "log_repeated_runs": 8.0,
            "log_protocol_change_count": 0.0,
        },
    }


def _toy_payload() -> dict:
    examples = []
    # Four utility labels and two public feature regimes, with question clusters
    # repeated across worlds as in the benchmark.  Hidden target-action fields are
    # intentionally absent: labels come from branch utility argmax.
    for split, count, offset in (("train", 16, 0.0), ("dev", 8, 0.01), ("diagnostic_ood", 8, -0.01)):
        for index in range(count):
            label = index % len(ACTION_NAMES)
            utilities = [0.0] * len(ACTION_NAMES)
            utilities[label] = 1.0
            examples.append({
                "observation": _observation(
                    effect=(label - 1.5) * 0.1 + offset,
                    overlap=float(label),
                    confirmation=0.1 * label,
                ),
                "branch_utilities": utilities,
                "split": split,
                "question_id": f"{split}_q_{index // 4:03d}",
                "metadata": {"family": "toy_family"},
            })
    return {"schema_version": "toy_shortcut", "examples": examples}


class ShortcutProbeTests(unittest.TestCase):
    def test_feature_set_removes_only_confirmation_receipt(self) -> None:
        all_names = feature_names("all_raw")
        no_names = feature_names("without_confirmation")
        self.assertIn("log_confirmation_pass_rate", all_names)
        self.assertNotIn("log_confirmation_pass_rate", no_names)
        self.assertEqual(set(all_names) - set(no_names), {"log_confirmation_pass_rate"})
        self.assertEqual(len(observation_features(_observation(effect=0.1), "all_raw")), len(all_names))
        self.assertEqual(len(observation_features(_observation(effect=0.1), "without_confirmation")), len(no_names))
        self.assertEqual(feature_names("all_current_raw"), all_names)
        self.assertEqual(feature_names("raw_no_confirmation"), no_names)

    def test_feature_schema_follows_removed_confirmation_field(self) -> None:
        observation = _observation(effect=0.1)
        observation["raw_evidence"] = {
            key: value
            for key, value in observation["raw_evidence"].items()
            if key != "log_confirmation_pass_rate"
        }
        available = observation["raw_evidence"].keys()
        self.assertNotIn("log_confirmation_pass_rate", feature_names("all_raw", available))
        self.assertEqual(
            len(observation_features(observation, "all_raw", available)),
            len(observation_features(observation, "without_confirmation", available)),
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = _toy_payload()
            for record in payload["examples"]:
                record["observation"]["raw_evidence"].pop("log_confirmation_pass_rate", None)
            result = run_shortcut_probe(payload, output_dir=directory, bootstrap_replicates=5)
            self.assertFalse(result["raw_feature_schema"]["confirmation_summary_present"])
            self.assertIn("log_confirmation_pass_rate", result["raw_feature_schema"]["omitted_names"])
            self.assertFalse(result["feature_sets"]["all_raw"]["confirmation_feature_included"])

    def test_public_feature_vector_does_not_depend_on_ids_or_metadata(self) -> None:
        first = _observation(effect=0.2, overlap=3, confirmation=0.4)
        second = dict(first)
        second["question_id"] = "different_hidden-looking-id"
        second["task_family"] = "another_public_label"
        np.testing.assert_array_equal(observation_features(first), observation_features(second))

    def test_probe_runs_without_sklearn_using_explicit_numpy_implementations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_shortcut_probe(
                _toy_payload(),
                output_dir=directory,
                bootstrap_replicates=20,
                seed=11,
            )
            status = sklearn_status()
            if status["available"]:
                self.assertEqual(result["status"], "completed_sklearn")
            else:
                self.assertEqual(result["status"], "completed_numpy_fallback")
                self.assertTrue(result["fallback_used"])
            self.assertEqual(set(result["models"]), {
                f"{feature_set}:{model}" for feature_set in FEATURE_SETS for model in MODEL_NAMES
            })
            for model_result in result["models"].values():
                self.assertEqual(model_result["status"], "completed")
                self.assertIn("metrics_by_split", model_result)
                self.assertEqual(model_result["metrics_by_split"]["diagnostic_ood"]["row_count"], 8)
            saved = json.loads((Path(directory) / "shortcut_probe_result.json").read_text())
            self.assertEqual(saved["schema_version"], result["schema_version"])
            self.assertEqual(saved["label_source"], "argmax_evaluator_branch_utility_not_hidden_target_action")
            self.assertTrue(saved["sklearn_required_for_formal_claim"])
            self.assertFalse(saved["formal_comparison_authorized"])

    def test_strict_sklearn_is_fail_closed_when_dependency_missing(self) -> None:
        if sklearn_status()["available"]:
            self.skipTest("sklearn is installed in this environment")
        result = run_shortcut_probe(_toy_payload(), strict_sklearn=True)
        self.assertEqual(result["status"], "fail_closed_sklearn_unavailable")
        self.assertFalse(result["fallback_used"])

    def test_public_dataset_without_branch_labels_fails_closed(self) -> None:
        payload = {"schema_version": "public", "examples": [{"observation": _observation(effect=0.0), "split": "train"}]}
        result = run_shortcut_probe(payload)
        self.assertEqual(result["status"], "fail_closed_invalid_dataset")

    def test_train_and_holdout_question_overlap_fails_closed(self) -> None:
        payload = _toy_payload()
        payload["examples"][16]["question_id"] = payload["examples"][0]["question_id"]
        result = run_shortcut_probe(payload)
        self.assertEqual(result["status"], "fail_closed_split_cluster_overlap")
        self.assertFalse(result["split_boundary_pass"])

    def test_missing_audit_question_id_does_not_use_neutral_policy_id_as_cluster(self) -> None:
        payload = _toy_payload()
        for record in payload["examples"]:
            record.pop("question_id", None)
        result = run_shortcut_probe(payload, bootstrap_replicates=5)
        self.assertTrue(result["split_boundary_pass"])
        metrics = result["models"]["all_raw:logistic_regression"]["metrics_by_split"]["diagnostic_ood"]
        self.assertEqual(metrics["question_cluster_count"], metrics["row_count"])

    def test_requested_model_failure_is_fail_closed(self) -> None:
        original = shortcut_module._make_model

        def failing_model(name, *, seed, sklearn_modules):
            if name == "random_forest":
                raise RuntimeError("injected model failure")
            return original(name, seed=seed, sklearn_modules=sklearn_modules)

        with mock.patch.object(shortcut_module, "_make_model", side_effect=failing_model):
            result = run_shortcut_probe(_toy_payload(), models=("logistic_regression", "random_forest"))
        self.assertEqual(result["status"], "fail_closed_model_error")
        self.assertIn("all_raw:random_forest", result["failed_models"])
        self.assertFalse(result["formal_comparison_authorized"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.evaluation.tier1_v05_frozen_final import (
    V05_FINAL_ID_FAMILIES,
    V05_FINAL_OOD_FAMILIES,
    audit_latent_generator_signatures,
    audit_public_manifest,
    build_baseline_selection_receipt,
    build_freeze_receipt,
    build_tier1_v05_frozen_final_benchmark,
    collect_v05_environment_receipts,
    latent_signature,
    _selection_results_have_dev_evidence as _producer_selection_results_have_dev_evidence,
)
from scripts.audit_tier1_v05_frozen_final import (
    _baseline_receipt_valid,
    _environment_confirmation_receipt_checks,
    _selection_results_have_dev_evidence as _audit_selection_results_have_dev_evidence,
)


class Tier1V05FrozenFinalTests(unittest.TestCase):
    @staticmethod
    def _write_valid_algorithm_and_hyperparameters(config: Path, hyper: Path) -> None:
        """Minimal structured fixtures accepted by the freeze validator."""

        config.write_text(json.dumps({
            "schema_version": "test_algorithm_v1",
            "algorithm": {"name": "fixture", "budget": {"max_optimizer_steps": 8}},
        }), encoding="utf-8")
        hyper.write_text(json.dumps({
            "schema_version": "test_hyperparameters_v1",
            "hyperparameters": {
                "trainer": {"max_optimizer_steps": 8, "epochs": 1, "batch_size": 1},
            },
        }), encoding="utf-8")

    @staticmethod
    def _write_dev_and_selection(dev: Path, selection: Path, payload: dict) -> None:
        dev.write_text(json.dumps({
            "schema_version": "test_dev_manifest_v1",
            "questions": [{"question_id": "dev_q_1", "split": "dev", "world_count": 4}],
        }), encoding="utf-8")
        selection.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _unrelated_development_manifest(benchmark):
        world = benchmark.worlds[0]
        return {
            "questions": [{
                "question_id": "development_only",
                "family": "legacy_family",
                "variant": 1,
                "worlds": [{
                    "world_id": "development_world",
                    "kind": world.kind,
                    "true_effect_a": world.true_effect_a + 0.001234,
                    "true_effect_b": world.true_effect_b + 0.002345,
                    "noise_scale": world.noise_scale + 0.003456,
                    "initial_samples": world.initial_samples + 1,
                    "leakage": world.leakage,
                    "confounding": world.confounding,
                    "metric_mismatch": world.metric_mismatch,
                    "protocol_invalid": world.protocol_invalid,
                }],
            }],
        }

    def test_frozen_profile_has_48_id_and_48_ood_clusters(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        self.assertEqual(len(benchmark.questions), 96)
        self.assertEqual(len(benchmark.worlds), 384)
        self.assertEqual(len(benchmark.final_id_questions), 48)
        self.assertEqual(len(benchmark.final_ood_questions), 48)
        self.assertEqual(
            {family: sum(q.family == family for q in benchmark.final_id_questions) for family in V05_FINAL_ID_FAMILIES},
            {family: 8 for family in V05_FINAL_ID_FAMILIES},
        )
        self.assertEqual(
            {
                family: sum(q.family == family for q in benchmark.final_ood_questions)
                for family in V05_FINAL_OOD_FAMILIES
            },
            {family: 12 for family in V05_FINAL_OOD_FAMILIES},
        )

    def test_latent_and_generator_signatures_are_unique_and_ood_is_new(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        self.assertEqual(len({latent_signature(world) for world in benchmark.worlds}), 384)
        self.assertEqual(len({q.generator_signature for q in benchmark.questions}), 96)
        audit = audit_latent_generator_signatures(
            benchmark, [self._unrelated_development_manifest(benchmark)]
        )
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["new_ood_family_count"], 4)
        self.assertEqual(audit["latent_overlap_count"], 0)

    def test_development_latent_overlap_is_detected_without_id_overlap(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        world = benchmark.worlds[0]
        # A synthetic development manifest with a copied latent world must close the
        # latent gate even though its IDs are unrelated.
        hidden = {
            "questions": [{
                "question_id": "development_only",
                "family": "legacy_family",
                "variant": 1,
                "worlds": [{
                    "world_id": "development_world",
                    "kind": world.kind,
                    "true_effect_a": world.true_effect_a,
                    "true_effect_b": world.true_effect_b,
                    "noise_scale": world.noise_scale,
                    "initial_samples": world.initial_samples,
                    "leakage": world.leakage,
                    "confounding": world.confounding,
                    "metric_mismatch": world.metric_mismatch,
                    "protocol_invalid": world.protocol_invalid,
                    "question_family": "renamed_legacy_family",
                }],
            }],
        }
        audit = audit_latent_generator_signatures(benchmark, [hidden])
        self.assertFalse(audit["gates"]["latent_outputs_disjoint"])
        self.assertGreaterEqual(audit["latent_overlap_count"], 1)

    def test_public_manifest_is_opaque_and_locked(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        public = benchmark.manifest(include_hidden=False)
        audit = audit_public_manifest(public)
        self.assertTrue(audit["pass"])
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("true_effect_a", encoded)
        self.assertNotIn("true_effect_b", encoded)
        self.assertNotIn("legacy_target_actions_audit_only", encoded)
        self.assertTrue(public["final_access"]["locked"])

    def test_hidden_manifest_contains_recipe_and_public_does_not(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        hidden = benchmark.manifest(include_hidden=True)
        public = benchmark.manifest(include_hidden=False)
        self.assertIn("generator_recipes", hidden)
        self.assertIn("worlds", hidden["questions"][0])
        self.assertNotIn("generator_recipes", public)
        self.assertNotIn("worlds", public["questions"][0])

    def test_environment_observation_does_not_expose_hidden_family(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        question = benchmark.final_ood_questions[0]
        world = question.worlds[0]
        env = benchmark.make_environment(question.question_id)
        env.reset(question.policy_question_id, world.world_id, seed=17)
        self.assertEqual(env.visible_observation().task_family, "v05_final_public_task")
        self.assertNotEqual(env.visible_observation().task_family, question.family)

    def test_bounded_environment_receipts_have_atomic_sums_and_eight_seeds(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        receipt = collect_v05_environment_receipts(benchmark, question_limit=1)
        self.assertEqual(receipt["question_count_collected"], 1)
        self.assertEqual(receipt["world_count_collected"], 4)
        self.assertEqual(receipt["action_seed_receipt_count"], 4 * 4 * 8)
        self.assertTrue(receipt["independent_confirmation_not_copied"])
        for row in receipt["rows"]:
            self.assertAlmostEqual(
                row["utility"], sum(row["reward_components"].values()), places=12,
            )
        self.assertTrue(all(
            _environment_confirmation_receipt_checks(receipt["rows"]).values()
        ))

    def test_independent_confirmation_audit_rejects_tampered_private_rows(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        receipt = collect_v05_environment_receipts(benchmark, question_limit=1)
        eligible = next(row for row in receipt["rows"] if row["confirmation_eligible"])
        ineligible = next(row for row in receipt["rows"] if not row["confirmation_eligible"])

        eligible["confirmation_seed"] = 997
        checks = _environment_confirmation_receipt_checks(receipt["rows"])
        self.assertFalse(checks["seeds_preregistered"])
        eligible["confirmation_seed"] = 103

        eligible["confirmation_dataset_hash"] = eligible["dataset_hash"]
        checks = _environment_confirmation_receipt_checks(receipt["rows"])
        self.assertFalse(checks["eligible_hashes_independent"])
        eligible["confirmation_dataset_hash"] = "sha256:" + "a" * 64
        eligible["confirmation_split_hash"] = eligible["split_hash"]
        checks = _environment_confirmation_receipt_checks(receipt["rows"])
        self.assertFalse(checks["eligible_hashes_independent"])
        eligible["confirmation_split_hash"] = "sha256:" + "b" * 64

        ineligible["confirmation_passed"] = False
        checks = _environment_confirmation_receipt_checks(receipt["rows"])
        self.assertFalse(checks["eligibility_semantics_consistent"])
        ineligible["confirmation_passed"] = None
        ineligible["confirmation_split_hash"] = "sha256:" + "b" * 64
        checks = _environment_confirmation_receipt_checks(receipt["rows"])
        self.assertFalse(checks["eligibility_semantics_consistent"])

    def test_freeze_receipt_fails_closed_without_tag_and_pre_final_baseline(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        signature_audit = audit_latent_generator_signatures(
            benchmark, [self._unrelated_development_manifest(benchmark)]
        )
        public_audit = audit_public_manifest(benchmark.manifest(include_hidden=False))
        receipt = build_freeze_receipt(
            repo_root=Path(__file__).resolve().parents[2],
            benchmark=benchmark,
            signature_audit=signature_audit,
            public_audit=public_audit,
            baseline_selection=None,
            explicit_sign=True,
        )
        self.assertFalse(receipt["signed"])
        self.assertEqual(receipt["status"], "pending_clean_commit_tag")

    def test_baseline_selection_receipt_binds_dev_and_frozen_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "dev.json"
            config = root / "algorithm.yaml"
            hyper = root / "hyperparameters.json"
            selection = root / "selection.json"
            dev.write_text(json.dumps({
                "schema_version": "test_dev_manifest_v1",
                "questions": [{"question_id": "dev_q_1", "split": "dev", "world_count": 4}],
            }), encoding="utf-8")
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            selection.write_text(json.dumps({
                "selection_split": "dev",
                "selected_baseline": "NoFlip",
                "candidate_metrics": {"NoFlip": {"normalized_regret": 0.1}},
            }), encoding="utf-8")
            receipt = build_baseline_selection_receipt(
                selected_baseline="NoFlip",
                selection_split="dev",
                development_manifest=dev,
                algorithm_config=config,
                hyperparameters=hyper,
                selection_results=selection,
            )
            self.assertTrue(receipt["selection_locked_before_final"])
            self.assertTrue(receipt["algorithm_hyperparameters_frozen"])
            self.assertTrue(receipt["selection_receipt_digest"].startswith("sha256:"))
            for key in (
                "development_manifest_digest", "algorithm_config_digest",
                "hyperparameters_digest",
            ):
                self.assertTrue(receipt[key].startswith("sha256:"))

    def test_baseline_selection_rejects_empty_reserved_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "dev.json"
            config = root / "algorithm.yaml"
            hyper = root / "hyperparameters.json"
            dev.write_text(json.dumps({
                "split": "dev", "question_ids": [], "status": "reserved_before_tier1",
            }), encoding="utf-8")
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            with self.assertRaises(ValueError):
                build_baseline_selection_receipt(
                    selected_baseline="NoFlip",
                    selection_split="dev",
                    development_manifest=dev,
                    algorithm_config=config,
                    hyperparameters=hyper,
                )

    def test_selection_must_choose_minimum_normalized_regret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, config, hyper, selection = (root / name for name in (
                "dev.json", "algorithm.json", "hyper.json", "selection.json",
            ))
            self._write_dev_and_selection(dev, selection, {
                "selection_split": "dev",
                "selected_baseline": "SFT",
                "candidate_metrics": {
                    "SFT": {"normalized_regret": 0.20, "n_questions": 2},
                    "NoFlip": {"normalized_regret": 0.10, "n_questions": 2},
                },
            })
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            self.assertFalse(_producer_selection_results_have_dev_evidence(
                selection, selected_baseline="SFT", selection_split="dev",
            ))
            self.assertFalse(_audit_selection_results_have_dev_evidence(
                selection, selected_baseline="SFT", selection_split="dev",
            ))
            with self.assertRaises(ValueError):
                build_baseline_selection_receipt(
                    selected_baseline="SFT", selection_split="dev",
                    development_manifest=dev, algorithm_config=config,
                    hyperparameters=hyper, selection_results=selection,
                )

    def test_selection_tie_policy_is_deterministic_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, config, hyper, selection = (root / name for name in (
                "dev.json", "algorithm.json", "hyper.json", "selection.json",
            ))
            self._write_dev_and_selection(dev, selection, {
                "selection_split": "dev",
                "selected_baseline": "Alpha",
                "candidate_metrics": {
                    "Beta": {"normalized_regret": 0.10},
                    "Alpha": {"normalized_regret": 0.10},
                },
            })
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            self.assertTrue(_producer_selection_results_have_dev_evidence(
                selection, selected_baseline="Alpha", selection_split="dev",
            ))
            self.assertTrue(_audit_selection_results_have_dev_evidence(
                selection, selected_baseline="Alpha", selection_split="dev",
            ))
            selection_payload = json.loads(selection.read_text(encoding="utf-8"))
            selection_payload["selected_baseline"] = "Beta"
            selection.write_text(json.dumps(selection_payload), encoding="utf-8")
            self.assertFalse(_producer_selection_results_have_dev_evidence(
                selection, selected_baseline="Beta", selection_split="dev",
            ))

            # An explicit first-in-file policy permits the first tied candidate,
            # while still rejecting a later tied candidate.
            selection_payload.update({
                "selected_baseline": "Beta",
                "tie_policy": "first_in_file",
                "candidate_metrics": [
                    {"method": "Beta", "normalized_regret": 0.10},
                    {"method": "Alpha", "normalized_regret": 0.10},
                ],
            })
            selection.write_text(json.dumps(selection_payload), encoding="utf-8")
            self.assertTrue(_producer_selection_results_have_dev_evidence(
                selection, selected_baseline="Beta", selection_split="dev",
            ))
            self.assertFalse(_audit_selection_results_have_dev_evidence(
                selection, selected_baseline="Alpha", selection_split="dev",
            ))

    def test_selection_metrics_reject_nonfinite_missing_and_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            base = {"selection_split": "dev", "selected_baseline": "NoFlip"}
            for candidates in (
                {"NoFlip": {}},
                {"NoFlip": {"normalized_regret": "0.1"}},
                {"NoFlip": {"normalized_regret": float("nan")}},
                [
                    {"method": "NoFlip", "normalized_regret": 0.1},
                    {"method": "NoFlip", "normalized_regret": 0.2},
                ],
            ):
                payload = dict(base, candidate_metrics=candidates)
                selection.write_text(json.dumps(payload), encoding="utf-8")
                self.assertFalse(_producer_selection_results_have_dev_evidence(
                    selection, selected_baseline="NoFlip", selection_split="dev",
                ))
                self.assertFalse(_audit_selection_results_have_dev_evidence(
                    selection, selected_baseline="NoFlip", selection_split="dev",
                ))

    def test_algorithm_and_hyperparameter_files_need_structured_budget_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, config, hyper, selection = (root / name for name in (
                "dev.json", "algorithm.yaml", "hyper.json", "selection.json",
            ))
            self._write_dev_and_selection(dev, selection, {
                "selection_split": "dev",
                "selected_baseline": "NoFlip",
                "candidate_metrics": {"NoFlip": {"normalized_regret": 0.1}},
            })
            # YAML is accepted when the optional parser is present; JSON remains
            # the portable fallback for hyperparameters.
            config.write_text(
                "schema_version: test_algorithm_v1\n"
                "algorithm:\n  name: fixture\n  budget:\n    max_optimizer_steps: 8\n",
                encoding="utf-8",
            )
            hyper.write_text(json.dumps({"hyperparameters": {"learning_rate": 0.1}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_baseline_selection_receipt(
                    selected_baseline="NoFlip", selection_split="dev",
                    development_manifest=dev, algorithm_config=config,
                    hyperparameters=hyper, selection_results=selection,
                )
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            receipt = build_baseline_selection_receipt(
                selected_baseline="NoFlip", selection_split="dev",
                development_manifest=dev, algorithm_config=config,
                hyperparameters=hyper, selection_results=selection,
            )
            self.assertEqual(receipt["selection_rule"], "min_normalized_regret")
            self.assertEqual(receipt["tie_policy"], "lexicographic")

    def test_freeze_receipt_rejects_invented_baseline_digests(self) -> None:
        benchmark = build_tier1_v05_frozen_final_benchmark()
        signature_audit = audit_latent_generator_signatures(
            benchmark, [self._unrelated_development_manifest(benchmark)]
        )
        public_audit = audit_public_manifest(benchmark.manifest(include_hidden=False))
        baseline = {
            "selected_baseline": "NoFlip",
            "selection_split": "dev",
            "selection_locked_before_final": True,
            "selection_receipt_digest": "sha256:" + "1" * 64,
            "development_manifest_digest": "sha256:" + "2" * 64,
            "algorithm_config_digest": "sha256:" + "3" * 64,
            "hyperparameters_digest": "sha256:" + "4" * 64,
            "algorithm_hyperparameters_frozen": True,
            "selection_evidence": {
                "development_manifest_path": "data/manifests/dev_v0_2.json",
                "algorithm_config_path": "research_strategy_optimization/configs/algorithms/pesco_cpu.yaml",
                "hyperparameters_path": "research_strategy_optimization/configs/algorithms/pesco_cpu.yaml",
            },
        }
        receipt = build_freeze_receipt(
            repo_root=Path(__file__).resolve().parents[2],
            benchmark=benchmark,
            signature_audit=signature_audit,
            public_audit=public_audit,
            baseline_selection=baseline,
            explicit_sign=True,
        )
        self.assertFalse(receipt["signed"])
        self.assertFalse(receipt["baseline_selection_valid"])
        self.assertFalse(receipt["algorithm_hyperparameters_frozen"])

    def test_baseline_selection_rejects_promotion_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "dev.json"
            config = root / "algorithm.yaml"
            hyper = root / "hyperparameters.json"
            for path in (dev, config, hyper):
                path.write_text("fixture", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_baseline_selection_receipt(
                    selected_baseline="NoFlip",
                    selection_split="promotion",
                    development_manifest=dev,
                    algorithm_config=config,
                    hyperparameters=hyper,
                )

    def test_clean_tagged_fixture_can_sign_receipt(self) -> None:
        # Exercise the positive signing path in an isolated temporary repository;
        # the real workspace intentionally remains unsigned until a clean tagged
        # commit is supplied by the release process.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "v05-test"], check=True)
            marker = root / "marker.txt"
            marker.write_text("frozen", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "marker.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "freeze"], check=True)
            benchmark = build_tier1_v05_frozen_final_benchmark()
            signature_audit = audit_latent_generator_signatures(
                benchmark, [self._unrelated_development_manifest(benchmark)]
            )
            public_audit = audit_public_manifest(benchmark.manifest(include_hidden=False))
            dev = root / "dev.json"
            config = root / "algorithm.yaml"
            hyper = root / "hyperparameters.json"
            selection = root / "selection.json"
            dev.write_text(json.dumps({
                "schema_version": "test_dev_manifest_v1",
                "questions": [{"question_id": "dev_q_1", "split": "dev", "world_count": 4}],
            }), encoding="utf-8")
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            selection.write_text(json.dumps({
                "selection_split": "dev",
                "selected_baseline": "NoFlip",
                "candidate_metrics": {"NoFlip": {"normalized_regret": 0.1}},
            }), encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", "dev.json", "algorithm.yaml", "hyperparameters.json", "selection.json"],
                check=True,
            )
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "freeze inputs"], check=True)
            subprocess.run(["git", "-C", str(root), "tag", "pesco-v0.5-test"], check=True)
            baseline = build_baseline_selection_receipt(
                selected_baseline="NoFlip",
                selection_split="dev",
                development_manifest=dev,
                algorithm_config=config,
                hyperparameters=hyper,
                selection_results=selection,
            )
            receipt = build_freeze_receipt(
                repo_root=root,
                benchmark=benchmark,
                signature_audit=signature_audit,
                public_audit=public_audit,
                baseline_selection=baseline,
                explicit_sign=True,
            )
            self.assertTrue(receipt["signed"])
            self.assertEqual(receipt["status"], "frozen")

    def test_independent_audit_rechecks_selection_results_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "dev.json"
            config = root / "algorithm.yaml"
            hyper = root / "hyperparameters.json"
            selection = root / "selection.json"
            dev.write_text(json.dumps({
                "questions": [{"question_id": "dev_q_1", "split": "dev", "world_count": 4}],
            }), encoding="utf-8")
            self._write_valid_algorithm_and_hyperparameters(config, hyper)
            selection.write_text(json.dumps({
                "selection_split": "dev",
                "selected_baseline": "NoFlip",
                "candidate_metrics": {"NoFlip": {"normalized_regret": 0.1}},
            }), encoding="utf-8")
            baseline = build_baseline_selection_receipt(
                selected_baseline="NoFlip",
                selection_split="dev",
                development_manifest=dev,
                algorithm_config=config,
                hyperparameters=hyper,
                selection_results=selection,
            )
            receipt = {
                "signed": False,
                "baseline_selection": baseline,
            }
            self.assertTrue(_baseline_receipt_valid(receipt, repo_root=root))
            selection.write_text(json.dumps({
                "selection_split": "dev",
                "selected_baseline": "NoFlip",
                "candidate_metrics": {"SFT": {"normalized_regret": 0.1}},
            }), encoding="utf-8")
            self.assertFalse(_baseline_receipt_valid(receipt, repo_root=root))

    def test_independent_audit_rejects_empty_frozen_baseline(self) -> None:
        self.assertFalse(_baseline_receipt_valid({
            "status": "frozen",
            "signed": False,
            "baseline_selection": {},
        }, repo_root=Path(__file__).resolve().parents[2]))


if __name__ == "__main__":
    unittest.main()

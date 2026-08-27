from __future__ import annotations

import unittest
import copy

try:
    import torch as _torch  # noqa: F401 - presence check for the optional suite
except ImportError:  # pragma: no cover - exercised in minimal NumPy/stdlib installs
    raise unittest.SkipTest("PyTorch is an optional dependency for differentiable tests")

from research_strategy_optimization.algorithms.differentiable_strategy import (
    DecisionDataset,
    DifferentiableStrategyTrainer,
    DifferentiableTrainerConfig,
    ReversalExample,
    observation_to_features,
)
from research_strategy_optimization.environments.tier1_benchmark import build_tier1_v03_benchmark
from research_strategy_optimization.evaluation.tier1_differentiable_suite import (
    collect_tier1_v03_dataset,
    evaluate_differentiable_policy,
    is_invalid_local_optimization,
)
from research_strategy_optimization.schemas import EvidenceState, HypothesisBelief, Observation, Protocol
from research_strategy_optimization.schemas import ResearchAction


class DifferentiableStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = build_tier1_v03_benchmark()
        cls.dataset = collect_tier1_v03_dataset(cls.benchmark, Protocol(protocol_version="pesco_v0_2"))

    def test_tier1_dataset_has_independent_questions_and_matched_branches(self):
        self.assertEqual(len(self.dataset.examples), 48)
        self.assertEqual(self.dataset.provenance["branch_groups"], 48)
        self.assertEqual(self.dataset.provenance["question_world_group_count"], 48)
        self.assertEqual(self.dataset.provenance["action_level_row_count"], 192)
        self.assertEqual(self.dataset.provenance["exploration_seed_observations"], 768)
        self.assertEqual(self.dataset.provenance["seed_level_observation_count"], 768)
        self.assertEqual(len({example.question_id for example in self.dataset.examples}), 12)
        self.assertEqual(sum(example.split == "train" for example in self.dataset.examples), 32)
        self.assertEqual(sum(example.split == "dev" for example in self.dataset.examples), 8)
        self.assertEqual(sum(example.split == "diagnostic_ood" for example in self.dataset.examples), 8)
        self.assertGreaterEqual(len(self.dataset.reversals), 1)

    def test_public_feature_vector_does_not_depend_on_world_identifier(self):
        first = Observation("q", 1, "method_a", .1, .01, .2, 120, 4, 5)
        second = Observation("q", 1, "method_a", .1, .01, .2, 120, 4, 5)
        self.assertTrue((observation_to_features(first) == observation_to_features(second)).all())
        self.assertEqual(observation_to_features(first).ndim, 1)

    def test_dataset_round_trip_preserves_named_hypothesis_beliefs(self):
        original = copy.deepcopy(self.dataset)
        observation = original.examples[0].observation
        original.examples[0].observation = Observation(
            question_id=observation.question_id,
            turn=observation.turn,
            current_method=observation.current_method,
            effect_estimate=observation.effect_estimate,
            ci_low=observation.ci_low,
            ci_high=observation.ci_high,
            sample_size=observation.sample_size,
            seed_count=observation.seed_count,
            remaining_budget=observation.remaining_budget,
            metric_name=observation.metric_name,
            validity_signals=observation.validity_signals,
            history_summary=observation.history_summary,
            hypothesis_probability=0.81,
            active_hypothesis_id="H_A",
            hypothesis_beliefs=(
                HypothesisBelief("H_A", 0.81, turn=observation.turn),
                HypothesisBelief("H_B", 0.23, turn=observation.turn),
            ),
            task_family=observation.task_family,
        )
        restored = DecisionDataset.from_dict(original.to_dict())
        self.assertEqual(restored.examples[0].observation.belief_map(), {"H_A": 0.81, "H_B": 0.23})

    def test_collector_uses_neutral_policy_question_ids(self):
        by_evaluator_id = {question.question_id: question for question in self.benchmark.questions}
        for example in self.dataset.examples:
            question = by_evaluator_id[example.question_id]
            self.assertEqual(example.observation.question_id, question.policy_question_id)
            self.assertNotIn(question.family, example.observation.question_id)

    def test_all_named_methods_use_real_optimizer_and_log_metrics(self):
        methods = (
            "SFT", "GRPO-Terminal", "GRPO-FourState", "StateGateOnly",
            "PESCO-NoBranch", "PESCO-BranchOnly", "PESCO-NoFlipLoss",
            "PESCO-Full", "Evidence-Gated SMOPD",
        )
        config = DifferentiableTrainerConfig(epochs=2, batch_size=16, max_optimizer_steps=6, seed=7)
        trainer = DifferentiableStrategyTrainer(config)
        for method in methods:
            policy, log = trainer.fit(self.dataset, method)
            self.assertGreater(log.optimizer_steps, 0, method)
            self.assertEqual(log.optimizer_steps, config.max_optimizer_steps, method)
            self.assertEqual(log.implementation_status, "genuine_cpu_differentiable_reference")
            self.assertTrue(log.epochs, method)
            self.assertIn("policy_entropy", log.epochs[-1])
            self.assertIn("reference_kl", log.epochs[-1])
            self.assertIn("belief_loss", log.epochs[-1])
            if method == "PESCO-Full":
                self.assertGreater(log.reversal_count, 0)
                self.assertGreater(log.epochs[-1]["flip_loss"], 0.0)
                self.assertGreater(log.flip_gradient_norm, 0.0)
                self.assertGreater(log.flip_update_norm, 0.0)
                self.assertGreater(log.flip_updates_applied, 0)
                self.assertGreater(log.flip_gradient_probe_count, 0)
            if method == "Evidence-Gated SMOPD":
                self.assertGreater(log.teacher_optimizer_steps, 0)
                self.assertGreater(log.teacher_parameter_count, 0)
            metrics = evaluate_differentiable_policy(policy, self.dataset, "diagnostic_ood", state_gate=method == "StateGateOnly")
            self.assertEqual(metrics["example_count"], 8)
            self.assertEqual(metrics["confidence_interval_status"], "estimable")
            self.assertIn("mean_belief_log_loss", metrics)

    def test_sft_ignores_hidden_target_action_audit_field(self):
        """SFT must consume public branch winners, never target-table labels."""
        altered = copy.deepcopy(self.dataset)
        for example in altered.examples:
            # Deliberately poison the audit-only field with a conflicting action.
            example.metadata["target_action"] = "repair_data_split" if example.best_action.value != "repair_data_split" else "continue_current_method"
        config = DifferentiableTrainerConfig(epochs=1, batch_size=16, max_optimizer_steps=5, seed=13)
        clean_policy, clean_log = DifferentiableStrategyTrainer(config).fit(self.dataset, "SFT")
        altered_policy, altered_log = DifferentiableStrategyTrainer(config).fit(altered, "SFT")
        self.assertEqual(clean_log.supervised_target_source, "public_branch_utility_best_action")
        self.assertEqual(altered_log.supervised_target_source, "public_branch_utility_best_action")
        for clean, changed in zip(clean_policy.parameters(), altered_policy.parameters()):
            self.assertTrue((clean.detach() == changed.detach()).all())

    def test_metrics_keep_public_and_audit_conditional_denominators_separate(self):
        policy, _ = DifferentiableStrategyTrainer(
            DifferentiableTrainerConfig(epochs=1, batch_size=16, max_optimizer_steps=4, seed=19)
        ).fit(self.dataset, "SFT")
        metrics = evaluate_differentiable_policy(policy, self.dataset, "train")
        self.assertIn("required_switch_n", metrics)
        self.assertIn("audit_required_switch_n", metrics)
        self.assertIn("invalid_repair_n", metrics)
        self.assertIn("audit_invalid_repair_n", metrics)
        self.assertIn("flip_eligible_n", metrics)
        self.assertGreater(metrics["flip_eligible_n"], 0)
        self.assertLessEqual(metrics["flip_correct_n"], metrics["flip_eligible_n"])
        self.assertLessEqual(metrics["confirmation_passed_n"], metrics["confirmation_eligible_n"])

    def test_reversal_pairs_are_same_question_and_question_normalized(self):
        by_question = {}
        for pair in self.dataset.reversals:
            left = self.dataset.examples[pair.left]
            right = self.dataset.examples[pair.right]
            self.assertEqual(left.question_id, right.question_id)
            by_question.setdefault(left.question_id, 0.0)
            by_question[left.question_id] += float(pair.weight)
        self.assertTrue(by_question)
        for total in by_question.values():
            self.assertAlmostEqual(total, 1.0, places=8)

    def test_metrics_expose_pairrank_and_exact_top1_separately(self):
        policy, _ = DifferentiableStrategyTrainer(
            DifferentiableTrainerConfig(epochs=1, batch_size=16, max_optimizer_steps=2, seed=31)
        ).fit(self.dataset, "PESCO-Full")
        metrics = evaluate_differentiable_policy(policy, self.dataset, "train")
        self.assertIn("pairwise_reversal_ranking_accuracy", metrics)
        self.assertIn("exact_top1_reversal_accuracy", metrics)
        self.assertIn("pairwise_reversal_question_rows", metrics)
        self.assertEqual(metrics["reversal_pair_aggregation"], "question_macro_equal_weight")

    def test_d_metrics_expose_cost_and_invalid_local_optimization_diagnostics(self):
        policy, _ = DifferentiableStrategyTrainer(
            DifferentiableTrainerConfig(epochs=1, batch_size=16, max_optimizer_steps=2, seed=29)
        ).fit(self.dataset, "PESCO-Full")
        metrics = evaluate_differentiable_policy(policy, self.dataset, "diagnostic_ood")
        for key in (
            "research_regret", "utility_per_cost", "mean_utility_per_cost",
            "erroneous_repair_n", "erroneous_repair_eligible_n", "erroneous_repair_rate",
            "invalid_local_optimization_n", "invalid_local_optimization_eligible_n",
            "invalid_local_optimization_rate", "selected_invalid_branch_rate",
        ):
            self.assertIn(key, metrics)
        self.assertLessEqual(metrics["erroneous_repair_n"], metrics["erroneous_repair_eligible_n"])
        self.assertLessEqual(
            metrics["invalid_local_optimization_n"],
            metrics["invalid_local_optimization_eligible_n"],
        )

    def test_invalid_switch_correct_action_is_not_local_optimization(self):
        """Invalid→SWITCH is valid for the subgroup-metric family."""
        self.assertFalse(is_invalid_local_optimization(
            EvidenceState.INVALID,
            ResearchAction.SWITCH,
            ResearchAction.SWITCH,
        ))
        self.assertTrue(is_invalid_local_optimization(
            EvidenceState.INVALID,
            ResearchAction.SWITCH,
            ResearchAction.REPAIR,
        ))
        self.assertFalse(is_invalid_local_optimization(
            EvidenceState.SUPPORTED,
            ResearchAction.SWITCH,
            ResearchAction.CONTINUE,
        ))
        subgroup_invalid = [
            example for example in self.dataset.examples
            if example.metadata.get("family") == "subgroup_metric_mismatch"
            and example.state_target is EvidenceState.INVALID
            and example.best_action is ResearchAction.SWITCH
        ]
        self.assertGreater(len(subgroup_invalid), 0)
        self.assertTrue(all(
            not is_invalid_local_optimization(
                example.state_target,
                ResearchAction.SWITCH,
                example.best_action,
            )
            for example in subgroup_invalid
        ))

    def test_flip_training_does_not_read_dev_or_ood_pairs(self):
        """Held-out reversal annotations must not alter the fitted train policy."""
        altered = copy.deepcopy(self.dataset)
        heldout_index = next(
            index for index, pair in enumerate(altered.reversals)
            if altered.examples[pair.left].split != "train"
            or altered.examples[pair.right].split != "train"
        )
        pair = altered.reversals[heldout_index]
        altered.reversals[heldout_index] = ReversalExample(
            left=pair.left,
            right=pair.right,
            action_left=ResearchAction.REPAIR,
            action_right=ResearchAction.SAMPLE,
            margin=pair.margin,
            confirmed=True,
            weight=100.0,
        )
        config = DifferentiableTrainerConfig(epochs=1, batch_size=16, max_optimizer_steps=5, seed=23)
        clean_policy, _ = DifferentiableStrategyTrainer(config).fit(self.dataset, "PESCO-Full")
        altered_policy, _ = DifferentiableStrategyTrainer(config).fit(altered, "PESCO-Full")
        for clean, changed in zip(clean_policy.parameters(), altered_policy.parameters()):
            self.assertTrue((clean.detach() == changed.detach()).all())


if __name__ == "__main__":
    unittest.main()

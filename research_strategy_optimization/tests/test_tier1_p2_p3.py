from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch as _torch  # noqa: F401 - presence check for optional differentiable diagnostics
except ImportError:  # pragma: no cover - exercised in minimal NumPy/stdlib installs
    raise unittest.SkipTest("PyTorch is an optional dependency for P2/P3 differentiable tests")

from research_strategy_optimization.algorithms.differentiable_strategy import (
    DecisionDataset,
    DecisionExample,
    DifferentiableStrategyPolicy,
)
from research_strategy_optimization.evaluation.tier1_differentiable_suite import (
    _pairwise_reversal_ranking_correct,
    _question_macro_bootstrap_ci,
    evaluate_differentiable_policy,
)
from research_strategy_optimization.evaluation.tier1_p2_experiments import (
    _perturbed_winner_stability,
    normalized_regret,
    top1_top2_gap,
    tie_set,
)
from research_strategy_optimization.evaluation.tier1_p3_gate import run_p3_gate
from research_strategy_optimization.schemas import EvidenceState, Observation, ResearchAction


class Tier1P2P3GateTests(unittest.TestCase):
    def test_tie_set_is_practical_tie_aware(self) -> None:
        self.assertEqual(tie_set([1.0, 0.99, 0.8], tolerance=0.02), (0, 1))

    def test_normalized_regret_is_bounded_and_scale_invariant(self) -> None:
        self.assertAlmostEqual(normalized_regret([0.0, 10.0, 5.0], 2), 0.5)
        self.assertEqual(normalized_regret([3.0, 3.0], 0), 0.0)

    def test_reward_non_tie_uses_top1_top2_gap(self) -> None:
        self.assertAlmostEqual(top1_top2_gap([1.0, 0.99, 0.2, -0.5]), 0.01)
        self.assertAlmostEqual(top1_top2_gap([1.0, 0.2, 0.1, 0.0]), 0.8)

    def test_pairwise_ranking_is_distinct_from_exact_top1(self) -> None:
        left = {
            ResearchAction.CONTINUE.value: 0.40,
            ResearchAction.SAMPLE.value: 0.35,
        }
        right = {
            ResearchAction.CONTINUE.value: 0.20,
            ResearchAction.SAMPLE.value: 0.30,
        }
        self.assertTrue(_pairwise_reversal_ranking_correct(
            left, right, ResearchAction.CONTINUE, ResearchAction.SAMPLE
        ))
        # The left endpoint's preferred action need not be the global top-1 action
        # among all four candidates; ranking evaluates only the registered pair.
        self.assertEqual(
            _question_macro_bootstrap_ci(
                [
                    {"question_id": "q1", "value": 1.0},
                    {"question_id": "q1", "value": 0.0},
                    {"question_id": "q2", "value": 0.0},
                ],
                "value",
                seed=7,
                replicates=100,
            )["point"],
            0.25,
        )

    def test_selected_action_confirmation_uses_replicate_receipts(self) -> None:
        action_names = [action.value for action in ResearchAction.mvp_actions()]
        receipts = {
            action: [
                {"confirmation_eligible": True, "passed": False}
                for _ in range(4)
            ]
            for action in action_names
        }
        observation = Observation(
            question_id="public",
            turn=1,
            current_method="method_a",
            effect_estimate=0.1,
            ci_low=0.05,
            ci_high=0.15,
            sample_size=100,
            seed_count=4,
            remaining_budget=4,
        )
        example = DecisionExample(
            observation=observation,
            branch_utilities=(1.0, 0.5, 0.25, 0.0),
            state_target=EvidenceState.SUPPORTED,
            split="train",
            question_id="q_receipt",
            world_id="q_receipt__supported",
            metadata={
                "branch_seed_confirmations": receipts,
                "branch_costs": {action: 1.0 for action in action_names},
                "branch_validity": {action: True for action in action_names},
            },
        )
        policy = DifferentiableStrategyPolicy(seed=11)
        metrics = evaluate_differentiable_policy(
            policy,
            DecisionDataset([example]),
            "train",
        )
        self.assertEqual(metrics["confirmation_eligible_n"], 4)
        self.assertEqual(metrics["confirmation_passed_n"], 0)
        self.assertEqual(metrics["confirmation_rate"], 0.0)
        self.assertEqual(metrics["confirmation_metric_unit"], "selected_action_replicate_receipt")

    def test_selected_confirmation_excludes_unattempted_receipts_but_keeps_failures(self) -> None:
        action_names = [action.value for action in ResearchAction.mvp_actions()]
        receipts = {
            action: [
                {"confirmation_eligible": True, "passed": True},
                {"confirmation_eligible": True, "passed": False},
                {"confirmation_eligible": False, "passed": None},
                {"confirmation_eligible": False, "passed": None},
            ]
            for action in action_names
        }
        observation = Observation(
            question_id="public",
            turn=1,
            current_method="method_a",
            effect_estimate=0.1,
            ci_low=0.05,
            ci_high=0.15,
            sample_size=100,
            seed_count=4,
            remaining_budget=4,
        )
        example = DecisionExample(
            observation=observation,
            branch_utilities=(1.0, 0.5, 0.25, 0.0),
            state_target=EvidenceState.SUPPORTED,
            split="train",
            question_id="q_receipt_mixed",
            world_id="q_receipt_mixed__supported",
            metadata={
                "branch_seed_confirmations": receipts,
                "branch_costs": {action: 1.0 for action in action_names},
                "branch_validity": {action: True for action in action_names},
            },
        )
        metrics = evaluate_differentiable_policy(
            DifferentiableStrategyPolicy(seed=11),
            DecisionDataset([example]),
            "train",
        )
        self.assertEqual(metrics["confirmation_eligible_n"], 2)
        self.assertEqual(metrics["confirmation_observed_n"], 4)
        self.assertEqual(metrics["confirmation_receipt_n"], 4)
        self.assertEqual(metrics["confirmation_ineligible_n"], 2)
        self.assertEqual(metrics["confirmation_passed_n"], 1)
        self.assertAlmostEqual(metrics["confirmation_rate"], 0.5)

    def test_reward_stability_reports_top1_top2_tie_strata(self) -> None:
        action_names = [action.value for action in ResearchAction.mvp_actions()]
        def make_example(question_id: str, utilities: tuple[float, ...]) -> DecisionExample:
            components = {
                action: {"utility": float(value)}
                for action, value in zip(action_names, utilities)
            }
            observation = Observation(
                question_id="public",
                turn=1,
                current_method="method_a",
                effect_estimate=0.1,
                ci_low=0.05,
                ci_high=0.15,
                sample_size=100,
                seed_count=4,
                remaining_budget=4,
            )
            return DecisionExample(
                observation=observation,
                branch_utilities=utilities,
                state_target=EvidenceState.SUPPORTED,
                split="train",
                question_id=question_id,
                world_id=f"{question_id}__supported",
                metadata={"reward_components": components},
            )
        dataset = DecisionDataset([
            make_example("q_tie", (1.0, 0.99, 0.0, 0.0)),
            make_example("q_non_tie", (1.0, 0.5, 0.0, 0.0)),
        ])
        result = _perturbed_winner_stability(dataset, tolerance=0.02, replicates=8, seed=3)
        self.assertEqual(result["tie_winner_n"], 1)
        self.assertEqual(result["non_tie_winner_n"], 1)
        self.assertIsNotNone(result["tie_stable_winner_fraction"])
        self.assertIsNotNone(result["non_tie_stable_winner_fraction"])
        self.assertEqual(result["non_tie_definition"], "top1_minus_top2 > tolerance")

    def test_p3_missing_p2_fails_closed_without_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_p3_gate(Path(tmp) / "out", Path(tmp) / "missing_p2.json")
            self.assertEqual(result["status"], "no_go_prerequisites_not_met")
            self.assertFalse(result["experiment_executed"])
            self.assertIn("p2_result_missing", result["blocking_reasons"])
            self.assertFalse(result["online_rl_or_7b_authorized"])


if __name__ == "__main__":
    unittest.main()

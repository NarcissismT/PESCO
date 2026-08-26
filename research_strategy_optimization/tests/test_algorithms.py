from __future__ import annotations

import math
import unittest

from research_strategy_optimization.algorithms.branch_rollout import BranchRolloutManager
from research_strategy_optimization.algorithms.leave_one_out_advantage import (
    leave_one_out_advantages,
    leave_one_out_baselines,
)
from research_strategy_optimization.algorithms.paired_world_sampler import (
    PairedWorldSampler,
    identify_confirmed_reversal,
)
from research_strategy_optimization.algorithms.preference_reversal_loss import (
    batched_preference_reversal_loss,
    preference_reversal_loss,
)
from research_strategy_optimization.algorithms.pesco_trainer import PESCOTrainer, TrainerConfig
from research_strategy_optimization.algorithms.strategy_policy import TabularStrategyPolicy
from research_strategy_optimization.environments.tier0_simulator import (
    Tier0ResearchEnvironment,
    default_mvp_worlds,
)
from research_strategy_optimization.schemas import Observation, ResearchAction


class LeaveOneOutTests(unittest.TestCase):
    def test_excludes_current_branch(self) -> None:
        self.assertEqual(leave_one_out_baselines([1.0, 3.0, 5.0]), [4.0, 3.0, 2.0])
        self.assertEqual(leave_one_out_advantages([1.0, 3.0, 5.0]), [-3.0, 0.0, 3.0])

    def test_normalized_constant_group_is_finite(self) -> None:
        values = leave_one_out_advantages([2.0, 2.0, 2.0], normalize=True)
        self.assertEqual(values, [0.0, 0.0, 0.0])


class BranchRolloutTests(unittest.TestCase):
    def test_options_restore_one_common_state(self) -> None:
        env = Tier0ResearchEnvironment()
        before = env.reset("rq_test", "world_01", seed=123)
        manager = BranchRolloutManager(env, utility_fn=lambda output, verdict, _: output.effect_estimate)
        snapshot = manager.create_snapshot()
        results = manager.execute_paired_options(
            snapshot,
            [ResearchAction.CONTINUE, ResearchAction.SWITCH],
            seeds=(17, 29, 41),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([result.snapshot_digest for result in results], [snapshot.digest] * 2)
        self.assertNotEqual(results[0].utility, results[1].utility)
        after = env.visible_observation()
        self.assertEqual(before.to_dict(), after.to_dict())
        effects = manager.estimate_paired_effects(results)
        self.assertIn((ResearchAction.CONTINUE.value, ResearchAction.SWITCH.value), effects)

    def test_duplicate_or_empty_seeds_fail(self) -> None:
        env = Tier0ResearchEnvironment()
        env.reset("rq_test", "world_01")
        manager = BranchRolloutManager(env)
        snapshot = manager.create_snapshot()
        with self.assertRaises(ValueError):
            manager.execute_paired_options(snapshot, [ResearchAction.CONTINUE], seeds=())
        with self.assertRaises(ValueError):
            manager.execute_paired_options(snapshot, [ResearchAction.CONTINUE], seeds=(1, 1))


class PairedWorldTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_public_view_hides_ids(self) -> None:
        first = PairedWorldSampler(default_mvp_worlds(), seed=9).sample(question_id="rq_x")[0]
        second = PairedWorldSampler(default_mvp_worlds(), seed=9).sample(question_id="rq_x")[0]
        self.assertEqual(first.pair_id, second.pair_id)
        self.assertEqual(first.common_seeds, second.common_seeds)
        public = first.public_manifest()
        self.assertFalse(public["world_ids_visible_to_policy"])
        self.assertNotIn(first.world_a.world_id, public.values())

    def test_confirmed_reversal_requires_both_confidence_sides(self) -> None:
        reversal = identify_confirmed_reversal(
            question_id="rq_x",
            world_a="a",
            world_b="b",
            action_left=ResearchAction.CONTINUE,
            action_right=ResearchAction.SWITCH,
            values_a_left=[1.0, 1.1, 0.9, 1.05],
            values_a_right=[0.1, 0.0, 0.2, 0.05],
            values_b_left=[0.0, 0.1, -0.1, 0.05],
            values_b_right=[0.8, 0.9, 0.7, 0.85],
            margin=0.05,
        )
        self.assertTrue(reversal.confirmed)
        self.assertGreater(reversal.lcb_a, 0.05)
        self.assertLess(reversal.ucb_b, -0.05)


class PreferenceLossTests(unittest.TestCase):
    def test_scalar_loss_prefers_the_correct_order(self) -> None:
        good = preference_reversal_loss(0.0, -2.0, -2.0, 0.0, beta=2.0)
        bad = preference_reversal_loss(-2.0, 0.0, 0.0, -2.0, beta=2.0)
        self.assertTrue(math.isfinite(good))
        self.assertLess(good, bad)

    def test_torch_batch_preserves_shape_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        examples = []
        for _ in range(2):
            examples.append(
                (
                    torch.tensor(0.0, requires_grad=True),
                    torch.tensor(-1.0, requires_grad=True),
                    torch.tensor(-1.0, requires_grad=True),
                    torch.tensor(0.0, requires_grad=True),
                )
            )
        values = [preference_reversal_loss(*example, beta=1.0) for example in examples]
        batch = batched_preference_reversal_loss(
            [
                __import__("research_strategy_optimization.algorithms.preference_reversal_loss", fromlist=["PreferenceReversalExample"]).PreferenceReversalExample(*example)
                for example in examples
            ]
        )
        self.assertTrue(torch.isfinite(batch))
        batch.backward()
        self.assertTrue(all(example[0].grad is not None for example in examples))
        self.assertAlmostEqual(float(batch.detach()), sum(float(value.detach()) for value in values) / 2.0, places=6)

    def test_tabular_flip_loss_gradient_changes_parameters_and_reduces_loss(self) -> None:
        policy = TabularStrategyPolicy(seed=3)
        supported = Observation("q", 0, "method_a", 0.10, 0.04, 0.16, 120, 4, 5)
        refuted = Observation("q", 0, "method_a", 0.00, -0.05, 0.05, 120, 4, 5)
        # Materialise both public states and freeze the reference before optimization.
        policy.probabilities(supported)
        policy.probabilities(refuted)
        policy.freeze_reference()
        before_checksum = policy.parameter_checksum()
        before = preference_reversal_loss(
            policy.action_log_ratios(supported),
            policy.action_log_ratios(refuted),
            ResearchAction.CONTINUE.value,
            ResearchAction.SWITCH.value,
            1.0,
        )
        diagnostics = policy.apply_preference_reversal_loss(
            supported,
            refuted,
            ResearchAction.CONTINUE,
            ResearchAction.SWITCH,
            beta=1.0,
            learning_rate=0.2,
        )
        after = preference_reversal_loss(
            policy.action_log_ratios(supported),
            policy.action_log_ratios(refuted),
            ResearchAction.CONTINUE.value,
            ResearchAction.SWITCH.value,
            1.0,
        )
        self.assertTrue(diagnostics["applied"])
        self.assertGreater(diagnostics["gradient_norm"], 0.0)
        self.assertGreater(diagnostics["update_norm"], 0.0)
        self.assertNotEqual(diagnostics["parameter_checksum_before"], diagnostics["parameter_checksum_after"])
        self.assertNotEqual(before_checksum, policy.parameter_checksum())
        self.assertAlmostEqual(before, diagnostics["loss_before"], places=8)
        self.assertAlmostEqual(after, diagnostics["loss_after"], places=8)
        self.assertLess(after, before)

    def test_trainer_logs_loss_driven_flip_update(self) -> None:
        trainer = PESCOTrainer(config=TrainerConfig(epochs=1, reversal_learning_rate=0.2))
        trainer.fit(
            lambda: Tier0ResearchEnvironment(),
            [world.world_id for world in default_mvp_worlds()],
            question_id="q",
        )
        epoch = trainer.log.epochs[0]
        self.assertTrue(epoch["loss_driven_flip_update"])
        self.assertGreater(float(epoch["flip_gradient_norm"]), 0.0)
        self.assertGreater(float(epoch["flip_parameter_update_norm"]), 0.0)
        self.assertLess(float(epoch["flip_loss_after"]), float(epoch["flip_loss_before"]))


if __name__ == "__main__":
    unittest.main()

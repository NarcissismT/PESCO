from __future__ import annotations

import unittest

from research_strategy_optimization.algorithms.option_executor import OptionExecutor
from research_strategy_optimization.algorithms.paired_world_sampler import PairedWorldPreferenceBuilder
from research_strategy_optimization.environments.tier0_simulator import Tier0ResearchEnvironment
from research_strategy_optimization.evidence.equivalence_tests import practical_equivalence
from research_strategy_optimization.evidence.optional_stopping_controls import StoppingSchedule
from research_strategy_optimization.schemas import ResearchAction


class PlanInterfaceTests(unittest.TestCase):
    def test_equivalence_and_stopping_guards(self):
        self.assertTrue(practical_equivalence(0.0, (-0.01, 0.01), 0.02).equivalent)
        schedule = StoppingSchedule.preregistered((1, 3), max_looks=3)
        schedule.require(3)
        with self.assertRaises(RuntimeError):
            schedule.require(2)

    def test_option_executor_and_preference_builder(self):
        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_01")
        receipt = OptionExecutor(env).execute(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        self.assertEqual(receipt.action, ResearchAction.CONTINUE)
        builder = PairedWorldPreferenceBuilder()
        aligned = builder.align_candidate_options({"a": ResearchAction.mvp_actions(), "b": ResearchAction.mvp_actions()})
        self.assertEqual(set(aligned), set(ResearchAction.mvp_actions()))


if __name__ == "__main__":
    unittest.main()

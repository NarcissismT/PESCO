from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.environments.tier1_benchmark import (
    MECHANISM_FAMILIES,
    build_tier1_v03_benchmark,
)
from research_strategy_optimization.schemas import ResearchAction


class Tier1BenchmarkTests(unittest.TestCase):
    def test_frozen_benchmark_has_12_questions_48_worlds_and_four_families(self) -> None:
        benchmark = build_tier1_v03_benchmark()
        self.assertEqual(len(benchmark.questions), 12)
        self.assertEqual(len(benchmark.worlds), 48)
        self.assertEqual(
            {question.family for question in benchmark.questions},
            set(MECHANISM_FAMILIES),
        )
        self.assertEqual(
            {question.family: sum(q.family == question.family for q in benchmark.questions) for question in benchmark.questions},
            {family: 3 for family in MECHANISM_FAMILIES},
        )
        self.assertEqual(
            benchmark.manifest()["counts_by_split"],
            {"train": 8, "dev": 2, "diagnostic_ood": 2},
        )

    def test_same_evidence_state_has_family_dependent_targets(self) -> None:
        benchmark = build_tier1_v03_benchmark()
        insufficient_targets = {
            question.target_action(next(world for world in question.worlds if world.kind == "insufficient").world_id)
            for question in benchmark.questions
        }
        self.assertIn(ResearchAction.SAMPLE, insufficient_targets)
        self.assertIn(ResearchAction.SWITCH, insufficient_targets)
        invalid_targets = {
            question.target_action(next(world for world in question.worlds if world.kind == "invalid").world_id)
            for question in benchmark.questions
        }
        self.assertIn(ResearchAction.REPAIR, invalid_targets)
        self.assertIn(ResearchAction.SAMPLE, invalid_targets)
        self.assertIn(ResearchAction.SWITCH, invalid_targets)

    def test_manifest_is_json_serializable_and_world_ids_remain_evaluator_side(self) -> None:
        manifest = build_tier1_v03_benchmark().manifest(include_hidden=True)
        encoded = json.dumps(manifest, sort_keys=True)
        self.assertIn("manifest_digest", manifest)
        self.assertIn("t1_group_leakage_01__invalid", encoded)
        public = build_tier1_v03_benchmark().manifest(include_hidden=False)
        public_encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("true_effect_a", public_encoded)
        self.assertNotIn("noise_scale", public_encoded)
        self.assertNotIn("target_actions", public_encoded)
        self.assertNotIn("__invalid", public_encoded)

    def test_policy_question_ids_are_neutral_and_do_not_reveal_world_or_target(self) -> None:
        benchmark = build_tier1_v03_benchmark()
        for question in benchmark.questions:
            self.assertNotIn(question.family, question.policy_question_id)
            env = benchmark.make_environment(question.question_id)
            observation = env.reset(
                question.policy_question_id,
                question.worlds[0].world_id,
                seed=17,
            )
            payload = json.dumps(observation.to_dict(), sort_keys=True)
            # The task family is registered public question context (needed to make
            # same-state action choices identifiable); the hidden world and target
            # action must remain evaluator-only.
            self.assertEqual(observation.task_family, question.family)
            self.assertNotIn(question.worlds[0].world_id, payload)
            self.assertNotIn(question.target_action(question.worlds[0].world_id).value, payload)


if __name__ == "__main__":
    unittest.main()

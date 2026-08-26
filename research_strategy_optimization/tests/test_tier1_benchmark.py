from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.environments.tier1_benchmark import (
    MECHANISM_FAMILIES,
    TIER1_REWARD_COMPONENT_NAMES,
    build_tier1_v03_benchmark,
    tier1_scientific_utility,
    tier1_scientific_utility_components,
)
from research_strategy_optimization.environments.tier0_simulator import TrustedVerifier
from research_strategy_optimization.schemas import Protocol
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

    def test_scientific_utility_components_are_fixed_and_sum_to_scalar(self) -> None:
        benchmark = build_tier1_v03_benchmark()
        question = benchmark.questions[0]
        protocol = Protocol(
            protocol_version="pesco_v0_2",
            exploration_seeds=(17,),
            confirmation_seeds=(103,),
        )
        for kind in ("supported", "invalid"):
            world = next(item for item in question.worlds if item.kind == kind)
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            initial = env.reset(question.policy_question_id, world.world_id, seed=17)
            output = env.execute_option(ResearchAction.CONTINUE, seeds=(17,))
            verdict = TrustedVerifier(protocol).evaluate(output, env, confirm=False)
            components = tier1_scientific_utility_components(
                question,
                world,
                ResearchAction.CONTINUE,
                output,
                verdict,
                protocol,
                initial_observation=initial,
            )
            self.assertEqual(tuple(components), TIER1_REWARD_COMPONENT_NAMES)
            self.assertTrue(all(isinstance(value, float) for value in components.values()))
            scalar = tier1_scientific_utility(
                question,
                world,
                ResearchAction.CONTINUE,
                output,
                verdict,
                protocol,
                initial_observation=initial,
            )
            self.assertAlmostEqual(sum(components.values()), scalar, places=12)

    def test_scientific_utility_components_are_available_for_every_action(self) -> None:
        benchmark = build_tier1_v03_benchmark()
        question = benchmark.questions[0]
        world = question.worlds[0]
        protocol = Protocol(
            protocol_version="pesco_v0_2",
            exploration_seeds=(17,),
            confirmation_seeds=(103,),
        )
        env = benchmark.make_environment(question.question_id, protocol=protocol)
        initial = env.reset(question.policy_question_id, world.world_id, seed=17)
        verifier = TrustedVerifier(protocol)
        for action in ResearchAction.mvp_actions():
            branch = env.clone_from_snapshot(env.snapshot())
            output = branch.execute_option(action, seeds=(17,))
            verdict = verifier.evaluate(output, branch, confirm=False)
            components = tier1_scientific_utility_components(
                question,
                world,
                action,
                output,
                verdict,
                protocol,
                initial_observation=initial,
            )
            self.assertEqual(set(components), set(TIER1_REWARD_COMPONENT_NAMES))
            self.assertAlmostEqual(
                sum(components.values()),
                tier1_scientific_utility(
                    question,
                    world,
                    action,
                    output,
                    verdict,
                    protocol,
                    initial_observation=initial,
                ),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()

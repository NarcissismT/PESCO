from __future__ import annotations

import unittest
from dataclasses import replace

from research_strategy_optimization.environments.tier0_simulator import (
    Tier0ResearchEnvironment,
    TrustedVerifier,
    default_mvp_worlds,
)
from research_strategy_optimization.environments.tier1_tabular_env import (
    Tier1TabularEnvironment,
)
from research_strategy_optimization.schemas import EvidenceState, Protocol, ResearchAction


class TrackingTier1Environment(Tier1TabularEnvironment):
    """Test-only subclass that records the class used by confirmation clones."""

    confirmation_types: list[type] = []

    def _simulate(self, method, option, seeds, confirmation):
        if confirmation:
            type(self).confirmation_types.append(type(self))
        return super()._simulate(method, option, seeds, confirmation)


class Tier1EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        base = list(default_mvp_worlds())
        self.protocol = Protocol()
        self.worlds = [
            replace(base[0], world_id="tier1_supported"),
            replace(base[1], world_id="tier1_confounded", kind="invalid", confounding=True),
            replace(base[3], world_id="tier1_leakage", confounding=False),
        ]
        TrackingTier1Environment.confirmation_types.clear()

    def _branch(self, world_id: str, cls=Tier1TabularEnvironment):
        env = cls(worlds=self.worlds, protocol=self.protocol)
        env.reset(question_id="tier1_test", world_id=world_id, seed=17)
        snapshot = env.snapshot()
        return env, env.clone_from_snapshot(snapshot), snapshot

    def test_clone_and_confirmation_preserve_tier1_backend(self) -> None:
        env, branch, _ = self._branch("tier1_supported", TrackingTier1Environment)
        self.assertIsInstance(branch, TrackingTier1Environment)

        output = branch.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        verdict = TrustedVerifier(self.protocol).evaluate(output, branch)
        self.assertTrue(verdict.validity_pass)
        self.assertTrue(verdict.independent_confirmation_passed)
        self.assertTrue(verdict.confirmation_data_independent)
        self.assertEqual(output.backend, "tier1_numpy")
        self.assertTrue(TrackingTier1Environment.confirmation_types)
        self.assertTrue(all(cls is TrackingTier1Environment for cls in TrackingTier1Environment.confirmation_types))

        # A direct Tier-0 run cannot share Tier-1 provenance hashes.
        tier0 = Tier0ResearchEnvironment(worlds=self.worlds, protocol=self.protocol)
        tier0.reset(question_id="tier1_test", world_id="tier1_supported", seed=17)
        tier0_output = tier0.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        self.assertNotEqual(output.backend, tier0_output.backend)
        self.assertNotEqual(output.code_hash, tier0_output.code_hash)
        self.assertNotEqual(output.dataset_hash, tier0_output.dataset_hash)

    def test_nonconfounded_assignment_and_confounded_adjustment(self) -> None:
        _, supported, _ = self._branch("tier1_supported")
        supported_output = supported.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        self.assertLess(abs(supported_output.treatment_confounder_correlation), 0.2)
        self.assertEqual(supported_output.estimator, supported.ESTIMATOR_DIFF)

        _, confounded, snapshot = self._branch("tier1_confounded")
        naive = confounded.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        naive_verdict = TrustedVerifier(self.protocol).evaluate(naive, confounded)
        self.assertFalse(naive_verdict.validity_pass)
        confounded_output_correlation = naive.treatment_confounder_correlation
        self.assertGreater(confounded_output_correlation, 0.3)
        self.assertGreater(abs(naive.effect_estimate - naive.latent_effect), 0.08)

        repaired = confounded.clone_from_snapshot(snapshot)
        adjusted = repaired.execute_option(ResearchAction.REPAIR, seeds=self.protocol.exploration_seeds)
        adjusted_verdict = TrustedVerifier(self.protocol).evaluate(adjusted, repaired)
        self.assertTrue(adjusted_verdict.validity_pass)
        self.assertEqual(adjusted.estimator, repaired.ESTIMATOR_ADJUSTED)
        self.assertLess(abs(adjusted.effect_estimate - adjusted.latent_effect), abs(naive.effect_estimate - naive.latent_effect))
        self.assertNotEqual(naive.code_hash, adjusted.code_hash)
        self.assertNotEqual(naive.split_hash, adjusted.split_hash)
        self.assertEqual(naive.dataset_hash, adjusted.dataset_hash)
        self.assertAlmostEqual(
            confounded_output_correlation,
            adjusted.treatment_confounder_correlation,
            places=12,
        )

    def test_leakage_repair_changes_actual_split_protocol(self) -> None:
        _, leaky, snapshot = self._branch("tier1_leakage")
        before = leaky.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        before_verdict = TrustedVerifier(self.protocol).evaluate(before, leaky)
        self.assertFalse(before_verdict.validity_pass)
        self.assertEqual(before_verdict.evidence_state, EvidenceState.INVALID)
        self.assertTrue(before.leakage)
        self.assertGreater(before.group_overlap_count, 0)
        self.assertIn("row_random_overlap_v1", before.data_partition)

        repaired = leaky.clone_from_snapshot(snapshot)
        after = repaired.execute_option(ResearchAction.REPAIR, seeds=self.protocol.exploration_seeds)
        after_verdict = TrustedVerifier(self.protocol).evaluate(after, repaired)
        self.assertTrue(after_verdict.validity_pass)
        self.assertFalse(after.leakage)
        self.assertEqual(after.group_overlap_count, 0)
        self.assertEqual(after.data_partition, "group_held_out_v1")
        self.assertNotEqual(before.split_hash, after.split_hash)
        self.assertLess(abs(after.effect_estimate - after.latent_effect), abs(before.effect_estimate - before.latent_effect))

    def test_tier1_confirmation_hashes_are_independent(self) -> None:
        _, branch, _ = self._branch("tier1_supported")
        output = branch.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        verdict = TrustedVerifier(self.protocol).evaluate(output, branch)
        self.assertTrue(verdict.independent_confirmation_performed)
        self.assertNotEqual(verdict.confirmation_dataset_hash, output.dataset_hash)
        self.assertNotEqual(verdict.confirmation_split_hash, output.split_hash)
        self.assertTrue(verdict.confirmation_data_independent)

    def test_tier1_provenance_tampering_fails_closed(self) -> None:
        _, branch, _ = self._branch("tier1_supported")
        output = branch.execute_option(ResearchAction.CONTINUE, seeds=self.protocol.exploration_seeds)
        forged = replace(output, dataset_hash="sha256:" + "0" * 64)
        verdict = TrustedVerifier(self.protocol).evaluate(forged, branch, confirm=False)
        self.assertFalse(verdict.validity_pass)
        self.assertIn("provenance_hash_mismatch", verdict.invalid_reasons)


if __name__ == "__main__":
    unittest.main()

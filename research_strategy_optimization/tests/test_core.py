from __future__ import annotations

import json
import math
import unittest

from research_strategy_optimization.algorithms.branch_rollout import BranchRolloutManager
from research_strategy_optimization.algorithms.discovery_certificate import make_discovery_certificate
from research_strategy_optimization.algorithms.strategy_policy import TabularStrategyPolicy
from research_strategy_optimization.algorithms.objectives import (
    ObjectiveWeights,
    clipped_option_loss,
    factorized_evidence_loss,
    masked_token_advantages,
    objective_breakdown,
    pesco_objective,
)
from research_strategy_optimization.evidence.evidence_classifier import classify_evidence
from research_strategy_optimization.evidence.proper_scoring import belief_delta, log_score
from research_strategy_optimization.environments.tier0_simulator import Tier0ResearchEnvironment, TrustedVerifier, default_mvp_worlds
from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment
from research_strategy_optimization.environments.tier2_posttraining_env import Tier2PostTrainingEnvironment
from research_strategy_optimization.evaluation.final_decision import freeze_check, mvp_gate, stage_status
from research_strategy_optimization.evaluation.multiple_testing import benjamini_hochberg, holm_adjust
from research_strategy_optimization.schemas import EvidenceState, Protocol, ResearchAction
from research_strategy_optimization.utils.ledger import AuditLedger
from research_strategy_optimization.utils.public_view import public_trajectory


class EvidenceRuleTests(unittest.TestCase):
    def test_invalid_precedes_interval(self):
        decision = classify_evidence(False, 1.0, (0.9, 1.1), Protocol(), invalid_reasons=["leakage"])
        self.assertEqual(decision.state, EvidenceState.INVALID)

    def test_insufficient_is_not_refuted(self):
        decision = classify_evidence(True, 0.02, (0.0, 0.04), Protocol())
        self.assertEqual(decision.state, EvidenceState.INSUFFICIENT)

    def test_supported_and_refuted_interval_rules(self):
        protocol = Protocol(delta_min=0.02)
        self.assertEqual(classify_evidence(True, 0.1, (0.04, 0.16), protocol).state, EvidenceState.SUPPORTED)
        self.assertEqual(classify_evidence(True, 0.0, (-0.01, 0.01), protocol).state, EvidenceState.REFUTED)

    def test_proper_score_rewards_correct_support_and_refutation(self):
        self.assertGreater(belief_delta(0.1, 0.9, 1), 0.0)
        self.assertGreater(belief_delta(0.9, 0.1, 0), 0.0)
        self.assertGreater(log_score(0.9, 1), log_score(0.1, 1))


class EnvironmentIsolationTests(unittest.TestCase):
    def test_public_observation_has_no_hidden_world_fields(self):
        env = Tier0ResearchEnvironment()
        observation = env.reset(world_id="world_04")
        payload = json.dumps(observation.to_dict(), sort_keys=True).lower()
        for token in ("world_04", "latent_effect", "true_effect_a", "hidden_world_id"):
            self.assertNotIn(token, payload)

    def test_dynamic_invalid_and_insufficient_transitions(self):
        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_04")
        output = env.execute_option(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        verdict = TrustedVerifier().evaluate(output, env)
        self.assertEqual(verdict.evidence_state, EvidenceState.INVALID)
        repaired = env.clone_from_snapshot(env.snapshot())
        output = repaired.execute_option(ResearchAction.REPAIR, seeds=(17, 29, 41, 53))
        self.assertNotEqual(TrustedVerifier().evaluate(output, repaired).evidence_state, EvidenceState.INVALID)

        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_03")
        output = env.execute_option(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        self.assertEqual(TrustedVerifier().evaluate(output, env).evidence_state, EvidenceState.INSUFFICIENT)
        sampled = env.clone_from_snapshot(env.snapshot())
        output = sampled.execute_option(ResearchAction.SAMPLE, seeds=(17, 29, 41, 53))
        self.assertEqual(TrustedVerifier().evaluate(output, sampled).evidence_state, EvidenceState.SUPPORTED)

    def test_public_trajectory_excludes_verdict(self):
        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_01")
        manager = BranchRolloutManager(environment=env, verifier=TrustedVerifier())
        result = manager.execute_paired_options(manager.create_snapshot(), [ResearchAction.CONTINUE], seeds=(17, 29, 41, 53))[0]
        payload = public_trajectory(result.record.trajectory)
        self.assertNotIn("verdicts", payload)
        self.assertEqual(payload["world_id"], "hidden_from_agent")

    def test_hypothesis_beliefs_remain_bound_after_method_switch(self):
        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_02")
        env.execute_option(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        before = env.visible_observation()
        self.assertEqual(before.active_hypothesis_id, "H_A")
        self.assertIn("H_B", before.belief_map())
        branch = env.clone_from_snapshot(env.snapshot())
        branch.execute_option(ResearchAction.SWITCH, seeds=(17, 29, 41, 53))
        after = branch.visible_observation()
        self.assertEqual(after.active_hypothesis_id, "H_B")
        self.assertNotEqual(before.belief_map()["H_A"], after.belief_map()["H_B"])
        self.assertEqual(after.to_dict()["active_hypothesis_id"], "H_B")

    def test_confirmation_uses_distinct_dataset_and_split_hashes(self):
        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_01")
        output = env.execute_option(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        verdict = TrustedVerifier().evaluate(output, env)
        self.assertTrue(verdict.independent_confirmation_performed)
        self.assertTrue(verdict.confirmation_data_independent)
        self.assertNotEqual(output.dataset_hash, verdict.confirmation_dataset_hash)
        self.assertNotEqual(output.split_hash, verdict.confirmation_split_hash)

    def test_tier1_clone_preserves_numpy_backend(self):
        env = Tier1TabularEnvironment()
        env.reset(world_id="world_01")
        clone = env.clone_from_snapshot(env.snapshot())
        self.assertIsInstance(clone, Tier1TabularEnvironment)
        output = clone.execute_option(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        self.assertEqual(output.backend, Tier1TabularEnvironment.BACKEND)
        self.assertIn("tier1_numpy_backend", output.validity_signals)
        self.assertNotEqual(output.code_hash, env._hash("code:method_a"))


class GateAndLedgerTests(unittest.TestCase):
    def test_freeze_and_mvp_gate(self):
        self.assertTrue(freeze_check({
            "question_manifest_sealed": True,
            "final_split_inaccessible": True,
            "world_id_hidden": True,
            "verifier_immutable": True,
            "contamination_audit_pass": True,
            "resource_budget_defined": True,
        })["pass"])
        gate = mvp_gate({})
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["tier2_llm_training_authorized"])

    def test_hash_chain_detects_tampering(self):
        ledger = AuditLedger()
        ledger.append("a", {"x": 1})
        ledger.append("b", {"y": 2})
        self.assertTrue(ledger.verify())
        ledger.events[0]["payload"]["x"] = 9
        self.assertFalse(ledger.verify())

    def test_tier2_is_fail_closed_without_scientific_gate(self):
        with self.assertRaises(RuntimeError):
            Tier2PostTrainingEnvironment().reset("q", "hidden", 0)

    def test_authorized_tier2_delegates_to_injected_bundle(self):
        class Bundle:
            def reset(self, question_id, world_id, seed):
                return (question_id, world_id, seed)

        env = Tier2PostTrainingEnvironment({"scientific_hard_gate_pass": True, "executor": Bundle()})
        self.assertEqual(env.reset("q", "hidden", 3), ("q", "hidden", 3))


class DiscoveryTests(unittest.TestCase):
    def test_invalid_or_unconfirmed_candidate_does_not_get_certificate(self):
        env = Tier0ResearchEnvironment()
        env.reset(world_id="world_04")
        output = env.execute_option(ResearchAction.CONTINUE, seeds=(17, 29, 41, 53))
        verdict = TrustedVerifier().evaluate(output, env)
        cert = make_discovery_certificate(
            method_family="new_family",
            proposed_without_method_hint=True,
            structurally_distinct=True,
            actually_executed=True,
            verdict=verdict,
            lower_confidence_gain=1.0,
        )
        self.assertFalse(cert.certificate_pass)


class ObjectiveAndStatisticsTests(unittest.TestCase):
    def test_objective_components_and_token_mask(self):
        self.assertLess(clipped_option_loss(0.0, 1.0), 0.0)
        self.assertEqual(masked_token_advantages([1.0, 2.0, 3.0], [True, False, True]), [1.0, 0.0, 3.0])
        state = factorized_evidence_loss(
            {"valid": -0.1, "invalid": -2.0},
            {"precise": -0.1, "imprecise": -2.0},
            {"positive": -0.1, "negative": -2.0},
            validity_target="valid", precision_target="precise", direction_target="positive",
        )
        self.assertTrue(math.isfinite(state))
        total = pesco_objective(1.0, 2.0, 3.0, weights=ObjectiveWeights(flip=0.5, state=0.2))
        self.assertAlmostEqual(total, 2.6)
        breakdown = objective_breakdown(1.0, 2.0, 3.0, weights=ObjectiveWeights(flip=0.5, state=0.2))
        self.assertAlmostEqual(breakdown["flip_loss_weighted"], 1.0)
        self.assertAlmostEqual(breakdown["total_loss"], total)

    def test_protocol_version_freeze_and_stage_status_are_evidence_bound(self):
        protocol = Protocol()
        self.assertEqual(protocol.protocol_version, "pesco_v0_2")
        self.assertTrue(freeze_check({
            "question_manifest_sealed": True,
            "final_split_inaccessible": True,
            "world_id_hidden": True,
            "verifier_immutable": True,
            "contamination_audit_pass": True,
            "resource_budget_defined": True,
            "protocol_version": "pesco_v0_2",
            "expected_protocol_version": "pesco_v0_2",
        })["pass"])
        self.assertFalse(freeze_check({
            "question_manifest_sealed": True,
            "final_split_inaccessible": True,
            "world_id_hidden": True,
            "verifier_immutable": True,
            "contamination_audit_pass": True,
            "resource_budget_defined": True,
            "protocol_version": "pesco_v0_1",
            "expected_protocol_version": "pesco_v0_2",
        })["pass"])
        self.assertEqual(stage_status("stage_3_zero_shot", {"pass": True, "diagnostic_only": True})["status"], "NO-GO")

    def test_multiple_testing_corrections_are_monotone_and_bounded(self):
        p = [0.001, 0.02, 0.5]
        h = holm_adjust(p)
        b = benjamini_hochberg(p)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in h + b))
        self.assertLessEqual(h[0], h[1])


if __name__ == "__main__":
    unittest.main()

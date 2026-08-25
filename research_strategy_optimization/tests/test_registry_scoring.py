from __future__ import annotations

import unittest

from research_strategy_optimization.evidence.hypothesis_registry import GENESIS_HASH, HypothesisRegistry
from research_strategy_optimization.evidence.proper_scoring import multiclass_belief_delta, multiclass_log_score
from research_strategy_optimization.schemas import Hypothesis


def _hypothesis(identifier: str = "h1") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=identifier,
        question_id="q1",
        claim="method A has a positive effect",
        estimand="effect_delta",
        delta_min=0.02,
        protocol_version="pesco_v0_2",
        timestamp="2026-08-25T00:00:00+00:00",
    )


class HypothesisRegistryTests(unittest.TestCase):
    def test_duplicate_and_non_monotone_belief_turns_are_rejected(self):
        registry = HypothesisRegistry()
        registry.register_before_experiment(_hypothesis())
        registry.commit_belief("h1", 0.5, turn=0)
        with self.assertRaises(ValueError):
            registry.commit_belief("h1", 0.6, turn=0)
        with self.assertRaises(ValueError):
            registry.commit_belief("h1", 0.6, turn=-1)
        with self.assertRaises(ValueError):
            registry.commit_belief("h1", 0.6, turn=0)
        registry.commit_belief("h1", 0.6, turn=1)
        self.assertTrue(registry.verify_hash_chain())

    def test_belief_and_evidence_share_a_tamper_evident_chain(self):
        registry = HypothesisRegistry()
        registry.register_before_experiment(_hypothesis())
        first = registry.commit_belief("h1", 0.5, turn=0)
        second = registry.append_evidence("h1", {"state": "insufficient", "turn": 0})
        third = registry.commit_belief("h1", 0.55, turn=1)
        self.assertEqual(registry.hash_chain("h1")[0]["previous_event_hash"], GENESIS_HASH)
        self.assertEqual(registry.hash_chain_tip("h1"), third)
        self.assertNotEqual(first, second)
        self.assertTrue(registry.verify())
        # Accessors are copies; mutating one cannot alter the committed chain.
        public = registry.records()
        public["beliefs"]["h1"][0]["probability"] = 0.99
        self.assertTrue(registry.verify())
        # Direct tampering is detected by comparing typed views with the chain.
        registry._beliefs["h1"][0]["probability"] = 0.99  # noqa: SLF001 - audit test
        self.assertFalse(registry.verify())

    def test_registry_rejects_forged_evidence_metadata(self):
        registry = HypothesisRegistry()
        registry.register_before_experiment(_hypothesis())
        with self.assertRaises(ValueError):
            registry.append_evidence("h1", {"event_index": 99, "state": "supported"})


class MulticlassScoringValidationTests(unittest.TestCase):
    def test_valid_distribution_and_belief_delta(self):
        score = multiclass_log_score({"supported": 0.7, "refuted": 0.2, "insufficient": 0.1}, "supported")
        self.assertAlmostEqual(score, -0.3566749439, places=8)
        self.assertGreater(
            multiclass_belief_delta(
                {"supported": 0.4, "refuted": 0.4, "insufficient": 0.2},
                {"supported": 0.7, "refuted": 0.2, "insufficient": 0.1},
                "supported",
            ),
            0.0,
        )

    def test_invalid_distribution_inputs_fail_closed(self):
        invalid = (
            {},
            {"supported": 0.8, "refuted": 0.1},
            {"supported": 1.1, "refuted": -0.1},
            {"supported": float("nan"), "refuted": 0.0},
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises((ValueError, TypeError)):
                    multiclass_log_score(probabilities, "supported")
        with self.assertRaises(KeyError):
            multiclass_log_score({"supported": 1.0}, "refuted")


if __name__ == "__main__":
    unittest.main()

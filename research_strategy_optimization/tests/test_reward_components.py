from __future__ import annotations

import unittest

try:
    import torch as _torch  # noqa: F401 - collector returns the optional DecisionDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal NumPy/stdlib installs
    _TORCH_AVAILABLE = False

from research_strategy_optimization.evaluation.tier1_v04_extended import (
    ACTION_SET,
    TIER1_REWARD_COMPONENT_NAMES,
    TRACK_RAW_EVIDENCE,
    build_tier1_v04_extended_benchmark,
    collect_tier1_v04_extended,
)
from research_strategy_optimization.schemas import Protocol


class RewardComponentReceiptTests(unittest.TestCase):
    def test_bounded_extended_collector_records_atomic_receipts(self) -> None:
        """The one-question smoke exercises aggregation without a full benchmark run."""

        if not _TORCH_AVAILABLE:
            self.skipTest("PyTorch is an optional dependency for the differentiable collector")

        protocol = Protocol(
            protocol_version="pesco_v0_2",
            exploration_seeds=(17, 29, 41, 53, 67, 71, 83, 97),
            confirmation_seeds=(103, 107, 109, 113, 127, 131, 137, 139),
            max_budget=6,
        )
        dataset, _audit = collect_tier1_v04_extended(
            build_tier1_v04_extended_benchmark(),
            protocol,
            track=TRACK_RAW_EVIDENCE,
            question_limit=1,
        )
        self.assertEqual(len(dataset.examples), 4)
        action_names = tuple(action.value for action in ACTION_SET)
        for example in dataset.examples:
            aggregate = example.metadata["reward_components"]
            self.assertEqual(set(aggregate), set(action_names))
            self.assertEqual(tuple(example.metadata["reward_component_names"]), TIER1_REWARD_COMPONENT_NAMES)
            self.assertEqual(
                example.metadata["reward_components_aggregation"],
                "mean_over_independent_replicates",
            )
            self.assertTrue(example.metadata["pre_action_observation_constructed_before_candidate_branches"])
            self.assertTrue(example.metadata["cross_candidate_confirmation_feature_excluded"])
            self.assertEqual(set(example.metadata["branch_replicate_confirmation"]), set(action_names))
            for index, action_name in enumerate(action_names):
                terms = aggregate[action_name]
                self.assertEqual(tuple(terms), TIER1_REWARD_COMPONENT_NAMES)
                self.assertAlmostEqual(sum(terms.values()), example.branch_utilities[index], places=9)
                receipts = example.metadata["branch_replicate_confirmation"][action_name]
                self.assertTrue(receipts)
                self.assertTrue(all("reward_components" in receipt for receipt in receipts))
                self.assertTrue(
                    all(
                        tuple(receipt["reward_components"]) == TIER1_REWARD_COMPONENT_NAMES
                        for receipt in receipts
                    )
                )


if __name__ == "__main__":
    unittest.main()

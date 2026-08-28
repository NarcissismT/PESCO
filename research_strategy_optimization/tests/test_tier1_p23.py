from __future__ import annotations

import json
import unittest

try:
    from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
    from research_strategy_optimization.evaluation.tier1_p23_dataset import (
        P23_COUNTS, P23_GENERATOR_VERSION, build_tier1_p23_promotion_v2_benchmark,
    )
    from research_strategy_optimization.evaluation.tier1_p23_diagnostics import P23_COMPONENT_MAPPING, P23Config, _canonical_reversals
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch extra is required for differentiable P2.3 tests")
class Tier1P23Tests(unittest.TestCase):
    def test_promotion_v2_split_and_family_contract(self) -> None:
        benchmark = build_tier1_p23_promotion_v2_benchmark()
        counts = {split: sum(q.split == split for q in benchmark.questions) for split in benchmark.split_names}
        self.assertEqual(counts, P23_COUNTS)
        self.assertEqual(len({q.family for q in benchmark.questions}), 8)
        self.assertEqual(P23_GENERATOR_VERSION, benchmark.manifest(include_hidden=True)["generator_version"])
        self.assertGreaterEqual(counts["promotion"], 48)

    def test_canonical_contract_is_one_pair_per_question(self) -> None:
        dataset = DecisionDataset.from_json("artifacts/tier1_p23_promotion_v2/dataset_raw_evidence.json")
        pairs, audit = _canonical_reversals(dataset, P23Config())
        by_question = {}
        for pair in pairs:
            qid = dataset.examples[pair.left].question_id
            by_question[qid] = by_question.get(qid, 0) + 1
        self.assertTrue(audit["canonical_top1"])
        self.assertTrue(all(count <= 1 for count in by_question.values()))
        promotion = [pair for pair in pairs if dataset.examples[pair.left].split == "promotion" and dataset.examples[pair.right].split == "promotion"]
        self.assertGreaterEqual(len(promotion), 30)
        self.assertGreaterEqual(len({dataset.examples[pair.left].question_id for pair in promotion}), 20)

    def test_component_mapping_and_optimizer_scope_are_explicit(self) -> None:
        self.assertEqual(P23_COMPONENT_MAPPING["Base"], "GRPO-MatchedAtomic")
        self.assertEqual(P23_COMPONENT_MAPPING["Branch+Flip"], "GRPO+Branch+Flip")
        receipt = json.load(open("artifacts/tier1_p23_frozen_raw_10seed/p23_gate_receipt.json", encoding="utf-8"))
        self.assertFalse(receipt["p3a_authorized"])
        self.assertFalse(receipt["online_rl_authorized"])


if __name__ == "__main__":
    unittest.main()

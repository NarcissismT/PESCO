from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset, DifferentiableStrategyPolicy
    from research_strategy_optimization.evaluation.tier1_p231_diagnostics import (
        FOUR_STATE_COMPONENTS, P231Config, REWARD_COMPONENTS, TERMINAL_COMPONENTS,
        canonical_pair_payload, fit_rollout_method, reward_tensors, verify_canonical_pair_payload,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch extra is required for differentiable P2.3.1 tests")
class Tier1P231Tests(unittest.TestCase):
    def test_artificial_reward_sample_has_three_distinct_objectives(self) -> None:
        action_names = [action.value for action in __import__(
            "research_strategy_optimization.algorithms.differentiable_strategy",
            fromlist=["ACTION_SET"],
        ).ACTION_SET]
        components = {action: {name: 0.0 for name in REWARD_COMPONENTS} for action in action_names}
        components[action_names[0]]["validity_gate"] = 0.2
        components[action_names[1]]["repair_protocol_bonus"] = 0.3
        components[action_names[2]]["sample_precision_bonus"] = 0.4
        components[action_names[3]]["execution_cost_penalty"] = -0.1
        example = SimpleNamespace(metadata={"reward_components": components})
        rewards = reward_tensors(example)
        self.assertFalse((rewards["terminal"] == rewards["four_state"]).all().item())
        self.assertFalse((rewards["terminal"] == rewards["atomic"]).all().item())
        self.assertFalse((rewards["four_state"] == rewards["atomic"]).all().item())
        self.assertTrue(TERMINAL_COMPONENTS <= set(REWARD_COMPONENTS))
        self.assertTrue(FOUR_STATE_COMPONENTS <= set(REWARD_COMPONENTS))

    def test_canonical_pair_digest_recomputes_exactly(self) -> None:
        dataset = DecisionDataset.from_json("artifacts/tier1_p23_promotion_v2/dataset_raw_evidence.json")
        rows, payload = canonical_pair_payload(dataset, P231Config())
        audit = verify_canonical_pair_payload(dataset, payload, P231Config())
        self.assertTrue(rows)
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["training_pair_digest"], audit["evaluation_pair_digest"])
        self.assertEqual(audit["evaluation_pair_digest"], audit["gate_pair_digest"])
        self.assertGreater(float(payload["top1_gap_threshold"]), 0.0)

    def test_rloo_and_grpo_contracts_are_distinct(self) -> None:
        dataset = DecisionDataset.from_json("artifacts/tier1_p23_promotion_v2/dataset_raw_evidence.json")
        rows, _ = canonical_pair_payload(dataset, P231Config())
        from research_strategy_optimization.evaluation.tier1_p231_diagnostics import _pair_objects
        pairs = _pair_objects(dataset, rows)
        config = P231Config(sft_steps=1, finetune_steps=1, batch_size=8, hidden_dim=8, minibatch_epochs=2)
        policy = DifferentiableStrategyPolicy(hidden_dim=8, seed=17)
        _, rloo = fit_rollout_method(dataset, policy, "RLOO", config, pairs)
        _, grpo = fit_rollout_method(dataset, policy, "GRPO-Terminal", config, pairs)
        self.assertFalse(rloo["importance_ratio_used"])
        self.assertFalse(rloo["clipped_surrogate_used"])
        self.assertTrue(grpo["importance_ratio_used"])
        self.assertTrue(grpo["clipped_surrogate_used"])
        self.assertTrue(all(row["frozen_rollout"] for row in grpo["logs"]))
        self.assertNotEqual(rloo["final_policy_digest"], grpo["final_policy_digest"])


if __name__ == "__main__":
    unittest.main()

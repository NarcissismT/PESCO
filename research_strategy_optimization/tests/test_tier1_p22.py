from __future__ import annotations

import unittest

import torch

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p22_diagnostics import (
    P22Config,
    _canonical_reversals,
    _aggregate,
    _utility_soft_targets,
)
from scripts.write_p22_gate_receipt import build as build_p22_gate


class Tier1P22Tests(unittest.TestCase):
    def test_soft_utility_targets_are_normalized_and_monotone(self) -> None:
        targets = _utility_soft_targets(torch.tensor([[1.0, 0.0, -1.0, 1.0]]), 0.25)
        self.assertAlmostEqual(float(targets.sum()), 1.0, places=6)
        self.assertGreater(float(targets[0, 0]), float(targets[0, 1]))
        self.assertAlmostEqual(float(targets[0, 0]), float(targets[0, 3]), places=6)

    def test_canonical_pairs_use_unique_top1_and_gap(self) -> None:
        dataset = DecisionDataset.from_json("artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
        pairs, audit = _canonical_reversals(dataset, P22Config(top1_gap_threshold=0.05))
        self.assertEqual(audit["canonical_top1"], True)
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            left, right = dataset.examples[pair.left], dataset.examples[pair.right]
            self.assertEqual(pair.action_left, left.best_action)
            self.assertEqual(pair.action_right, right.best_action)
            self.assertNotEqual(pair.action_left, pair.action_right)

    def test_bootstrap_uses_question_level_pairrank_rows(self) -> None:
        records = []
        for seed in (17, 23):
            for method, rank_rows in (
                ("SFT", [("q1", 0.2, 0.2), ("q2", 0.8, 0.8)]),
                ("SFT→NoFlip", [("q1", 0.4, 0.4), ("q2", 0.6, 0.6)]),
                ("SFT→Pairwise-Full", [("q1", 0.9, 0.9), ("q2", 0.1, 0.1)]),
            ):
                records.append({
                    "method": method,
                    "seed": seed,
                    "split": "promotion",
                    "normalized_regret": 0.2,
                    "pairwise_reversal_ranking_accuracy": sum(row[1] for row in rank_rows) / 2,
                    "exact_top1_reversal_accuracy": sum(row[2] for row in rank_rows) / 2,
                    "question_metric_rows": [
                        {
                            "question_id": question_id,
                            "normalized_regret": 0.2,
                            "pairwise_reversal_ranking_accuracy": pairrank,
                            "exact_top1_reversal_accuracy": top1,
                        }
                        for question_id, pairrank, top1 in rank_rows
                    ],
                })
        aggregate = _aggregate(records, ("SFT→Pairwise-Full", "SFT→NoFlip"), (17, 23), bootstrap_replicates=200)
        ci = aggregate["gate_checks"]["SFT→Pairwise-Full"]["pairrank_delta_vs_noflip_ci"]
        self.assertEqual(ci["question_count"], 2)
        self.assertLess(ci["lower"], ci["upper"])

    def test_gate_reads_nested_strict_shortcut_metrics_and_rejects_fallback(self) -> None:
        result = {
            "matrix_complete": True,
            "aggregation": {
                "promotion_summary": {
                    "SFT": {"required_switch_rate": 0.10},
                    "SFT→NoFlip": {"required_switch_rate": 0.05},
                    "SFT→Pairwise-Full": {
                        "mean_regret": 0.20,
                        "required_switch_rate": 0.20,
                        "confirmation_rate": 1.0,
                        "erroneous_repair_rate": 0.0,
                        "validity_rate": 1.0,
                    },
                },
                "gate_checks": {
                    "SFT→Pairwise-Full": {
                        "regret_gate": True,
                        "pairrank_gate": True,
                        "canonical_top1_gate": True,
                    }
                },
            },
        }
        family = {"gate": {"regret_non_single_family": True, "pairrank_non_single_family": True}}
        strict = {
            "models": {
                "all_raw:logistic_regression": {
                    "model": "logistic_regression",
                    "status": "completed",
                    "implementation": "sklearn",
                    "metrics_by_split": {"promotion": {"normalized_regret": 0.10}},
                },
                "all_raw:gradient_boosting": {
                    "model": "gradient_boosting",
                    "status": "completed",
                    "implementation": "sklearn",
                    "metrics_by_split": {"promotion": {"normalized_regret": 0.15}},
                },
                "fallback": {
                    "status": "completed",
                    "implementation": "numpy_fallback",
                    "metrics_by_split": {"promotion": {"normalized_regret": 0.01}},
                },
            }
        }
        reward = {"scenarios": {"base": {"stability_rate": 1.0}}, "method_ranking_available": True, "method_ranking_stable": True}
        receipt = build_p22_gate(result, family, strict, reward, method="SFT→Pairwise-Full")
        self.assertFalse(receipt["gates"]["strict_shortcut_probe"])
        result["aggregation"]["promotion_summary"]["SFT→Pairwise-Full"]["mean_regret"] = 0.05
        receipt = build_p22_gate(result, family, strict, reward, method="SFT→Pairwise-Full")
        self.assertTrue(receipt["gates"]["strict_shortcut_probe"])


if __name__ == "__main__":
    unittest.main()

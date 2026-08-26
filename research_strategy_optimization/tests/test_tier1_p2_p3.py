from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.evaluation.tier1_p2_experiments import (
    normalized_regret,
    tie_set,
)
from research_strategy_optimization.evaluation.tier1_p3_gate import run_p3_gate


class Tier1P2P3GateTests(unittest.TestCase):
    def test_tie_set_is_practical_tie_aware(self) -> None:
        self.assertEqual(tie_set([1.0, 0.99, 0.8], tolerance=0.02), (0, 1))

    def test_normalized_regret_is_bounded_and_scale_invariant(self) -> None:
        self.assertAlmostEqual(normalized_regret([0.0, 10.0, 5.0], 2), 0.5)
        self.assertEqual(normalized_regret([3.0, 3.0], 0), 0.0)

    def test_p3_missing_p2_fails_closed_without_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_p3_gate(Path(tmp) / "out", Path(tmp) / "missing_p2.json")
            self.assertEqual(result["status"], "no_go_prerequisites_not_met")
            self.assertFalse(result["experiment_executed"])
            self.assertIn("p2_result_missing", result["blocking_reasons"])
            self.assertFalse(result["online_rl_or_7b_authorized"])


if __name__ == "__main__":
    unittest.main()

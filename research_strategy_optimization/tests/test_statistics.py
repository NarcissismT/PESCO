from __future__ import annotations

import unittest

from research_strategy_optimization.evaluation.statistics import (
    holm_adjust,
    holm_reject,
    paired_binary_power,
    paired_binary_required_n,
    paired_bootstrap_ci,
    paired_sign_permutation_pvalue,
)


class StatisticsTests(unittest.TestCase):
    def test_holm_preserves_order_and_controls_family(self):
        self.assertEqual(holm_adjust((0.01, 0.04, 0.20)), (0.03, 0.08, 0.20))
        self.assertEqual(holm_reject((0.01, 0.04, 0.20), alpha=0.05), (True, False, False))

    def test_paired_bootstrap_is_reproducible_and_respects_pairing(self):
        first = paired_bootstrap_ci(left=(2.0, 4.0, 6.0), right=(1.0, 1.0, 1.0), n_bootstrap=300, seed=9)
        second = paired_bootstrap_ci(left=(2.0, 4.0, 6.0), right=(1.0, 1.0, 1.0), n_bootstrap=300, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 3.0)
        self.assertLessEqual(first[1], first[0])
        self.assertLessEqual(first[0], first[2])

    def test_sign_flip_and_power_helpers(self):
        # With four matched units the exact sign-flip resolution is 1/16; use a
        # conservative threshold above that finite-sample floor.
        self.assertLess(paired_sign_permutation_pvalue((1.0, 1.0, 1.0, 1.0), n_resamples=2000, seed=3), 0.15)
        required = paired_binary_required_n(0.25, 0.10)
        self.assertGreaterEqual(required, 120)
        self.assertGreaterEqual(paired_binary_power(0.25, 0.10, required), 0.79)


if __name__ == "__main__":
    unittest.main()

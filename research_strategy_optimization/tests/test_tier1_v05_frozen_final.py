from __future__ import annotations

import unittest

from research_strategy_optimization.evaluation import tier1_v05_frozen_final as v05


class Tier1V05ConsumedBoundaryTests(unittest.TestCase):
    def test_v05_is_explicitly_consumed(self) -> None:
        self.assertEqual(v05.V05_BOUNDARY_STATUS, "final_boundary_rehearsal_consumed")
        self.assertFalse(v05.V05_PUBLIC)
        self.assertFalse(v05.V05_FORMAL_COMPARISON_AUTHORIZED)

    def test_public_contract_contains_no_generator(self) -> None:
        contract = v05.public_contract()
        self.assertFalse(contract["generator_available"])
        self.assertFalse(contract["formal_comparison_authorized"])
        self.assertNotIn("generator_recipes", contract)

    def test_commitment_loader_is_opaque(self) -> None:
        commitment = v05.load_public_commitment()
        self.assertEqual(commitment["status"], "final_boundary_rehearsal_consumed")
        self.assertNotIn("worlds", commitment)
        self.assertNotIn("generator_recipes", commitment)


if __name__ == "__main__":
    unittest.main()

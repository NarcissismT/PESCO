from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from research_strategy_optimization.evaluation import tier1_v06_frozen_final as v06


class Tier1V06BoundaryTests(unittest.TestCase):
    def test_public_contract_is_private_and_excludes_hidden_fields(self) -> None:
        contract = v06.public_contract()
        self.assertEqual(contract["status"], "private_evaluator_only")
        self.assertFalse(set(contract["public_exposes"]) & set(contract["public_excludes"]))
        self.assertIn("latent_parameters", contract["public_excludes"])
        self.assertIn("target_actions", contract["public_excludes"])

    def test_public_module_has_no_generator_like_exports(self) -> None:
        names = [name for name in dir(v06) if not name.startswith("__")]
        forbidden = [name for name in names if re.search(r"(?:build|generate|recipe|latent|target|world|seed)", name, re.I)]
        self.assertEqual(forbidden, [])

    def test_private_commitment_has_expected_counts_without_public_recipes(self) -> None:
        path = Path("artifacts/tier1_v06_evaluator_private/public_commitment.json")
        if not path.exists():
            self.skipTest("private evaluator bundle is intentionally absent in a clean public checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["split_counts"], {"final_id": 40, "final_ood": 40})
        self.assertEqual(payload["world_count"], 320)
        self.assertEqual(payload["ood_family_count"], 5)
        self.assertEqual(v06.load_public_commitment(path)["status"], "private_evaluator_only")


if __name__ == "__main__":
    unittest.main()

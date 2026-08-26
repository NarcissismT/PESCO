from __future__ import annotations

import unittest

from research_strategy_optimization.evaluation.tier1_zero_shot_robustness import (
    ACTION_LETTER_ROTATIONS,
    PROMPT_TEMPLATES,
    action_mapping_payload,
    null_prior_summary,
    render_public_prompt,
)
from research_strategy_optimization.schemas import ResearchAction


class _FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, *args, **kwargs):
        raise ImportError("jinja2 unavailable")


class Tier1ZeroShotRobustnessTests(unittest.TestCase):
    def test_action_rotations_are_bijective_and_semantically_distinct(self):
        expected = {action.value for action in ResearchAction.mvp_actions()}
        self.assertEqual(expected, set(action.value for action in ACTION_LETTER_ROTATIONS["canonical"].values()))
        checksums = set()
        for mapping in ACTION_LETTER_ROTATIONS.values():
            self.assertEqual(set(mapping), {"A", "B", "C", "D"})
            self.assertEqual(set(action.value for action in mapping.values()), expected)
            checksums.add(str(action_mapping_payload(mapping)))
        self.assertEqual(len(checksums), len(ACTION_LETTER_ROTATIONS))

    def test_prompt_whitelist_excludes_evaluator_fields_for_every_template(self):
        observation = {
            "task_family": "public_family",
            "effect_estimate": 0.1,
            "confidence_interval": [0.01, 0.2],
            "world_id": "secret_world",
            "target_action": "repair_data_split",
            "verifier_labels": "secret",
        }
        for template in PROMPT_TEMPLATES:
            prompt, provenance = render_public_prompt(
                observation,
                template=template,
                mapping=ACTION_LETTER_ROTATIONS["canonical"],
            )
            self.assertNotIn("secret_world", prompt)
            self.assertNotIn("verifier_labels", prompt)
            self.assertEqual(provenance["chat_rendering"], "plain_text")

    def test_chat_fallback_is_explicitly_non_native(self):
        prompt, provenance = render_public_prompt(
            None,
            template="compact_json",
            mapping=ACTION_LETTER_ROTATIONS["canonical"],
            chat=True,
            tokenizer=_FakeTokenizer(),
        )
        self.assertIn("<|im_start|>assistant", prompt)
        self.assertFalse(provenance["native_chat_template"])
        self.assertEqual(provenance["chat_rendering"], "manual_qwen_fallback")

    def test_null_prior_is_relabelled_semantically(self):
        mapping = ACTION_LETTER_ROTATIONS["cyclic_1"]
        summary = null_prior_summary({"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}, mapping)
        self.assertEqual(summary["semantic_prior_winner"], ResearchAction.SAMPLE.value)
        self.assertAlmostEqual(sum(summary["semantic_action_probabilities"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()

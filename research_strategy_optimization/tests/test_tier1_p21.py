from __future__ import annotations

import unittest

try:
    import torch as _torch  # noqa: F401 - P2.1 differentiable diagnostics are optional
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is an optional dependency for P2.1 diagnostics")

from research_strategy_optimization.algorithms.differentiable_strategy import (  # noqa: E402
    DecisionDataset,
    DecisionExample,
    ReversalExample,
)
from research_strategy_optimization.evaluation.tier1_p21_dataset import (  # noqa: E402
    P21_GENERATOR_VERSION,
    P21_SCHEMA,
    build_tier1_p21_diagnostic_benchmark,
    p21_latent_signature,
)
from research_strategy_optimization.evaluation.tier1_p21_diagnostics import (  # noqa: E402
    select_top_candidate_reversals,
)
from research_strategy_optimization.schemas import (  # noqa: E402
    EvidenceState,
    Observation,
    ResearchAction,
)


def _example(index: int, utilities: tuple[float, ...], question: str = "q") -> DecisionExample:
    observation = Observation(
        question_id=f"policy-{question}",
        turn=1,
        current_method="method_a",
        effect_estimate=0.1,
        ci_low=-0.01,
        ci_high=0.02,
        sample_size=100,
        seed_count=4,
        remaining_budget=3,
        task_family="public_family",
        raw_evidence=(("effect_estimate", 0.1),),
    )
    best = max(range(len(utilities)), key=lambda i: utilities[i])
    actions = tuple(ResearchAction.mvp_actions())
    return DecisionExample(
        observation=observation,
        branch_utilities=utilities,
        branch_states=tuple(EvidenceState.SUPPORTED for _ in utilities),
        state_target=EvidenceState.SUPPORTED,
        split="train",
        question_id=question,
        world_id=f"w{index}",
        world_pair_id=f"pair-{question}",
        confirmation_passed=True,
        branch_count=len(utilities),
        metadata={"family": "synthetic"},
    )


class Tier1P21Tests(unittest.TestCase):
    def test_fresh_benchmark_schema_and_split_boundary(self) -> None:
        benchmark = build_tier1_p21_diagnostic_benchmark()
        self.assertEqual(benchmark.schema_version, P21_SCHEMA)
        self.assertEqual(
            benchmark.manifest(include_hidden=True)["generator_version"],
            P21_GENERATOR_VERSION,
        )
        self.assertEqual(len(benchmark.questions), 64)
        self.assertEqual(len(benchmark.worlds), 256)
        self.assertEqual(
            {split: sum(q.split == split for q in benchmark.questions) for split in ("train", "tune", "promotion")},
            {"train": 24, "tune": 20, "promotion": 20},
        )
        self.assertEqual(
            len({p21_latent_signature(world) for world in benchmark.worlds}),
            256,
        )

    def test_top_candidate_reversal_weights_are_question_normalized(self) -> None:
        examples = [
            _example(0, (0.9, 0.2, 0.1, 0.0)),
            _example(1, (0.1, 0.8, 0.2, 0.0)),
            _example(2, (0.8, 0.3, 0.2, 0.0)),
        ]
        pair_a = ReversalExample(
            left=0,
            right=1,
            action_left=ResearchAction.CONTINUE,
            action_right=ResearchAction.SAMPLE,
            weight=0.9,
            lcb_left=0.9,
            ucb_right=-0.8,
            margin=0.05,
            confirmed=True,
        )
        pair_b = ReversalExample(
            left=2,
            right=1,
            action_left=ResearchAction.CONTINUE,
            action_right=ResearchAction.SAMPLE,
            weight=0.1,
            lcb_left=0.8,
            ucb_right=-0.8,
            margin=0.05,
            confirmed=True,
        )
        selected, audit = select_top_candidate_reversals(
            DecisionDataset(examples, [pair_a, pair_b]),
            top_k=2,
            max_pairs_per_question=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(sum(pair.weight for pair in selected), 1.0, places=12)
        self.assertEqual(audit["selected_reversal_count"], 2)
        self.assertEqual(audit["question_count_with_reversals"], 1)
        self.assertAlmostEqual(audit["weights_sum_by_question"]["q"], 1.0, places=12)

    def test_non_top_candidate_is_dropped(self) -> None:
        examples = [
            _example(0, (0.9, 0.2, 0.1, 0.0)),
            _example(1, (0.1, 0.8, 0.2, 0.0)),
        ]
        pair = ReversalExample(
            left=0,
            right=1,
            action_left=ResearchAction.REPAIR,
            action_right=ResearchAction.SAMPLE,
            lcb_left=0.9,
            ucb_right=-0.8,
            confirmed=True,
        )
        selected, audit = select_top_candidate_reversals(DecisionDataset(examples, [pair]), top_k=2)
        self.assertFalse(selected)
        self.assertEqual(audit["dropped"]["non_top_candidate"], 1)


if __name__ == "__main__":
    unittest.main()

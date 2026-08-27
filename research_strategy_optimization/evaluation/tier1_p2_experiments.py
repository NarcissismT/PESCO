"""Pre-registered Tier-1 CPU promotion experiment (P2).

The v0.3/v0.4 benchmark builders are intentionally separate from this module.  This
runner consumes a frozen :class:`DecisionDataset`, trains every named policy from the
same data and optimizer budget for ten independent seeds, and writes a fail-closed
gate report.  It is useful both for the small checked-in diagnostic dataset and for a
larger v0.4 dataset once one is available.

The only primary comparison is ``Delta_R`` on the promotion split::

    R(PESCO-Full) - R(best non-Full training baseline)

Action accuracy is retained as a secondary diagnostic.  Near ties are represented by
a pre-registered utility tolerance and regret is normalized by the action utility
range.  Inference-time branch search is reported as an evaluator upper bound; it is
not allowed to become the learned-method baseline or a claim of model superiority.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..algorithms.differentiable_strategy import (
    ACTION_SET,
    DecisionDataset,
    DifferentiableStrategyPolicy,
    DifferentiableStrategyTrainer,
    DifferentiableTrainerConfig,
    policy_action,
)
from ..schemas import ResearchAction
from ..utils.run_manifest import build_run_manifest, write_run_manifest
from .tier1_differentiable_suite import evaluate_differentiable_policy


TRAINING_METHODS: Tuple[str, ...] = (
    "SFT",
    "GRPO-Terminal",
    "GRPO-FourState",
    "PESCO-BranchOnly",
    "PESCO-NoFlipLoss",
    "PESCO-Full",
    "Evidence-Gated SMOPD",
)
PESCO_METHOD = "PESCO-Full"
BASELINE_METHODS: Tuple[str, ...] = tuple(
    method for method in TRAINING_METHODS if method != PESCO_METHOD
)
INFERENCE_SEARCH = "Inference-Time Branch Search"
DEFAULT_SEEDS: Tuple[int, ...] = tuple(range(10))
DEFAULT_TIE_TOLERANCE = 0.02


def _dataset_splits(dataset: DecisionDataset) -> Tuple[str, ...]:
    """Return deterministic split names present in a frozen dataset.

    Older diagnostic artifacts use ``diagnostic_ood`` while the formal follow-up
    uses separate ``final_id``/``final_ood`` partitions.  P2 must never silently
    evaluate a formal dataset under the old names, so all loops derive their split
    universe from the serialized examples and preserve the registered order.
    """

    preferred = ("train", "dev", "tune", "promotion", "final_id", "final_ood", "diagnostic_ood", "test")
    present = {str(example.split) for example in dataset.examples}
    ordered = [split for split in preferred if split in present]
    ordered.extend(sorted(present.difference(ordered)))
    return tuple(ordered)


@dataclass(frozen=True)
class P2Config:
    """Execution and gate settings.

    ``promotion_split`` defaults to ``dev`` because the old checked-in dataset has
    no separately opened promotion partition.  The gate records that fact and will
    fail closed until the v0.4 split has enough independent clusters/pairs.
    """

    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    promotion_split: str = "dev"
    heldout_split: str = "diagnostic_ood"
    # Baselines must be selected on a development/tune split before promotion is
    # inspected.  ``None`` chooses dev, then tune, and only falls back to the
    # promotion split for legacy datasets that have no development partition; the
    # fallback is recorded as unauthorized for formal promotion.
    baseline_selection_split: Optional[str] = None
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE
    trainer: DifferentiableTrainerConfig = field(
        default_factory=lambda: DifferentiableTrainerConfig(
            epochs=8,
            batch_size=16,
            max_optimizer_steps=64,
            learning_rate=3e-3,
            hidden_dim=48,
        )
    )
    bootstrap_replicates: int = 2000
    min_formal_pairs: int = 30
    min_formal_clusters: int = 20


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def tie_set(utilities: Sequence[float], tolerance: float = DEFAULT_TIE_TOLERANCE) -> Tuple[int, ...]:
    """Return all actions within the registered practical-tie tolerance."""

    if not utilities:
        return ()
    best = max(float(value) for value in utilities)
    return tuple(index for index, value in enumerate(utilities) if best - float(value) <= float(tolerance))


def top1_top2_gap(utilities: Sequence[float]) -> float:
    """Return the winner margin against the runner-up, not against the minimum.

    Reward perturbation stability has a meaningful non-tie denominator only when the
    best action is separated from the second-best action.  Comparing top-1 with the
    worst action incorrectly declares nearly every four-action row a non-tie.
    """

    if len(utilities) < 2:
        return 0.0
    ordered = sorted((float(value) for value in utilities), reverse=True)
    return float(ordered[0] - ordered[1])


def normalized_regret(utilities: Sequence[float], selected_index: int, *, eps: float = 1e-8) -> float:
    """Regret divided by the available action utility range."""

    if not utilities:
        return 0.0
    best = max(float(value) for value in utilities)
    worst = min(float(value) for value in utilities)
    scale = max(best - worst, float(eps))
    selected = float(utilities[int(selected_index)])
    return max(0.0, (best - selected) / scale)


def _bootstrap_ci(values: Sequence[float], *, seed: int, replicates: int) -> dict:
    """Cluster-level percentile bootstrap with explicit NA semantics."""

    numbers = [float(value) for value in values if math.isfinite(float(value))]
    if len(numbers) < 2:
        point = sum(numbers) / len(numbers) if numbers else None
        return {
            "point": point,
            "lower": None,
            "upper": None,
            "n": len(numbers),
            "status": "NA_less_than_two_clusters",
        }
    generator = random.Random(int(seed))
    samples = []
    n = len(numbers)
    for _ in range(max(100, int(replicates))):
        samples.append(sum(numbers[generator.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    lower_index = max(0, int(0.025 * len(samples)) - 1)
    upper_index = min(len(samples) - 1, int(0.975 * len(samples)))
    return {
        "point": sum(numbers) / n,
        "lower": float(samples[lower_index]),
        "upper": float(samples[upper_index]),
        "n": n,
        "status": "estimable",
    }


def _two_level_seed_question_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
    seed: int,
    replicates: int,
    exclude_family: str | None = None,
) -> dict:
    """Bootstrap paired estimates at both the training-seed and question levels.

    The old P2 report resampled only ten initialization seeds.  That interval can
    be narrow even when the method changes substantially across held-out questions.
    This helper first forms one macro contribution per question *within each seed*,
    then resamples seeds and, inside every selected seed, resamples its question
    clusters.  Family leave-one-out calls use the same estimator after removing one
    mechanism family.  Missing clusters are reported as NA rather than silently
    collapsing to a seed-only interval.
    """

    grouped: dict[int, dict[str, list[float]]] = {}
    families: dict[str, str] = {}
    for row in rows:
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        family = str(row.get("family", "unknown"))
        if exclude_family is not None and family == str(exclude_family):
            continue
        try:
            seed_id = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        question_id = str(row.get("question_id", "unknown"))
        families[question_id] = family
        grouped.setdefault(seed_id, {}).setdefault(question_id, []).append(value)
    per_seed: dict[int, list[float]] = {}
    for seed_id, by_question in grouped.items():
        question_values = [sum(values) / len(values) for values in by_question.values() if values]
        if question_values:
            per_seed[seed_id] = question_values
    if not per_seed:
        return {
            "point": None,
            "lower": None,
            "upper": None,
            "seed_count": 0,
            "question_cluster_count": 0,
            "status": "NA_no_seed_question_clusters",
            "bootstrap": "seed_then_question_cluster",
            "excluded_family": exclude_family,
        }
    seed_means = [sum(values) / len(values) for values in per_seed.values()]
    point = sum(seed_means) / len(seed_means)
    total_questions = len({question_id for by_question in grouped.values() for question_id in by_question})
    if len(per_seed) < 2 or any(len(values) < 2 for values in per_seed.values()):
        return {
            "point": point,
            "lower": None,
            "upper": None,
            "seed_count": len(per_seed),
            "question_cluster_count": total_questions,
            "status": "NA_insufficient_two_level_clusters",
            "bootstrap": "seed_then_question_cluster",
            "excluded_family": exclude_family,
        }
    rng = random.Random(int(seed))
    seed_ids = tuple(sorted(per_seed))
    draws: list[float] = []
    for _ in range(max(100, int(replicates))):
        selected_seed_means: list[float] = []
        for _outer in seed_ids:
            chosen_seed = rng.choice(seed_ids)
            question_values = per_seed[chosen_seed]
            inner = [rng.choice(question_values) for _inner in question_values]
            selected_seed_means.append(sum(inner) / len(inner))
        draws.append(sum(selected_seed_means) / len(selected_seed_means))
    draws.sort()
    lower_index = max(0, int(0.025 * len(draws)) - 1)
    upper_index = min(len(draws) - 1, int(0.975 * len(draws)))
    return {
        "point": point,
        "lower": float(draws[lower_index]),
        "upper": float(draws[upper_index]),
        "seed_count": len(per_seed),
        "question_cluster_count": total_questions,
        "status": "estimable",
        "bootstrap": "seed_then_question_cluster",
        "excluded_family": exclude_family,
        "per_seed_question_counts": {str(seed_id): len(values) for seed_id, values in sorted(per_seed.items())},
    }


def _family_leave_one_out_cis(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    seed: int,
    replicates: int,
) -> dict[str, dict]:
    families = sorted({str(row.get("family", "unknown")) for row in rows})
    return {
        family: _two_level_seed_question_ci(
            rows,
            value_key=value_key,
            seed=int(seed) + index + 1,
            replicates=replicates,
            exclude_family=family,
        )
        for index, family in enumerate(families)
    }


def _method_metrics(policy: DifferentiableStrategyPolicy, dataset: DecisionDataset, split: str, *, tie_tol: float) -> dict:
    """Evaluate a policy and add normalized/tie-aware metrics."""

    base = dict(evaluate_differentiable_policy(policy, dataset, split))
    examples = [example for example in dataset.examples if example.split == split]
    tie_correct = []
    normalized = []
    for example in examples:
        action = policy_action(policy, example.observation)
        selected_index = ACTION_SET.index(action)
        ties = tie_set(example.branch_utilities, tie_tol)
        tie_correct.append(selected_index in ties)
        normalized.append(normalized_regret(example.branch_utilities, selected_index))
    base.update({
        "tie_tolerance": float(tie_tol),
        "tie_aware_action_accuracy": sum(tie_correct) / len(tie_correct) if tie_correct else None,
        "normalized_regret": sum(normalized) / len(normalized) if normalized else None,
        "normalized_regret_ci": _bootstrap_ci(normalized, seed=17, replicates=500),
        "question_cluster_count": len({example.question_id for example in examples}),
        "confirmed_same_question_pair_count": sum(
            bool(pair.confirmed)
            and dataset.examples[pair.left].split == split
            and dataset.examples[pair.right].split == split
            and dataset.examples[pair.left].question_id == dataset.examples[pair.right].question_id
            for pair in dataset.reversals
        ),
    })
    # Preserve question-cluster contributions in the per-seed record.  Aggregate
    # metrics alone cannot support the required seed×question bootstrap or family
    # leave-one-out audit.  These rows contain only evaluator-visible outcomes and
    # are not fed back into policy training.
    record_by_world = {
        str(row.get("world_id")): row
        for row in base.get("records", [])
        if row.get("world_id") is not None
    }
    question_groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        row = record_by_world.get(str(example.world_id), {})
        selected_raw = row.get("selected_action")
        try:
            selected_index = ACTION_SET.index(ResearchAction(selected_raw))
        except (TypeError, ValueError):
            selected_index = 0
        question_groups.setdefault(str(example.question_id), []).append({
            "question_id": str(example.question_id),
            "family": str(example.metadata.get("family", "unknown")),
            "world_id": str(example.world_id),
            "normalized_regret": normalized_regret(example.branch_utilities, selected_index),
            "regret": float(max(example.branch_utilities) - example.branch_utilities[selected_index]),
        })
    normalized_question_rows = [
        {
            "question_id": question_id,
            "family": str(rows[0].get("family", "unknown")),
            "normalized_regret": sum(float(row["normalized_regret"]) for row in rows) / len(rows),
            "regret": sum(float(row["regret"]) for row in rows) / len(rows),
            "world_count": len(rows),
        }
        for question_id, rows in sorted(question_groups.items())
        if rows
    ]
    existing_question_rows = {
        str(row.get("question_id")): dict(row)
        for row in base.get("question_metric_rows", [])
        if row.get("question_id") is not None
    }
    for row in normalized_question_rows:
        existing = existing_question_rows.setdefault(str(row["question_id"]), dict(row))
        existing.update({
            "family": str(row.get("family", existing.get("family", "unknown"))),
            "normalized_regret": float(row["normalized_regret"]),
            "regret": float(row["regret"]),
            "world_count": int(row["world_count"]),
        })
    base["question_metric_rows"] = [existing_question_rows[key] for key in sorted(existing_question_rows)]
    # PairRankAcc is a pair statistic; retain one question-macro row per question
    # so its difference can be bootstrapped at the same two levels as regret.
    pair_rows_by_world = {
        str(row.get("world_id")): row
        for row in base.get("records", [])
        if row.get("world_id") is not None
    }
    pair_question_values: dict[str, list[tuple[float, float]]] = {}
    for pair in dataset.reversals:
        left = dataset.examples[int(pair.left)]
        right = dataset.examples[int(pair.right)]
        if not bool(pair.confirmed) or left.split != split or right.split != split:
            continue
        if left.question_id != right.question_id:
            continue
        left_row = pair_rows_by_world.get(str(left.world_id), {})
        right_row = pair_rows_by_world.get(str(right.world_id), {})
        left_probs = left_row.get("action_probabilities", {})
        right_probs = right_row.get("action_probabilities", {})
        try:
            left_ok = float(left_probs.get(pair.action_left.value, 0.0)) > float(left_probs.get(pair.action_right.value, 0.0))
            right_ok = float(right_probs.get(pair.action_right.value, 0.0)) > float(right_probs.get(pair.action_left.value, 0.0))
            value = 1.0 if left_ok and right_ok else 0.0
        except (TypeError, ValueError):
            value = 0.0
        try:
            pair_weight = float(pair.weight)
        except (TypeError, ValueError):
            pair_weight = 0.0
        if not math.isfinite(pair_weight) or pair_weight < 0.0:
            pair_weight = 0.0
        pair_question_values.setdefault(str(left.question_id), []).append((value, pair_weight))
    base["pairwise_reversal_question_rows"] = [
        {
            "question_id": question_id,
            "family": str(next((item.get("family", "unknown") for item in base["question_metric_rows"] if item["question_id"] == question_id), "unknown")),
            "pairwise_reversal_ranking_accuracy": (
                sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
                if sum(weight for _, weight in values) > 0.0
                else sum(value for value, _ in values) / len(values)
            ),
            "pair_count": len(values),
            "pair_weight_sum": sum(weight for _, weight in values),
        }
        for question_id, values in sorted(pair_question_values.items())
        if values
    ]
    # The v0.4 evaluator exposes Pairwise Reversal Ranking Accuracy as the primary
    # flip-loss-aligned metric.  Keep an explicit fallback for older serialized
    # datasets whose evaluator only emitted the transitional alias.
    if "pairwise_reversal_ranking_accuracy" not in base:
        base["pairwise_reversal_ranking_accuracy"] = base.get("pair_flip_accuracy")
    if "exact_top1_reversal_accuracy" not in base:
        base["exact_top1_reversal_accuracy"] = base.get("flip_accuracy")
    return base


def _branch_search_metrics(dataset: DecisionDataset, split: str, *, tie_tol: float) -> dict:
    """Evaluator upper bound from the frozen branch utility vector."""

    examples = [example for example in dataset.examples if example.split == split]
    if not examples:
        return {"split": split, "example_count": 0, "oracle_upper_bound": True}
    regrets = [0.0 for _ in examples]
    normalized = [0.0 for _ in examples]
    tie_correct = [bool(tie_set(example.branch_utilities, tie_tol)) for example in examples]
    return {
        "split": split,
        "example_count": len(examples),
        "action_accuracy": 1.0,
        "tie_aware_action_accuracy": 1.0,
        "mean_regret": 0.0,
        "normalized_regret": 0.0,
        "normalized_regret_ci": _bootstrap_ci(normalized, seed=19, replicates=200),
        "oracle_upper_bound": True,
        "formal_baseline_eligible": False,
    }


def _perturbed_winner_stability(dataset: DecisionDataset, *, tolerance: float, replicates: int, seed: int) -> dict:
    """Audit winner stability only when atomic reward receipts are available.

    A scalar ``branch_utilities`` value is not an auditable decomposition of the
    reward.  Earlier versions reconstructed pseudo-components from that scalar and
    consequently made the ±20% stability check almost tautological.  The collector
    provides, per action,
    ``metadata["reward_components"][action]`` as a mapping of named atomic terms;
    that is the only schema accepted by this gate.  The non-tie denominator is based
    on the top-1 minus top-2 reward gap; tie and non-tie stability are reported
    separately.
    """

    missing_examples = [
        str(example.question_id)
        for example in dataset.examples
        if not isinstance(example.metadata.get("reward_components"), Mapping)
    ]
    non_tie = 0
    if missing_examples:
        return {
            "non_tie_winner_n": int(sum(
                top1_top2_gap(example.branch_utilities) > float(tolerance)
                for example in dataset.examples
            )),
            "tie_winner_n": int(sum(
                top1_top2_gap(example.branch_utilities) <= float(tolerance)
                for example in dataset.examples
            )),
            "replicate_n_per_example": max(1, int(replicates)),
            "stable_winner_fraction": None,
            "stable_winner_threshold": 0.90,
            "status": "NA_missing_atomic_reward_receipts",
            "authorized": False,
            "required_receipt": "metadata.reward_components[action] -> named atomic numeric terms",
            "missing_example_count": len(missing_examples),
            "missing_example_ids_sample": missing_examples[:10],
        }

    generator = random.Random(int(seed))
    winners = []
    baseline_winners = []
    baseline_non_tie = []
    term_names: Optional[Tuple[str, ...]] = None
    for example in dataset.examples:
        reward_components = example.metadata.get("reward_components")
        if not isinstance(reward_components, Mapping):
            # This is defensive for a caller that mutates the dataset between the
            # initial schema check and this loop.
            return {
                "non_tie_winner_n": int(non_tie),
                "replicate_n_per_example": max(1, int(replicates)),
                "stable_winner_fraction": None,
                "stable_winner_threshold": 0.90,
                "status": "NA_missing_atomic_reward_receipts",
                "authorized": False,
                "required_receipt": "metadata.reward_components[action] -> named atomic numeric terms",
            }
        action_components: list[Dict[str, float]] = []
        for action in ACTION_SET:
            raw_terms = reward_components.get(action.value)
            if not isinstance(raw_terms, Mapping) or not raw_terms:
                return {
                    "non_tie_winner_n": int(non_tie),
                    "replicate_n_per_example": max(1, int(replicates)),
                    "stable_winner_fraction": None,
                    "stable_winner_threshold": 0.90,
                    "status": "NA_malformed_atomic_reward_receipts",
                    "authorized": False,
                    "required_receipt": "metadata.reward_components[action] -> named atomic numeric terms",
                    "question_id": str(example.question_id),
                    "action": action.value,
                }
            terms: Dict[str, float] = {}
            for key, value in raw_terms.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = float("nan")
                if not math.isfinite(numeric):
                    return {
                        "non_tie_winner_n": int(non_tie),
                        "replicate_n_per_example": max(1, int(replicates)),
                        "stable_winner_fraction": None,
                        "stable_winner_threshold": 0.90,
                        "status": "NA_malformed_atomic_reward_receipts",
                        "authorized": False,
                        "required_receipt": "metadata.reward_components[action] -> named atomic numeric terms",
                        "question_id": str(example.question_id),
                        "action": action.value,
                        "term": str(key),
                    }
                terms[str(key)] = numeric
            action_components.append(terms)
        names = tuple(sorted(set().union(*(set(terms) for terms in action_components))))
        if not names or any(set(terms) != set(names) for terms in action_components):
            return {
                "non_tie_winner_n": int(non_tie),
                "replicate_n_per_example": max(1, int(replicates)),
                "stable_winner_fraction": None,
                "stable_winner_threshold": 0.90,
                "status": "NA_inconsistent_atomic_reward_terms",
                "authorized": False,
                "required_receipt": "same named atomic terms for every action",
                "question_id": str(example.question_id),
            }
        if term_names is None:
            term_names = names
        elif term_names != names:
            return {
                "non_tie_winner_n": int(non_tie),
                "replicate_n_per_example": max(1, int(replicates)),
                "stable_winner_fraction": None,
                "stable_winner_threshold": 0.90,
                "status": "NA_inconsistent_atomic_reward_terms",
                "authorized": False,
                "required_receipt": "same named atomic terms for every example",
                "question_id": str(example.question_id),
            }
        baseline_values = [sum(terms[name] for name in names) for terms in action_components]
        if any(abs(float(baseline_values[index]) - float(example.branch_utilities[index])) > 1e-6 for index in range(len(ACTION_SET))):
            return {
                "non_tie_winner_n": int(non_tie),
                "replicate_n_per_example": max(1, int(replicates)),
                "stable_winner_fraction": None,
                "stable_winner_threshold": 0.90,
                "status": "NA_atomic_receipt_scalar_mismatch",
                "authorized": False,
                "required_receipt": "sum(reward_components[action].values()) == branch_utilities[action]",
                "question_id": str(example.question_id),
            }
        baseline_gap = top1_top2_gap(baseline_values)
        is_non_tie = baseline_gap > float(tolerance)
        if is_non_tie:
            non_tie += 1
        baseline_non_tie.append(is_non_tie)
        baseline_winner = max(range(len(baseline_values)), key=lambda index: baseline_values[index])
        baseline_winners.append(baseline_winner)
        for _ in range(max(1, int(replicates))):
            weights = {
                name: 1.0 + generator.uniform(-0.20, 0.20)
                for name in names
            }
            values = [
                sum(weights[name] * terms[name] for name in names)
                for terms in action_components
            ]
            winners.append(max(range(len(values)), key=lambda index: values[index]))
    if not winners:
        return {"non_tie_winner_n": non_tie, "stable_fraction": None, "status": "NA_empty"}
    # Compare each perturbation winner to the unperturbed winner.  Report tie and
    # non-tie strata separately; the promotion gate uses only non-tie rows.
    stable = 0
    non_tie_stable = 0
    tie_stable = 0
    cursor = 0
    for base_winner, is_non_tie in zip(baseline_winners, baseline_non_tie):
        for _ in range(max(1, int(replicates))):
            is_stable = int(winners[cursor] == base_winner)
            stable += is_stable
            if is_non_tie:
                non_tie_stable += is_stable
            else:
                tie_stable += is_stable
            cursor += 1
    replicate_count = max(1, int(replicates))
    non_tie_perturbation_n = non_tie * replicate_count
    tie_n = len(baseline_winners) - non_tie
    tie_perturbation_n = tie_n * replicate_count
    return {
        "non_tie_winner_n": non_tie,
        "tie_winner_n": tie_n,
        "non_tie_perturbation_n": non_tie_perturbation_n,
        "tie_perturbation_n": tie_perturbation_n,
        "replicate_n_per_example": replicate_count,
        "stable_winner_fraction": (
            non_tie_stable / non_tie_perturbation_n
            if non_tie_perturbation_n else None
        ),
        "non_tie_stable_winner_fraction": (
            non_tie_stable / non_tie_perturbation_n
            if non_tie_perturbation_n else None
        ),
        "tie_stable_winner_fraction": (
            tie_stable / tie_perturbation_n
            if tie_perturbation_n else None
        ),
        "overall_stable_winner_fraction": stable / len(winners),
        "stable_winner_threshold": 0.90,
        "non_tie_definition": "top1_minus_top2 > tolerance",
        "status": "estimable",
        "authorized": True,
    }


def _record_rate(
    row: Mapping[str, Any],
    *,
    rate_key: str,
    numerator_key: str,
    denominator_key: str,
) -> Optional[float]:
    """Read a rate only when its numerator/denominator receipt is valid.

    We intentionally do not coerce missing counters to zero.  A missing receipt is
    an unestimable safety metric, not evidence of a perfect rate.
    """

    try:
        numerator = float(row[numerator_key])
        denominator = float(row[denominator_key])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return None
    if numerator < 0.0 or numerator > denominator:
        return None
    value = numerator / denominator
    recorded = row.get(rate_key)
    if recorded is not None:
        try:
            recorded_value = float(recorded)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(recorded_value) or abs(recorded_value - value) > 1e-8:
            return None
    return float(value)


def _safety_metric_comparison(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    promotion_split: str,
    heldout_split: str,
    bootstrap_replicates: int,
) -> dict:
    """Compare safety/replication rates from actual per-seed evaluation records.

    For each seed and split, PESCO-Full is compared with the strongest available
    non-Full training baseline: minimum rate for validity/FDR and maximum rate for
    independent confirmation (replication proxy).  Deltas are oriented so values
    <= 0 are non-worse for every metric.  Any missing or inconsistent receipt makes
    that metric NA and closes the aggregate gate.
    """

    specs = {
        "validity": {
            "rate_key": "selected_invalid_branch_rate",
            "numerator_key": "selected_invalid_branch_n",
            "denominator_key": "example_count",
            "direction": "lower",
            "source": "selected_invalid_branch_rate",
        },
        "replication": {
            "rate_key": "confirmation_rate",
            "numerator_key": "confirmation_passed_n",
            "denominator_key": "confirmation_eligible_n",
            "direction": "higher",
            "source": "independent_confirmation_rate_proxy",
        },
        "false_discovery": {
            "rate_key": "invalid_local_optimization_rate",
            "numerator_key": "invalid_local_optimization_n",
            "denominator_key": "invalid_local_optimization_eligible_n",
            "direction": "lower",
            "source": "invalid_local_optimization_rate",
        },
    }
    splits = tuple(dict.fromkeys((str(promotion_split), str(heldout_split))))
    rows_by_key: Dict[tuple[int, str, str], Mapping[str, Any]] = {
        (int(row["seed"]), str(row["split"]), str(row["method"])): row
        for row in records
        if row.get("seed") is not None
        and row.get("method") in TRAINING_METHODS
        and row.get("split") in splits
    }
    output: Dict[str, Any] = {
        "status": "estimable",
        "comparison_splits": list(splits),
        "orientation": "delta <= 0 means PESCO-Full is non-worse",
        "definition": {
            "validity": "PESCO selected-invalid-branch rate versus the lowest baseline rate",
            "replication": "PESCO independent-confirmation rate versus the highest baseline rate",
            "false_discovery": "PESCO invalid-local-optimization rate versus the lowest baseline rate",
            "gate_rule": "all paired receipts present and the bootstrap 95% CI upper bound of each oriented delta is <= 0",
        },
        "metrics": {},
        "missing_receipts": [],
    }
    all_estimable = True
    for metric_name, spec in specs.items():
        deltas: list[float] = []
        paired_rows: list[dict] = []
        for split in splits:
            for seed in seeds:
                full = rows_by_key.get((int(seed), split, PESCO_METHOD))
                baselines = [
                    row for (row_seed, row_split, method), row in rows_by_key.items()
                    if row_seed == int(seed)
                    and row_split == split
                    and method in BASELINE_METHODS
                ]
                full_rate = _record_rate(
                    full or {},
                    rate_key=spec["rate_key"],
                    numerator_key=spec["numerator_key"],
                    denominator_key=spec["denominator_key"],
                )
                baseline_rates = [
                    value
                    for value in (
                        _record_rate(
                            row,
                            rate_key=spec["rate_key"],
                            numerator_key=spec["numerator_key"],
                            denominator_key=spec["denominator_key"],
                        )
                        for row in baselines
                    )
                    if value is not None
                ]
                if full_rate is None or not baseline_rates:
                    all_estimable = False
                    output["missing_receipts"].append({
                        "metric": metric_name,
                        "split": split,
                        "seed": int(seed),
                        "full_receipt": full is not None,
                        "baseline_receipt_count": len(baseline_rates),
                        "required_fields": [
                            spec["rate_key"],
                            spec["numerator_key"],
                            spec["denominator_key"],
                        ],
                    })
                    continue
                reference = min(baseline_rates) if spec["direction"] == "lower" else max(baseline_rates)
                delta = full_rate - reference if spec["direction"] == "lower" else reference - full_rate
                deltas.append(float(delta))
                paired_rows.append({
                    "seed": int(seed),
                    "split": split,
                    "full": float(full_rate),
                    "baseline_reference": float(reference),
                    "delta": float(delta),
                })
        ci = _bootstrap_ci(
            deltas,
            seed=4100 + sum(ord(char) for char in metric_name),
            replicates=bootstrap_replicates,
        )
        metric_estimable = bool(deltas) and len(deltas) == len(seeds) * len(splits) and ci.get("status") == "estimable"
        if not metric_estimable:
            all_estimable = False
        output["metrics"][metric_name] = {
            "source": spec["source"],
            "direction": spec["direction"],
            "paired_count": len(deltas),
            "expected_paired_count": len(seeds) * len(splits),
            "ci": ci,
            "gate": bool(metric_estimable and ci.get("upper") is not None and float(ci["upper"]) <= 0.0),
            "paired_rows": paired_rows,
        }
    output["status"] = "estimable_from_evaluation_records" if all_estimable else "NA_missing_or_invalid_receipts"
    output["validity"] = bool(output["metrics"]["validity"]["gate"])
    output["replication"] = bool(output["metrics"]["replication"]["gate"])
    output["false_discovery"] = bool(output["metrics"]["false_discovery"]["gate"])
    return output


def _best_baseline(records: Sequence[Mapping[str, Any]], split: str) -> Optional[str]:
    eligible = [
        row for row in records
        if row.get("method") in BASELINE_METHODS and row.get("split") == split
        and math.isfinite(_finite(row.get("normalized_regret"), float("nan")))
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: _finite(row.get("normalized_regret")))["method"]


def run_p2_experiment(
    output_dir: str | Path,
    dataset: DecisionDataset,
    *,
    config: Optional[P2Config] = None,
    repo_root: str | Path | None = None,
    command: Sequence[str] | None = None,
    data_paths: Iterable[str | Path] = (),
) -> dict:
    """Run the ten-seed CPU comparison and write all audit artifacts.

    ``command`` and ``data_paths`` let the CLI attach an execution-boundary
    provenance record.  They are optional so existing programmatic callers keep
    working; the manifest still records the interpreter and semantic seeds when
    they are omitted.
    """

    config = config or P2Config()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_splits = _dataset_splits(dataset)
    available_splits = set(dataset_splits)
    if config.baseline_selection_split is not None:
        baseline_selection_split = str(config.baseline_selection_split)
    elif "dev" in available_splits:
        baseline_selection_split = "dev"
    elif "tune" in available_splits:
        baseline_selection_split = "tune"
    else:
        baseline_selection_split = config.promotion_split
    baseline_selection_on_promotion = baseline_selection_split == config.promotion_split
    full_compute_budget = int(config.trainer.max_optimizer_steps) >= 64
    if len(config.seeds) != 10:
        # The feedback's decisive experiment is explicitly a ten-seed experiment.
        # A smaller smoke run is allowed only when marked diagnostic in the output.
        seed_count_status = "diagnostic_non_ten_seed"
    else:
        seed_count_status = "ten_seed_full_budget" if full_compute_budget else "ten_seed_diagnostic_budget"
    records: list[dict] = []
    logs: dict[str, dict] = {}
    trainer_template = config.trainer
    for seed in config.seeds:
        for method in TRAINING_METHODS:
            trainer_config = DifferentiableTrainerConfig(
                epochs=trainer_template.epochs,
                batch_size=trainer_template.batch_size,
                learning_rate=trainer_template.learning_rate,
                hidden_dim=trainer_template.hidden_dim,
                seed=int(seed),
                max_optimizer_steps=trainer_template.max_optimizer_steps,
                state_loss_weight=trainer_template.state_loss_weight,
                belief_loss_weight=trainer_template.belief_loss_weight,
                flip_loss_weight=trainer_template.flip_loss_weight,
                kl_weight=trainer_template.kl_weight,
                entropy_weight=trainer_template.entropy_weight,
                constraint_loss_weight=trainer_template.constraint_loss_weight,
                gradient_clip_norm=trainer_template.gradient_clip_norm,
                temperature=trainer_template.temperature,
            )
            policy, log = DifferentiableStrategyTrainer(trainer_config).fit(dataset, method)
            key = f"seed_{int(seed):02d}/{method}"
            logs[key] = log.to_dict()
            for split in dataset_splits:
                metrics = _method_metrics(policy, dataset, split, tie_tol=config.tie_tolerance)
                metrics.update({"seed": int(seed), "method": method})
                records.append(metrics)
            # A ten-seed comparison can instantiate dozens of CPU PyTorch policies;
            # release each one before the next method/seed so allocator-retained
            # blocks do not turn a diagnostic run into an accidental OOM.
            del policy
            gc.collect()
    # The evaluator branch search is one deterministic upper bound, not a trained
    # method.  It is recorded once per split so downstream tables cannot mistake it
    # for a tenth training seed.
    for split in dataset_splits:
        row = _branch_search_metrics(dataset, split, tie_tol=config.tie_tolerance)
        row.update({"seed": None, "method": INFERENCE_SEARCH})
        records.append(row)

    deltas: list[float] = []
    sft_deltas: list[float] = []
    flip_deltas: list[float] = []
    best_question_deltas: list[dict[str, Any]] = []
    sft_question_deltas: list[dict[str, Any]] = []
    flip_question_deltas: list[dict[str, Any]] = []
    flip_best_question_deltas: list[dict[str, Any]] = []
    per_seed: list[dict] = []
    for seed in config.seeds:
        full = next((row for row in records if row.get("seed") == seed and row.get("method") == PESCO_METHOD and row.get("split") == config.promotion_split), None)
        selection_rows = [
            row for row in records
            if row.get("seed") == seed
            and row.get("split") == baseline_selection_split
            and row.get("method") in BASELINE_METHODS
        ]
        selected_baseline = min(
            selection_rows,
            key=lambda row: _finite(row.get("normalized_regret"), 1e9),
        ) if selection_rows else None
        selected_baseline_method = selected_baseline.get("method") if selected_baseline else None
        best = next(
            (
                row for row in records
                if row.get("seed") == seed
                and row.get("split") == config.promotion_split
                and row.get("method") == selected_baseline_method
            ),
            None,
        )
        delta = (_finite(full.get("normalized_regret"), float("nan")) - _finite(best.get("normalized_regret"), float("nan"))) if full and best else float("nan")
        if math.isfinite(delta):
            deltas.append(delta)
        if full and best:
            best_by_question = {
                str(row.get("question_id")): row
                for row in best.get("question_metric_rows", [])
            }
            for row in full.get("question_metric_rows", []):
                question_id = str(row.get("question_id"))
                reference = best_by_question.get(question_id)
                if reference is None:
                    continue
                value = _finite(row.get("normalized_regret"), float("nan")) - _finite(reference.get("normalized_regret"), float("nan"))
                if math.isfinite(value):
                    best_question_deltas.append({
                        "seed": int(seed),
                        "question_id": question_id,
                        "family": str(row.get("family", reference.get("family", "unknown"))),
                        "value": value,
                    })
        sft = next((row for row in records if row.get("seed") == seed and row.get("method") == "SFT" and row.get("split") == config.promotion_split), None)
        sft_delta = (_finite(full.get("normalized_regret"), float("nan")) - _finite(sft.get("normalized_regret"), float("nan"))) if full and sft else float("nan")
        if math.isfinite(sft_delta):
            sft_deltas.append(sft_delta)
        if full and sft:
            sft_by_question = {
                str(row.get("question_id")): row
                for row in sft.get("question_metric_rows", [])
            }
            for row in full.get("question_metric_rows", []):
                question_id = str(row.get("question_id"))
                reference = sft_by_question.get(question_id)
                if reference is None:
                    continue
                value = _finite(row.get("normalized_regret"), float("nan")) - _finite(reference.get("normalized_regret"), float("nan"))
                if math.isfinite(value):
                    sft_question_deltas.append({
                        "seed": int(seed),
                        "question_id": question_id,
                        "family": str(row.get("family", reference.get("family", "unknown"))),
                        "value": value,
                    })
        held_full = next((row for row in records if row.get("seed") == seed and row.get("method") == PESCO_METHOD and row.get("split") == config.heldout_split), None)
        held_best = next(
            (
                row for row in records
                if row.get("seed") == seed
                and row.get("split") == config.heldout_split
                and row.get("method") == selected_baseline_method
            ),
            None,
        )
        held_noflip = next((row for row in records if row.get("seed") == seed and row.get("method") == "PESCO-NoFlipLoss" and row.get("split") == config.heldout_split), None)
        full_flip = _finite(held_full.get("pairwise_reversal_ranking_accuracy"), float("nan")) if held_full else float("nan")
        noflip_flip = _finite(held_noflip.get("pairwise_reversal_ranking_accuracy"), float("nan")) if held_noflip else float("nan")
        best_flip = _finite(held_best.get("pairwise_reversal_ranking_accuracy"), float("nan")) if held_best else float("nan")
        if math.isfinite(full_flip) and math.isfinite(noflip_flip):
            # PairRankAcc promotion gate is preregistered against NoFlip, the
            # ablation that isolates the reversal loss.  Keep best-non-Full as a
            # secondary diagnostic below.
            flip_deltas.append(full_flip - noflip_flip)
        if held_full and held_noflip:
            full_pairs = {str(row.get("question_id")): row for row in held_full.get("pairwise_reversal_question_rows", [])}
            noflip_pairs = {str(row.get("question_id")): row for row in held_noflip.get("pairwise_reversal_question_rows", [])}
            for question_id, row in full_pairs.items():
                reference = noflip_pairs.get(question_id)
                if reference is None:
                    continue
                value = _finite(row.get("pairwise_reversal_ranking_accuracy"), float("nan")) - _finite(reference.get("pairwise_reversal_ranking_accuracy"), float("nan"))
                if math.isfinite(value):
                    flip_question_deltas.append({
                        "seed": int(seed),
                        "question_id": question_id,
                        "family": str(row.get("family", reference.get("family", "unknown"))),
                        "value": value,
                    })
        if held_full and held_best:
            full_pairs = {str(row.get("question_id")): row for row in held_full.get("pairwise_reversal_question_rows", [])}
            best_pairs = {str(row.get("question_id")): row for row in held_best.get("pairwise_reversal_question_rows", [])}
            for question_id, row in full_pairs.items():
                reference = best_pairs.get(question_id)
                if reference is None:
                    continue
                value = _finite(row.get("pairwise_reversal_ranking_accuracy"), float("nan")) - _finite(reference.get("pairwise_reversal_ranking_accuracy"), float("nan"))
                if math.isfinite(value):
                    flip_best_question_deltas.append({
                        "seed": int(seed),
                        "question_id": question_id,
                        "family": str(row.get("family", reference.get("family", "unknown"))),
                        "value": value,
                    })
        per_seed.append({
            "seed": int(seed),
            "baseline_selection_split": baseline_selection_split,
            "baseline_selected_on_promotion": baseline_selection_on_promotion,
            "promotion_best_baseline": selected_baseline_method,
            "delta_regret": delta if math.isfinite(delta) else None,
            "full_better_regret": bool(math.isfinite(delta) and delta < 0.0),
            "full_better_than_sft": bool(math.isfinite(sft_delta) and sft_delta < 0.0),
            "delta_regret_vs_sft": sft_delta if math.isfinite(sft_delta) else None,
            "heldout_pairwise_reversal_ranking_accuracy_delta": (
                full_flip - noflip_flip
                if math.isfinite(full_flip) and math.isfinite(noflip_flip) else None
            ),
            # Compatibility alias for earlier artifact readers.
            "heldout_flip_delta": (
                full_flip - noflip_flip
                if math.isfinite(full_flip) and math.isfinite(noflip_flip) else None
            ),
            "heldout_pairrank_reference": "PESCO-NoFlipLoss",
            "heldout_pairrank_delta_vs_best_non_full": (
                full_flip - best_flip
                if math.isfinite(full_flip) and math.isfinite(best_flip) else None
            ),
        })
    # Primary uncertainty is explicitly two-level.  The seed-only values above are
    # retained only as a compatibility/audit count and are never used for the gate.
    primary_ci = _two_level_seed_question_ci(
        best_question_deltas,
        value_key="value",
        seed=2026,
        replicates=config.bootstrap_replicates,
    )
    primary_ci["seed_only_audit"] = _bootstrap_ci(deltas, seed=2026, replicates=config.bootstrap_replicates)
    primary_ci["family_leave_one_out"] = _family_leave_one_out_cis(
        best_question_deltas,
        value_key="value",
        seed=20260,
        replicates=config.bootstrap_replicates,
    )
    sft_ci = _two_level_seed_question_ci(
        sft_question_deltas,
        value_key="value",
        seed=2028,
        replicates=config.bootstrap_replicates,
    )
    sft_ci["seed_only_audit"] = _bootstrap_ci(sft_deltas, seed=2028, replicates=config.bootstrap_replicates)
    sft_ci["family_leave_one_out"] = _family_leave_one_out_cis(
        sft_question_deltas,
        value_key="value",
        seed=20280,
        replicates=config.bootstrap_replicates,
    )
    flip_ci = _two_level_seed_question_ci(
        flip_question_deltas,
        value_key="value",
        seed=2027,
        replicates=config.bootstrap_replicates,
    )
    flip_ci["seed_only_audit"] = _bootstrap_ci(flip_deltas, seed=2027, replicates=config.bootstrap_replicates)
    flip_ci["family_leave_one_out"] = _family_leave_one_out_cis(
        flip_question_deltas,
        value_key="value",
        seed=20270,
        replicates=config.bootstrap_replicates,
    )
    flip_ci["reference"] = "PESCO-NoFlipLoss"
    flip_ci["best_non_full_secondary"] = _two_level_seed_question_ci(
        flip_best_question_deltas,
        value_key="value",
        seed=20271,
        replicates=config.bootstrap_replicates,
    )
    # Safety/replication gates must be derived from the per-seed evaluation
    # receipts.  Missing counters are NA (never silently interpreted as zero), and
    # the aggregate gate is therefore fail-closed when a required comparison is not
    # estimable.  ``confirmation_rate`` is explicitly labelled as the available
    # independent-confirmation replication proxy; a future benchmark can replace it
    # with a dedicated replication-success receipt without changing the gate API.
    no_worse = _safety_metric_comparison(
        records,
        seeds=config.seeds,
        promotion_split=config.promotion_split,
        heldout_split=config.heldout_split,
        bootstrap_replicates=config.bootstrap_replicates,
    )
    # Require an actual CI entirely below zero; point estimates alone never promote.
    gates = {
        "ten_training_seeds": len(config.seeds) == 10,
        "preregistered_optimizer_step_budget": full_compute_budget,
        "baseline_selected_before_promotion": not baseline_selection_on_promotion,
        "promotion_regret_ci_entirely_below_zero": primary_ci.get("lower") is not None and primary_ci.get("upper") is not None and float(primary_ci["upper"]) < 0.0,
        "heldout_pairwise_reversal_ranking_ci_entirely_above_zero": flip_ci.get("lower") is not None and flip_ci.get("upper") is not None and float(flip_ci["lower"]) > 0.0,
        "at_least_8_of_10_seeds_same_regret_direction": sum(bool(row["full_better_regret"]) for row in per_seed) >= 8 if len(config.seeds) == 10 else False,
        "reward_perturbation_winner_stability_90_percent": False,
        "no_worse_validity_replication_false_discovery": bool(
            no_worse.get("status") == "estimable_from_evaluation_records"
            and no_worse.get("validity")
            and no_worse.get("replication")
            and no_worse.get("false_discovery")
        ),
        "beats_sft": bool(
            sft_ci.get("lower") is not None
            and sft_ci.get("upper") is not None
            and float(sft_ci["upper"]) < 0.0
            and sum(bool(row.get("full_better_than_sft")) for row in per_seed) >= 8
        ),
        "promotion_pair_count_minimum": sum(
            bool(pair.confirmed)
            and dataset.examples[pair.left].split == config.promotion_split
            and dataset.examples[pair.right].split == config.promotion_split
            and dataset.examples[pair.left].question_id == dataset.examples[pair.right].question_id
            for pair in dataset.reversals
        ) >= config.min_formal_pairs,
        "promotion_cluster_count_minimum": len({example.question_id for example in dataset.examples if example.split == config.promotion_split}) >= config.min_formal_clusters,
    }
    # Compatibility alias retained for readers of the pre-P2.1 schema.  The gate is
    # evaluated from the explicitly named PairRankAcc field above.
    gates["heldout_same_question_flip_ci_entirely_above_zero"] = gates[
        "heldout_pairwise_reversal_ranking_ci_entirely_above_zero"
    ]
    # Reward stability is evaluated independently only when the dataset carries an
    # auditable atomic reward decomposition; the helper also reports tie/non-tie
    # strata using the top1-minus-top2 gap.
    stability = _perturbed_winner_stability(
        dataset,
        tolerance=config.tie_tolerance,
        replicates=100,
        seed=2030,
    )
    gates["reward_perturbation_winner_stability_90_percent"] = bool(
        stability.get("authorized", False)
        and stability.get("stable_winner_fraction") is not None
        and float(stability["stable_winner_fraction"]) >= 0.90
    )
    promotion_status = "GO" if all(gates.values()) else "NO-GO"
    result = {
        "schema_version": "pesco_tier1_p2_experiment_v0.1",
        "status": (
            "completed_cpu_ten_seed" if len(config.seeds) == 10 and full_compute_budget
            else "completed_cpu_ten_seed_diagnostic_budget" if len(config.seeds) == 10
            else "completed_cpu_diagnostic"
        ),
        "promotion_status": promotion_status,
        "primary_metric": "Delta_R = R_PESCO-Full - R_best_non_Full_training_baseline",
        "primary_metric_definition": "normalized research regret on promotion split; lower is better",
        "superiority_claim": "no proof of superiority unless the two-level seed×question CI is entirely below zero",
        "regret_uncertainty_method": "two-level bootstrap: seed resampling then question-cluster resampling; family leave-one-out reported",
        "promotion_split": config.promotion_split,
        "baseline_selection_split": baseline_selection_split,
        "baseline_selection_on_promotion": baseline_selection_on_promotion,
        "baseline_selection_status": (
            "legacy_fallback_on_promotion_not_formal"
            if baseline_selection_on_promotion
            else "locked_before_promotion"
        ),
        "heldout_split": config.heldout_split,
        "seed_count": len(config.seeds),
        "seed_count_status": seed_count_status,
        "seeds": list(config.seeds),
        "methods": list(TRAINING_METHODS) + [INFERENCE_SEARCH],
        "per_seed": per_seed,
        "primary_delta_regret": primary_ci,
        "primary_question_delta_count": len(best_question_deltas),
        "sft_question_delta_count": len(sft_question_deltas),
        "heldout_pairrank_question_delta_count": len(flip_question_deltas),
        "heldout_pairwise_reversal_ranking_accuracy_delta": flip_ci,
        "heldout_flip_accuracy_delta": flip_ci,
        "promotion_delta_regret_vs_sft": sft_ci,
        "reward_perturbation_stability": stability,
        "no_worse_safety_metrics": no_worse,
        "gates": gates,
        "records": records,
        "training_logs": logs,
        "formal_comparison_authorized": bool(all(gates.values())),
        "diagnostic_only": not bool(all(gates.values())),
        "dataset_provenance": dict(dataset.provenance),
        "limitations": [
            "The checked-in v0.3 dataset is underpowered when promotion/held-out pair counts are below preregistered thresholds.",
            "Inference-time branch search is an evaluator upper bound and is excluded from the learned-baseline winner.",
            "SMOPD parameter parity is reported explicitly; teacher/student update counts are not silently treated as equal FLOPs.",
            "A non-negative or interval-overlapping regret delta is reported as no proof of superiority; it is not described as a significant degradation.",
        ],
    }
    (output / "p2_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "p2_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "p2_gate.json").write_text(json.dumps({
        "promotion_status": promotion_status,
        "superiority_claim": "no proof of superiority unless two-level seed×question regret CI is entirely below zero",
        "regret_uncertainty_method": "seed_then_question_cluster_bootstrap",
        "gates": gates,
        "primary_delta_regret": primary_ci,
        "heldout_pairwise_reversal_ranking_accuracy_delta": flip_ci,
        "heldout_flip_accuracy_delta": flip_ci,
        "promotion_delta_regret_vs_sft": sft_ci,
        "no_worse_safety_metrics": no_worse,
        "reward_perturbation_stability": stability,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    manifest = build_run_manifest(
        experiment="tier1_p2_ten_seed_cpu",
        repo_root=root,
        command=command,
        runner_paths=[
            root / "scripts/run_tier1_p2.py",
            root / "research_strategy_optimization/evaluation/tier1_p2_experiments.py",
            root / "research_strategy_optimization/algorithms/differentiable_strategy.py",
            root / "research_strategy_optimization/schemas.py",
            root / "research_strategy_optimization/utils/run_manifest.py",
        ],
        data_paths=data_paths,
        seeds={"training": list(config.seeds)},
        checkpoint=None,
        status="completed" if len(config.seeds) == 10 and full_compute_budget else "diagnostic",
        diagnostics={
            "capture_mode": "in_run",
            "promotion_status": promotion_status,
            "dataset_schema": dataset.schema_version,
            "seed_count": len(config.seeds),
            "methods": list(TRAINING_METHODS),
            "optimizer_step_cap": config.trainer.max_optimizer_steps,
            "promotion_split": config.promotion_split,
            "heldout_split": config.heldout_split,
            "formal_comparison_authorized": bool(all(gates.values())),
            "diagnostic_only": not bool(all(gates.values())),
            "safety_metric_status": no_worse.get("status"),
            "reward_perturbation_status": stability.get("status"),
        },
    )
    write_run_manifest(output / "run_manifest.json", manifest)
    return result


__all__ = [
    "TRAINING_METHODS",
    "BASELINE_METHODS",
    "INFERENCE_SEARCH",
    "P2Config",
    "tie_set",
    "top1_top2_gap",
    "normalized_regret",
    "run_p2_experiment",
]

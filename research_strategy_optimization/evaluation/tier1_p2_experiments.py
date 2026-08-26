"""Pre-registered Tier-1 CPU promotion experiment (P2).

The v0.3/v0.4 benchmark builders are intentionally separate from this module.  This
runner consumes a frozen :class:`DecisionDataset`, trains every named policy from the
same data and optimizer budget for ten independent seeds, and writes a fail-closed
gate report.  It is useful both for the small checked-in diagnostic dataset and for a
larger v0.4 dataset once one is available.

The only primary comparison is ``Delta_R`` on the promotion split::

    R(PESCO-Full) - R(best non-PESCO training baseline)

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

    preferred = ("train", "dev", "final_id", "final_ood", "diagnostic_ood", "test")
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
    consequently made the ±20% stability check almost tautological.  The v0.4
    collector currently stores no ``reward_components`` receipt, so fail closed and
    return an explicit NA record.  A future collector may provide, per action,
    ``metadata["reward_components"][action]`` as a mapping of named atomic terms;
    that is the only schema accepted by this gate.
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
                max(example.branch_utilities) - min(example.branch_utilities) > float(tolerance)
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
        baseline_gap = max(baseline_values) - min(baseline_values)
        if baseline_gap > float(tolerance):
            non_tie += 1
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
    # Compare each perturbation winner to the unperturbed winner.  The order is
    # example-major, so the parallel baseline_winners list is sufficient.
    stable = 0
    cursor = 0
    for base_winner in baseline_winners:
        for _ in range(max(1, int(replicates))):
            stable += int(winners[cursor] == base_winner)
            cursor += 1
    return {
        "non_tie_winner_n": non_tie,
        "replicate_n_per_example": max(1, int(replicates)),
        "stable_winner_fraction": stable / len(winners),
        "stable_winner_threshold": 0.90,
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
    non-PESCO training baseline: minimum rate for validity/FDR and maximum rate for
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
    per_seed: list[dict] = []
    for seed in config.seeds:
        full = next((row for row in records if row.get("seed") == seed and row.get("method") == PESCO_METHOD and row.get("split") == config.promotion_split), None)
        baseline_rows = [row for row in records if row.get("seed") == seed and row.get("split") == config.promotion_split and row.get("method") in BASELINE_METHODS]
        best = min(baseline_rows, key=lambda row: _finite(row.get("normalized_regret"), 1e9)) if baseline_rows else None
        delta = (_finite(full.get("normalized_regret"), float("nan")) - _finite(best.get("normalized_regret"), float("nan"))) if full and best else float("nan")
        if math.isfinite(delta):
            deltas.append(delta)
        sft = next((row for row in records if row.get("seed") == seed and row.get("method") == "SFT" and row.get("split") == config.promotion_split), None)
        sft_delta = (_finite(full.get("normalized_regret"), float("nan")) - _finite(sft.get("normalized_regret"), float("nan"))) if full and sft else float("nan")
        if math.isfinite(sft_delta):
            sft_deltas.append(sft_delta)
        held_full = next((row for row in records if row.get("seed") == seed and row.get("method") == PESCO_METHOD and row.get("split") == config.heldout_split), None)
        held_best = min((row for row in records if row.get("seed") == seed and row.get("split") == config.heldout_split and row.get("method") in BASELINE_METHODS), key=lambda row: _finite(row.get("normalized_regret"), 1e9), default=None)
        full_flip = _finite(held_full.get("pair_flip_accuracy"), float("nan")) if held_full else float("nan")
        best_flip = _finite(held_best.get("pair_flip_accuracy"), float("nan")) if held_best else float("nan")
        if math.isfinite(full_flip) and math.isfinite(best_flip):
            flip_deltas.append(full_flip - best_flip)
        per_seed.append({
            "seed": int(seed),
            "promotion_best_baseline": best.get("method") if best else None,
            "delta_regret": delta if math.isfinite(delta) else None,
            "full_better_regret": bool(math.isfinite(delta) and delta < 0.0),
            "full_better_than_sft": bool(math.isfinite(sft_delta) and sft_delta < 0.0),
            "delta_regret_vs_sft": sft_delta if math.isfinite(sft_delta) else None,
            "heldout_flip_delta": (full_flip - best_flip) if math.isfinite(full_flip) and math.isfinite(best_flip) else None,
        })
    primary_ci = _bootstrap_ci(deltas, seed=2026, replicates=config.bootstrap_replicates)
    sft_ci = _bootstrap_ci(sft_deltas, seed=2028, replicates=config.bootstrap_replicates)
    flip_ci = _bootstrap_ci(flip_deltas, seed=2027, replicates=config.bootstrap_replicates)
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
        "promotion_regret_ci_entirely_below_zero": primary_ci.get("lower") is not None and primary_ci.get("upper") is not None and float(primary_ci["upper"]) < 0.0,
        "heldout_same_question_flip_ci_entirely_above_zero": flip_ci.get("lower") is not None and flip_ci.get("upper") is not None and float(flip_ci["lower"]) > 0.0,
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
    # Reward stability is evaluated independently only when the dataset carries an
    # auditable atomic reward decomposition.  Current v0.4 rows carry a scalar
    # utility, so this intentionally returns NA and closes the gate.
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
        "primary_metric": "Delta_R = R_PESCO-Full - R_best_non_PESCO_training_baseline",
        "primary_metric_definition": "normalized research regret on promotion split; lower is better",
        "promotion_split": config.promotion_split,
        "heldout_split": config.heldout_split,
        "seed_count": len(config.seeds),
        "seed_count_status": seed_count_status,
        "seeds": list(config.seeds),
        "methods": list(TRAINING_METHODS) + [INFERENCE_SEARCH],
        "per_seed": per_seed,
        "primary_delta_regret": primary_ci,
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
        ],
    }
    (output / "p2_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "p2_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "p2_gate.json").write_text(json.dumps({
        "promotion_status": promotion_status,
        "gates": gates,
        "primary_delta_regret": primary_ci,
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
    "normalized_regret",
    "run_p2_experiment",
]

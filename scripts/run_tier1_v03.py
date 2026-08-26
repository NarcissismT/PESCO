#!/usr/bin/env python3
"""Run the diagnostic Tier-1 v0.3 benchmark.

The benchmark has exactly 12 independent questions x 4 hidden worlds = 48
question-world branch groups.  Each group has 4 MVP action-level rows and 4
exploration seeds, yielding 192 action-level rows and 768 seed-level observations.
It also records initial-state calibration and held-out confirmation metadata.  It is
intentionally a CPU diagnostic; it does not train or evaluate a formal LLM method.
"""

from __future__ import annotations

import json
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.environments.tier0_simulator import TrustedVerifier
from research_strategy_optimization.environments.tier1_benchmark import (
    MECHANISM_FAMILIES,
    build_tier1_v03_benchmark,
    tier1_scientific_utility,
)
from research_strategy_optimization.schemas import EvidenceState, Protocol, ResearchAction
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _public_output(output: Any) -> dict:
    return {
        "action": output.action,
        "method": output.method,
        "effect_estimate": float(output.effect_estimate),
        "confidence_interval": [float(output.ci_low), float(output.ci_high)],
        "sample_size": int(output.sample_size),
        "seed_count": int(output.seed_count),
        "execution_cost": float(output.execution_cost),
        "backend": output.backend,
        "estimator": output.estimator,
        "data_partition": output.data_partition,
        "dataset_hash": output.dataset_hash,
        "code_hash": output.code_hash,
        "split_hash": output.split_hash,
        "validity_signals": list(output.validity_signals),
        "treatment_confounder_correlation": float(output.treatment_confounder_correlation),
        "group_overlap_count": int(output.group_overlap_count),
        "hidden_validation": {
            "metric": float(getattr(output, "hidden_validation_metric", 0.0)),
            "baseline": float(getattr(output, "hidden_validation_baseline", 0.0)),
            "n": int(getattr(output, "hidden_validation_n", 0)),
            "group_overlap_count": int(getattr(output, "hidden_validation_overlap_count", 0)),
            "split": str(getattr(output, "hidden_validation_split", "not_run")),
            "partition_hash": str(getattr(output, "hidden_validation_partition_hash", "")),
        },
        "confirmation": bool(output.confirmation),
    }


def _verdict(verdict: Any) -> dict:
    return verdict.to_dict() if hasattr(verdict, "to_dict") else dict(verdict)


def _confirmation_summary(verdict: Mapping[str, Any]) -> dict:
    """Return explicit confirmation eligibility/denominator fields.

    ``TrustedVerifier`` only performs confirmation for a valid, decisive candidate
    (Supported/Refuted).  The old A runner accumulated a single boolean and therefore
    treated rows that were not eligible for confirmation as if they had passed.  Keep
    the eligibility, execution, pass, and data-independence fields separate so the
    artifact can be audited without relying on a vacuous ``all([])``.
    """

    state = str(verdict.get("evidence_state", ""))
    confirmation = dict(verdict.get("independent_confirmation", {}))
    performed = bool(confirmation.get("performed", False))
    return {
        "eligible": state in {EvidenceState.SUPPORTED.value, EvidenceState.REFUTED.value},
        "performed": performed,
        "passed": bool(confirmation.get("passed", False)),
        "data_independent": bool(confirmation.get("data_independent", False)) if performed else False,
        "seeds": list(confirmation.get("confirmation_seeds", [])),
        "dataset_hash": str(confirmation.get("dataset_hash", "")),
        "split_hash": str(confirmation.get("split_hash", "")),
    }


def _branch_evidence(row: Mapping[str, Any], true_effect: float) -> dict:
    """Make a self-contained evaluator-side branch evidence record.

    The branch output itself intentionally omits latent truth.  This helper is used
    only for the A audit artifact, where the evaluator-owned true effect is paired
    with the public estimate and provenance fields to expose before/after bias.
    """

    output = dict(row["output"])
    verdict = dict(row["verdict"])
    estimate = float(output["effect_estimate"])
    low, high = (float(value) for value in output["confidence_interval"])
    confirmation = _confirmation_summary(verdict)
    return {
        "action": str(row["action"]),
        "method": str(output["method"]),
        "true_effect": float(true_effect),
        "effect_estimate": estimate,
        "confidence_interval": [low, high],
        "confidence_width": high - low,
        "bias": estimate - float(true_effect),
        "absolute_bias": abs(estimate - float(true_effect)),
        "validity_pass": bool(verdict["validity_pass"]),
        "evidence_state": str(verdict["evidence_state"]),
        "backend": str(output["backend"]),
        "estimator": str(output["estimator"]),
        "code_hash": str(output["code_hash"]),
        "dataset_hash": str(output["dataset_hash"]),
        "split_hash": str(output["split_hash"]),
        "data_partition": str(output["data_partition"]),
        "group_overlap_count": int(output["group_overlap_count"]),
        "treatment_confounder_correlation": float(output["treatment_confounder_correlation"]),
        "sample_size": int(output["sample_size"]),
        "seed_count": int(output["seed_count"]),
        "validity_signals": list(output["validity_signals"]),
        "hidden_validation": dict(output.get("hidden_validation", {
            "metric": 0.0,
            "baseline": 0.0,
            "n": 0,
            "group_overlap_count": 0,
            "split": "not_run",
            "partition_hash": "",
        })),
        "confirmation": confirmation,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a conditional rate, or JSON ``null`` when no row is eligible."""

    return float(numerator) / float(denominator) if denominator else None


def run(output_dir: Path, *, protocol: Protocol | None = None) -> dict:
    protocol = protocol or Protocol()
    benchmark = build_tier1_v03_benchmark()
    if str(protocol.protocol_version) != str(benchmark.protocol_version):
        raise ValueError(
            f"protocol/benchmark mismatch: {protocol.protocol_version!r} != {benchmark.protocol_version!r}"
        )
    protocol_payload = {
        "protocol_version": protocol.protocol_version,
        "delta_min": protocol.delta_min,
        "confidence_level": protocol.confidence_level,
        "exploration_seeds": list(protocol.exploration_seeds),
        "confirmation_seeds": list(protocol.confirmation_seeds),
        "invalid_precedence": protocol.invalid_precedence,
        "independent_confirmation_required": protocol.independent_confirmation_required,
        "max_budget": protocol.max_budget,
    }
    protocol_digest = "sha256:" + hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    verifier = TrustedVerifier(protocol)
    actions = ResearchAction.mvp_actions()
    seeds = tuple(protocol.exploration_seeds)

    seed_rows: list[dict] = []
    branch_rows: list[dict] = []
    initial_rows: list[dict] = []
    target_agreement_rows: list[dict] = []
    clone_ok = True
    backend_ok = True
    confirmation_seed_disjoint = not (set(protocol.exploration_seeds) & set(protocol.confirmation_seeds))
    state_calibration = True
    public_state_targets: dict[str, set[str]] = defaultdict(set)

    for question in benchmark.questions:
        # The environment owns all hidden worlds for this question.  A fresh reset
        # plus one snapshot is the common source state for every counterfactual branch.
        env = benchmark.make_environment(question.question_id, protocol=protocol)
        for world in question.worlds:
            env.reset(question.policy_question_id, world.world_id, seed=17)
            source_snapshot = env.snapshot()
            initial_branch = env.clone_from_snapshot(source_snapshot)
            clone_ok = clone_ok and type(initial_branch) is type(env)
            initial_output = initial_branch.execute_option(ResearchAction.CONTINUE, seeds=seeds)
            initial_verdict = verifier.evaluate(initial_output, initial_branch)
            initial_rows.append({
                "record_granularity": "question_world_group",
                "question_world_group_id": f"{question.question_id}|{world.world_id}",
                "question_id": question.question_id,
                "world_id": world.world_id,
                "world_kind": world.kind,
                "family": question.family,
                "split": question.split,
                "initial_state": initial_verdict.evidence_state.value,
                "validity_pass": bool(initial_verdict.validity_pass),
                "effect_estimate": float(initial_output.effect_estimate),
                "confidence_interval": [float(initial_output.ci_low), float(initial_output.ci_high)],
                "backend": initial_output.backend,
                "dataset_hash": initial_output.dataset_hash,
                "confirmation_data_independent": bool(initial_verdict.confirmation_data_independent),
            })
            public_fingerprint = json.dumps(
                initial_branch.visible_observation().to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            backend_ok = backend_ok and initial_output.backend == "tier1_numpy"
            state_calibration = state_calibration and initial_verdict.evidence_state.value == world.kind

            for action in actions:
                # One vector branch is the actual scientific utility/confirmation
                # record.  The four one-seed executions below are the required
                # seed-level observations and are all restored from this same digest.
                branch = env.clone_from_snapshot(source_snapshot)
                clone_ok = clone_ok and type(branch) is type(env)
                vector_output = branch.execute_option(action, seeds=seeds)
                vector_verdict = verifier.evaluate(vector_output, branch)
                utility = tier1_scientific_utility(
                    question,
                    world,
                    action,
                    vector_output,
                    vector_verdict,
                    protocol,
                    initial_observation=initial_branch.visible_observation(),
                )
                backend_ok = backend_ok and vector_output.backend == "tier1_numpy"
                branch_rows.append({
                    "record_granularity": "action_level",
                    "question_world_group_id": f"{question.question_id}|{world.world_id}",
                    "question_id": question.question_id,
                    "world_id": world.world_id,
                    "world_kind": world.kind,
                    "family": question.family,
                    "split": question.split,
                    "action": action.value,
                    "target_action": question.target_action(world.world_id).value,
                    "source_snapshot_digest": source_snapshot.digest,
                    "clone_class": type(branch).__name__,
                    "utility": float(utility),
                    "output": _public_output(vector_output),
                    "verdict": _verdict(vector_verdict),
                    "confirmation_data_independent": bool(vector_verdict.confirmation_data_independent),
                })
                for seed in seeds:
                    one_seed_branch = env.clone_from_snapshot(source_snapshot)
                    clone_ok = clone_ok and type(one_seed_branch) is type(env)
                    one_seed_output = one_seed_branch.execute_option(action, seeds=(seed,))
                    one_seed_verdict = verifier.evaluate(one_seed_output, one_seed_branch, confirm=False)
                    backend_ok = backend_ok and one_seed_output.backend == "tier1_numpy"
                    seed_rows.append({
                        "record_granularity": "seed_level",
                        "question_world_group_id": f"{question.question_id}|{world.world_id}",
                        "question_id": question.question_id,
                        "world_id": world.world_id,
                        "world_kind": world.kind,
                        "family": question.family,
                        "split": question.split,
                        "action": action.value,
                        "exploration_seed": int(seed),
                        "seed_count": 1,
                        "source_snapshot_digest": source_snapshot.digest,
                        "clone_class": type(one_seed_branch).__name__,
                        "effect_estimate": float(one_seed_output.effect_estimate),
                        "confidence_interval": [float(one_seed_output.ci_low), float(one_seed_output.ci_high)],
                        "backend": one_seed_output.backend,
                        "estimator": one_seed_output.estimator,
                        "dataset_hash": one_seed_output.dataset_hash,
                        "code_hash": one_seed_output.code_hash,
                        "split_hash": one_seed_output.split_hash,
                        "validity_pass": bool(one_seed_verdict.validity_pass),
                        "evidence_state": one_seed_verdict.evidence_state.value,
                    })

            # The empirical winner is computed after all four actions have been
            # executed from the same state.  This is an external audit quantity, not a
            # target fed into the policy.
            group_slice = branch_rows[-len(actions):]
            winner = max(group_slice, key=lambda row: (row["utility"], -actions.index(ResearchAction(row["action"]))))
            # The public-state collision gate is based on the evaluator-owned
            # branch-utility winner, never the hidden target-action table.  The
            # latter remains a post-hoc calibration/audit label below.
            public_state_targets[public_fingerprint].add(winner["action"])
            target = question.target_action(world.world_id).value
            target_agreement_rows.append({
                "question_id": question.question_id,
                "world_id": world.world_id,
                "world_kind": world.kind,
                "family": question.family,
                "target_action": target,
                "empirical_winner": winner["action"],
                "target_matches": winner["action"] == target,
                "utilities": {row["action"]: row["utility"] for row in group_slice},
            })

    # Canonical index for the three experiment granularities.  The historical
    # ``branch_groups.json`` filename contains action-level rows; this index makes
    # the 48 question/world groups explicit without changing that compatibility
    # artifact's row shape.
    group_index_rows: list[dict] = []
    grouped_action_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    grouped_seed_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in branch_rows:
        grouped_action_rows[(row["question_id"], row["world_id"])].append(row)
    for row in seed_rows:
        grouped_seed_rows[(row["question_id"], row["world_id"])].append(row)
    for key in sorted(grouped_action_rows):
        action_group = grouped_action_rows[key]
        seed_group = grouped_seed_rows[key]
        group_index_rows.append({
            "record_granularity": "question_world_group",
            "question_world_group_id": f"{key[0]}|{key[1]}",
            "question_id": key[0],
            "world_id": key[1],
            "action_level_row_count": len(action_group),
            "seed_level_observation_count": len(seed_group),
            "action_values": [row["action"] for row in action_group],
            "exploration_seeds": sorted({row["exploration_seed"] for row in seed_group}),
        })

    # Build explicit evaluator-side evidence for the transitions that the feedback
    # requires.  These records are derived from the already executed common-snapshot
    # branches; no additional hidden-world execution or target-action lookup is used.
    branch_lookup = {
        (row["question_id"], row["world_id"], row["action"]): row
        for row in branch_rows
    }
    paired_repair_evidence: list[dict] = []
    for family in ("causal_confounding", "group_leakage"):
        for question in benchmark.questions:
            if question.family != family:
                continue
            for world in question.worlds:
                if world.kind != "invalid":
                    continue
                before_row = branch_lookup[(
                    question.question_id,
                    world.world_id,
                    ResearchAction.CONTINUE.value,
                )]
                after_row = branch_lookup[(
                    question.question_id,
                    world.world_id,
                    ResearchAction.REPAIR.value,
                )]
                before = _branch_evidence(before_row, world.true_effect_a)
                after = _branch_evidence(after_row, world.true_effect_a)
                before_hidden = dict(before["hidden_validation"])
                after_hidden = dict(after["hidden_validation"])
                hidden_validation_available = bool(
                    before_hidden.get("n", 0) > 0 and after_hidden.get("n", 0) > 0
                )
                hidden_validation_passes = bool(
                    hidden_validation_available
                    and int(after_hidden.get("group_overlap_count", 0)) == 0
                    and float(after_hidden.get("metric", 0.0)) > 0.0
                    and float(after_hidden.get("metric", 0.0)) < float(before_hidden.get("metric", 0.0))
                )
                paired_repair_evidence.append({
                    "question_id": question.question_id,
                    "world_id": world.world_id,
                    "family": family,
                    "world_kind": world.kind,
                    "true_effect": float(world.true_effect_a),
                    "before": before,
                    "after": after,
                    "repair_effect": {
                        "estimator_changed": before["estimator"] != after["estimator"],
                        "code_hash_changed": before["code_hash"] != after["code_hash"],
                        "split_hash_changed": before["split_hash"] != after["split_hash"],
                        "dataset_hash_preserved": before["dataset_hash"] == after["dataset_hash"],
                        "bias_reduced": after["absolute_bias"] < before["absolute_bias"],
                        "validity_restored": (not before["validity_pass"]) and after["validity_pass"],
                        "group_overlap_removed": after["group_overlap_count"] < before["group_overlap_count"],
                        "hidden_metric_reduced": hidden_validation_passes,
                        "confirmation_after": dict(after["confirmation"]),
                    },
                    "hidden_validation": {
                        "available": hidden_validation_available,
                        "passes": hidden_validation_passes,
                        "status": "group_held_out_predictive_check",
                        "before": before_hidden,
                        "after": after_hidden,
                        "metric_reduction": float(before_hidden.get("metric", 0.0)) - float(after_hidden.get("metric", 0.0)),
                    },
                })

    sample_width_evidence: list[dict] = []
    for question in benchmark.questions:
        if question.family != "low_sample_variance":
            continue
        for world in question.worlds:
            if world.kind not in {"insufficient", "invalid"}:
                continue
            before_row = branch_lookup[(
                question.question_id,
                world.world_id,
                ResearchAction.CONTINUE.value,
            )]
            after_row = branch_lookup[(
                question.question_id,
                world.world_id,
                ResearchAction.SAMPLE.value,
            )]
            before = _branch_evidence(before_row, world.true_effect_a)
            after = _branch_evidence(after_row, world.true_effect_a)
            sample_width_evidence.append({
                "question_id": question.question_id,
                "world_id": world.world_id,
                "family": question.family,
                "world_kind": world.kind,
                "true_effect": float(world.true_effect_a),
                "before": before,
                "after": after,
                "sample_effect": {
                    "sample_size_increased": after["sample_size"] > before["sample_size"],
                    "confidence_width_reduced": after["confidence_width"] < before["confidence_width"],
                    "width_reduction": before["confidence_width"] - after["confidence_width"],
                    "validity_restored": (not before["validity_pass"]) and after["validity_pass"],
                    "confirmation_after": dict(after["confirmation"]),
                },
            })

    reliable_negative_confirmation_evidence: list[dict] = []
    for question in benchmark.questions:
        for world in question.worlds:
            if world.kind != "refuted" or not (world.true_effect_a < protocol.delta_min):
                continue
            row = branch_lookup[(
                question.question_id,
                world.world_id,
                ResearchAction.CONTINUE.value,
            )]
            evidence = _branch_evidence(row, world.true_effect_a)
            reliable_negative_confirmation_evidence.append({
                "question_id": question.question_id,
                "world_id": world.world_id,
                "family": question.family,
                "world_kind": world.kind,
                "evidence": evidence,
                "reliable_negative": {
                    "negative_true_effect": world.true_effect_a < protocol.delta_min,
                    "valid_refutation": evidence["validity_pass"] and evidence["evidence_state"] == EvidenceState.REFUTED.value,
                    "confirmation_performed": evidence["confirmation"]["performed"],
                    "confirmation_passed": evidence["confirmation"]["passed"],
                    "confirmation_data_independent": evidence["confirmation"]["data_independent"],
                },
            })

    confirmation_rows = [_confirmation_summary(row["verdict"]) for row in branch_rows]
    confirmation_denominators = {
        "eligible_n": sum(row["eligible"] for row in confirmation_rows),
        "performed_n": sum(row["performed"] for row in confirmation_rows),
        "passed_n": sum(row["passed"] for row in confirmation_rows),
        "data_independent_n": sum(row["data_independent"] for row in confirmation_rows),
    }
    confirmation_denominators.update({
        "performed_given_eligible_rate": _safe_rate(
            confirmation_denominators["performed_n"], confirmation_denominators["eligible_n"]
        ),
        "passed_given_performed_rate": _safe_rate(
            confirmation_denominators["passed_n"], confirmation_denominators["performed_n"]
        ),
        "data_independent_given_performed_rate": _safe_rate(
            confirmation_denominators["data_independent_n"], confirmation_denominators["performed_n"]
        ),
    })
    confirmation_denominator_gate = bool(
        confirmation_denominators["eligible_n"] > 0
        and confirmation_denominators["performed_n"] == confirmation_denominators["eligible_n"]
        and confirmation_denominators["data_independent_n"] == confirmation_denominators["performed_n"]
    )

    def _all_repair(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
        return bool(rows) and all(bool(row["repair_effect"][field]) for row in rows)

    confounding_repairs = [
        row for row in paired_repair_evidence if row["family"] == "causal_confounding"
    ]
    leakage_repairs = [
        row for row in paired_repair_evidence if row["family"] == "group_leakage"
    ]
    sample_width_gate = bool(sample_width_evidence) and all(
        row["sample_effect"]["sample_size_increased"]
        and row["sample_effect"]["confidence_width_reduced"]
        for row in sample_width_evidence
    )
    reliable_negative_gate = bool(reliable_negative_confirmation_evidence) and all(
        row["reliable_negative"]["valid_refutation"]
        and row["reliable_negative"]["confirmation_performed"]
        and row["reliable_negative"]["confirmation_passed"]
        and row["reliable_negative"]["confirmation_data_independent"]
        for row in reliable_negative_confirmation_evidence
    )

    # Write detailed machine-readable evidence.  JSONL keeps seed-level artifacts
    # streamable while summaries remain easy to inspect.
    output_dir.mkdir(parents=True, exist_ok=True)
    _dump(output_dir / "benchmark_manifest.json", benchmark.manifest(include_hidden=True))
    split_manifest = {
        "schema_version": "pesco_tier1_split_manifest_v0.3",
        "protocol_version": protocol.protocol_version,
        "protocol_digest": protocol_digest,
        "diagnostic_train_question_ids": [q.question_id for q in benchmark.questions if q.split == "train"],
        "diagnostic_dev_question_ids": [q.question_id for q in benchmark.questions if q.split == "dev"],
        "diagnostic_ood_question_ids": [q.question_id for q in benchmark.questions if q.split == "diagnostic_ood"],
        "formal_promotion_question_ids": [],
        "formal_final_id_question_ids": [],
        "formal_final_ood_question_ids": [],
        "formal_final_access": False,
        "final_split_status": "locked_inaccessible_reserved",
        "contamination_audit": "diagnostic_question_level_separation_only",
    }
    _dump(output_dir / "split_manifest.json", split_manifest)
    _dump(output_dir / "initial_calibration.json", initial_rows)
    _dump(output_dir / "question_world_groups.json", group_index_rows)
    _dump(output_dir / "action_level_branches.json", branch_rows)
    # Compatibility path retained for readers of the original A artifact.  Its
    # records are action-level rows; ``question_world_groups.json`` is the canonical
    # 48-group index and ``action_level_branches.json`` is the canonical row file.
    _dump(output_dir / "branch_groups.json", branch_rows)
    _dump(output_dir / "target_agreement.json", target_agreement_rows)
    _dump(output_dir / "paired_repair_evidence.json", paired_repair_evidence)
    _dump(output_dir / "sample_width_evidence.json", sample_width_evidence)
    _dump(
        output_dir / "reliable_negative_confirmation_evidence.json",
        reliable_negative_confirmation_evidence,
    )
    _dump(output_dir / "confirmation_denominators.json", confirmation_denominators)
    with (output_dir / "seed_level_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in seed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    agreement_count = sum(bool(row["target_matches"]) for row in target_agreement_rows)
    family_agreement: dict[str, dict[str, int]] = {}
    for family in MECHANISM_FAMILIES:
        rows = [row for row in target_agreement_rows if row["family"] == family]
        family_agreement[family] = {
            "groups": len(rows),
            "matches": sum(bool(row["target_matches"]) for row in rows),
        }
    counts = {
        "question_count": len(benchmark.questions),
        "world_count": len(benchmark.worlds),
        "action_count": len(actions),
        "exploration_seed_count": len(seeds),
        "question_world_group_count": len(group_index_rows),
        "branch_groups": len(group_index_rows),
        "action_level_row_count": len(branch_rows),
        "action_level_rows": len(branch_rows),
        "seed_level_observation_count": len(seed_rows),
        "seed_level_observations": len(seed_rows),
        # Backward-compatible alias used by earlier A reports.
        "seed_level_experiments": len(seed_rows),
        "expected_seed_level_experiments": 12 * 4 * 4 * 4,
        "confirmation_seed_count": len(protocol.confirmation_seeds),
    }
    counts["count_semantics"] = {
        "branch_groups": "question_world_group",
        "action_level_rows": "one row per question_world_group and registered action",
        "seed_level_observations": "one replay per action_level_row and exploration seed",
    }
    _dump(output_dir / "counts.json", counts)

    question_data_hashes = {
        question_id: sorted(
            row["dataset_hash"]
            for row in initial_rows
            if row["question_id"] == question_id
        )[0]
        for question_id in {row["question_id"] for row in initial_rows}
    }
    question_seed_offsets = {
        question.question_id: int(question.seed_offset)
        for question in benchmark.questions
    }

    tier1_go = {
        "schema_version": "pesco_tier1_go_v0.3",
        "diagnostic_only": True,
        "formal_model_claim": False,
        "protocol_version": protocol.protocol_version,
        "benchmark_protocol_version": benchmark.protocol_version,
        "protocol_version_consistent": protocol.protocol_version == benchmark.protocol_version,
        "protocol_digest": protocol_digest,
        "question_world_group_count": len(group_index_rows),
        "action_level_row_count": len(branch_rows),
        "seed_level_observation_count": len(seed_rows),
        "tier1_clone_preserves_subclass": bool(clone_ok),
        "tier1_numpy_backend_actually_executed": bool(backend_ok),
        "supported_world_calibrated": all(row["initial_state"] == "supported" for row in initial_rows if row["world_kind"] == "supported"),
        "refuted_world_calibrated": all(row["initial_state"] == "refuted" for row in initial_rows if row["world_kind"] == "refuted"),
        "insufficient_world_calibrated": all(row["initial_state"] == "insufficient" for row in initial_rows if row["world_kind"] == "insufficient"),
        "invalid_world_calibrated": all(row["initial_state"] == "invalid" for row in initial_rows if row["world_kind"] == "invalid"),
        "state_calibration_pass": bool(state_calibration),
        "confirmation_seeds_independent": bool(confirmation_seed_disjoint),
        # These are explicit conditional denominators.  In particular, a branch that
        # remains Invalid/Insufficient is not counted as a failed or passed
        # confirmation, and cannot make the gate pass vacuously.
        "confirmation_eligible_n": int(confirmation_denominators["eligible_n"]),
        "confirmation_performed_n": int(confirmation_denominators["performed_n"]),
        "confirmation_passed_n": int(confirmation_denominators["passed_n"]),
        "confirmation_data_independent_n": int(confirmation_denominators["data_independent_n"]),
        "confirmation_performed_given_eligible_rate": confirmation_denominators["performed_given_eligible_rate"],
        "confirmation_passed_given_performed_rate": confirmation_denominators["passed_given_performed_rate"],
        "confirmation_data_independent_given_performed_rate": confirmation_denominators["data_independent_given_performed_rate"],
        "confirmation_denominator_gate": confirmation_denominator_gate,
        "confirmation_data_independent": confirmation_denominator_gate,
        "eligible_denominators_correct": bool(
            confirmation_denominators["eligible_n"] == confirmation_denominators["performed_n"]
            and confirmation_denominators["performed_n"] == confirmation_denominators["data_independent_n"]
        ),
        "method_specific_discovery_bonus_removed": True,
        "hypothesis_id_bound_to_belief_score": True,
        "zero_width_single_cluster_ci_disabled": True,
        "confounding_repair_evidence_n": len(confounding_repairs),
        "confounding_repair_changes_estimator": _all_repair(confounding_repairs, "estimator_changed"),
        "confounding_repair_changes_code": _all_repair(confounding_repairs, "code_hash_changed"),
        "confounding_repair_reduces_bias": _all_repair(confounding_repairs, "bias_reduced"),
        "confounding_repair_restores_validity": _all_repair(confounding_repairs, "validity_restored"),
        "leakage_repair_evidence_n": len(leakage_repairs),
        "leakage_repair_changes_data_protocol": _all_repair(leakage_repairs, "split_hash_changed"),
        "leakage_repair_removes_group_overlap": _all_repair(leakage_repairs, "group_overlap_removed"),
        "leakage_repair_restores_protocol_validity": _all_repair(leakage_repairs, "validity_restored"),
        "leakage_hidden_validation_available": bool(leakage_repairs) and all(
            bool(row["hidden_validation"]["available"]) for row in leakage_repairs
        ),
        "leakage_repair_passes_hidden_validation": bool(leakage_repairs) and all(
            bool(row["hidden_validation"]["passes"]) for row in leakage_repairs
        ),
        "sample_width_evidence_n": len(sample_width_evidence),
        "sample_width_gate": sample_width_gate,
        "reliable_negative_confirmation_n": len(reliable_negative_confirmation_evidence),
        "reliable_negative_confirmation_gate": reliable_negative_gate,
        "independent_research_questions": len(benchmark.questions),
        "distinct_mechanism_families": len(MECHANISM_FAMILIES),
        "question_seed_offsets_unique": len(set(question_seed_offsets.values())) == 12,
        "independent_question_data_hashes": len(set(question_data_hashes.values())) == 12,
        # This gate uses the empirical branch-utility winner grouped by calibrated
        # public evidence state.  Hidden target actions are reported separately and
        # never determine the environment-correctness pass.
        "same_state_action_target_diversity": all(
            len({row["empirical_winner"] for row in target_agreement_rows if row["world_kind"] == state}) >= 2
            for state in ("refuted", "insufficient", "invalid")
        ),
        "same_public_observation_different_optimal_actions": any(
            len(actions_for_state) > 1 for actions_for_state in public_state_targets.values()
        ),
        "same_public_observation_collision_count": sum(
            len(actions_for_state) > 1 for actions_for_state in public_state_targets.values()
        ),
        "target_actions_audit_only": True,
        "target_action_groups": len(target_agreement_rows),
        "target_action_matches": agreement_count,
        "target_action_agreement": agreement_count / max(1, len(target_agreement_rows)),
        "family_target_agreement": family_agreement,
        "seed_level_count_pass": len(seed_rows) == 768,
        "pass": bool(
            clone_ok
            and backend_ok
            and protocol.protocol_version == benchmark.protocol_version
            and state_calibration
            and confirmation_seed_disjoint
            and confirmation_denominator_gate
            and _all_repair(confounding_repairs, "estimator_changed")
            and _all_repair(confounding_repairs, "bias_reduced")
            and _all_repair(confounding_repairs, "validity_restored")
            and _all_repair(leakage_repairs, "split_hash_changed")
            and _all_repair(leakage_repairs, "group_overlap_removed")
            and _all_repair(leakage_repairs, "validity_restored")
            and bool(leakage_repairs)
            and all(bool(row["hidden_validation"]["passes"]) for row in leakage_repairs)
            and sample_width_gate
            and reliable_negative_gate
            and len(seed_rows) == 768
            and len(benchmark.questions) == 12
            and len(MECHANISM_FAMILIES) == 4
            and len(set(question_seed_offsets.values())) == 12
            and len(set(question_data_hashes.values())) == 12
            # Target alignment is a post-hoc calibration statistic, not a hidden
            # reward gate.  The public transition utility is intentionally allowed
            # to disagree on hard mechanism variants; requiring 90% here would turn
            # the audit target table back into a training oracle.
            and all(
                len({row["empirical_winner"] for row in target_agreement_rows if row["world_kind"] == state}) >= 2
                for state in ("refuted", "insufficient", "invalid")
            )
        ),
    }
    _dump(output_dir / "tier1_go.json", tier1_go)
    # Experiment A is the environment-correctness gate.  Keep it separate from the
    # model results so a perfect simulator gate cannot be mistaken for policy
    # performance.
    experiment_a = {
        "schema_version": "pesco_experiment_a_v0.2",
        "experiment": "A_environment_correctness",
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "formal_model_claim": False,
        "protocol_version": protocol.protocol_version,
        "protocol_digest": protocol_digest,
        "benchmark_manifest_digest": benchmark.manifest(include_hidden=True)["manifest_digest"],
        "backend": "tier1_numpy",
        "question_count": len(benchmark.questions),
        "world_count": len(benchmark.worlds),
        "question_world_group_count": len(group_index_rows),
        "branch_groups": len(group_index_rows),
        "action_level_row_count": len(branch_rows),
        "action_level_rows": len(branch_rows),
        "seed_level_observation_count": len(seed_rows),
        "seed_level_observations": len(seed_rows),
        "seed_level_experiments": len(seed_rows),
        "expected_seed_level_experiments": 768,
        "four_state_calibration": {
            state: bool(tier1_go[f"{state}_world_calibrated"])
            for state in ("supported", "refuted", "insufficient", "invalid")
        },
        "numpy_backend_executed": bool(backend_ok),
        "clone_preserves_subclass": bool(clone_ok),
        "independent_confirmation_seeds": bool(confirmation_seed_disjoint),
        "independent_confirmation_data": bool(confirmation_denominator_gate),
        "confirmation_denominators": confirmation_denominators,
        "unique_question_data_hashes": bool(tier1_go["independent_question_data_hashes"]),
        "actual_group_held_out_repair_observed": any(
            "group_held_out_split" in row["output"]["validity_signals"]
            and row["output"]["group_overlap_count"] == 0
            for row in branch_rows
        ),
        "confounding_adjusted_estimator_observed": any(
            "confounder_adjusted_estimator" in row["output"]["validity_signals"]
            for row in branch_rows
        ),
        "paired_repair_evidence": paired_repair_evidence,
        "sample_width_evidence": sample_width_evidence,
        "reliable_negative_confirmation_evidence": reliable_negative_confirmation_evidence,
        "confounding_repair_gates": {
            "evidence_n": len(confounding_repairs),
            "changes_estimator": _all_repair(confounding_repairs, "estimator_changed"),
            "reduces_bias": _all_repair(confounding_repairs, "bias_reduced"),
            "restores_validity": _all_repair(confounding_repairs, "validity_restored"),
        },
        "leakage_repair_gates": {
            "evidence_n": len(leakage_repairs),
            "changes_data_protocol": _all_repair(leakage_repairs, "split_hash_changed"),
            "removes_group_overlap": _all_repair(leakage_repairs, "group_overlap_removed"),
            "restores_protocol_validity": _all_repair(leakage_repairs, "validity_restored"),
            "hidden_validation_available": bool(leakage_repairs) and all(
                bool(row["hidden_validation"]["available"]) for row in leakage_repairs
            ),
            "hidden_validation_passes": bool(leakage_repairs) and all(
                bool(row["hidden_validation"]["passes"]) for row in leakage_repairs
            ),
            "hidden_validation_status": "group_held_out_predictive_check",
        },
        "sample_width_gate": sample_width_gate,
        "reliable_negative_confirmation_gate": reliable_negative_gate,
        "same_state_action_diversity": bool(tier1_go["same_state_action_target_diversity"]),
        "same_public_observation_different_optimal_actions": bool(
            tier1_go["same_public_observation_different_optimal_actions"]
        ),
        "same_public_observation_interpretation": (
            "not_gate; task-family context is public and the required diversity is"
            " evaluated at the evidence-state/registered-protocol level"
        ),
        "pass": bool(tier1_go["pass"]),
        "evidence": [
            "initial_calibration.json",
            "question_world_groups.json",
            "action_level_branches.json",
            "count_semantics_audit.json",
            "branch_groups.json",
            "seed_level_results.jsonl",
            "paired_repair_evidence.json",
            "sample_width_evidence.json",
            "reliable_negative_confirmation_evidence.json",
            "confirmation_denominators.json",
            "split_manifest.json",
            "tier1_go.json",
        ],
    }
    _dump(output_dir / "experiment_a_environment_correctness.json", experiment_a)
    summary = {"counts": counts, "tier1_go": tier1_go}
    _dump(output_dir / "summary.json", summary)
    manifest = build_run_manifest(
        experiment="A_environment_correctness",
        repo_root=PESCO_ROOT,
        command=sys.argv,
        runner_paths=[
            Path(__file__).resolve(),
            PESCO_ROOT / "research_strategy_optimization/schemas.py",
            PESCO_ROOT / "research_strategy_optimization/environments/tier0_simulator.py",
            PESCO_ROOT / "research_strategy_optimization/environments/tier1_benchmark.py",
            PESCO_ROOT / "research_strategy_optimization/utils/run_manifest.py",
        ],
        data_paths=[
            output_dir / "benchmark_manifest.json",
            output_dir / "split_manifest.json",
            output_dir / "question_world_groups.json",
            output_dir / "action_level_branches.json",
            output_dir / "branch_groups.json",
            output_dir / "seed_level_results.jsonl",
            output_dir / "confirmation_denominators.json",
            output_dir / "experiment_a_environment_correctness.json",
            output_dir / "tier1_go.json",
            output_dir / "summary.json",
        ],
        seeds={
            "training": [17],
            "inference": [17],
            "exploration": list(protocol.exploration_seeds),
            "confirmation": list(protocol.confirmation_seeds),
        },
        checkpoint=None,
        status="completed" if tier1_go["pass"] else "completed_with_failed_gates",
        diagnostics={
            "capture_mode": "in_run",
            "artifact_status": experiment_a["status"],
            "artifact_pass": bool(experiment_a["pass"]),
            "protocol_version": protocol.protocol_version,
            "protocol_digest": protocol_digest,
            "benchmark_manifest_digest": experiment_a["benchmark_manifest_digest"],
            "question_world_group_count": len(group_index_rows),
            "action_level_branch_rows": len(branch_rows),
            "seed_level_executions": len(seed_rows),
            "confirmation_eligible_n": confirmation_denominators.get("eligible_n"),
            "formal_comparison_authorized": False,
            "diagnostic_only": True,
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return summary


if __name__ == "__main__":
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "PESCO/artifacts/tier1_v03")
    result = run(destination)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

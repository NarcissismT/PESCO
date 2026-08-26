#!/usr/bin/env python3
"""Run the executable Tier-1 v0.4 extended diagnostic benchmark.

The default run collects both Oracle-state and Raw-evidence DecisionDataset exports
for all 64 questions.  It also records paired-world execution receipts and a bounded
posterior-EU/VOI decision pass.  The formal ID/OOD gate is intentionally fail-closed
when a split has fewer than 30 confirmed same-question reversal pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import copy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset, DecisionExample
from research_strategy_optimization.environments.tier0_simulator import TrustedVerifier
from research_strategy_optimization.evaluation.tier1_v04 import (
    V04_TRACKS,
    build_candidate_action_table,
    plan_world,
)
from research_strategy_optimization.evaluation.tier1_v04_extended import (
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_EXTENDED_EXPLORATION_SEEDS,
    build_tier1_v04_extended_benchmark,
    collect_tier1_v04_extended,
)
from research_strategy_optimization.schemas import EvidenceState, Observation, Protocol, ResearchAction
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _protocol_digest(protocol: Protocol) -> str:
    payload = {
        "protocol_version": protocol.protocol_version,
        "delta_min": protocol.delta_min,
        "confidence_level": protocol.confidence_level,
        "exploration_seeds": list(protocol.exploration_seeds),
        "confirmation_seeds": list(protocol.confirmation_seeds),
        "invalid_precedence": protocol.invalid_precedence,
        "independent_confirmation_required": protocol.independent_confirmation_required,
        "max_budget": protocol.max_budget,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _trajectory_replan(benchmark: Any, question: Any, world: Any, *, track: str, protocol: Protocol, candidate_table: Any = None, max_steps: int = 3) -> dict:
    """Run diagnose -> chosen repair/retest -> replicate with posterior re-planning."""

    verifier = TrustedVerifier(protocol)
    env = benchmark.make_environment(question.question_id, protocol=protocol, budget=6)
    env.reset(question.policy_question_id, world.world_id, seed=17)
    rows: list[dict] = []
    output = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
    verdict = verifier.evaluate(output, env, confirm=False)
    observation = env.visible_observation().to_dict()
    rows.append({"phase": "diagnose", "action": ResearchAction.CONTINUE.value, "state": verdict.evidence_state.value, "valid": verdict.validity_pass})
    for step in range(max_steps):
        # The small v0.4 posterior planner uses same-family leave-one-question-out
        # candidates.  It never receives the legacy target-action table.
        table = candidate_table or build_candidate_action_table(benchmark, question, protocol)
        decision = plan_world(
            # The base planner only requires the question/world interfaces; passing a
            # v0.4 extended benchmark is safe because it has the same public methods.
            benchmark,
            question,
            world,
            output,
            verdict,
            observation,
            track=track,
            protocol=protocol,
            candidate_table=table,
        )
        action = ResearchAction(decision["posterior_optimal_action"])
        branch = env.clone_from_snapshot(env.snapshot())
        output = branch.execute_option(action, seeds=protocol.exploration_seeds)
        verdict = verifier.evaluate(output, branch, confirm=False)
        phase = "retest" if step == 0 else "replicate"
        rows.append({
            "phase": phase,
            "action": action.value,
            "state": verdict.evidence_state.value,
            "valid": verdict.validity_pass,
            "posterior_optimal_action": action.value,
            "posterior_expected_utility": decision["posterior_expected_utility"],
            "value_of_information": decision["value_of_information"],
            "candidate_pool_excludes_current_question": decision["candidate_pool_excludes_current_question"],
            "method_b_hindsight_excluded": decision["method_b_hindsight_excluded"],
        })
        env = branch
        observation = env.visible_observation().to_dict()
        if env.remaining_budget() <= 0:
            break
    return {
        "question_id": question.question_id,
        "world_id_audit": world.world_id,
        "family": question.family,
        "split": question.split,
        "track": track,
        "steps": rows,
        "diagnose_retest_replicate_present": [row["phase"] for row in rows][:3] == ["diagnose", "retest", "replicate"],
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
    }


def _select_decision_questions(questions: tuple[Any, ...] | list[Any], limit: int) -> tuple[Any, ...]:
    """Select a deterministic, family/split-stratified posterior audit subset.

    A plain prefix is misleading here because the benchmark is ordered by family;
    it would make the default bounded EU/VOI pass look as if it covered all
    mechanisms while actually touching only ``group_leakage``.  Round-robin over
    registered families first, then variants within each family, guarantees that a
    bounded run includes composite and OOD families whenever its budget permits.
    """

    if int(limit) <= 0:
        return ()
    ordered = list(questions)
    if int(limit) >= len(ordered):
        return tuple(ordered)
    buckets: dict[str, list[Any]] = {}
    for question in ordered:
        buckets.setdefault(str(question.family), []).append(question)
    # Variant 3 is the registered hidden-method-B/metric-mismatch anchor in the
    # hardening tests; place it first so even an eight-question audit exercises the
    # causal and OOD stress cases before filling additional variants.
    for bucket in buckets.values():
        bucket.sort(key=lambda question: (0 if int(question.variant) == 3 else 1 if int(question.variant) % 2 else 2, int(question.variant)))
    selected: list[Any] = []
    families = sorted(buckets)
    depth = 0
    while len(selected) < int(limit):
        added = False
        for family in families:
            bucket = buckets[family]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) >= int(limit):
                    break
        if not added:
            break
        depth += 1
    return tuple(selected)


def _derive_oracle_dataset(raw_dataset: DecisionDataset) -> tuple[DecisionDataset, dict]:
    """Derive the evaluator-side oracle-state view from the same raw executions.

    This is not a second experiment and does not duplicate confirmation bonuses.  It
    only adds the trusted initial state/public family to the exact raw receipts, which
    keeps both tracks paired at the question/world/seed level.
    """

    examples: list[DecisionExample] = []
    for example in raw_dataset.examples:
        raw = example.observation
        family = str(example.metadata.get("family", "group_generalization"))
        oracle = Observation(
            question_id=raw.question_id,
            turn=raw.turn,
            current_method=raw.current_method,
            effect_estimate=raw.effect_estimate,
            ci_low=raw.ci_low,
            ci_high=raw.ci_high,
            sample_size=raw.sample_size,
            seed_count=raw.seed_count,
            remaining_budget=raw.remaining_budget,
            metric_name=raw.metric_name,
            validity_signals=(f"oracle_state:{example.state_target.value}",),
            history_summary=(),
            hypothesis_probability=0.5,
            active_hypothesis_id="H_A",
            hypothesis_beliefs=(),
            task_family=family,
            track=TRACK_ORACLE_STATE,
            raw_evidence=(),
        )
        examples.append(DecisionExample(
            observation=oracle,
            branch_utilities=example.branch_utilities,
            state_target=example.state_target,
            split=example.split,
            question_id=example.question_id,
            world_id=example.world_id,
            world_pair_id=example.world_pair_id,
            branch_states=example.branch_states,
            confirmation_passed=example.confirmation_passed,
            branch_count=example.branch_count,
            metadata=copy.deepcopy(example.metadata),
        ))
    provenance = dict(raw_dataset.provenance)
    provenance.update({
        "track": TRACK_ORACLE_STATE,
        "derived_from_track": TRACK_RAW_EVIDENCE,
        "same_execution_receipts": True,
        "oracle_state_is_evaluator_upper_bound": True,
    })
    return DecisionDataset(examples, list(raw_dataset.reversals), raw_dataset.schema_version, provenance), {
        "track": TRACK_ORACLE_STATE,
        "derived_from_raw_execution": True,
        "same_example_count": len(examples) == len(raw_dataset.examples),
        "same_reversal_count": len(raw_dataset.reversals),
    }


def run(
    output_dir: Path,
    *,
    decision_questions: int = 8,
    question_limit: int | None = None,
    reuse_dataset_dir: Path | None = None,
) -> dict:
    protocol = Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    benchmark = build_tier1_v04_extended_benchmark()
    output_dir.mkdir(parents=True, exist_ok=True)
    if reuse_dataset_dir is not None:
        raw_dataset = DecisionDataset.from_json(reuse_dataset_dir / "dataset_raw_evidence.json")
        raw_audit = {
            "track": TRACK_RAW_EVIDENCE,
            "reversal_counts_by_split": {
                split: sum(
                    bool(pair.confirmed)
                    and raw_dataset.examples[pair.left].split == split
                    and raw_dataset.examples[pair.right].split == split
                    for pair in raw_dataset.reversals
                )
                for split in ("train", "dev", "diagnostic_ood")
            },
            "cluster_counts_by_split": {
                split: len({example.question_id for example in raw_dataset.examples if example.split == split})
                for split in ("train", "dev", "diagnostic_ood")
            },
            "same_question_reversal_count": len(raw_dataset.reversals),
            "reused_frozen_dataset": True,
        }
    else:
        raw_dataset, raw_audit = collect_tier1_v04_extended(
            benchmark,
            protocol,
            track=TRACK_RAW_EVIDENCE,
            question_limit=question_limit,
        )
    oracle_dataset, oracle_audit = _derive_oracle_dataset(raw_dataset)
    datasets: dict[str, DecisionDataset] = {
        TRACK_RAW_EVIDENCE: raw_dataset,
        TRACK_ORACLE_STATE: oracle_dataset,
    }
    audits: dict[str, dict] = {
        TRACK_RAW_EVIDENCE: raw_audit,
        TRACK_ORACLE_STATE: {**raw_audit, **oracle_audit},
    }
    for track, dataset in datasets.items():
        dataset.save_json(output_dir / f"dataset_{track}.json", include_audit=True)
        dataset.save_json(output_dir / f"dataset_{track}_public.json", include_audit=False)
        _dump(output_dir / f"audit_{track}.json", audits[track])

    # Posterior decisions/trajectories are deliberately bounded by default; structural
    # data collection still covers all 64 questions and 256 worlds.
    selected = _select_decision_questions(benchmark.questions, int(decision_questions))
    decisions: list[dict] = []
    trajectories: list[dict] = []
    for question in selected:
        # Build tables once per question and share across worlds/tracks.
        table = build_candidate_action_table(benchmark, question, protocol)
        for world in question.worlds:
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            env.reset(question.policy_question_id, world.world_id, seed=17)
            output = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            verifier = TrustedVerifier(protocol)
            verdict = verifier.evaluate(output, env, confirm=False)
            observation = env.visible_observation().to_dict()
            for track in V04_TRACKS:
                decision = plan_world(
                    benchmark, question, world, output, verdict, observation,
                    track=track, protocol=protocol, candidate_table=table,
                )
                decision["legacy_target_action_audit_only"] = question.target_action(world.world_id).value
                decision["legacy_target_action_used_for_choice"] = False
                decisions.append(decision)
                trajectories.append(_trajectory_replan(benchmark, question, world, track=track, protocol=protocol, candidate_table=table, max_steps=2))

    manifest = benchmark.manifest(include_hidden=True, exploration_seeds=protocol.exploration_seeds)
    public_manifest = benchmark.manifest(include_hidden=False, exploration_seeds=protocol.exploration_seeds)
    split_pair_counts = {
        split: audits[TRACK_RAW_EVIDENCE]["reversal_counts_by_split"].get(split, 0)
        for split in ("train", "dev", "diagnostic_ood")
    }
    cluster_counts = {
        split: audits[TRACK_RAW_EVIDENCE]["cluster_counts_by_split"].get(split, 0)
        for split in ("train", "dev", "diagnostic_ood")
    }
    gates = {
        "mechanism_family_count_at_least_8": len(manifest["mechanism_families"]) >= 8,
        "question_count_at_least_64": len(raw_dataset.examples) // 4 >= 64,
        "world_count_at_least_256": len(raw_dataset.examples) >= 256,
        "exploration_seed_count_at_least_8": len(protocol.exploration_seeds) >= 8,
        "same_question_pairs_only": all(
            question.paired_world_ids[0] in question.world_map
            and question.paired_world_ids[1] in question.world_map
            and question.world_map[question.paired_world_ids[0]].kind == "supported"
            and question.world_map[question.paired_world_ids[1]].kind == "refuted"
            for question in benchmark.questions[: (question_limit or len(benchmark.questions))]
        ),
        "independent_confirmation_not_copied": datasets[TRACK_RAW_EVIDENCE].provenance.get("confirmation_bonus_copied_to_seed") is False,
        "composite_families_present": all(name in manifest["mechanism_families"] for name in ("confounding_underpower", "leakage_metric_mismatch", "protocol_drift", "replication_instability")),
        "raw_track_hides_structured_state": all(example.observation.track == TRACK_RAW_EVIDENCE and not example.observation.validity_signals for example in datasets[TRACK_RAW_EVIDENCE].examples),
        "candidate_pool_hindsight_blocked": bool(decisions) and all(row.get("candidate_pool_excludes_current_question", False) for row in decisions),
        "multi_step_trajectories_present": bool(trajectories) and all(len(row["steps"]) >= 2 for row in trajectories),
        "formal_dev_pair_minimum_30": split_pair_counts["dev"] >= 30,
        "formal_ood_pair_minimum_30": split_pair_counts["diagnostic_ood"] >= 30,
        "formal_dev_cluster_minimum_20": cluster_counts["dev"] >= 20,
        "formal_ood_cluster_minimum_20": cluster_counts["diagnostic_ood"] >= 20,
    }
    structural_gate_names = [
        "mechanism_family_count_at_least_8", "question_count_at_least_64",
        "world_count_at_least_256", "exploration_seed_count_at_least_8",
        "same_question_pairs_only", "independent_confirmation_not_copied",
        "composite_families_present", "raw_track_hides_structured_state",
        "candidate_pool_hindsight_blocked", "multi_step_trajectories_present",
    ]
    structural_pass = all(bool(gates[name]) for name in structural_gate_names)
    formal_gate_status = "OPEN" if all(
        gates[name] for name in (
            "formal_dev_pair_minimum_30", "formal_ood_pair_minimum_30",
            "formal_dev_cluster_minimum_20", "formal_ood_cluster_minimum_20",
        )
    ) else "CLOSED_underpowered"
    result = {
        "schema_version": "pesco_tier1_v04_extended_result_v0.1",
        "status": "completed_cpu_diagnostic",
        "pass": bool(structural_pass),
        "diagnostic_structure_pass": bool(structural_pass),
        "formal_gate_status": formal_gate_status,
        "gates": gates,
        "question_count": len(raw_dataset.examples) // 4,
        "world_count": len(raw_dataset.examples),
        "mechanism_families": list(manifest["mechanism_families"]),
        "exploration_seeds": list(protocol.exploration_seeds),
        "confirmation_seeds": list(protocol.confirmation_seeds),
        "tracks": list(V04_TRACKS),
        "decision_count": len(decisions),
        "trajectory_count": len(trajectories),
        "decision_question_ids": [question.question_id for question in selected],
        "decision_selection_policy": "family_round_robin_variant3_anchor",
        "reversal_counts_by_split": split_pair_counts,
        "cluster_counts_by_split": cluster_counts,
        "benchmark_manifest_digest": manifest["manifest_digest"],
        "public_manifest_digest": public_manifest["manifest_digest"],
        "protocol_digest": _protocol_digest(protocol),
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
        "tier2_claim": False,
        "llm_claim": False,
    }
    _dump(output_dir / "benchmark_manifest.json", manifest)
    _dump(output_dir / "benchmark_public_manifest.json", public_manifest)
    _dump(output_dir / "decisions.json", decisions)
    _dump(output_dir / "trajectories.json", trajectories)
    _dump(output_dir / "tier1_v04_extended_go.json", result)
    root = ROOT
    run_manifest = build_run_manifest(
        experiment="tier1_v04_extended_diagnostic",
        repo_root=root,
        command=sys.argv,
        runner_paths=[
            root / "research_strategy_optimization/evaluation/tier1_v04_extended.py",
            root / "research_strategy_optimization/evaluation/tier1_v04.py",
            root / "scripts/run_tier1_v04_extended.py",
            root / "research_strategy_optimization/environments/tier0_simulator.py",
            root / "research_strategy_optimization/environments/tier1_tabular_env.py",
            root / "research_strategy_optimization/schemas.py",
            root / "research_strategy_optimization/utils/run_manifest.py",
        ],
        data_paths=[
            path
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        ],
        seeds={
            "training": [17],
            "inference": [17],
            "environment_reset": [17],
            "exploration": list(protocol.exploration_seeds),
            "confirmation": list(protocol.confirmation_seeds),
        },
        checkpoint=None,
        status="completed_diagnostic",
        diagnostics={
            "capture_mode": "in_run",
            "artifact_status": result["status"],
            "artifact_pass": bool(result["pass"]),
            "formal_comparison_authorized": False,
            "formal_gate_status": formal_gate_status,
            "gates": gates,
            "protocol_version": protocol.protocol_version,
            "protocol_digest": result["protocol_digest"],
            "question_count": result["question_count"],
            "world_count": result["world_count"],
            "decision_count": result["decision_count"],
            "trajectory_count": result["trajectory_count"],
            "tracks": list(V04_TRACKS),
            "mechanism_families": list(manifest["mechanism_families"]),
            "formal_final_splits_opened": False,
            "tier2_claim": False,
            "llm_claim": False,
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", run_manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/tier1_v04_extended")
    parser.add_argument("--decision-questions", type=int, default=8)
    parser.add_argument("--question-limit", type=int, default=None, help="diagnostic smoke limit; omit for all 64 questions")
    parser.add_argument("--reuse-dataset-dir", type=Path, default=None, help="reuse a completed raw dataset and only run decisions/audits")
    args = parser.parse_args(argv)
    result = run(
        Path(args.output),
        decision_questions=args.decision_questions,
        question_limit=args.question_limit,
        reuse_dataset_dir=args.reuse_dataset_dir,
    )
    print(json.dumps({"output": args.output, "pass": result["pass"], "status": result["status"], "gates": result["gates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

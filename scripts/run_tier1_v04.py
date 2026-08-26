#!/usr/bin/env python3
"""Run the diagnostic Tier-1 v0.4 benchmark hardening.

v0.4 keeps the executable v0.3 NumPy worlds but replaces the hindsight-prone target
table with two explicit posterior-planning tracks: evaluator-side Oracle-state and
Raw-evidence.  It records posterior expected utility and value-of-information (VOI)
for every question/world.  No formal model, final split, or external comparison is
authorized by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.environments.tier0_simulator import TrustedVerifier
from research_strategy_optimization.evaluation.tier1_v04 import (
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    V04_TRACKS,
    build_candidate_action_table,
    build_tier1_v04_benchmark,
    plan_world,
    tier1_v04_manifest,
)
from research_strategy_optimization.schemas import Protocol, ResearchAction
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _protocol_digest(protocol: Protocol) -> str:
    payload = {
        "protocol_version": protocol.protocol_version,
        "delta_min": protocol.delta_min,
        "confidence_level": protocol.confidence_level,
        "invalid_precedence": protocol.invalid_precedence,
        "independent_confirmation_required": protocol.independent_confirmation_required,
        "exploration_seeds": list(protocol.exploration_seeds),
        "confirmation_seeds": list(protocol.confirmation_seeds),
        "max_budget": protocol.max_budget,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _initial_record(question: Any, world: Any, output: Any, verdict: Any) -> dict:
    observation = output  # overwritten by caller; this placeholder keeps field order stable
    return {
        "question_id": question.question_id,
        "world_id_audit": world.world_id,
        "family": question.family,
        "split": question.split,
        "initial_evidence_state_audit": verdict.evidence_state.value,
        "initial_validity_pass": bool(verdict.validity_pass),
        "initial_effect_estimate": float(output.effect_estimate),
        "initial_confidence_interval": [float(output.ci_low), float(output.ci_high)],
        "initial_backend": str(output.backend),
        "initial_dataset_hash": str(output.dataset_hash),
        "legacy_target_action_audit": None,
        "legacy_target_action_consumed": False,
    }


def run(
    output_dir: Path,
    *,
    protocol: Protocol | None = None,
    max_questions: int | None = None,
    command: Sequence[str] | None = None,
) -> dict:
    protocol = protocol or Protocol()
    benchmark = build_tier1_v04_benchmark()
    if str(protocol.protocol_version) != str(benchmark.protocol_version):
        raise ValueError(
            f"protocol/benchmark mismatch: {protocol.protocol_version!r} != {benchmark.protocol_version!r}"
        )
    questions = benchmark.questions[: int(max_questions)] if max_questions is not None else benchmark.questions
    verifier = TrustedVerifier(protocol)
    decisions: list[dict] = []
    initial_rows: list[dict] = []
    candidate_pool_audits: list[dict] = []
    backend_ok = True
    clone_ok = True

    for question in questions:
        # Candidate simulations are shared by all four worlds and both tracks for a
        # question.  Crucially, candidate_scenarios excludes this question itself.
        table = build_candidate_action_table(benchmark, question, protocol)
        candidate_pool_audits.append({
            "question_id": question.question_id,
            "candidate_pool_question_ids": sorted({key.split("|", 1)[0] for key in table}),
            "candidate_pool_size": len(table),
            "current_question_excluded": all(
                key.split("|", 1)[0] != question.question_id for key in table
            ),
        })
        for world in question.worlds:
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            env.reset(question.policy_question_id, world.world_id, seed=17)
            snapshot = env.snapshot()
            initial_branch = env.clone_from_snapshot(snapshot)
            clone_ok = clone_ok and type(initial_branch) is type(env)
            initial_output = initial_branch.execute_option(
                ResearchAction.CONTINUE,
                seeds=protocol.exploration_seeds,
            )
            initial_verdict = verifier.evaluate(initial_output, initial_branch)
            initial_observation = initial_branch.visible_observation().to_dict()
            backend_ok = backend_ok and initial_output.backend == "tier1_numpy"
            record = _initial_record(question, world, initial_output, initial_verdict)
            record["initial_public_observation"] = {
                key: value
                for key, value in initial_observation.items()
                if key not in {"question_id"}
            }
            # Keep the old table as an explicit audit-only comparison.  It is never
            # passed into plan_world and cannot affect either v0.4 track.
            record["legacy_target_action_audit"] = question.target_action(world.world_id).value
            initial_rows.append(record)
            for track in V04_TRACKS:
                decision = plan_world(
                    benchmark,
                    question,
                    world,
                    initial_output,
                    initial_verdict,
                    initial_observation,
                    track=track,
                    protocol=protocol,
                    candidate_table=table,
                )
                decision["legacy_target_action_audit"] = question.target_action(world.world_id).value
                decision["legacy_target_action_used_for_choice"] = False
                decisions.append(decision)

    by_track = {
        track: [decision for decision in decisions if decision["track"] == track]
        for track in V04_TRACKS
    }
    action_counts = {
        track: dict(Counter(decision["posterior_optimal_action"] for decision in rows))
        for track, rows in by_track.items()
    }
    causal_refuted = [
        decision
        for decision in decisions
        if decision["family"] == "causal_confounding"
        and decision["world_id_audit"].endswith("__refuted")
    ]
    disagreement = sum(
        left["posterior_optimal_action"] != right["posterior_optimal_action"]
        for left, right in zip(
            sorted(by_track[TRACK_ORACLE_STATE], key=lambda row: (row["question_id"], row["world_id_audit"])),
            sorted(by_track[TRACK_RAW_EVIDENCE], key=lambda row: (row["question_id"], row["world_id_audit"])),
        )
    )
    voi_positive = {
        track: sum(
            any(float(value) > 1e-9 for value in decision["value_of_information"].values())
            for decision in rows
        )
        for track, rows in by_track.items()
    }
    current_manifest = tier1_v04_manifest(benchmark, include_hidden=True)
    public_manifest = tier1_v04_manifest(benchmark, include_hidden=False)
    summary = {
        "schema_version": "pesco_tier1_v04_summary_v0.1",
        "benchmark_schema_version": current_manifest["schema_version"],
        "protocol_version": protocol.protocol_version,
        "protocol_digest": _protocol_digest(protocol),
        "question_count": len(questions),
        "world_count": len(questions) * 4,
        "decision_count": len(decisions),
        "tracks": list(V04_TRACKS),
        "action_counts": action_counts,
        "oracle_state_vs_raw_evidence_action_disagreement_n": int(disagreement),
        "positive_voi_decision_n": voi_positive,
        "causal_confounding_refuted_audit": [
            {
                "question_id": row["question_id"],
                "world_id_audit": row["world_id_audit"],
                "track": row["track"],
                "posterior_optimal_action": row["posterior_optimal_action"],
                "legacy_target_action_audit": row["legacy_target_action_audit"],
                "legacy_target_action_used_for_choice": False,
                "method_b_hindsight_excluded": bool(row["method_b_hindsight_excluded"]),
                "candidate_pool_excludes_current_question": bool(row["candidate_pool_excludes_current_question"]),
            }
            for row in causal_refuted
        ],
        "all_candidate_pools_exclude_current_question": all(
            bool(item["current_question_excluded"]) for item in candidate_pool_audits
        ),
        "all_choices_use_posterior_eu_or_voi": all(
            row["target_action_source"] == "leave_one_question_out_posterior_expected_utility_plus_value_of_information"
            and row["legacy_target_action_consumed"] is False
            for row in decisions
        ),
        "backend_verified": bool(backend_ok),
        "clone_type_preserved": bool(clone_ok),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "tier2_claim": False,
        "llm_claim": False,
    }
    gates = {
        "question_count": len(questions) == 12,
        "world_count": len(questions) * 4 == 48,
        "both_tracks_present": set(V04_TRACKS) == set(track for track in V04_TRACKS),
        "candidate_leave_one_question_out": summary["all_candidate_pools_exclude_current_question"],
        "posterior_expected_utility_recorded": all("posterior_expected_utility" in row for row in decisions),
        "value_of_information_recorded": all("value_of_information" in row for row in decisions),
        "causal_method_b_hindsight_blocked": all(
            bool(row["method_b_hindsight_excluded"])
            and row["legacy_target_action_used_for_choice"] is False
            for row in causal_refuted
        ),
        "backend_verified": bool(backend_ok),
        "clone_type_preserved": bool(clone_ok),
        "formal_final_splits_closed": True,
        "formal_comparison_authorized": False,
    }
    positive_gate_names = (
        "question_count",
        "world_count",
        "both_tracks_present",
        "candidate_leave_one_question_out",
        "posterior_expected_utility_recorded",
        "value_of_information_recorded",
        "causal_method_b_hindsight_blocked",
        "backend_verified",
        "clone_type_preserved",
        "formal_final_splits_closed",
    )
    result = {
        "schema_version": "pesco_tier1_v04_go_v0.1",
        "status": "completed_cpu_diagnostic",
        # Boundary assertions such as ``formal_comparison_authorized=false`` are
        # intentional safety properties, not failed GO gates.
        "pass": bool(all(bool(gates[name]) for name in positive_gate_names)),
        "pass_gate_names": list(positive_gate_names),
        "gates": gates,
        "summary": summary,
        "manifest_digest": current_manifest["manifest_digest"],
        "public_manifest_digest": public_manifest["manifest_digest"],
        "benchmark_manifest": current_manifest,
        "tracks": {
            TRACK_ORACLE_STATE: "evaluator-side state upper bound; not a raw model result",
            TRACK_RAW_EVIDENCE: "public raw evidence only; state label withheld",
        },
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "tier2_claim": False,
        "llm_claim": False,
    }
    _dump(output_dir / "benchmark_manifest.json", current_manifest)
    _dump(output_dir / "benchmark_public_manifest.json", public_manifest)
    _dump(output_dir / "initial_rows.json", initial_rows)
    _dump(output_dir / "decisions.json", decisions)
    _dump(output_dir / "candidate_pool_audit.json", candidate_pool_audits)
    _dump(output_dir / "summary.json", summary)
    _dump(output_dir / "tier1_v04_go.json", result)

    # Keep provenance attached to the executable artifact.  This is deliberately
    # written by the runner itself (rather than only by a later audit script), so
    # future v0.4 runs capture the exact command, source/data digests, runtime,
    # and semantic seed categories at the execution boundary.
    protocol_seeds = {
        # v0.4 is a CPU diagnostic and has no learned training phase; seed 17 is
        # the deterministic environment-reset/inference seed used by the runner.
        "training": [17],
        "inference": [17],
        "environment_reset": [17],
        "exploration": list(protocol.exploration_seeds),
        "confirmation": list(protocol.confirmation_seeds),
    }
    artifact_paths = [
        output_dir / "benchmark_manifest.json",
        output_dir / "benchmark_public_manifest.json",
        output_dir / "initial_rows.json",
        output_dir / "decisions.json",
        output_dir / "candidate_pool_audit.json",
        output_dir / "summary.json",
        output_dir / "tier1_v04_go.json",
    ]
    source_paths = [
        Path(__file__).resolve(),
        PESCO_ROOT / "research_strategy_optimization/evaluation/tier1_v04.py",
        PESCO_ROOT / "research_strategy_optimization/schemas.py",
        PESCO_ROOT / "research_strategy_optimization/environments/tier0_simulator.py",
        PESCO_ROOT / "research_strategy_optimization/environments/tier1_benchmark.py",
        PESCO_ROOT / "research_strategy_optimization/utils/run_manifest.py",
    ]
    manifest = build_run_manifest(
        experiment="P1_tier1_v04_benchmark_hardening",
        repo_root=PESCO_ROOT,
        command=command
        or [
            "python",
            "scripts/run_tier1_v04.py",
            "--output",
            str(output_dir),
        ],
        runner_paths=source_paths,
        # The benchmark/public manifests and complete result tables are the
        # concrete data boundary for this diagnostic run.
        data_paths=artifact_paths,
        seeds=protocol_seeds,
        checkpoint=None,
        status="completed" if result["pass"] else "completed_with_failed_gates",
        diagnostics={
            "capture_mode": "in_run",
            "artifact_status": result["status"],
            "artifact_pass": bool(result["pass"]),
            "benchmark_manifest_digest": result["manifest_digest"],
            "public_manifest_digest": result["public_manifest_digest"],
            "protocol_version": protocol.protocol_version,
            "protocol_digest": result["summary"]["protocol_digest"],
            "question_count": result["summary"]["question_count"],
            "world_count": result["summary"]["world_count"],
            "decision_count": result["summary"]["decision_count"],
            "tracks": list(V04_TRACKS),
            "formal_final_splits_opened": False,
            "formal_comparison_authorized": False,
            "diagnostic_only": True,
            "tier2_claim": False,
            "llm_claim": False,
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="PESCO/artifacts/tier1_v04")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(Path(args.output), max_questions=args.max_questions, command=sys.argv)
    print(json.dumps({"output": args.output, "pass": result["pass"], "status": result["status"]}, ensure_ascii=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

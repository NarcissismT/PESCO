#!/usr/bin/env python3
"""Write P0 provenance manifests for the completed Tier-1 artifacts.

This is intentionally a post-hoc, metadata-only command: it hashes existing
artifacts/checkpoint files and never reruns A, B, or the differentiable suite.
It is useful when an expensive run predates the provenance helper or when a
checkpoint is available outside the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing(root: Path, *relative_paths: str) -> list[Path]:
    return [root / relative for relative in relative_paths if (root / relative).exists()]


def _protocol_seeds() -> dict[str, Any]:
    protocol = Protocol(protocol_version="pesco_v0_2")
    return {
        "training": [17],
        "inference": [17],
        "exploration": list(protocol.exploration_seeds),
        "confirmation": list(protocol.confirmation_seeds),
    }


def _common_source_paths(root: Path) -> list[Path]:
    return _existing(
        root,
        "research_strategy_optimization/schemas.py",
        "research_strategy_optimization/environments/tier0_simulator.py",
        "research_strategy_optimization/environments/tier1_benchmark.py",
        "research_strategy_optimization/environments/tier1_tabular_env.py",
    )


def _write_a(root: Path, output_root: Path) -> dict[str, Any]:
    artifact_dir = output_root / "tier1_v03"
    go = _load(artifact_dir / "tier1_go.json")
    experiment = _load(artifact_dir / "experiment_a_environment_correctness.json")
    benchmark_manifest_path = artifact_dir / "benchmark_manifest.json"
    benchmark_manifest = _load(benchmark_manifest_path) if benchmark_manifest_path.exists() else {}
    data_paths = _existing(
        root,
        "artifacts/tier1_v03/benchmark_manifest.json",
        "artifacts/tier1_v03/split_manifest.json",
        "artifacts/tier1_v03/question_world_groups.json",
        "artifacts/tier1_v03/action_level_branches.json",
        "artifacts/tier1_v03/count_semantics_audit.json",
        "artifacts/tier1_v03/branch_groups.json",
        "artifacts/tier1_v03/seed_level_results.jsonl",
        "artifacts/tier1_v03/confirmation_denominators.json",
        "artifacts/tier1_v03/experiment_a_environment_correctness.json",
    )
    sources = _common_source_paths(root) + _existing(root, "scripts/run_tier1_v03.py")
    manifest = build_run_manifest(
        experiment="A_environment_correctness",
        repo_root=root,
        command=["python", "scripts/run_tier1_v03.py", "artifacts/tier1_v03"],
        runner_paths=sources,
        data_paths=data_paths,
        seeds=_protocol_seeds(),
        diagnostics={
            "capture_mode": "posthoc_existing_artifact",
            "artifact_status": "completed_diagnostic",
            "artifact_pass": bool(experiment.get("pass", go.get("pass", False))),
            "protocol_version": go.get("protocol_version"),
            "protocol_digest": go.get("protocol_digest"),
            # A's GO summary predates the benchmark manifest digest field.  Read
            # the canonical benchmark manifest directly so the provenance record
            # always links the exact data contract used by the artifact.
            "benchmark_manifest_digest": benchmark_manifest.get("manifest_digest")
            or go.get("benchmark_manifest_digest"),
            "question_world_group_count": int(
                experiment.get("question_world_group_count", experiment.get("branch_groups", 48))
            ),
            "snapshot_branch_groups": int(
                experiment.get("question_world_group_count", experiment.get("branch_groups", 48))
            ),
            "action_level_branch_rows": int(
                experiment.get("action_level_row_count", experiment.get("action_level_rows", 192))
            ),
            "action_level_row_count": int(
                experiment.get("action_level_row_count", experiment.get("action_level_rows", 192))
            ),
            "seed_level_executions": int(
                experiment.get("seed_level_observation_count", experiment.get("expected_seed_level_experiments", 768))
            ),
            "seed_level_observation_count": int(
                experiment.get("seed_level_observation_count", experiment.get("expected_seed_level_experiments", 768))
            ),
            "confirmation_eligible_n": experiment.get("confirmation_denominators", {}).get("eligible_n"),
        },
    )
    write_run_manifest(artifact_dir / "run_manifest.json", manifest)
    return manifest


def _write_b(root: Path, output_root: Path) -> dict[str, Any]:
    artifact = _load(output_root / "tier1_zero_shot.json")
    status_path = output_root / "tier1_zero_shot_current_status.json"
    status = _load(status_path) if status_path.exists() else {}
    checkpoint_raw = artifact.get("checkpoint") or status.get("checkpoint")
    checkpoint = Path(checkpoint_raw) if checkpoint_raw else None
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    data_paths = _existing(
        root,
        "artifacts/tier1_v03/benchmark_manifest.json",
        "artifacts/tier1_zero_shot.json",
        "artifacts/tier1_zero_shot_current_status.json",
    )
    sources = _common_source_paths(root) + _existing(root, "scripts/run_tier1_zero_shot.py")
    manifest = build_run_manifest(
        experiment="B_zero_shot_failure_diagnosis",
        repo_root=root,
        command=[
            "python",
            "scripts/run_tier1_zero_shot.py",
            "--checkpoint",
            str(checkpoint) if checkpoint is not None else "<missing-checkpoint>",
            "--output",
            "artifacts/tier1_zero_shot.json",
            "--batch-size",
            "8",
        ],
        runner_paths=sources,
        data_paths=data_paths,
        seeds=_protocol_seeds(),
        checkpoint=checkpoint,
        diagnostics={
            "capture_mode": "posthoc_existing_artifact",
            "artifact_status": artifact.get("status"),
            "artifact_pass": bool(artifact.get("pass", False)),
            "formal_comparison_authorized": bool(artifact.get("formal_comparison_authorized", False)),
            "protocol_version": artifact.get("protocol_version"),
            "protocol_digest": artifact.get("protocol_digest"),
            "benchmark_manifest_digest": artifact.get("benchmark_manifest_digest"),
            "benchmark_freshness_match": artifact.get("benchmark_freshness_match"),
            "current_full_forward_completed": artifact.get("current_full_forward_completed"),
            "model_forward_pass_executed_this_run": artifact.get("model_forward_pass_executed_this_run"),
            "model_row_provenance": artifact.get("model_row_provenance"),
            "row_count": artifact.get("row_count"),
            "action_accuracy_audit": artifact.get("action_accuracy_audit"),
            "state_accuracy_audit": artifact.get("state_accuracy_audit"),
            "current_status_file": str(status_path.relative_to(root)) if status_path.exists() else None,
        },
    )
    write_run_manifest(output_root / "tier1_zero_shot_run_manifest.json", manifest)
    return manifest


def _write_cdef(root: Path, output_root: Path) -> dict[str, Any]:
    artifact_dir = output_root / "tier1_differentiable_suite"
    suite = _load(artifact_dir / "suite.json")
    data_paths = _existing(
        root,
        "artifacts/tier1_v03/benchmark_manifest.json",
        "artifacts/tier1_differentiable_suite/dataset.json",
        "artifacts/tier1_differentiable_suite/dataset_public.json",
        "artifacts/tier1_differentiable_suite/suite.json",
        "artifacts/tier1_differentiable_suite/experiment_c_state_reward.json",
        "artifacts/tier1_differentiable_suite/experiment_d_branch_ablation.json",
        "artifacts/tier1_differentiable_suite/experiment_e_flip_ablation.json",
        "artifacts/tier1_differentiable_suite/experiment_f_discovery_boundary.json",
    )
    data_paths += sorted(artifact_dir.glob("policy_*.pt"))
    sources = _common_source_paths(root) + _existing(
        root,
        "scripts/run_tier1_differentiable_suite.py",
        "research_strategy_optimization/algorithms/differentiable_strategy.py",
        "research_strategy_optimization/evaluation/tier1_differentiable_suite.py",
    )
    matched = suite.get("matched_compute", {})
    seeds = {
        "training": [17],
        "inference": [17],
        "exploration": matched.get("same_exploration_seeds", [17, 29, 41, 53]),
        "confirmation": matched.get("same_confirmation_seeds", [103, 107, 109, 113]),
    }
    training_logs = suite.get("training_logs", {})
    manifest = build_run_manifest(
        experiment="C_D_E_F_differentiable_suite",
        repo_root=root,
        command=[
            "python",
            "scripts/run_tier1_differentiable_suite.py",
            "--output",
            "artifacts/tier1_differentiable_suite",
            "--epochs",
            "16",
            "--max-optimizer-steps",
            str(matched.get("optimizer_step_cap", 128)),
            "--seed",
            "17",
        ],
        runner_paths=sources,
        data_paths=data_paths,
        seeds=seeds,
        diagnostics={
            "capture_mode": "posthoc_existing_artifact",
            "artifact_status": suite.get("implementation_status"),
            "protocol_version": suite.get("protocol_version"),
            "protocol_digest": suite.get("protocol_digest"),
            "benchmark_manifest_digest": suite.get("benchmark_manifest", {}).get("manifest_digest"),
            "experiments": ["C", "D", "E", "F"],
            "methods": sorted(training_logs),
            "optimizer_step_cap": matched.get("optimizer_step_cap"),
            "question_world_group_count": suite.get("dataset_provenance", {}).get(
                "question_world_group_count", suite.get("dataset_provenance", {}).get("branch_groups")
            ),
            "branch_groups": suite.get("dataset_provenance", {}).get(
                "question_world_group_count", suite.get("dataset_provenance", {}).get("branch_groups")
            ),
            "action_level_branch_rows": suite.get("dataset_provenance", {}).get(
                "action_level_row_count", suite.get("dataset_provenance", {}).get("action_level_rows", 192)
            ),
            "action_level_row_count": suite.get("dataset_provenance", {}).get(
                "action_level_row_count", suite.get("dataset_provenance", {}).get("action_level_rows", 192)
            ),
            "seed_level_executions": suite.get("dataset_provenance", {}).get(
                "seed_level_observation_count", suite.get("dataset_provenance", {}).get("exploration_seed_observations")
            ),
            "seed_level_observation_count": suite.get("dataset_provenance", {}).get(
                "seed_level_observation_count", suite.get("dataset_provenance", {}).get("exploration_seed_observations")
            ),
            "reversal_count": suite.get("dataset_provenance", {}).get("reversal_count"),
            "formal_final_splits_opened": False,
            "tier2_claim": suite.get("tier2_claim"),
            "llm_claim": suite.get("llm_claim"),
        },
    )
    write_run_manifest(artifact_dir / "run_manifest.json", manifest)
    return manifest


def _write_v04(root: Path, output_root: Path) -> dict[str, Any]:
    """Attach provenance to an existing v0.4 artifact without rerunning it."""

    artifact_dir = output_root / "tier1_v04"
    result = _load(artifact_dir / "tier1_v04_go.json")
    summary = _load(artifact_dir / "summary.json")
    benchmark = _load(artifact_dir / "benchmark_manifest.json")
    data_paths = _existing(
        root,
        "artifacts/tier1_v04/benchmark_manifest.json",
        "artifacts/tier1_v04/benchmark_public_manifest.json",
        "artifacts/tier1_v04/initial_rows.json",
        "artifacts/tier1_v04/decisions.json",
        "artifacts/tier1_v04/candidate_pool_audit.json",
        "artifacts/tier1_v04/summary.json",
        "artifacts/tier1_v04/tier1_v04_go.json",
    )
    sources = _common_source_paths(root) + _existing(
        root,
        "scripts/run_tier1_v04.py",
        "research_strategy_optimization/evaluation/tier1_v04.py",
        "research_strategy_optimization/utils/run_manifest.py",
    )
    protocol = Protocol()
    manifest = build_run_manifest(
        experiment="P1_tier1_v04_benchmark_hardening",
        repo_root=root,
        command=["python", "scripts/run_tier1_v04.py", "--output", "artifacts/tier1_v04"],
        runner_paths=sources,
        data_paths=data_paths,
        seeds={
            "training": [17],
            "inference": [17],
            "environment_reset": [17],
            "exploration": list(protocol.exploration_seeds),
            "confirmation": list(protocol.confirmation_seeds),
        },
        checkpoint=None,
        status="completed" if result.get("pass") else "completed_with_failed_gates",
        diagnostics={
            "capture_mode": "posthoc_existing_artifact",
            "artifact_status": result.get("status"),
            "artifact_pass": bool(result.get("pass", False)),
            "benchmark_manifest_digest": benchmark.get("manifest_digest"),
            "public_manifest_digest": result.get("public_manifest_digest"),
            "protocol_version": summary.get("protocol_version"),
            "protocol_digest": summary.get("protocol_digest"),
            "question_count": summary.get("question_count"),
            "world_count": summary.get("world_count"),
            "decision_count": summary.get("decision_count"),
            "tracks": summary.get("tracks"),
            "formal_final_splits_opened": False,
            "formal_comparison_authorized": False,
            "diagnostic_only": True,
            "tier2_claim": False,
            "llm_claim": False,
        },
    )
    write_run_manifest(artifact_dir / "run_manifest.json", manifest)
    return manifest


def write_all_manifests(
    root: Path = ROOT,
    output_root: Path | None = None,
    *,
    only: tuple[str, ...] = ("A", "B", "CDEF"),
) -> dict[str, dict[str, Any]]:
    """Write selected A, B, and unified C--F manifests without rerunning experiments.

    ``only`` is useful when refreshing a lightweight post-hoc record (for example,
    fixing A's metadata) without re-hashing the multi-gigabyte B checkpoint.
    """

    root = Path(root).resolve()
    output = (output_root or root / "artifacts").resolve()
    writers = {
        "A": lambda: _write_a(root, output),
        "B": lambda: _write_b(root, output),
        "CDEF": lambda: _write_cdef(root, output),
        "V04": lambda: _write_v04(root, output),
    }
    requested = tuple(only)
    unknown = sorted(set(requested).difference(writers))
    if unknown:
        raise ValueError(f"unknown manifest groups: {unknown}")
    return {key: writers[key]() for key in requested}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("A", "B", "CDEF", "V04"),
        default=["A", "B", "CDEF"],
        help="manifest groups to refresh (omit to refresh all; B hashes the full checkpoint)",
    )
    args = parser.parse_args(argv)
    manifests = write_all_manifests(
        Path(args.repo_root),
        Path(args.output_root) if args.output_root else None,
        only=tuple(args.only),
    )
    print(
        json.dumps(
            {
                key: {
                    "manifest_digest": value["manifest_digest"],
                    "path": {
                        "A": "artifacts/tier1_v03/run_manifest.json",
                        "B": "artifacts/tier1_zero_shot_run_manifest.json",
                        "CDEF": "artifacts/tier1_differentiable_suite/run_manifest.json",
                        "V04": "artifacts/tier1_v04/run_manifest.json",
                    }[key],
                    "checkpoint_full_digest": value["checkpoint"].get("full_checkpoint_digest"),
                }
                for key, value in manifests.items()
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

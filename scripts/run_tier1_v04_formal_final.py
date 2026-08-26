#!/usr/bin/env python3
"""Build and optionally collect the Tier-1 v0.4 formal final splits.

The default command is structural-only: it writes public/hidden manifests and a
whole-family holdout audit while leaving final access locked.  Environment
execution is intentionally opt-in and requires an authorization receipt.  This
keeps merely constructing a benchmark from being mistaken for a completed formal
evaluation or a model-comparison authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_v04_extended import (
    FORMAL_FINAL_ID_FAMILIES,
    FORMAL_FINAL_OOD_FAMILIES,
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_EXTENDED_EXPLORATION_SEEDS,
    build_tier1_v04_extended_benchmark,
    build_tier1_v04_formal_final_benchmark,
    collect_tier1_v04_extended,
)
from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest
from scripts.run_tier1_v04_extended import _derive_oracle_dataset, _dump


AUTHORIZATION_FIELDS = (
    "allow_final_access",
    "freeze_check_pass",
    "contamination_audit_pass",
    "model_frozen",
    "hyperparameters_frozen",
)


def _digest_manifest(payload: dict) -> dict:
    unsigned = dict(payload)
    unsigned.pop("manifest_digest", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["manifest_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return payload


def _dump_local(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _load_authorization(path: Path | None) -> dict[str, Any]:
    """Load and validate the explicit final-access receipt.

    A missing receipt is not an error for structural-only mode, but collection
    mode must fail closed unless every required field is explicitly ``true``.
    """

    if path is None:
        raise ValueError(
            "formal environment collection requires --authorization-file with "
            "all final-access fields set to true"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("formal authorization receipt must be a JSON object")
    missing = [field for field in AUTHORIZATION_FIELDS if payload.get(field) is not True]
    if missing:
        raise ValueError(
            "formal authorization receipt is incomplete; required true fields: "
            + ", ".join(missing)
        )
    return dict(payload)


def _split_counts(dataset: Any) -> tuple[dict[str, int], dict[str, int]]:
    splits = tuple(dict.fromkeys(str(example.split) for example in dataset.examples))
    pair_counts = {
        split: sum(
            bool(pair.confirmed)
            and dataset.examples[pair.left].split == split
            and dataset.examples[pair.right].split == split
            and dataset.examples[pair.left].question_id == dataset.examples[pair.right].question_id
            for pair in dataset.reversals
        )
        for split in splits
    }
    cluster_counts = {
        split: len({example.question_id for example in dataset.examples if example.split == split})
        for split in splits
    }
    return pair_counts, cluster_counts


def _structural_audit(benchmark: Any, diagnostic: Any, hidden_manifest: dict, public_manifest: dict) -> tuple[dict, dict, dict]:
    """Return structural gates, split counts, and final world counts."""

    diagnostic_hidden = diagnostic.manifest(include_hidden=True)
    diagnostic_question_ids = {item["question_id"] for item in diagnostic_hidden["questions"]}
    diagnostic_world_ids = {
        world["world_id"]
        for item in diagnostic_hidden["questions"]
        for world in item.get("worlds", [])
    }
    formal_question_ids = {item["question_id"] for item in hidden_manifest["questions"]}
    formal_world_ids = {
        world["world_id"]
        for item in hidden_manifest["questions"]
        for world in item.get("worlds", [])
    }
    id_families = {
        item["family"] for item in hidden_manifest["questions"] if item["split"] == "final_id"
    }
    ood_families = {
        item["family"] for item in hidden_manifest["questions"] if item["split"] == "final_ood"
    }
    counts = dict(hidden_manifest["counts_by_split"])
    world_counts = {
        split: sum(
            item["world_count"]
            for item in hidden_manifest["questions"]
            if item["split"] == split
        )
        for split in ("final_id", "final_ood")
    }
    public_questions_encoded = json.dumps(public_manifest["questions"], sort_keys=True)
    structural_gates = {
        "final_id_question_count_at_least_20": counts.get("final_id", 0) >= 20,
        "final_ood_question_count_at_least_20": counts.get("final_ood", 0) >= 20,
        "final_id_world_count_at_least_80": world_counts.get("final_id", 0) >= 80,
        "final_ood_world_count_at_least_80": world_counts.get("final_ood", 0) >= 80,
        "whole_family_ood_holdout": (
            id_families == set(FORMAL_FINAL_ID_FAMILIES)
            and ood_families == set(FORMAL_FINAL_OOD_FAMILIES)
            and not id_families.intersection(ood_families)
            and all(
                item["family"] not in ood_families
                for item in hidden_manifest["questions"]
                if item["split"] in {"train", "dev", "final_id"}
            )
        ),
        "final_question_ids_disjoint_from_diagnostic": not formal_question_ids.intersection(diagnostic_question_ids),
        "final_world_ids_disjoint_from_diagnostic": not formal_world_ids.intersection(diagnostic_world_ids),
        "public_manifest_omits_family_labels": all(
            "family" not in item for item in public_manifest["questions"]
        ) and "family" not in public_questions_encoded,
        "public_manifest_omits_legacy_targets": (
            "legacy_target_actions_audit_only" not in public_questions_encoded
            and "true_effect_b" not in json.dumps(public_manifest, sort_keys=True)
        ),
        "final_access_locked": True,
    }
    audit = {
        "schema_version": "pesco_tier1_v04_formal_final_holdout_audit_v0.1",
        "final_id_families": sorted(id_families),
        "final_ood_families": sorted(ood_families),
        "train_dev_final_id_excludes_final_ood_families": structural_gates["whole_family_ood_holdout"],
        "question_id_overlap_with_diagnostic": sorted(formal_question_ids.intersection(diagnostic_question_ids)),
        "world_id_overlap_with_diagnostic": sorted(formal_world_ids.intersection(diagnostic_world_ids)),
        "counts_by_split": counts,
        "world_counts_by_final_split": world_counts,
        "mechanism_families_id": list(FORMAL_FINAL_ID_FAMILIES),
        "mechanism_families_ood": list(FORMAL_FINAL_OOD_FAMILIES),
    }
    return structural_gates, counts, world_counts, audit


def run(
    output_dir: Path,
    *,
    collect_environment_audit: bool = False,
    authorization_file: Path | None = None,
    question_limit: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    benchmark = build_tier1_v04_formal_final_benchmark()
    diagnostic = build_tier1_v04_extended_benchmark()
    hidden_manifest = benchmark.manifest(
        include_hidden=True, exploration_seeds=protocol.exploration_seeds
    )
    public_manifest = benchmark.manifest(
        include_hidden=False, exploration_seeds=protocol.exploration_seeds
    )
    structural_gates, counts, world_counts, holdout_audit = _structural_audit(
        benchmark, diagnostic, hidden_manifest, public_manifest
    )
    _dump_local(output_dir / "whole_family_holdout_audit.json", holdout_audit)

    datasets: dict[str, Any] = {}
    audits: dict[str, dict] = {}
    pair_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    authorization: dict[str, Any] | None = None
    if collect_environment_audit:
        authorization = _load_authorization(authorization_file)
        raw_dataset, raw_audit = collect_tier1_v04_extended(
            benchmark,
            protocol,
            track=TRACK_RAW_EVIDENCE,
            question_limit=question_limit,
        )
        oracle_dataset, oracle_audit = _derive_oracle_dataset(raw_dataset)
        datasets = {TRACK_RAW_EVIDENCE: raw_dataset, TRACK_ORACLE_STATE: oracle_dataset}
        audits = {
            TRACK_RAW_EVIDENCE: raw_audit,
            TRACK_ORACLE_STATE: {**raw_audit, **oracle_audit},
        }
        for track, dataset in datasets.items():
            dataset.save_json(output_dir / f"dataset_{track}.json", include_audit=True)
            dataset.save_json(output_dir / f"dataset_{track}_public.json", include_audit=False)
            _dump_local(output_dir / f"audit_{track}.json", audits[track])
        pair_counts, cluster_counts = _split_counts(raw_dataset)
        all_raw_public = all(
            example.observation.track == TRACK_RAW_EVIDENCE
            and not example.observation.validity_signals
            and not example.observation.history_summary
            for example in raw_dataset.examples
        )
        structural_gates.update({
            "final_id_confirmed_reversal_minimum_30": pair_counts.get("final_id", 0) >= 30,
            "final_ood_confirmed_reversal_minimum_30": pair_counts.get("final_ood", 0) >= 30,
            "independent_confirmation_not_copied": raw_dataset.provenance.get("confirmation_bonus_copied_to_seed") is False,
            "raw_track_hides_structured_state": all_raw_public,
            "same_question_pair_scope": raw_dataset.provenance.get("reversal_pair_scope") == "same_question_world_pairs_only",
            "multi_step_trajectories_complete": int(raw_dataset.provenance.get("trajectory_complete_count", 0)) == len(raw_dataset.examples),
        })

    # A bounded ``--question-limit`` smoke is not a completed formal evaluation.
    # Keep the final-access lock and completion flag fail-closed unless every
    # registered formal cluster was collected and all receipt gates passed.
    full_collection = bool(
        collect_environment_audit
        and question_limit is None
        and counts.get("final_id", 0) >= 20
        and counts.get("final_ood", 0) >= 20
        and pair_counts.get("final_id", 0) >= 30
        and pair_counts.get("final_ood", 0) >= 30
    )
    completed = full_collection and bool(all(structural_gates.values()))
    final_access = {
        "locked": True,
        "first_access_completed": bool(collect_environment_audit),
        "authorization_required": True,
        "formal_evaluation_completed": completed,
        "formal_comparison_authorized": False,
        "environment_receipts_collected": bool(collect_environment_audit),
        "full_collection": full_collection,
    }
    for manifest in (hidden_manifest, public_manifest):
        manifest["final_access"] = dict(final_access)
        _digest_manifest(manifest)
    _dump_local(output_dir / "benchmark_manifest.json", hidden_manifest)
    _dump_local(output_dir / "benchmark_public_manifest.json", public_manifest)
    # The collector audit is written before the final-access status is attached;
    # refresh its embedded manifests so the receipt is self-consistent.
    if completed:
        for track, audit_payload in audits.items():
            audit_payload["benchmark_manifest"] = hidden_manifest
            audit_payload["public_benchmark_manifest"] = public_manifest
            _dump_local(output_dir / f"audit_{track}.json", audit_payload)
    for split in ("final_id", "final_ood"):
        public_items = [
            item for item in public_manifest["questions"] if item["split"] == split
        ]
        hidden_items = [
            item for item in hidden_manifest["questions"] if item["split"] == split
        ]
        _dump_local(output_dir / f"{split}_manifest.json", {
            "schema_version": hidden_manifest["schema_version"],
            "split": split,
            "question_count": counts.get(split, 0),
            "world_count": world_counts.get(split, 0),
            "questions": public_items,
            "final_access": dict(final_access),
        })
        _dump_local(output_dir / f"{split}_audit_hidden.json", {
            "schema_version": hidden_manifest["schema_version"],
            "split": split,
            "question_count": counts.get(split, 0),
            "world_count": world_counts.get(split, 0),
            "questions": hidden_items,
            "final_access": dict(final_access),
            "scope": "evaluator_audit_only_not_for_public_release",
        })
    _dump_local(output_dir / "formal_final_go.json", {
        "schema_version": "pesco_tier1_v04_formal_final_result_v0.1",
        "status": "completed_cpu_formal_final" if completed else "locked_manifest_ready",
        "pass": bool(all(structural_gates.values())),
        "gates": structural_gates,
        "counts_by_split": counts,
        "pair_counts_by_split": pair_counts,
        "cluster_counts_by_split": cluster_counts,
        "mechanism_families_id": list(FORMAL_FINAL_ID_FAMILIES),
        "mechanism_families_ood": list(FORMAL_FINAL_OOD_FAMILIES),
        "whole_family_ood_holdout": True,
        "formal_evaluation_completed": completed,
        "formal_comparison_authorized": False,
        "tier2_claim": False,
        "llm_claim": False,
        "authorization_receipt": authorization,
        "raw_dataset": "dataset_raw_evidence.json" if completed else None,
        "oracle_dataset": "dataset_oracle_state.json" if completed else None,
    })
    run_manifest = build_run_manifest(
        experiment="tier1_v04_formal_final_cpu",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            ROOT / "scripts/run_tier1_v04_formal_final.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_v04_extended.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_v04.py",
            ROOT / "research_strategy_optimization/environments/tier0_simulator.py",
            ROOT / "research_strategy_optimization/environments/tier1_tabular_env.py",
            ROOT / "research_strategy_optimization/schemas.py",
            ROOT / "research_strategy_optimization/utils/run_manifest.py",
        ],
        data_paths=[
            path for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        ],
        seeds={
            "environment_reset": [17],
            "exploration": list(protocol.exploration_seeds),
            "confirmation": list(protocol.confirmation_seeds),
        },
        checkpoint=None,
        status="completed_formal_final" if completed else "locked_manifest_ready",
        diagnostics={
            "capture_mode": "in_run" if completed else "structural_only",
            "formal_evaluation_completed": completed,
            "formal_comparison_authorized": False,
            "final_id_cluster_count": cluster_counts.get("final_id", 0),
            "final_ood_cluster_count": cluster_counts.get("final_ood", 0),
            "final_id_pair_count": pair_counts.get("final_id", 0),
            "final_ood_pair_count": pair_counts.get("final_ood", 0),
            "whole_family_ood_holdout": True,
            "tier2_claim": False,
            "llm_claim": False,
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", run_manifest)
    return {
        "output": str(output_dir),
        "pass": bool(all(structural_gates.values())),
        "status": "completed_cpu_formal_final" if completed else "locked_manifest_ready",
        "formal_evaluation_completed": completed,
        "formal_comparison_authorized": False,
        "gates": structural_gates,
        "pair_counts": pair_counts,
        "cluster_counts": cluster_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/tier1_v04_formal_final")
    parser.add_argument(
        "--collect-environment-audit",
        action="store_true",
        help="execute the full formal benchmark; requires --authorization-file",
    )
    parser.add_argument(
        "--authorization-file",
        type=Path,
        help="JSON receipt with all required final-access booleans true",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="optional bounded collection smoke (will generally fail minimum gates)",
    )
    args = parser.parse_args(argv)
    result = run(
        Path(args.output),
        collect_environment_audit=args.collect_environment_audit,
        authorization_file=args.authorization_file,
        question_limit=args.question_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

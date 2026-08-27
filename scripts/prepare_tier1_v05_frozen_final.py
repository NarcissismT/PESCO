#!/usr/bin/env python3
"""Prepare and audit the independent Tier-1 v0.5 frozen final.

The command creates two explicitly separate outputs:

* a public commitment directory containing opaque split manifests and summaries;
* an evaluator directory containing hidden worlds, generator recipes, and (optional)
  CPU transition receipts.

No model comparison is performed here.  ``--collect-environment-receipts`` only
executes the trusted synthetic environment.  A clean freeze receipt is signed only
with ``--sign-freeze`` when a pre-final baseline receipt, a clean tagged Git HEAD,
and all overlap/public-boundary audits pass.  On an ordinary working tree the
command intentionally emits a pending receipt rather than fabricating a freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_v05_frozen_final import (
    V05_CONFIRMATION_SEEDS,
    V05_EXPLORATION_SEEDS,
    V05_EVALUATOR_VERSION,
    V05_GENERATOR_VERSION,
    V05_SPLITS,
    audit_latent_generator_signatures,
    audit_public_manifest,
    build_freeze_receipt,
    build_tier1_v05_frozen_final_benchmark,
    collect_v05_environment_receipts,
)
from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import digest_paths


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: Any, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    if private:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _default_development_manifests() -> list[Path]:
    candidates = [
        ROOT / "artifacts/tier1_v04_extended/benchmark_manifest.json",
        ROOT / "artifacts/tier1_v04_formal_final/benchmark_manifest.json",
    ]
    return [path for path in candidates if path.exists()]


def _public_signature_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(audit.get("gates", {}))
    return {
        "schema_version": "pesco_v05_public_signature_summary_v0.1",
        "generator_version": audit.get("generator_version"),
        "final_question_count": audit.get("final_question_count", 0),
        "final_world_count": audit.get("final_world_count", 0),
        "final_latent_signature_count": audit.get("final_latent_signature_count", 0),
        "final_generator_signature_count": audit.get("final_generator_signature_count", 0),
        "development_manifest_count": audit.get("development_manifest_count", 0),
        "question_id_overlap_count": audit.get("question_id_overlap_count", 0),
        "world_id_overlap_count": audit.get("world_id_overlap_count", 0),
        "latent_overlap_count": audit.get("latent_overlap_count", 0),
        "generator_signature_overlap_count": audit.get("generator_signature_overlap_count", 0),
        "new_ood_family_count": audit.get("new_ood_family_count", 0),
        "gates": gates,
        "pass": bool(audit.get("pass") is True),
    }


def _split_public_manifest(manifest: Mapping[str, Any], split: str) -> dict[str, Any]:
    questions = [item for item in manifest.get("questions", []) if item.get("split") == split]
    payload = {
        "schema_version": manifest.get("schema_version"),
        "profile": manifest.get("profile"),
        "split": split,
        "question_count": len(questions),
        "world_count": sum(int(item.get("world_count", 0)) for item in questions),
        "questions": questions,
        "final_access": dict(manifest.get("final_access", {})),
    }
    payload["manifest_digest"] = _digest(payload)
    return payload


def _split_hidden_manifest(manifest: Mapping[str, Any], split: str) -> dict[str, Any]:
    questions = [item for item in manifest.get("questions", []) if item.get("split") == split]
    payload = {
        "schema_version": manifest.get("schema_version"),
        "profile": manifest.get("profile"),
        "split": split,
        "question_count": len(questions),
        "world_count": sum(int(item.get("world_count", 0)) for item in questions),
        "questions": questions,
        "final_access": dict(manifest.get("final_access", {})),
        "scope": "evaluator_audit_only_not_for_public_release",
    }
    payload["manifest_digest"] = _digest(payload)
    return payload


def _lightweight_run_manifest(
    *,
    source_inventory: Mapping[str, Any],
    data_inventory: Mapping[str, Any],
    status: str,
    diagnostics: Mapping[str, Any],
    seeds: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the standard provenance shape without a potentially hanging git scan.

    ``utils.run_manifest.build_run_manifest`` performs a full porcelain scan.  On
    the shared research filesystem that scan can outlive a bounded evaluator run
    when large generated bundles are present.  The strict clean/tag decision is
    still handled separately by :func:`build_freeze_receipt`; this lightweight
    manifest records the exact source/data inventories and a bounded HEAD lookup.
    """

    try:
        git_sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    payload: dict[str, Any] = {
        "schema_version": "pesco_run_manifest_v0.1",
        "experiment": "tier1_v05_frozen_final_prepare",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(str(item) for item in sys.argv),
        "git_sha": git_sha,
        "git": {
            "repository": str(ROOT),
            "sha": git_sha,
            "status_available": False,
            "dirty": None,
            "status_reason": "bounded_manifest_writer; see freeze_receipt for strict clean check",
        },
        "source": dict(source_inventory),
        "source_digest": source_inventory.get("digest"),
        "data": dict(data_inventory),
        "data_digest": data_inventory.get("digest"),
        "seeds": dict(seeds),
        "training_seed": None,
        "checkpoint": {
            "supplied": False,
            "available": False,
            "reason": "no_checkpoint_supplied",
        },
        "status": status,
        "diagnostics": dict(diagnostics),
    }
    payload["manifest_digest"] = _digest(payload)
    return payload


def run(
    output_dir: Path,
    evaluator_dir: Path,
    *,
    development_manifests: Sequence[Path] = (),
    collect_environment: bool = False,
    question_limit: int | None = None,
    baseline_selection: Mapping[str, Any] | None = None,
    sign_freeze: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    evaluator_dir = evaluator_dir.resolve()
    if output_dir == evaluator_dir:
        raise ValueError("public and evaluator outputs must be separate directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(evaluator_dir, 0o700)
    except OSError:
        pass

    benchmark = build_tier1_v05_frozen_final_benchmark()
    dev_paths = list(development_manifests) or _default_development_manifests()
    dev_payloads = [_load_json(path) for path in dev_paths]
    signature_audit = audit_latent_generator_signatures(benchmark, dev_payloads)
    hidden_manifest = benchmark.manifest(include_hidden=True)
    public_manifest = benchmark.manifest(include_hidden=False)
    public_audit = audit_public_manifest(public_manifest)

    # Keep evaluator-owned fields out of the public directory.  The hidden bundle
    # is written first to the private directory and receives restrictive permissions.
    _write_json(evaluator_dir / "benchmark_hidden_manifest.json", hidden_manifest, private=True)
    _write_json(evaluator_dir / "latent_signature_audit.json", signature_audit, private=True)
    for split in V05_SPLITS:
        _write_json(
            evaluator_dir / f"{split}_audit_hidden.json",
            _split_hidden_manifest(hidden_manifest, split), private=True,
        )
    _write_json(output_dir / "benchmark_public_manifest.json", public_manifest)
    for split in V05_SPLITS:
        _write_json(output_dir / f"{split}_manifest.json", _split_public_manifest(public_manifest, split))
    public_signature_summary = _public_signature_summary(signature_audit)
    _write_json(output_dir / "signature_audit_summary.json", public_signature_summary)

    environment_receipt = None
    if collect_environment:
        protocol = Protocol(
            protocol_version="pesco_v0_2",
            exploration_seeds=V05_EXPLORATION_SEEDS,
            confirmation_seeds=V05_CONFIRMATION_SEEDS,
            max_budget=6,
        )
        environment_receipt = collect_v05_environment_receipts(
            benchmark, protocol, question_limit=question_limit,
        )
        _write_json(evaluator_dir / "environment_receipts.json", environment_receipt, private=True)
        summary = {
            key: value for key, value in environment_receipt.items() if key != "rows"
        }
        summary["rows_hidden_in_evaluator_bundle"] = True
        _write_json(output_dir / "environment_receipt_summary.json", summary)

    # This contract is explicit about local limitations: a separate directory is an
    # evaluator boundary for CI packaging, but it is not a container security claim.
    evaluator_contract = {
        "schema_version": "pesco_v05_independent_evaluator_contract_v0.1",
        "evaluator_version": V05_EVALUATOR_VERSION,
        "private_bundle_path": str(evaluator_dir),
        "public_bundle_path": str(output_dir),
        "hidden_manifest_not_in_public_bundle": True,
        "independent_process_required_for_claim": True,
        "container_isolation_verified": False,
        "model_evaluation_completed": False,
        "formal_comparison_authorized": False,
    }
    _write_json(evaluator_dir / "evaluator_contract.json", evaluator_contract, private=True)
    public_contract = dict(evaluator_contract)
    public_contract.pop("private_bundle_path", None)
    public_contract["private_bundle_digest"] = _digest({
        "hidden_manifest": hidden_manifest.get("manifest_digest"),
        "signature_audit": signature_audit.get("pass"),
    })
    _write_json(output_dir / "evaluator_contract.json", public_contract)

    structural_gates = {
        "final_id_clusters_at_least_40": len(benchmark.final_id_questions) >= 40,
        "final_ood_clusters_at_least_40": len(benchmark.final_ood_questions) >= 40,
        "whole_family_ood_holdout": set(q.family for q in benchmark.final_id_questions).isdisjoint(
            {q.family for q in benchmark.final_ood_questions}
        ),
        "signature_audit_pass": bool(signature_audit.get("pass") is True),
        "public_boundary_audit_pass": bool(public_audit.get("pass") is True),
        "public_manifest_locked": public_manifest.get("final_access", {}).get("locked") is True,
    }
    structural_gates["all_structural_gates_pass"] = bool(all(structural_gates.values()))
    public_freeze_audit = {
        "schema_version": "pesco_v05_frozen_final_audit_v0.1",
        "generator_version": V05_GENERATOR_VERSION,
        "evaluator_version": V05_EVALUATOR_VERSION,
        "counts_by_split": dict(public_manifest.get("counts_by_split", {})),
        "world_count": int(public_manifest.get("world_count", 0)),
        "structural_gates": structural_gates,
        "signature_summary": public_signature_summary,
        "public_manifest_audit": public_audit,
        "model_evaluation_completed": False,
        "formal_comparison_authorized": False,
    }
    _write_json(output_dir / "v05_freeze_audit.json", public_freeze_audit)
    _write_json(evaluator_dir / "v05_freeze_audit_hidden.json", {
        **public_freeze_audit,
        "hidden_manifest_digest": hidden_manifest.get("manifest_digest"),
        "hidden_signature_audit": signature_audit,
        "development_manifest_paths": [str(path) for path in dev_paths],
    }, private=True)

    freeze_receipt = build_freeze_receipt(
        repo_root=ROOT,
        benchmark=benchmark,
        signature_audit=signature_audit,
        public_audit=public_audit,
        baseline_selection=baseline_selection,
        explicit_sign=sign_freeze,
    )
    _write_json(output_dir / "freeze_receipt.json", freeze_receipt)
    _write_json(evaluator_dir / "freeze_receipt.json", freeze_receipt, private=True)

    # Source/data hashes bind the public receipt to the exact generator and runner.
    source_inventory = digest_paths(
        [
            ROOT / "research_strategy_optimization/evaluation/tier1_v05_frozen_final.py",
            ROOT / "scripts/prepare_tier1_v05_frozen_final.py",
            ROOT / "scripts/audit_tier1_v05_frozen_final.py",
            ROOT / "scripts/record_v05_baseline_selection.py",
        ], root=ROOT, role="source",
    )
    # ``independent_audit.json`` is a derived post-prepare report written by
    # ``write_v05_independent_audit.py``.  Exclude it from the prepare-time data
    # inventory so rerunning the independent auditor does not invalidate the
    # producer manifest; all benchmark/receipt inputs remain content-bound here.
    public_data_paths = [
        path for path in output_dir.iterdir()
        if path.name not in {"run_manifest.json", "independent_audit.json"}
    ]
    data_inventory = digest_paths(public_data_paths, root=ROOT, role="data")
    run_manifest = _lightweight_run_manifest(
        source_inventory=source_inventory,
        data_inventory=data_inventory,
        seeds={
            "exploration": list(V05_EXPLORATION_SEEDS),
            "confirmation": list(V05_CONFIRMATION_SEEDS),
        },
        status="prepared_locked" if freeze_receipt["status"] != "frozen" else "frozen",
        diagnostics={
            "source_inventory_digest": source_inventory.get("digest"),
            "evaluator_bundle_separate": True,
            "container_isolation_verified": False,
            "final_id_cluster_count": len(benchmark.final_id_questions),
            "final_ood_cluster_count": len(benchmark.final_ood_questions),
            "model_evaluation_completed": False,
            "formal_comparison_authorized": False,
        },
    )
    _write_json(output_dir / "run_manifest.json", run_manifest)
    private_run_manifest = {
        **run_manifest,
        "private_bundle_digest": _digest({
            "hidden_manifest": hidden_manifest.get("manifest_digest"),
            "freeze_receipt": freeze_receipt.get("receipt_digest"),
        }),
    }
    # Adding the private-bundle binding changes the manifest payload, so recompute
    # its canonical digest rather than copying the public digest verbatim.
    private_run_manifest["manifest_digest"] = _digest({
        key: value for key, value in private_run_manifest.items()
        if key != "manifest_digest"
    })
    _write_json(evaluator_dir / "run_manifest.json", private_run_manifest, private=True)
    return {
        "output": str(output_dir),
        "evaluator_output": str(evaluator_dir),
        "status": freeze_receipt["status"],
        "signed": bool(freeze_receipt["signed"]),
        "structural_gates": structural_gates,
        "signature_audit_pass": bool(signature_audit.get("pass") is True),
        "public_manifest_audit_pass": bool(public_audit.get("pass") is True),
        "environment_receipts_collected": environment_receipt is not None,
        "environment_question_count": (
            environment_receipt.get("question_count_collected", 0)
            if environment_receipt else 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/tier1_v05_frozen_final")
    parser.add_argument("--evaluator-output", default="artifacts/tier1_v05_evaluator_private")
    parser.add_argument(
        "--development-manifest", action="append", type=Path, default=None,
        help="hidden development manifest; may be supplied more than once",
    )
    parser.add_argument("--collect-environment-receipts", action="store_true")
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument(
        "--baseline-selection-file", type=Path, default=None,
        help=("pre-final receipt requiring selected_baseline, selection_split, "
              "development_manifest_digest, selection_receipt_digest, "
              "algorithm_config_digest, hyperparameters_digest, selection_results_digest, and "
              "algorithm_hyperparameters_frozen=true"),
    )
    parser.add_argument("--sign-freeze", action="store_true")
    parser.add_argument(
        "--require-frozen", action="store_true",
        help="return nonzero unless the strict clean/tag/baseline receipt is signed",
    )
    args = parser.parse_args(argv)
    baseline = _load_json(args.baseline_selection_file) if args.baseline_selection_file else None
    result = run(
        Path(args.output),
        Path(args.evaluator_output),
        development_manifests=tuple(args.development_manifest or ()),
        collect_environment=bool(args.collect_environment_receipts),
        question_limit=args.question_limit,
        baseline_selection=baseline,
        sign_freeze=bool(args.sign_freeze),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if args.require_frozen and not result["signed"]:
        return 2
    return 0 if result["structural_gates"].get("all_structural_gates_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

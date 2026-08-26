#!/usr/bin/env python3
"""Run a small Tier-1 data-generation/verifier smoke experiment.

This is intentionally separate from the 64-branch Tier-0 MVP.  It exercises the
NumPy-backed grouped-data adapter on explicit confounding and leakage worlds, proving
that the same validity gate catches attractive but invalid surface estimates and that
repair makes a subsequent run eligible for evidence classification.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.environments.tier0_simulator import TrustedVerifier, default_mvp_worlds
from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment
from research_strategy_optimization.schemas import Protocol, ResearchAction


def run(output: Path) -> dict:
    protocol = Protocol()
    base = list(default_mvp_worlds())
    worlds = [
        replace(base[0], world_id="tier1_supported"),
        replace(base[1], world_id="tier1_confounded", kind="invalid", confounding=True),
        replace(base[3], world_id="tier1_leakage", confounding=False),
    ]
    rows = []
    clone_type_preserved = True
    tier1_backend_seen = True
    repair_snapshots = {}
    for world in worlds:
        for action in ResearchAction.mvp_actions():
            env = Tier1TabularEnvironment(worlds=worlds, protocol=protocol)
            env.reset(question_id="rq_tier1_smoke", world_id=world.world_id, seed=17)
            before = env.snapshot()
            branch = env.clone_from_snapshot(before)
            clone_type_preserved = clone_type_preserved and isinstance(branch, Tier1TabularEnvironment)
            raw = branch.execute_option(action, seeds=protocol.exploration_seeds)
            verdict = TrustedVerifier(protocol).evaluate(raw, branch)
            tier1_backend_seen = tier1_backend_seen and raw.backend == Tier1TabularEnvironment.BACKEND
            rows.append({
                "world_id": world.world_id,
                "world_kind": world.kind,
                "action": action.value,
                "surface_effect": raw.effect_estimate,
                "validity_pass": verdict.validity_pass,
                "evidence_state": verdict.evidence_state.value,
                "confirmation_passed": verdict.independent_confirmation_passed,
                "confirmation_data_independent": verdict.confirmation_data_independent,
                "confirmation_dataset_hash": verdict.confirmation_dataset_hash,
                "confirmation_split_hash": verdict.confirmation_split_hash,
                "backend": raw.backend,
                "estimator": raw.estimator,
                "treatment_confounder_correlation": raw.treatment_confounder_correlation,
                "group_overlap_count": raw.group_overlap_count,
                "data_partition": raw.data_partition,
                "code_hash": raw.code_hash,
                "dataset_hash": raw.dataset_hash,
                "split_hash": raw.split_hash,
                "repair_or_sample_transition": False,
            })

        # Explicit repair transition for both invalid mechanisms.
        if world.world_id in {"tier1_confounded", "tier1_leakage"}:
            env = Tier1TabularEnvironment(worlds=worlds, protocol=protocol)
            env.reset(question_id="rq_tier1_smoke", world_id=world.world_id, seed=17)
            repaired = env.clone_from_snapshot(env.snapshot())
            clone_type_preserved = clone_type_preserved and isinstance(repaired, Tier1TabularEnvironment)
            raw = repaired.execute_option(ResearchAction.REPAIR, seeds=protocol.exploration_seeds)
            verdict = TrustedVerifier(protocol).evaluate(raw, repaired)
            tier1_backend_seen = tier1_backend_seen and raw.backend == Tier1TabularEnvironment.BACKEND
            repair_snapshots[world.world_id] = {
                "effect": raw.effect_estimate,
                "estimator": raw.estimator,
                "split_hash": raw.split_hash,
                "dataset_hash": raw.dataset_hash,
                "overlap": raw.group_overlap_count,
                "valid": verdict.validity_pass,
            }
            rows.append({
                "world_id": world.world_id,
                "world_kind": world.kind,
                "action": ResearchAction.REPAIR.value,
                "surface_effect": raw.effect_estimate,
                "validity_pass": verdict.validity_pass,
                "evidence_state": verdict.evidence_state.value,
                "confirmation_passed": verdict.independent_confirmation_passed,
                "confirmation_data_independent": verdict.confirmation_data_independent,
                "confirmation_dataset_hash": verdict.confirmation_dataset_hash,
                "confirmation_split_hash": verdict.confirmation_split_hash,
                "backend": raw.backend,
                "estimator": raw.estimator,
                "treatment_confounder_correlation": raw.treatment_confounder_correlation,
                "group_overlap_count": raw.group_overlap_count,
                "data_partition": raw.data_partition,
                "code_hash": raw.code_hash,
                "dataset_hash": raw.dataset_hash,
                "split_hash": raw.split_hash,
                "repair_or_sample_transition": True,
            })

    confounded_before = next(
        row for row in rows
        if row["world_id"] == "tier1_confounded" and row["action"] == ResearchAction.CONTINUE.value
    )
    leakage_before = next(
        row for row in rows
        if row["world_id"] == "tier1_leakage" and row["action"] == ResearchAction.CONTINUE.value
    )
    confounded_after = repair_snapshots["tier1_confounded"]
    leakage_after = repair_snapshots["tier1_leakage"]
    confirmation_rows = [row for row in rows if row["confirmation_passed"]]

    payload = {
        "schema_version": "pesco_tier1_smoke_v0.2",
        "diagnostic_only": True,
        "tier2_claim": False,
        "backend": Tier1TabularEnvironment.BACKEND,
        "clone_class": "Tier1TabularEnvironment",
        "rows": rows,
        "negative_controls": {
            "tier1_clone_preserves_subclass": clone_type_preserved,
            "tier1_numpy_backend_actually_executed": tier1_backend_seen,
            "confounding_invalid_before_repair": any(
                row["world_id"] == "tier1_confounded"
                and row["action"] == ResearchAction.CONTINUE.value
                and not row["validity_pass"]
                for row in rows
            ),
            "leakage_invalid_before_repair": any(
                row["world_id"] == "tier1_leakage"
                and row["action"] == ResearchAction.CONTINUE.value
                and not row["validity_pass"]
                for row in rows
            ),
            "repair_restores_validity": all(
                row["validity_pass"]
                for row in rows
                if row["repair_or_sample_transition"]
            ),
            "confounding_repair_changes_estimator": confounded_before["estimator"] != confounded_after["estimator"],
            "confounding_repair_reduces_bias": abs(confounded_after["effect"]) < abs(confounded_before["surface_effect"]),
            "leakage_repair_changes_data_protocol": leakage_before["split_hash"] != leakage_after["split_hash"],
            "leakage_repair_removes_group_overlap": leakage_before["group_overlap_count"] > leakage_after["overlap"],
            "confirmation_data_independent": all(row["confirmation_data_independent"] for row in confirmation_rows),
        },
    }
    payload["negative_controls"]["pass"] = all(bool(value) for value in payload["negative_controls"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "PESCO/artifacts/tier1_smoke.json")
    result = run(destination)
    print(json.dumps({"output": str(destination), "rows": len(result["rows"]), "negative_controls": result["negative_controls"]}, indent=2))

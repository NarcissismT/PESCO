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
    for world in worlds:
        for action in ResearchAction.mvp_actions():
            env = Tier1TabularEnvironment(worlds=worlds, protocol=protocol)
            env.reset(question_id="rq_tier1_smoke", world_id=world.world_id, seed=17)
            before = env.snapshot()
            branch = env.clone_from_snapshot(before)
            raw = branch.execute_option(action, seeds=protocol.exploration_seeds)
            verdict = TrustedVerifier(protocol).evaluate(raw, branch)
            rows.append({
                "world_id": world.world_id,
                "world_kind": world.kind,
                "action": action.value,
                "surface_effect": raw.effect_estimate,
                "validity_pass": verdict.validity_pass,
                "evidence_state": verdict.evidence_state.value,
                "confirmation_passed": verdict.independent_confirmation_passed,
                "repair_or_sample_transition": False,
            })

        # Explicit repair transition for both invalid mechanisms.
        if world.world_id in {"tier1_confounded", "tier1_leakage"}:
            env = Tier1TabularEnvironment(worlds=worlds, protocol=protocol)
            env.reset(question_id="rq_tier1_smoke", world_id=world.world_id, seed=17)
            repaired = env.clone_from_snapshot(env.snapshot())
            raw = repaired.execute_option(ResearchAction.REPAIR, seeds=protocol.exploration_seeds)
            verdict = TrustedVerifier(protocol).evaluate(raw, repaired)
            rows.append({
                "world_id": world.world_id,
                "world_kind": world.kind,
                "action": ResearchAction.REPAIR.value,
                "surface_effect": raw.effect_estimate,
                "validity_pass": verdict.validity_pass,
                "evidence_state": verdict.evidence_state.value,
                "confirmation_passed": verdict.independent_confirmation_passed,
                "repair_or_sample_transition": True,
            })

    payload = {
        "schema_version": "pesco_tier1_smoke_v0.1",
        "diagnostic_only": True,
        "tier2_claim": False,
        "rows": rows,
        "negative_controls": {
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
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "PESCO/artifacts/tier1_smoke.json")
    result = run(destination)
    print(json.dumps({"output": str(destination), "rows": len(result["rows"]), "negative_controls": result["negative_controls"]}, indent=2))

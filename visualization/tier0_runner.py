"""Run the repository's deterministic Tier-0 simulator and emit report rows.

Unlike ``demo.py`` (which fabricates policy outcomes), this runner executes the
actual four-world environment and trusted verifier from
``research_strategy_optimization``. It is still a pilot diagnostic: the
``task_utility`` field uses the preregistered MVP oracle action only to quantify
branch quality after execution, and must not be used as a training label for a
policy without the planned masking/credit-allocation protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    # When invoked as ``python -m visualization.tier0_runner`` with ``PESCO``
    # on PYTHONPATH, visualization is a top-level package.
    from research_strategy_optimization.environments.tier0_simulator import (
        Tier0ResearchEnvironment,
        TrustedVerifier,
        default_mvp_worlds,
    )
    from research_strategy_optimization.schemas import Protocol, ResearchAction
except ImportError:  # pragma: no cover - package-style invocation fallback
    from ..research_strategy_optimization.environments.tier0_simulator import (
        Tier0ResearchEnvironment,
        TrustedVerifier,
        default_mvp_worlds,
    )
    from ..research_strategy_optimization.schemas import Protocol, ResearchAction


WORLD_STATE = {
    "supported": "Supported",
    "refuted": "Refuted",
    "insufficient": "Insufficient",
    "invalid": "Invalid",
}
BEST_ACTION = {
    "supported": ResearchAction.CONTINUE.value,
    "refuted": ResearchAction.SWITCH.value,
    "insufficient": ResearchAction.SAMPLE.value,
    "invalid": ResearchAction.REPAIR.value,
}


def run_tier0_records(
    *,
    question_id: str = "rq_tier0_001",
    protocol: Optional[Protocol] = None,
    world_ids: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Execute 4 worlds × 4 MVP actions × 4 seeds by default."""
    protocol = protocol or Protocol()
    exploration_seeds = tuple(int(seed) for seed in (seeds or protocol.exploration_seeds))
    worlds = list(default_mvp_worlds())
    if world_ids is not None:
        allowed = set(world_ids)
        worlds = [world for world in worlds if world.world_id in allowed]
    actions = ResearchAction.mvp_actions()
    records: List[Dict[str, Any]] = []
    for world in worlds:
        state = WORLD_STATE[world.kind]
        env = Tier0ResearchEnvironment(worlds=worlds, protocol=protocol)
        env.reset(question_id=question_id, world_id=world.world_id, seed=world.seed_offset)
        snapshot = env.snapshot()
        for action in actions:
            for seed in exploration_seeds:
                branch = env.clone_from_snapshot(snapshot)
                output = branch.execute_option(action, seeds=(seed,))
                verdict = TrustedVerifier(protocol).evaluate(output, branch)
                predicted = verdict.evidence_state.value.title()
                action_value = action.value
                action_correct = action_value == BEST_ACTION[world.kind]
                state_correct = predicted == state
                confirmed = bool(verdict.independent_confirmation_passed)
                valid_claim = bool(verdict.validity_pass and verdict.scientific_claim_consistency)
                records.append({
                    "record_type": "tier0_branch",
                    "schema_version": "pesco_results_v0.1",
                    "method": "Tier0-BranchRollout",
                    "split": "pilot",
                    "question_id": question_id,
                    "world_id": world.world_id,
                    "world_kind": world.kind,
                    "world_pair_id": "pair_supported_refuted" if world.kind in {"supported", "refuted"} else "pair_insufficient_invalid",
                    "snapshot_id": snapshot.digest,
                    "branch_id": f"{world.world_id}:{action_value}:{seed}",
                    "seed": seed,
                    "true_state": state,
                    "predicted_state": predicted,
                    "selected_action": action_value,
                    "valid_claim": valid_claim,
                    "belief_score": 1.0 if state_correct else 0.0,
                    "task_utility": 1.0 if action_correct else 0.0,
                    "replication_utility": 1.0 if confirmed else 0.0,
                    "discovery_utility": 0.0,
                    "cost": float(verdict.execution_cost),
                    "utility": (1.0 if action_correct else 0.0) + (0.25 if state_correct else 0.0),
                    "switch": action is ResearchAction.SWITCH,
                    "switch_beneficial": action is ResearchAction.SWITCH and action_correct,
                    "effective_switch": action is ResearchAction.SWITCH and action_correct,
                    "unnecessary_switch": action is ResearchAction.SWITCH and not action_correct,
                    "persisted": action is ResearchAction.CONTINUE,
                    "current_strategy_optimal": action_correct and action is ResearchAction.CONTINUE,
                    "persistence_correct": action_correct and action is ResearchAction.CONTINUE,
                    "refutation_accept": state == "Refuted" and predicted == "Refuted",
                    "underpower_handled": state != "Insufficient" or action is ResearchAction.SAMPLE,
                    "invalid_repaired": state == "Invalid" and action is ResearchAction.REPAIR and valid_claim,
                    "invalid_claim": state == "Invalid" and predicted == "Supported",
                    "independent_confirmed": confirmed,
                    "entered_confirmation": bool(verdict.independent_confirmation_performed),
                    "new_path_verified": False,
                    "turn": 1,
                    "effect_estimate": float(output.effect_estimate),
                    "ci_low": float(output.ci_low),
                    "ci_high": float(output.ci_high),
                    "sample_size": int(output.sample_size),
                    "seed_count": int(output.seed_count),
                    "validity_signals": list(output.validity_signals),
                    "audit_signature": verdict.audit_signature,
                })
    return records


def write_tier0_results(destination: str | Path, **kwargs: Any) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = run_tier0_records(**kwargs)
    payload = {
        "schema_version": "pesco_results_v0.1",
        "source": "PESCO.research_strategy_optimization.environments.tier0_simulator",
        "synthetic_pilot": True,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run actual PESCO Tier-0 branches")
    parser.add_argument("output", nargs="?", default="tier0_results.json")
    args = parser.parse_args()
    print(write_tier0_results(args.output))

"""Deterministic Tier-0 style data for smoke-testing the report pipeline.

The demo is deliberately synthetic and must never be presented as a real
scientific result.  It mirrors the four minimum-pilot worlds in plan §25 and
includes a paired-world preference reversal for PESCO.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


WORLDS = {
    "supported": {"true_state": "Supported", "best_action": "continue_current_method", "effect": 0.16, "leakage": False},
    "refuted": {"true_state": "Refuted", "best_action": "switch_to_alternative_method", "effect": -0.01, "leakage": False},
    "insufficient": {"true_state": "Insufficient", "best_action": "add_samples_or_seeds", "effect": 0.01, "leakage": False},
    "invalid": {"true_state": "Invalid", "best_action": "repair_data_split", "effect": 0.35, "leakage": True},
}

ACTIONS = ["continue_current_method", "add_samples_or_seeds", "repair_data_split", "switch_to_alternative_method"]


def _action_score(world: Dict[str, Any], action: str, rng: random.Random) -> float:
    base = {
        "continue_current_method": 0.58,
        "add_samples_or_seeds": 0.55,
        "repair_data_split": 0.48,
        "switch_to_alternative_method": 0.52,
    }[action]
    if action == world["best_action"]:
        base += 0.25
    if world["true_state"] == "Invalid" and action == "continue_current_method":
        base += 0.22  # surface metric is high, trusted validity gate should reject it
    return max(0.01, min(0.99, base + rng.gauss(0, 0.025)))


def generate_demo_records(seed: int = 17, *, questions: int = 8) -> List[Dict[str, Any]]:
    """Return records suitable for ``python -m PESCO.visualization --demo``."""
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    methods = ["Base", "GRPO-FourState", "PESCO-Offline", "PESCO-Full"]
    split_by_q = {q: ("final_ood" if q >= max(1, questions - 2) else "final_id") for q in range(questions)}
    for q in range(questions):
        world_name = list(WORLDS)[q % len(WORLDS)]
        world = WORLDS[world_name]
        for method in methods:
            for seed_offset in range(4):
                local = random.Random(seed * 10000 + q * 100 + seed_offset * 7 + len(method))
                if method == "PESCO-Full":
                    action = world["best_action"]
                    predicted = world["true_state"]
                    if local.random() < 0.07:
                        action = rng.choice(ACTIONS)
                    if local.random() < 0.06:
                        predicted = rng.choice([state for state in ("Supported", "Refuted", "Insufficient", "Invalid") if state != predicted])
                elif method == "PESCO-Offline":
                    action = world["best_action"] if local.random() < 0.68 else "continue_current_method"
                    predicted = world["true_state"] if local.random() < 0.78 else "Insufficient"
                elif method == "GRPO-FourState":
                    action = "continue_current_method" if world_name in {"supported", "invalid"} else "add_samples_or_seeds"
                    predicted = world["true_state"] if local.random() < 0.58 else "Supported"
                else:
                    action = "continue_current_method"
                    predicted = "Supported" if local.random() < 0.8 else "Insufficient"
                switched = action == "switch_to_alternative_method"
                current_optimal = world["best_action"] == "continue_current_method"
                effective_switch = switched and action == world["best_action"]
                invalid_repaired = world_name == "invalid" and action == "repair_data_split"
                valid_claim = not (world_name == "invalid" and predicted == "Supported")
                replication = bool(local.random() < ({"PESCO-Full": .92, "PESCO-Offline": .81, "GRPO-FourState": .67, "Base": .6}[method]))
                # The demo uses the same fixed four-action MVP as the pilot.  A
                # method-name-specific discovery bonus would make this visualization
                # smoke test encode an unfair comparison, so discovery is disabled
                # for every method until an open-ended candidate protocol exists.
                discovery = False
                announced = False
                # Synthetic paired-world annotation used only to exercise the
                # FlipAcc chart.  A real runner should emit this field only
                # after the paired-world reversal has passed its statistical
                # confirmation gate.
                flip_correct = bool(
                    method == "PESCO-Full"
                    and world_name in {"supported", "refuted"}
                    and local.random() < .9
                )
                cost = {"Base": 1.0, "GRPO-FourState": 1.3, "PESCO-Offline": 1.55, "PESCO-Full": 1.9}[method] * (1 + .06 * seed_offset)
                budget_profile = {
                    "Base": {"cost_teacher": 0.0, "cost_training": 0.20, "cost_rollout": 0.50, "cost_verification": 0.15, "cost_confirmation": 0.15},
                    "GRPO-FourState": {"cost_teacher": 0.05, "cost_training": 0.28, "cost_rollout": 0.62, "cost_verification": 0.18, "cost_confirmation": 0.17},
                    "PESCO-Offline": {"cost_teacher": 0.08, "cost_training": 0.32, "cost_rollout": 0.73, "cost_verification": 0.20, "cost_confirmation": 0.19},
                    "PESCO-Full": {"cost_teacher": 0.10, "cost_training": 0.40, "cost_rollout": 0.85, "cost_verification": 0.24, "cost_confirmation": 0.22},
                }[method]
                utility = _action_score(world, action, local)
                records.append({
                    "record_type": "episode",
                    "method": method,
                    "split": split_by_q[q],
                    "question_id": f"rq_{q:03d}",
                    "world_id": f"world_{world_name}_{q:03d}",
                    "world_pair_id": f"pair_{q // len(WORLDS):03d}",
                    "world_state": world["true_state"],
                    "true_state": world["true_state"],
                    "predicted_state": predicted,
                    "selected_action": action,
                    "switch": switched,
                    "switch_beneficial": effective_switch,
                    "effective_switch": effective_switch,
                    "unnecessary_switch": switched and not effective_switch,
                    "flip_correct": flip_correct,
                    "persisted": action == "continue_current_method",
                    "current_strategy_optimal": current_optimal,
                    "persistence_correct": action == "continue_current_method" and current_optimal,
                    "refutation_accept": world_name == "refuted" and predicted == "Refuted",
                    "underpower_handled": world_name != "insufficient" or action in {"add_samples_or_seeds", "stop_and_report_insufficient", "stop"},
                    "invalid_repaired": invalid_repaired,
                    "invalid_claim": world_name == "invalid" and predicted == "Supported",
                    "flip_eligible": world_name in {"supported", "refuted"},
                    "required_switch": world_name == "refuted",
                    "switch_required": world_name == "refuted",
                    "invalid_repair_eligible": world_name == "invalid",
                    "invalid_initial": world_name == "invalid",
                    "insufficient_handling_eligible": world_name == "insufficient",
                    "insufficient_initial": world_name == "insufficient",
                    "confirmation_eligible": world_name in {"supported", "refuted"},
                    "valid_claim": valid_claim,
                    "new_path_verified": discovery,
                    "new_path_announced": announced,
                    "candidate_group_id": f"discovery_{q:03d}_{seed_offset}",
                    "candidate_rank": {"PESCO-Full": 1, "PESCO-Offline": 2, "GRPO-FourState": 3, "Base": 4}[method],
                    "candidate_success": discovery,
                    "candidate_utility": 1.0 if discovery else utility,
                    "discovery_opportunity": False,
                    "discovery_eligible": False,
                    "discovery_bonus_policy": "disabled_fixed_action_space",
                    "independent_confirmed": replication,
                    "entered_confirmation": True,
                    "belief_score": 0.93 if predicted == world["true_state"] else 0.25,
                    "hypothesis_probability": 0.93 if predicted == "Supported" else 0.25,
                    "task_utility": utility,
                    "replication_utility": 1.0 if replication else 0.0,
                    "discovery_utility": 1.0 if discovery else 0.0,
                    "cost": cost,
                    "utility": utility,
                    "turn": seed_offset + 1,
                    "branch_id": action,
                    "inference_mode": "fixed_branch" if method == "GRPO-FourState" else "single_path",
                    **budget_profile,
                })
    return records


def write_demo(path: str | Path, seed: int = 17, *, questions: int = 8) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": "pesco_results_v0.1", "records": generate_demo_records(seed, questions=questions)}, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="demo_results.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(write_demo(args.output, args.seed))

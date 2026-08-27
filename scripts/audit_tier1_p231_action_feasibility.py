#!/usr/bin/env python3
"""Audit the P2.3.1 action-feasibility and tie-set benchmark contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_p23_dataset import build_tier1_p23_promotion_v2_benchmark
from research_strategy_optimization.evaluation.tier1_p23_v3_dataset import build_tier1_p23_promotion_v3_benchmark
from research_strategy_optimization.evaluation.tier1_v04_extended import TrustedVerifier, V04_EXTENDED_CONFIRMATION_SEEDS, V04_EXTENDED_EXPLORATION_SEEDS
from research_strategy_optimization.schemas import EvidenceState, Protocol, ResearchAction


def run(output: Path, *, tolerance: float = 0.02, benchmark_name: str = "v2") -> dict:
    benchmark = build_tier1_p23_promotion_v2_benchmark() if benchmark_name == "v2" else build_tier1_p23_promotion_v3_benchmark()
    protocol = Protocol(
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    verifier = TrustedVerifier(protocol)
    actions = tuple(ResearchAction.all_actions())
    rows = []
    counters = Counter()
    by_split = Counter()
    by_family = Counter()
    for question in benchmark.questions:
        for world in question.worlds:
            env = benchmark.make_environment(question.question_id, protocol=protocol)
            env.reset(question.policy_question_id, world.world_id, seed=17)
            baseline = env.execute_option(ResearchAction.CONTINUE, seeds=protocol.exploration_seeds)
            baseline_verdict = verifier.evaluate(baseline, env, confirm=False)
            snapshot = env.snapshot()
            outcomes = {}
            valid_actions = []
            utilities = {}
            for action in actions:
                branch = env.clone_from_snapshot(snapshot)
                output_row = branch.execute_option(action, seeds=protocol.exploration_seeds)
                verdict = verifier.evaluate(output_row, branch, confirm=False)
                valid = bool(verdict.validity_pass)
                outcomes[action.value] = {
                    "validity_pass": valid,
                    "state": verdict.evidence_state.value,
                    "invalid_reasons": list(verdict.invalid_reasons),
                    "effect_estimate": float(output_row.effect_estimate),
                    "execution_cost": float(output_row.execution_cost),
                }
                if valid:
                    valid_actions.append(action.value)
                # A conservative action utility used only for tie/non-tie auditing;
                # formal rewards remain the evaluator's atomic receipts.
                utilities[action.value] = float(
                    (0.25 if valid else -0.30)
                    + (0.20 if verdict.independent_confirmation_passed else 0.0)
                    - 0.03 * float(output_row.execution_cost)
                )
            # The promotion action metric remains defined on the four registered
            # policy actions.  STOP/REVISE/REDESIGN are executable safety actions,
            # but including STOP's zero-cost receipt in the utility ranking would
            # manufacture a non-tie for every world.
            metric_actions = tuple(ResearchAction.mvp_actions())
            ordered = sorted((utilities[action.value] for action in metric_actions), reverse=True)
            gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
            non_tie = gap > float(tolerance)
            target = question.target_action(world.world_id)
            target_valid = bool(outcomes.get(target.value, {}).get("validity_pass", False))
            target_not_repair = target is not ResearchAction.REPAIR
            repair_outcome = outcomes.get(ResearchAction.REPAIR.value, {})
            switch_outcome = outcomes.get(ResearchAction.SWITCH.value, {})
            sample_outcome = outcomes.get(ResearchAction.SAMPLE.value, {})
            row = {
                "question_id": question.question_id,
                "world_id": world.world_id,
                "split": question.split,
                "family": question.family,
                "kind": world.kind,
                "initial_state": baseline_verdict.evidence_state.value,
                "audit_target_action": target.value,
                "valid_actions": valid_actions,
                "all_actions_invalid": not bool(valid_actions),
                "target_action_valid": target_valid,
                "necessary_switch": target is ResearchAction.SWITCH,
                "necessary_switch_valid": target is ResearchAction.SWITCH and bool(switch_outcome.get("validity_pass", False)),
                "successful_repair": target is ResearchAction.REPAIR and bool(repair_outcome.get("validity_pass", False)),
                "harmful_or_unnecessary_repair": target_not_repair and (not bool(repair_outcome.get("validity_pass", False)) or float(utilities[ResearchAction.REPAIR.value]) < max(utilities.values())),
                "insufficient_handling": (target is ResearchAction.SAMPLE or baseline_verdict.evidence_state is EvidenceState.INSUFFICIENT),
                "insufficient_handling_valid": (target is ResearchAction.SAMPLE or baseline_verdict.evidence_state is EvidenceState.INSUFFICIENT) and bool(sample_outcome.get("validity_pass", False)),
                "top1_minus_top2_gap": gap,
                "non_tie": non_tie,
                "utilities_for_tie_audit": {action.value: utilities[action.value] for action in metric_actions},
                "outcomes": outcomes,
            }
            rows.append(row)
            counters["world_count"] += 1
            counters["all_actions_invalid_n"] += int(row["all_actions_invalid"])
            counters["target_action_valid_n"] += int(target_valid)
            counters["necessary_switch_n"] += int(row["necessary_switch"])
            counters["necessary_switch_valid_n"] += int(row["necessary_switch_valid"])
            counters["successful_repair_n"] += int(row["successful_repair"])
            counters["harmful_or_unnecessary_repair_n"] += int(row["harmful_or_unnecessary_repair"])
            counters["insufficient_handling_n"] += int(row["insufficient_handling"])
            counters["insufficient_handling_valid_n"] += int(row["insufficient_handling_valid"])
            counters["non_tie_n"] += int(non_tie)
            counters["tie_n"] += int(not non_tie)
            by_split[question.split] += 1
            by_family[question.family] += 1
    out = {
        "schema_version": "pesco_tier1_p231_action_feasibility_audit_v0.1",
        "benchmark": benchmark_name,
        "action_catalogue": [action.value for action in actions],
        "tolerance": float(tolerance),
        "non_tie_definition": "top1_minus_top2 > tolerance",
        "required_minima": {
            "necessary_switch_n": 20,
            "successful_repair_n": 20,
            "harmful_or_unnecessary_repair_n": 20,
            "insufficient_handling_n": 20,
            "insufficient_handling_valid_n": 20,
            "nonzero_safety_denominators": True,
        },
        "counts": dict(counters),
        "counts_by_split": dict(by_split),
        "counts_by_family": dict(by_family),
        "pass": bool(
            counters["all_actions_invalid_n"] == 0
            and counters["necessary_switch_n"] >= 20
            and counters["successful_repair_n"] >= 20
            and counters["harmful_or_unnecessary_repair_n"] >= 20
            and counters["insufficient_handling_n"] >= 20
            and counters["insufficient_handling_valid_n"] >= 20
            and counters["non_tie_n"] > 0
            and counters["tie_n"] > 0
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--benchmark", choices=("v2", "v3"), default="v2")
    args = parser.parse_args(argv)
    result = run(args.output, tolerance=args.tolerance, benchmark_name=args.benchmark)
    print(json.dumps({"pass": result["pass"], "counts": result["counts"]}, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

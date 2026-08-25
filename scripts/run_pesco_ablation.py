#!/usr/bin/env python3
"""Run one transparent CPU ablation or emit its preregistered status.

Only mechanisms that have a faithful tabular switch are executed here.  The remaining
ablation names are recorded as reserved rather than being replaced with fabricated
numbers; formal comparison still requires the multi-question frozen splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PESCO_ROOT = ROOT / "PESCO"
if str(PESCO_ROOT) not in sys.path:
    sys.path.insert(0, str(PESCO_ROOT))

from research_strategy_optimization.algorithms.pesco_trainer import PESCOTrainer, TrainerConfig
from research_strategy_optimization.environments.tier0_simulator import Tier0ResearchEnvironment, default_mvp_worlds
from research_strategy_optimization.evaluation.ablations import CORE_ABLATIONS
from research_strategy_optimization.schemas import Protocol


def run(name: str, output: Path, epochs: int = 3) -> dict:
    specs = {spec.name: spec for spec in CORE_ABLATIONS}
    if name not in specs:
        raise ValueError(f"unknown ablation {name!r}; choose one of {sorted(specs)}")
    runnable = {
        "No-PairedWorld": {"use_paired_world": False},
        "No-FlipLoss": {"use_flip_loss": False},
        "No-Branch": {"use_branch_advantage": False},
        "No-ValidityGate": {"use_validity_gate": False},
    }
    spec = specs[name]
    payload = {"schema_version": "pesco_ablation_result_v0.1", "spec": spec.to_dict()}
    if name not in runnable:
        payload.update({"status": "registered_not_run_on_final_splits", "training_log": None})
    else:
        config = TrainerConfig(epochs=max(1, int(epochs)), **runnable[name])
        trainer = PESCOTrainer(config=config)
        worlds = list(default_mvp_worlds())
        trainer.fit(lambda: Tier0ResearchEnvironment(protocol=Protocol()), [world.world_id for world in worlds], "rq_tier0_001")
        payload.update({"status": "cpu_reference_ablation_executed", "training_log": trainer.log.to_dict(), "config": config.__dict__})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[spec.name for spec in CORE_ABLATIONS])
    parser.add_argument("--output", "-o", default="PESCO/artifacts/ablation.json")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args(argv)
    result = run(args.name, Path(args.output), args.epochs)
    print(json.dumps({"output": args.output, "name": args.name, "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

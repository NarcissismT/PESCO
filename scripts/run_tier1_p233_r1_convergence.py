#!/usr/bin/env python3
"""Run the frozen-config 64..1024 convergence ladder in isolated processes."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=(64,128,256,512,1024))
    p.add_argument("--seeds", type=int, nargs="+", default=(17,23,29))
    p.add_argument("--sft-steps", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--branch-loss-weight", type=float, default=0.30)
    p.add_argument("--pairwise-weight", type=float, default=0.10)
    p.add_argument("--flip-reference-kl-weight", type=float, default=0.5)
    p.add_argument("--gradient-mode", choices=("sum", "pcgrad"), default="sum")
    p.add_argument("--pcgrad-aux-pair", action="store_true")
    p.add_argument("--pcgrad-aux-orthogonal", action="store_true")
    p.add_argument("--branch-head-isolated", action="store_true")
    p.add_argument("--flip-head-isolated", action="store_true")
    p.add_argument("--branch-formulation", choices=("sibling_advantage", "expected_utility", "utility_cross_entropy", "soft_utility_cross_entropy", "utility_improvement_soft_ce", "top1_hinge"), default="expected_utility")
    p.add_argument("--utility-target-weight", type=float, default=0.25)
    p.add_argument("--atomic-target-weight", type=float, default=0.0)
    p.add_argument("--state-weight", type=float, default=0.2)
    p.add_argument("--branch-trust-region", action="store_true")
    p.add_argument("--branch-trust-epsilon", type=float, default=0.005)
    args = p.parse_args(argv); args.root.mkdir(parents=True, exist_ok=True)
    methods=("GRPO-Atomic", "Atomic+Branch", "PESCO-Full")
    for step in args.steps:
        for seed in args.seeds:
            out = args.root / f"steps_{int(step)}_seed{int(seed)}"
            marker = out / "p231_result.json"
            if marker.exists():
                continue
            cmd=[sys.executable, str(ROOT/"scripts/run_tier1_p233_chunk.py"), "--dataset", str(args.dataset), "--output-dir", str(out), "--seed", str(seed), "--methods", *methods, "--finetune-steps", str(step), "--sft-steps", str(args.sft_steps), "--hidden-dim", str(args.hidden_dim), "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate), "--branch-loss-weight", str(args.branch_loss_weight), "--pairwise-weight", str(args.pairwise_weight), "--flip-reference-kl-weight", str(args.flip_reference_kl_weight), "--state-weight", str(args.state_weight), "--utility-target-weight", str(args.utility_target_weight), "--atomic-target-weight", str(args.atomic_target_weight), "--gradient-mode", str(args.gradient_mode), "--branch-formulation", str(args.branch_formulation)] + (["--pcgrad-aux-pair"] if args.pcgrad_aux_pair else []) + (["--pcgrad-aux-orthogonal"] if args.pcgrad_aux_orthogonal else []) + (["--branch-head-isolated"] if args.branch_head_isolated else []) + (["--flip-head-isolated"] if args.flip_head_isolated else []) + (["--branch-trust-region", "--branch-trust-epsilon", str(args.branch_trust_epsilon)] if args.branch_trust_region else [])
            print("running", step, seed, flush=True); subprocess.run(cmd, check=True)
    assemble=[sys.executable, str(ROOT/"scripts/assemble_tier1_p233_convergence.py"), "--root", str(args.root), "--output", str(args.output), "--steps", *[str(x) for x in args.steps], "--seeds", *[str(x) for x in args.seeds]]
    subprocess.run(assemble, check=True); return 0

if __name__ == "__main__":
    raise SystemExit(main())

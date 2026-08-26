# Tier-1 formal final-ID/final-OOD P2 ten-seed experiment

This is the complete preregistered CPU P2 run on the locked formal environment
dataset. It uses 10 training seeds, 64 optimizer steps, 8 epochs, promotion split
`final_id`, and held-out split `final_ood`. The dataset contains evaluator-side
atomic reward receipts; public exports remain sanitized.

The promotion result is **NO-GO**:

- primary normalized-regret delta: `0.0074862`, 95% bootstrap CI
  `[0.0006630, 0.0148329]`;
- held-out same-question FlipAcc delta: `-0.0333333`, 95% CI
  `[-0.0482456, -0.0166667]`;
- atomic-receipt reward perturbation winner stability: `0.9899457` over 368
  non-tied examples (100 perturbations per example), passing its 0.90 gate;
- regret direction, held-out FlipAcc, safety, and the ≥8/10 direction gates do
  not all pass.

Therefore formal model comparison, P3 LoRA/QLoRA, online RL, and 7B evaluation
remain locked. See `p2_result.json`, `p2_gate.json`, and `run_manifest.json`.

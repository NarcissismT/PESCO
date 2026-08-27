# P2.1 bounded seed sweep diagnostic

This directory records an isolated CPU supplement to the main P2.1 algorithm
diagnostic.  Twelve worker processes (seeds 17, 23, and 29 × SFT,
PESCO-BranchOnly, PESCO-NoFlipLoss, and PESCO-Full) were run independently with
32 optimizer steps and four epochs.  Each worker evaluates the untouched fresh
train/tune/promotion dataset; no final v0.4/v0.5 artifact is read for training or
baseline selection.

The aggregate uses seed resampling followed by question-cluster resampling and
reports mechanism-family leave-one-out intervals.  Results are diagnostic-only:

- Full minus the tune-selected baseline (all three seeds selected
  `PESCO-BranchOnly`) normalized-regret delta: point `-0.0167`, 95% CI
  `[-0.0506, 0.0000]`.
- Full minus NoFlipLoss promotion PairRankAcc delta: point `+0.1284`, 95% CI
  `[+0.0766, +0.1779]`.
- Worker failures: `0`.

These three seeds are a bounded stability check, not the preregistered 10-seed
promotion gate.  The result cannot authorize a clean final, LoRA/QLoRA, 7B, or
online-RL comparison.

Machine-readable outputs:

- `seed_sweep_result.json` — aggregate estimates and family leave-one-out rows.
- `run_manifest.json` — command, source/data digests, seeds, and dirty-worktree
  provenance.
- `workers/` — one auditable `p21_result.json` per method/seed worker.

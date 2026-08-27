# Tier-1 P2 ten-seed CPU promotion experiment

> **Consumption status (2026-08-26): `diagnostic_v04_consumed`.**  This is a
> historical audit of the v0.4 dataset.  Read
> [`consumption_notice.json`](consumption_notice.json) before interpreting any
> metric; the original result/manifest bytes remain unchanged, and this run is
> not valid for post-fix algorithm selection or a paper final comparison.

This is the canonical P2 result on the frozen v0.4 extended raw-evidence dataset,
including the evaluator-side atomic reward receipts used by the perturbation gate.
Seven registered training methods and the evaluator-only inference-time branch
search were run for ten independent seeds under the same optimizer budget.

The pre-registered promotion result is **NO-GO**:

- primary normalized-regret delta: `-0.0161708`, 95% bootstrap CI
  `[-0.0286126, -0.0015162]`;
- held-out same-question FlipAcc delta: `-0.0580153`, 95% CI
  `[-0.0610687, -0.0541985]`;
- atomic-receipt reward perturbation winner stability: `0.991875` over 256
  non-tied examples (100 perturbations per example);
- ten-seed, preregistered-budget, regret, reward-stability, beats-SFT, and
  pair/cluster gates pass, but held-out FlipAcc, 8/10 same-direction, and the
  false-discovery safety gate do not.

Consequently this result does not authorize P3, online RL, or a 7B/QLoRA claim.
See `p2_result.json`, `p2_gate.json`, and `run_manifest.json`.

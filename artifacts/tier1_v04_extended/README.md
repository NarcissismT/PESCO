# Tier-1 v0.4 extended hardening diagnostic

This is the canonical CPU diagnostic artifact for the feedback coverage envelope.
It does not make a Tier-2/LLM claim, but its pre-registered dev and diagnostic-OOD
formal pair/cluster checks are OPEN:

- 8 mechanism families x 8 question variants = 64 independent questions and
  256 paired-world records;
- 8 exploration seeds;
- supported/refuted/insufficient/invalid worlds are executed for every question;
  451 confirmed same-question reversals are retained (142 dev, 131 diagnostic-OOD,
  178 train);
- 4 explicit composite mechanism families;
- train/dev/diagnostic-OOD question clusters (24/20/20), with final ID/OOD model
  evaluation still closed;
- posterior expected utility + value-of-information decisions on diagnostic
  train/dev paired worlds; and
- two-step re-plan trajectories for both oracle_state and raw_evidence.

The canonical run includes 64 posterior-EU/VOI decisions and 64 diagnose--retest--
replicate trajectories on one variant-3 anchor per mechanism family (both
`oracle_state` and `raw_evidence` tracks).  Selection is family round-robin so the
causal hidden-method-B anchor and all three composite OOD families are covered.
`decisions.json` and `trajectories.json` are evaluator-only audit records; they are
not policy inputs and contain no public target-action table.

Run:

PYTHONPATH=. python scripts/run_tier1_v04_extended.py --output artifacts/tier1_v04_extended

tier1_v04_extended_go.json is a structural/diagnostic GO record only:
`formal_gate_status=OPEN` for the CPU dev/OOD reversal checks,
`formal_comparison_authorized=false`, `formal_final_access=false`, and all
Tier-2/LLM claims remain false.

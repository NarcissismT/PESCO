# Tier-1 v0.4 benchmark-hardening diagnostic

This directory is a quick, machine-readable diagnostic for the feedback hardening
pass.  It reuses the executable NumPy worlds from Tier-1 v0.3 but changes the
decision/evaluation boundary:

- `oracle_state` is an evaluator-side upper-bound track that supplies the trusted
  initial evidence state.
- `raw_evidence` withholds that state and uses only public/raw observation and
  experiment-output fields.
- Both tracks use a uniform, same-family, leave-one-question-out latent candidate
  bank.  The current question is excluded from its own posterior, so the hidden
  method-B effect in causal-confounding variant 3 cannot become a hindsight target.
- `posterior_expected_utility` and `value_of_information` are recorded for every
  question/world/track.  `posterior_optimal_action` is derived from those values;
  the legacy v0.3 target table is audit-only and never consumed.

The artifact is diagnostic-only.  `tier1_v04_go.json` keeps formal comparison,
final ID/OOD access, Tier-2, and LLM claims closed.  Recreate it with:

```bash
PYTHONPATH=. python scripts/run_tier1_v04.py --output artifacts/tier1_v04
```

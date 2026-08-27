# Tier-1 v0.5 frozen-final boundary

This directory is the public side of the v0.5 final boundary.  It contains only
opaque question commitments, split counts, source/evaluator digests, and fail-closed
status receipts.  World parameters, mechanism labels, target actions, and latent
generator recipes are kept in the separate evaluator bundle
`artifacts/tier1_v05_evaluator_private/` and must not be published with a paper.

The prepared profile contains 48 final-ID and 48 final-OOD question clusters (384
worlds).  The OOD side is a whole-family holdout with four mechanism families absent
from the v0.4 development registry.  `signature_audit_summary.json` records the
question-ID, world-ID, latent-output, and generator-signature overlap checks.

The current `freeze_receipt.json` is intentionally `pending_clean_commit_tag`: this
working tree has not supplied a pre-final baseline selection and does not have a
clean tagged HEAD.  The independent evaluator script therefore cannot sign a final
freeze or authorize model comparison.  The evaluator bundle contains a complete CPU
transition/confirmation receipt collection (12,288 action×seed rows); these are
environment receipts only and are not a model evaluation.

A signable baseline receipt must additionally include digests for a non-empty
development manifest, the actual dev candidate-metrics/selection-results file, the
algorithm configuration, and the frozen hyperparameters, plus
`algorithm_hyperparameters_frozen: true`.  The receipt digest and all four file
hashes are recomputed at freeze time; invented hash strings and reserved empty dev
placeholders fail closed.

To re-audit without invoking the generator:

```bash
PYTHONPATH=. python scripts/audit_tier1_v05_frozen_final.py \
  --public artifacts/tier1_v05_frozen_final \
  --evaluator artifacts/tier1_v05_evaluator_private
```

# Tier-1 v0.3 benchmark artifact

This directory is a diagnostic environment artifact, not a formal model-comparison
result. It contains 12 independent question instances (three variants in each of
four mechanism families), 48 hidden worlds, 48 same-state branch groups (192
action-level rows), and 768 seed-level exploration observations (`12 × 4 × 4 × 4`).

The four families are:

- `group_leakage`
- `causal_confounding`
- `low_sample_variance`
- `subgroup_metric_mismatch`

The action target table is evaluator-side only. `tier1_scientific_utility` computes
branch utility from the generated output, trusted verifier, confirmation, protocol
repair, and cost; the target table is used only for external agreement/regret audits.

Useful files:

- `benchmark_manifest.json`: frozen question/world metadata and digest;
- `initial_calibration.json`: four-state calibration for all 48 worlds;
- `branch_groups.json`: common-snapshot branch records and held-out confirmations;
- `seed_level_results.jsonl`: the 768 seed-level records;
- `paired_repair_evidence.json`: evaluator-paired confounding/leakage repair records
  with before/after estimates, confidence intervals, bias, estimator/provenance,
  hidden group-validation metrics, and confirmation receipts;
- `sample_width_evidence.json`: before/after sample-size and confidence-width records;
- `reliable_negative_confirmation_evidence.json`: independent confirmation records for
  the reliable negative-result worlds;
- `confirmation_denominators.json`: explicit eligible/performed/passed/data-independent
  counts and conditional rates (non-eligible rows are never counted as passes);
- `target_agreement.json`: empirical branch winner vs pre-registered family target;
- `tier1_go.json`: machine-readable diagnostic gate;
- `summary.json`: counts and gate summary.

Reproduce with:

```bash
PYTHONPATH=PESCO python PESCO/scripts/run_tier1_v03.py PESCO/artifacts/tier1_v03
```

The artifact deliberately sets `diagnostic_only=true` and
`formal_model_claim=false`; it does not establish LLM/OOD generalization or a formal
algorithm comparison.

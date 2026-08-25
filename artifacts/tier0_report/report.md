# PESCO experimental report

> This report is generated from executable result records. `--demo` values are a pipeline smoke test; `--tier0` values are a deterministic pilot diagnostic. Neither is a trained-model scientific claim.

- Generated (UTC): `2026-08-25T11:22:07.965418+00:00`
- Records: **64**
- Methods: `Tier0-BranchRollout`
- Splits: `pilot`
- Statistical unit: question/task cluster where `question_id` is available; bootstrap interval uses deterministic pilot resampling.
- VRS weights: `{"alpha": 1.0, "beta": 1.0, "eta": 1.0, "gamma": 1.0, "lambda": 0.1}`

## Overall result template (§26.1)

| method | vrs | state_macro_f1 | flip_accuracy | effective_switch_rate | invalid_claim_rate | fdr | replication_rate | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tier0-BranchRollout | 1.277 | 68.0% | NA | 6.2% | 6.2% | NA | 87.0% | 1.449 |

## Evidence-state breakdown (§17.2)

See [evidence_confusion.csv](evidence_confusion.csv) and [evidence_confusion_matrices.png](evidence_confusion_matrices.png). Invalid→Supported and Insufficient→Refuted errors deserve explicit audit.

## Implementation boundary

Named external methods in the CPU pilot are labelled adapters/reference policies; this is not an external-paper or LLM-checkpoint reimplementation.

## Strategy correction and discovery (§17.5–§17.12)

See [strategy_metrics.png](strategy_metrics.png), [action_choice_heatmap.png](action_choice_heatmap.png), [cost_frontier.png](cost_frontier.png), [replication_fdr.png](replication_fdr.png), [preference_reversal.png](preference_reversal.png) when the corresponding fields are present.

## Paired-world / branch diagnostics (§9–§10)

Branch trajectory plots are emitted when records contain `turn`/`step` and `utility`; they aggregate mean and min/max branch spread and cap legends via method-level small multiples. The data contract keeps `question_id`, `world_id`, `snapshot_id`, `branch_id`, and `seed` available for replay audits.

## Reproducibility and interpretation boundary

1. Keep the input result file, freeze manifest, verifier digest, and generated `summary.json` together.
2. Do not treat demo records as model evidence; replace them with frozen Tier 0/1/2 runner outputs before making claims.
3. Report ID and OOD splits separately, with cluster bootstrap and preregistered multiple-comparison correction.
4. Missing VRS components remain `NA`; do not interpret them as zero scientific value.
5. A higher VRS with lower cost is desirable, but no single chart proves scientific validity, independent confirmation, or global novelty.

## Generated artifacts

- [action_choice_heatmap.png](action_choice_heatmap.png)
- [action_choice_heatmap.svg](action_choice_heatmap.svg)
- [branch_trajectories.png](branch_trajectories.png)
- [branch_trajectories.svg](branch_trajectories.svg)
- [cost_frontier.png](cost_frontier.png)
- [cost_frontier.svg](cost_frontier.svg)
- [evidence_confusion_matrices.png](evidence_confusion_matrices.png)
- [evidence_confusion_matrices.svg](evidence_confusion_matrices.svg)
- [overview_metrics.png](overview_metrics.png)
- [overview_metrics.svg](overview_metrics.svg)
- [preference_reversal.png](preference_reversal.png)
- [preference_reversal.svg](preference_reversal.svg)
- [replication_fdr.png](replication_fdr.png)
- [replication_fdr.svg](replication_fdr.svg)
- [strategy_metrics.png](strategy_metrics.png)
- [strategy_metrics.svg](strategy_metrics.svg)
- [metrics_by_method_split.csv](metrics_by_method_split.csv)
- [metrics_overall.csv](metrics_overall.csv)
- [evidence_confusion.csv](evidence_confusion.csv)
- [vrs_cluster_bootstrap.csv](vrs_cluster_bootstrap.csv)
- [summary.json](summary.json)
- [report_metadata.json](report_metadata.json)

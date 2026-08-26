# PESCO experimental report

> This report is generated from executable result records. `--demo` values are a pipeline smoke test; `--tier0` values are a deterministic pilot diagnostic. Neither is a trained-model scientific claim.

- Generated (UTC): `2026-08-25T16:48:22.563515+00:00`
- Records: **120**
- Methods: `Base, CVT-RL, DiscoPO, Ecpo, Evidence-Gated SMOPD, GDPO, GRPO-FourState, GRPO-Terminal, PESCO-Full, PESCO-Offline, Rule-Based, SFT, SMOPD, Search-Only, TCPO`
- Splits: `pilot_id, pilot_ood`
- Statistical unit: question/task cluster where `question_id` is available; conditional rates use only eligible denominators; a one-cluster bootstrap interval is reported as `NA` (not a zero-width interval).
- VRS weights: `{"alpha": 1.0, "beta": 1.0, "eta": 1.0, "gamma": 1.0, "lambda": 0.1}`

## Overall result template (§26.1)

| method | comparison_role | formal_comparison_eligible | vrs | state_macro_f1 | flip_accuracy | flip_eligible_n | effective_switch_rate | required_switch_n | invalid_repair_rate | invalid_repair_n | underpower_handling | insufficient_handling_n | invalid_claim_rate | fdr | replication_rate | confirmation_eligible_n | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | diagnostic_reference | false | 0.5311 | NA | 0.0% | 2 | 0.0% | 2 | 0 | 2 | 0 | 2 | 0.0% | NA | 100.0% | 4 | 2.189 |
| CVT-RL | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| DiscoPO | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| Ecpo | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| Evidence-Gated SMOPD | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| GDPO | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| GRPO-FourState | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| GRPO-Terminal | external_name_adapter_excluded | false | 0.5311 | NA | 0.0% | 2 | 0.0% | 2 | 0 | 2 | 0 | 2 | 0.0% | NA | 100.0% | 4 | 2.189 |
| PESCO-Full | diagnostic_pesco_reference | false | 1.727 | NA | 100.0% | 2 | 100.0% | 2 | 0 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| PESCO-Offline | diagnostic_pesco_reference | false | 1.334 | NA | 100.0% | 2 | 100.0% | 2 | 0 | 2 | 0 | 2 | 0.0% | NA | 100.0% | 6 | 2.189 |
| Rule-Based | diagnostic_control | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| SFT | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| SMOPD | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |
| Search-Only | diagnostic_oracle_upper_bound | false | 1.786 | NA | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 5.743 |
| TCPO | external_name_adapter_excluded | false | 2.135 | 100.0% | 100.0% | 2 | 100.0% | 2 | 1 | 2 | 1 | 2 | 0.0% | NA | 100.0% | 8 | 2.256 |

## Evidence-state breakdown (§17.2)

See [evidence_confusion.csv](evidence_confusion.csv) and [evidence_confusion_matrices.png](evidence_confusion_matrices.png). Invalid→Supported and Insufficient→Refuted errors deserve explicit audit.

## Implementation boundary

Named external methods in the CPU pilot are labelled adapters/reference policies; this is not an external-paper or LLM-checkpoint reimplementation.

| method | implementation_status | comparison_role | formal_comparison_eligible |
| --- | --- | --- | --- |
| Base | fixed_policy_reference | diagnostic_reference | false |
| CVT-RL | reference_cpu_adapter | external_name_adapter_excluded | false |
| DiscoPO | reference_cpu_adapter | external_name_adapter_excluded | false |
| Ecpo | reference_cpu_adapter | external_name_adapter_excluded | false |
| Evidence-Gated SMOPD | reference_cpu_adapter | external_name_adapter_excluded | false |
| GDPO | reference_cpu_adapter | external_name_adapter_excluded | false |
| GRPO-FourState | reference_cpu_adapter | external_name_adapter_excluded | false |
| GRPO-Terminal | fixed_policy_reference | external_name_adapter_excluded | false |
| PESCO-Full | tabular_pesco_reference | diagnostic_pesco_reference | false |
| PESCO-Offline | tabular_pesco_reference | diagnostic_pesco_reference | false |
| Rule-Based | transparent_rule_based_control | diagnostic_control | false |
| SFT | reference_cpu_adapter | external_name_adapter_excluded | false |
| SMOPD | reference_cpu_adapter | external_name_adapter_excluded | false |
| Search-Only | oracle_search_diagnostic | diagnostic_oracle_upper_bound | false |
| TCPO | reference_cpu_adapter | external_name_adapter_excluded | false |

## Strategy correction and discovery (§17.5–§17.12)

See [strategy_metrics.png](strategy_metrics.png), [action_choice_heatmap.png](action_choice_heatmap.png), [cost_frontier.png](cost_frontier.png), [replication_fdr.png](replication_fdr.png), [preference_reversal.png](preference_reversal.png), [split_vrs_heatmap.png](split_vrs_heatmap.png) when the corresponding fields are present.

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
- [split_vrs_heatmap.png](split_vrs_heatmap.png)
- [split_vrs_heatmap.svg](split_vrs_heatmap.svg)
- [strategy_metrics.png](strategy_metrics.png)
- [strategy_metrics.svg](strategy_metrics.svg)
- [metrics_by_method_split.csv](metrics_by_method_split.csv)
- [metrics_overall.csv](metrics_overall.csv)
- [evidence_confusion.csv](evidence_confusion.csv)
- [vrs_cluster_bootstrap.csv](vrs_cluster_bootstrap.csv)
- [summary.json](summary.json)
- [report_metadata.json](report_metadata.json)

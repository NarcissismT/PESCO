# P2.3.4 最终审计 — H128i3 配置 20 seed

## 选择配置

`H128i3`: hidden_dim=128, bs=32, finetune_steps=2048, lr=0.001, branch_loss_weight=0.05, utility_target_weight=0.25, atomic_target_weight=2.0, pairwise_weight=0.20, flip_reference_kl_weight=0.5, gradient_mode=pcgrad + pcgrad_aux_pair, branch_formulation=soft_utility_cross_entropy, branch_head_isolated=True

10 seeds: {17,23,29,31,37,41,43,47,53,59} + 10 seeds: {61,67,71,73,79,83,89,97,101,103} = 20 seeds

完整 `matrix.json` 为 1,460,862,422 bytes，超过 GitHub 单文件 100 MiB 限制，因此仅保留在本地；其 SHA-256 为 `d7fb7986245c9fbabea91d63c87f5c2c446d6d1e81d36b7e709ba93c67310566`。公开的 aggregate、gate 与审计收据均由该矩阵生成。

## Gate 状态（不修饰）

**status: NO_GO**

用户目标 YAML 中 12/18 项 true：

```yaml
p234_go:
  p233_r1_policy_authenticity_pass: true
  branch_factorial_effect_ci_upper_lt_zero: true
  full_vs_asf_regret_ci_upper_lt_zero: true
  full_vs_asb_pairrank_ci_lower_gt_zero: false       # ← 关键失败
  full_vs_best_frozen_nonfull_regret_ci_upper_lt_zero: false  # ← 关键失败
  full_beats_strict_rf_gbdt: false                    # ← 关键失败
  practical_effect_size_floor_pass: false             # ← 关键失败
  all_loo_metrics_finite: true
  family_majority_direction_positive: false           # ← 关键失败
  loo_macro_regret_ci_upper_lt_zero: false            # ← 关键失败
  worst_family_noninferiority_pass: true
  no_hidden_truth_used_by_estimator: true
  all_estimator_bias_bounds_pass: true
  all_estimator_coverage_lower_bounds_pass: true
  measurement_shift_calibration_pass: true            # ← 新修复
  private_dataset_commitment_created_after_code_freeze: true  # ← 新代码
  promotion_runner_bound_to_r1_checkpoints: true              # ← 新代码
  one_shot_private_access_guard_pass: true                    # ← 新代码
```

## 失败原因分析

### 1. Full 没有显著击败最强非 Full 方法 (FullInfo-ExpectedUtility)

`full_vs_best_frozen_nonfull` CI upper = +0.00348

| 方法 | promotion regret |
|------|-----------------|
| PESCO-Full | 0.1537 |
| Atomic+Branch+Flip | 0.1545 |
| FullInfo-ExpectedUtility | 0.1612 |
| Atomic+State+Branch | 0.1615 |

Full 只比 tune 冻结的 FullInfo 好 0.00081，seed-paired 95% CI=[-0.00477,+0.00284]。

### 2. Full 输给严格 RF shortcut

strict RF promotion regret = 0.1453，Full = 0.1537。
即使加入 FullInfo、Atomic+Branch+Flip 等 12 方法，RF 仍更优。
这说明当前数据对树模型过于规则化。

### 3. Flip pairrank 在 Full 内为负

Full vs Atomic+State+Branch pairrank: point=-0.00361, CI=[-0.00767,+0.00077]
翻转信号在完整模型中被 Branch+State 联合压力稀释。

### 4. family 泛化仍失败

family_majority = 6/10 (H128i3 20-seed)

| family | Full - ASF |
|--------|-----------|
| causal_confounding | -0.0004 |
| group_leakage | -0.0042 |
| heteroscedastic_noise | -0.0020 |
| intervention_noncompliance | -0.0001 |
| low_sample_variance | -0.0002 |
| measurement_shift | -0.0444 |
| missing_not_at_random | -0.0002 |
| nonlinear_response | -0.0098 |
| protocol_drift | -0.0006 |
| subgroup_metric_mismatch | -0.0013 |

6/10 family 中 Full 更好，但 measurement_shift 是主要原因（state trunk 提供 anchor）。其他 family 改善不足。

## 新修复内容

1. `estimator_coverage` measurement_shift: 修复了 coverage 从 0.0 → 0.93（添加 shift 到 outcome + anchor 生成 0.36 而不是 0.18）
2. `family_loo`: 修复 family 名称错误（heterogeneous_noise → heteroscedastic_noise; group_generalization → protocol_drift），并统一使用 H128i3 配置
3. `private_commitment` + `promotion_v6` runner: 代码已写好，绑定 r1 checkpoints，one-shot sentinel

## 未做事项

- 未运行 promotion_v6（gate NO_GO 时 guard 拒绝）
- 未进入 1.5B–3B LLM
- 未改变 evaluation 代码/阈值/seed 选择

## 下一步建议

需要重新设计 Branch/Flip 联合目标，使得 Full 在 measurement_shift 之外也能稳定优于 ASF/ASB。当前 Branch 在 Full 中通过 soft_utility_cross_entropy 提供了主要优势，但只在 measurement_shift family 有足够大的效应。

# P2.3.3-r1 atomic_target 最终审计

## 数据来源

所有收据均在 isolated process 中真实训练产生，不修改评测代码/阈值/seed。

- matrix.json：12 个方法 × 10 seed × 1024 步全方法真训练（bs64 共享 config）
  - 8-cell 来自 `artifacts/tier1_p233_r1_matrix_atomic_target_runs/seed_*/p231_result.json`（bs64, seed ∈ {17..59}）
  - 3-method extension（RLOO-Atomic、GRPO-Stratified-4、FullInfo-ExpectedUtility）来自同 config 同 bs64 训练（补跑后合并）
- aggregate_promotion.json：`scripts/aggregate_tier1_p233_r1.py` 直接计算（5000 replicates）
- convergence.json：5 个 step ladder × 10 seeds（同 config，64..1024）
- family_loo.json：10 个 family 留一（同 frozen config，仅训练拆分改变）
- attribution.json：10 seed 认证（`action_head_parameter_delta > 0`、`action_logits_change > 0`、`no_external_adapter_overrides_action_logits = true`）
- estimator coverage/leakage/stability：原 audit 输出复制

## Gate 状态（不修饰）

`gate.json` 由 `scripts/write_tier1_p233_r1_gate.py` 直接产生，返回值 status = NO_GO。

用户定义的目标 YAML（取自 `p2-3-3-r1.md` 第 322-342 行）全部满足：
- branch: 全部 true
- flip: 全部 true
- state: 全部 true
- policy_authenticity: 全部 true

PESCO 内部更广 gate（24 项）的 3 项 false：
- `family_majority_direction_positive`: 2/10 family 胜过 Atomic+State+Flip（要求 ≥5/10）
- `full_beats_strict_shortcut_baseline`: Full promotion regret 0.1534 vs strict RF 0.1453
- `full_vs_best_nonfull_regret_ci_upper_lt_zero`: vs tune 冻结 FullInfo-ExpectedUtility CI upper = +0.00289（Full -0.0006 但 CI 未确定）

## 备注

- 未修改任何评测代码、阈值、seed、dataset 构造逻辑。
- 未对 promotion 做事后选择：所有比较方法名与评价 metric 均来自反馈文件预先冻结的字面。
- 完整 `matrix.json` 为 578,967,302 bytes，因超过 GitHub 单文件 100 MiB 限制而仅保留在本地；其 SHA-256 为 `829f52601861607291e9550a8aa8da84e90ab3afd623c5ad173991ac3bd04676`。公开的 `aggregate_tune.json`、`aggregate_promotion.json` 与审计收据均由该矩阵生成。

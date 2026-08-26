# PESCO 实现状态与逐项验收证据

更新时间：2026-08-26。这里的“完成”仅表示当前 CPU/Tier 0/Tier 1 诊断实现已经有可运行代码和验证证据；不等同于计划中的 Tier 2 真实模型论文实验已经完成。

完整的机器可读覆盖边界见 [`implementation_manifest.json`](implementation_manifest.json)，可用 `python PESCO/scripts/audit_implementation.py` 校验所有证据路径。

| 计划要求 | 当前实现 | 证据 |
| --- | --- | --- |
| R1 冻结/隔离 | `pesco_v0_2.yaml`、manifest、快照白名单、隐藏 world ID、只读 verifier、实验前假设登记、假设/证据与运行审计链 | `artifacts/pesco_pilot/freeze_check.json`、`hypothesis_registry.json`、`audit_ledger.jsonl`、`tests/test_core.py`、`tests/test_registry_scoring.py` |
| R2 四类动态证据 | `evidence_classifier.py`；Invalid precedence；Supported/Refuted/Insufficient interval rules；Tier 0/NumPy Tier 1 修复/采样转移 | `mvp_gate.json`、`negative_controls.json`、`artifacts/tier1_smoke.json`、`evidence_confusion_matrices.*` |
| R3 同状态真实分支 | `SnapshotManager`、`Tier0ResearchEnvironment.clone_from_snapshot`、共同 seeds、预算、逐世界一步最优动作表 | `same_state_branches.json`、`optimal_action_table.json`、`test_algorithms.py`、`branch_trajectories.*` |
| R4 策略信用 | LOO baseline/advantage、PESCO branch utility、token mask helper；tabular branch update 与 flip objective 分开记录 | `training_log_full.json`、LOO tests、`algorithms/objectives.py`、`algorithms/pesco_trainer.py` |
| R5 跨世界反转 | paired sampler、LCB/UCB margin、double difference、flip loss；CPU reference 对表格 logits 应用 flip loss 的解析梯度并记录 loss/gradient/update | `reversal_pairs.json`、`training_log_full.json`、`preference_reversal.*`、reversal tests |
| R6 新路径证书 | structure/execution/validity/confirmation/gain gates；区分 policy/teacher | `certificate_*.json`、`discovery_pass_at_k.*`（有候选时）、discovery tests |
| R7 训练阶段与实验 B 入口 | CPU reference offline/full loop；PPO/state/constraint objective API；解析 flip-loss 更新（非仅记录）；Tier 1 smoke；Tier 2 fail-closed seam；真实 zero-shot B 的本地 checkpoint runner（checkpoint 不入库，本工作区已完成一次外部 checkpoint 诊断） | `training_log_*.json`、`artifacts/tier1_smoke.json`、`training_authorization.json`、`stage_status.json`、`artifacts/pesco_pilot/zero_shot_diagnostic.json`、`artifacts/tier1_zero_shot.json`、`artifacts/experiment_b_real_zero_shot.json`、`scripts/run_tier1_zero_shot.py` |
| R8 基线公平 | Rule-Based 透明控制、Search-Only/PESCO controls 与 15 个命名 CPU diagnostic rows（其中 10 个是 external-name adapters）；共享环境、验证器、seed、预算；每行和 manifest 明确标注 `comparison_role`，当前所有 CPU rows 均 `formal_comparison_eligible=false` | `configs/baselines/matched_cpu_suite.yaml`、`baseline_manifest.json`、`results.json` |
| R9 指标/统计 | VRS、Macro-F1、FlipAcc、switch/persistence/refutation/underpower/repair/VNPR/FDR/replication/cost、cluster bootstrap、Holm/BH helpers；12 项消融注册表 | `report/metrics_*.csv`、`summary.json`、`evaluation/ablations.py`、`ablation_manifest.json` |
| R10 MVP gate | 4 worlds × 4 actions × 4 exploration seeds + independent confirmation；Tier 0 GO 表；证据打乱/隐藏证据/文件名替换/无效高分/可靠负结果/表面新颖/假复现负对照；replay | `mvp_counts.json`（64/16 计数）、`tier0_go.json`、`mvp_gate.json`（全部 required true）、`negative_controls.json` |
| R13 Tier 1 v0.3 A–F 诊断实验 | A/C/D/E 共用 12 独立问题 × 4 隐藏世界 × 4 动作 × 4 exploration seeds 的 frozen branch dataset；B 是单独的 12×4=48 world-level 外部 checkpoint zero-shot 诊断；F 是固定动作 MVP discovery boundary；C 为 state-reward，D 为 branch ablation，E 为 flip-loss ablation + Evidence-Gated SMOPD teacher-distillation；C/D/E 使用小型 PyTorch MLP genuine optimizer updates | `artifacts/tier1_v03/tier1_go.json`、`artifacts/tier1_v03/summary.json`、`artifacts/tier1_v03/experiment_a_environment_correctness.json`、`artifacts/tier1_v03/benchmark_manifest.json`、`artifacts/tier1_zero_shot.json`、`artifacts/experiment_b_real_zero_shot.json`、`artifacts/tier1_differentiable_suite/suite.json`、`artifacts/tier1_differentiable_suite/dataset.json`、`artifacts/tier1_differentiable_suite/experiment_c_state_reward.json`、`artifacts/tier1_differentiable_suite/experiment_d_branch_ablation.json`、`artifacts/tier1_differentiable_suite/experiment_e_flip_ablation.json`、`artifacts/tier1_differentiable_suite/experiment_f_discovery_boundary.json`、`scripts/run_tier1_differentiable_suite.py`、`configs/algorithms/tier1_differentiable_suite_v0_3.yaml`、`algorithms/differentiable_strategy.py`、`evaluation/tier1_differentiable_suite.py` |

## 公平评分与实验 A–F 边界

- `flip_accuracy` 使用可确认的 world-pair 分母；必要切换、Invalid 修复、Insufficient 处理和独立确认分别使用 `required_switch_n`、`invalid_repair_n`、`insufficient_handling_n`、`confirmation_eligible_n`，无资格样本为 `NA`。
- 单一 question/task cluster 不产生零宽 bootstrap 区间；`vrs_cluster_bootstrap.csv` 明确写入 `NA_single_cluster`。
- 固定四动作 MVP 中 `switch_to_alternative_method` 不是自主发现，所有方法的 discovery utility 均为 0，见 `discovery_policy.json`。
- `experiment_b_diagnostic.json` 和 `experiment_c_diagnostic.json` 是 pilot 中的 fail-closed 脚手架：它们不授权正式比较，也不替代真实 B/C 产物。
- 实验 B 的真实 zero-shot runner 是 `scripts/run_tier1_zero_shot.py`；本工作区使用外部本地 checkpoint 生成了当前 benchmark 的 fresh `artifacts/tier1_zero_shot.json`（及 `experiment_b_real_zero_shot.json` 别名），记录 digest、公开输入白名单和独立审计。由于 checkpoint 不入库，缺少 checkpoint 的干净 checkout 仍只会得到 `zero_shot_diagnostic.json`（CPU diagnostic、`pass=false`），不会把 Rule-Based 或 evaluator diagnostic 冒充真实模型。当前 B JSON 的 benchmark digest 与冻结 benchmark 均为 `sha256:542ec9…`，48/48 条公开 observation 已由 frozen checkpoint 完成 forward；JSON 顶层 `freshness_audit.status=fresh_current`、`benchmark_freshness_match=true` 和 `current_full_forward_completed=true` 是机器可读的 freshness 标记。B 仍保持 `diagnostic_only=true`、`formal_comparison_authorized=false`。
- 实验 A 的环境正确性、B 的 fresh 真实模型零样本诊断、C/D/E 的可微机制消融和 F 的发现边界分别见 `artifacts/tier1_v03/experiment_a_environment_correctness.json`、`artifacts/tier1_zero_shot.json`、`artifacts/tier1_differentiable_suite/experiment_c_state_reward.json`、`experiment_d_branch_ablation.json`、`experiment_e_flip_ablation.json` 与 `experiment_f_discovery_boundary.json`；A、B、C/D/E 为当前诊断性证据，F 明确是当前 MVP 的 deferred boundary。当前 B 的 48 条 world-level rows 上 action/state audit accuracy 分别为 0.3541667/0.25，且模型 48/48 行都输出 `continue_current_method + supported`，明确记录 zero-shot 输出坍缩；`formal_comparison_authorized=false`。
- 实验 C/D/E 共用 `artifacts/tier1_differentiable_suite/suite.json`：C 是普通 state-reward sufficiency 对照，D 是 branch ablation，E 是 confirmed flip-loss ablation + Evidence-Gated SMOPD teacher-distillation。该 suite 的 `diagnostic_ood` 是机制实例诊断 split，不是正式 final OOD，也不打开 Tier-2/LLM 门禁；E 只有在相同 optimizer-step 且所有报告 split 的严格比较门禁满足时才记录 Full > NoFlip，SMOPD 的教师/参数计算不匹配，故不作正式比较。
- C/D/E 还显式记录 seed-level utility 的真实波动审计：当前 192 个 action-level seed utility 数组中 7 个有波动、185 个为离散协议/成本项导致的常数数组；因此 `formal_seed_variance_claim_authorized=false`，这些反转置信度和方法差异只能作为 CPU 机制诊断，不能解释为论文级方差或正式泛化证据。
- 结果记录把策略真实输出的 `policy_predicted_state` 与 evaluator-only 的 `evaluator_diagnostic_state` 分开；Base/Search/PESCO action-only policy 的状态指标为 `NA`，不会把外部分类器的标签冒充模型识别。
- Tier 1 的同状态难例按证据状态/注册协议分组，要求同一证据状态出现多个 evaluator branch winner；公开任务族保留为上下文，字节级公开 Observation collision 仅作审计字段而非通过门禁。

## 机器可读 pilot 结果

`mvp_gate.json` 当前应包含：

```json
{
  "all_worlds_execute": true,
  "branch_group_count_16": true,
  "exploration_experiments_64": true,
  "negative_controls_pass": true,
  "invalid_world_detected": true,
  "insufficient_not_refuted": true,
  "confirmed_reversal": true,
  "no_world_identifier_leakage": true,
  "reproducible_branches": true,
  "cpu_reference_loop_authorized": true,
  "tier2_llm_training_authorized": false
}
```

## 尚未宣称完成的内容

1. 真实 LLM/7B/QLoRA 后训练、执行器联合训练和参数内化验证。
2. 计划最低清单中的外部算法论文复现；当前名称对应共享协议下的透明 CPU adapter，不是外部方法的声称复现。
3. 足够数量的独立研究问题、正式 promotion/final ID/final OOD、功效达标的论文级统计推断。
4. 全球原创性或现实科学发现。新路径证书只支持局部、可执行、独立确认的自主性声明。
5. `stage_3_zero_shot` 不再硬编码 GO：pilot 的 `zero_shot_diagnostic.json` 与 `stage_status.json` 明确为 NO-GO。当前 B artifact 是一次独立外部 checkpoint 诊断，不等于打开 `stage_3` 或 Tier 2；虽然 fresh forward 已完成，仍需冻结执行器、最终分割、污染审计及其他 scientific hard gates，才能申请打开该阶段。
6. Tier 1 v0.3 differentiable suite 是小型 CPU MLP 的机制实验，不是 LLM、QLoRA 或外部论文方法复现；其 `diagnostic_ood` 是机制实例留出的诊断 split，不等价于已锁定的正式 final OOD。

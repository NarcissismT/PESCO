# PESCO 实现状态与逐项验收证据

更新时间：2026-08-25。这里的“完成”仅表示当前 CPU/Tier 0 参考实现已经有可运行代码和验证证据；不等同于计划中的 Tier 2 真实模型论文实验已经完成。

完整的机器可读覆盖边界见 [`implementation_manifest.json`](implementation_manifest.json)，可用 `python PESCO/scripts/audit_implementation.py` 校验所有证据路径。

| 计划要求 | 当前实现 | 证据 |
| --- | --- | --- |
| R1 冻结/隔离 | `pesco_v0_2.yaml`、manifest、快照白名单、隐藏 world ID、只读 verifier、实验前假设登记、假设/证据与运行审计链 | `artifacts/pesco_pilot/freeze_check.json`、`hypothesis_registry.json`、`audit_ledger.jsonl`、`tests/test_core.py`、`tests/test_registry_scoring.py` |
| R2 四类动态证据 | `evidence_classifier.py`；Invalid precedence；Supported/Refuted/Insufficient interval rules；Tier 0/NumPy Tier 1 修复/采样转移 | `mvp_gate.json`、`negative_controls.json`、`artifacts/tier1_smoke.json`、`evidence_confusion_matrices.*` |
| R3 同状态真实分支 | `SnapshotManager`、`Tier0ResearchEnvironment.clone_from_snapshot`、共同 seeds、预算、逐世界一步最优动作表 | `same_state_branches.json`、`optimal_action_table.json`、`test_algorithms.py`、`branch_trajectories.*` |
| R4 策略信用 | LOO baseline/advantage、PESCO branch utility、token mask helper | `training_log_full.json`、LOO tests、`algorithms/objectives.py` |
| R5 跨世界反转 | paired sampler、LCB/UCB margin、double difference、flip loss | `reversal_pairs.json`、`preference_reversal.*`、reversal tests |
| R6 新路径证书 | structure/execution/validity/confirmation/gain gates；区分 policy/teacher | `certificate_*.json`、`discovery_pass_at_k.*`（有候选时）、discovery tests |
| R7 训练阶段 | CPU reference offline/full loop；PPO/state/constraint objective API；Tier 1 smoke；Tier 2 fail-closed seam | `training_log_*.json`、`artifacts/tier1_smoke.json`、`training_authorization.json`、`stage_status.json` |
| R8 基线公平 | 14 个命名 adapter，共享环境、验证器、seed、预算；每行和 manifest 明确标注 adapter proxy | `configs/baselines/matched_cpu_suite.yaml`、`baseline_manifest.json`、`results.json` |
| R9 指标/统计 | VRS、Macro-F1、FlipAcc、switch/persistence/refutation/underpower/repair/VNPR/FDR/replication/cost、cluster bootstrap、Holm/BH helpers；12 项消融注册表 | `report/metrics_*.csv`、`summary.json`、`evaluation/ablations.py`、`ablation_manifest.json` |
| R10 MVP gate | 4 worlds × 4 actions × 4 exploration seeds + independent confirmation；Tier 0 GO 表；证据打乱/隐藏证据/文件名替换/无效高分/可靠负结果/表面新颖/假复现负对照；replay | `mvp_counts.json`（64/16 计数）、`tier0_go.json`、`mvp_gate.json`（全部 required true）、`negative_controls.json` |

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

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
| R9 指标/统计 | VRS、Macro-F1、Pairwise Reversal Ranking Accuracy（与 flip loss 对齐）与普通 action exact top-1 分离；selected-action confirmation receipt、question-macro reversal 权重、seed×question 两层 bootstrap、family LOO、reward top1−top2 tie/non-tie、shortcut probe 与多重校正 | `evaluation/tier1_differentiable_suite.py`、`evaluation/tier1_p2_experiments.py`、`evaluation/shortcut_probes.py`、`tests/test_tier1_p2_p3.py`、`tests/test_shortcut_probes.py` |
| R10 MVP gate | 4 worlds × 4 actions × 4 exploration seeds + independent confirmation；Tier 0 GO 表；证据打乱/隐藏证据/文件名替换/无效高分/可靠负结果/表面新颖/假复现负对照；replay | `mvp_counts.json`（64/16 计数）、`tier0_go.json`、`mvp_gate.json`（全部 required true）、`negative_controls.json` |
| R13 Tier 1 v0.3 A–F 诊断实验 | A/C/D/E 共用 12 独立问题 × 4 隐藏世界 × 4 动作 × 4 exploration seeds 的 frozen branch dataset；B 是单独的 12×4=48 world-level 外部 checkpoint zero-shot 诊断；F 是固定动作 MVP discovery boundary；C 为 state-reward，D 为 branch ablation，E 为 flip-loss ablation + Evidence-Gated SMOPD teacher-distillation；C/D/E 使用小型 PyTorch MLP genuine optimizer updates；每个执行边界保存完整 provenance manifest | `artifacts/tier1_v03/tier1_go.json`、`artifacts/tier1_v03/summary.json`、`artifacts/tier1_v03/experiment_a_environment_correctness.json`、`artifacts/tier1_v03/benchmark_manifest.json`、`artifacts/tier1_v03/run_manifest.json`、`artifacts/tier1_zero_shot.json`、`artifacts/experiment_b_real_zero_shot.json`、`artifacts/tier1_zero_shot_run_manifest.json`、`artifacts/tier1_differentiable_suite/suite.json`、`artifacts/tier1_differentiable_suite/dataset.json`、`artifacts/tier1_differentiable_suite/run_manifest.json`、`artifacts/tier1_differentiable_suite/experiment_c_state_reward.json`、`artifacts/tier1_differentiable_suite/experiment_d_branch_ablation.json`、`artifacts/tier1_differentiable_suite/experiment_e_flip_ablation.json`、`artifacts/tier1_differentiable_suite/experiment_f_discovery_boundary.json`、`scripts/run_tier1_differentiable_suite.py`、`configs/algorithms/tier1_differentiable_suite_v0_3.yaml`、`algorithms/differentiable_strategy.py`、`evaluation/tier1_differentiable_suite.py` |
| R14 v0.4 base/extended | 已完成历史双轨 CPU 诊断，但现标记 `diagnostic_v04_consumed`；latent-template 重用及旧指标边界使其仅可作审计/调试，不作为 post-fix final | `artifacts/tier1_v04/consumption_notice.json`、`artifacts/tier1_v04_extended/consumption_notice.json`、`artifacts/tier1_v04_extended/tier1_v04_extended_go.json`、`artifacts/tier1_v04_extended/run_manifest.json`、`artifacts/tier1_p21_diagnostic/p21_diagnostic_result.json` |
| R15 P2/P3 | 旧 v0.4 P2（含 formal-final 与 v3 budget）均保留原始 NO-GO 并标记 `diagnostic_v04_consumed`；P3 继续 fail-closed，未执行 LoRA/QLoRA；P2.1 fresh/shortcut/algorithm diagnostics 已完成 | `artifacts/tier1_p2_v04_ten_seed/consumption_notice.json`、`artifacts/tier1_p2_v04_ten_seed_v3/consumption_notice.json`、`artifacts/tier1_p2_v04_formal_final_ten_seed/consumption_notice.json`、`artifacts/tier1_p21_diagnostic/p21_diagnostic_result.json`、`artifacts/tier1_p21_algorithm_diagnostic/p21_algorithm_diagnostic.json`、`artifacts/tier1_p21_shortcut_probe/shortcut_probe_result.json`、`artifacts/tier1_p3_small_model_gate/small_model_gate.json` |
| R16 zero-shot robustness | 单 checkpoint bounded robustness 仍 fail-closed；不把重复权重或未执行的 rotation/template forward 当作独立证据 | `artifacts/tier1_zero_shot_robustness_null_only.json`、`artifacts/tier1_zero_shot_robustness_null_only.run_manifest.json` |
| R17 formal final-ID/OOD | v0.4 formal environment receipts 已留存但标记 `diagnostic_v04_consumed`；latent overlap 使其不是独立 clean final，formal model comparison 保持锁定；v0.5 structural boundary 已准备（私有 evaluator bundle 仅在受控工作区提供，不纳入公开 evidence），且 v0.5 baseline receipt 只允许在未打开 final 数据前的 dev split 选择 | `artifacts/tier1_v04_formal_final/consumption_notice.json`、`artifacts/tier1_v04_formal_final/formal_final_go.json`、`artifacts/tier1_v04_formal_final/whole_family_holdout_audit.json`、`artifacts/tier1_v05_frozen_final/v05_freeze_audit.json`、`artifacts/tier1_v05_frozen_final/freeze_receipt.json`、`artifacts/tier1_v05_frozen_final/evaluator_contract.json` |
| R18 P2.1 fresh/algorithm diagnostics | fresh train/tune/promotion diagnostic、counterfactual leakage、question-macro/seed×question/family-LOO、七方法比较、gradient-conflict、dynamic constraint/PCGrad 与 baseline-on-tune selection 均有独立 receipts；shortcut strict mode 在 sklearn 缺失时 fail-closed；保持 diagnostic-only | `artifacts/tier1_p21_diagnostic/p21_diagnostic_result.json`、`artifacts/tier1_p21_diagnostic/collection_audit.json`、`artifacts/tier1_p21_diagnostic/counterfactual_leakage_audit.json`、`artifacts/tier1_p21_algorithm_diagnostic/p21_algorithm_diagnostic.json`、`artifacts/tier1_p21_algorithm_diagnostic/run_manifest.json`、`artifacts/tier1_p21_algorithm_diagnostic/shortcut_probe_strict/shortcut_probe_result.json`、`scripts/run_tier1_p21_diagnostics.py`、`scripts/run_tier1_p21_constrained.py`、`scripts/aggregate_tier1_p21_diagnostics.py` |
| R19 shortcut/v0.5 freeze | Logistic/Random Forest/GBDT shortcut probes 已完成并显式标注 fallback；v0.5 final-ID/OOD 各 48 clusters、4 个新 OOD families、latent/generator signatures 与独立 evaluator contract 审计通过；baseline receipt 现在还必须绑定非空 dev 数据、实际 candidate-metrics 选择文件及可重算的四类文件 digest。clean commit/tag、算法/基线 freeze 和模型评测仍待完成（私有 evaluator bundle 不纳入公开 evidence） | `artifacts/tier1_p21_shortcut_probe/shortcut_probe_result.json`、`research_strategy_optimization/evaluation/shortcut_probes.py`、`artifacts/tier1_v05_frozen_final/signature_audit_summary.json`、`artifacts/tier1_v05_frozen_final/v05_freeze_audit.json`、`artifacts/tier1_v05_frozen_final/independent_audit.json`、`artifacts/tier1_v05_frozen_final/freeze_receipt.json`、`artifacts/tier1_v05_frozen_final/evaluator_contract.json`、`research_strategy_optimization/tests/test_tier1_v05_frozen_final.py` |

## 公平评分与实验 A–F 边界

- 主要反转指标是与 flip loss 对齐的 `pairwise_reversal_ranking_accuracy`；普通 action 的精确 top-1 另记为 `action_exact_top1_accuracy`，不再把端点 top-1 冲突命名为 FlipAcc。必要切换、Invalid 修复、Insufficient 处理和 selected-action 独立确认分别使用显式条件分母；无资格样本为 `NA`。
- reversal 按 question macro 计分，每个 question 的 pair 权重和归一化为 1；P2.1 的不确定性使用 seed×question 两层 bootstrap，并报告 mechanism-family leave-one-out，而不是只重采样训练 seed。
- selected-action confirmation 只统计模型实际选择动作的 replicate receipts；已尝试但失败的 confirmation 保留在分母，未尝试/缺失 receipt 不会被静默当作通过。
- reward stability 的 non-tie 使用 top1−top2 gap（而非 max−min），并分别报告 tie/non-tie 稳定性；`best_non_Full` 是当前 baseline 选择名称，避免把 PESCO 消融误称为外部 best non-PESCO。
- 单一 question/task cluster 不产生零宽 bootstrap 区间；`vrs_cluster_bootstrap.csv` 明确写入 `NA_single_cluster`。
- 固定四动作 MVP 中 `switch_to_alternative_method` 不是自主发现，所有方法的 discovery utility 均为 0，见 `discovery_policy.json`。旧 `certificate_pass=true` 文件保留历史字段但带有 `legacy=true`、`artifact_scope=legacy_certificate_evidence` 和 `legacy_scope`；`legacy_certificates_manifest.json` 明确其不属于当前 discovery scope。
- `experiment_b_diagnostic.json` 和 `experiment_c_diagnostic.json` 是 pilot 中的 fail-closed 脚手架：它们不授权正式比较，也不替代真实 B/C 产物。
- 实验 B 的真实 zero-shot runner 是 `scripts/run_tier1_zero_shot.py`；本工作区使用外部本地 checkpoint 生成了当前 benchmark 的 fresh `artifacts/tier1_zero_shot.json`（及 `experiment_b_real_zero_shot.json` 别名），记录 digest、公开输入白名单和独立审计。由于 checkpoint 不入库，缺少 checkpoint 的干净 checkout 仍只会得到 `zero_shot_diagnostic.json`（CPU diagnostic、`pass=false`），不会把 Rule-Based 或 evaluator diagnostic 冒充真实模型。当前 B JSON 的 benchmark digest 与冻结 benchmark 均为 `sha256:542ec9…`，48/48 条公开 observation 已由 frozen checkpoint 完成 forward；JSON 顶层 `freshness_audit.status=fresh_current`、`benchmark_freshness_match=true` 和 `current_full_forward_completed=true` 是机器可读的 freshness 标记。B 仍保持 `diagnostic_only=true`、`formal_comparison_authorized=false`。
- 实验 A 的环境正确性、B 的 fresh 真实模型零样本诊断、C/D/E 的可微机制消融和 F 的发现边界分别见 `artifacts/tier1_v03/experiment_a_environment_correctness.json`、`artifacts/tier1_zero_shot.json`、`artifacts/tier1_differentiable_suite/experiment_c_state_reward.json`、`experiment_d_branch_ablation.json`、`experiment_e_flip_ablation.json` 与 `experiment_f_discovery_boundary.json`；A、B、C/D/E 为当前诊断性证据，F 明确是当前 MVP 的 deferred boundary。当前 B 的 48 条 world-level rows 上 action/state audit accuracy 分别为 0.3541667/0.25，且模型 48/48 行都输出 `continue_current_method + supported`，明确记录 zero-shot 输出坍缩；`formal_comparison_authorized=false`。
- 实验 C/D/E 共用 `artifacts/tier1_differentiable_suite/suite.json`：C 是普通 state-reward sufficiency 对照，D 是 branch ablation，E 是 confirmed flip-loss ablation + Evidence-Gated SMOPD teacher-distillation。该 suite 的 `diagnostic_ood` 是机制实例诊断 split，不是正式 final OOD，也不打开 Tier-2/LLM 门禁；E 只有在相同 optimizer-step 且所有报告 split 的严格比较门禁满足时才记录 Full > NoFlip，SMOPD 的教师/参数计算不匹配，故不作正式比较。
- C/D/E 还显式记录 seed-level utility 的真实波动审计：当前 192 个 action-level seed utility 数组中 7 个有波动、185 个为离散协议/成本项导致的常数数组；因此 `formal_seed_variance_claim_authorized=false`，这些反转置信度和方法差异只能作为 CPU 机制诊断，不能解释为论文级方差或正式泛化证据。
- 结果记录把策略真实输出的 `policy_predicted_state` 与 evaluator-only 的 `evaluator_diagnostic_state` 分开；Base/Search/PESCO action-only policy 的状态指标为 `NA`，不会把外部分类器的标签冒充模型识别。
- Tier 1 的同状态难例按证据状态/注册协议分组，要求同一证据状态出现多个 evaluator branch winner；公开任务族保留为上下文，字节级公开 Observation collision 仅作审计字段而非通过门禁。
- Tier 1 v0.3 的计数语义固定为 48 个 `question_world_group`、192 条 `action_level` rows、768 条 `seed_level` observations。A 的 `question_world_groups.json` 是 48 组索引，`action_level_branches.json` 是 192 行 canonical 文件；旧 `branch_groups.json` 仅是 action-level 兼容别名。`invalid_local_optimization` 只统计 Invalid 状态下选择 CONTINUE/SWITCH 且输给 evaluator-owned public branch winner 的行，因此正确 Invalid+SWITCH 不计入。
- A/B/C/D/E/F 与 v0.4/P2 runner 的 `run_manifest.json` 记录 Git SHA/dirty 状态、实际命令、Python/NumPy/PyTorch 及依赖版本 digest、训练 seed、source/data digest；B 另记录外部 checkpoint 的完整文件、权重和 tokenizer digest。无 checkpoint 时 manifest 显式写明 `no_checkpoint_supplied`，不会伪造空模型摘要。
- v0.4 extended、formal-final 与 P2 的 CPU 产物保留独立 provenance，并在各目录新增 `consumption_notice.json`；这些 notice 是加性解释层，未修改历史主 JSON 或 run-manifest digest。P2 已真实执行 10 个 training seeds，但旧 promotion 门禁记录为 `NO-GO`，不能解读为 PESCO 优越性结论或 P2.1 后的算法结论。

## 反馈新增实验：v0.4、P2、P2.1、P3、v0.5

- v0.4 extended 已完成双轨 CPU 诊断：8 个机制族、64 个问题、256 个世界，8 个探索种子和 8 个独立确认种子；每个问题保留四状态世界。原始产物仍在 `artifacts/tier1_v04_extended/`，但其 `consumption_notice.json` 已将状态设为 `diagnostic_v04_consumed`：latent-template 重用与旧评测边界使它只可作历史调试/审计，不能作为修改算法后的 final 或 paper comparison。
- Raw-evidence 与 oracle-state 是同一批执行回执的双视图；raw track 只含数值 receipt，结构化状态/家族标签被屏蔽。canonical bounded posterior-EU/VOI pass 选取每族 variant-3 anchor，覆盖 8 个家族（含 causal hidden-method-B 与 composite OOD），输出 64 个决策和 64 条 diagnose→retest→replicate 轨迹；决策/轨迹 JSON 只用于 evaluator audit，不进入 policy input。
- P2 在含 atomic reward receipts 的冻结 raw dataset 上以 10 个 training seeds、64 optimizer steps、7 个训练方法和 inference-time branch-search upper bound 完成；extended 与 formal-final 两个 P2 目录现在都带 `status=diagnostic_v04_consumed` 的消费通知。历史 promotion 均为 `NO-GO`，且旧 exact top-1 FlipAcc/seed-only CI 已被 P2.1 的 PairRankAcc、question-macro 与 seed×question CI 取代，因此不能用来选择修改后的算法或宣称 formal superiority。
- P3 gate 因 P2 未晋级、`peft`/`bitsandbytes` 缺失且无 CUDA 设备而 fail-closed；`experiment_executed=false`。没有虚构 LoRA/QLoRA、在线 RL、7B 或参数内化结果。
- 实验 B 的 robustness harness 已补齐 action-letter rotation（4 个固定置换）、3 个 prompt templates、plain/native-chat/manual-chat provenance、masked constrained decoding、bounded `generate` control 和 null-prompt prior calibration：`scripts/run_tier1_zero_shot_robustness.py`。当前工作区只有一个唯一完整 checkpoint；bounded artifact `artifacts/tier1_zero_shot_robustness_null_only.json` 因 `unique_checkpoints=1<3`、instruction-role panel 缺失且未执行完整 rotation/template forward 保持 `pass=false`，并记录完整 checkpoint/weights/tokenizer digest。checkpoint tokenizer 虽声明 chat template，但当前 `jinja2=3.0.3` 不满足 native `apply_chat_template` 的 `>=3.1` 要求，故手工 Qwen rendering 明确标为 non-native；没有把重复权重算作独立 checkpoint。
- formal final 环境实验已实际完成：`artifacts/tier1_v04_formal_final/` 有 train/dev 24/24、final ID 24 clusters（6 个机制族）、final OOD 20 clusters（2 个 whole-family holdout），final ID/OOD confirmed reversals 196/114，raw/oracle 双轨和完整多步轨迹均通过结构门禁。其 `consumption_notice.json` 现在标记 `diagnostic_v04_consumed`；环境 receipts 仍可审计，但 latent-template overlap 使它不再是独立 clean final，formal model comparison 继续锁定。
- `artifacts/tier1_p2_v04_formal_final_ten_seed/` 保存的是历史 formal-ID/final-OOD 64-step P2；消费通知明确它不再作为独立 final 或 post-fix 结果引用。其原始 promotion 仍为 `NO-GO`，因此不打开 final model-comparison authorization。
- P2.1 fresh diagnostic 已完成于 `artifacts/tier1_p21_diagnostic/`：64 问题、256 世界、8 个 exploration/8 个 confirmation seeds、395 个同问题 reversal、按 question 归一化权重、pre-action raw observation 与 256/256 counterfactual leakage audit；`diagnostic_only=true`、`formal_comparison_authorized=false`。`artifacts/tier1_p21_shortcut_probe/` 已完成 Logistic/Random Forest/GBDT shortcut probes，并把无 sklearn 时的 NumPy fallback 明确标注为 fallback。另有 `artifacts/tier1_p21_seed_sweep_diagnostic/` 的 3-seed×4-method 隔离补充，以及 `artifacts/tier1_p21_seed_sweep_10seed_diagnostic/` 的 10-seed×4-method（40/40 成功）补充，均报告 seed→question 两层 bootstrap 与 family LOO；后者的 Full−NoFlip PairRankAcc CI 为 `[0.1050, 0.1740]`，仍不替代 final gate。`artifacts/tier1_p21_sensitivity_audit/` 进一步记录 1000 次全局 atomic-reward ±20% 权重扰动、top1−top2 稳定性、family 方向和 selected-action receipt 分母；另有独立 `artifacts/tier1_p21_constrained_diagnostic/` receipt；全部保持 diagnostic-only。
- v0.5 fresh-final 边界已准备但尚未签署：`artifacts/tier1_v05_frozen_final/` 有 final-ID/OOD 各 48 clusters、384 worlds、4 个新 OOD 机制族、disjoint generator/latent signatures 和独立 evaluator contract，独立审计 40/40 structural gates 通过；另有 evaluator-side ±20% atomic-reward 稳定性诊断（non-tie winner stability `0.9972`，overall `0.9986`）。但 `freeze_receipt.json` 仍为 `status=pending_clean_commit_tag`，算法超参与 dev baseline 尚未冻结，未执行 model evaluation，故 formal comparison 保持关闭。
- 最终回归命令为 `PYTHONPATH=. python -m unittest discover -s research_strategy_optimization/tests -p 'test*.py' -v`；新增 P2.1 evaluator、shortcut-probe、v0.5 signature 测试已纳入集合，PyTorch/scikit-learn 依赖按环境明确通过或跳过，不把 optional extra 缺失伪装成正式模型实验。
- C/D/E 既有 v0.3 suite 的模型数值没有因本轮审计重训；本轮只对既有完整 suite 做确定性边界/指标审计，并明确保留其 diagnostic-only 状态。若要宣称新的 v0.4 模型数值，必须另行授权并重训。

当前消费/冻结边界可概括为：

| 产物 | 状态 | 可支持的结论 |
| --- | --- | --- |
| v0.4 extended、v0.4 formal、旧 P2 | `diagnostic_v04_consumed` | 历史执行、收据和失败原因审计；不支持 post-fix 选择或 paper final |
| P2.1 fresh diagnostic | `completed_cpu_diagnostic` | 修复后评测器、反事实无泄漏、question-macro/seed×question/family-LOO、3-seed 与 10-seed bounded sweep、奖励敏感性/family 审计及 shortcut 诊断；仍非 formal comparison |
| v0.5 public/private boundary | `pending_clean_commit_tag` | 结构/签名/独立 evaluator 40/40 gates 与 reward-sensitivity 诊断；clean tag、baseline receipt 和模型评测尚未完成 |
| P3 小模型/LLM | `NO-GO` / fail-closed | 不执行 LoRA/QLoRA、在线 RL 或 7B 评测 |

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

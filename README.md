# PESCO 实现交付

这里是 `PESCO_Research_Algorithm_and_Experimental_Plan_v1.md` 的可运行、CPU 优先参考实现。目录按计划第 20 节拆分为环境、可信证据、策略算法、基线、评价和可视化；所有源码、配置、数据清单、审计记录和图片都保存在本目录。

Tier 0 核心不依赖第三方包；Tier 1 NumPy 环境和 PNG/SVG 图表需要 `numpy`、`matplotlib`（见 [requirements.txt](requirements.txt)）。反馈实验 C/D/E 与 P2/P3 的真实可微 MLP 更新需要单独安装与平台匹配的 PyTorch；标准 requirements 不强制拉取体积较大的平台专用 wheel，安装提示见 [requirements-optional.txt](requirements-optional.txt)。

## 当前已实现

- Tier 0 隐藏配对世界：Supported、Refuted、Insufficient、Invalid 四类动态证据状态。
- 严格证据规则：Invalid 优先；置信区间跨越 `delta_min` 时判定 Insufficient，而不是把它误判为 Refuted。
- 独立可信验证器：有效性、效应区间、独立确认种子、哈希审计和预算记录。
- 同状态快照分支：共同随机数、快照哈希/恢复、留一优势。
- 跨世界偏好反转：置信下界/上界门控、双重差分、PESCO flip loss。
- 严格对数评分、因子化状态损失、PPO clipped option loss、策略 token mask、约束目标组件。
- 无提示新路径证书：结构差异、真实执行、有效性、独立确认和收益下界全部通过才计入。
- CPU tabular PESCO-Offline / PESCO-Full 参考训练循环，以及共享环境/验证器/预算的基线适配器。
- Tier 1 v0.3 多问题机制实验：12 个独立问题（4 类机制 × 3 variants）、48 个隐藏世界、768 个 exploration seed 观测；共享 branch dataset 上运行小型 PyTorch MLP 的 SFT、GRPO-Terminal、GRPO-FourState、StateGateOnly、PESCO branch/flip-loss controls 和 Evidence-Gated SMOPD 教师蒸馏。该 suite 是 genuine CPU differentiable reference，不是 LLM 或论文复现。
- Tier 1 的硬案例不是把隐藏目标动作塞进输入：相同证据状态在不同注册机制下会产生不同 evaluator branch winner；公开任务族仍是允许的上下文，字节级 Observation collision 不作为门禁。
- 反馈实验 A/B/C/D/E/F 均有独立 machine-readable 产物：A 环境正确性、B 外部 checkpoint 真实零样本、C state-reward、D branch ablation、E flip ablation，以及 F 固定动作 MVP 的 open-ended discovery 边界；A–E 是诊断性证据，F 明确不宣称开放式发现已完成。
- 反馈实验 C/D/E 的统一可微 suite 产物位于 `artifacts/tier1_differentiable_suite/suite.json`：C 对比普通 state-reward（SFT/GRPO-Terminal/GRPO-FourState/StateGateOnly），D 是 branch ablation，E 是 confirmed flip-loss ablation 并包含 Evidence-Gated SMOPD teacher-distillation；三者共用冻结 Tier 1 v0.3 branch dataset 和 matched compute，均为机制诊断而非正式 final ID/OOD 结论。
- 反馈新增的 v0.4 extended 双轨 CPU 诊断位于 `artifacts/tier1_v04_extended/`（更早的 `artifacts/tier1_v04/` 也已标记 consumed）：8 个机制族、64 个问题、256 个世界、8 个探索/8 个独立确认种子；dev/diagnostic-OOD 同问题确认反转为 142/131，canonical posterior-EU/VOI pass 覆盖每族 variant-3 anchor，并产生 64 个决策和 64 条多步轨迹。该目录现在标记为 `diagnostic_v04_consumed`，仅可用于历史审计/调试；`consumption_notice.json` 说明旧 latent-template 边界与评测器口径已被 P2.1 取代，不能作为修改算法后的 final 或 paper comparison。
- P2 十种子晋级实验位于 `artifacts/tier1_p2_v04_ten_seed/`，在含 atomic reward receipts 的修订 raw-evidence 数据上完成 10 seeds/64 optimizer steps；其历史 promotion 为 NO-GO。正式 final-ID/final-OOD 划分的完整 P2 结果位于 `artifacts/tier1_p2_v04_formal_final_ten_seed/`，同样是历史 NO-GO。两者及旧 `tier1_p2_v04_ten_seed_v3/` 均有 `status=diagnostic_v04_consumed` 的消费通知；旧 exact top-1 FlipAcc/seed-only CI 不再用于算法选择。P3 小模型门禁位于 `artifacts/tier1_p3_small_model_gate/`，因 P2、依赖和 CUDA 前置条件失败而 fail-closed，未执行 LoRA/QLoRA。
- 旧 formal final 环境位于 `artifacts/tier1_v04_formal_final/`：train/dev 各 24 clusters、final ID 24 clusters（6 个机制族）、final OOD 20 clusters（2 个整族留出），confirmed reversals 为 196/114。环境 receipts 保留用于审计，但 latent-template overlap 使其标记为 `diagnostic_v04_consumed`，`formal_comparison_authorized=false`；不得再作为独立 clean final。默认命令只生成锁定结构 manifest；全量环境采集需显式 authorization receipt：`PYTHONPATH=. python scripts/run_tier1_v04_formal_final.py --output artifacts/tier1_v04_formal_final --collect-environment-audit --authorization-file artifacts/formal_final_access_authorization.json`。
- P2.1 fresh diagnostic 位于 `artifacts/tier1_p21_diagnostic/`：64 个问题、256 个世界、8 个 exploration/8 个 confirmation seeds、395 个同问题 reversal、按 question 归一化权重，以及 256/256 counterfactual raw-observation leakage audit；该产物保持 `diagnostic_only=true` 和 `formal_comparison_authorized=false`。`artifacts/tier1_p21_shortcut_probe/` 已完成 Logistic、Random Forest、GBDT shortcut probes，并在无 scikit-learn 时明确记录 NumPy fallback。
- P2.1 algorithm diagnostic 位于 `artifacts/tier1_p21_algorithm_diagnostic/`：在 fresh train/tune/promotion split 上隔离运行七个注册方法（SFT、GRPO-Terminal、GRPO-FourState、BranchOnly、NoFlipLoss、Full、SMOPD-inspired adapter），并额外运行 PESCO-Constrained-PCGrad、branch/state↔flip gradient-cosine probes 与 shortcut baselines；baseline 在 tune split 锁定。shortcut strict mode 在 sklearn 不可用时明确 `fail_closed`，结果保持 `diagnostic_only=true`，不打开 formal comparison、LoRA 或在线 RL。
- 另有一个隔离的 3-seed×4-method bounded sweep 位于 `artifacts/tier1_p21_seed_sweep_diagnostic/`（seeds 17/23/29，32 optimizer steps）；它用 seed→question 两层 bootstrap 和 mechanism-family leave-one-out 检查方向稳定性。该补充仍是 diagnostic-only：Full−tune-selected baseline 的 normalized-regret CI 为 `[-0.0506, 0.0000]`，Full−NoFlip 的 PairRankAcc CI 为 `[0.0766, 0.1779]`，不能替代预注册的 10-seed/final gate。
- 另完成一组 10-seed×4-method 的补充扫参（seeds 17/23/29/31/37/41/43/47/53/59，32 optimizer steps，40/40 workers 无失败），位于 `artifacts/tier1_p21_seed_sweep_10seed_diagnostic/`：Full−tune-selected normalized-regret 点估计 `-0.00802`，95% CI `[-0.02359, 0.00268]`；Full−NoFlip PairRankAcc 点估计 `+0.13803`，95% CI `[0.10503, 0.17402]`。它仍明确是 diagnostic-only，不替代正式冻结门禁。
- `artifacts/tier1_p21_sensitivity_audit/` 记录了保留逐题逐世界策略记录上的全局 atomic-reward 权重 `±20%` 扰动（1000 次共享权重抽样）、top1−top2 tie/non-tie 稳定性、family-wise 方向和 selected-action confirmation/validity 分母；该审计是诊断性二次分析，不重新训练策略，也不打开 formal comparison。
- v0.5 仅作为 `final_boundary_rehearsal_consumed` 保留：其公开模块可重建 final recipes/latent formula/targets/IDs/seeds，故不得作为论文 final 或 formal comparison。新的 v0.6 只允许由私有 evaluator bundle 生成；公开仓库仅保留接口、计数与 commitment 校验。v0.5 的结构审计与 ±20% reward sensitivity 结果仍可用于历史边界审计，但不再签署或复用。
- PESCO-Full 的 CPU 参考循环会对确认的跨世界 flip loss 直接执行表格 logits 的解析梯度更新，并在 `training_log_full.json` 中记录更新前后 loss、梯度范数、参数更新范数和参数 checksum；这不是 LLM autograd 训练的替代品，而是可验证的机制参考。
- 计划指标：VRS、状态 Macro-F1/混淆矩阵、Pairwise Reversal Ranking Accuracy（主要 flip-loss 对齐指标）、普通 action exact top-1、遗憾、有效/无效切换、负结果接受、不足处理、无效修复、VNPR、FDR、selected-action 复现率、成本和聚类 bootstrap。Pair/switch/repair/insufficient/confirmation 指标均使用显式条件分母；reversal 按 question macro 归一化，CI 使用 seed×question 两层 bootstrap 并报告 family LOO，不把 exact top-1 冲突写成 FlipAcc。
- PNG/SVG/CSV/Markdown 报告流水线，支持 demo、实际 Tier 0 pilot 和 JSON/JSONL 输入。

## 一键运行最小实验

在工作区根目录执行：

```bash
python PESCO/scripts/run_pesco_pilot.py \
  --output PESCO/artifacts/pesco_pilot \
  --epochs 8

python -m PESCO.visualization \
  PESCO/artifacts/pesco_pilot/results.json \
  --output PESCO/artifacts/pesco_pilot/report \
  --bootstrap 200
```

该 pilot 会真实执行 4 世界 × 4 动作 × 4 探索 seed，即 64 个 seed-level 分支观测（同状态记录为 16 个分支组，并另保存 64 次单 seed 配对审计重跑），另使用独立 confirmation seeds；确认只对 Supported/Refuted 的 12 个决策分支执行，因此 `mvp_counts.json` 中有 48 次 held-out confirmation 实验。`mvp_counts.json` 给出机器可检查的计数，`tier0_go.json` 给出四状态可达、动态转移、一步最优动作和 evidence-blind gap 门槛，`hypothesis_registry.json` 保存实验前冻结的先验与证据链，`negative_controls.json` 执行方案第 18.2 节的证据打乱、隐藏证据、文件名替换、无效高分、可靠负结果、表面新颖和假复现检查；`mvp_gate.json`、`reversal_pairs.json`、`audit_ledger.jsonl` 是其余验收证据。`baseline_manifest.json` 标明哪些命名基线是 CPU adapter，而非外部论文复现；每行 `formal_comparison_eligible=false`，避免把 adapter smoke rows 当作正式外部算法比较，`Rule-Based` 是透明四状态控制，`Search-Only` 是 oracle diagnostic。

固定四动作 pilot 不把 `switch_to_alternative_method` 当作自主发现；`discovery_policy.json` 对所有方法统一关闭 discovery utility。旧版 `certificate_*.json` 中可能保留 `certificate_pass=true`/`autonomous=true`，但已机器标记为 `legacy=true`、`artifact_scope=legacy_certificate_evidence`，并由 `legacy_certificates_manifest.json` 集中登记；这些字段只代表历史输出，不授权当前 discovery claim。方法切换后，结果记录以新的 `active_hypothesis_id`（H_B）和 H_B 真值计算 proper score，并保留 `hypothesis_beliefs`、`beliefs_before`、`beliefs_after`，不会把方法 B 的概率继续按 H_A 评分。

只想运行可视化 smoke test：

```bash
python -m PESCO.visualization --demo --output PESCO/artifacts/demo_report
python -m PESCO.visualization --tier0 --output PESCO/artifacts/tier0_report

python PESCO/scripts/run_tier1_smoke.py PESCO/artifacts/tier1_smoke.json

python PESCO/scripts/run_pesco_ablation.py No-FlipLoss \
  --output PESCO/artifacts/ablation_no_flip.json --epochs 2
```

`demo_report` 是确定性合成数据的管线测试；`tier0_report` 是实际 Tier 0 模拟器/验证器的 pilot 诊断。二者都不是大模型科学结论。
`tier1_smoke.json` 额外验证 NumPy 分组数据、混杂和泄漏任务族，以及修复后的有效性恢复；它同样只是小规模诊断。
消融脚本对已有 CPU 开关的设置执行真实 tabular 训练，其余设置只生成“registered, not run on final splits”状态，不伪造正式统计结果。
本仓库不内置冻结模型或 checkpoint。没有 checkpoint 的干净 checkout 会保留 `zero_shot_diagnostic.json`（CPU policy 的公开输入/输出诊断），并将 `stage_3_zero_shot` 保持为 NO-GO；该文件不能解释为真实模型 zero-shot 结果。当前工作区使用仓库外的本地冻结 checkpoint 完成了 fresh B 诊断运行；给定同一外部 checkpoint 可复核，产物保存在 `artifacts/tier1_zero_shot.json` 及其别名 `artifacts/experiment_b_real_zero_shot.json`。`experiment_b_diagnostic.json` 与 `experiment_c_diagnostic.json` 仍是 pilot 中的 fail-closed 脚手架：它们不能替代真实 B，也不授权正式外部比较。

实验 B 的真实模型入口是 `scripts/run_tier1_zero_shot.py`。它只把 Tier 1 v0.3 的公开 observation 放入 prompt，记录 checkpoint digest、输入白名单和独立审计；当前 fresh 产物的 `status=completed_diagnostic`、`pass=true` 表示 48 条当前 benchmark world-level rows 已完成 frozen-model forward，但 `diagnostic_only=true` 且 `formal_comparison_authorized=false` 仍然有效。没有 checkpoint 时 runner 会 fail-closed，而不会生成伪造的真实模型结果。

该 fresh B 诊断共有 48 条 world-level rows；真实模型的 action audit accuracy 为 0.3541667、state audit accuracy 为 0.25，且 48/48 行均输出 `A=continue_current_method, S=supported`，这是需要保留的 zero-shot 输出坍缩诊断。它不能解读为方法优于基线或已完成正式 zero-shot 科学比较。保存的 artifact 与当前冻结 benchmark 均绑定到 digest `sha256:542ec9…`，48/48 条公开 observation 均由 frozen checkpoint 在本轮 forward；`freshness_audit.status=fresh_current`、`benchmark_freshness_match=true` 和 `current_full_forward_completed=true` 已通过机器审计。checkpoint 不入库，故 fresh B 结果仍只代表可复核的外部 checkpoint 诊断，不打开 `stage_3_zero_shot` 或 formal comparison。

上述 48-row fresh diagnostic 与新增 robustness gate 是两件事。`artifacts/tier1_zero_shot_robustness_null_only.json` 只完成了同一唯一 checkpoint 的 null-prompt prior 和完整 hash 审计；action-letter rotation、三模板、chat/constrained controls 已实现但未完成模型 forward，多 checkpoint instruction panel 也不存在，因此 robustness `pass=false`。不能用旧 B 的 48-row forward 代替这些新增鲁棒性要求。

```bash
python PESCO/scripts/run_tier1_zero_shot.py \
  --checkpoint /path/to/frozen/local-checkpoint \
  --output PESCO/artifacts/tier1_zero_shot.json
```

运行 Tier 1 小型可微机制实验：

```bash
python PESCO/scripts/run_tier1_differentiable_suite.py \
  --output PESCO/artifacts/tier1_differentiable_suite \
  --epochs 16 --max-optimizer-steps 128
```

它输出每个方法的训练 loss、flip/state/option 分量、entropy、reference KL、gradient norm、ESS、optimizer/teacher steps，以及 train/dev/diagnostic_ood 的 action accuracy、regret、state Macro-F1。所有方法使用同一 branch dataset、动作权限、seed 和 optimizer-step 上限；`diagnostic_ood` 不代表正式 final OOD。计数语义固定为：48 个 `question_world_group`，每组 4 个候选动作即 192 条 `action_level` rows，另有 768 条 `seed_level` observations；产物同时保留旧 `branch_groups` 别名但不再把它解释成 192。
该命令生成的 `suite.json` 是 C/D/E 的合并审计产物；`tier1_v03/` 目录中的 `tier1_go.json`、`summary.json`、`experiment_a_environment_correctness.json` 和 benchmark manifest 则是 A 的独立问题/世界/seed 校准证据。`artifacts/tier1_differentiable_suite/experiment_f_discovery_boundary.json` 明确记录了当前固定四动作 MVP 的发现边界：发现 utility 为 0，尚未执行开放式候选生成，因此不把 `switch_to_alternative_method` 冒充为自主发现。
如需只检查 48/192/768 的跨文件一致性，可运行 `python PESCO/scripts/audit_tier1_count_semantics.py`；该审计不重新执行实验。

## 测试

```bash
PYTHONPATH=PESCO python -m unittest discover \
  -s PESCO -p 'test*.py' -v

python -m compileall -q PESCO/research_strategy_optimization PESCO/visualization
```

当前测试覆盖快照隔离、共同 seed、LOO 优势、反转置信门槛、解析 flip-loss 更新、四状态转移、严格评分、隐藏字段白名单、假设/证据哈希链、发现证书、目标函数、多重比较校正、Tier 1 多问题数据集、可微方法训练、atomic reward receipts、P2.1 counterfactual leakage/PairRank/selected-receipt 边界、shortcut probes 和 v0.5 signature provenance；测试集合会随可选 extra 展开，运行上述命令即可得到当前环境的通过/跳过明细。

其中可微训练与 P2/P3 诊断测试属于可选 PyTorch extra：请按主机 CPU/CUDA 平台从 PyTorch 官方安装选择器安装对应 wheel（不要直接假设某个 CUDA 版本），再运行上述测试即可；未安装 PyTorch 的干净环境会将这些模块标记为 skipped，不会因模块导入失败而中断标准 unittest discovery。快捷模型探针的 scikit-learn 依赖见 [requirements-optional.txt](requirements-optional.txt)，缺少该依赖时仅运行明确标注的 NumPy fallback（`--strict-sklearn` 可改为 fail-closed）。

## 明确的阶段边界

本交付完成的是计划要求的 Tier 0 / CPU 参考闭环和可审计实验基础设施，不把它冒充为 Tier 2 LLM 后训练结果。`PESCO/artifacts/pesco_pilot/training_authorization.json` 明确记录：

- CPU reference loop：GO；
- Tier 1 online RL：NO-GO；
- Tier 2 LLM RL / QLoRA：NO-GO。

进入真实模型训练前仍需按计划冻结模型与执行器、五分割数据、污染审计、容器/权限隔离、强基线和最终 ID/OOD 评测。

## 主要文件

- [研究计划](PESCO_Research_Algorithm_and_Experimental_Plan_v1.md)
- [实现状态与验收矩阵](docs/IMPLEMENTATION_STATUS.md)
- [机器可读实现覆盖清单](docs/implementation_manifest.json)
- [Tier 0 pilot 报告](artifacts/pesco_pilot/report/report.md)
- [Tier 0 实际分支报告](artifacts/tier0_report/report.md)

反馈实验产物：

- [A 环境正确性](artifacts/tier1_v03/experiment_a_environment_correctness.json) · [Tier 1 GO 校准](artifacts/tier1_v03/tier1_go.json)
- [B 真实 zero-shot fresh 产物](artifacts/tier1_zero_shot.json) · [B 当前 freshness 状态](artifacts/tier1_zero_shot_current_status.json) · [B robustness fail-closed 审计](artifacts/tier1_zero_shot_robustness_null_only.json)
- [C state-reward](artifacts/tier1_differentiable_suite/experiment_c_state_reward.json) · [D branch ablation](artifacts/tier1_differentiable_suite/experiment_d_branch_ablation.json)
- [E flip-loss/SMOPD](artifacts/tier1_differentiable_suite/experiment_e_flip_ablation.json) · [F discovery boundary](artifacts/tier1_differentiable_suite/experiment_f_discovery_boundary.json) · [C–E 合并 suite](artifacts/tier1_differentiable_suite/suite.json)
- [v0.4 extended 双轨 benchmark](artifacts/tier1_v04_extended/tier1_v04_extended_go.json) · [v0.4 raw dataset](artifacts/tier1_v04_extended/dataset_raw_evidence.json) · [v0.4 posterior/VOI decisions](artifacts/tier1_v04_extended/decisions.json) · [v0.4 run manifest](artifacts/tier1_v04_extended/run_manifest.json)
- [v0.4 consumed notice](artifacts/tier1_v04_extended/consumption_notice.json) · [formal v0.4 consumed notice](artifacts/tier1_v04_formal_final/consumption_notice.json)
- [P2.1 fresh diagnostic](artifacts/tier1_p21_diagnostic/p21_diagnostic_result.json) · [counterfactual leakage audit](artifacts/tier1_p21_diagnostic/counterfactual_leakage_audit.json) · [shortcut probes](artifacts/tier1_p21_shortcut_probe/shortcut_probe_result.json)
- [P2.1 algorithm diagnostic](artifacts/tier1_p21_algorithm_diagnostic/p21_algorithm_diagnostic.json) · [algorithm diagnostic README](artifacts/tier1_p21_algorithm_diagnostic/README.md)
- [P2.1 isolated seed sweep](artifacts/tier1_p21_seed_sweep_diagnostic/seed_sweep_result.json)
- [P2.1 ten-seed supplementary sweep](artifacts/tier1_p21_seed_sweep_10seed_diagnostic/seed_sweep_result.json) · [reward-weight/family sensitivity audit](artifacts/tier1_p21_sensitivity_audit/sensitivity_audit.json)
- [P2.1 constrained-PCGrad receipt](artifacts/tier1_p21_constrained_diagnostic/constrained_result.json)
- [formal final historical gate](artifacts/tier1_v04_formal_final/formal_final_go.json) · [whole-family holdout audit](artifacts/tier1_v04_formal_final/whole_family_holdout_audit.json) · [formal consumed notice](artifacts/tier1_v04_formal_final/consumption_notice.json)
- [P2 extended historical gate](artifacts/tier1_p2_v04_ten_seed/p2_result.json) · [P2 formal historical gate](artifacts/tier1_p2_v04_formal_final_ten_seed/p2_result.json) · [P3 fail-closed gate](artifacts/tier1_p3_small_model_gate/small_model_gate.json)
- [v0.5 freeze audit](artifacts/tier1_v05_frozen_final/v05_freeze_audit.json) · [v0.5 independent audit](artifacts/tier1_v05_frozen_final/independent_audit.json) · [v0.5 reward sensitivity summary](artifacts/tier1_v05_frozen_final/reward_sensitivity_summary.json) · [v0.5 pending freeze receipt](artifacts/tier1_v05_frozen_final/freeze_receipt.json) · public evaluator contract (private hidden bundle is not published)
- [P2.2 3-seed common-SFT screening](artifacts/tier1_p22_screening_final_objective_v2/p22_matrix_result.json) · [global tune-only baseline selection](artifacts/tier1_p22_screening_final_objective_v2/global_baseline_selection.json) · [P2.2 all-method 10-seed frozen matrix](artifacts/tier1_p22_frozen_listwise_pcgrad_10seed/p22_matrix_result.json) · [P2.2 gate receipt](artifacts/tier1_p22_frozen_listwise_pcgrad_10seed/p22_gate_receipt.json) · [strict Logistic/GBDT probes](artifacts/tier1_p22_strict_shortcut/shortcut_probe_result.json)
- [P2.2 family-LOO and reward sensitivity](artifacts/tier1_p22_frozen_listwise_pcgrad_10seed/family_loo.json) · [method-ranking reward audit](artifacts/tier1_p22_frozen_listwise_pcgrad_10seed/method_reward_sensitivity.json) · [gradient-conflict audit](artifacts/tier1_p22_frozen_listwise_pcgrad_10seed/gradient_conflict_summary.json) · [v0.6 public leakage audit](artifacts/tier1_v06_public_leakage_audit.json) · [v0.6 private-boundary signature audit](artifacts/tier1_v06_evaluator_private/private_boundary_audit.json)
- [P2.2 design-only power analysis](artifacts/tier1_p22_power_analysis/power_analysis.json) · [repair-safety sensitivity 10-seed](artifacts/tier1_p22_frozen_listwise_safety_10seed/p22_gate_receipt.json)
- [demo 可视化报告](artifacts/demo_report/report.md)
- [可视化输入契约](visualization/schema.json)
- [冻结协议](research_strategy_optimization/configs/freeze/pesco_v0_2.yaml)

覆盖清单可独立审计：

```bash
python PESCO/scripts/audit_implementation.py
```

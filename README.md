# PESCO 实现交付

这里是 `PESCO_Research_Algorithm_and_Experimental_Plan_v1.md` 的可运行、CPU 优先参考实现。目录按计划第 20 节拆分为环境、可信证据、策略算法、基线、评价和可视化；所有源码、配置、数据清单、审计记录和图片都保存在本目录。

Tier 0 核心不依赖第三方包；Tier 1 和 PNG/SVG 图表需要 `numpy`、`matplotlib`（见 [requirements.txt](requirements.txt)），PyTorch 仅用于可选的可微损失测试。

## 当前已实现

- Tier 0 隐藏配对世界：Supported、Refuted、Insufficient、Invalid 四类动态证据状态。
- 严格证据规则：Invalid 优先；置信区间跨越 `delta_min` 时判定 Insufficient，而不是把它误判为 Refuted。
- 独立可信验证器：有效性、效应区间、独立确认种子、哈希审计和预算记录。
- 同状态快照分支：共同随机数、快照哈希/恢复、留一优势。
- 跨世界偏好反转：置信下界/上界门控、双重差分、PESCO flip loss。
- 严格对数评分、因子化状态损失、PPO clipped option loss、策略 token mask、约束目标组件。
- 无提示新路径证书：结构差异、真实执行、有效性、独立确认和收益下界全部通过才计入。
- CPU tabular PESCO-Offline / PESCO-Full 参考训练循环，以及共享环境/验证器/预算的基线适配器。
- 计划指标：VRS、状态 Macro-F1/混淆矩阵、FlipAcc、遗憾、有效/无效切换、负结果接受、不足处理、无效修复、VNPR、FDR、复现率、成本和聚类 bootstrap。
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

该 pilot 会真实执行 4 世界 × 4 动作 × 4 探索 seed，即 64 个 seed-level 分支观测（同状态记录为 16 个分支组，并另保存 64 次单 seed 配对审计重跑），另使用独立 confirmation seeds；确认只对 Supported/Refuted 的 12 个决策分支执行，因此 `mvp_counts.json` 中有 48 次 held-out confirmation 实验。`mvp_counts.json` 给出机器可检查的计数，`tier0_go.json` 给出四状态可达、动态转移、一步最优动作和 evidence-blind gap 门槛，`hypothesis_registry.json` 保存实验前冻结的先验与证据链，`negative_controls.json` 执行方案第 18.2 节的证据打乱、隐藏证据、文件名替换、无效高分、可靠负结果、表面新颖和假复现检查；`mvp_gate.json`、`reversal_pairs.json`、`audit_ledger.jsonl` 是其余验收证据。`baseline_manifest.json` 标明哪些命名基线是 CPU adapter，而非外部论文复现。

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

## 测试

```bash
PYTHONPATH=PESCO python -m unittest discover \
  -s PESCO -p 'test*.py' -v

python -m compileall -q PESCO/research_strategy_optimization PESCO/visualization
```

当前测试覆盖快照隔离、共同 seed、LOO 优势、反转置信门槛、四状态转移、严格评分、隐藏字段白名单、假设/证据哈希链、发现证书、目标函数、多重比较校正和可视化报告；当前全量 stdlib suite 为 38 项。

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
- [demo 可视化报告](artifacts/demo_report/report.md)
- [可视化输入契约](visualization/schema.json)
- [冻结协议](research_strategy_optimization/configs/freeze/pesco_v0_2.yaml)

覆盖清单可独立审计：

```bash
python PESCO/scripts/audit_implementation.py
```

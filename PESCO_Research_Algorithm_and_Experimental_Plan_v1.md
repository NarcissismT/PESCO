# PESCO：面向自主科研探索的配对世界证据条件化策略优化研究计划

> 文档版本：v1.0  
> 编制日期：2026-08-25  
> 研究性质：算法研究与实验设计方案，尚未执行正式实验  
> 核心目标：训练模型根据真实实验反馈自主判断证据、修正研究策略，并发现经过独立验证的新路径

---

## 目录

1. 研究背景、动机与问题界定
2. 文献定位与原创性边界
3. 科学问题、研究假设与成功标准
4. 形式化建模
5. 实验世界与动态证据状态
6. 独立可信验证器
7. PESCO 算法整体设计
8. 严格评分规则与认知奖励
9. 同状态实验分支与策略级信用分配
10. 跨世界策略偏好反转
11. 自主新路径发现与验证约束
12. 完整优化目标与训练流程
13. 理论研究计划
14. 实验环境设计
15. 数据集与环境划分
16. 基线方法与公平比较
17. 评价指标
18. 消融实验与负对照
19. 统计分析、重复实验与错误发现控制
20. 工程实现方案
21. 初始超参数与资源预算
22. 项目阶段、里程碑与决策门槛
23. 与现有项目冻结协议的衔接
24. 风险、失败模式与应对
25. 最小可行实验
26. 最终结果汇报模板
27. 预期贡献与论文结构
28. 参考文献

---

## 1. 研究背景、动机与问题界定

### 1.1 研究背景

目前的大语言模型智能体已经可以完成相当复杂的科研执行工作，例如：

- 理解研究目标与现有代码。
- 编写并运行实验程序。
- 调整超参数。
- 调用训练框架。
- 读取实验指标。
- 修复工程错误。
- 汇总实验结果。

但是，这些能力不等于具备自主科研策略。

一个模型可能能顺利执行十次训练实验，却始终沿着最初选择的方法进行局部调整，即使已有证据表明这个方法存在数据泄漏、统计功效不足、假设不成立，或者明显弱于另一条研究路径。

本研究重点区分：

$$
\text{Execution Capability}
\neq
\text{Research Strategy Capability}.
$$

执行能力回答：

> 模型能否把已经确定的研究方法落实成可运行实验？

策略能力回答：

> 当实验产生新的证据时，模型能否重新判断应该继续、补充、修复、停止，还是切换到另一条研究路径？

2026 年的研究发现，在 3557 对相邻后训练实验中，只有约 2.1% 真正尝试了替代策略；大部分智能体在确定初始方案后，只进行方案内部的局部调整。[What is Missing from AI Post-Training AI](https://arxiv.org/abs/2608.19072)

### 1.2 研究问题

本项目研究以下问题：

> 能否设计一种后训练算法，使模型在没有人工提供具体方法提示的情况下，根据真实实验输出自主判断证据状态，并选择经过真实执行与独立验证、更符合当前证据的研究策略？

其中，关键能力包括：

1. 判断现有实验是否支持研究假设。
2. 判断现有实验是否可靠反驳研究假设。
3. 区分证据不足与真实反驳。
4. 判断实验是否受到数据泄漏、混杂或错误评测影响。
5. 识别何时应该继续原有路径。
6. 识别何时应该追加样本、重复实验或修复实验。
7. 识别何时应该接受负结果。
8. 识别何时应该提出新的研究假设或方法。
9. 对新路径进行真实执行、独立验证和预算控制。

### 1.3 不属于本研究直接目标的内容

本计划不直接宣称：

- 模型已经能够发现此前任何人类都不知道的新科学定律。
- 只要方法名称未出现在提示中，就可以证明全球范围内的原创性。
- 所有科研问题都能由四种离散证据状态完整表示。
- 真实科学结论可以完全由单个验证器或单个统计检验自动决定。
- 仅凭自建 benchmark 得分提升，就能证明跨所有领域的自主科研能力。

本研究最初能够合理争取的结论是：

> 在定义清晰、可执行、可复现的研究环境中，模型能够根据实验反馈形成更可靠的策略修正，并在部分任务上自主发现经过独立验证的新方法路径。

---

## 2. 文献定位与原创性边界

### 2.1 与现有工作的关系

| 工作 | 已有能力 | 对本研究的启发 | 本研究需要避免的重复 |
| --- | --- | --- | --- |
| SMOPD | 多奖励专家训练与在线策略蒸馏 | 能将不同能力整合到统一策略 | 不能只提出多个证据奖励和教师混合 |
| GDPO | 分别归一化多个奖励维度 | 提供多奖励优化基线 | 不能只调整四类证据奖励权重 |
| GiGPO | 同状态动作分组与多层优势估计 | 支持同状态研究动作比较 | 同状态优势估计本身不是创新 |
| Tree-GRPO | 树状采样与过程信用分配 | 支持多路径研究分支 | 多分支树搜索本身不是创新 |
| Spark | 关键状态动态分叉 | 支持有限预算下的选择性探索 | 不能只提出关键状态多尝试几次 |
| SciDisco / DiscoPO | 科研证据图和逐轮验证奖励 | 提供科学过程监督基线 | 不能只把实验步骤完成情况转成奖励 |
| TCPO | 多轮信用分配、未来收益和固定历史反事实 | 支持延迟研究动作价值估计 | 不能只提出固定历史分叉 |
| CVT-RL | 反事实信用、验证约束、防奖励作弊 | 提供强约束与可信验证基线 | 不能只组合反事实和有效性约束 |
| Ecpo | 同状态动作统计校准和方差门控 | 支持控制随机种子与小样本噪声 | 不能只提出动作优势收缩估计 |
| IGPO | 基于信息增益的多轮策略优化 | 支持从认知变化构造过程奖励 | 信息增益奖励本身不是创新 |
| AutoDiscovery | 贝叶斯惊喜和树搜索 | 支持开放式问题与假设探索 | 贝叶斯惊喜和 MCTS 本身不是创新 |
| HEP | 可审计的假设、实验、证据和信念记录 | 支持结构化研究历史 | 假设登记与证据日志本身不是创新 |
| RTPO | 多轮交互的反向信用分配 | 支持长期研究轨迹训练 | 长轨迹反向更新本身不是创新 |

### 2.2 SMOPD 的真实定位

SMOPD 主要解决：

$$
\text{多个奖励信号如何在同一模型中有效优化}.
$$

其基本形式为：

$$
\mathcal L_{\mathrm{SMOPD}}
=
\mathcal L_{\mathrm{anchor}}
+
\lambda_{\mathrm{OPD}}
\mathcal L_{\mathrm{OPD}}.
$$

奖励专家混合为：

$$
p_{\mathrm{mix}}(v\mid s)
=
\sum_{m=1}^{M}
\alpha_m p_{\pi_m}(v\mid s).
$$

它能够帮助模型兼顾工具调用准确率、输出格式或有用性与安全性，但没有直接定义：

$$
Q(
\text{研究动作}
\mid
\text{实验历史},
\text{证据状态},
\text{预算}
).
$$

并且 SMOPD 已在附录测试过动态教师混合。因此，简单增加一个教师门控，不足以形成独立的方法贡献。[SMOPD](https://arxiv.org/abs/2608.03092)

### 2.3 本研究的候选原创性

本研究拟提出：

> 在同一研究问题的多个可执行实验世界中，学习由真实证据变化导致的研究策略偏好反转。

核心研究对象为：

$$
\operatorname{sign}
\left[
Q_{w_A}(s_A,u_1)
-
Q_{w_A}(s_A,u_2)
\right]
\neq
\operatorname{sign}
\left[
Q_{w_B}(s_B,u_1)
-
Q_{w_B}(s_B,u_2)
\right].
$$

也就是说：

- 相同研究问题。
- 相同或经过语义对齐的候选研究动作。
- 不同的真实实验机制和实验反馈。
- 不同的最优研究策略。

新算法需要学习：

> 当实验反馈改变时，策略偏好为什么发生变化，以及这种变化是否能提升真实科研价值。

### 2.4 原创性成立的必要条件

只有满足以下条件，才能较有说服力地将该方法视为独立算法研究：

1. 不只是四种证据标签与手工奖励。
2. 不只是标准 GRPO 或 PPO 加奖励权重。
3. 不只是推理阶段增加树搜索。
4. 明确提出跨世界策略偏好反转的学习机制。
5. 明确提出策略层与执行层的信用隔离。
6. 使用真实可执行实验分支，而不是只让 LLM 评判策略。
7. 对反驳、证据不足和实验无效采用不同的科学处理。
8. 在强基线、同预算和隐藏测试集上证明提升。
9. 将统计有效性、独立复现和错误发现控制纳入实验协议。

---

## 3. 科学问题、研究假设与成功标准

### 3.1 主要科学问题

**RQ1：证据响应性。**

模型是否能够针对相同研究问题，在不同实验反馈下改变研究策略？

**RQ2：可信负结果。**

模型是否能够将可靠反驳视为科学进展，而不是错误或需要掩盖的失败？

**RQ3：实验修复。**

模型是否能够判断表面成功结果实际上由泄漏或混杂产生，并主动修复实验设计？

**RQ4：证据不足。**

模型是否能够区分统计信息不足与真实的无效或反驳？

**RQ5：自主策略发现。**

在没有具体方法提示时，模型是否能够提出与当前路径实质不同、并通过独立验证的新策略？

**RQ6：能力内化。**

提升是否已经体现在模型参数中，而不仅仅来自推理阶段执行更多分支？

### 3.2 预注册研究假设

**H1：配对世界训练提高证据响应性。**

相对于不使用配对世界的同等训练方法，完整算法能够显著提高跨世界策略偏好反转准确率。

**H2：严格评分规则降低确认偏差。**

相对于支持加分、反驳扣分的奖励设计，严格评分规则能够提高可靠负结果接受率，并降低虚假支持率。

**H3：同状态真实分支提高策略决策质量。**

相对于只使用单条轨迹终局奖励的方法，同状态真实执行分支能够降低策略动作遗憾。

**H4：策略层信用分配提高有效切换率。**

相对于直接将整条轨迹回报广播给所有输出 token 的方法，策略层信用能够更准确地奖励必要且有效的研究路径切换。

**H5：经过验证的新颖性奖励降低探索坍塌。**

相对于普通终局奖励或文本新颖度奖励，经过独立复现门控的策略新颖性奖励能够提高有效方法多样性与经过验证的新路径发现率。

**H6：可信约束降低错误发现。**

相对于只通过负奖励惩罚无效实验的方法，硬门控或约束优化能够降低虚假结论、验证器绕过与数据污染利用。

### 3.3 不支持假设的情况

以下结果均应被视为研究假设未成立：

- 完整算法只提高实验执行率，但没有提高策略切换质量。
- 通过增加实验次数取得提升，但同预算下没有优势。
- 支持性结果增加，同时虚假发现率显著增加。
- 策略切换次数增加，但有效切换率没有提高。
- 训练任务提升明显，隐藏问题或未见方法族没有提升。
- 模型依赖真实 world 标签、文件名或其他泄漏信息。
- 移除推理期树搜索后，模型的单路径能力没有提升。
- 在强基线 TCPO、CVT-RL 或 Ecpo 面前没有额外收益。

---

## 4. 形式化建模

### 4.1 部分可观察研究环境

将科研过程建模为一个具有预算和可信约束的部分可观察决策过程：

$$
\mathcal M
=
(
\mathcal W,
\mathcal S,
\mathcal O,
\mathcal U,
P,
R,
\mathcal C,
B
).
$$

其中：

- $\mathcal W$：真实但不可直接观察的实验世界。
- $\mathcal S$：研究状态空间。
- $\mathcal O$：实验观测空间。
- $\mathcal U$：高层研究动作空间。
- $P$：实验执行后的环境状态转移。
- $R$：科学价值与认知价值奖励。
- $\mathcal C$：实验有效性、结论可信性与预算约束。
- $B$：每个研究问题的总实验预算。

### 4.2 世界状态与模型可见状态

真实环境状态：

$$
\omega_t
=
(
w,
\Theta,
D_t,
\eta_t,
c_t
).
$$

其中：

- $w$：数据生成机制、数据来源或真实实验设置。
- $\Theta$：目标假设及替代机制的真实状态。
- $D_t$：当前数据、程序、实验配置及可信验证状态。
- $\eta_t$：环境随机性，例如随机种子与观测噪声。
- $c_t$：剩余预算。

模型可见状态：

$$
s_t
=
(
q,
h_t,
o_t,
\widehat b_t,
c_t
).
$$

其中：

- $q$：研究任务描述。
- $h_t$：模型可见的研究历史。
- $o_t$：实际实验输出、日志和可访问统计量。
- $\widehat b_t$：模型报告的假设可信度。
- $c_t$：剩余预算。

必须保证：

$$
w\notin s_t
$$

且：

$$
\Theta\notin s_t.
$$

模型不能直接读取：

- 隐藏世界编号。
- 隐藏真实效应。
- 可信验证器内部标签。
- 最终确认集。
- 隐藏测试数据。
- 训练环境生成参数。

### 4.3 研究假设与固定估计对象

对每一个研究假设：

$$
H_j
$$

需要记录：

$$
H_j
=
(
\text{claim}_j,
\text{estimand}_j,
\delta_j,
\text{protocol}_j,
\text{timestamp}_j
).
$$

其中：

- $\text{claim}_j$：具体可检验的科学主张。
- $\text{estimand}_j$：明确的被估计对象。
- $\delta_j$：最小具有实际意义的效应。
- $\text{protocol}_j$：预先登记的检验与复现协议。
- $\text{timestamp}_j$：假设提出并冻结的时间。

如果模型提出一个新假设：

$$
H_{j+1},
$$

不能回头用已经观察过的测试结果对其进行无约束确认。

必须：

1. 在查看独立确认结果之前登记假设。
2. 分配新的验证数据或独立随机种子。
3. 记录与既有假设的关系。
4. 处理多重检验或错误率预算。

### 4.4 研究动作空间

初始动作空间建议包含：

$$
\mathcal U
=
\{
u_{\mathrm{continue}},
u_{\mathrm{replicate}},
u_{\mathrm{sample}},
u_{\mathrm{repair}},
u_{\mathrm{metric}},
u_{\mathrm{revise}},
u_{\mathrm{switch}},
u_{\mathrm{stop}}
\}.
$$

具体解释：

| 动作 | 科学含义 | 典型适用条件 | 不应机械执行的原因 |
| --- | --- | --- | --- |
| continue | 继续当前研究方法 | 已有证据支持且仍有改进空间 | 可能掩盖已经被反驳的路径 |
| replicate | 更换种子、数据或实现进行复现 | 已经出现可信但尚未确认的发现 | 已充分复现后可能浪费预算 |
| sample | 增加样本、重复次数或观测精度 | 结果不确定且追加成本合理 | 如果实验设计无效，更多样本无法解决问题 |
| repair | 修复泄漏、混杂、错误切分或实现问题 | 当前实验存在有效性缺陷 | 修复成本过高时应考虑停止 |
| metric | 改变或补充评测指标 | 原指标不能回答登记的研究问题 | 事后挑选指标可能导致作弊 |
| revise | 修正假设、机制或适用范围 | 原假设过强或与真实现象不一致 | 必须重新登记验证协议 |
| switch | 切换方法族或研究策略 | 当前策略被可靠反驳或出现更优机会 | 无意义切换会增加成本 |
| stop | 提交结论或报告证据不足 | 问题已经解决或预算耗尽 | 不能把无效实验伪装成可信结论 |

### 4.5 策略层与执行层

高层策略：

$$
u_t
\sim
\pi_{\theta}^{H}
(
\cdot
\mid
s_t
).
$$

低层执行：

$$
a_{t:t+\ell-1}
\sim
\pi_{\phi}^{L}
(
\cdot
\mid
s_t,u_t
).
$$

一个高层动作可以包含多个低层执行步骤：

$$
u_t
\Rightarrow
(
\text{编写代码},
\text{运行实验},
\text{收集结果},
\text{记录证据}
).
$$

研究初期建议固定或部分固定执行器：

$$
\phi=\phi_0.
$$

这样可以隔离：

> 改进究竟来自研究策略，还是来自更强的代码执行能力。

在完整模型训练阶段，再考虑：

$$
(\theta,\phi)
$$

的联合优化，但必须单独汇报执行层改进与策略层改进。

---

## 5. 实验世界与动态证据状态

### 5.1 实验世界不是固定证据标签

必须明确区分：

$$
w
\neq
z_t.
$$

其中：

- $w$ 是真实实验世界或生成机制。
- $z_t$ 是当前时刻的实验结论状态。

同一个世界中，证据状态可能发生动态变化：

$$
\text{Invalid}
\xrightarrow{\text{修复切分}}
\text{Insufficient}
\xrightarrow{\text{增加样本}}
\text{Refuted}.
$$

或者：

$$
\text{Insufficient}
\xrightarrow{\text{追加实验}}
\text{Supported}
\xrightarrow{\text{独立复现}}
\text{Confirmed}.
$$

因此：

> 四类状态不应被理解为每个问题永久不变的标签，而应被理解为研究过程中的动态证据状态。

### 5.2 证据状态的因子化

建议不直接将证据理解成单个四分类变量，而是分解为：

$$
e_t
=
(
v_t,
p_t,
d_t,
r_t^{\mathrm{rep}}
).
$$

其中：

- $v_t$：实验设计是否有效。
- $p_t$：信息精度或是否足以判断登记效应。
- $d_t$：效应方向与估计结果。
- $r_t^{\mathrm{rep}}$：是否已通过独立复现。

四分类标签由冻结规则产生：

$$
z_t
=
g(
v_t,
p_t,
d_t,
r_t^{\mathrm{rep}}
).
$$

### 5.3 Invalid：实验无效

满足任一冻结条件即可判定为无效：

- 训练集和测试集存在重复或污染。
- 划分方式造成目标泄漏。
- 随机种子或数据切分不是独立重复。
- 评测脚本被模型修改。
- 对照组不满足协议要求。
- 样本独立性假设明显不成立。
- 关键混杂因素未按预设要求控制。
- 统计方法与登记的估计对象不匹配。
- 从隐藏验证器或目标论文读取结果。
- 实验程序未真实执行或结果无法重放。

优先级：

$$
v_t=0
\Longrightarrow
z_t=\mathrm{Invalid}.
$$

即使表面性能很高，也不允许绕过这一规则。

### 5.4 Supported：有效证据支持假设

设：

$$
\widehat\Delta_t
$$

为效应估计。

设：

$$
[
L_t,U_t
]
$$

为适用的置信区间或置信序列。

对一个声称存在正向实际效应的假设，可以使用：

$$
v_t=1,
\qquad
L_t>\delta_{\min}.
$$

其中：

$$
\delta_{\min}>0
$$

是预先冻结的最小实际有效效应。

对于不同研究问题，应分别制定合理的支持规则。

### 5.5 Refuted：有效证据反驳假设

可靠反驳不等于：

$$
p>0.05.
$$

对一个主张正向效果超过：

$$
\delta_{\min}
$$

的研究假设，如果：

$$
v_t=1,
\qquad
U_t<\delta_{\min},
$$

则可以认为当前证据反驳该主张。

如果研究问题是：

> 是否存在具有实际意义的非零效果？

则应使用：

- 等效检验。
- 预先定义的等效区间。
- 充分精度的置信区间。
- 合理的独立复现。

例如：

$$
[
L_t,U_t
]
\subseteq
[
-\delta_{\min},
\delta_{\min}
].
$$

### 5.6 Insufficient：证据不足

如果实验有效，但区间仍跨越关键决策阈值：

$$
L_t
\leq
\delta_{\min}
\leq
U_t,
$$

且无法可靠支持或反驳登记假设，则判定：

$$
z_t=\mathrm{Insufficient}.
$$

不得采用：

> 因为没有显著性，所以已经反驳。

也不应使用仅由当前观察结果反推的事后功效作为唯一判断标准。

### 5.7 Confirmed：独立确认状态

四分类可以额外增加确认标识：

$$
\kappa_t
\in
\{
\text{unconfirmed},
\text{confirmed}
\}.
$$

例如：

$$
(
\mathrm{Supported},
\mathrm{confirmed}
)
$$

与：

$$
(
\mathrm{Refuted},
\mathrm{confirmed}
)
$$

都应具有较高科学价值。

---

## 6. 独立可信验证器

### 6.1 验证器职责

可信验证器需要独立完成：

1. 验证实验程序是否真实执行。
2. 检查数据与评测环境是否符合冻结协议。
3. 判断实验设计是否有效。
4. 计算效应量和不确定性。
5. 生成当前证据状态。
6. 判断是否存在重复使用测试集。
7. 执行独立确认实验。
8. 检查模型最终报告是否与实际实验输出一致。
9. 记录预算、耗时和随机种子。
10. 对最终结论生成不可被模型直接修改的审计记录。

### 6.2 验证器输入

可信验证器输入建议为：

~~~json
{
  "research_question_id": "rq_001",
  "hypothesis_id": "h_001",
  "world_id": "hidden_from_agent",
  "branch_id": "branch_004",
  "experiment_manifest": {
    "dataset_hash": "sha256:...",
    "code_hash": "sha256:...",
    "evaluator_hash": "sha256:...",
    "train_split_hash": "sha256:...",
    "test_split_hash": "sha256:...",
    "random_seeds": [17, 29, 41],
    "container_digest": "sha256:...",
    "network_policy": "none"
  },
  "observed_metrics": {
    "effect_estimate": 0.031,
    "confidence_interval": [0.014, 0.058],
    "sample_size": 1200
  },
  "registered_protocol": {
    "minimum_meaningful_effect": 0.01,
    "alpha": 0.05,
    "independent_confirmation_required": true
  }
}
~~~

### 6.3 验证器输出

~~~json
{
  "validity_pass": true,
  "evidence_state": "supported",
  "effect_estimate": 0.031,
  "confidence_interval": [0.014, 0.058],
  "independent_confirmation": {
    "performed": true,
    "passed": true,
    "confirmation_seeds": [103, 107, 109]
  },
  "scientific_claim_consistency": true,
  "budget_cost": {
    "execution_seconds": 41.7,
    "gpu_seconds": 0.0,
    "model_tokens": 1840
  },
  "audit_signature": "trusted_signature_placeholder"
}
~~~

### 6.4 安全边界

模型运行环境不能拥有：

- 验证器修改权限。
- 隐藏标签读取权限。
- 最终确认集读取权限。
- 最终评测输出文件写权限。
- 目标论文未授权内容读取权限。
- 最终测试世界编号读取权限。
- 验证密钥访问权限。

最小要求：

$$
\text{agent workspace}
\cap
\text{trusted evaluator secrets}
=
\varnothing.
$$

### 6.5 行政格式错误与科学失败分离

根据现有项目经验，必须区分：

- 输出格式不完整。
- JSON 字符串转义问题。
- 冻结的行政常量缺失。
- 程序真实无法执行。
- 科学假设错误。
- 实验设计无效。
- 研究策略选择失败。

不允许把 parser 格式噪声直接当作：

> 模型不具备科学探索能力。

同样也不允许通过 parser 修改：

- 科学假设。
- 算法代码。
- 方法类别。
- 实验目标。
- 评估指标。
- 问题定义。

---

## 7. PESCO 算法整体设计

### 7.1 方法名称

暂定：

> Paired-world Evidence-conditioned Strategy Counterfactual Optimization。

简称：

> PESCO。

名称为研究阶段暂定名，不代表已经完成商标、论文标题或命名冲突审查。

### 7.2 核心思想

PESCO 包含五个主要机制：

1. 高层科研策略与低层执行器分离。
2. 使用严格评分规则奖励真实认知改进。
3. 在同一实验状态下真实执行多个候选研究动作。
4. 在配对实验世界中学习策略偏好反转。
5. 只奖励经过独立确认的新研究路径。

### 7.3 信息流

~~~mermaid
flowchart TD
    A["研究问题与历史"] --> B["提出假设与策略"]
    B --> C["执行真实实验"]
    C --> D["模型解释原始证据"]
    C --> E["隐藏验证器独立审核"]
    D --> F["生成候选研究动作"]
    F --> G["复制状态并执行分支"]
    G --> H["计算分支科学价值"]
    E --> H
    H --> I["估计同状态动作优势"]
    H --> J["学习跨世界偏好反转"]
    I --> K["更新高层研究策略"]
    J --> K
~~~

### 7.4 推理时不能依赖隐藏标签

训练时：

$$
z_t^{*}
=
\operatorname{Verifier}
(
\omega_t,
\tau_t
).
$$

模型预测：

$$
\widehat z_t
=
\operatorname{Model}
(
h_t,o_t
).
$$

训练可以比较：

$$
\widehat z_t
\quad
\text{与}
\quad
z_t^{*}.
$$

但推理时，策略只能使用：

$$
\pi_{\theta}
(
u_t
\mid
h_t,o_t,\widehat z_t,c_t
).
$$

禁止使用：

$$
\pi_{\theta}
(
u_t
\mid
h_t,o_t,z_t^{*},c_t
)
$$

作为正常评测结果。

可以另外报告一个 oracle-state 上界，但必须与真实自主模式分开。

---

## 8. 严格评分规则与认知奖励

### 8.1 二元假设的对数评分

设：

$$
Y
\in
\{
0,1
\}
$$

表示隐藏真实结果。

设：

$$
b_t
=
P(
Y=1
\mid
h_t
).
$$

定义：

$$
S(
b_t,Y
)
=
Y\log b_t
+
(1-Y)\log(1-b_t).
$$

为避免数值不稳定，可以设：

$$
\widetilde b_t
=
\operatorname{clip}
(
b_t,
\epsilon,
1-\epsilon
).
$$

实践中可以从：

$$
\epsilon=10^{-4}
$$

或：

$$
\epsilon=10^{-3}
$$

开始，并在开发集冻结。

### 8.2 认知改进奖励

定义：

$$
r_t^{\mathrm{belief}}
=
S(
b_{t+1},Y
)
-
S(
b_t,Y
).
$$

如果：

$$
Y=1,
\qquad
b_t=0.1,
\qquad
b_{t+1}=0.9,
$$

则：

$$
r_t^{\mathrm{belief}}
=
\log 9.
$$

如果：

$$
Y=0,
\qquad
b_t=0.9,
\qquad
b_{t+1}=0.1,
$$

同样有：

$$
r_t^{\mathrm{belief}}
=
\log 9.
$$

因此：

> 正确支持与正确反驳都可以获得正向学习信号。

### 8.3 多假设扩展

对于多个备选机制：

$$
\Theta
\in
\{
\theta_1,\ldots,\theta_M
\},
$$

模型维护分布：

$$
b_t(\theta)
=
P(
\Theta=\theta
\mid
h_t
).
$$

对真实机制：

$$
\Theta^{*},
$$

使用：

$$
S(
b_t,\Theta^{*}
)
=
\log
b_t(
\Theta^{*}
).
$$

由此：

$$
r_t^{\mathrm{belief}}
=
\log
b_{t+1}
(
\Theta^{*}
)
-
\log
b_t
(
\Theta^{*}
).
$$

### 8.4 与信息增益的关系

如果：

$$
b_t(\theta)
=
p(
\theta
\mid
h_t
)
$$

且：

$$
b_{t+1}(\theta)
=
p(
\theta
\mid
h_t,o_{t+1},u_t
),
$$

则：

$$
\mathbb E
\left[
r_t^{\mathrm{belief}}
\mid
h_t,u_t
\right]
=
I(
\Theta;
O_{t+1}
\mid
h_t,u_t
).
$$

需要明确说明：

- 该结论依赖正确后验与适用的数据生成模型。
- 未校准的模型主观置信度不自动满足该等式。
- 真正开放的现实问题通常没有已知隐藏真值。
- 现实环境需要通过独立预测评分、预注册结果和确认实验近似评价。

### 8.5 防止主观置信度作弊

如果让模型随意报告：

$$
b_t,
$$

模型可能故意先报告很低的正确概率，再通过下一次回答获得高额改进奖励。

因此需要：

1. 实验前提交并冻结预测。
2. 将预测写入不可修改的实验账本。
3. 禁止读取结果后倒写之前的概率。
4. 使用可信后验、隐藏标签或独立预测评分。
5. 对整条轨迹使用统一的初始参考。
6. 检查信念随有效证据的连续性。
7. 对明显不一致或无证据调整进行惩罚。

对于无折扣有限轨迹：

$$
\sum_{t=0}^{T-1}
\left[
S(
b_{t+1},Y
)
-
S(
b_t,Y
)
\right]
=
S(
b_T,Y
)
-
S(
b_0,Y
).
$$

这样能够减少通过反复降低和抬高置信度刷取累计奖励的问题。

若使用：

$$
\gamma<1,
$$

需要重新检查奖励塑形是否改变原始优化目标；初始实验建议采用有限时域且：

$$
\gamma=1.
$$

### 8.6 真实世界没有隐藏真值时怎么办

对于真实研究问题，可采用以下替代评价：

1. 独立确认集上的预测对数得分。
2. 预注册效应估计的误差。
3. 新随机种子上的复现成功率。
4. 独立数据来源的泛化结果。
5. 对不可见实验输出的提前预测。
6. 与可信 reference implementation 的一致性。
7. 专家审核和外部重复实验。

需要将：

$$
\text{synthetic known truth}
$$

与：

$$
\text{real-world independently verified evidence}
$$

明确分开汇报。

---

## 9. 同状态实验分支与策略级信用分配

### 9.1 环境快照

在关键研究状态：

$$
s_t
$$

创建冻结快照：

$$
\chi_t
=
\operatorname{Snapshot}
(
\omega_t,
h_t,
c_t
).
$$

快照需要包括：

- 当前代码版本。
- 数据及切分。
- 已运行实验。
- 模型 checkpoint。
- 可访问上下文。
- 当前预算。
- 已登记假设。
- 实验随机数生成器状态。
- 不可变验证器版本。

### 9.2 候选研究动作

采样：

$$
u_t^{(1)},\ldots,u_t^{(K)}
\sim
\pi_{\mathrm{old}}^{H}
(
\cdot
\mid
s_t
).
$$

对每个动作：

$$
\chi_t^{(i)}
=
\operatorname{Restore}
(
\chi_t
).
$$

然后：

$$
\tau_t^{(i)}
=
\operatorname{Execute}
(
\chi_t^{(i)},
u_t^{(i)}
).
$$

### 9.3 分支科学效用

定义：

$$
G_t^{(i)}
=
\sum_{\ell=0}^{H_i-1}
\left[
r_{t+\ell}^{\mathrm{belief}}
+
\lambda_{\mathrm{task}}
r_{t+\ell}^{\mathrm{task}}
+
\lambda_{\mathrm{disc}}
r_{t+\ell}^{\mathrm{discovery}}
-
\lambda_{\mathrm{cost}}
c_{t+\ell}
\right].
$$

同时满足：

$$
\operatorname{Valid}
(
\tau_t^{(i)}
)
=
1
$$

才能对相关科学结论发放奖励。

### 9.4 留一优势估计

定义：

$$
\widehat A_t^{(i)}
=
G_t^{(i)}
-
\frac{1}{K-1}
\sum_{j\neq i}
G_t^{(j)}.
$$

该量表示：

> 当前研究动作相对于同一状态其他真实可执行动作的科学优势。

### 9.5 留一而不是包含自身的平均值

如果将当前回报也放进基线：

$$
\overline G
=
\frac1K
\sum_{j=1}^{K}
G_t^{(j)},
$$

则：

$$
G_t^{(i)}
-
\overline G
$$

与当前动作之间存在额外耦合。

留一基线：

$$
\overline G_{-i}
=
\frac{1}{K-1}
\sum_{j\neq i}
G_t^{(j)}
$$

在独立采样条件下更容易给出清晰的无偏性说明。

如果实际实现中仍使用组内标准差归一化，应将其描述为工程稳定化处理，而不是未经说明地宣称完全无偏。

### 9.6 长期修复动作的信用

某些关键动作的即时回报可能很低：

$$
r_t^{\mathrm{repair}}
\approx
0.
$$

但它可能让后续实验从：

$$
\mathrm{Invalid}
$$

转为：

$$
\mathrm{Refuted}
$$

或者：

$$
\mathrm{Supported}.
$$

因此：

$$
G_t^{\mathrm{repair}}
>
G_t^{\mathrm{continue}}.
$$

这使得模型可以学习：

> 即使修复行为暂时降低表面分数，只要它使后续判断更真实，就具有正向策略价值。

### 9.7 公共随机数与配对种子

对于两个动作：

$$
u_i,u_j,
$$

尽可能使用相同的冻结随机条件：

$$
\xi_1,\ldots,\xi_n.
$$

定义配对差值：

$$
d_k
=
G(
u_i,\xi_k
)
-
G(
u_j,\xi_k
).
$$

则：

$$
\widehat\Delta_{ij}
=
\frac1n
\sum_{k=1}^{n}
d_k.
$$

如果公共随机数带来正相关，则：

$$
\operatorname{Var}
(
G_i-G_j
)
=
\operatorname{Var}
(
G_i
)
+
\operatorname{Var}
(
G_j
)
-
2
\operatorname{Cov}
(
G_i,G_j
).
$$

因此有机会降低动作比较噪声。

但公共随机数不是任何情况下都能降低方差，需要在先导实验中实际检查。

### 9.8 外部候选动作与重要性修正

如果动作不是直接从：

$$
\pi_{\mathrm{old}}
$$

采样，而是来自：

- 外部教师。
- 人工指定动作。
- 强制覆盖所有动作类型。
- 多样性采样器。

则动作实际来自：

$$
\mu(
u
\mid
s
).
$$

原则上需要：

$$
\rho(
u,s
)
=
\frac{
\pi_{\mathrm{old}}
(
u
\mid
s
)
}{
\mu
(
u
\mid
s
)
}.
$$

如果长文本动作的重要性权重过于不稳定，可以采用：

1. 先将动作标准化为离散策略类别。
2. 用外部候选构造离线偏好学习数据。
3. 在真正在线 RL 阶段使用 on-policy 候选。
4. 单独标记教师生成动作，避免错误宣称自主发现。

---

## 10. 跨世界策略偏好反转

### 10.1 配对世界构造

对同一个研究问题：

$$
q,
$$

构造：

$$
w_A,w_B
\in
\mathcal W_q.
$$

要求：

1. 任务描述一致。
2. 可见工具接口一致。
3. 初始方法空间一致。
4. 总预算一致。
5. 候选动作具有可比性。
6. 不提供世界编号。
7. 世界差异来自真实机制或冻结的数据生成过程。

允许不同世界产生不同观测：

$$
o_A
\neq
o_B.
$$

模型必须依据这些观测改变策略。

### 10.2 动作价值反转

设：

$$
u_1
=
\text{继续当前方法},
$$

$$
u_2
=
\text{切换研究方法}.
$$

在世界 A：

$$
\Delta_A
=
Q_{w_A}
(
s_A,u_1
)
-
Q_{w_A}
(
s_A,u_2
)
>
0.
$$

在世界 B：

$$
\Delta_B
=
Q_{w_B}
(
s_B,u_1
)
-
Q_{w_B}
(
s_B,u_2
)
<
0.
$$

因此：

$$
\operatorname{sign}
(
\Delta_A
)
\neq
\operatorname{sign}
(
\Delta_B
).
$$

### 10.3 双重差分形式

定义：

$$
D_{A,B}
(
u_1,u_2
)
=
\Delta_A
-
\Delta_B.
$$

展开：

$$
\begin{aligned}
D_{A,B}
(
u_1,u_2
)
=
&
\left[
Q_{w_A}
(
s_A,u_1
)
-
Q_{w_A}
(
s_A,u_2
)
\right]
\\
&
-
\left[
Q_{w_B}
(
s_B,u_1
)
-
Q_{w_B}
(
s_B,u_2
)
\right].
\end{aligned}
$$

若满足适当的可比性假设，该量有助于识别：

> 世界条件与研究动作之间的交互关系。

注意：

> 不能在未说明干预假设时，把该量直接描述成证据文本对最终结果的无条件因果效应。

### 10.4 只学习统计上可靠的反转

对世界 A：

$$
\operatorname{LCB}
(
\Delta_A
)
>
\epsilon_{\mathrm{flip}}.
$$

对世界 B：

$$
\operatorname{UCB}
(
\Delta_B
)
<
-\epsilon_{\mathrm{flip}}.
$$

只有在两侧均通过置信门槛时，才将其记为可靠反转样本。

否则：

- 继续追加重复实验。
- 将该样本标记为不确定。
- 降低其训练权重。
- 不将随机波动误当成策略差异。

### 10.5 跨世界偏好反转损失

定义：

$$
\ell_{\theta}
(
u,s
)
=
\log
\frac{
\pi_{\theta}^{H}
(
u
\mid
s
)
}{
\pi_{\mathrm{ref}}^{H}
(
u
\mid
s
)
}.
$$

对于：

$$
u_1
\succ_A
u_2
$$

和：

$$
u_2
\succ_B
u_1,
$$

定义：

$$
\begin{aligned}
\mathcal L_{\mathrm{flip}}
=
&
-
\log
\sigma
\left(
\beta
\left[
\ell_{\theta}
(
u_1,s_A
)
-
\ell_{\theta}
(
u_2,s_A
)
\right]
\right)
\\
&
-
\log
\sigma
\left(
\beta
\left[
\ell_{\theta}
(
u_2,s_B
)
-
\ell_{\theta}
(
u_1,s_B
)
\right]
\right).
\end{aligned}
$$

这个损失直接要求：

- 在世界 A 中更偏好动作一。
- 在世界 B 中更偏好动作二。
- 但两个世界的问题描述保持一致。

模型只能通过真正理解实验反馈来满足上述要求。

### 10.6 为什么这比固定教师融合更重要

考虑一个固定教师：

$$
\pi_1
\approx
\text{总是继续}.
$$

另一个教师：

$$
\pi_2
\approx
\text{总是切换}.
$$

简单混合：

$$
\pi_{\mathrm{mix}}
=
\frac12
\pi_1
+
\frac12
\pi_2
$$

不能保证：

$$
\pi_{\mathrm{mix}}
(
\text{continue}
\mid
s_A
)
\gg
\pi_{\mathrm{mix}}
(
\text{switch}
\mid
s_A
),
$$

同时：

$$
\pi_{\mathrm{mix}}
(
\text{switch}
\mid
s_B
)
\gg
\pi_{\mathrm{mix}}
(
\text{continue}
\mid
s_B
).
$$

但是应避免过度宣称：

- 如果 SMOPD 教师自身已经能根据证据改变动作，它也可能学习条件化策略。
- 如果普通 GRPO 能够看到完整证据且有足够探索，也可能学会策略反转。

本研究需要验证的不是：

> 其他方法理论上绝不可能学会。

而是：

> 通过配对世界反转信号，是否能更高效、更稳定地学到证据响应性的研究策略。

### 10.7 多动作扩展

对多个候选动作：

$$
\mathcal U_s
=
\{
u_1,\ldots,u_K
\},
$$

可以构造参考改进分布：

$$
p_w^{*}
(
u
\mid
s
)
\propto
\pi_{\mathrm{old}}
(
u
\mid
s
)
\exp
\left(
\frac{
\widehat Q_w
(
s,u
)
}{
\tau
}
\right).
$$

这相当于求解：

$$
\max_p
\;
\sum_u
p(
u
)
\widehat Q_w
(
s,u
)
-
\tau
D_{\mathrm{KL}}
\left(
p
\Vert
\pi_{\mathrm{old}}
\right).
$$

随后可以通过：

$$
\mathcal L_{\mathrm{route}}
=
\sum_w
D_{\mathrm{KL}}
\left(
p_w^{*}
\Vert
\pi_{\theta}^{H}
\right)
$$

进行策略蒸馏。

研究初期优先采用简单、可解释的成对反转损失；多动作分布式蒸馏可作为后续扩展或消融。

---

## 11. 自主新路径发现与验证约束

### 11.1 不应奖励文本层面的新颖性

错误做法：

$$
r_{\mathrm{novel}}
=
\text{文本 embedding 距离}.
$$

这会鼓励：

- 改写方法描述。
- 更换术语。
- 增加表面复杂度。
- 生成不可执行的新概念。
- 反复提出与已有方法等价的策略。

### 11.2 方法族与策略结构

对研究动作建立独立的结构化表示：

$$
f(
u
)
=
(
\text{method-family},
\text{estimand},
\text{intervention},
\text{data-regime},
\text{evaluation-protocol}
).
$$

方法结构判断可以结合：

- 冻结的方法分类器。
- 代码语法树。
- 训练目标变更。
- 数据生成或采样机制变化。
- 实验干预变量变化。
- 对照与检验协议变化。
- 真实运行行为。

不能只由生成模型自行声明：

> 我提出了新的研究方法。

### 11.3 新路径证书

定义：

$$
\operatorname{Cert}
(
u
)
=
\mathbf 1[
\text{无具体方法提示}
]
\cdot
\mathbf 1[
\text{方法结构实质不同}
]
\cdot
\mathbf 1[
\text{真实执行}
]
\cdot
\mathbf 1[
\text{实验有效}
]
\cdot
\mathbf 1[
\text{独立确认通过}
]
\cdot
\mathbf 1[
\operatorname{LCB}
(
\Delta Q
)
>
\delta_{\mathrm{disc}}
].
$$

对应奖励：

$$
r_{\mathrm{discovery}}
=
\operatorname{Cert}
(
u
)
\cdot
\operatorname{clip}
\left(
\widehat{\Delta Q},
0,
r_{\max}
\right).
$$

### 11.4 自主性证据

需要记录：

- 初始任务提示。
- 系统提示。
- 工具输出。
- 可访问文献。
- 教师参与情况。
- 候选动作生成来源。
- 方法首次提出时间。
- 是否由用户或外部教师直接给出方法名称。

对于由外部教师直接提出的策略，只能记为：

$$
\text{teacher-proposed discovery},
$$

不能记为：

$$
\text{autonomous discovery}.
$$

### 11.5 全局原创性与局部自主性

本项目优先验证：

$$
\text{未提示}
+
\text{未见任务}
+
\text{独立复现}.
$$

除非另行完成：

- 时间切分。
- 文献全面排查。
- 专家评审。
- 独立真实实验。

否则不能宣称：

> 首次发现人类未知的方法或科学规律。

---

## 12. 完整优化目标与训练流程

### 12.1 总体目标

定义：

$$
\max_{\pi_{\theta}}
\;
\mathbb E
\left[
S(
b_T,\Theta^{*}
)
+
\lambda_{\mathrm{task}}
U_T
+
\lambda_{\mathrm{disc}}
D_T
-
\lambda_{\mathrm{cost}}
C_T
\right]
$$

满足：

$$
\mathbb P
\left(
\text{输出无效或错误科学结论}
\right)
\leq
\delta_{\mathrm{claim}}
$$

以及：

$$
\mathbb E
[
C_T
]
\leq
B.
$$

其中：

- $S$：严格评分规则。
- $U_T$：冻结任务目标上的真实收益。
- $D_T$：经过确认的自主新路径价值。
- $C_T$：实验、推理与训练成本。

### 12.2 优化损失

建议采用：

$$
\mathcal L_{\mathrm{PESCO}}
=
\mathcal L_{\mathrm{option}}
+
\lambda_{\mathrm{flip}}
\mathcal L_{\mathrm{flip}}
+
\lambda_{\mathrm{state}}
\mathcal L_{\mathrm{state}}
+
\lambda_{\mathrm{ref}}
\mathcal L_{\mathrm{ref}}.
$$

必要时加入：

$$
+
\lambda_{\mathrm{constraint}}
\mathcal L_{\mathrm{constraint}}.
$$

### 12.3 策略层 RL 损失

定义：

$$
\rho_t
=
\frac{
\pi_{\theta}^{H}
(
u_t
\mid
s_t
)
}{
\pi_{\mathrm{old}}^{H}
(
u_t
\mid
s_t
)
}.
$$

采用：

$$
\mathcal L_{\mathrm{option}}
=
-
\mathbb E
\left[
\min
\left(
\rho_t\widehat A_t,
\operatorname{clip}
\left(
\rho_t,
1-\epsilon,
1+\epsilon
\right)
\widehat A_t
\right)
\right].
$$

其中：

$$
\widehat A_t
$$

来自同状态真实实验分支。

### 12.4 证据状态监督

定义：

$$
\mathcal L_{\mathrm{state}}
=
-
\sum_{z}
y_z
\log
p_{\theta}
\left(
z
\mid
h_t,o_t
\right).
$$

也可拆解为：

$$
\mathcal L_{\mathrm{state}}
=
\lambda_v
\mathcal L_{\mathrm{validity}}
+
\lambda_p
\mathcal L_{\mathrm{precision}}
+
\lambda_d
\mathcal L_{\mathrm{direction}}.
$$

因子化设计可以避免将：

- 无效实验。
- 信息不足。
- 效应方向。

混为同一类现象。

### 12.5 约束优化

定义违规成本：

$$
g(
\tau
)
=
\mathbf 1[
\text{无效结论}
]
+
\mathbf 1[
\text{验证器绕过}
]
+
\mathbf 1[
\text{未确认结果冒充确认}
].
$$

可采用拉格朗日形式：

$$
\max_{\pi}
\min_{\lambda\geq0}
\;
\mathbb E
\left[
R(
\tau
)
-
\lambda
\left(
g(
\tau
)
-
\delta
\right)
\right].
$$

对于明确不可接受的违规，优先使用硬门控：

$$
\operatorname{Valid}
(
\tau
)
=0
\Longrightarrow
\text{对应科学结论不发放奖励}.
$$

### 12.6 Token 信用屏蔽

如果高层计划与低层代码由同一个 LLM 生成，建议对 token 分组：

~~~text
<research_belief>
...
</research_belief>

<research_strategy>
...
</research_strategy>

<execution_plan>
...
</execution_plan>

<tool_observation>
...
</tool_observation>
~~~

原则：

- 高层策略优势主要回传到研究策略 token。
- 证据分类损失回传到证据解释 token。
- 工具返回结果不作为模型生成 token 计入梯度。
- 系统提示、用户提示和隐藏观察不参与生成损失。
- 低层执行器可在独立实验中固定。

### 12.7 训练阶段

**阶段 A：可执行性冷启动。**

训练模型稳定完成：

- 结构化假设输出。
- 工具调用。
- 实验启动。
- 结果读取。
- 基本证据记录。

**阶段 B：证据识别监督。**

训练模型区分：

- 支持。
- 反驳。
- 证据不足。
- 实验无效。

**阶段 C：离线策略偏好学习。**

使用已执行分支构造：

$$
(s,u^{+},u^{-})
$$

以及：

$$
(s_A,s_B,u_1,u_2).
$$

验证跨世界反转损失的有效性。

**阶段 D：在线策略级 RL。**

模型生成候选路径，执行真实分支，并根据科学效用更新高层策略。

**阶段 E：去除方法提示。**

逐步减少：

- 具体算法名称。
- 可选动作建议。
- 人工诊断结论。
- 教师给出的修复方向。

**阶段 F：泛化和隐藏评测。**

在未见问题、未见世界机制、未见方法族与冻结隐藏测试集上评估。

### 12.8 训练伪代码

~~~python
def train_pesco(policy, executor, verifier, training_tasks):
    for task in training_tasks:
        worlds = sample_paired_worlds(task)
        world_records = []

        for world in worlds:
            state = world.reset()

            while state.remaining_budget > 0:
                observation = world.observe()

                belief = policy.report_belief(
                    history=state.history,
                    observation=observation,
                )

                evidence_prediction = policy.predict_evidence_state(
                    history=state.history,
                    observation=observation,
                )

                candidate_options = policy.sample_research_options(
                    history=state.history,
                    observation=observation,
                    belief=belief,
                    evidence_prediction=evidence_prediction,
                )

                snapshot = world.snapshot()
                branch_records = []

                for option in candidate_options:
                    branch_world = world.restore(snapshot)

                    trajectory = executor.execute_option(
                        world=branch_world,
                        option=option,
                        continuation_policy=policy,
                    )

                    verdict = verifier.evaluate(
                        trajectory=trajectory,
                        protocol=task.frozen_protocol,
                    )

                    utility = compute_scientific_utility(
                        verdict=verdict,
                        belief_before=belief,
                        belief_after=trajectory.final_belief,
                        cost=trajectory.total_cost,
                        confirmed_novelty=verdict.confirmed_novelty,
                    )

                    branch_records.append(
                        {
                            "option": option,
                            "trajectory": trajectory,
                            "verdict": verdict,
                            "utility": utility,
                        }
                    )

                advantages = leave_one_out_advantages(
                    branch_records
                )

                world_records.append(
                    {
                        "task": task,
                        "world": world,
                        "state": state,
                        "branches": branch_records,
                        "advantages": advantages,
                    }
                )

                state = choose_training_continuation(
                    world,
                    branch_records,
                )

        reversal_pairs = construct_verified_reversal_pairs(
            world_records,
            require_independent_confirmation=True,
        )

        optimize_policy(
            policy=policy,
            option_advantages=extract_advantages(world_records),
            evidence_labels=extract_verifier_labels(world_records),
            reversal_pairs=reversal_pairs,
            validity_constraints=True,
        )

    return policy
~~~

该伪代码描述研究方案，不代表现有项目中已经存在这些函数或模块。

---

## 13. 理论研究计划

### 13.1 命题一：严格评分规则鼓励真实信念

研究命题：

> 对给定的隐藏结果分布，严格适当评分规则在模型报告真实后验时取得最大期望得分。

可研究：

$$
\arg\max_{\widehat b}
\;
\mathbb E_{Y\sim b}
\left[
S(
\widehat b,Y
)
\right]
=
b.
$$

这为模型输出：

- 校准的假设概率。
- 真实的证据不确定性。
- 对可靠反驳的正确接受。

提供数学基础。

需要明确的前提：

- 假设定义在观察结果之前冻结。
- 评分所用的隐藏结果不能被模型访问。
- 信念报告不能在观察确认数据之后倒写。
- 连续更新的奖励不能被反复重置利用。

### 13.2 命题二：认知奖励与实验信息增益

研究命题：

$$
\mathbb E_{\Theta,O}
\left[
\log
p(
\Theta
\mid
O,h,u
)
-
\log
p(
\Theta
\mid
h
)
\right]
=
I(
\Theta;
O
\mid
h,u
).
$$

这说明：

> 在正确建模的可控环境中，奖励真实信念更新可以同时奖励具有信息价值的实验。

应单独讨论：

- 后验错误时的偏差。
- 近似推断造成的误差。
- 真实科学环境中未知真值的替代估计。
- 无效实验对后验更新的屏蔽。

### 13.3 命题三：留一基线的合法性

给定固定状态：

$$
s,
$$

若：

$$
u_i
\sim
\pi_{\theta}
(
\cdot
\mid
s
),
$$

并且：

$$
b_{-i}
\perp
u_i
\mid
s,
$$

则：

$$
\mathbb E
\left[
\nabla_{\theta}
\log
\pi_{\theta}
(
u_i
\mid
s
)
b_{-i}
\mid
s
\right]
=0.
$$

因此：

$$
\mathbb E
\left[
\nabla_{\theta}
\log
\pi_{\theta}
(
u_i
\mid
s
)
\left(
G_i-b_{-i}
\right)
\right]
$$

可作为策略梯度估计。

潜在破坏条件：

- 候选动作强制互不相同。
- 第一个动作决定其他候选动作。
- 动作由不同教师产生但没有重要性修正。
- 分支运行环境无法真正恢复。
- 验证器版本或预算在分支之间变化。
- 基线对当前动作进行反向传播。

### 13.4 命题四：证据盲策略的性能上界

考虑：

$$
Z
\in
\{
z_1,z_2
\}
$$

且：

$$
P(
Z=z_1
)
=
P(
Z=z_2
)
=
\frac12.
$$

设：

$$
u^{*}
(
z_1
)
=
u_1
$$

和：

$$
u^{*}
(
z_2
)
=
u_2.
$$

对于忽略证据的策略：

$$
\pi(
u
\mid
z_1
)
=
\pi(
u
\mid
z_2
),
$$

则：

$$
\sup_{\pi\ \mathrm{evidence\text{-}blind}}
V(
\pi
)
\leq
\frac12.
$$

理想证据条件化策略可以达到：

$$
V(
\pi^{*}
)
=1.
$$

四类状态均匀且动作一一对应时，证据盲策略在理想化设定中可能只有：

$$
\frac14
$$

的动作匹配率。

但真实研究环境并不存在固定的一一对应，实际证明与实验必须允许：

- 一个状态存在多个合理动作。
- 不同预算改变最优动作。
- 研究目标改变最终决策。
- 支持状态下可能应该停止而不是复现。
- 反驳状态下可能应该提交负结果而不是切换。

### 13.5 命题五：配对比较的噪声消除

设：

$$
G_{q,w,u}
=
\alpha_q
+
\beta_w
+
\gamma_u
+
\delta_{w,u}
+
\varepsilon_{q,w,u}.
$$

其中：

- $\alpha_q$：研究问题固有难度。
- $\beta_w$：实验世界固有难度。
- $\gamma_u$：某种动作整体上的平均优势。
- $\delta_{w,u}$：世界与动作之间的交互项。

在适当设计下：

$$
\left(
G_{q,w_A,u_1}
-
G_{q,w_A,u_2}
\right)
-
\left(
G_{q,w_B,u_1}
-
G_{q,w_B,u_2}
\right)
$$

可以消除公共问题项与部分固定动作项，从而更加直接地反映：

$$
\delta_{w_A,u_1}
-
\delta_{w_A,u_2}
-
\delta_{w_B,u_1}
+
\delta_{w_B,u_2}.
$$

需要说明：

- 该结论依赖可加模型与可比性。
- 不同世界的动作需要具有可比的执行语义。
- 不能凭该式自动断言识别所有科学因果机制。

### 13.6 命题六：适应性实验中的错误率控制

如果：

$$
E_t
$$

是零假设下适当构造的非负检验过程，则可利用：

$$
P_{H_0}
\left(
\sup_t
E_t
\geq
\frac1\alpha
\right)
\leq
\alpha.
$$

在应用到研究代理前必须检查：

- 假设是否在使用确认数据前冻结。
- 新假设是否分配了独立错误率预算。
- 数据访问顺序是否满足所需条件。
- 是否存在多重检验或选择偏差。
- 是否采用独立确认切分。

如果证明和实现成本过高，首版可以优先采用：

1. 冻结探索集。
2. 完全独立确认集。
3. 多重比较校正。
4. 预先冻结显著性或置信区间标准。

---

## 14. 实验环境设计

### 14.1 总体实验分层

建议构建三层实验：

| 层级 | 环境类型 | 主要用途 | 算力要求 |
| --- | --- | --- | --- |
| Tier 0 | 可解析的数学模拟环境 | 验证算法机制、最优动作和理论预测 | 主要使用 CPU |
| Tier 1 | 真实可执行的小规模统计或机器学习实验 | 验证数据泄漏、反驳、证据不足和策略切换 | CPU 或少量 GPU |
| Tier 2 | 真实 LLM 后训练或科研方法实验 | 验证复杂长程研究策略与跨任务泛化 | 受控 GPU 预算 |

必须通过：

$$
\mathrm{Tier\ 0}
\rightarrow
\mathrm{Tier\ 1}
\rightarrow
\mathrm{Tier\ 2}
$$

的阶段门槛，不能在验证器尚不可靠时直接进入大规模训练。

### 14.2 Tier 0：有限状态研究环境

构造：

$$
\Theta
\in
\{
\theta_{\mathrm{effective}},
\theta_{\mathrm{null}},
\theta_{\mathrm{alternative}},
\theta_{\mathrm{confounded}}
\}.
$$

动作：

$$
\mathcal U
=
\{
\text{continue},
\text{sample},
\text{repair},
\text{switch},
\text{replicate},
\text{stop}
\}.
$$

观测：

$$
O_t
=
g(
\Theta,u_t,\xi_t
).
$$

预算更新：

$$
c_{t+1}
=
c_t
-
\operatorname{Cost}
(
u_t
).
$$

通过动态规划、穷举或高精度蒙特卡洛计算：

$$
Q^{*}
(
s,u
).
$$

研究：

- 证据不足时何时追加样本。
- 实验无效时何时修复。
- 可靠反驳后何时切换。
- 预算不足时何时停止。
- 支持证据下何时复现。
- 新方法探索的机会成本。

### 14.3 Tier 1：基础数据生成机制

可使用：

$$
Y
=
\theta T
+
\beta C
+
\epsilon.
$$

其中：

- $T$：研究干预或待检验方法。
- $C$：潜在混杂变量。
- $\theta$：真实干预效应。
- $\beta$：混杂变量影响。
- $\epsilon$：随机误差。

如果：

$$
T
\not\perp
C,
$$

忽略：

$$
C
$$

会产生伪效应。

为了模拟数据泄漏，可额外构造：

$$
L
=
Y
+
\eta
$$

并将：

$$
L
$$

错误地放入预测特征中。

这样：

$$
\text{observed score}
\uparrow
$$

但：

$$
\text{valid scientific evidence}
\downarrow.
$$

### 14.4 四个基础世界

| 世界 | 真实参数或机制 | 初始观测 | 可能的有效后续动作 |
| --- | --- | --- | --- |
| 支持机制 | 正向效应大于预设最小有效值 | 有效实验出现稳定提升 | 复现、检查适用边界或提交结论 |
| 反驳机制 | 效应接近零或方向与假设相反 | 高精度实验不支持原主张 | 提交可靠负结果或探索替代方法 |
| 信息不足机制 | 存在不确定效应，但样本不足或方差过大 | 区间跨越决策阈值 | 增加样本、减少噪声或停止并报告不足 |
| 无效机制 | 数据泄漏、混杂、错误切分或评测缺陷 | 表面效果显著但不可解释 | 修复实验、重新运行或声明无法判断 |

注意：

> 表格中列的是世界机制与典型初始状态，不表示该世界在所有时间点永久保持同一证据标签。

### 14.5 扩展世界

建议逐步加入：

- 方法只对特定子群有效。
- 效应存在但评价指标不合适。
- 原方法失败，但替代机制有效。
- 方法有效，但实现存在工程错误。
- 方法对训练数据有效，但 OOD 环境失效。
- 方法平均无效，但与另一策略组合后有效。
- 假设方向正确，但效应幅度小于实际意义阈值。
- 初始负结果来自独立重复不足。
- 原方法可靠，但无必要继续消耗预算。

### 14.6 Tier 1 任务族建议

可从以下任务族中选择：

1. 表格预测与数据泄漏识别。
2. 分组切分和时间切分。
3. 特征处理方法有效性。
4. 类别不平衡与指标错配。
5. 因果干预与混杂变量控制。
6. 模型校准与错误概率解释。
7. 时间序列中的未来信息泄漏。
8. 不同随机种子的性能波动。
9. 子群效果与平均效果不一致。
10. 训练与测试分布偏移。

每个任务族都应同时包含：

- 正结果。
- 负结果。
- 不确定结果。
- 无效实验。

### 14.7 Tier 2：真实模型后训练任务

建议选择轻量可执行问题：

- 小模型数学能力提升。
- 小模型代码能力提升。
- 工具格式与任务正确性冲突。
- 偏好优化与监督微调选择。
- 训练样本质量与覆盖率差异。
- 数据污染或重复样本的影响。
- 奖励模型偏差。
- 评测脚本错误。
- 小样本评估导致的不稳定结论。
- 新训练目标或数据构造方法的发现。

对于每个任务，必须冻结：

1. 基础模型 checkpoint。
2. tokenizer。
3. 训练数据可访问范围。
4. 训练时长和 GPU 数量。
5. 评测程序。
6. 隐藏确认数据。
7. 禁止网络或外部泄漏策略。
8. 结果写入与验证权限。

### 14.8 可执行环境 API

建议抽象出如下接口：

~~~python
class ResearchEnvironment:
    def reset(self, question_id, world_id, seed):
        raise NotImplementedError

    def visible_observation(self):
        raise NotImplementedError

    def snapshot(self):
        raise NotImplementedError

    def restore(self, snapshot):
        raise NotImplementedError

    def execute_option(self, option):
        raise NotImplementedError

    def remaining_budget(self):
        raise NotImplementedError

    def final_submission(self, claim):
        raise NotImplementedError


class TrustedScientificVerifier:
    def assess_validity(self, trajectory):
        raise NotImplementedError

    def classify_evidence(self, trajectory, protocol):
        raise NotImplementedError

    def confirm_independently(self, candidate):
        raise NotImplementedError

    def compute_scientific_utility(self, trajectory):
        raise NotImplementedError

    def produce_audit_record(self, trajectory):
        raise NotImplementedError
~~~

该接口为目标设计，不代表当前仓库已经具备对应实现。

---

## 15. 数据集与环境划分

### 15.1 划分原则

严格区分：

$$
\mathcal E
=
\mathcal E_{\mathrm{train}}
\cup
\mathcal E_{\mathrm{dev}}
\cup
\mathcal E_{\mathrm{promotion}}
\cup
\mathcal E_{\mathrm{final,ID}}
\cup
\mathcal E_{\mathrm{final,OOD}}.
$$

其中：

- Train：训练环境。
- Dev：开发、调试和早期诊断。
- Promotion：模型是否进入最终评估的晋级验证。
- Final ID：冻结的同分布最终测试。
- Final OOD：冻结的分布外最终测试。

### 15.2 按研究问题而非随机种子划分

不能采用：

> 同一研究问题的一部分随机种子训练，另一部分随机种子测试。

应优先保证：

$$
\mathcal Q_{\mathrm{train}}
\cap
\mathcal Q_{\mathrm{test}}
=
\varnothing.
$$

进一步保证：

$$
\mathcal F_{\mathrm{train}}
\cap
\mathcal F_{\mathrm{OOD}}
=
\varnothing,
$$

其中：

$$
\mathcal F
$$

表示方法族或研究机制类别。

### 15.3 世界配对规则

每个研究问题：

$$
q
$$

对应：

$$
\mathcal W_q
=
\{
w_1,\ldots,w_m
\}.
$$

需要保证：

1. 世界差异具有明确机制解释。
2. 原始任务描述不包含隐藏世界标签。
3. 模型无法通过路径、文件名、样本编号直接识别世界。
4. 研究问题的真值不是由同一个固定文本提示决定。
5. 配对动作在不同世界中尽可能具有一致语义。
6. 训练、验证和测试机制之间不存在简单模板泄漏。

### 15.4 分支轨迹记录

建议数据结构：

~~~json
{
  "question_id": "rq_017",
  "question_family": "tabular_generalization",
  "world_id": "private_world_reference",
  "snapshot_id": "snap_003",
  "branch_id": "branch_002",
  "hypothesis": {
    "id": "h_004",
    "claim": "Method A improves group-held-out accuracy",
    "minimum_effect": 0.02,
    "registered_before_confirmation": true
  },
  "visible_history_hash": "sha256:...",
  "option": {
    "family": "repair_split",
    "free_form_plan": "replace random split with group split",
    "proposed_without_method_hint": true
  },
  "observations": [
    {
      "turn": 2,
      "visible_output": "validation accuracy changed",
      "tool_call_succeeded": true
    }
  ],
  "trusted_evidence": {
    "state_before": "invalid",
    "state_after": "insufficient",
    "confirmation_pass": false
  },
  "returns": {
    "belief_delta": 0.14,
    "task_utility": 0.05,
    "cost": 0.03,
    "novelty_bonus": 0.0,
    "total": 0.16
  },
  "provenance": {
    "proposal_source": "policy_on_policy",
    "executor_model": "frozen_executor",
    "evaluator_version": "frozen_v1",
    "trainable_token_mask": "strategy_and_belief_only"
  }
}
~~~

### 15.5 配对世界反转数据

~~~json
{
  "question_id": "rq_017",
  "paired_worlds": [
    "world_effective",
    "world_refuted"
  ],
  "candidate_options": [
    "continue_current_method",
    "switch_method_family"
  ],
  "estimated_values": {
    "world_effective": {
      "continue_current_method": 0.74,
      "switch_method_family": 0.31
    },
    "world_refuted": {
      "continue_current_method": -0.08,
      "switch_method_family": 0.46
    }
  },
  "paired_confidence": {
    "delta_effective_lcb": 0.18,
    "delta_refuted_ucb": -0.12,
    "confirmed_reversal": true
  },
  "world_ids_visible_to_policy": false
}
~~~

示例数字仅说明数据格式，不代表已经运行过任何实验。

### 15.6 训练集构成

建议保留：

- 正确支持路径。
- 正确反驳路径。
- 证据不足后追加实验路径。
- 无效实验后修复路径。
- 错误策略坚持路径。
- 不必要策略切换路径。
- 真正有效的新方法路径。
- 虚假新颖但不可执行路径。
- 表面有效但无法独立复现路径。
- 预算不足时正确停止的路径。

不得只保存：

> 最终得到支持性结论的成功轨迹。

否则模型会系统性忽略科学上同样有价值的负结果。

---

## 16. 基线方法与公平比较

### 16.1 基线列表

| 编号 | 基线 | 核心机制 | 需要回答的问题 |
| --- | --- | --- | --- |
| B0 | Base | 不进行针对性训练 | 基础模型天然具有多少科研策略能力 |
| B1 | SFT | 模仿高质量研究轨迹 | 监督学习是否足以解决问题 |
| B2 | GRPO-Terminal | 使用终局科学效用 | 只看最后结果是否足够 |
| B3 | GRPO-FourState | 使用四类证据奖励 | 简单任务奖励工程是否足够 |
| B4 | GDPO | 分开归一化多种奖励 | 多奖励平衡是否足够 |
| B5 | SMOPD | 多教师专长蒸馏 | 融合多种奖励专家是否足够 |
| B6 | Evidence-Gated SMOPD | 根据证据预测选择教师 | 简单条件化教师混合是否足够 |
| B7 | DiscoPO | 逐轮可信证据奖励 | 科研过程监督是否足够 |
| B8 | GiGPO / Ecpo | 同状态动作优势与统计校准 | 同状态动作比较是否足够 |
| B9 | TCPO | 多轮信用与固定历史反事实 | 通用反事实信用是否足够 |
| B10 | CVT-RL | 反事实信用与验证约束 | 一般性因果信用加安全约束是否足够 |
| B11 | Search-Only | 不训练模型，只增加推理期分叉 | 提升是否只是执行更多搜索 |
| B12 | PESCO-Offline | 配对世界偏好反转，不做在线 RL | 离线策略反转是否已经有效 |
| B13 | PESCO-Full | 完整方法 | 完整算法是否存在额外收益 |

### 16.2 最低必需基线

如果算力有限，至少保留：

1. Base。
2. SFT。
3. GRPO-FourState。
4. SMOPD。
5. DiscoPO。
6. TCPO。
7. Ecpo 或 CVT-RL。
8. Search-Only。
9. PESCO-Offline。
10. PESCO-Full。

### 16.3 统一奖励和环境信息

为了避免不公平，应区分两个实验设置。

**设置一：环境信息匹配。**

所有方法可访问相同：

- 原始实验输出。
- 工具接口。
- 预算。
- 验证器反馈粒度。
- 数据集。
- 研究历史。

**设置二：奖励信息匹配。**

所有方法尽量共享：

- 相同隐藏验证器。
- 相同终局效用。
- 相同证据分类标准。
- 相同独立复现标准。

主要差异限定在：

> 如何从这些信息中构造策略更新与信用分配。

### 16.4 三种计算预算匹配

必须分别报告：

**训练 GPU 成本匹配。**

$$
\mathrm{GPUHours}_{\mathrm{train}}
\approx
\mathrm{constant}.
$$

**环境实验成本匹配。**

$$
\#\mathrm{EnvironmentRuns}
\approx
\mathrm{constant}.
$$

**总成本匹配。**

$$
C_{\mathrm{total}}
=
C_{\mathrm{teacher}}
+
C_{\mathrm{training}}
+
C_{\mathrm{rollout}}
+
C_{\mathrm{verification}}
+
C_{\mathrm{confirmation}}.
$$

特别注意：

- SMOPD 的多个教师训练成本必须计算。
- PESCO 的额外实验分支必须计算。
- Search-Only 的推理搜索成本必须计算。
- 人工提示和外部教师参与需要单独记录。

### 16.5 执行器一致性

初始对比应固定：

$$
\pi_{\mathrm{executor}}
=
\pi_0.
$$

否则：

> 某个方法使用更强执行器导致的提升，可能被错误归因到研究策略算法。

后续可以增加：

- 固定执行器实验。
- 联合训练实验。
- 交叉替换执行器实验。

用来分离：

$$
\text{strategy improvement}
$$

与：

$$
\text{execution improvement}.
$$

### 16.6 推理阶段预算

正式评测建议同时报告：

1. 单路径推理。
2. 固定小规模分支推理。
3. 与基线完全相同的搜索预算。
4. 更大搜索预算下的性能上限。

如果只有第四种表现提升，则不能证明能力已经内化到模型参数中。

---

## 17. 评价指标

### 17.1 主指标：成本归一化可信科研价值

定义：

$$
\mathrm{VRS}
=
\mathbb E
\left[
\mathbf 1[
\mathrm{ValidClaim}
]
\cdot
\left(
\alpha
S_{\mathrm{belief}}
+
\beta
U_{\mathrm{task}}
+
\gamma
U_{\mathrm{replication}}
+
\eta
U_{\mathrm{discovery}}
\right)
-
\lambda
C
\right].
$$

其中：

- ValidClaim：最终结论符合可信协议。
- Belief：假设判断校准程度。
- Task：研究问题的真实目标收益。
- Replication：独立复现质量。
- Discovery：经过验证的新路径价值。
- Cost：实验和推理预算。

各项权重必须在正式实验前冻结。

### 17.2 证据状态识别准确率

建议使用：

$$
\mathrm{MacroF1}_{\mathrm{state}}.
$$

分别报告：

- Supported F1。
- Refuted F1。
- Insufficient F1。
- Invalid F1。

同时报告混淆矩阵，重点观察：

$$
\mathrm{Insufficient}
\rightarrow
\mathrm{Refuted}
$$

以及：

$$
\mathrm{Invalid}
\rightarrow
\mathrm{Supported}.
$$

### 17.3 研究动作遗憾

定义：

$$
\operatorname{Regret}
(
s
)
=
\max_u
Q^{*}
(
s,u
)
-
Q^{*}
(
s,u_{\pi}
).
$$

总体：

$$
\mathrm{ResearchRegret}
=
\frac1N
\sum_{i=1}^{N}
\operatorname{Regret}
(
s_i
).
$$

在 Tier 0 可以精确或近似计算最优值；在 Tier 1 和 Tier 2 使用冻结的真实分支结果估计。

### 17.4 跨世界偏好反转准确率

对经确认的反转对：

$$
\mathcal P_{\mathrm{flip}},
$$

定义：

$$
\mathrm{FlipAcc}
=
\frac1{
|
\mathcal P_{\mathrm{flip}}
|
}
\sum_{
(A,B,i,j)
\in
\mathcal P_{\mathrm{flip}}
}
\mathbf 1
\left[
\pi(
u_i
\mid
s_A
)
>
\pi(
u_j
\mid
s_A
)
\right]
\cdot
\mathbf 1
\left[
\pi(
u_j
\mid
s_B
)
>
\pi(
u_i
\mid
s_B
)
\right].
$$

这是候选核心指标之一。

### 17.5 有效策略切换率

定义：

$$
\mathrm{EffectiveSwitchRate}
=
\frac{
\#\{
\text{切换且独立确认更优}
\}
}{
\#\{
\text{确实存在更优替代路径的状态}
\}
}.
$$

同时报告：

$$
\mathrm{UnnecessarySwitchRate}
=
\frac{
\#\{
\text{切换但没有科学收益}
\}
}{
\#\{
\text{全部切换行为}
\}
}.
$$

### 17.6 正确坚持率

定义：

$$
\mathrm{AppropriatePersistence}
=
P
\left(
\text{继续当前策略}
\mid
\text{当前策略确实最优}
\right).
$$

该指标避免误把：

> 更频繁地切换。

当作：

> 更好的研究决策。

### 17.7 可靠反驳接受率

定义：

$$
\mathrm{RefutationAcceptance}
=
P
\left(
\text{正确接受负结果}
\mid
\text{实验有效且证据足以反驳}
\right).
$$

同时检查：

- 是否继续寻找支持性指标。
- 是否错误宣称实验失败。
- 是否重新定义研究目标。
- 是否在需要时提交可靠负结果。

### 17.8 证据不足处理率

定义：

$$
\mathrm{UnderpowerHandling}
=
P
\left(
\text{追加有效信息或正确声明不足}
\mid
\text{证据不足}
\right).
$$

追加样本并非永远正确；如果预算不足或信息价值不高，正确停止也应计为合理行为。

### 17.9 无效实验修复率

定义：

$$
\mathrm{InvalidRepairRate}
=
P
\left(
\text{定位并修复实验缺陷}
\mid
\text{存在可修复的无效实验}
\right).
$$

额外报告：

$$
\mathrm{InvalidClaimRate}
=
P
\left(
\text{提交支持性结论}
\mid
\text{实验无效}
\right).
$$

### 17.10 无提示新路径验证率

定义：

$$
\mathrm{VNPR}
=
\frac{
\#\{
\text{无提示、可执行、独立复现、确实更优的新路径}
\}
}{
\#\{
\text{提供探索机会的研究任务}
\}
}.
$$

单独报告：

- 由策略模型自主提出的路径。
- 由外部教师提出的路径。
- 由人工提示触发的路径。
- 已知方法族重组。
- 未见方法族泛化。

### 17.11 策略方法多样性

对经过有效性验证的方法族：

$$
\mathcal F,
$$

计算：

$$
H(
\mathcal F
)
=
-
\sum_f
p(
f
)
\log
p(
f
).
$$

有效方法族数量：

$$
N_{\mathrm{effective}}
=
\exp
\left(
H(
\mathcal F
)
\right).
$$

不统计：

- 单纯换措辞。
- 无效代码。
- 无法复现的方法。
- 对原方法的无意义语法变体。

### 17.12 方法发现上限

同时报告：

$$
\mathrm{pass@1},
\qquad
\mathrm{pass@k},
\qquad
\mathrm{best\text{-}of\text{-}k}.
$$

避免出现：

$$
\mathrm{pass@1}\uparrow
$$

但：

$$
\mathrm{pass@k}\downarrow.
$$

这类结果可能意味着训练提升平均保守性能，但降低探索上限。

### 17.13 错误发现率与复现率

定义：

$$
\mathrm{FDR}
=
\frac{
\#\{
\text{被宣布有效但独立确认失败的方法}
\}
}{
\max
\left(
1,
\#\{
\text{全部宣布有效的方法}
\}
\right)
}.
$$

定义：

$$
\mathrm{ReplicationRate}
=
\frac{
\#\{
\text{独立确认通过}
\}
}{
\#\{
\text{进入确认流程}
\}
}.
$$

### 17.14 成本归一化收益

定义：

$$
\mathrm{Efficiency}
=
\frac{
\mathrm{VerifiedScientificUtility}
}{
C_{\mathrm{GPU}}
+
\lambda_{\mathrm{CPU}}
C_{\mathrm{CPU}}
+
\lambda_{\mathrm{exp}}
N_{\mathrm{exp}}
+
\lambda_{\mathrm{token}}
N_{\mathrm{token}}
}.
$$

所有系数在正式比较前冻结。

---

## 18. 消融实验与负对照

### 18.1 核心消融

| 编号 | 消融设置 | 被移除的机制 | 主要观察指标 |
| --- | --- | --- | --- |
| A1 | No-PairedWorld | 去除配对世界训练 | FlipAcc、OOD 泛化 |
| A2 | No-FlipLoss | 去除策略反转损失 | FlipAcc、有效切换率 |
| A3 | No-Branch | 去除同状态真实分支 | 研究动作遗憾、修复率 |
| A4 | No-ProperScore | 改为支持奖励和反驳惩罚 | 负结果接受率、虚假支持率 |
| A5 | No-ValidityGate | 去除实验有效性硬约束 | InvalidClaimRate、FDR |
| A6 | No-Replication | 去除独立确认 | FDR、ReplicationRate |
| A7 | No-NoveltyCertificate | 去除新路径验证门控 | 文本新颖度与真实有效新方法差异 |
| A8 | No-StrategyMask | 将回报广播至所有生成 token | 策略改进与执行改进分离 |
| A9 | No-CostPenalty | 去除预算约束 | 实验次数、成本归一化收益 |
| A10 | No-FactorizedEvidence | 使用简单四分类，不分解有效性和精度 | 状态混淆、修复行为 |
| A11 | StaticTeacherMix | 改成固定教师混合 | 配对世界策略响应 |
| A12 | StateGateOnly | 只使用证据状态门控，不用真实分支 | 门控是否足以替代策略价值学习 |

### 18.2 关键负对照

**证据打乱对照。**

保持问题不变，打乱原始实验输出与世界之间的对应关系。

预期：

> 真正依赖证据的模型应该受到明显影响。

**隐藏证据对照。**

只提供研究问题，不提供实验结果。

预期：

> 若模型仍具有接近原来的策略反转准确率，则需要检查隐藏提示或世界 ID 泄漏。

**世界文件名替换对照。**

随机替换：

- 数据路径。
- 文件名。
- 世界编号。
- 配置顺序。
- 样本索引。

预期：

> 结果不应依赖与科学机制无关的环境标识。

**无效评测高分对照。**

提供明显高于真实成绩的泄漏结果。

预期：

> 模型应优先修复或报告无效，而不是直接提交支持性结论。

**可靠负结果对照。**

构造高精度、可复现、明确反驳原假设的世界。

预期：

> 模型能够接受反驳，而不是无限制调参寻找偶然正结果。

**表面新颖对照。**

将已有方法重新命名或更换表述。

预期：

> 不应获得新路径奖励。

**假复现对照。**

多次使用相同数据切分或同一随机种子。

预期：

> 验证器不能将其认定为独立复现。

### 18.3 Oracle 上界分析

额外报告：

1. 已知真实证据状态的 oracle-state。
2. 已知最佳动作候选集合的 oracle-option。
3. 已知真实世界机制的 oracle-world。
4. 高预算树搜索 oracle。

通过比较区分：

- 模型不会解释证据。
- 模型能解释但不会生成候选方法。
- 模型会生成候选但不会选择。
- 执行器无法落实正确策略。
- 环境预算不足。

---

## 19. 统计分析、重复实验与错误发现控制

### 19.1 独立统计单位

正式推断的主要统计单位应为：

$$
\text{研究问题或研究机制族}.
$$

不能将：

- 同一问题的多个世界。
- 同一世界的多个随机种子。
- 同一快照的多个动作分支。

直接视为相互独立的研究问题。

### 19.2 聚类 bootstrap

对于主指标：

$$
\mathrm{VRS},
$$

建议按：

$$
q
$$

进行重采样。

如果一个研究问题包含：

$$
w_1,\ldots,w_m,
$$

则在 bootstrap 时应整体采样该问题的所有对应世界。

### 19.3 配对比较

对同一个研究问题：

$$
q,
$$

分别运行：

$$
\pi_{\mathrm{PESCO}}
$$

和：

$$
\pi_{\mathrm{baseline}}.
$$

定义：

$$
d_q
=
U_q
\left(
\pi_{\mathrm{PESCO}}
\right)
-
U_q
\left(
\pi_{\mathrm{baseline}}
\right).
$$

报告：

$$
\overline d
=
\frac1N
\sum_{q=1}^{N}
d_q
$$

及其问题级置信区间。

根据数据类型，可以采用：

- 配对 bootstrap。
- 配对置换检验。
- McNemar 检验。
- 分层混合效应模型。
- 预注册的非参数检验。

### 19.4 多重比较

如果同时比较多个 baseline 或多个主指标，使用：

- Holm 校正。
- Benjamini–Hochberg。
- 预先指定的层级检验。

在正式实验前区分：

- 主要假设。
- 次要指标。
- 探索性分析。

不得在看到结果之后任意改变主要结论指标。

### 19.5 统计功效初步估算

对于二元配对成功指标，如果：

$$
p_{01}
=
0.25
$$

和：

$$
p_{10}
=
0.10,
$$

则净提升为：

$$
\Delta
=
0.15.
$$

近似样本量：

$$
n
\approx
\frac{
\left(
z_{1-\alpha/2}
+
z_{1-\beta}
\right)^2
\left(
p_{01}+p_{10}
\right)
}{
\left(
p_{01}-p_{10}
\right)^2
}.
$$

若：

$$
\alpha=0.05,
\qquad
1-\beta=0.8,
$$

则：

$$
n
\approx
\frac{
\left(
1.96+0.84
\right)^2
\cdot
0.35
}{
0.15^2
}
\approx
122.
$$

这里的：

$$
n
$$

指独立研究问题或近似独立的研究决策单元，不是随机种子数量。

该数值只是示例。正式研究应根据先导实验中的实际：

- 不一致率。
- 聚类结构。
- 指标方差。
- 基线性能。
- 多重检验策略。

重新进行功效分析。

### 19.6 独立探索与确认

对每个候选新方法，区分：

$$
\mathcal D_{\mathrm{explore}}
$$

和：

$$
\mathcal D_{\mathrm{confirm}}.
$$

要求：

$$
\mathcal D_{\mathrm{explore}}
\cap
\mathcal D_{\mathrm{confirm}}
=
\varnothing.
$$

同时：

$$
\mathcal S_{\mathrm{explore}}
\cap
\mathcal S_{\mathrm{confirm}}
=
\varnothing,
$$

其中：

$$
\mathcal S
$$

表示随机种子集合。

### 19.7 假设自适应提出与错误率分配

如果模型在执行过程中提出：

$$
H_1,H_2,\ldots,H_m,
$$

需要处理：

$$
\text{multiple testing}
$$

与：

$$
\text{adaptive hypothesis selection}.
$$

首版建议：

- 每个新假设必须在确认集访问前登记。
- 使用独立的确认种子或数据切分。
- 对同一问题内的多个确认检验做校正。
- 限制每个任务允许确认的新假设数量。
- 单独报告探索结果和确认结果。

---

## 20. 工程实现方案

### 20.1 模块划分

建议模块结构如下：

~~~text
research_strategy_optimization/
    configs/
        freeze/
        environments/
        algorithms/
        baselines/
        evaluation/
    environments/
        abstract_research_env.py
        world_registry.py
        snapshot_manager.py
        budget_tracker.py
        tier0_simulator.py
        tier1_tabular_env.py
        tier1_confounding_env.py
        tier1_leakage_env.py
        tier2_posttraining_env.py
    evidence/
        hypothesis_registry.py
        evidence_schema.py
        evidence_classifier.py
        interval_rules.py
        equivalence_tests.py
        validity_checks.py
        replication_protocol.py
        optional_stopping_controls.py
    algorithms/
        strategy_policy.py
        option_executor.py
        proper_scoring.py
        branch_rollout.py
        leave_one_out_advantage.py
        paired_world_sampler.py
        preference_reversal_loss.py
        discovery_certificate.py
        constrained_objective.py
        pesco_trainer.py
    baselines/
        sft.py
        grpo_terminal.py
        grpo_four_state.py
        gdpo.py
        smopd_adapter.py
        disco_adapter.py
        tcpo_adapter.py
        ecpo_adapter.py
        search_only.py
    evaluation/
        evidence_metrics.py
        strategy_regret.py
        preference_reversal.py
        refutation_metrics.py
        replication_metrics.py
        novelty_metrics.py
        compute_accounting.py
        cluster_bootstrap.py
        final_decision.py
    tests/
        test_world_isolation.py
        test_snapshot_equivalence.py
        test_evidence_state_transitions.py
        test_refutation_vs_insufficient.py
        test_leakage_detection.py
        test_leave_one_out_advantage.py
        test_preference_reversal.py
        test_novelty_certificate.py
        test_train_final_separation.py
        test_compute_accounting.py
~~~

该目录为设计蓝图，不表示已在当前仓库创建。

### 20.2 关键工程接口

~~~python
class HypothesisRegistry:
    def register_before_experiment(self, hypothesis, protocol):
        raise NotImplementedError

    def commit_belief(self, hypothesis_id, probability):
        raise NotImplementedError

    def append_evidence(self, hypothesis_id, evidence_record):
        raise NotImplementedError

    def freeze_confirmation_protocol(self, hypothesis_id):
        raise NotImplementedError


class BranchRolloutManager:
    def create_snapshot(self, environment):
        raise NotImplementedError

    def execute_paired_options(self, snapshot, options, seeds):
        raise NotImplementedError

    def estimate_paired_effects(self, branch_results):
        raise NotImplementedError


class PairedWorldPreferenceBuilder:
    def align_candidate_options(self, world_records):
        raise NotImplementedError

    def identify_confirmed_reversals(self, world_records):
        raise NotImplementedError

    def build_training_pairs(self, reversal_records):
        raise NotImplementedError
~~~

### 20.3 冻结协议文件建议

~~~yaml
protocol_version: pesco_v0_1

task:
  objective: evidence_conditioned_research_strategy
  question_manifest: frozen_question_manifest.json
  task_cutoff_date: "2026-08-25"

worlds:
  mechanism_manifest: frozen_world_manifest.json
  expose_world_id_to_agent: false
  expose_hidden_truth_to_agent: false
  allow_runtime_world_mutation: false

evidence:
  minimum_meaningful_effect: 0.02
  confidence_level: 0.95
  invalid_precedence: true
  allow_posthoc_power_as_refutation: false
  require_hypothesis_registration: true

verification:
  evaluator_version: trusted_verifier_v1
  evaluator_digest: sha256_placeholder
  immutable_evaluator: true
  independent_confirmation_required: true
  confirmation_seeds: [103, 107, 109, 113]

budget:
  max_strategy_turns: 6
  max_candidate_options_per_turn: 4
  max_confirmation_attempts: 2
  normalize_gpu_cpu_and_token_cost: true

security:
  network_policy: none
  allow_evaluator_modification: false
  allow_hidden_test_access: false
  allow_target_paper_access: false

splits:
  train_manifest: train_manifest.json
  dev_manifest: dev_manifest.json
  promotion_manifest: promotion_manifest.json
  final_id_manifest: final_id_manifest.json
  final_ood_manifest: final_ood_manifest.json
~~~

所有配置值仅为示例；正式值必须根据现有冻结协议和先导实验重新确定。

### 20.4 状态快照验证

每一个快照至少验证：

$$
\operatorname{Hash}
(
\chi_t^{(1)}
)
=
\operatorname{Hash}
(
\chi_t^{(2)}
)
$$

在执行分支之前成立。

需额外检查：

- 文件内容一致。
- 数据内容一致。
- 内存中的关键变量一致。
- 随机数生成器状态一致。
- 预算一致。
- 允许访问的上下文一致。
- 验证器版本一致。

如果无法保证快照一致，则该数据只能称为：

> 不同轨迹之间的比较。

不能直接称为：

> 同状态反事实分支。

### 20.5 证据记录与审计账本

每次实验需要记录：

~~~json
{
  "event_type": "research_experiment_completed",
  "event_index": 14,
  "question_id": "rq_017",
  "hypothesis_id": "h_004",
  "snapshot_id": "snap_003",
  "branch_id": "branch_002",
  "previous_event_hash": "sha256:...",
  "experiment_code_hash": "sha256:...",
  "dataset_hash": "sha256:...",
  "visible_output_hash": "sha256:...",
  "trusted_verdict_hash": "sha256:...",
  "budget_before": 4,
  "budget_after": 3,
  "confirmation_data_accessed": false
}
~~~

必须保证：

- 事件顺序可核对。
- 之前的事件不能被覆盖。
- 确认数据访问可审计。
- 最终结果可以重放。

### 20.6 训练器必须检查的前置条件

在任何正式在线训练前检查：

~~~python
def assert_training_allowed(freeze, environment, verifier):
    assert freeze.is_bound
    assert freeze.question_manifest_is_sealed
    assert freeze.final_split_is_inaccessible
    assert verifier.is_immutable
    assert verifier.negative_controls_pass
    assert environment.snapshot_replay_is_valid
    assert environment.world_id_is_hidden
    assert environment.confirmation_split_is_hidden
    assert freeze.contamination_audit_pass
    assert freeze.resource_budget_is_defined
    assert freeze.scientific_gate_is_open
~~~

如果其中任何一项失败：

$$
\text{formal online RL}
=
\text{NO-GO}.
$$

---

## 21. 初始超参数与资源预算

### 21.1 超参数属于先导建议

本节参数仅用于启动先导实验，并不表示已经完成调参或具有最优性。

正式论文实验需要：

1. 先在开发集进行小规模搜索。
2. 冻结搜索空间。
3. 冻结最终参数。
4. 使用冻结参数评估隐藏测试任务。
5. 避免根据最终测试结果重复调参。

### 21.2 模型规模

建议按以下顺序：

| 阶段 | 模型建议 | 目标 |
| --- | --- | --- |
| Tier 0 | 表格策略或小型神经网络 | 验证算法机制和理论 |
| Tier 1 | 1.5B 至 3B 级开放模型 | 验证证据解释与策略分叉 |
| Tier 2 初始 | 7B 级开放模型 | 验证复杂研究任务和参数内化 |
| Tier 2 扩展 | 14B 或其他更大模型 | 仅在 7B 方案已证明有效后考虑 |

如果沿用现有冻结基座，应继续使用已经审计的：

- checkpoint。
- revision。
- tokenizer。
- 时间截止。
- 模型知识污染检查。

如果引入新的 1.5B、3B 或 14B 基座，则需要额外冻结与审计，不能直接沿用 7B 的原冻结身份。

### 21.3 初始算法参数

~~~yaml
algorithm:
  name: pesco

  horizon:
    max_high_level_turns: 6
    branch_continuation_depth: 2
    finite_horizon_discount: 1.0

  branching:
    options_per_critical_state: 4
    minimum_exploration_seeds: 2
    preferred_exploration_seeds: 3
    independent_confirmation_seeds: 4
    use_common_random_numbers: true
    branch_only_at_selected_states: true

  advantages:
    estimator: leave_one_out
    detach_baseline: true
    normalize_within_dev_frozen_groups: false
    minimum_reversal_margin: 0.05

  objective:
    option_clip_epsilon: 0.2
    state_loss_weight: 0.2
    reversal_loss_weight: 0.5
    reference_kl_weight: 0.02
    validated_novelty_weight: 0.1
    normalized_cost_weight: 0.05

  evidence:
    probability_clip: 0.001
    invalid_claim_hard_gate: true
    independent_confirmation_required: true
    reward_supported_and_refuted_symmetrically: true

  optimization:
    policy_learning_rate: 0.000005
    gradient_clip_norm: 1.0
    train_adapter_seeds: [17, 29, 41]
~~~

这些参数必须根据：

- 执行器成本。
- 任务长度。
- 奖励分布。
- 模型规模。
- GPU 可用性。
- 环境噪声。

在开发集重新校准。

### 21.4 LoRA 或 QLoRA 参数

若且仅若科学门槛已经开启，可以考虑：

~~~yaml
adapter:
  method: lora_or_qlora
  rank_candidates: [16, 32]
  alpha_candidates: [32, 64]
  dropout: 0.05
  precision:
    quantization: nf4
    double_quantization: true
    compute_dtype: bfloat16
  candidate_learning_rates:
    - 0.000005
    - 0.00001
  sequence_length_candidates:
    - 8192
    - 16384
~~~

不得将：

> 已经完成过工程 smoke test。

等同于：

> 正式科学训练已经获得授权。

### 21.5 预算核算

单次训练总成本：

$$
C_{\mathrm{total}}
=
C_{\mathrm{policy}}
+
C_{\mathrm{executor}}
+
C_{\mathrm{branch}}
+
C_{\mathrm{verifier}}
+
C_{\mathrm{confirm}}
+
C_{\mathrm{teacher}}.
$$

每个实验配置单独记录：

- GPU 小时。
- CPU 小时。
- 峰值显存。
- 总生成 token。
- 环境执行次数。
- 平均实验耗时。
- 独立确认次数。
- 失败重试次数。
- 外部教师调用次数。
- 人工介入次数。

### 21.6 选择性分支

由于真实实验分支昂贵，建议只在关键状态开启：

$$
\operatorname{Branch}
(
s_t
)
=
\mathbf 1
\left[
\operatorname{VOI}
(
s_t
)
>
\tau_{\mathrm{branch}}
\right].
$$

可使用的候选触发条件包括：

- 当前证据突然变化。
- 模型对证据状态高度不确定。
- 当前实验被判为无效。
- 原假设被可靠反驳。
- 替代动作的价值估计接近。
- 策略长时间停留在单一方法族。
- 独立复现失败。

但选择机制必须记录清楚：

- 是否仅依赖动作采样前的状态。
- 是否引入动作相关选择偏差。
- 是否需要倾向概率或重要性修正。

---

## 22. 项目阶段、里程碑与决策门槛

### 22.1 阶段 0：冻结问题和研究假设

工作内容：

1. 冻结研究问题定义。
2. 冻结证据状态判定规则。
3. 冻结实验世界机制。
4. 冻结候选动作语义。
5. 冻结训练与评测划分。
6. 冻结评测器与独立确认协议。
7. 冻结主要指标与统计分析计划。
8. 冻结资源和网络策略。

交付物：

- 问题清单。
- 世界生成器。
- 验证器规范。
- 数据划分文件。
- 假设注册表。
- 指标冻结文件。
- 风险与泄漏审计清单。

进入下一阶段条件：

$$
\mathrm{freeze\_check}
=
\mathrm{PASS}.
$$

### 22.2 阶段 1：Tier 0 最小环境

工作内容：

- 建立有限状态实验世界。
- 定义高层动作和预算。
- 计算近似最优策略。
- 实现四类动态证据。
- 构造支持、反驳、不足和无效的可控轨迹。

初始 GO 标准：

~~~yaml
tier0_go:
  world_generation_reproducible: true
  all_four_evidence_states_reachable: true
  invalid_to_valid_transition_supported: true
  insufficient_to_resolved_transition_supported: true
  optimal_action_computable: true
  confirmed_preference_reversals_exist: true
  evidence_blind_policy_underperforms_oracle: true
~~~

如果这些条件不成立，说明研究问题还没有被环境正确表达。

### 22.3 阶段 2：验证器与快照工程

工作内容：

- 实现数据泄漏检查。
- 实现等效检验与区间判定。
- 实现独立随机种子验证。
- 实现快照、恢复与预算一致性。
- 编写无效实验和假复现负对照。
- 建立审计账本。

初始 GO 标准：

~~~yaml
verifier_go:
  validity_negative_controls_pass: true
  refutation_vs_insufficient_tests_pass: true
  leakage_detection_tests_pass: true
  fake_replication_tests_pass: true
  immutable_evaluator_check_pass: true
  snapshot_replay_consistency_pass: true
  hidden_world_id_access_denied: true
  hidden_confirmation_data_access_denied: true
~~~

### 22.4 阶段 3：现有模型零样本诊断

工作内容：

1. 运行基础模型。
2. 运行具备工具调用的研究代理。
3. 记录原始实验结果解释。
4. 记录实际高层动作。
5. 测量策略锁定和错误发现。
6. 识别模型最常见的科学失败模式。

建议输出：

~~~yaml
zero_shot_diagnosis:
  evidence_macro_f1: pending
  supported_action_accuracy: pending
  refuted_action_accuracy: pending
  insufficient_action_accuracy: pending
  invalid_action_accuracy: pending
  paired_world_flip_accuracy: pending
  invalid_claim_rate: pending
  effective_switch_rate: pending
  oracle_gap: pending
~~~

只有当：

$$
V(
\pi^{*}
)
-
V(
\pi_{\mathrm{base}}
)
$$

存在具有实际意义的差距时，才值得投入算法训练。

### 22.5 阶段 4：离线偏好反转学习

工作内容：

- 收集真实分支结果。
- 构造普通动作偏好对。
- 构造跨世界偏好反转对。
- 训练 SFT。
- 训练普通 DPO 或偏好基线。
- 训练 PESCO-Offline。

初始 GO 标准：

~~~yaml
offline_go:
  paired_world_flip_accuracy_improves: true
  invalid_claim_rate_does_not_increase: true
  refutation_acceptance_improves_or_matches: true
  matched_compute_comparison_pass: true
  no_final_split_access: true
~~~

如果离线反转学习无法优于普通偏好学习，应先检查：

- 反转数据是否真实。
- 训练标签是否存在噪声。
- 模型能否看到足够实验信息。
- 候选动作是否具有可比性。
- 信号是否被代码执行 token 淹没。

### 22.6 阶段 5：在线 PESCO 强化学习

工作内容：

- 启用真实 on-policy 策略采样。
- 执行关键状态分支。
- 计算留一策略优势。
- 构造配对世界反转训练样本。
- 加入独立确认与有效性约束。
- 对比 GRPO、SMOPD、DiscoPO、TCPO 等基线。

初始 GO 标准：

~~~yaml
online_go:
  scientific_hard_gate_pass: true
  executor_success_rate_acceptable: true
  verifier_negative_controls_pass: true
  paired_world_interactions_non_degenerate: true
  promotion_split_reserved: true
  final_id_split_locked: true
  final_ood_split_locked: true
  compute_accounting_enabled: true
~~~

### 22.7 阶段 6：冻结最终评测

工作内容：

- 锁定模型。
- 锁定超参数。
- 锁定方法选择。
- 首次访问最终隐藏评测。
- 运行同分布和 OOD 实验。
- 完成预注册统计检验。
- 执行独立重复与人工审计。

正式完成标准：

~~~yaml
final_completion:
  freeze_check_pass: true
  contamination_audit_pass: true
  all_primary_baselines_completed: true
  matched_budget_tables_completed: true
  ablations_completed: true
  negative_controls_completed: true
  replication_results_completed: true
  final_id_evaluation_completed: true
  final_ood_evaluation_completed: true
  cluster_bootstrap_completed: true
  primary_hypotheses_reported_without_posthoc_changes: true
~~~

### 22.8 建议进度

以下仅是研究组织建议，不保证实际完成时间：

| 周期 | 重点内容 | 主要产出 |
| --- | --- | --- |
| 第 1 至 2 周 | Tier 0、证据分类规则、动作空间、问题冻结 | 可运行模拟器与最优动作 oracle |
| 第 3 至 4 周 | Tier 1、验证器、快照、负对照、零样本诊断 | 可执行配对实验世界与失败分析 |
| 第 5 至 6 周 | SFT、GRPO、普通偏好学习与 PESCO-Offline | 策略反转学习的初步证据 |
| 第 7 至 9 周 | 在线 PESCO、预算控制、关键状态分叉 | 完整算法与主要 baseline |
| 第 10 至 12 周 | 消融、OOD、复现、统计检验、最终冻结评测 | 方法论文级实验结果 |

若验证器或实验环境尚未成熟，应延长早期阶段，而不是提前进入大规模训练。

---

## 23. 与现有项目冻结协议的衔接

### 23.1 当前已知项目状态

根据此前讨论，现有实验状态为：

~~~yaml
canonical_parser_valid: 5/5
executable_cases: 4/5
all_four_seeds: PASS
negative_controls: 7/7 PASS
scientific_hard_pass: 0/5
sdk_contract_gate: PASS
scientific_capability: NOT_ESTABLISHED
qlora: NO_GO
~~~

这意味着：

- 接口和行政格式问题已经基本解决。
- 多数示例可以进入执行环境。
- 负对照具备一定基础。
- 但真正的科学能力尚未建立。
- 正式 QLoRA 晋级条件尚未满足。

### 23.2 当前阶段允许推进的内容

可以优先推进：

1. Tier 0 模拟器。
2. 四类证据的可信规则。
3. 世界机制与动态状态转换。
4. 同状态分支快照。
5. 闭源教师 engineering dry-run。
6. 非最终评测集的离线分支数据。
7. 研究动作分类器。
8. 小规模 CPU 环境验证。
9. 反转数据格式与损失函数单元测试。

### 23.3 当前阶段不应直接解锁的内容

在：

$$
\mathrm{scientific\_hard\_pass}
=
0/5
$$

时，不应直接宣称进入：

- 7B 正式 QLoRA。
- 14B 正式模型训练。
- champion 晋级。
- held-out 最终结果评估。
- OOD 最终测试访问。
- 递归多代自进化。

### 23.4 五分割协议兼容

如果现有项目已经冻结：

- Train。
- Dev。
- Promotion。
- Final ID。
- Final OOD。

则 PESCO 应复用该分层结构。

若新增：

- 新模型。
- 新研究问题。
- 新世界生成机制。
- 新验证器。
- 新 OOD 定义。
- 新 promotion 规则。

应创建新的冻结版本，例如：

$$
\mathrm{freeze\_v0.2}
$$

而不是修改已冻结的：

$$
\mathrm{freeze\_v0.1}.
$$

### 23.5 与现有 7B smoke test 的关系

此前单卡 QLoRA smoke test 只能证明：

> 工程上可以加载模型、执行若干更新、处理 token mask 和生成 adapter。

不能证明：

- 证据判断能力。
- 科学假设提出能力。
- 策略切换能力。
- 无提示方法发现能力。
- OOD 泛化。
- 独立复现能力。

因此：

$$
\mathrm{engineering\ smoke\ test}
\neq
\mathrm{scientific\ training\ authorization}.
$$

### 23.6 现有最终样本量的限制

如果现有冻结方案包含：

$$
N_{\mathrm{final,ID}}
=
24
$$

以及：

$$
N_{\mathrm{final,OOD}}
=
16,
$$

这些数量可以支持：

- 初步严谨评测。
- 大效应诊断。
- 工程可行性分析。

但未必足以稳定识别较小的方法差异。

如果需要扩展，必须：

1. 先完成先导实验和功效估计。
2. 新建冻结协议。
3. 明确替代或扩展规则。
4. 保持历史结果的解释边界。
5. 不因已经看到最终结果而随意调整测试样本。

---

## 24. 风险、失败模式与应对

| 风险 | 典型表现 | 可能原因 | 应对策略 |
| --- | --- | --- | --- |
| world ID 泄漏 | 不看实验结果也能选择正确动作 | 文件名、元数据或数据顺序泄漏 | 随机化路径、隐藏标签、运行证据遮蔽负对照 |
| 验证器作弊 | 模型修改评测程序获得高分 | 权限隔离不足 | 独立只读验证器、容器隔离、审计日志 |
| 负结果惩罚 | 模型不断寻找支持性结果 | 奖励把 Refuted 当失败 | 严格评分规则、可靠负结果专项指标 |
| 不足被误判反驳 | 低样本量结果被报告为无效 | 不恰当的显著性解释 | 最小实际效应、置信区间、等效检验 |
| 无效高分 | 泄漏实验产生漂亮指标 | 只关注最终分数 | 实验有效性硬门控 |
| 假复现 | 多次使用相同种子或相同切分 | 复现协议不完整 | 冻结独立确认种子与数据哈希 |
| 文本新颖性作弊 | 改写方法名称获得奖励 | 使用 embedding 距离或 LLM 自评 | 结构化方法族、代码行为和独立复现 |
| 策略乱切换 | 频繁更换方法但收益下降 | 奖励切换次数 | 用真实策略优势奖励有效切换 |
| 保守模式坍塌 | 只保留容易执行的常见策略 | 稀疏发现奖励与终局分数偏置 | 保留有效策略多样性、报告 pass@k |
| 小样本幸运动作 | 单个随机种子产生虚假优势 | 分支估计高方差 | 配对种子、收缩估计、置信下界 |
| 离线分布偏移 | 在教师轨迹上有效，在自身轨迹上失效 | 仅训练外部示范 | on-policy 分支和逐步提示退火 |
| 策略执行混淆 | 模型提高写代码能力但没有学会选方向 | 全轨迹回报广播 | 固定执行器、策略 token 屏蔽 |
| 伪因果表述 | 将世界差异直接说成证据因果效应 | 干预假设不足 | 明确 estimand、配对条件和识别假设 |
| 测试集污染 | 在最终隐藏任务上调参 | 冻结协议未隔离 | 五分割、审计账本、单次最终评估 |
| 算力不公平 | 额外分支带来更高收益 | 基线没有相同实验预算 | 同 GPU、同环境调用、同总成本比较 |
| 后验评分作弊 | 模型故意先报告错误信念 | 允许回写或重置信念 | 实验前承诺、不可修改账本、轨迹级评分 |
| 过强模板依赖 | 训练状态固定对应动作 | 环境构造过于简单 | 引入预算、目标、多个合理动作与 OOD 机制 |
| 反转标签噪声 | 两个世界最优动作差异来自随机误差 | 重复实验不足 | 配对置信区间、独立确认、低置信样本剔除 |

### 24.1 最危险的学术风险

最危险的情况是：

> 实际方法只是 GRPO 加四个奖励和一个偏好损失，但论文将其描述为新的科学发现算法。

需要通过以下证据避免该问题：

1. 明确给出新的策略反转学习目标。
2. 给出与 TCPO、CVT-RL、Ecpo 的实质区别。
3. 说明动作语义、世界机制和科学验证条件。
4. 给出统计、理论或样本效率分析。
5. 在强基线和公平预算下验证新增机制。

### 24.2 最危险的工程风险

最危险的工程问题是：

> 环境不具备可信分叉、标签不可靠、验证器可以绕过，但直接进入大模型在线 RL。

解决顺序应为：

$$
\text{可信验证}
\rightarrow
\text{可信分叉}
\rightarrow
\text{可靠动作差异}
\rightarrow
\text{离线学习}
\rightarrow
\text{在线训练}.
$$

---

## 25. 最小可行实验

### 25.1 单问题最小实验

选择一个明确问题：

> 方法 A 是否能够提升分组独立测试集上的预测表现？

创建四个冻结世界：

~~~yaml
worlds:
  supported:
    true_effect: positive
    leakage: false
    sample_size: adequate

  refuted:
    true_effect: negligible
    leakage: false
    sample_size: adequate

  insufficient:
    true_effect: uncertain_or_small
    leakage: false
    sample_size: inadequate

  invalid:
    true_effect: negligible
    leakage: true
    sample_size: apparently_adequate
~~~

每个世界的初始提示保持一致：

~~~text
研究问题：

请判断方法 A 是否能够改善目标任务表现。你可以运行实验、修改实现、
检查数据划分，并在预算内决定下一步研究动作。

注意：

任务没有指定应该继续原方法、增加样本、修复实验还是切换方法。
~~~

该提示不能暴露任何隐藏世界编号。

### 25.2 固定四个候选动作

~~~yaml
candidate_options:
  - continue_current_method
  - add_samples_or_seeds
  - repair_data_split
  - switch_to_alternative_method
~~~

这一步是环境机制验证，不等于正式评估模型自主生成候选动作的能力。

### 25.3 真实执行分支

对每个世界：

$$
4
\text{ 个候选动作}
\times
4
\text{ 个冻结种子}.
$$

总计：

$$
4
\text{ 个世界}
\times
4
\text{ 个动作}
\times
4
\text{ 个种子}
=
64
$$

次轻量实验。

另行保留独立确认种子，不将探索种子重复当作复现证据。

### 25.4 最小实验要回答的问题

1. 四个世界是否确实产生不同的可信实验状态？
2. 无效世界的表面高分是否被验证器阻止？
3. 证据不足世界能否通过追加实验转化为明确状态？
4. 可靠反驳世界是否能产生正确的负结果？
5. 支持世界是否支持原方法的独立复现？
6. 不同世界中是否出现可确认的动作偏好反转？
7. 当前模型是否会忽略实验结果并坚持默认策略？
8. 反转数据是否足够干净，可以训练离线偏好模型？

### 25.5 期望输出表

| 世界 | 继续 | 增加样本 | 修复实验 | 切换方法 | 最佳动作 | 是否存在可识别反转 |
| --- | --- | --- | --- | --- | --- | --- |
| Supported | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Refuted | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Insufficient | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Invalid | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

必须根据实际结果填表，而不是预先把某个动作写死为正确答案。

### 25.6 单问题 GO / NO-GO

~~~yaml
minimum_pilot_decision:
  all_worlds_execute: required
  scientific_verifier_is_independent: required
  invalid_world_is_detected: required
  insufficient_is_not_labeled_refuted: required
  supported_and_refuted_are_distinguishable: required
  at_least_one_confirmed_strategy_reversal: required
  no_world_identifier_leakage: required
  reproducible_branch_results: required
~~~

若任一 required 条件失败：

$$
\text{algorithm training}
=
\text{NO-GO}.
$$

### 25.7 从单问题扩展到先导实验

初始扩展：

$$
10
\text{ 个研究问题}
\times
4
\text{ 个世界}
\times
4
\text{ 个随机种子}
=
160
$$

个环境实例。

如果每个关键状态平均执行：

$$
3\sim5
$$

个动作分支，则大约得到：

$$
480\sim800
$$

个分支结果。

这一规模用于：

- 工程验证。
- 初步诊断。
- 先导效应估计。
- 基线可行性。
- 数据质量检查。

不能直接将其描述为足够强的正式泛化结论。

---

## 26. 最终结果汇报模板

### 26.1 总体实验结果

| 方法 | VRS | 状态 Macro-F1 | 策略反转准确率 | 有效切换率 | 错误发现率 | 独立复现率 | 总成本 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| SFT | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| GRPO-FourState | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| SMOPD | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| DiscoPO | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| TCPO | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| CVT-RL / Ecpo | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| PESCO-Offline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| PESCO-Full | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

### 26.2 分证据状态结果

| 方法 | Supported | Refuted | Insufficient | Invalid | 状态平均 |
| --- | --- | --- | --- | --- | --- |
| Base | 待测 | 待测 | 待测 | 待测 | 待测 |
| SFT | 待测 | 待测 | 待测 | 待测 | 待测 |
| GRPO-FourState | 待测 | 待测 | 待测 | 待测 | 待测 |
| SMOPD | 待测 | 待测 | 待测 | 待测 | 待测 |
| TCPO | 待测 | 待测 | 待测 | 待测 | 待测 |
| PESCO-Full | 待测 | 待测 | 待测 | 待测 | 待测 |

### 26.3 单路径与推理期搜索比较

| 方法 | 训练是否更新参数 | 推理分支数 | 单路径成功率 | 多分支成功率 | 总实验调用 |
| --- | --- | --- | --- | --- | --- |
| Base | 否 | 1 | 待测 | 不适用 | 待测 |
| Search-Only | 否 | 4 | 待测 | 待测 | 待测 |
| PESCO-Full | 是 | 1 | 待测 | 不适用 | 待测 |
| PESCO-Full | 是 | 4 | 待测 | 待测 | 待测 |

如果：

$$
\mathrm{PESCO}_{\mathrm{one\ path}}
>
\mathrm{Base}_{\mathrm{one\ path}},
$$

才能支持：

> 模型参数中内化了一部分策略修正能力。

如果：

$$
\mathrm{PESCO}_{\mathrm{one\ path}}
\approx
\mathrm{Base}_{\mathrm{one\ path}}
$$

但：

$$
\mathrm{PESCO}_{\mathrm{tree}}
>
\mathrm{Base}_{\mathrm{one\ path}},
$$

则不能排除提升主要来自额外搜索。

### 26.4 消融结果模板

| 配置 | VRS | FlipAcc | RefutationAcceptance | InvalidClaimRate | VNPR |
| --- | --- | --- | --- | --- | --- |
| 完整 PESCO | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除配对世界 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除反转损失 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除分支优势 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除严格评分 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除有效性门控 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除独立复现 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 去除新路径证书 | 待测 | 待测 | 待测 | 待测 | 待测 |

### 26.5 OOD 汇报模板

| OOD 类型 | 示例 | 目标 |
| --- | --- | --- |
| 未见研究问题 | 新任务描述 | 检查是否记住训练问题 |
| 未见方法族 | 新的可行训练方法 | 检查无提示方法泛化 |
| 未见混杂机制 | 新型数据污染或泄漏 | 检查实验有效性迁移 |
| 未见数据来源 | 不同领域数据 | 检查跨数据分布能力 |
| 未见预算 | 更少实验机会 | 检查预算条件化决策 |
| 未见效应强度 | 更弱或更接近阈值 | 检查证据校准 |
| 未见执行器 | 换用不同低层执行器 | 分离策略能力和执行能力 |

### 26.6 最终声明边界

如果实验成立，可以声明：

> 在冻结的可执行科研环境中，PESCO 相对于同预算基线提高了证据条件化的研究策略修正能力。

如果额外具备充分证据，可以进一步声明：

> PESCO 在未见研究问题上提高了无提示且经过独立验证的新方法发现率。

除非满足额外证据条件，不应直接声明：

- 模型已经具备完整人类科研能力。
- 模型可以可靠解决任意未知科学问题。
- 模型发现了历史上从未存在过的方法。
- 方法已经证明能够递归自我改进。

---

## 27. 预期贡献与论文结构

### 27.1 预期方法贡献

1. 提出面向自主科研策略修正的部分可观察、受约束决策建模。
2. 提出配对实验世界中的策略偏好反转学习目标。
3. 提出基于真实同状态实验分支的策略级信用分配。
4. 通过严格评分规则统一可靠支持与可靠反驳的学习信号。
5. 将实验有效性、独立确认和预算纳入研究决策优化。
6. 提出可验证的无提示新路径发现证书。

### 27.2 预期实证贡献

1. 证明当前模型存在显著的策略锁定。
2. 展示模型会混淆反驳、证据不足和实验无效。
3. 证明固定奖励专家融合不足以保证跨世界策略反转。
4. 证明同预算下配对世界反转训练能够改善研究动作质量。
5. 证明可信负结果可以成为正向学习信号。
6. 证明验证门控能够减少虚假发现。
7. 检查模型参数是否真正内化策略调整能力。

### 27.3 预期理论贡献

1. 严格评分规则与实验信息增益的关系。
2. 留一分支优势估计的条件无偏性。
3. 证据盲策略与条件化策略的性能差距。
4. 配对世界双重比较的噪声抵消条件。
5. 自适应实验中的错误发现控制框架。

### 27.4 推荐论文结构

1. Introduction。
2. Why execution success does not imply strategy revision。
3. Related work and novelty boundary。
4. Evidence-conditioned research decision process。
5. Paired-world strategy preference reversals。
6. PESCO algorithm。
7. Trusted scientific verification。
8. Theory and estimator assumptions。
9. Experimental environments and frozen protocols。
10. Main results under matched compute。
11. Ablations, negative controls and OOD evaluation。
12. Limitations and failure cases。
13. Conclusion。

### 27.5 最重要的中心主张

最终论文需要证明的核心不是：

> 模型可以尝试更多方法。

而是：

> 模型能够判断当前实验结果意味着什么，并根据不同的真实证据自主决定是否继续、追加、修复、停止或切换研究策略。

对应的最核心实验现象为：

$$
\boxed{
\text{同一研究问题}
+
\text{不同实验反馈}
\Longrightarrow
\text{不同且经过验证的最优研究动作}
}.
$$

---

## 28. 参考文献

1. Wang et al. SMOPD: Multi-Reward Reinforcement Learning via Specialize-and-Merge Online Policy Distillation. 2026. [https://arxiv.org/abs/2608.03092](https://arxiv.org/abs/2608.03092)

2. Liu et al. GDPO: Group reward-Decoupled Normalization Policy Optimization. 2026. [https://arxiv.org/abs/2601.05242](https://arxiv.org/abs/2601.05242)

3. Feng et al. Group-in-Group Policy Optimization for LLM Agent Training. 2025. [https://arxiv.org/abs/2505.10978](https://arxiv.org/abs/2505.10978)

4. Ji et al. Tree Search for LLM Agent Reinforcement Learning. 2025. [https://arxiv.org/abs/2509.21240](https://arxiv.org/abs/2509.21240)

5. Wu et al. Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning. 2026. [https://arxiv.org/abs/2601.20209](https://arxiv.org/abs/2601.20209)

6. Xu et al. Scaling Scientific Discovery Environments for Turn-Level Agentic RL. 2026. [https://arxiv.org/abs/2607.28990](https://arxiv.org/abs/2607.28990)

7. Liao et al. TCPO: Turn-Level Credit Policy Optimization. 2026. [https://arxiv.org/abs/2608.01667](https://arxiv.org/abs/2608.01667)

8. Meng. Policy-Conditioned Counterfactual Credit for Verifiable Reinforcement Learning of Long-Horizon Language Agents. 2026. [https://arxiv.org/abs/2606.05263](https://arxiv.org/abs/2606.05263)

9. Li et al. When Denser Credit Is Not Enough: Evidence-Calibrated Policy Optimization for Long-Horizon LLM Agent Training. 2026. [https://arxiv.org/html/2606.05885v1](https://arxiv.org/html/2606.05885v1)

10. Wang et al. Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents. 2025–2026. [https://arxiv.org/abs/2510.14967](https://arxiv.org/abs/2510.14967)

11. Agarwal et al. AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise. 2025–2026. [https://arxiv.org/abs/2507.00310](https://arxiv.org/abs/2507.00310)

12. Takahara and Mizoguchi. Toward Auditable AI Scientists: A Hypothesis Evolution Protocol for LLM Agents. 2026. [https://arxiv.org/abs/2607.09195](https://arxiv.org/abs/2607.09195)

13. Li et al. RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training. 2026. [https://arxiv.org/abs/2608.18682](https://arxiv.org/abs/2608.18682)

14. Lim et al. What is Missing from AI Post-Training AI: An Empirical Analysis. 2026. [https://arxiv.org/abs/2608.19072](https://arxiv.org/abs/2608.19072)

15. Si et al. Towards Execution-Grounded Automated AI Research. 2026. [https://arxiv.org/abs/2601.14525](https://arxiv.org/abs/2601.14525)

16. Stoffl et al. VERITAS: A Multi-Agent Co-Scientist for Verifiable Image-Derived Hypothesis Testing. 2026. [https://arxiv.org/abs/2604.12144](https://arxiv.org/abs/2604.12144)

17. How Do Agents Fail on AutoResearch: End-to-End Diagnostic Evaluation on 100 Real-World Frontier Research Tasks. 2026. [https://arxiv.org/abs/2608.14905](https://arxiv.org/abs/2608.14905)

18. Ramdas et al. Game-theoretic statistics and safe anytime-valid inference. 2022–2023. [https://arxiv.org/abs/2210.01948](https://arxiv.org/abs/2210.01948)

19. Kumar et al. Training Language Models to Self-Correct via Reinforcement Learning. 2024. [https://arxiv.org/abs/2409.12917](https://arxiv.org/abs/2409.12917)

20. Antoniades et al. Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty. 2026. [https://arxiv.org/abs/2606.25198](https://arxiv.org/abs/2606.25198)

---

## 附录 A：一页式执行摘要

### A.1 算法设计

PESCO 的核心结构是：

$$
\text{严格评分规则}
+
\text{同状态实验分支}
+
\text{跨世界策略反转}
+
\text{可信验证与独立复现}.
$$

训练时：

1. 提供同一研究问题的多个真实实验世界。
2. 让模型自行解释当前实验结果。
3. 在同一状态下执行多个真实研究动作。
4. 通过隐藏验证器评估动作的科学价值。
5. 识别不同世界中的策略偏好反转。
6. 将反转信号和分支优势用于高层策略更新。
7. 对无提示、真实有效且可复现的新路径提供额外奖励。

### A.2 实验设计

先验证：

$$
\mathrm{Tier\ 0}
$$

数学模拟。

再验证：

$$
\mathrm{Tier\ 1}
$$

真实轻量实验。

最后进入：

$$
\mathrm{Tier\ 2}
$$

真实后训练研究任务。

重点 baseline：

- SFT。
- GRPO。
- SMOPD。
- DiscoPO。
- TCPO。
- CVT-RL / Ecpo。
- Search-Only。

### A.3 核心评价

主要检查：

1. 是否正确识别四种证据状态。
2. 是否接受可靠负结果。
3. 是否修复无效实验。
4. 是否根据实验世界改变策略。
5. 是否自主提出经过确认的新方法。
6. 是否在同等预算下优于强基线。
7. 是否在隐藏问题和 OOD 环境中泛化。

### A.4 当前最优先事项

现阶段最先完成：

> 一个研究问题、四个实验世界、四个候选动作、真实同状态分叉、独立可信验证，以及至少一组经过确认的策略偏好反转。

在此之前：

$$
\text{正式大模型在线 RL}
=
\text{NO-GO}.
$$

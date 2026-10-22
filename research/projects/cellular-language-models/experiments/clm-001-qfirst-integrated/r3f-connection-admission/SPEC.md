# MiniCells R3e Connection 更新准入与联合训练修复规格

**版本：** 1.0  
**工作类型：** 实验协议修复、路由稳定性修复、联合训练验证  
**目标：** 修复 Connection 提案的幅度控制和准入问题，使有效更新能够进入 Cell 适应训练，并通过严格的实验协议验证其实际收益。

---

## 1. 背景与当前问题

### 1.1 R3e 的实验结果

R3e 已完成全部预定运行。

- Source commit：`db5b3b5fea4fcc14b0d65451d4495c00c77aed58`
- Config hash：`cd56bfdf6fb53b470d37558e3a45dec34ec43293974662bf12919735d7225113`
- 三个训练 Seed：`151001301`、`151001302`、`151001303`

实验涉及以下策略：

- A-F：固定 Top-2 路由。
- A-T：Connection 更新通过 Trust Region 后，使用当前 Cell 进行质量评价。
- A-C：Connection 更新通过 Trust Region 后，进行 Cell 适应训练再评价。
- B-F：固定 Gate 的单 Cell 顺序执行。
- B-R：确定性哈希随机路由。
- B-C：学习单 Cell 顺序路由，并进行 Cell 适应训练。

R3e 的 36 次 Connection 提案全部被 Trust Region 拒绝。

结果包括：

1. A-T、A-C、A-F 的每个 Seed 均具有完全相同的 Development NLL。
2. B-C 与 B-F 的每个 Seed 均具有完全相同的 Development NLL。
3. A-C、B-C 均未真正进入 Connection 更新后的 Cell 适应评价阶段。
4. 部分提案在极低 KL 下仍产生极高的硬动作变化率和完整路径变化率。

因此，R3e 尚未实质检验新 Connection 的联合优化效果。

### 1.2 已确认的实现风险

当前源码存在以下值得修复的问题。

**问题 A：A 组在零修正与非零修正之间切换动作规则。**

当前 `stable_connection.py` 使用类似逻辑：

```python
action = (
    base
    if (delta == 0).all(-1)
    else q.argmax(-1)
)
```

严格零修正时使用稳定 Top-2；出现任意非零修正时则切换到 Pair 分数的 `argmax`。

两套选择规则在浮点并列和舍入情况下可能不一致。因此，即使 Connection 修正极小，也可能产生非预期的路由翻转。

**问题 B：参数插值不等于已经验证的路由稳定性。**

当前通过以下方式缩放提案：

\[
\theta_\alpha
=
\theta_{\mathrm{old}}
+\alpha(\theta_{\mathrm{new}}-\theta_{\mathrm{old}})
\]

对于当前线性 Connection，这可以控制固定状态上的 Logit 修正幅度，但不能保证 Argmax 路由变化率随 \(\alpha\) 平滑下降。

**问题 C：Trust Region 的 KL 与硬路径变化衡量不同性质。**

小 KL 表明分类分布变化有限，却不能保证硬 Argmax 动作不变。连续状态下的早期动作变化还可能改变后续隐藏状态，进一步造成路径分叉。

**问题 D：当前双分支适应使用独立路由随机流。**

`adaptation.py` 为 old/new 分支分别设置不同的 `route_rng` 种子。这会使两条分支即使使用完全相同的 Connection，也可能因为 Gumbel 采样差异而产生不同的 Cell 更新。

这些问题需要在新实验协议中分别处理，不能仅凭放宽阈值来掩盖。

---

## 2. 实施目标与非目标

### 2.1 必须实现的目标

1. 使 Connection 修正的实际 Logit 幅度可测量、可缩放、可复现。
2. 消除零修正与极小非零修正之间不必要的硬路由不连续性。
3. 保留 KL、硬动作变化率和完整路径变化率三个独立准入门槛。
4. 使通过 Trust Region 的 Connection 能够真正进入 A-C/B-C 的 Cell 适应训练。
5. 使 old/new Cell 适应对照具有相同的数据顺序、Cell 初始状态、优化器状态和配对随机数。
6. 完整记录提案、回溯、准入、适应、最终接受和拒绝原因。
7. 保留原始 R3e 结果，不覆盖历史产物。

### 2.2 明确不做的事情

本次不进行以下修改：

- 不改变 Cell 网络结构。
- 不改变 A 组 Top-2 计算机制。
- 不改变 B 组单 Cell 顺序执行的基本架构。
- 不增加新的模型容量。
- 不使用 Development 或 Confirmation 数据选择超参数。
- 不通过放宽门槛强迫候选通过。
- 不把随机路由优于原始 Gate 直接解释为随机路由具有普遍优势。
- 不宣称本轮结果已经证明 Cell 专业化、小世界拓扑或功能分工。

本次工作首先修复实验机制。是否改善模型质量，必须由后续结果决定。

---

## 3. 实验分支与数据保护

### 3.1 新实验身份

在原始 R3e 的源代码与配置基础上建立独立工作分支。

建议分支名：

`codex/r3e-connection-admission-fix`

建议新的实验标识：

`r3f-connection-admission`

R3e 原始 Source、Config、Checkpoints、Metrics、Adaptation Results 和 Graph Summary 必须保持不变。

如果后续发现需要改变策略定义，而不仅仅是修复数值一致性，应在新实验配置中明确记录，不得将其描述为原始 R3e 的重跑。

### 3.2 数据隔离

继续遵守以下边界：

| 数据 | 允许用途 |
|---|---|
| Train | Connection 提案优化、Cell 训练 |
| Calibration | 路由稳定性诊断、Trust Region、质量准入、固定阈值校准 |
| Development | 所有参数和协议锁定后的终端评价 |
| Confirmation | 保持封存，不读取 |

Development 曾用于历史实验，因此本轮不能把它作为全新的独立确认集。

所有阈值、回溯尺度、门槛和配置必须在终端评价前锁定。

---

## 4. 第一阶段：建立只读诊断工具

本阶段不得启动新的完整训练。

### 4.1 检查零修正一致性

对 A-T、A-C、B-C 分别构造：

- 原始 Connection。
- 深拷贝且 Connection 参数完全相同的候选模型。
- 相同 Calibration 输入。
- 相同隐藏状态、前驱动作和随机状态。

验证以下指标：

\[
D_{\mathrm{KL}}=0
\]

\[
R_{\mathrm{action}}=0
\]

\[
R_{\mathrm{path}}=0
\]

实际断言使用适当的浮点容差，但硬动作与路径变化次数必须严格为零。

如果同参数模型仍然出现动作或路径差异，立即停止后续实验，并报告不一致发生的位置。

### 4.2 记录 Gate 的 Logit Margin

对固定 Calibration 参考状态，记录原始 Gate 的动作 Margin。

A 组使用原始 Gate 第 2 名和第 3 名 Cell 的分数差：

\[
m_A=z_{(2)}-z_{(3)}
\]

因为 Top-2 路由需要用第三名 Cell 替换当前第二名 Cell 才能改变选中集合。

B 组使用第 1 名和第 2 名 Cell 的分数差：

\[
m_B=z_{(1)}-z_{(2)}
\]

分别输出：

- 最小值、最大值、中位数。
- 第 1、5、10、25、50、75、90、95、99 百分位。
- 精确并列率。
- 数值近并列率。
- 原始 Gate 动作与 Pair 分数 `argmax` 的不一致率。
- 各动作变化发生时的原始 Margin。

所有统计只使用 Calibration 参考状态。

### 4.3 执行尺度扫描

在不修改模型参数的情况下，评估：

\[
\alpha\in
\{0,10^{-8},10^{-6},10^{-4},10^{-3},
10^{-2},1/64,1/16,1/4,1\}
\]

每个尺度均输出：

- 参数变化范数。
- Connection 修正 Logit 的 RMS 和最大绝对值。
- 新旧策略 KL。
- 硬动作变化率。
- 完整路径变化率。
- 动作变化与原始 Margin 的关系。

尺度扫描必须使用相同参考输入和确定性路由。

**本阶段只进行诊断，不根据 Development 结果选择尺度。**

输出文件建议为：

`ROUTING_DIAGNOSTICS.json`

---

## 5. 第二阶段：统一路由选择规则

修改：

`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/stable_connection.py`

必要时同步修改 `graph.py`、`evaluation.py` 和相关测试。

### 5.1 明确区分三个概念

代码必须明确区分：

1. 原始 Gate 分数。
2. Connection 修正后的原始分数。
3. 确定性硬路由动作。

A 组：

\[
q_{ij}=z_i+z_j+\delta_{ij}
\]

B 组：

\[
q_i=z_i+\delta_i
\]

不得在不同函数中隐式切换这些分数的定义。

### 5.2 统一零修正语义

当 Connection 修正为零时：

- A 组必须严格复现原始稳定 Top-2。
- B 组必须严格复现原始 Gate 的稳定 Top-1。
- 不得因为 Pair 分数求和的浮点舍入而选择不同动作。

为此，新增统一的基线路由与硬路由选择函数。不要继续依赖“修正是否逐元素恰好等于零”来切换两套动作算法。

### 5.3 引入明确的路由切换 Margin

为了防止极小修正引起大量硬路由翻转，引入显式参数：

`route_switch_margin_fraction`

建议初始值固定为 `0.05`，作为新实验的预注册起始值，不得在观察 Development 后调整。

在每个 Seed 的 Calibration 参考状态上计算：

\[
\tau=
\max\left(
\tau_{\mathrm{numeric}},
0.05\cdot\operatorname{median}(m)
\right)
\]

其中：

- \(m\) 是对应架构的基线路由 Margin。
- \(\tau_{\mathrm{numeric}}\) 是根据浮点精度和分数尺度确定的数值容差。
- \(\tau\) 在实验启动时校准一次并冻结。
- 不得针对每次候选单独调整 \(\tau\) 来强迫通过。

硬路由规则：

1. 计算基线路由 `base_action`。
2. 计算修正后的最佳动作 `candidate_action`。
3. 如果两者相同，保留基线路由。
4. 如果不同，只有当修正后的最佳分数比基线路由对应分数高出超过 \(\tau\) 时，才采用新动作。
5. 否则保留基线路由。

概念伪代码：

```python
base_action = baseline_action(z)
candidate_action = raw_scores.argmax(dim=-1)

base_score = raw_scores.gather(
    -1, base_action[:, None]
).squeeze(-1)

candidate_score = raw_scores.gather(
    -1, candidate_action[:, None]
).squeeze(-1)

switch = (
    (candidate_action != base_action)
    & ((candidate_score - base_score) > route_switch_margin)
)

action = torch.where(
    switch,
    candidate_action,
    base_action,
)
```

该规则必须同时应用于 A、B 两组，并保持各自的合法动作空间。

这里的 Margin 是明确的策略设计，而非单纯的数值修补。配置、日志和实验报告必须记录它。

### 5.4 区分随机训练策略与确定性推理策略

保留现有 Gumbel-Max 训练采样与 epsilon 探索机制，不得在此次修复中静默移除。

具体要求：

- 训练时继续从修正后的原始分数生成 Gumbel 采样。
- 确定性评价时使用上述 Margin 稳定的硬路由规则。
- Trust Region 的 KL 衡量修正前后的原始分类分布。
- 硬动作变化率与路径变化率衡量 Margin 稳定后的实际确定性路由。
- 记录两种路由语义，不得将训练采样的变化率与确定性硬路由变化率混为一谈。

现有训练策略包含 epsilon 均匀探索。因此，基于原始 Softmax 的 KL 是不含共同均匀混合项的保守分布比较；保持这一口径，并在文档和指标中注明。

---

## 6. 第三阶段：使更新幅度真正可控

修改：

`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/policy_proposal.py`

### 6.1 保留当前梯度提案优化

继续使用现有的反事实终端 Q、中心化成本和 Connection 参数优化机制。

不得使用 Development 标签或 Confirmation 数据优化 Connection。

梯度阶段负责提出候选，不负责决定候选是否可以进入正式训练。

### 6.2 以实际 Logit 变化验证尺度

当前 Connection 为线性映射，因此在固定输入、隐藏状态和 Gate 输出上，参数插值应当与 Logit 插值一致。

对于旧参数与未缩放候选参数：

\[
\theta_\alpha=
\theta_{\mathrm{old}}
+\alpha(\theta_{\mathrm{candidate}}-\theta_{\mathrm{old}})
\]

必须验证：

\[
\delta_\alpha
\approx
\delta_{\mathrm{old}}
+\alpha(\delta_{\mathrm{candidate}}-\delta_{\mathrm{old}})
\]

在固定 Calibration 参考状态上检查该等式的数值误差。

如果实际 Logit 变化与预测尺度不一致，必须停止提案并报告原因。

每个回溯尺度均记录：

- `scale`
- `parameter_delta_norm`
- `score_delta_rms`
- `score_delta_max_abs`
- `kl`
- `action_change`
- `path_change`
- `passed`
- `reject_reasons`

### 6.3 扩展回溯范围

检查当前配置中的 `proposal.backtrack`，确保尺度按严格递减顺序排列，并覆盖：

\[
1,\frac12,\frac14,\frac18,\ldots,\frac1{1024}
\]

如果当前配置已有更合适的尺度集合，可以保留，但必须具有足够的分辨率来定位通过边界。

不得假设动作变化率一定随尺度单调下降。必须实际评估每个尺度的指标。

### 6.4 不得接受数值意义上的空更新

在每个候选尺度上，检查有效 Logit 变化和策略变化。

如果候选在数值精度范围内等同于旧策略，且确定性动作与路径完全不变，应拒绝该候选：

`NO_EFFECTIVE_UPDATE`

不能仅仅因为 KL 为零、动作变化为零而将其判定为“稳定且成功”。

反过来，也不能仅仅因为硬动作变化率较高就忽略 KL 和 Margin 诊断。所有准入指标必须分别记录。

---

## 7. 第四阶段：重建 Trust Region 准入流程

修改：

`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/evaluation.py`

### 7.1 保留三个独立准入门槛

本轮默认保留原有门槛，不因 R3e 失败而放宽：

| 指标 | 准入条件 |
|---|---:|
| KL | 不超过 0.01 |
| 硬动作变化率 | 不超过 10% |
| 完整路径变化率 | 不超过 20% |

实际阈值必须从配置读取，不得在代码中另写一套隐含默认值。

### 7.2 统一指标定义

**KL**

在相同参考状态上比较旧 Connection 与候选 Connection 的原始分类分布。

**硬动作变化率**

在旧模型产生的相同隐藏状态上，比较旧、新模型通过统一 Margin 规则选出的硬动作。

**完整路径变化率**

分别运行旧模型和候选模型的确定性完整前向过程，比较整条动作序列是否存在任何差异。

路径变化必须使用独立完整前向计算，不得用同一隐藏状态上的局部动作变化代替。

### 7.3 输出明确的准入原因

每个候选必须分别记录：

- `kl_pass`
- `action_change_pass`
- `path_change_pass`
- `effective_update`
- `trust_region_passed`
- `chosen_scale`

如果没有任何尺度通过，使用：

`ALL_TRUST_REGION_SCALES_REJECTED`

同时记录所有尺度的失败指标。

如果全部通过的尺度均属于数值空更新，则使用：

`NO_EFFECTIVE_UPDATE`

不得把两种情况合并。

---

## 8. 第五阶段：确保通过准入的 Connection 真正进入 Cell 适应训练

修改：

`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/adaptation.py`

这是本轮最重要的执行门禁。

### 8.1 明确区分三种接受状态

当前 `accepted` 字段不足以清晰表达全部流程。新实验必须分别记录：

1. `trust_region_passed`
2. `cell_adaptation_executed`
3. `final_quality_accepted`

其含义如下：

| 状态 | 含义 |
|---|---|
| Trust Region 通过 | Connection 候选符合路由稳定性约束 |
| Cell 适应已执行 | 候选确实进入新分支并完成配置指定的 Cell 更新 |
| 最终质量接受 | 新分支在 Calibration 上取得满足门槛的改进，并被正式提交 |

A-T 通过 Trust Region 后，执行现有 Cell 的质量比较，不需要额外 Cell 适应。

A-C、B-C 通过 Trust Region 后，必须执行配置指定的完整双分支 Cell 适应流程。

不得在 Trust Region 通过后，因为缺少某个旧字段、错误的状态恢复或事务恢复逻辑而静默跳过适应训练。

### 8.2 使用配对随机数控制

当前 old/new 分支采用不同的 `route_rng` 种子。新实验应改为配对随机数设计。

要求：

- old/new 使用相同的初始 `route_rng` 状态。
- 每一步使用相同的训练数据。
- 两条分支使用相同的 Gumbel 随机数、epsilon 掩码及均匀探索随机数。
- 两条分支保留各自独立的模型参数和优化器状态。
- 允许因 Connection 不同而产生不同动作。
- 在每次更新前后检查配对随机流的状态是否一致。

如果两条分支的随机流消费数量发生不一致，必须报告：

`ADAPTATION_ROUTE_STREAMS_DIFFER`

并停止本次对照，不得继续将结果解释为纯粹的 Connection 效果。

### 8.3 保留配对适应的其他约束

每次双分支适应都必须验证：

- 初始 Cell 参数完全相同。
- 初始 Cell 优化器状态完全相同。
- 数据 RNG 状态相同。
- 数据顺序与数据游标一致。
- 每一步训练 Token 数相同。
- 两条分支执行相同数量的 Cell 更新。
- 旧分支和新分支分别完成 Calibration 评价。
- 被丢弃分支的全部额外训练成本被记录。

### 8.4 保留事务恢复能力

必须测试以下情况：

- 在提案生成后中断。
- 在第一个 Cell 适应步骤后中断。
- 在中间 Cell 适应步骤后中断。
- 在最终 Calibration 评价后中断。
- 从最新事务 Checkpoint 恢复。

恢复后不得重复执行已经完成的适应步骤，也不得漏记成本或提交不同的模型状态。

---

## 9. 第六阶段：训练成本与实验日志

修改 `training.py` 及相关结果序列化逻辑，保留现有成本核算体系。

至少记录以下成本：

- 正式 Cell 主训练更新与 Token。
- Connection 提案优化更新。
- Counterfactual 状态与后缀执行工作。
- Trust Region 各尺度的额外前向工作。
- Calibration 评价成本。
- Cell 适应分支的更新次数和 Token。
- 被丢弃分支的训练成本。
- 总训练时间和终端推理成本。

特别注意：

**相同的主训练 Token 数不意味着相同的总计算量。**

新旧方案比较必须分别报告主训练预算、提案成本、适应成本与总成本。

建议每次提案生成独立的 `TRUST_REGION_TRACE.jsonl`，并在 `ADAPTATION_RESULTS.json` 中增加明确的准入与提交字段。

---

## 10. 必须增加的自动化测试

### 10.1 路由一致性测试

- A 组零修正严格复现稳定 Top-2。
- B 组零修正严格复现稳定 Top-1。
- 同参数模型的 KL、硬动作变化和路径变化均为零。
- 零修正与极小非零修正不会仅因不同 tie-breaking 规则产生动作翻转。
- 所有选中动作均属于合法动作空间。

### 10.2 幅度控制测试

- 固定状态下，线性 Connection 的 Logit 插值与参数插值一致。
- `scale=0` 严格复现旧策略。
- 每个尺度均生成完整诊断指标。
- 不对动作变化率作未经验证的单调性假设。
- 数值空更新不能被接受。
- 非有限 Logit、KL 或梯度必须触发失败。

### 10.3 Trust Region 测试

- 三个准入条件分别生效。
- 任何一个门槛失败，候选都不得进入正式 Cell 适应。
- 所有尺度失败时，正确记录拒绝原因。
- 所有尺度均为空更新时，正确拒绝。
- 通过准入的候选保留对应的缩放参数和完整溯源信息。

### 10.4 Cell 适应测试

至少构造一个受控候选，使其：

1. 通过 Trust Region。
2. 成功创建新适应分支。
3. 完成配置指定数量的 Cell 更新。
4. 完成两条分支的 Calibration 评价。
5. 正确执行最终接受或拒绝。
6. 正确记录被丢弃分支的成本。

该测试必须覆盖 A-C 与 B-C。

这里需要区分两种结果：通过 Trust Region 并成功执行适应，属于机制测试通过；最终质量是否改善，则属于模型效果测试。

### 10.5 Checkpoint 与可复现性测试

- 保存并恢复后，训练状态完全一致。
- 配对随机流可以复现。
- 已完成的适应步骤不会重复执行。
- 冻结参数指纹保持不变。
- Development 不参与提案优化或阈值选择。
- Confirmation 保持封存。

---

## 11. 执行顺序与停止条件

### 阶段 A：只读诊断

完成零修正检查、Margin 分布和尺度扫描。

**门禁：** 如果同参数模型无法复现完全相同的确定性路由，不得进入后续阶段。

### 阶段 B：实现路由和 Trust Region 修复

完成统一路由规则、Margin 稳定机制、实际 Logit 幅度验证与完整回溯日志。

运行所有单元测试，不启动完整训练。

**门禁：** 零修正一致性、尺度验证和 Trust Region 测试全部通过。

### 阶段 C：单 Seed 小规模试运行

只运行受控试验，检查 Connection 提案是否能够：

- 产生有效 Logit 修正。
- 至少有一个尺度通过 Trust Region。
- 使 A-C 或 B-C 真正进入 Cell 适应流程。
- 正确完成适应与成本记录。

如果仍然没有候选通过，不得自动放宽 KL、硬动作或路径门槛。应检查 Margin 分布、修正幅度和候选方向，报告失败原因后再决定下一步。

### 阶段 D：冻结协议并执行正式实验

在单 Seed 门禁通过后，冻结新配置、代码版本、数据清单和阈值。

之后才可以运行完整 Seed 集合，并在所有选择步骤结束后执行 Development 终端评价。

原始 R3e 的结果继续保留；Confirmation 不得解封。

---

## 12. 最终验收标准

只有同时满足以下条件，才能宣布本次工程修复完成。

1. 原始 R3e 结果未被覆盖。
2. 零修正路由一致性测试通过。
3. Connection 的 Logit 更新幅度经过实际验证。
4. Trust Region 三个门槛均被保留且独立生效。
5. 空更新不能被误判为成功。
6. 至少一个受控候选通过 Trust Region。
7. A-C 和 B-C 的受控候选均能够实际进入 Cell 适应流程。
8. 配对适应使用一致的数据顺序、Cell 初始状态、优化器状态和路由随机流。
9. Checkpoint 恢复、成本核算和数据隔离测试通过。
10. 所有训练、准入、适应和评价结果均可追溯到代码版本、配置哈希、Seed 与 Checkpoint。

**机制修复完成不等于科学假设获得支持。**

即使新实验成功接受 Connection 更新，如果 Development NLL 没有改善，也必须如实报告。三个 Seed 仍然属于探索性证据，不能据此作出强确认结论。

---

## 13. 最终交付物

Codex 完成后必须提供：

1. 修改文件列表及每项修改的原因。
2. 新实验 Source commit 和 Config hash。
3. `ROUTING_DIAGNOSTICS.json`。
4. `TRUST_REGION_TRACE.jsonl`。
5. 更新后的 `ADAPTATION_RESULTS.json`。
6. 更新后的成本核算结果。
7. 所有自动化测试与 CI 结果。
8. 单 Seed 试运行结果。
9. 每个提案的准入率、实际适应执行次数、最终接受率。
10. 与 R3e 原始结果的明确对照。
11. 尚未解决的问题及是否允许进入正式多 Seed 训练的结论。

执行时以这份规格为约束。不得只修改阈值、只修复一个分支，或者在没有实际 Cell 适应记录的情况下宣称联合训练机制已经得到验证。
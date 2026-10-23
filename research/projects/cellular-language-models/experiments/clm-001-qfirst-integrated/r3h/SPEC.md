# MiniCells R3h — Train–Inference Routing Alignment

**训练—推理路由对齐与 Cell 学习组织实验**

**实验 ID：** `CLM-001-R3h-Routing-Alignment-001`  
**版本：** v1.0  
**仓库：** `ArcheLabs/mini-cells`  
**研究性质：** 探索性对照训练  
**运行环境：** Kaggle T4 × 2  
**执行方式：** 完成工程验证后一次启动，自动执行全部实验  
**Confirmation：** `SEALED`

---

## 一、研究背景

### 1.1 R3g 的核心发现

R3g 已完成只读的 Cell–Connection 协同适应归因。

历史实验：

```text
CLM-001-R3g-CoAdaptation-Attribution-001

Source:
753ec7a61b32b2b0ff18a05744fb0087fd4916c1

Results:
results/r3g-d865425e15395198-f87ad21a4104
```

主要结果：

**A-C：**

- Connection 替换直接带来约 0.001443 的 Calibration NLL 改善。
- 新 Cell 带来的额外改善仅约 0.000006。
- Cell 参数出现了可测量的变化。
- 有利交互项约为 -0.00000383。

**B-C：**

- Connection 替换带来约 0.000744 的 Calibration NLL 改善。
- 新旧 Cell 参数逐位相同。
- 没有检测到独立的 Cell 适应收益。

更重要的是，R3g 发现：

| 指标 | A-C | B-C |
|---|---:|---:|
| 训练策略归一化熵 | 约 1 | 约 1 |
| 训练采样匹配硬路由概率 | 6.67% | 16.68% |
| 均匀随机理论概率 | 1/15 | 1/6 |

这说明目前训练期间的 Cell 选择接近均匀随机，而推理使用确定性路由。

### 1.2 当前机制

训练时：

\[
\pi_F(a|s)
=
(1-\epsilon)\operatorname{Softmax}(q(s))_a
+
\epsilon U(a)
\]

推理时：

\[
a^*(s)=\operatorname{HardRoute}_{\tau}(q(s))
\]

其中：

- \(q(s)\) 是 Connection 给出的动作 Logit。
- \(\tau\) 是稳定硬路由的切换 Margin。
- \(U\) 是均匀动作分布。
- \(\epsilon\) 是训练探索率。

目前 \(q\) 的分数差异很小，因此温度为 1 的 Softmax 接近均匀分布。

这导致 Cell 主要接受随机路径上的梯度，而不是推理组织所决定的路径上的梯度。

### 1.3 本次研究目标

本轮研究三个问题：

1. 提高训练路由与推理路由的一致性，能否降低 Development NLL？
2. 不同程度的路由集中化，对 Cell 梯度覆盖、训练稳定性和模型泛化有什么影响？
3. 硬路由与随机探索之间，是否存在比完全确定或近乎均匀采样更好的折中？

本次首先冻结 Connection。

只有明确这些问题后，才进入下一轮 Cell–Connection 联合优化。

---

## 二、研究假设

### H1：Hard Routing Alignment

使用与确定性推理一致的稳定硬路由进行训练，在保留少量均匀探索的情况下，能够改善模型质量。

比较：

\[
H \quad \text{vs.} \quad F
\]

分别在 A、B 两种架构上检验。

### H2：Mixture Routing Alignment

将稳定硬路由与随机采样结合，可以在提高路径一致性的同时保留 Cell 探索和梯度覆盖，并可能优于原始高熵采样。

比较：

\[
M \quad \text{vs.} \quad F
\]

### H3：Temperature Calibration

降低采样温度可以提高训练路由的集中程度，但不一定能与带 Margin 的稳定硬路由完全一致。

本轮检验其：

- 实际采样集中程度。
- 硬路由动作匹配概率。
- Cell 覆盖情况。
- 终端 NLL。

该假设是独立的机制分析，不预设低温采样一定优于稳定硬路由。

### H4：Routing Alignment and Cell Learning

提高训练—推理路径一致性，能够改变 Cell 的实际训练分配，并可能改善推理组织下的模型质量。

本轮通过路径一致性、Cell 梯度覆盖、训练损失和终端质量共同分析。

不得仅凭路由熵降低就宣布 H4 得到支持。

---

# 三、实验架构

本轮保留 A、B 两种原始结构。

## 3.1 A：四阶段 Top-2 Cell

保持 R3e/R3f 的架构。

- Cell 数量：6。
- 执行阶段：4。
- 每阶段选择两个 Cell。
- 每 Token 总 Cell 调用次数：8。
- 合法动作：15 个 Cell Pair。
- 原始 Gate 冻结。
- Connection 参数冻结。
- 只有 Cell 参数参与梯度训练。

计算路径保持：

\[
h_{t+1}
=
h_t
+
0.125
\sum_{i\in S_t}
w_{t,i}C_i(\operatorname{Norm}(h_t))
\]

不得修改原始 Pair 内混合权重语义。

## 3.2 B：八步顺序 Cell 图

保持原始 R3e Graph 结构。

- Cell 数量：6。
- 顺序执行：8 步。
- 每步选择一个 Cell。
- 每 Token 总 Cell 调用次数：8。
- 合法动作：6。
- 允许重复执行同一 Cell。
- START 和前驱 Cell 语义保持不变。
- 原始 Gate 冻结。
- Connection 参数冻结。

计算路径：

\[
h_{k+1}
=
h_k+
0.0625C_{a_k}(\operatorname{Norm}(h_k))
\]

不允许修改：

- 图节点数量。
- 合法边集合。
- 自连接许可。
- Graph Alpha。
- Cell 参数规模。

A 与 B 属于不同的计算架构。

**不能直接将 A 与 B 的 NLL 差异解释为训练路由策略的效果。**

主要比较必须分别在各自架构内部完成。

---

# 四、四种训练路由方式

每种架构均执行以下四个 Arm。

| Arm | 名称 | 核心机制 |
|---|---|---|
| F | Original Softmax | 原始 Gumbel-Max + ε |
| H | Stable Hard Routing | 稳定硬路由 + ε |
| T | Temperature-Calibrated | 经温度校准的 Softmax + ε |
| M | Hard–Soft Mixture | 稳定硬路由与 Softmax 混合 + ε |

共八个逻辑 Arm：

```text
A-F
A-H
A-T
A-M

B-F
B-H
B-T
B-M
```

这里的 F/H/T/M 仅表示训练路由方式，不代表重新引入 R3e 的 Connection 更新。

## 4.1 F：原始 Softmax 基线

保持 R3e 原有训练机制：

\[
\boxed{
\pi_F(a|s)
=
(1-\epsilon)
\operatorname{Softmax}(q(s))
+
\epsilon U
}
\]

采样使用原有 Gumbel-Max。

Softmax 温度：

\[
T_F=1
\]

该组是本次的核心基线。

其作用是检验：仅改变 Cell 的训练路由方式，相对于当前原始训练机制究竟有什么收益。

## 4.2 H：稳定硬路由

定义：

\[
a^*(s)=\operatorname{HardRoute}_{\tau}(q(s))
\]

训练分布：

\[
\boxed{
\pi_H(a|s)
=
(1-\epsilon)\mathbf1[a=a^*(s)]
+
\epsilon U(a)
}
\]

即：

- 大部分时间执行稳定硬路由。
- 以 ε 概率执行均匀探索。
- 硬路由动作必须与推理时完全相同。

本组不使用普通 `argmax(q)` 替代稳定硬路由。

所有动作应当经过 R3f 已修复的 Baseline-Preserving Margin 规则。

**H 的核心价值是：训练路径与推理组织使用同一套动作选择机制。**

## 4.3 T：温度校准的 Softmax

定义：

\[
\boxed{
\pi_T(a|s)
=
(1-\epsilon)
\operatorname{Softmax}\left(\frac{q(s)}{T}\right)
+
\epsilon U
}
\]

其中：

\[
0<T\le1
\]

温度不能随意指定一个未经测量的数值。

本轮根据初始模型、固定 Admission 输入上的动作分布，执行一次不使用目标标签的温度校准。

目标：

\[
\mathbb E_s[
\operatorname{Softmax}(q(s)/T)_{a^*(s)}
]
\ge0.80
\]

使用事先固定的候选温度：

\[
T\in
\{2^{-k}\mid k=0,\ldots,20\}
\]

按照从高温到低温的顺序选择第一个满足目标的温度。

如果全部温度都无法达到目标：

- 选择最小预注册温度作为诊断性 T Arm。
- 明确记录 `TEMPERATURE_ALIGNMENT_TARGET_NOT_REACHED`。
- 继续运行 T Arm，但不得宣称它实现了目标匹配率。
- 不得增加温度搜索范围或调整目标值。

必须特别检查：由于推理使用带 Margin 的稳定硬路由，原始分数的普通 Argmax 可能与稳定硬路由不同。

温度降低不能保证消除这类不一致。

温度在每个 Seed、每种架构下初始校准一次，之后保持不变。

不得随着训练过程重新优化温度。

## 4.4 M：Hard–Soft 混合路由

本组用于比较：硬路由训练是否可以保留一定随机分工，避免 Cell 过度集中在少数路径上。

定义：

\[
\boxed{
\pi_M(a|s)
=
(1-\epsilon)
\left[
\rho\mathbf1[a=a^*(s)]
+
(1-\rho)\operatorname{Softmax}(q(s))_a
\right]
+
\epsilon U
}
\]

冻结：

\[
\rho=0.75
\]

含义：

- 75% 的非探索路由来自稳定硬路由。
- 25% 的非探索路由来自温度 1 的 Softmax。
- 额外 ε 探索机制保持一致。

不得根据 Development 表现重新选择 \(\rho\)。

本轮只研究一个固定混合比例，不进行网格搜索。

---

# 五、统一的确定性推理

**这一点必须严格保证。**

虽然 F/H/T/M 的训练方式不同，但终端评价必须全部使用同一个确定性路由函数：

\[
\boxed{
a_{\mathrm{eval}}
=
\operatorname{HardRoute}_{\tau}(q(s))
}
\]

要求：

- 不执行 Gumbel 采样。
- ε=0。
- 不使用训练温度选择动作。
- 不使用 M 的混合采样。
- 使用相同的稳定 tie-breaking 和 Margin 规则。
- 保留原始 Gate 和 Connection 语义。

因此，四组模型的质量差异只能来自训练路由造成的 Cell 参数演化，而不是评价时选择了不同的推理算法。

在同一架构、同一 Seed 的训练开始前，四组确定性推理必须严格一致。

如果初始模型输出或推理路径不一致，立即停止该组实验。

---

# 六、初始化与冻结边界

## 6.1 训练起点

为了获得三个 Seed 的严格匹配对照，本轮统一从各 Seed 对应的 **R3c D Step 500 Checkpoint** 初始化。

Seed：

```text
151001301
151001302
151001303
```

原因：

- R3c D500 是 R3e 的共同历史起点。
- 三个 Seed 都有对应的既有父状态。
- 不依赖仅在一个 Seed 中完成的 R3f-v2 新 Connection。
- 不需要重新执行 R3f 的前 100 Step。
- 有利于隔离训练路由机制。

父 Checkpoint、Tokenizer、数据资产和哈希应直接复用 R3e 已验证的冻结输入声明。

必须逐项核对当前 HF Manifest 与内容寻址对象，不得从未经验证的最新模型版本推断父状态。

## 6.2 参数初始化

同一个 Seed、同一种架构中的 F/H/T/M：

- Cell 参数完全相同。
- Gate 参数完全相同。
- Connection 参数完全相同。
- Embedding 完全相同。
- ContextRead 完全相同。
- LM Head 完全相同。
- 其他冻结参数完全相同。
- Cell Optimizer 初始状态完全相同。

A 与 B 使用各自架构的统一初始化，但两种架构无需具有相同的中间计算路径。

## 6.3 Connection 状态

本轮使用原始冻结 Connection：

\[
\phi=\phi_0
\]

Connection 修正参数采用零初始化。

不进行：

- Connection Proposal。
- Trust Region。
- Calibration 质量接受。
- Connection Optimizer 更新。
- Cell–Connection 双分支竞争。

允许 Gate 因输入隐藏状态变化而选择不同动作，但 Gate 和 Connection 权重始终冻结。

本轮只训练 Cell 参数。

## 6.4 模型身份

所有训练任务必须记录：

```text
parent_source_sha
parent_checkpoint_sha256
initial_cell_fingerprint
initial_connection_fingerprint
initial_frozen_parameter_fingerprint
experiment_source_sha
experiment_config_sha256
seed
architecture
training_router
```

同架构四组的初始指纹必须一致。

---

# 七、训练预算

## 7.1 主实验

每组训练：

\[
500\text{ Cell Updates}
\]

每次更新：

\[
512\text{ Tokens}
\]

共：

\[
256000\text{ Tokens/模型}
\]

执行：

- 2 种架构。
- 4 种路由。
- 3 个 Seed。

共：

\[
2\times4\times3=24
\]

个独立模型。

总主训练预算：

\[
24\times500=12000
\]

次 Cell Optimizer 更新。

总主训练 Token：

\[
\boxed{6,144,000}
\]

这只是主训练 Token 预算，不包含路由监测、Calibration/Admission 检查和 Development 终端评价。

## 7.2 统一训练配置

继续使用现有训练参数：

| 参数 | 值 |
|---|---:|
| Cell learning rate | 0.0003 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Warmup | 50 Step |
| Microbatch | 1 |
| Effective batch | 8 |
| Sequence length | 64 |
| Tokens/Step | 512 |
| 主训练 Step | 500 |
| Precision | float32 |

训练 ε：

\[
\epsilon_0=0.15
\]

\[
\epsilon_{499}=0.02
\]

采用固定线性衰减。

所有 Arm 使用相同的 ε 调度。

不为 H/T/M 单独调整学习率、Warmup、Gradient Clip 或 Token 预算。

## 7.3 不增加独立 Pilot

本次已有多轮真实模型训练经验，因此不需要额外运行完整 Pilot。

仅需在正式执行前完成：

- 路由概率正确性测试。
- 配对随机数测试。
- 初始模型一致性检查。
- Checkpoint 与身份测试。
- 单个非更新前向的数值检查。

这些属于必要的工程门禁，不用于选择最佳训练策略。

正式运行后，一次完成全部 24 组模型。

不得因为某个 Arm 的早期 NLL 较低而提前停止其他 Arm。

---

# 八、配对随机数与数据公平性

## 8.1 完全相同的 Train 数据

同一个 Seed、同一种架构：

F/H/T/M 必须使用相同的：

- Train Token 序列。
- Batch 顺序。
- Microbatch 顺序。
- Token 数。
- 数据游标。
- 初始 Cell 权重。
- Optimizer 超参数。

不同路由可以改变隐藏状态、梯度和后续 Cell 参数。

这些差异属于实验效果。

## 8.2 统一随机数输入

当前 F 使用 Gumbel-Max。

H/T/M 必须使用统一的路由随机数协议，避免因为不同 Arm 消耗不同数量的随机数而造成难以归因的差异。

建议定义固定的随机流：

```text
gumbel_uniforms
epsilon_uniforms
exploration_actions
mixture_uniforms
```

每个随机变量由以下信息确定：

```text
seed
main_step
microbatch_index
token_position
routing_stage
stream_identifier
```

无论某个 Arm 是否需要使用某种随机变量，都不得改变其他随机流。

对于 F：

- 使用原始 Gumbel-Max 机制。
- 使用统一 ε 探索掩码。

对于 H：

- 非探索部分直接选择稳定硬路由。
- 探索部分使用统一的随机动作。

对于 T：

- 使用同一套 Gumbel 随机数对温度校准后的 Logit 采样。

对于 M：

- 使用独立的混合选择随机数决定硬路由还是 Softmax。
- Softmax 部分使用同一套 Gumbel 随机数。
- ε 探索使用统一掩码。

必须保证所有随机抽样都遵循各自定义的分布。

不应为了共享随机数而改变 F 的边际采样分布。

## 8.3 配对随机性不等于强制动作相同

F/H/T/M 允许产生不同动作。

即使接收相同随机变量，由于路由策略不同，实际 Cell 选择也可能不同。

这正是本次需要研究的差异。

只要求外生随机条件一致，不要求最终动作一致。

---

# 九、训练—推理一致性指标

本次不仅评价 NLL，还必须直接测量路由机制是否按照设计运行。

## 9.1 训练策略熵

每个 Arm 计算：

\[
H(\pi)
=
-\sum_a \pi(a|s)\log\pi(a|s)
\]

归一化：

\[
\boxed{
H_{\mathrm{norm}}
=
\frac{H(\pi)}{\log|\mathcal A|}
}
\]

分别记录：

- 均值。
- 中位数。
- 分位数。
- 不同 Stage 的分布。
- 训练早期和晚期的变化。

H 组由于 ε 探索，归一化熵不等于严格零。

M 应当具有比 F 更集中的动作分布。

T 必须报告实际集中程度，而不是假定低温一定达成对齐。

## 9.2 确定性路由动作匹配概率

在固定状态上计算：

\[
a^*(s)=\operatorname{HardRoute}_{\tau}(q(s))
\]

再计算：

\[
\boxed{
P_{\mathrm{match}}
=
\mathbb E_s[
\pi_{\mathrm{train}}(a^*(s)|s)
]
}
\]

同时报告各 Stage 的分布。

该指标明确衡量训练采样选择推理动作的概率。

## 9.3 实际动作匹配率

训练期间，对当前隐藏状态记录：

- 训练实际选中的动作。
- 同状态下稳定硬路由会选择的动作。
- 是否一致。

计算：

\[
R_{\mathrm{match}}
=
\frac{\text{相同动作数}}{\text{全部动作数}}
\]

注意：该指标是同一状态上的局部匹配，而不是训练随机路径与完整确定性路径的闭环比较。

两个定义必须在报告中明确区分。

## 9.4 完整路径匹配概率

在固定模型和输入上，先执行一次确定性硬路由，得到：

\[
a_1^*,\ldots,a_K^*
\]

沿这条确定性路径记录每一步的条件状态 \(s_k^*\)。

计算：

\[
\boxed{
P_{\mathrm{hard\ path}}(x)
=
\prod_{k=1}^{K}
\pi_{\mathrm{train}}(a_k^*|s_k^*)
}
\]

这表示在该固定输入上，随机训练路由完整选择这条确定性路径的概率。

必须使用每一步在确定性前缀下的实际状态。

不得直接将各阶段的平均匹配概率相乘。

如果现有实现的路由采样存在跨阶段或跨 Token 的随机相关性，需要采用与实际随机协议一致的联合概率计算，不能错误使用独立性假设。

同时分别报告局部匹配率和完整路径匹配概率。

---

# 十、Cell 梯度与训练覆盖

## 10.1 Cell 使用频率

记录：

- 六个 Cell 的总调用次数。
- 每个 Stage 的调用分布。
- 各 Cell 接受梯度的更新次数。
- 各 Cell 的 Gradient Norm。
- 未被使用的 Cell 数量。
- 相邻 Step 的 Cell 调用变化。

A 组还应记录 Pair 分布。

B 组还应记录顺序转移矩阵：

\[
T_{ij}
=
P(a_{k+1}=j\mid a_k=i)
\]

包括自连接比例。

## 10.2 不主动引入 Balance Loss

本轮不能引入额外的路由平衡损失。

因为这会将研究问题从：

“训练路由对齐是否有效？”

变成：

“路由对齐加平衡正则化是否有效？”

这两者需要分别研究。

因此，Cell 覆盖不均衡首先作为结果记录。

仅当出现非有限数值、全部 Cell 梯度无效或训练机制不满足预定义正确性条件时，才标记运行无效。

某个 Cell 被较少使用不自动意味着运行错误。

## 10.3 允许出现负面结果

稳定硬路由可能导致：

- 一些 Cell 频繁被选中。
- 一些 Cell 使用次数较少。
- 梯度集中。
- 部分 Cell 学习不足。
- Development NLL 变差。

这些都是有价值的实验结果。

不得因结果不理想，在运行过程中增加 Balance Loss 或改变 ε。

---

# 十一、评估协议

## 11.1 数据边界

| 数据 | 本轮用途 |
|---|---|
| Train | Cell 训练 |
| Calibration | 初始 Margin/温度辅助校准；有限工程诊断 |
| Admission | 固定状态的路由一致性与覆盖分析 |
| Development | 全部训练完成后的终端模型质量评价 |
| Confirmation | SEALED |

温度校准不使用目标标签。

不得利用 Development NLL 选择温度、混合权重或训练策略。

## 11.2 主要质量指标

主要指标：

\[
\boxed{
\text{Development NLL}
}
\]

每个 Arm、Seed 在 Step 500 评价一次。

使用完全相同的：

- Development 数据。
- Tokenizer。
- 评价 Window。
- 确定性路由规则。
- 模型精度。
- Loss 计算方法。

## 11.3 次要指标

包括：

- Training CE 曲线。
- 路由归一化熵。
- 推理动作匹配概率。
- 完整路径匹配概率。
- Cell 调用分布。
- 梯度覆盖。
- Cell 参数变化范数。
- 推理 Tokens/s。
- 峰值显存。
- 训练总成本。

不得使用训练 CE 单独判定模型质量收益。

## 11.4 参考模型评价

在训练开始前，对每个 Seed/架构共同的初始模型执行一次确定性质量评价。

这用于检查：

- 四个训练 Arm 的起点一致。
- 原始模型确实可以复现。
- 训练过程相对于共同起点产生了什么变化。

该初始评价不用于选择实验策略。

---

# 十二、统计判定

## 12.1 主要比较

预注册四个主要比较：

```text
A-H vs A-F
A-M vs A-F

B-H vs B-F
B-M vs B-F
```

T 组与 F 的比较单独报告，但作为次要分析，不参与这四个主要假设的选择。

## 12.2 NLL 差异

定义：

\[
\Delta=
NLL_{\mathrm{candidate}}
-
NLL_F
\]

负值表示候选路由训练方式更好。

支持性探索证据需要同时满足：

1. 三个 Seed 的改善方向一致。
2. 三 Seed 平均改善至少 0.005 NLL。
3. 训练预算和数据身份有效。
4. 没有明显的协议违规。
5. 报告 Story 配对差异及不确定性。

其中：

\[
\overline\Delta\le-0.005
\]

属于本轮固定的实际效果门槛，不代表统计确认。

## 12.3 Bootstrap 和多重比较

继续使用既有 Story-cluster paired bootstrap。

建议：

```text
bootstrap_replicates = 2000
bootstrap_seed = 2026102301
multiplicity = HOLM
primary_comparisons = 4
```

需要明确：

- Story Bootstrap 不能替代独立训练 Seed。
- 只有三个 Seed。
- Development 曾用于历史研究。
- 即使达到阈值，也只能标记为探索性支持。

不能宣称已经得到独立确认。

---

# 十三、训练执行与恢复

## 13.1 独立模型状态

每个 Seed/Arm 保存：

```text
model
cell_optimizer
data_rng
route_rng
main_step
data_cursor
training_curve
routing_metrics
cell_counts
gradient_norms
cost
source_identity
config_identity
parent_identity
```

训练路由模式和温度参数也必须进入 Checkpoint。

恢复时不得通过当前默认配置重新推断此前使用的路由模式。

## 13.2 Checkpoint

建议保存：

```text
0
50
100
150
200
250
300
350
400
450
500
```

以及必要的独立事务记录。

沿用已有：

- 内容寻址对象。
- 私有 HF 存储。
- Manifest。
- Store Receipt。
- Checkpoint 身份检查。
- 中断恢复。

不重新设计存储系统。

## 13.3 Kaggle GPU

使用 T4 ×2。

两个 Worker 分别执行互相独立的模型训练任务。

统一协调器负责：

- HF 上传。
- Checkpoint Manifest。
- GitHub 发布。
- 最终结果汇总。

避免两个 GPU Worker 同时修改同一个远端存储指针。

## 13.4 恢复正确性

恢复后必须确保：

- Cell 权重一致。
- Cell Optimizer 状态一致。
- 数据游标一致。
- 外生随机数流一致。
- 已完成的训练更新不重复执行。
- 训练 Token 总数不重复计算。
- 路由模式和温度保持不变。

如已有恢复测试覆盖这些行为，直接复用，不需要在 Kaggle Notebook 中重新运行完整测试套件。

---

# 十四、计算成本

本轮总主训练预算：

```text
Architectures: 2
Routing modes: 4
Seeds: 3

Physical models: 24
Main updates/model: 500
Main updates total: 12000

Tokens/update: 512
Main training tokens: 6144000
```

必须单独记录：

- 主训练前向工作。
- 主训练反向工作。
- 路由采样开销。
- 额外路由监测工作。
- 终端 Development 评价。
- Checkpoint 上传时间。
- 总 Wall Time。
- GPU 显存峰值。

不得将相同 Token 数解释为相同训练 FLOPs。

四种路由策略可能具有不同的计算开销。

不得追加无关的大规模性能 Benchmark。

---

# 十五、自动化实现

建立独立研究目录：

```text
research/projects/
  cellular-language-models/
    experiments/
      clm-001-qfirst-integrated/
        r3h/
          SPEC.md
          README.md

          config/
            frozen.json

          implementation/
            minicells_r3h/
              parent_import.py
              routing.py
              sampler.py
              trainer.py
              alignment.py
              instrumentation.py
              evaluation.py
              statistics.py
              storage.py
              reports.py
              runner.py

          notebooks/
            kaggle-development.ipynb

          reports/
            INPUT_VERIFICATION.json
            ROUTING_CALIBRATION.json
            TRAINING_ALIGNMENT.json
            CELL_EXPOSURE.json
            DEVELOPMENT_NLL.json
            TRAINING_COST.json
            SCIENTIFIC-FINDINGS.md
            RUN_MANIFEST.json
```

优先复用 R3e/R3f 已有模型、训练器、Store、HF 和 GitHub 发布模块。

避免复制整个 Cell 模型实现。

新增代码应当主要围绕：

- 不同训练路由分布。
- 固定随机流。
- 路由一致性统计。
- 实验矩阵调度。
- 结果汇总。

## 15.1 统一执行入口

创建：

```bash
python scripts/run_r3h.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3h/config/frozen.json \
  --run-all
```

该入口目前需要 Codex 创建。

Notebook 负责：

1. 检出固定 Source SHA。
2. 确认环境和私有存储。
3. 运行统一入口。
4. 保留 Output。
5. 输出最终研究状态。

不在 Notebook 中加入多套重复的独立 Runner。

---

# 十六、自动运行流程

本轮应当采用一次启动、自动完成的方式。

### 阶段 A：输入与初始化验证

- 验证三个 Seed 的 R3c D500 父 Checkpoint。
- 验证 Tokenizer 和数据资产。
- 初始化八种架构/策略组合。
- 验证同架构四组起点一致。
- 验证 Connection 冻结。
- 完成训练采样器的概率测试。
- 冻结三个 Seed 的 T 温度和 Route Margin 记录。

如发现模型初始状态不一致，停止受影响的实验。

### 阶段 B：正式训练

完整运行：

```text
A-F, A-H, A-T, A-M
B-F, B-H, B-T, B-M
```

每组三个 Seed、500 Step。

不得根据训练期质量提前选择或淘汰 Arm。

训练期间定期保存 Checkpoint 和路由统计。

### 阶段 C：终端评价

所有有效训练完成后：

- 读取 Development。
- 执行统一确定性推理。
- 计算最终 NLL。
- 汇总 Cell 使用与梯度指标。
- 计算对齐程度和完整路径匹配概率。

### 阶段 D：统计与发布

生成：

- 三 Seed 模型质量表。
- 四项主要比较。
- T 组次要比较。
- 训练路由熵变化。
- Cell 覆盖结果。
- 成本对比。
- 科学结论。

最后发布到独立 Results 分支。

---

# 十七、结果文件

## 17.1 DEVELOPMENT_NLL.json

至少包含：

```text
seed
architecture
routing_mode
development_nll
baseline_development_nll
delta_from_F
parent_checkpoint_sha256
final_checkpoint_sha256
```

## 17.2 TRAINING_ALIGNMENT.json

至少包含：

```text
seed
architecture
routing_mode
initial_hard_match_probability
final_hard_match_probability
initial_normalized_entropy
final_normalized_entropy
empirical_local_action_match
hard_path_match_probability
temperature
epsilon_schedule
```

## 17.3 CELL_EXPOSURE.json

至少包含：

```text
cell_id
total_invocations
gradient_update_count
mean_gradient_norm
stage_usage
pair_usage
transition_usage
unused_or_low_exposure_flags
```

必须支持六个 Cell 的独立对比。

## 17.4 SCIENTIFIC-FINDINGS.md

报告必须回答：

1. H 是否明显提高训练—推理路由一致性？
2. M 是否提供更好的集中化与探索折中？
3. T 实际达到多大程度的路由集中化？
4. 哪些方案改善了 Development NLL？
5. Cell 梯度分配是否形成可观察差异？
6. 高对齐是否伴随 Cell 使用集中与潜在饥饿？
7. A/B 两种架构是否得到一致的研究结论？
8. 是否有理由进一步恢复 Connection 的联合更新？
9. 哪些结果仍只属于探索性证据？

---

# 十八、科学结论判定

每个主要质量假设使用：

- `SUPPORTED_EXPLORATORY`
- `NOT_SUPPORTED`
- `INCONCLUSIVE`
- `INVALID_RUN`

训练机制验证单独使用：

- `ROUTING_DISTRIBUTION_VERIFIED`
- `ALIGNMENT_INCREASED`
- `ALIGNMENT_TARGET_NOT_REACHED`
- `CELL_EXPOSURE_IMBALANCED`

不得将上述工程或描述性状态当作模型质量结论。

## 18.1 H 的结果

如果 H 明显提高路由对齐，但 NLL 变差：

说明单纯提高确定性路径训练比例不一定有利。

可能需要更多探索或不同的 Cell 分工机制。

## 18.2 M 的结果

如果 M 改善 NLL，并保留较均衡 Cell 覆盖：

说明硬路由与随机探索的结合可能比完全高熵或高度集中训练更有效。

下一阶段可以研究随训练进度变化的混合比例。

## 18.3 T 的结果

如果低温 T 能提高匹配率，但收益明显弱于 H：

说明问题不只是 Softmax 温度过高，还可能涉及普通 Argmax 与带 Margin 硬路由的差异。

如果 T 优于 H：

说明一定程度的软采样与梯度覆盖可能具有独立价值。

## 18.4 所有方法均未改善

如果 H/T/M 均不能改善 Development NLL：

不应直接认定训练—推理路由不一致无害。

应结合 Cell 梯度覆盖、训练稳定性和实际路径匹配分析原因。

尤其需要区分：

- 更好的路由对齐没有产生质量收益。
- 过度集中导致部分 Cell 训练不足。
- 原始 Gate 所定义的硬路径本身不够好。

这些属于不同的失败机制。

---

# 十九、下一阶段决策

R3h 结束后，不自动开始 R3i。

根据结果进入下一步。

### 情况 A：H 最好

优先采用稳定硬路由训练作为下一阶段联合优化的基线。

之后再引入 R3f-v2 已验证的局部 Connection 更新，观察 Cell 是否能围绕新组织真正适应。

### 情况 B：M 最好

优先采用混合路由。

下一阶段研究：

- 固定与变化的混合比例。
- Connection 更新前后的探索强度。
- Cell 分工逐步稳定的训练过程。

### 情况 C：T 最好

研究适合当前 Logit 尺度的温度机制，以及温度是否应随训练阶段变化。

但不在 R3h 中进行事后温度搜索。

### 情况 D：全部表现不佳

优先重新检查原始 Gate 的动作价值与硬路由组织能力，而不是立即扩大训练规模。

必要时研究显式的 Cell 路径价值估计、局部稀疏 Connection 更新，以及路由组织的功能意义。

---

# 二十、最终验收门禁

Codex 完成后必须确认：

- R3c D500 父模型恢复正确。
- 三 Seed 的初始状态有效。
- 同一 Seed/架构下四组 Cell 初始参数一致。
- Connection 始终冻结。
- 四种路由分布符合定义。
- 统一随机数协议有效。
- 全部有效模型完成 500 主更新。
- 每模型训练 Token 数严格一致。
- H/T/M 的实际路由分布得到验证。
- 所有评价使用相同稳定硬路由。
- Development 只在终端评价使用。
- Confirmation 保持 SEALED。
- 模型和训练过程可恢复。
- 主训练及额外计算成本完整记录。
- GitHub 结果回传有远端 SHA 核验。
- 科学负结果不会被覆盖或触发追加训练。

如果出现真实存储中断，可以从最近的已验证 Checkpoint 恢复。

不得因恢复而重复执行已经完成的训练更新。

---

# 二十一、给 Codex 的执行指令

请在 `ArcheLabs/mini-cells` 中实施：

`CLM-001-R3h-Routing-Alignment-001`

本次研究目标是验证：**让 Cell 训练时执行的计算路径，更加接近 Connection 在确定性推理时选择的路径，是否能够改善语言模型质量。**

实施要求：

1. 从 R3c D500 的三个已验证 Seed 初始化。
2. 保持 Cell 结构、Gate、Connection 和推理算法不变。
3. Connection 完全冻结，只训练 Cell。
4. 每种架构分别实现 F/H/T/M 四种训练路由。
5. F 复现原始 Gumbel-Max + ε。
6. H 使用与推理完全一致的稳定硬路由 + ε。
7. T 使用一次性温度校准的 Softmax + ε。
8. M 使用固定 75% 硬路由和 25% 原始 Softmax 的混合采样 + ε。
9. 所有组使用匹配的训练数据、预算和外生随机流。
10. 完整执行三个 Seed、每组 500 Cell Step。
11. 保存 Cell 梯度覆盖、路由熵、动作匹配率与完整路径匹配概率。
12. 使用统一的确定性路由评价 Development NLL。
13. 完成预注册的对照比较。
14. 不使用 Development 调参。
15. 不读取 Confirmation。
16. 保留完整 Checkpoint、运行身份与结果。
17. 只向 GitHub 公开聚合指标。
18. 不自动进入下一阶段联合训练。

所有有效实验无论结果正负都必须完成。

最终需要明确回答：

**MiniCells 当前接近均匀随机的训练路由，是否正在限制 Cell 对确定性计算组织的学习？**

以及：

**稳定硬路由、低温采样、硬软混合采样，哪一种更有利于 Cell 的持续训练与模型质量？**

只有回答这些问题，下一阶段才能合理地重新引入 Connection 更新，进一步研究真正稳定的 Cell–Connection 共同适应。

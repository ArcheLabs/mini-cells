# MiniCells R3i — Stable Cell–Connection Co-Adaptation

**稳定路由下的 Cell–Connection 联合训练与持续组织更新实验**

**版本：** v1.0  
**实验 ID：** `CLM-001-R3i-Stable-CoAdaptation-001`  
**仓库：** `ArcheLabs/mini-cells`  
**研究性质：** 探索性多 Seed 联合训练  
**计算环境：** Kaggle T4 × 2  
**执行方式：** 工程门禁通过后，一次启动、自动执行与恢复  
**Confirmation：** `SEALED`  
**科学状态：** `EXPLORATORY_DEVELOPMENT`

---

# 1. 研究背景

## 1.1 已完成研究

### R3e：Stable Co-Adaptive Learning

R3e 初次尝试让 Connection 和 Cell 共同训练。

结果：

- 36 次 Connection 提案全部被拒绝。
- 新 Connection 未进入正式 Cell 训练。
- A-C 与 A-F 的 NLL 完全相同。
- B-C 与 B-F 的 NLL 完全相同。
- 不能据此否定联合训练机制。

### R3f-v2：Local Trust-Region Connection Update

R3f-v2 解决了 Connection 更新准入问题。

在单 Seed、Step 100 的局部实验中：

- A-C、B-C 均成功通过 Trust Region。
- 两组均在尺度 1/4096 下产生有效的硬路径变化。
- 两组均完成 K=10 的 Cell 配对适应。
- A-C 满足局部 Calibration 质量接受条件。
- B-C 获得小幅质量改善，但未达到接受阈值。

这证明稳定、非空的 Connection 更新在工程上可行。

### R3g：Co-Adaptation Attribution

R3g 进一步区分了 Connection 与 Cell 的收益。

结果：

- A-C 的局部改善绝大部分来自 Connection 的直接推理路径选择。
- Cell 额外适应收益非常小。
- B-C 的旧、新 Cell 参数逐位相同。
- 原有训练采样接近均匀分布，与确定性推理路径明显不匹配。

### R3h：Train–Inference Routing Alignment

R3h 冻结 Connection，只改变 Cell 训练路由。

三个 Seed、两种架构、四种路由方式全部完成。

平均 Development NLL：

| 训练方式 | A：Top-2 | B：顺序图 |
|---|---:|---:|
| F：原始 Softmax | 4.539648 | 4.561129 |
| H：稳定硬路由 | **4.512753** | **4.519418** |
| T：校准低温 | 4.513944 | 4.520858 |
| M：硬软混合 | 4.517296 | 4.525682 |

H 相对于 F：

\[
\Delta_A=-0.026895
\]

\[
\Delta_B=-0.041711
\]

三个 Seed 均保持相同改善方向。

R3h 因此提供了目前最明确的探索性证据：让 Cell 训练路由更接近确定性推理路由，可以改善模型质量。

## 1.2 尚未解决的问题

R3h 中的 Connection 始终冻结。

因此，我们仍然不知道：

**当 Cell 已经沿稳定计算路径学习时，允许 Connection 根据任务反馈进行局部更新，能否进一步改善模型质量？**

还需要区分：

- Connection 更新自身改善了推理路径。
- Cell 在新 Connection 下学出了更有利的参数。
- 两者形成了持续的协同适应。
- 更新带来的额外计算成本是否合理。
- 多次 Connection 更新是否导致 Cell 使用过度集中。

这是 R3i 的研究范围。

---

# 2. 核心研究目标

R3i 需要回答五个问题。

**Q1：质量收益**

稳定 Connection 联合训练能否超过同样路由模式下的冻结 Connection 基线？

**Q2：实际适应**

新的 Connection 被接受后，是否真正改变后续 Cell 的训练路径和梯度分配？

**Q3：持续性**

在 500 次主 Cell 更新中，多个 Connection 更新能否持续产生收益，而不只是一次局部改善？

**Q4：训练路由依赖**

联合训练收益是否依赖稳定硬路由 H，还是同样能够出现在低温路由 T 中？

**Q5：组织稳定性**

Connection 持续变化是否造成 Cell 使用集中、训练不稳定或部分 Cell 梯度不足？

---

# 3. 实验假设

## H1：A 架构硬路由联合训练收益

\[
NLL(A\text{-}CH)<NLL(A\text{-}FH)
\]

其中：

- FH：Frozen Connection + Hard Routing。
- CH：Co-Adaptive Connection + Hard Routing。

## H2：B 架构硬路由联合训练收益

\[
NLL(B\text{-}CH)<NLL(B\text{-}FH)
\]

## H3：A 架构低温联合训练收益

\[
NLL(A\text{-}CT)<NLL(A\text{-}FT)
\]

## H4：B 架构低温联合训练收益

\[
NLL(B\text{-}CT)<NLL(B\text{-}FT)
\]

四个质量假设采用统一的探索性判定：

1. 三 Seed 改善方向一致。
2. 平均 NLL 改善至少 0.005。
3. 数据、模型和训练预算有效。
4. 结果来自统一确定性推理。
5. 不读取 Confirmation。

同时设置独立的机制问题：

**M1：** 是否存在通过 Trust Region 的非空 Connection 更新？

**M2：** 通过准入的更新是否进入 K=10 配对 Cell 适应？

**M3：** 接受后的 Connection 是否在后续实际训练中改变 Cell 执行与梯度分配？

**M4：** 是否观察到超出纯路由效果的正向 Cell 适应收益？

这些机制指标不能代替质量假设。

尤其不能仅凭 M1/M2 成立就宣布实现了有益的共同学习。

---

# 4. 实验矩阵

## 4.1 两种架构

继续使用既有架构。

### A：Top-2

- 6 个 Cell。
- 4 个执行阶段。
- 每阶段两个 Cell。
- 15 个合法 Cell Pair。
- 每 Token 共 8 次 Cell 调用。
- 原始 Gate 冻结。

### B：顺序 Cell 图

- 6 个 Cell。
- 8 步串行执行。
- 每步选择一个 Cell。
- 允许自连接。
- 保留前驱 Cell 语义。
- Graph Alpha = 0.0625。
- 原始 Gate 冻结。

两种架构内部比较，不直接跨架构进行质量假设判定。

## 4.2 四类实验组

| Arm | Cell 训练路由 | Connection |
|---|---|---|
| FH | 稳定硬路由 H | 冻结 |
| CH | 稳定硬路由 H | 联合更新 |
| FT | 校准低温 T | 冻结 |
| CT | 校准低温 T | 联合更新 |

对应八个逻辑 Arm：

```text
A-FH
A-CH
A-FT
A-CT

B-FH
B-CH
B-FT
B-CT
```

每个 Arm 使用三个 Seed：

```text
151001301
151001302
151001303
```

总计 24 个逻辑模型。

其中：

- 12 个 Frozen 模型直接复用 R3h 的历史结果。
- 12 个 Co-Adaptive 模型执行新的正式训练。

不得混入 R3h 的 F 或 M 路由作为本轮主要比较。

F、M 的历史结果保留为背景参考。

---

# 5. 历史基线与模型初始化

## 5.1 冻结基线

R3i 必须复用已经完成的 R3h 结果：

```text
Experiment:
CLM-001-R3h-Routing-Alignment-001

Source:
01f226e259356bb06a8ef275e8314649062446e4

Config SHA-256:
f0090b76382e0b29042dc7a5a017927377c06413cc27dbb01d24b378ab338d65

Run ID:
7cb36823c6e1a705bc5200a108e3ab160ec2df0f4cb6639e512696d574706ba7
```

R3h 的历史 H/T 结果只读。

不得覆盖、续写或修改原始 Checkpoint。

冻结基线映射：

| R3i | R3h |
|---|---|
| A-FH | A-H |
| A-FT | A-T |
| B-FH | B-H |
| B-FT | B-T |

### 5.2 必须验证历史 Checkpoint

读取 R3h 的固定私有 HF Manifest 与内容寻址对象。

验证：

- Source SHA。
- Config SHA。
- Run ID。
- 训练 Seed。
- Arm。
- Step。
- Model Fingerprint。
- Cell Optimizer。
- Data Cursor。
- 训练路由模式。
- 温度。
- R3h 随机流协议。

优先使用 R3h 的历史固定 HF Revision 和 Manifest SHA，不使用未经验证的 `latest` 内容替代原始实验记录。

对每个 Frozen Arm，读取历史 Step 500 NLL，作为本次比较基线。

## 5.3 新联合训练的起点

CH/CT 从各自 Seed 的原始 R3c D500 父状态开始。

同一 Seed 和架构：

\[
\theta_0^{CH}=\theta_0^{FH}
\]

\[
\theta_0^{CT}=\theta_0^{FT}
\]

\[
\phi_0^{CH}=\phi_0^{FH}
\]

\[
\phi_0^{CT}=\phi_0^{FT}
\]

初始 Cell、Connection、冻结模型参数及 Cell Optimizer 必须一致。

不是从 R3h Step 500 继续增加 500 Step。

**新训练必须从 R3h 对照的相同起点重新执行 500 Step，才能衡量 Connection 在这一训练区间内持续更新的收益。**

## 5.4 R3h 训练兼容性门禁

因为 R3i 会扩展训练器，必须首先确认新训练器在尚未执行 Connection Proposal 时，能够复现原始 R3h 的行为。

每个 CH/CT 任务在到达 Step 100、执行第一次 Proposal 之前：

- Model 权重应与对应 R3h Frozen 模型的 Step 100 一致。
- Cell Optimizer 状态一致。
- Data Cursor 一致。
- 训练批次身份一致。
- 路由动作一致。
- Temperature、Margin 与路由随机流定义一致。

优先要求逐位一致。

如果实际设备存在非确定性行为，必须报告偏差来源，并通过额外确定性检查验证其不影响可比性；不得静默放宽检查。

如果无法证明 R3i 与历史 R3h 基线兼容：

`HISTORICAL_CONTROL_NOT_COMPARABLE`

停止正式科学比较。

不能直接使用不等价的历史 NLL 宣称联合训练收益。

不得未经新实验授权，自动改为一套不同训练配置的对照。

---

# 6. 统一的 Cell 训练路由

R3i 沿用 R3h 的路由实现。

## 6.1 H：稳定硬路由

设：

\[
a^*(s)
=
\operatorname{HardRoute}_{\tau}(q(s))
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

保留：

- Stable Baseline Action。
- Route Switch Margin。
- 原始稳定 tie-breaking。
- Epsilon Uniform Exploration。

不得改用未经 Margin 处理的 `argmax(q)`。

## 6.2 T：校准低温路由

训练分布：

\[
\boxed{
\pi_T(a|s)
=
(1-\epsilon)
\operatorname{Softmax}(q(s)/T)
+
\epsilon U(a)
}
\]

每个 Seed 和架构的 T：

**直接继承 R3h 已冻结的 Temperature。**

不得重新搜索。

不得因为 Connection 更新后 Logit 尺度变化，重新选择 T。

必须同时记录：

- 当前实际 Softmax 熵。
- 当前稳定硬动作匹配概率。
- 当前原始 Logit 幅度。
- 温度缩放后的分布变化。

Connection 更新可能改变 T 的有效集中程度，这属于实验效应。

## 6.3 Epsilon 调度

所有模型保持：

\[
\epsilon_0=0.15
\]

\[
\epsilon_{499}=0.02
\]

使用 R3h 原有线性调度。

不能因 Connection 更新而自动重置 Epsilon 或主训练 Step。

## 6.4 随机流

保留 R3h：

`KEYED_CPU_TORCH_STREAMS_V1`

统一键：

```text
seed
main_step
microbatch
routing_stage
stream_identifier
```

保留：

- `gumbel_uniforms`
- `epsilon_uniforms`
- `exploration_actions`
- `mixture_uniforms`

CH/CT 使用与对应 Frozen 模型一致的外生随机流定义。

Connection Proposal 的随机状态必须独立，不能意外改变后续正式 Cell 训练的数据或路由随机流。

---

# 7. Connection 更新机制

本节直接继承 R3f-v2 已验证的局部更新机制。

## 7.1 提案时刻

每个 Co-Adaptive 模型在：

\[
\boxed{
100,\ 200,\ 300,\ 400
}
\]

四个主训练 Step 执行 Connection Proposal。

每个时刻最多一个提案。

若提案失败，不增加额外候选搜索。

若某次提案被拒绝，继续正式 Cell 训练，后续按原定时刻生成下一次提案。

## 7.2 Connection Proposal

每次使用：

- 12 个 Train Counterfactual States。
- A：15 个候选 Pair。
- B：6 个候选 Cell。
- 原始条件反事实终端 CE。
- 原始硬路由重规划后缀。
- 8 次 Connection AdamW 优化。
- Connection LR = 0.0003。
- Weight Decay = 0.1。
- Gradient Clip = 1.0。

保持 R3f-v2 的中心化策略优化目标。

不在本轮修改成新的 Policy Gradient、REINFORCE 或路径价值网络。

Connection Proposal 的目标仍是改善当前硬推理组织的条件价值。

T 组虽使用低温随机训练，Connection 提案的反事实价值后缀仍按共同的确定性推理定义计算。

这一差异必须在科学报告中注明，不能将其描述为 T 随机策略下的精确期望价值。

## 7.3 先保存、后扫描

每次 Proposal 优化完成后，先保存：

- 正式模型快照。
- 原始 Connection。
- 完整未缩放候选参数。
- 参数差分。
- Proposal Optimizer 配置。
- Counterfactual State Identity。
- Source/Config/Checkpoint Hash。
- RNG 状态。
- 计算成本。

必须成功保存并回读，才允许开始回溯尺度扫描。

任何被拒绝的 Proposal 不得丢失完整候选参数。

---

# 8. Trust Region

## 8.1 保留原始约束

\[
\boxed{
D_{\mathrm{KL}}\le0.01
}
\]

\[
\boxed{
D_{\mathrm{action}}\le0.10
}
\]

\[
\boxed{
D_{\mathrm{path}}\le0.20
}
\]

三个指标独立生效。

继续使用：

- 同状态动作变化率。
- 确定性完整路径变化率。
- 原始分类分布 KL。

参考输入来自 Admission。

Calibration 不参与 Trust Region 阈值选择。

Development 不参与任何准入决定。

## 8.2 缩放尺度

使用 R3f-v2 的尺度集合：

\[
\boxed{
\alpha\in\{0\}\cup\{2^{-k}\mid k=0,\ldots,20\}
}
\]

共 22 档。

Connection 参数：

\[
\phi_\alpha
=
\phi_{\mathrm{old}}
+
\alpha(\phi_{\mathrm{proposal}}-\phi_{\mathrm{old}})
\]

扫描从最大非零尺度开始。

选择第一个同时满足全部约束、且具有真实硬路径变化的尺度。

\[
N_{\mathrm{changed\ paths}}>0
\]

数值空更新不得接受。

不得根据 Calibration NLL 选择最好的尺度。

## 8.3 实际训练策略变化

这里增加一个**独立监测指标**，但不改变 R3f-v2 的三个准入门槛。

对 H/T 分别计算：

\[
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{train}}^{old},
\pi_{\mathrm{train}}^{new}
\right)
\]

H：

使用当前 Epsilon 下的稳定硬路由混合策略。

T：

使用当前冻结 Temperature 与 Epsilon 的低温策略。

记录：

- Raw Softmax KL。
- Actual Training Policy KL。
- 确定性动作变化。
- 完整路径变化。
- 训练策略熵变化。
- 确定性硬路由动作匹配概率变化。

**本轮实际训练策略 KL 是诊断指标，不额外设置 0.01 的准入门槛。**

原因是 H 的策略包含离散硬动作切换，原始分布 KL 与实际训练混合策略 KL 不具有相同数值语义。

如果直接追加相同阈值，会改变 R3f-v2 的既有算法，需要另立实验验证。

但若出现非有限值、概率归一化错误或非法动作，必须停止本次 Proposal。

---

# 9. Cell 配对适应

只对通过 Trust Region 的 Connection 候选执行配对 Cell 适应。

## 9.1 两条分支

从同一个正式状态：

\[
(\theta_0,\phi_0)
\]

产生：

**Old：**

\[
(\theta_O,\phi_0)
\]

**New：**

\[
(\theta_N,\phi_\alpha)
\]

分别执行：

\[
K=10
\]

次 Cell 更新。

## 9.2 必须保持配对条件

两条分支具有相同的：

- 初始 Cell 参数。
- 初始 Cell Optimizer。
- Train Batch。
- Data Cursor。
- 外生随机输入。
- Epsilon 调度。
- 当前主训练 Step。
- Cell Learning Rate。
- Gradient Clip。
- Token 数。

允许因为 Connection 不同而：

- 选择不同 Cell。
- 产生不同隐藏状态。
- 接受不同梯度。
- 产生不同 Cell 参数。

这些差异是实验效应。

## 9.3 实际路由验证

每次配对 Cell 更新必须记录：

- Old/New 训练动作差异。
- 完整路径差异。
- 六个 Cell 的调用数。
- 各 Cell 梯度范数。
- 实际训练策略 KL。
- 相同输入的哈希。
- 累计 Token 数。

本轮特别需要避免 R3f-v2 的情况：

**Connection 虽然通过准入，但实际配对训练几乎没有执行不同的路径。**

因此必须将：

`TRAINING_PATH_DIVERGENCE`

与：

`DETERMINISTIC_INFERENCE_PATH_DIVERGENCE`

明确分开。

即使没有实际训练路径差异，也仍完成规定的配对适应并如实报告，不能事后放弃候选或改写准入阈值。

---

# 10. Calibration 质量接受

## 10.1 接受条件

完成 K=10 后计算：

\[
L_O=L(\theta_O,\phi_0)
\]

\[
L_N=L(\theta_N,\phi_\alpha)
\]

当：

\[
\boxed{
L_O-L_N\ge0.001
}
\]

接受新组织。

否则保留旧组织。

此条件沿用 R3f-v2。

不得在运行过程中更改。

## 10.2 正式训练状态提交

如果接受：

正式状态变成：

\[
(\theta_N,\phi_\alpha)
\]

如果拒绝：

正式状态变成：

\[
(\theta_O,\phi_0)
\]

提交必须包含：

- Cell 参数。
- Cell Optimizer 状态。
- Connection 参数。
- Data Cursor。
- 主训练 Step。
- RNG 状态。
- Connection Version。
- 完整成本与事务记录。

不允许只提交 Connection 而保留另一条分支的 Cell Optimizer。

## 10.3 配对适应计入主训练预算

K=10 次适应更新必须计入 500 次主 Cell 更新。

例如：

```text
Step 100
Connection Proposal
K=10 Paired Adaptation
Select Old/New
Continue at Step 110
```

下一次 Proposal 仍然在 Step 200。

不能在执行 K=10 之后额外增加主训练次数。

被丢弃分支的 10 次训练另外计入实际消耗计算成本。

---

# 11. 联合适应归因

R3g 已经证明，单独观察：

\[
L_N<L_O
\]

不足以说明 Cell 与 Connection 实现了有益的共同学习。

因此，R3i 在每个通过 Trust Region 的 Proposal 后，增加局部的 2×2 归因评价。

## 11.1 四格评价

| Cell 参数 | 旧 Connection | 新 Connection |
|---|---:|---:|
| Old Cell | \(L_{OO}\) | \(L_{ON}\) |
| New Cell | \(L_{NO}\) | \(L_{NN}\) |

分别计算：

\[
\Delta_{\mathrm{route}}
=
L_{ON}-L_{OO}
\]

\[
\Delta_{\mathrm{cell}}
=
L_{NN}-L_{ON}
\]

以及：

\[
I=
L_{NN}-L_{NO}-L_{ON}+L_{OO}
\]

解释：

- Route Effect：在旧 Cell 下直接更换 Connection 的影响。
- Cell Effect：在新 Connection 下，更换为新分支 Cell 的影响。
- Interaction：Cell 参数来源与 Connection 之间的非加性交互。

所有 NLL 使用相同 Calibration 数据和确定性推理。

这属于机制诊断，不改变原来的最终质量接受规则。

不能根据四格评价重新选择已经确定的 Scale 或改变质量门槛。

## 11.2 真实 Cell 适应

至少记录：

- Old/New Cell 参数是否逐位相同。
- 六个 Cell 的参数差异。
- 实际训练路径是否改变。
- 梯度覆盖是否改变。
- \(L_{NN}-L_{ON}\)。
- Interaction。

必须区分：

`ROUTE_ONLY_BENEFIT`

`CELL_ADAPTATION_BENEFIT`

`FAVORABLE_INTERACTION`

`NO_DETECTABLE_ADAPTATION`

不能把纯 Connection 推理收益描述为 Cell 能力形成了专业分工。

---

# 12. Cell 使用集中与组织稳定性

R3h 已发现 H/T 会使 Cell 使用明显集中。

尤其 B-H，部分 Seed 的最大 Cell 调用次数超过最小 Cell 的六倍。

本轮允许继续集中，但必须监测。

## 12.1 Cell 使用指标

每个模型记录：

- 六个 Cell 调用次数。
- 每个 Stage 的 Cell 使用比例。
- 每个 Cell 的 Gradient Norm。
- 各 Cell 获得梯度的更新次数。
- 最大/最小 Cell 调用比。
- A 架构的 Pair 分布。
- B 架构的转移矩阵。
- 自连接比例。
- 当前路由熵。
- Connection Version。

## 12.2 Connection 演化指标

每次接受 Connection 后记录：

- 更新前后的硬动作变化率。
- 完整路径变化率。
- Cell 调用分布变化。
- 图转移分布变化。
- 接受后后续若干主 Step 的训练路径变化。
- NLL 变化。

目的是判断：

Connection 是否只是局部调换一些路径，还是逐渐形成不同的 Cell 工作分配。

## 12.3 不增加 Balance Loss

本次不引入：

- Load Balance Loss。
- Cell Utilization Reward。
- Entropy Bonus。
- 专门的 Graph Sparsity Loss。
- 动态增加或删除 Cell。

这些都会改变研究变量。

Cell 使用集中属于需要报告的结果，不自动视为失败。

只有非法计算、非有限梯度或训练状态损坏才触发工程停止。

---

# 13. 数据边界

严格沿用 R3h 的数据划分。

| 数据 | 使用方式 |
|---|---|
| Train | Cell 更新、Connection Proposal |
| Admission | Trust Region 与路由稳定性 |
| Calibration | K=10 质量接受、局部归因 |
| Development | 全部训练结束后的终端评价 |
| Confirmation | 保持 SEALED |

必须保持三个 Seed 的历史数据身份。

不得重新采样不同的 Train 数据以使新模型获得更好结果。

Development 在训练与 Connection 选择阶段不得读取。

Confirmation 不得加载、物化或访问。

---

# 14. 训练与计算预算

## 14.1 主训练

新增 Co-Adaptive 任务：

\[
3\text{ Seeds}
\times2\text{ Architectures}
\times2\text{ Routing Modes}
=12
\]

每个模型：

\[
500\text{ Updates}\times512\text{ Tokens}
\]

因此：

\[
\boxed{
6000\text{ 新主 Cell Updates}
}
\]

\[
\boxed{
3,072,000\text{ 新主训练 Tokens}
}
\]

复用的 12 个 R3h Frozen 模型已经执行过另外 6000 次主 Cell 更新。

它们属于历史对照，不得重复计入本次实际新执行成本。

## 14.2 Connection Proposal

最多：

\[
12\times4=48
\]

个 Connection Proposal。

最多：

\[
48\times8=384
\]

次 Connection Optimizer 更新。

每次执行 22 档尺度扫描（含零更新对照）。

## 14.3 额外 Cell 适应成本

若 48 个 Proposal 全部通过 Trust Region：

- 每个 Proposal 执行两条 K=10 分支。
- 其中一条最终成为正式状态。
- 另一条被丢弃。

因此，最多存在：

\[
48\times10=480
\]

次额外的被丢弃分支 Cell 更新。

对应：

\[
480\times512=245760
\]

个额外训练 Token。

这些不属于 3,072,000 个正式主训练 Token。

另外还需单独核算：

- 反事实后缀计算。
- Proposal 优化。
- Trust Region 扫描。
- Admission 检查。
- Calibration 质量接受。
- 四格归因。
- 路由监测。
- Development 终端评价。

**相同主训练 Token 不代表相同总计算预算。**

必须同时比较质量与真实额外成本。

---

# 15. 执行与 Checkpoint

## 15.1 Checkpoint 时刻

保留：

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

并在 Connection Proposal 的事务阶段额外保存。

## 15.2 事务状态

每个 Proposal 使用：

```text
PROPOSED
→ TRUST_EVALUATED
→ ADAPTING(k)
→ EVALUATED
→ COMMITTED
```

对每次提案永久关联：

- Official State。
- Candidate Connection。
- Scale Scan。
- Old/New Adaptation。
- Quality Decision。
- Selected State。
- Discarded Work。
- Content Hash。
- Experiment Identity。

## 15.3 恢复正确性

必须支持以下中断场景：

- 提案参数生成后。
- Trust Region 扫描途中。
- K=10 适应的中间步骤。
- Calibration 评价之后。
- 正式状态提交之后。
- 结果回传途中。

恢复时：

- 不得重新优化已保存候选。
- 不得重复 Cell 更新。
- 不得重复计入 Token。
- 不得重新选择已有 Proposal 的尺度。
- 不得替换旧、新分支。
- 不得覆盖已提交的历史状态。

## 15.4 私有数据保存

沿用现有的：

- Immutable Store。
- Content Addressing。
- HF Private Model Repository。
- Manifest。
- Receipt。
- 单协调器远端上传。

每次上传后验证远端内容与本地哈希。

如果 GitHub 结果回传失败，支持仅重新发布已有结果。

不能因为发布失败重新训练模型。

---

# 16. 统计评价

## 16.1 主要比较

预注册四项：

```text
A-CH vs A-FH
B-CH vs B-FH
A-CT vs A-FT
B-CT vs B-FT
```

注意：Frozen 基线必须是 R3h 对应 H/T 模型，不能使用原始 F。

## 16.2 质量改善

定义：

\[
\Delta_s
=
NLL_{\mathrm{CoAdapt},s}
-
NLL_{\mathrm{Frozen},s}
\]

负值表示联合训练更好。

每项质量假设得到 `SUPPORTED_EXPLORATORY` 的条件：

\[
\forall s,\ \Delta_s<0
\]

并且：

\[
\boxed{
\frac13\sum_s\Delta_s\le-0.005
}
\]

此门槛用于探索性实际效果判定，不代表统计确认。

## 16.3 统计方法

继续使用：

- 三 Seed 配对比较。
- Story-cluster paired bootstrap。
- 2000 次 Bootstrap。
- 四项主要比较的 Holm 调整。
- 原始逐 Seed NLL 差异。
- 描述性置信区间。

需要明确：

- Story 不是独立训练重复。
- 三个 Seed 仍然有限。
- Development 在历史研究中已使用过。
- Calibration 被多次用于 Connection 选择。
- 本轮不属于独立 Confirmation。

即使结果显著改善，仍只能作为探索性证据。

## 16.4 终端评价门禁

必须在全部 12 个新增 Co-Adaptive 模型完成训练、且历史 Frozen 对照验证完成之后，才执行终端 Development 评价。

不得根据前几个 Seed 的 Development NLL 停止其他 Seed。

不得在看到 Development 结果后重新调整提案尺度、接受阈值或路由温度。

---

# 17. 研究结果必须区分三种成功

## 17.1 Connection Admission Success

真实 Connection 提案通过：

- KL。
- Hard Action Change。
- Full Path Change。
- Nonempty Update。

这属于优化准入成功。

## 17.2 Co-Adaptation Mechanism Success

更新进入配对适应，并产生可测量的：

- 训练路径变化。
- Cell 梯度变化。
- Cell 参数变化。
- 或局部 Cell–Connection 交互。

这属于联合训练机制证据。

仅完成 K=10 的代码流程，不自动等于观察到了实际适应。

## 17.3 Model Quality Success

Co-Adaptive 模型在相同训练预算下，优于 Frozen 基线。

这属于主要科学假设。

三种成功必须分别报告。

不能将局部准入成功自动提升为模型质量支持。

---

# 18. 推荐代码结构

新建：

```text
research/projects/
  cellular-language-models/
    experiments/
      clm-001-qfirst-integrated/
        r3i/
          SPEC.md
          README.md

          config/
            frozen.json
            releases.lock

          implementation/
            minicells_r3i/
              parent_import.py
              baseline_reference.py
              routing.py
              trainer.py
              proposal.py
              trust_region.py
              adaptation.py
              crossover.py
              organization.py
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
            BASELINE_VERIFICATION.json
            PROPOSAL_RESULTS.json
            TRUST_REGION_RESULTS.json
            ADAPTATION_RESULTS.json
            CELL_CONNECTION_ATTRIBUTION.json
            TRAINING_ALIGNMENT.json
            GRAPH_ORGANIZATION.json
            DEVELOPMENT_NLL.json
            COMPUTE_COST.json
            SCIENTIFIC-FINDINGS.md
            RUN_MANIFEST.json
```

优先复用：

- R3h 的 Model/Trainer。
- R3h 的 H/T 路由。
- R3h 的 Keyed Random Streams。
- R3f-v2 的 Proposal。
- R3f-v2 的 Scale Scan。
- R3f-v2 的配对适应事务。
- R3g 的四格归因。
- 现有 Store、HF 与 GitHub 发布器。

不要复制新的 Cell 网络实现。

需要特别注意：R3h 的训练器以 Connection 冻结为前提，因此 R3i 必须在独立派生实现中明确允许经过准入并正式提交的 Connection 更新。

不能简单删除所有 Connection 冻结检查。

新的正确边界是：

- Proposal 期间允许独立候选 Connection 参数优化。
- 正式 Cell 更新阶段不允许 Connection 梯度更新。
- Connection 只能通过已提交的事务改变。
- 其他冻结模型权重仍然严格不变。

---

# 19. 必须实施的工程测试

## 19.1 初始与历史基线一致性

- R3h H/T 历史 Checkpoint 可验证。
- 新 CH/CT 的初始参数与 Frozen 对照相同。
- Step 100 Proposal 前训练状态可复现。
- Cell Optimizer 和数据游标相同。
- T 温度与 R3h 历史记录相同。
- H 的硬路由语义没有改变。

## 19.2 Proposal

- 候选在尺度扫描前保存。
- Connection 更新仅在 Proposal 中发生。
- 22 档尺度完整执行。
- Scale=0 与原始策略完全一致。
- 非空更新准入。
- 三个 Trust Region 门槛独立生效。
- 失败后继续主 Cell 训练。

## 19.3 配对适应

- Old/New 初始 Cell 相同。
- Old/New Optimizer 相同。
- 数据顺序一致。
- 随机流配对。
- K=10 正确执行。
- 正确提交 Selected State。
- 丢弃分支成本正确记录。
- 不存在重复主训练更新。

## 19.4 多次更新

重点测试：

- Step 100 接受后，Step 200 基于已更新 Connection 提案。
- Step 100 拒绝后，Step 200 基于原始正式状态提案。
- 多次接受不会覆盖旧 Proposal 记录。
- Connection Version 随正式提交正确增长。
- 恢复后不重复接受某次 Proposal。
- Step 500 主预算严格有效。

## 19.5 数据安全

- Admission 只参与准入。
- Calibration 只参与质量接受与局部机制归因。
- Development 不进入训练过程。
- Confirmation 保持封存。
- GitHub 不上传私有权重或训练 Token。
- 源码、配置、Checkpoint 和结果身份一致。

---

# 20. 自动执行方案

创建：

```bash
python scripts/run_r3i.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3i/config/frozen.json \
  --run-all
```

该脚本由 Codex 创建，目前不假定已经存在。

## 阶段 A：准备

自动完成：

1. 验证 R3h 历史 H/T Frozen Checkpoint。
2. 验证三个 R3c D500 父模型。
3. 验证数据与 Tokenizer。
4. 加载 R3h 冻结的 Temperature/Margin。
5. 建立独立 R3i Run Identity。
6. 创建 12 个新 Co-Adaptive 任务。

## 阶段 B：可比性门禁

运行到 Step 100 首次 Proposal 前，验证新训练与 R3h 基线的一致性。

如果无法验证，停止进入 Connection 联合更新阶段。

不得自动更换实验定义来使运行继续。

## 阶段 C：正式训练

自动执行：

```text
3 Seeds ×

A-CH
A-CT
B-CH
B-CT
```

每个模型 500 主 Step。

在 Step 100/200/300/400 自动执行：

```text
Save Official State
→ Generate Connection Proposal
→ Save Full Candidate
→ Trust Region Scan
→ Paired K=10 Adaptation
→ Calibration Quality Selection
→ Attribution
→ Commit Official State
→ Continue Main Training
```

若 Trust Region 全部失败：

记录拒绝，并直接继续训练。

若通过但 Calibration 收益不足：

选择 Old 分支并继续训练。

两种情况不得中止整个科学实验。

## 阶段 D：终端评价

全部模型完成后：

- 验证所有最终 Checkpoint。
- 加载 Development。
- 统一稳定硬路由评价 NLL。
- 对比历史 Frozen 基线。
- 汇总所有 Seed。
- 汇总 Connection 接受与训练组织演化。
- 汇总计算成本。
- 完成预注册统计。

## 阶段 E：发布

发布聚合结果到新的 GitHub Results 分支。

本地及私有 HF 保留完整训练状态。

GitHub 推送后必须核验 Remote Commit SHA。

如果推送失败，记录可恢复状态，支持只重试回传。

不得自动进入下一轮实验。

---

# 21. 输出要求

## 21.1 DEVELOPMENT_NLL.json

包含：

```text
seed
architecture
routing_mode
connection_mode
frozen_baseline_nll
coadaptive_nll
delta_nll
parent_checkpoint_sha256
final_checkpoint_sha256
```

## 21.2 PROPOSAL_RESULTS.json

包含：

```text
seed
arm
main_step
proposal_id
candidate_fingerprint
chosen_scale
trust_region_passed
effective_update
raw_policy_kl
actual_training_policy_kl
hard_action_change
full_path_change
```

## 21.3 ADAPTATION_RESULTS.json

包含：

```text
proposal_id
adapt_steps
old_calibration_nll
new_calibration_nll
quality_accepted
training_path_difference
cell_parameter_difference
selected_branch
discarded_training_cost
```

## 21.4 CELL_CONNECTION_ATTRIBUTION.json

包含：

```text
proposal_id
L_OO
L_ON
L_NO
L_NN
route_effect
cell_effect
interaction_effect
numerical_resolution
attribution_status
```

## 21.5 GRAPH_ORGANIZATION.json

包含：

```text
seed
arm
connection_version
cell_usage
stage_usage
pair_usage
edge_usage
self_connection_fraction
conditional_edge_distribution
predecessor_dependence
organization_change
```

A/B 架构不适用的字段可以为空，但不得伪造对应统计。

## 21.6 COMPUTE_COST.json

包含：

```text
main_cell_updates
main_train_tokens
connection_proposals
policy_optimizer_updates
counterfactual_work
trust_region_work
paired_adaptation_work
discarded_adaptation_work
calibration_work
development_work
total_wall_time
gpu_memory_peak
```

必须区分实际测量与解析估计。

## 21.7 SCIENTIFIC-FINDINGS.md

完整回答：

1. H1–H4 哪些获得探索性支持？
2. Connection 提案有多少成功通过准入？
3. 实际质量接受率是多少？
4. 每个 Seed 平均接受多少次 Connection 更新？
5. Connection 更新后 Cell 训练路径是否真正改变？
6. 收益主要来自路由还是 Cell 适应？
7. 多次 Connection 更新是否产生累积收益？
8. Cell 使用是否变得更加集中？
9. H 与 T 哪种更适合联合训练？
10. 额外计算成本是否与质量收益相称？
11. 是否有理由进入下一阶段独立确认？

---

# 22. 最终科学判定

根据四个预注册质量假设、独立机制指标和真实成本，最终给出以下级别结论。

### A. 联合训练没有取得质量收益

Connection 更新可能成功，但并未优于 Frozen 模型。

结论：

`COADAPTATION_QUALITY_NOT_SUPPORTED`

保留所有负结果。

### B. 存在质量收益，但主要来自 Connection

如果 Co-Adaptive 模型质量改善，而 Cell 适应归因贡献接近零：

`CONNECTION_QUALITY_GAIN_WITH_LIMITED_CELL_ADAPTATION`

不能宣称已经证明了有益的 Cell–Connection 共同演化。

### C. 存在额外 Cell 适应证据

如果 Connection 更新改变了真实训练路径，并在归因中产生超出数值误差的正向 Cell 收益：

`CELL_ADAPTATION_EVIDENCE_OBSERVED`

必须报告实际影响大小。

### D. 质量收益与共同适应均获得探索性支持

若三 Seed 中取得一致质量改善，同时有可重复的实际 Cell 适应证据：

`STABLE_COADAPTATION_SUPPORTED_EXPLORATORY`

仍不等于统计确认，也不证明已经形成稳定的功能专业化。

任何情况都不允许直接使用：

`CONFIRMED`

或：

`PROVEN_SPECIALIZATION`

---

# 23. 完成标准

只有以下项目全部满足，R3i 才算完成：

- R3h Frozen 对照完整且身份正确。
- 新增 12 个 Co-Adaptive 模型完成。
- 三个 Seed 的训练起点一致。
- 每模型 500 次主 Cell 更新完成。
- Connection Proposal 按冻结时刻执行。
- 所有实际候选被保存。
- Trust Region 没有被放宽。
- 成功准入的候选完成 K=10 配对适应。
- Old/New 分支事务可恢复。
- Connection 更新确实来自正式质量接受。
- 所有结果均在统一确定性推理下评价。
- Development 未参与训练选择。
- Confirmation 保持 SEALED。
- 提供完整成本比较。
- 四项主要假设独立判定。
- 全部结果具有 Source、Config 和 Checkpoint 哈希。
- 私有 HF 状态可恢复。
- GitHub 结果推送得到远端确认。
- 不自动启动后续训练。

本次允许负面科学结果。

只有工程状态不完整或数据不合法，才标记实验无效。

---

# 24. Codex 最终执行指令

请在 `ArcheLabs/mini-cells` 中实施：

`CLM-001-R3i-Stable-CoAdaptation-001`

目标是验证：

**在 R3h 已经证明有效的稳定硬路由 H 和校准低温路由 T 训练条件下，持续进行受 Trust Region 约束的 Connection 局部更新，能否使 Cell–Connection 联合训练超过冻结 Connection 的模型质量。**

实施原则：

1. 以 R3h H/T 历史结果作为严格的 Frozen 对照。
2. 从同一 R3c D500 父模型初始化新增 Co-Adaptive 模型。
3. 三个 Seed、两种架构、两种训练路由。
4. 不改变 Cell 网络、Gate 或推理路由语义。
5. 复用 R3f-v2 的局部 Connection Proposal 与 Trust Region。
6. 在 Step 100/200/300/400 各执行一次 Proposal。
7. 严格保存候选，完成 22 档尺度扫描。
8. 每次通过准入后执行 K=10 配对 Cell 适应。
9. Calibration 改善至少 0.001 才提交新组织。
10. 旧、新分支适应执行计入 500 主 Step 的正确预算。
11. 增加 R3g 的 Cell–Connection 四格归因。
12. 完整监测训练路径、Cell 梯度、图组织及使用集中程度。
13. 使用 R3h 原有配对随机流。
14. 保持 T 温度冻结，不根据运行结果重新校准。
15. 所有新增模型完成后统一评价 Development。
16. 以四项预注册比较进行科学判断。
17. 不读取 Confirmation。
18. 保持 Checkpoint 与事务可恢复。
19. GitHub 只发布聚合结果。
20. 不自动开始 R3j 或正式 Confirmation。

本轮成功不以 Connection 更新次数最多、路由变化最大或 Cell 使用最均衡为标准。

**真正的成功标准是：在相同的主训练预算与可比的初始条件下，允许 Connection 受约束地持续更新，能否让 Cell 学出更好的计算组织，并取得超过冻结 Connection 的实际模型质量收益。**

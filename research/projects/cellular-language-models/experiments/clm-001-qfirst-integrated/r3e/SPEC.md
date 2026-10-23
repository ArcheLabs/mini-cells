# MiniCells R3e — Stable Co-Adaptive Learning & Cell Graph

**稳定 Cell–Connection 协同优化与顺序 Cell 图实验规范**

**版本：** v1.0  
**实验 ID：** `CLM-001-R3e-SCAGL-001`  
**目标仓库：** `ArcheLabs/mini-cells`  
**性质：** 探索性 Development 实验  
**执行：** 一次性实现、单次启动、自动完成  
**Confirmation：** `SEALED`

---

# 1. 研究背景

## 1.1 已有实验结论

MiniCells 的 R3c、R3d 和 R3d-A 已经揭示三个问题。

**R3c：可微路由不保证有效分工。**

普通 Top-2 门控没有稳定改善模型质量，并且部分 Cell 存在严重梯度覆盖不足。

**R3d：反事实偏好学习没有改善 Connection。**

长期价值 Connection 在冻结 Cell 时显著恶化质量。联合训练也未能超越冻结 Router。

R3d 三 Seed 平均 Development NLL：

| Arm | NLL |
|---|---:|
| F0 | 4.550880 |
| L | 4.604461 |
| M | 4.596800 |
| F | **4.513366** |
| P | 4.514825 |
| J | 4.529020 |

**R3d-A：Cell 能够适应新的 Connection，但整体训练收益不足。**

在 P 和 J 的最终 Checkpoint 上，保持 Cell 参数不变时，使用已学习的 Connection 比恢复原始 Router 更好。

这说明训练后的 Cell 与 Connection 存在功能依赖，但尚未证明形成有价值的专业化，也未证明整体学习效率更高。

## 1.2 研究方向调整

过去我们主要研究：

\[
\text{如何预测下一 Cell 的长期价值？}
\]

现在调整为：

\[
\boxed{
\text{如何使 Cell 能力与 Connection 组织共同演化，}
\quad
\text{同时保持计算路径稳定并持续降低任务损失？}
}
\]

新的训练机制应同时考虑：

- Cell 的计算能力。
- Connection 的路径选择。
- 策略变化造成的状态分布偏移。
- Cell 对新 Connection 的适应。
- 最终任务质量和资源成本。

---

# 2. 科学目标

R3e 包含两条独立研究线。

## 2.1 R3e-A：稳定联合训练

保持 R3c/R3d 的计算架构。

验证：

**H1：受约束的 Connection 更新，能否使联合训练优于冻结 Router？**

**H2：评价 Cell 适应后的质量，能否比只评价当前 Cell 下的策略效果产生更好的联合训练结果？**

## 2.2 R3e-B：顺序 Cell 图

将四阶段 Top-2 改为八步串行 Top-1。

验证：

**H3：在相同顺序 Cell 图架构中，状态依赖的 Connection 学习能否优于固定路由与确定性随机路由？**

**H4：学习的连接是否表现出可重复的状态依赖、前驱依赖或自连接使用差异？**

H4 是结构特征假设，不等于证明功能专业化，更不等于证明形成小世界网络。

## 2.3 两条研究线相互独立

A 和 B 可以在一次实验中自动连续或并行执行。

但必须独立统计。

R3e-A 的主要结果不得与 R3e-B 混合，用来证明训练机制或图结构单独有效。

---

# 3. 共同实验基础

## 3.1 初始化

全部 Arm 从对应 Seed 的 **R3c D Step 500 Checkpoint** 出发。

不从 R3d L/P/J 的已学习 Connection 初始化。

原因是 R3d 的 Connection 已经表现出不可靠的价值排序，继承它可能污染新机制的归因。

保留已有：

- Embedding。
- Position Embedding。
- InputMemory。
- ContextRead。
- LayerNorm。
- 六个 FeedForward Cell。
- 原始 Gate。
- 最终 LM Head。

公共模型参数冻结，仅允许指定的 Cell 与新 Connection 参数更新。

## 3.2 共同模型设置

| 参数 | 设置 |
|---|---|
| Hidden width | 128 |
| Cell 数量 | 6 |
| Cell hidden | 248 |
| Vocabulary | 4096 |
| Sequence length | 64 |
| Effective batch | 8 |
| 每 Step Token | 512 |
| Context Read | 1 次 |
| Cell 权重跨步骤共享 | 是 |
| 动态 STOP | 不使用 |
| 新型 Cell 内部结构 | 不使用 |

必须以 R3c 冻结配置和 Checkpoint 的真实参数身份为准。

不修改历史实验数据、Tokenizer 或模型权重。

## 3.3 Seed

```text
151001301
151001302
151001303
```

三个 Seed 对应的原始 Cell 参数与随机状态均独立。

各个对照 Arm 应尽可能使用相同的 Train 数据顺序，并记录不同路由策略造成的执行差异。

---

# 4. R3e-A：稳定 Cell–Connection 协同优化

## 4.1 保持原始执行结构

固定四个 Stage，每阶段执行两个 Cell。

\[
h_{t+1}
=
h_t+
0.125
\sum_{i\in S_t}
w_{t,i}C_i(\operatorname{Norm}(h_t))
\]

其中：

\[
|S_t|=2,\qquad T=4
\]

总共执行：

\[
4\times2=8
\]

次 Cell 调用 / Token。

仍然使用原始 Gate 的 Logit 计算所选两个 Cell 的混合权重，并保持 StopGrad。

## 4.2 Connection 结构

复用 R3d 已实现的 15 Cell-pair 动作空间。

\[
\mathcal A=
\{(i,j):0\le i<j<6\}
\]

评分：

\[
q_\phi(s,a)
=
b_D(s,a)+\Delta_\phi(s,a)
\]

其中 \(b_D\) 为原始 D Gate 的 Cell-pair 基础分数。

\(\Delta_\phi\) 使用 R3d 的轻量线性修正结构，初始全部为零。

因此初始硬路由必须严格等价于 R3c D。

新 Connection 仅学习 Cell-pair 的选择，不改变 Cell 内部计算与混合权重规则。

---

# 5. 新的稳定联合训练算法

暂定名称：

**SCAGL — Stable Co-Adaptive Graph Learning**

在 R3e-A 中，仅实现 SCAGL 的 Cell–Connection 部分，不引入 Graph 拓扑更新。

## 5.1 基本原则

使用两个不同时间尺度：

- Cell 在当前稳定的 Connection 下持续学习。
- Connection 仅在预先规定的组织更新时刻提出候选改变。

一次 Connection 更新不能立即覆盖全部执行路径。

必须经过有限的策略变化约束和适应后质量评价。

## 5.2 Cell 学习

在当前正式 Connection \(\pi_k\) 下执行真实路径：

\[
h_{t+1}=F_{\theta,a_t}(h_t)
\]

通过最终 CE 更新实际执行的 Cell：

\[
\theta\leftarrow
\theta-\eta_\theta
\nabla_\theta L_{\mathrm{CE}}
\]

与 R3d 相同，这里是实际采样路径的条件梯度近似，而不是完整离散随机路径期望的无偏梯度。

Cell 使用原有 AdamW 配置、50 Step Warmup 及 Gradient Clipping。

## 5.3 Connection 候选改进

冻结当前 Cell 和 Connection 参数快照。

从 Train 批次中选取有限的实际访问状态。

对一个状态的所有 15 个候选 Cell-pair 执行反事实后缀。

每个候选：

1. 从相同当前状态出发。
2. 强制执行指定 Cell-pair。
3. 使用相同的当前 Cell 参数。
4. 后续 Stage 由原 Connection 重新路由。
5. 使用最终 Token CE 评分。

得到：

\[
\widehat Q_k(s,a)
\]

本轮不使用 R3d 的成对偏好 BCE 作为主策略目标。

改为直接最小化估计的策略期望成本：

\[
\boxed{
L_{\mathrm{policy}}(\phi')
=
\mathbb E_{s\sim d_{\pi_k}}
\left[
\sum_{a\in\mathcal A}
\pi_{\phi'}(a\mid s)
\left(
\widehat Q_k(s,a)-\overline Q_k(s)
\right)
\right]
}
\]

其中 \(\overline Q_k(s)\) 为该状态下候选 Q 的均值，仅用于数值中心化，不影响完整动作集合下的精确期望目标梯度。

\(\widehat Q\) 在当前候选更新中视为固定评分。

Connection 输出使用 Softmax 形成训练期随机策略，但正式推理采用硬 Argmax。

因此需要同时控制概率策略变化与硬执行路径变化。

## 5.4 Connection Trust Region

候选策略必须满足三个约束。

### A. KL 约束

\[
\mathbb E_s
\left[
D_{\mathrm{KL}}
(\pi_k\Vert\pi')
\right]
\le\delta_{\mathrm{KL}}
\]

### B. 硬动作变化约束

在相同参考状态上：

\[
D_{\mathrm{action}}
=
\Pr[
a_{\pi'}(s)\ne a_{\pi_k}(s)
]
\le\delta_{\mathrm{action}}
\]

### C. 完整路径变化约束

从相同输入分别执行旧策略和候选策略。

定义：

\[
D_{\mathrm{path}}
=
\Pr[
(a_0',\ldots,a_{T-1}')
\ne
(a_0,\ldots,a_{T-1})
]
\]

限制：

\[
D_{\mathrm{path}}\le\delta_{\mathrm{path}}
\]

KL 只作为近似策略信赖域约束。

硬动作与整条路径的约束用于弥补 Argmax 不连续造成的问题。

三个指标必须使用固定的参考数据和冻结参数快照计算。

## 5.5 Cell 适应后的策略评价

这是 R3e 相对于 R3d 最重要的改变。

不能只评价候选 Connection 对当前 Cell 的即时影响。

对于通过信赖域约束的候选策略 \(\pi'\)，从相同 Cell 参数 \(\theta_k\) 出发，建立两条短期适应路径。

旧组织：

\[
\theta_K^{\mathrm{old}}
=
\operatorname{Adapt}_K(\theta_k,\pi_k)
\]

候选组织：

\[
\theta_K^{\mathrm{new}}
=
\operatorname{Adapt}_K(\theta_k,\pi')
\]

两条路径必须具有：

- 相同起始 Cell 参数。
- 相同起始 Cell Optimizer 状态。
- 相同 Train 数据批次顺序。
- 相同 Cell 更新次数。
- 相同训练超参数。
- 独立但可复现的执行 RNG。

评价：

\[
\boxed{
\Delta_{\mathrm{adapt}}
=
L_{\mathrm{meta}}(
\theta_K^{\mathrm{new}},\pi')
-
L_{\mathrm{meta}}(
\theta_K^{\mathrm{old}},\pi_k)
}
\]

这里 \(L_{\mathrm{meta}}\) 使用已经存在的 Calibration 数据。

Calibration 在本实验中属于训练期组织选择数据，不再是独立验证集。

Development 只用于全部训练完成后的探索性比较。

## 5.6 接受或拒绝候选 Connection

候选需要同时满足：

1. KL 约束。
2. 硬动作变化约束。
3. 完整路径变化约束。
4. 适应后的 Calibration NLL 达到预设改善阈值。

满足则接受新 Connection，同时保留其对应的 K 步适应后的 Cell 状态与 Optimizer 状态。

否则保留旧 Connection 及其对应的 K 步适应结果。

无论接受还是拒绝，这 K 步都计入正常 Cell 更新预算。

候选分支的额外计算单独记入训练成本。

不得因为某个候选没有改善而回滚并重复相同主训练 Token，以获取人为优势。

**这是一种有限 K 步适应后的保守组织搜索近似，不是 TRPO 原算法，也没有严格的单调改进保证。**

---

# 6. R3e-A 对照组

| Arm | Cell | Connection | 候选接受条件 |
|---|---|---|---|
| **A-F** | 正常训练 | 原始 D 冻结 | 不进行策略更新 |
| **A-T** | 正常训练 | Trust Region 更新 | 当前 Cell 下的候选闭环改善 |
| **A-C** | 正常训练 | Trust Region + Co-Adaptation | K 步 Cell 适应后的改善 |

三个 Arm 从相同 R3c D500 起点开始。

### A-F：冻结路由基线

持续训练 Cell，保持原始 Gate 参数冻结。

作为主要基线。

应记录原始 Gate 参数冻结后，因 Cell 隐藏状态变化而发生的实际路由变化。

### A-T：仅约束 Connection 更新

执行候选价值估计和策略信赖域约束。

在保持当前 Cell 参数不变的情况下，使用 Calibration 闭环 NLL 评价是否接受候选。

不使用额外 Cell 适应分支。

其目的在于区分策略稳定性本身的收益。

### A-C：完整稳定协同优化

使用相同候选价值学习和策略信赖域。

候选经过 K 步 Cell 适应后再评价。

主要检验：

\[
NLL(A-C)<NLL(A-F)
\]

以及：

\[
NLL(A-C)<NLL(A-T)
\]

A-C 相较 A-T 的额外训练计算必须单独计量，不能宣称两者使用相同总训练算力。

---

# 7. R3e-A 冻结运行规则

默认预注册以下配置：

| 参数 | 值 |
|---|---:|
| Cell 更新 | 500 Step / Arm / Seed |
| Connection 提案时刻 | Cell Step 100、200、300、400 |
| 每次提案采样状态 | 12 |
| 每状态候选动作 | 全部 15 个 |
| 提案参数优化 | 8 Step |
| K 步 Cell 适应 | 10 |
| 最大平均 KL | 0.01 |
| 最大硬动作变化率 | 10% |
| 最大完整路径变化率 | 20% |
| 接受所需适应后 NLL 改善 | ≥0.001 |
| 最终推理探索率 | 0 |

这些是事先规定的研究初始值，不是已验证的最优超参数。

提案优化从当前正式 Connection 的参数开始。

所有反事实 Q 均基于提案开始时冻结的模型。

提案优化和信赖域回退仅作用于候选副本，正式模型参数不能被失败提案污染。

若候选超出约束，按固定系数：

```text
1.0
0.5
0.25
0.125
0.0625
```

缩小当前提案相对正式 Connection 的参数位移。

逐个检查真实硬路由与 KL 约束。

全部不满足则拒绝提案，不自动扩大预算或重新搜索参数。

### Cell 曝光

A-F、A-T、A-C 统一采用训练期受控探索：

\[
\epsilon:0.15\rightarrow0.02
\]

线性衰减。

最终推理：

\[
\epsilon=0
\]

三者使用可比较的探索预算。

仍然记录各 Cell 梯度覆盖，但不强制推理时均衡。

---

# 8. R3e-B：顺序 Cell 图

第二条研究线不再使用四阶段并行 Top-2。

改为：

**八步串行 Top-1，允许 Cell 自连接。**

## 8.1 图定义

\[
G=(V,E)
\]

节点：

\[
V=\{C_0,\ldots,C_5\}
\]

初始允许边：

\[
E=V\times V
\]

包括全部有向边与自连接，共：

\[
|E|=36
\]

另外允许从 START 状态选择任意 Cell。

这是一张完整的候选连接图。

并不意味着每一步执行所有 Cell。

每步仅执行一个 Cell。

## 8.2 顺序执行方程

设每个 Token 的初始隐藏状态为 \(h_0\)。

第 \(k\) 步：

\[
a_k=
\pi_\phi(h_k,a_{k-1},k)
\]

\[
\boxed{
h_{k+1}
=
h_k+
\alpha_G
C_{a_k}(\operatorname{Norm}(h_k))
}
\]

固定：

\[
k=0,\ldots,7
\]

每个 Token 正好执行八次 Cell。

不使用动态 STOP。

不增加新的 Cell 数量。

## 8.3 残差系数

初始建议：

\[
\alpha_G=0.0625
\]

理由是原始 R3c 使用四阶段、每阶段混合两个 Cell、残差系数 0.125。

新的顺序图包含八次单 Cell 更新，采用一半残差系数，可以在一阶近似上维持相似的累计更新尺度。

但两种结构并不代数等价。

串行计算会使后续 Cell 使用更新后的状态，这与同一 Stage 内并行混合两个 Cell 存在本质区别。

因此不能声称 R3e-A 与 R3e-B 在计算语义上完全相同。

## 8.4 Graph Connection

当前状态：

\[
h_k
\]

前一个 Cell：

\[
a_{k-1}
\]

当前执行步：

\[
k
\]

共同作为 Connection 的输入。

基础评分沿用冻结 D Gate，在步 \(k\) 使用：

\[
t=\lfloor k/2\rfloor
\]

得到六个原始 Cell Logit：

\[
z_j(h_k,t)
\]

新的 Connection 修正：

\[
q_j=
z_j+\Delta_\phi(h_k,a_{k-1},k)_j
\]

\(\Delta_\phi\) 初始化为零。

建议使用轻量线性修正，其中输入包含：

- 当前归一化隐藏状态。
- 前驱 Cell 的 One-hot，包括 START。
- 当前步数 One-hot。

不引入额外 Attention、复杂 GNN 或独立 Cell 编码器。

### Graph 的作用

所有 36 条边目前均合法。

Connection 学习这些边在不同状态下的使用偏好。

本轮不根据低频使用情况删除边，也不自动增加新节点。

换句话说：

**R3e-B 学习的是图上的动态路径选择，而不是永久图拓扑剪枝。**

真正的增边、删边和层级社区形成留给后续实验。

---

# 9. R3e-B 的对照组

| Arm | 图结构 | Connection | Cell |
|---|---|---|---|
| **B-F** | 八步串行 | 冻结 D Gate Top-1 | 训练 |
| **B-R** | 八步串行 | 固定确定性随机路由 | 训练 |
| **B-C** | 八步串行 | 稳定联合学习 | 训练 |

### B-F

每一步使用冻结原始 D Gate 选择一个 Cell。

不额外训练 Connection。

此组是顺序图结构的主要固定路由基线。

### B-R

使用固定 Seed 的确定性哈希路由。

候选分配不依赖目标标签。

在合法六个 Cell 之间形成可复现的近似均衡采样。

该基线用于判断模型表现是否仅来自多 Cell 顺序执行，而与学习路由无关。

### B-C

采用与 A-C 相同的稳定协同优化原则：

- Cell 快速学习。
- Connection 低频提出候选。
- 控制策略 KL。
- 控制硬路径变化。
- 使用 K 步 Cell 适应比较候选组织。
- 仅接受满足预定约束和质量改善条件的候选。

区别在于动作空间：

\[
|\mathcal A_B|=6
\]

Connection 从六个下一节点中选择一个，而不是 15 个 Cell 对。

反事实评分必须重新执行八步顺序图的剩余后缀。

不能使用 A 的四阶段反事实逻辑近似替代。

B-C 的提案时刻、适应步数、信赖域阈值及 Cell 更新预算与 A-C 保持一致。

---

# 10. 图组织分析

本轮只需要记录与顺序图直接相关的指标。

## 10.1 基础连接统计

- 六个 Cell 的调用比例。
- 36 条有向边的使用次数。
- 各 Cell 的自连接比例。
- 不同执行步的 Cell 分布。
- 不同前驱下的后继分布。
- 完整八步路径的重复与变化。

## 10.2 条件连接

分析：

\[
P(a_{k+1}=j\mid a_k=i)
\]

以及在输入状态不同的情况下是否存在显著差异。

但仅有条件分布不同，不足以证明某条边存在有用的计算专业化。

必须结合 B-C 相对 B-F/B-R 的最终 NLL、相同模型参数下替换路由的质量变化，以及对状态分布的解释。

## 10.3 不证明小世界性质

本轮不要求出现小世界网络。

六节点全连接候选图不应被称为小世界结构。

即使训练后部分边使用频率较高，也不能直接把稀疏使用图等同于数学意义上的小世界网络。

不设计人为鼓励聚类系数或缩短平均路径长度的正则项。

---

# 11. 正式训练预算

## 11.1 总体预算

全部六个训练 Arm：

```text
A-F
A-T
A-C

B-F
B-R
B-C
```

每个 Arm：

- 3 Seed。
- 500 Cell 更新 Step。
- 每 Step 512 个训练 Token。

主训练 Step：

\[
6\times3\times500=9000
\]

主训练 Token：

\[
9000\times512=4,608,000
\]

这个数字不包含：

- Connection 提案优化。
- 反事实后缀前向。
- K 步候选适应分支的额外计算。
- Calibration 评价。
- 终点 Development 评价。

这些额外成本必须分开统计。

对于 A-C/B-C 的 K 步适应，两条分支中最终被保留的一条计入 500 次主 Cell 更新；被放弃的另一条分支属于额外训练计算，不能悄悄忽略。

## 11.2 计算预算的两种比较

本轮必须区分：

**相同主训练 Token 预算：**

比较学习机制在相同 Cell 更新次数下取得的结果。

**相同总训练算力：**

需要考虑策略提案、反事实评分和影子适应分支的全部计算。

R3e-A 可以在同一主 Token 预算下与 A-F 比较，但不能宣称它在相同总训练 FLOPs 下优于 A-F，除非额外提供对应的算力匹配对照。

R3e-B 同理。

终点报告必须明确这种区别。

---

# 12. 数据与评价

## 12.1 数据

完全复用现有 TinyStories 资产、Tokenizer 和数据划分。

- Train：Cell 与 Connection 的参数更新、反事实采样。
- Calibration：训练期间候选组织的接受或拒绝，以及辅助指标。
- Development：所有 Arm 的正式终点探索性评价。
- Confirmation：保持 SEALED。

不重新下载数据、不重新训练 Tokenizer、不重新生成历史数据划分。

由于 Calibration 会多次参与候选策略选择，其结果不得视作独立验证。

Development 也已用于历史 R3 系列研究，因此所有结论均标记为探索性。

## 12.2 主要指标

唯一主要任务质量指标：

\[
L_{\mathrm{CE}}
\]

对应 Development NLL。

必须报告：

- 每 Seed NLL。
- 三 Seed 平均 NLL。
- 主要配对差异。
- Story 聚类 Bootstrap 区间。
- Seed 间变化。
- 训练成本。
- 推理总参数。
- 实际每 Token Cell 调用。
- 估计 MatMul FLOPs。
- 实测评估吞吐。
- 峰值推理显存。

## 12.3 主要比较

R3e-A：

\[
\Delta_A=NLL(A-C)-NLL(A-F)
\]

\[
\Delta_{\mathrm{adapt}}
=
NLL(A-C)-NLL(A-T)
\]

R3e-B：

\[
\Delta_{BF}
=
NLL(B-C)-NLL(B-F)
\]

\[
\Delta_{BR}
=
NLL(B-C)-NLL(B-R)
\]

负值代表前者质量更好。

对于训练轨迹不同的 Arm，这些比较属于整体训练方法差异，不能单独归因于某一次 Router 更新。

## 12.4 判断门槛

预先冻结：

\[
\Delta NLL_{\mathrm{minimum}}=-0.005
\]

主要科学假设获得探索性支持，需要：

1. 三个 Seed 的配对差异均为负。
2. 平均改善至少达到 0.005 NLL。
3. 所有 Arm 满足对应计算预算。
4. 不存在严重的数值或数据完整性问题。

Story Bootstrap 和多重比较调整作为额外统计证据报告。

只有三个独立 Seed 时，不得声称获得强统计确认。

## 12.5 独立的稳定性判断

必须把任务质量与路由稳定性分别判断。

例如：

- Connection 的动作变化率降低。
- Cell 梯度覆盖改善。
- 图上连接分布更稳定。

这些即使成立，也不能自动让 H1–H3 获得支持。

同样，B-C NLL 较低，也不能单独证明 H4 的图结构专业化。

---

# 13. 资源效率评价

本轮已经接近 MiniCells 最初的资源目标，因此必须认真记录资源成本。

## 13.1 A 组

保持：

- 六个共享 Cell。
- 八次 Cell 调用。
- 一次 Context Read。

新增 Connection 参数单独统计。

## 13.2 B 组

保持：

- 六个共享 Cell。
- 八次 Cell 调用。
- 一次 Context Read。

但是串行深度更长、Connection 执行次数更多。

不能仅根据八次 Cell 调用相同，就宣称与 A 组具有相同 FLOPs 和推理延迟。

必须实际计量。

## 13.3 总参数

明确区分：

\[
P_{\mathrm{total}}
\]

和：

\[
P_{\mathrm{active}}
\]

本轮不进行 Cell 参数压缩。

因此，B 组即使成功，也只能说明顺序图及其训练机制有价值，不能证明总模型驻留参数已经减少。

同等总参数与推理预算下对单一递归核心或紧凑 FFN 的优势，属于后续独立研究问题。

---

# 14. 最小必要检查

用户要求精简实验，因此不执行额外大型测试流程。

只保留会影响科学正确性的检查。

## 14.1 初始化

- Checkpoint SHA、Seed、Step 与配置身份一致。
- A 组零修正时严格复现 D 路由。
- B 组每步只执行一个合法 Cell。
- 数据与 Tokenizer 身份一致。
- Confirmation 不访问。

## 14.2 运行中

- CE 有限。
- 梯度有意义且有限。
- 冻结参数不发生变化。
- 反事实当前动作改变后，后缀重新路由。
- 候选评估不使用 Development。
- 候选接受后，Cell 参数与 Optimizer 状态来自同一适应分支。
- 拒绝候选后，正式 Connection 不受污染。
- 正常训练 Cell 更新次数严格符合预算。

## 14.3 运行结束

- 每个 Arm 和 Seed 具有独立完成状态。
- 最终 Checkpoint 可验证。
- 完整训练成本与额外训练计算分别记录。
- 所有结果能追溯到源码 SHA 与冻结配置。

不再增加独立长时间 GPU Smoke、Pilot、硬件 Benchmark 或历史回归测试阶段。

---

# 15. 一次性执行与恢复

## 15.1 统一入口

Codex 创建：

```bash
python scripts/run_r3e.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/config/frozen.json \
  --run-all
```

`run_r3e.py` 是本次新增的统一入口。

同样创建 Kaggle Notebook：

```text
notebooks/kaggle-development.ipynb
```

Notebook 只负责：

- 准备固定源码。
- 使用现有 Secrets。
- 调用统一 Runner。
- 显示最终科学汇总。

不应在 Notebook 中重复实现训练循环。

## 15.2 Kaggle 环境

沿用现有：

- Kaggle T4 ×2。
- 现有数据资产。
- 现有 HF 私有 Checkpoint 仓库。
- 现有 GitHub 写权限配置。
- PyTorch/NumPy 运行环境。

无需重新创建 Kaggle Dataset 或 Hugging Face Repository。

若现有资源可用，直接进入正式训练。

## 15.3 自动执行所有 Arm

运行程序应按资源情况执行全部三个 Seed、六个 Arm。

支持两 GPU 并发不同的独立 Seed。

不能通过把不同 Seed 混合成数据并行训练来改变原有有效 Batch 定义。

一次 Run All 后不要求人工继续批准后续 Arm。

科学结果为负，不触发自动改变参数或追加训练。

## 15.4 Checkpoint 与恢复

复用既有 Store 与 HF Manifest 机制。

每 50 Cell Step 保存完整训练状态。

对于影子适应分支，仅保留当前必要的候选状态和最终选择结果，不永久保存所有被拒绝的完整参数副本。

Checkpoint 应包含：

- Cell 参数。
- Connection 参数。
- Cell Optimizer。
- Connection Optimizer（如存在）。
- 当前 Cell Step。
- 已完成策略提案数。
- 正式 Connection 版本。
- 当前分支适应状态（若在提案中断）。
- RNG。
- 数据游标。
- 运行配置与源码身份。

中断后恢复时，必须能够明确知道某次提案是否已经正式接受。

不能重复接受同一个提案或重复计入相同更新。

保留现有私有存储上传与即时回读。

跨独立 Kaggle Session 的持久化验证如未发生，继续如实标记为未验证。

---

# 16. 结果交付

建议独立目录：

```text
research/projects/cellular-language-models/
  experiments/
    clm-001-qfirst-integrated/
      r3e/
        SPEC.md
        README.md
        config/
          frozen.json
          releases.lock
        implementation/
          minicells_r3e/
            common.py
            stable_connection.py
            policy_proposal.py
            adaptation.py
            graph.py
            training.py
            evaluation.py
            storage.py
            runner.py
        notebooks/
          kaggle-development.ipynb
        reports/
          RUN_MANIFEST.json
          METRICS.json
          POLICY_STABILITY.json
          ADAPTATION_RESULTS.json
          GRAPH_SUMMARY.json
          TRAINING_COST.json
          SCIENTIFIC-FINDINGS.md
```

通用模块应从 R3c/R3d 已有实现复用，不复制无修改的大量代码。

私有模型状态和数据不进入公开 GitHub。

公开结果分支只提交可发布的配置、实验身份、聚合结果和科学结论。

---

# 17. 自动生成科学结论

最终 `SCIENTIFIC-FINDINGS.md` 需要逐项回答：

### H1：稳定的协同训练是否有效？

比较 A-C 与 A-F。

如果 A-C 不能改善 NLL，不能声称稳定联合训练问题已解决。

### H2：考虑 Cell 适应是否有帮助？

比较 A-C 与 A-T。

若 A-C 不优于 A-T，不能声称 K 步适应评价带来了独立收益。

同时报告两者额外训练计算成本的差异。

### H3：顺序 Cell 图中的连接学习是否有效？

比较 B-C、B-F、B-R。

若 B-C 没有稳定优于两个同结构基线，则没有证据支持本轮图式 Connection 学习。

### H4：图是否形成可重复的连接组织？

报告自连接、方向性、前驱依赖、状态条件差异与 Seed 间一致性。

结构差异只能支持「学习到非均匀或条件连接分布」，不能直接证明形成小世界网络或有意义的 Cell 功能专业化。

### 科学判断状态

统一使用：

- `SUPPORTED_EXPLORATORY`
- `NOT_SUPPORTED`
- `INCONCLUSIVE`
- `INVALID_RUN`

不得将工程成功、训练稳定或某个中间指标改善直接等同于科学假设成立。

---

# 18. Codex 最终实施指令

请在 `ArcheLabs/mini-cells` 仓库中实施：

`CLM-001-R3e-SCAGL-001`

本次任务不是撰写提案，而是实现并执行正式 R3e 实验。

实验包含两条独立对照：

**R3e-A：Stable Cell–Connection Co-Adaptation**

保持 R3c 的四阶段 Top-2 结构。

实现基于长期候选动作价值的 Connection 提案，采用策略 KL、硬动作变化和完整路径变化约束，并通过有限 K 步 Cell 适应评价决定是否接受新的 Connection。

执行 A-F、A-T、A-C 三个 Arm。

**R3e-B：Sequential Cell Graph**

构建六节点、36 条候选有向边（包括自连接）的 Cell 图。

使用八步串行 Top-1 计算，每一步根据当前状态、前驱 Cell 与计算步数选择下一 Cell。

采用同样的稳定协同优化原则训练 Connection。

执行 B-F、B-R、B-C 三个 Arm。

### 执行要求

1. 读取现有 R3c、R3d、R3d-A 代码与冻结结果，复用已有基础设施。
2. 新建 R3e 独立实现，不修改历史冻结实验。
3. 创建 `frozen.json` 和 `releases.lock`。
4. 完成后直接使用现有 Kaggle 环境运行正式实验。
5. 同一次执行中完成全部 Seed 与 Arm。
6. 不设置中途人工授权。
7. 不重复执行已有的大型工程测试。
8. 不新增无关 Pilot、Smoke 或冗余探针。
9. 保留最小正确性校验与可恢复 Checkpoint。
10. 不访问 Confirmation。
11. 不因早期科学负结果改变冻结参数或提前终止合法实验。
12. 结果完成后自动生成报告并发布允许公开的汇总。

### 最终必须返回

- 实验实际源码 SHA。
- 实际运行配置 SHA。
- 全部六组、三个 Seed 的 Development NLL。
- H1–H4 的判断。
- Connection 提案数量、接受数量和拒绝原因。
- 硬路由、完整路径稳定性指标。
- Cell 梯度覆盖。
- 顺序图的实际连接统计。
- 训练和推理资源消耗。
- 私有 Checkpoint 真实存储位置。
- GitHub 结果分支。
- 未完成或失败项的明确状态。

**实验结果应忠实反映数据。不能为了支持 SCAGL 而改变基线、阈值、训练预算或评价协议。**

---

# 19. R3e 的最终科学使命

R3e 不承担证明 MiniCells 已经具备高于成熟语言模型的资源效率这一目标。

它需要先回答两个基础问题。

**第一：**

\[
\boxed{
\text{Cell 与 Connection 能否在相对稳定的计算组织下，}
\text{形成可持续改善任务质量的联合学习机制？}
}
\]

**第二：**

\[
\boxed{
\text{在顺序 Cell 图中，学习连接是否比固定连接}
\text{更有利于组织有限的计算能力？}
}
\]

如果第一项成立，我们就拥有了进一步研究计算组织演化的训练基础。

如果第二项也成立，我们才有更充分的依据继续研究动态图拓扑、Cell 社区、稀疏跨组连接及更大规模的计算网络。

**只有在此之后，才进入以总参数、驻留内存和真实推理成本为主要目标的模型压缩与扩展研究。**

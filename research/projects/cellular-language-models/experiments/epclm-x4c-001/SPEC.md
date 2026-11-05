# EPCLM-X4C-001：X4 Cellular Computation Redesign

**项目：** MiniCells / Cellular Language Models
**实验类型：** 架构重构、计算效率、持久状态与语言能力验证
**状态：** Proposed — Implementation Specification
**基线：** Native CLM X4
**实施仓库：** `ArcheLabs/mini-cells`

---

# 1. 实验目标

## 1.1 背景

MiniCells 已经探索了两条不同的技术路线。

**Native CLM X4** 使用局部 Causal Attention、SwiGLU、RoPE、RMSNorm 和循环残差更新，在小型语言建模实验中取得了相对较好的结果。

但是，X4 仍然存在明显的架构问题：

- 固定执行 3 个 Stage，每个 Stage 4 次迭代。
- 每次迭代都重新执行 Attention 和 FFN。
- 计算深度无法根据当前任务需求变化。
- 不具备独立于局部 Token 表示的长期持久计算状态。
- 无法证明大量重复计算是必要的。
- 参数量接近 Transformer，不代表计算量或实际执行效率接近 Transformer。

**EPCLM Core-001** 则采用独立持久 Cell State、共享更新规则和局部消息传递，探索持续状态计算。

但现有实验尚未证明其通信机制能够产生明确的语言建模优势。尤其是 IC 与 R1 的结果接近，说明增加 Cell Communication 不一定能改善预测。

因此，本次研究不继续独立优化 EPCLM Ring 拓扑，也不直接在 X4 后面增加完整 EPCLM 网络。

本次目标是：

> **重新设计 X4 的计算架构，让具有持久状态的 Cell 承担部分原本由重复 Attention/FFN 完成的计算，在尽可能保留语言能力的前提下，降低计算成本，并验证持续状态对长程任务的价值。**

## 1.2 核心原则

1. **Attention 保留其必要功能。** 第一版不追求消灭 Attention，而是减少冗余计算。
2. **Cell 是计算单元。** Cell 不仅保存状态，还必须对语言表示产生有效计算贡献。
3. **State 是持续计算状态。** 不要求其无损保存历史 Token。
4. **State 与 Parameters 分离。** 模型参数是训练得到的计算规则；State 是运行过程中演化的动态表示。
5. **替换优先于叠加。** 新增 Cellular 模块应尽量替代已有计算，而不是无限增加计算开销。
6. **机制贡献必须可验证。** 每种新能力都必须对应独立消融实验。
7. **效率必须实测。** FLOPs 减少不等于 GPU 延迟或训练时间必然降低。

## 1.3 本次不实现

本次不实现在线权重学习、Cell Growth、Mitosis、复杂动态拓扑、分布式 Cell 部署、JAM/PVM 执行或完整 Agent。

不要求通过本次实验直接证明复杂工程任务的长程推理能力。

这次仅验证基础语言计算重构与跨 Chunk 持续状态利用机制。

---

# 2. 实现与证据依据

Codex 首先检查以下源码。

**Native X4：**

- `research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_runtime.py`
- `research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_phase_b.py`
- `research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_phase_c.py`
- `research/projects/cellular-language-models/experiments/native-clm-v0/implementation/README.md`

**EPCLM Core-001：**

- `research/projects/cellular-language-models/experiments/epclm-core-001/`
- `implementation/model.py`
- `gc001-baseline-repair/repair_models.py`
- `topology-001/topology.py`

**已有增量执行研究：**

- `research/projects/cellular-language-models/experiments/epclm-001-incremental-execution/`

需要检查这些源码在当前分支上的实际位置与状态。如果路径发生变化，按 Git 历史定位，不应创建未经核实的替代实现。

保留历史实验、Checkpoint、Protocol 和 Result。

本次使用独立实验目录：

`research/projects/cellular-language-models/experiments/epclm-x4c-001/`

不得覆盖原始 X4、T1 或 EPCLM Core-001 的训练结果。

---

# 3. 原始 X4 架构

## 3.1 基础配置

按照已核查的 Native CLM Phase-C 源码：

| 参数 | X4 |
|---|---|
| Trainable Parameters | 1,893,912 |
| Hidden Dimension | 188 |
| Attention Heads | 4 |
| Stages | 3 |
| Iterations per Stage | 4 |
| Attention Windows | 8、32、128 |
| FFN | SwiGLU |
| FFN Hidden Dimension | 639 |
| Position | RoPE |
| Normalization | RMSNorm |
| Update | Fixed Scaled Residual |
| Context | 128 |
| Vocabulary | 2,048 |

X4 包含三个独立的 Stage，每个 Stage 内部重复执行四次 Attention、FFN 和 Residual Update。

同一 Stage 内的主要参数共享，三个 Stage 之间的参数独立。

完整前向过程包含 12 次 Attention 和 12 次 FFN。

## 3.2 计算成本

根据现有解析 FLOPs 估算器，在 125 Token 输入长度下：

| 模型 | 估算每 Token Forward FLOPs |
|---|---:|
| X4，3×4 | 13.110 M |
| X4，3×2 | 6.940 M |
| X4，3×1 | 3.855 M |

上述数值是源码公式的解析估算，不是实测计算量或 GPU 延迟。

本次要求 Codex 重新执行估算并记录计算公式。

不能直接把这些估算当作最终性能结果。

## 3.3 需要验证的主要问题

X4 在每个 Stage 中重复执行四次相似结构的计算。

但目前尚未明确：

- 第 2、3、4 次迭代分别产生多少预测收益。
- 重复 Attention 和重复 FFN 哪一部分更加重要。
- 不同 Stage 是否需要相同的迭代次数。
- 是否能使用状态计算替代部分重复 FFN。
- 是否能保留少量 Attention，同时维持接近原始 X4 的语言质量。

本次必须从上述问题开始，而不是直接实施新架构。

---

# 4. 实验总设计

本次采用四个阶段，逐步增加机制。

| 阶段 | 名称 | 目标 |
|---|---|---|
| G0 | Baseline & Correctness | 固定 X4 基线，验证训练、推理与 FLOPs 统计 |
| G1 | Computation Depth | 识别重复迭代中可以减少的计算 |
| G2 | Persistent State | 比较 Flat State 与 Cellular State 的价值 |
| G3 | Selective Cell Computation | 使用 Cell 替代部分密集 FFN，验证质量与成本 |

不得跳过 G1，直接将 Cell 模块叠加到完整 X4 上。

每个阶段产生独立的模型定义、配置、训练身份和报告。

完成一个阶段，不代表下一个阶段的研究假设自动成立。

---

# 5. G0 — X4 基线与正确性验证

## 5.1 原始基线

保留：

**A0 — X4-3×4**

要求严格使用已冻结的 X4 结构：

- 三个 Stage。
- 每 Stage 四轮迭代。
- 原始窗口大小。
- 原始参数共享机制。
- 原始 RoPE、SwiGLU、RMSNorm。
- 原始训练数据和 Tokenizer 协议。

首先定位历史 X4 的实际训练结果及对应源码、参数、数据身份。

如果历史 Checkpoint 无法恢复，记录 `HISTORICAL_WEIGHTS_UNAVAILABLE`，使用可复现的全新训练建立基线。

不得把重新训练的模型冒充历史 Checkpoint。

## 5.2 原始实现正确性

Codex 应当检查：

- Causal Attention 不存在未来 Token 泄漏。
- RoPE Position ID 与真实 Token 位置一致。
- Attention Window 8/32/128 语义正确。
- 同一 Stage 的四轮更新按正确顺序执行。
- 各 Stage 参数独立。
- Embedding 与 LM Head 的权重绑定正确。
- Residual Update 系数与历史实现一致。

新增可重复的单元测试。

## 5.3 增量推理基线

必须评估 X4 的实际自回归推理成本。

优先复用已有 EPCLM-001 增量执行实现，不得再开发一套没有必要的重复执行框架。

需要支持：

- Prefill。
- 单 Token Decode。
- Stage/Iteration KV Cache。
- 固定窗口管理。
- Snapshot / Restore。
- Cached 与 Reference Forward 一致性检查。

**同一 Stage 不同迭代的中间表示不同，因此不能简单把四轮迭代共用一份 KV Cache。**

必须按实际计算依赖保存必要的缓存，或证明某种缓存复用是数值等价的。

对于超过历史 128 Token 上限的连续运行，必须先冻结新的绝对位置和滑动窗口语义。

不能把扩展后的 Sliding-Window 实现与旧版 128 Token 截断重算混为同一个参考函数。

## 5.4 G0 验收

通过以下要求：

- 原始 X4 参数量一致。
- 相同输入下 Reference Forward 稳定、确定。
- Cached/Uncached 在相同因果语义下通过数值等价测试。
- FLOPs 统计可复现。
- 能输出 PPL、生成文本、Prefill/Decode 性能报告。
- 不修改历史基线。

若失败，停止后续正式性能比较，先修复基础设施。

---

# 6. G1 — 计算深度消融

## 6.1 核心假设

**H1：X4 的部分重复 Attention/FFN 计算可以减少，而不会产生与节省成本不成比例的语言质量损失。**

## 6.2 实验组

| ID | Stage Iterations | 目标 |
|---|---|---|
| A0 | 4 / 4 / 4 | 原始基线 |
| A1 | 2 / 2 / 2 | 减少一半重复迭代 |
| A2 | 1 / 1 / 1 | 每 Stage 仅执行一次 |
| A3 | 1 / 2 / 1 | 检验中间 Stage 的额外计算价值 |

A3 是预先指定的探索性候选，不得根据测试集表现反复搜索不同组合。

## 6.3 实施要求

优先在现有 `PhaseBStage` 基础上引入可配置 Iterations，而不是复制完整 X4 实现。

需要保证：

- 不同实验组的 Stage 仍使用对应的原始参数结构。
- 减少迭代不会错误删除整个 Stage。
- Step Embedding 数量与迭代次数一致。
- 除指定的 Iteration Count 外，不修改 Attention、FFN 和训练目标。
- 参数量变化必须记录。
- 预训练 X4 权重裁剪只能作为诊断，不能替代重新训练后的公平比较。

在独立训练中，必须从新的模型初始化开始。

## 6.4 额外机制消融

在 G1 中增加一个低成本的诊断实验：

分别观察减少 Attention 重复次数与减少 FFN 重复次数的影响。

诊断组可以采用：

- 保留四轮 Attention，部分轮次不执行 FFN。
- 保留四轮 FFN，部分轮次不执行 Attention。

这些组用于区分计算贡献。

由于修改会改变计算图，不应只依据冻结模型的即时 PPL 判断长期训练表现。

## 6.5 G1 输出

必须形成：

- PPL vs Iterations。
- PPL vs Training FLOPs。
- Attention FLOPs / FFN FLOPs 分解。
- Training Tokens/s。
- Prefill Latency。
- Batch-1 Decode Latency。
- Peak VRAM。
- 参数量。
- 不同迭代下的实际生成文本。

选择后续实验的基础架构时，不要求最低 FLOPs 方案一定胜出。

应根据语言质量与计算成本共同选择，并明确其相对 A0 的质量损失。

---

# 7. G2 — Persistent State 计算

## 7.1 核心假设

**H2：在减少 X4 重复计算后，引入持续状态能够改善跨 Chunk 的信息利用能力，并以可控成本弥补部分计算损失。**

G2 从 G1 选择的基础模型出发。

不再恢复完整 3×4 结构。

## 7.2 State 定义

建立两种结构。

**B1 — Flat Persistent State**

状态形状：

`[Batch, 2048]`

这是非 Cellular 的状态对照。

**B2 — Cellular Persistent State**

状态形状：

`[Batch, 16, 128]`

每个 Cell 保存独立状态，但第一版允许共享主要更新参数。

两种结构具有相同的持久状态元素数量。

不过，Flat State 必须使用合理、有效的状态更新网络，而不能故意采用过弱的线性层。

参数量和计算量应尽可能匹配，无法匹配的部分必须报告。

## 7.3 双时间尺度

使用：

- Token Time：当前输入与语言预测。
- Cell Time：跨 Chunk 的状态更新。

第一版固定：

`Chunk Size = 32`

64 Token 只作为后续敏感性实验候选。

同一 Chunk 内，各 Token 使用进入该 Chunk 时的 State。

Chunk 完成后，使用该 Chunk 的因果计算表示更新持久状态。

下一 Chunk 才能读取更新后的 State。

## 7.4 State Read

令当前 Token 的语言表示为 \(h_t\)，当前 State 为 \(S_b\)。

使用 Query-conditioned State Read：

\[
r_t=\operatorname{Attention}(q(h_t),K(S_b),V(S_b))
\]

然后将读取结果参与语言计算：

\[
h'_t=h_t+g_t r_t
\]

该操作可以使用 Attention，但只能对当前持久 State 的位置执行寻址。

不得隐式恢复完整历史 Token Attention。

## 7.5 State Write

Chunk 完成后：

\[
S_{b+1}=U(S_b,H_b)
\]

其中 \(H_b\) 表示当前 Chunk 的计算结果。

第一版建议采用：

- 共享的基础 Update Network。
- Identity-preserving Residual State。
- 可学习的选择性写入门控。
- 显式 State Reset。
- Snapshot / Restore。

Write 不允许使用未来 Chunk 的信息。

## 7.6 训练要求

必须让训练损失能够对跨 Chunk 状态写入产生有效学习信号。

可以使用跨多个 Chunk 的截断反向传播。

明确记录：

- TBPTT 跨越的 Chunk 数。
- State Detach 位置。
- 梯度截断策略。
- 每次训练更新的有效预测 Token 数量。
- 是否存在长程依赖训练信号。

不能在每个 Chunk 更新 State 后立即无条件 Detach，又声称后续语言损失已经训练了跨 Chunk 的写入机制。

## 7.7 G2 实验组

| ID | 模型 | 目标 |
|---|---|---|
| B0 | G1 选定基础模型 | 无持久 State |
| B1 | B0 + Flat State | 验证持久状态本身 |
| B2 | B0 + 16 Cell State | 验证 Cellular 状态组织 |
| B3 | B2，但禁用 State Read | 验证输出是否利用 State |
| B4 | B2，但冻结 State 更新 | 验证持续更新是否有效 |

B3/B4 首先进行冻结权重因果消融；如需作结构质量比较，再分别独立训练。

不得混淆两种实验含义。

## 7.8 G2 任务

普通 TinyStories PPL 不能单独证明持续状态的价值。

需要新增严格分离的机制任务：

**M1：Delayed Dependency**

在早期 Chunk 提供信息，多个 Chunk 后要求使用该信息。

**M2：State Update**

提供一个初始条件，随后改变该条件，要求模型使用新的条件。

**M3：Interference**

连续输入多个相近信息，观察状态混淆与错误更新。

**M4：Cross-Chunk Constraint**

在较早的 Chunk 中定义任务约束，后续 Chunk 要求模型作出符合约束的选择。

**M5：State Continuity**

同一输入分别采用完整连续执行与 Snapshot/Restore，验证结果一致性。

任务必须包含未见过的随机组合，避免模型仅记忆训练样例。

如果加入额外任务训练，应将其与原始 TinyStories 训练分开记录，并保证相关对照组获得相同训练数据和任务预算。

---

# 8. G3 — Selective Cellular Computation

## 8.1 核心假设

**H3：Cell 可以替代部分密集 FFN，以较低的实际计算成本提供有效的语言计算，同时保持自己的持续状态。**

此阶段不是给 X4 附加更多 FFN。

必须真正替换一个现有 FFN 路径。

## 8.2 最小实施范围

第一版只替换一个 Stage 的 FFN。

建议选择 Stage 2 作为固定候选。

不要同时替换全部三个 Stage。

Stage 1 和 Stage 3 保留原有机制。

Stage 2 的 Attention 保留，原有密集 FFN 则替换为 Selective Cell Compute。

## 8.3 Cell 定义

每个 Cell：

\[
Cell_i=(\theta_i,s_i)
\]

其中：

- \(\theta_i\)：计算参数。
- \(s_i\)：持久状态。

第一版采用共享基础计算网络与少量 Cell-specific 参数。

不要求每个 Cell 都拥有完整独立的 639 宽度 SwiGLU。

## 8.4 Selective Routing

对当前 Token 表示和持久状态计算路由分数：

\[
a_i=R(h_t,s_i)
\]

选择 Top-k Cell：

\[
\mathcal A_t=\operatorname{TopK}(a,k)
\]

第一版固定：

`Cells = 16`

`Top-k = 2`

选中的 Cell 执行：

\[
u_i=F_i(h_t,s_i)
\]

输出：

\[
y_t=\sum_{i\in\mathcal A_t}g_i u_i
\]

必须保证真正只计算被激活 Cell 的主要计算路径。

不能先执行全部 Cell 的 FFN，再用 Gate 将未选中的结果乘零，并将其声称为稀疏计算优化。

Router 自身也必须纳入计算量统计。

## 8.5 重要对照

需要区分稀疏计算本身与 Persistent State 的价值。

| ID | 结构 | 目的 |
|---|---|---|
| C0 | B2，保留原始密集 FFN | Cellular State 基线 |
| C1 | Stateless Routed Cell FFN | 验证稀疏 FFN |
| C2 | Stateful Routed Cell FFN | 验证持久状态与 Cell Compute 的结合 |
| C3 | 参数/计算匹配的普通 FFN | 控制模型容量变化 |

如果 C1 与 C2 表现相同，不能宣称持久状态改善了计算能力。

如果 C2 的 FLOPs 降低，但实际延迟明显升高，也不能宣称获得工程效率优势。

## 8.6 Routing 诊断

至少记录：

- Cell Activation Frequency。
- Active Parameters per Token。
- Routing Entropy。
- Cell Utilization。
- Dead Cell Count。
- Routing Distribution by Input。
- Cell-specific 参数的梯度范数。
- Stateful vs Stateless 的性能差异。
- 实际 Kernel 执行与显存访问开销。

不要求 Cell 自动形成可解释的语义角色。

如果出现分工，应通过独立诊断证明，而不是根据可视化推断。

---

# 9. 推理与状态生命周期

## 9.1 Runtime Session

新架构必须提供明确的状态生命周期：

`init → prefill → step → snapshot → restore → continue → reset`

Runtime 应分别管理：

- 模型权重。
- 各 Stage 的局部 KV Cache。
- Persistent Cell State。
- Chunk 内尚未提交的状态更新信息。
- 全局 Token Position。
- RNG State。

## 9.2 固定 Chunk 边界

Chunk 边界必须由 Session 中的绝对 Token 位置决定。

不能因为用户一次调用输入 32 个 Token、另一次调用输入 8 个 Token，就改变模型的计算语义。

必须支持：

- 连续输入 128 Token。
- 四次输入 32 Token。
- 128 次输入单个 Token。

在相同固定 Chunk 分段规则和相同模型权重下，上述方式应产生一致的状态与预测。

## 9.3 Snapshot/Restore

Snapshot 必须包含：

- Model Identity。
- Tokenizer Identity。
- Absolute Position。
- Persistent State。
- 各 Stage/Iteration 的有效 KV Cache。
- 当前 Chunk 已处理 Token 数。
- 尚未提交的 Chunk 状态更新缓冲。
- RNG State。

恢复后必须能够继续生成，并与不中断的执行比较。

禁止通过重新输入完整历史 Token 来假装实现 State Restore。

## 9.4 State Reset

独立文档之间默认 Reset。

跨 Chunk 保留 State。

独立任务之间不得无意传播状态。

真实持续运行场景需要明确的 Session 边界，不应默认把多个用户或多个独立文档的状态混合。

---

# 10. 统一训练与数据协议

## 10.1 原始语言数据

优先复用 Native X4 的 TinyStories 数据协议：

- 原始固定数据 Revision。
- ByteLevel BPE。
- Vocabulary 2,048。
- Context 128。
- Training Sequence 125。
- Batch 8。
- AdamW。
- Warmup + Cosine。
- 原始学习率与梯度裁剪配置。

历史 X4 基线训练目标为 10M Token。

Codex 必须根据实际数据 Manifest 和训练源码核对这些设置，不得仅根据 README 重建一个看似相同的数据集。

所有训练输入和 Tokenizer 文件记录 SHA-256。

## 10.2 训练预算

采用两个层次：

**Pilot**

用于正确性、训练稳定性和初步筛选。

建议使用 100K Token Smoke，再使用最多 1M Token Development Pilot。

**Development**

对选定的候选模型执行完整 10M Token 训练。

与原始 X4 使用相同的有效训练目标计数规则。

不因为某个候选计算更快，就额外增加其训练 Token。

## 10.3 公平比较

必须分别比较：

**Parameter-Matched**

控制总可训练参数量。

**Token-Matched**

控制有效训练 Token 数量及数据顺序。

**Compute-Matched**

在相同累计训练 FLOPs 下比较模型质量。

**Inference-Matched**

在相同硬件、精度、Batch 和推理语义下比较延迟与内存。

这些比较不应混为一个综合评分。

本次建议优先保留参数量处于原始 X4 的 ±5% 范围内。

无法满足时记录具体原因与实际差异，不得通过隐式增加模型规模获得质量优势。

## 10.4 随机种子

第一轮使用开发种子进行机制筛选。

新架构筛选不得访问 Independent Test。

优先复用历史允许用于开发的 Seed，正式复核至少采用三个独立训练 Seed。

历史 Confirmation Seed 不应在探索过程中反复使用。

## 10.5 计算资源与失败边界

优先复用仓库已有 Kaggle Notebook 和 GPU 执行基础设施。

禁止在源码、日志和结果中保存凭据。

仅在当前环境具有明确可用且已授权的计算资源时执行正式 GPU 训练。

每个 GPU Job 都要有明确的最大运行时长、训练 Token 预算和失败恢复策略。

若运行资源不足，必须完成代码、测试、Notebook、Protocol 和 Ready-to-run 交付，并如实标记训练为 `NOT_RUN_RESOURCE_UNAVAILABLE`。

不得将未执行的训练写为 PASS。

---

# 11. 必须报告的指标

## 11.1 语言能力

- Validation NLL / PPL。
- Development Holdout NLL / PPL。
- Learning Curves。
- Greedy Generation。
- Sampling Generation。
- Repetition / Diversity。
- 长程信息利用正确率。

不同模型的 PPL 必须使用相同 Tokenizer、相同评估样本和相同预测目标位置。

## 11.2 训练性能

- Total Training FLOPs。
- Training Tokens/s。
- Training Wall Time。
- Peak VRAM。
- Optimizer State Memory。
- Backward / Forward FLOPs。
- Grad Norm。
- 数值溢出和异常。

## 11.3 推理性能

- Prefill 32 / 64 / 128。
- Batch-1 Decode Median / P95。
- Batch-8 Decode Throughput。
- Cached Decode。
- Persistent State Memory。
- KV Cache Memory。
- Temporary Activation Memory。
- Total Runtime Memory。
- Snapshot / Restore Cost。

## 11.4 Cellular 专项

- State Read Contribution。
- State Write Contribution。
- State Perturbation Effect。
- State Reset Effect。
- Flat vs Cellular 差异。
- Cell Utilization。
- Active Compute FLOPs。
- Routing Cost。
- 跨 Chunk 信息保持与更新能力。

---

# 12. 阶段判定与停止规则

实验允许产生否定结论。

## G1 判定

判断减少迭代是否产生有效的质量—计算权衡。

优先考虑在开发实验中相对原始 X4：

- 预测质量没有明显恶化。
- 前向 FLOPs 至少降低 20%。
- 实测 Decode 性能不明显恶化。

初步非劣阈值建议为 PPL 不高于基线的 1.02 倍。

这只是开发筛选标准，不能代替多 Seed 科学确认。

若所有减少迭代的候选都明显退化，则 G1 结论为 `DEPTH_REDUCTION_NOT_SUPPORTED`，不得假设 EPCLM 能自动补回缺失的能力。

## G2 判定

需要满足：

- State Read/Write 不存在因果泄漏。
- Snapshot/Restore 正确。
- 至少一类跨 Chunk 任务获得稳定改善。
- 与 Flat State 比较时，不能仅凭 Cellular 状态数量较多宣称优势。

若 B2 与 B1 没有差异，则结论为 `NO_CLEAR_CELLULAR_ORGANIZATION_GAIN`。

## G3 判定

需要满足：

- Cell 真正替代已有密集 FFN。
- 有效激活的 Cell 可以通过计算图和性能分析确认。
- 相对于 G2 的密集 FFN 基线，实际计算成本确实降低。
- 语言质量与长程指标没有不可接受的退化。

如果减少 FLOPs 但真实 GPU 延迟增加，应标记 `FLOPS_GAIN_ONLY`。

如果 Cell 完全未被有效使用，应标记 `NO_CELL_COMPUTE_CONTRIBUTION`。

这些判定不能通过修改测试结果或事后改变阈值消除。

---

# 13. 实现测试

Codex 至少新增以下自动化测试。

| 测试 | 检查内容 |
|---|---|
| `test_x4_reference_equivalence` | X4 原始计算与重构基线一致 |
| `test_iteration_configs` | 3×4、3×2、3×1 等迭代配置正确 |
| `test_causality` | 不使用未来 Token |
| `test_kv_cache_equivalence` | Cached 与 Reference Forward 一致 |
| `test_chunk_partition_equivalence` | 不同 API 输入分块产生相同结果 |
| `test_state_persistence` | State 跨 Chunk 保留 |
| `test_state_reset` | 独立 Session 不共享状态 |
| `test_snapshot_restore` | 恢复后连续计算一致 |
| `test_state_read_ablation` | 能正确关闭 State Read |
| `test_state_write_ablation` | 能正确关闭 State Write |
| `test_flat_vs_cellular_state` | 两种状态组织均可正常训练 |
| `test_sparse_cell_execution` | 未激活的 Cell 不执行主要 FFN |
| `test_parameter_accounting` | 参数量统计正确 |
| `test_flops_accounting` | 各计算模块分别统计 FLOPs |
| `test_training_gradients` | 必要梯度能跨 Chunk 传播 |

其中 Chunk Partition Equivalence 和因果正确性属于必须通过的门禁。

如果失败，不允许发布对应模型的正式语言质量结果。

---

# 14. 工程交付

建议创建目录：

`research/projects/cellular-language-models/experiments/epclm-x4c-001/`

至少包括：

- `README.md`：实验介绍、执行方式与当前状态。
- `SPEC.md`：本实施规范及冻结版本。
- `protocol.json`：机器可读实验协议。
- `architecture.py`：新架构模块。
- `cell.py`：Cell Compute 与 State 定义。
- `routing.py`：选择性 Cell 路由。
- `state.py`：State Update、Reset、Snapshot。
- `runtime.py`：Prefill、Decode 与 Chunk 运行时。
- `flops.py`：计算量分解。
- `train.py`：训练入口。
- `evaluate.py`：语言能力、State 和性能评估。
- `run.ipynb`：Kaggle GPU 实验入口。
- `tests/`：全部必要测试。
- `reports/`：阶段结果和研究总结。

具体文件可以根据已有基础设施调整，优先复用现有模块。

不得为了符合上述文件名而复制大量已有代码。

所有新文件应与旧 X4 和 EPCLM 实验保持清晰的依赖边界。

---

# 15. 结果报告格式

每个模型至少生成：

`config.json`
`training-metrics.json`
`validation-results.json`
`holdout-results.json`
`generation-samples.json`
`flops-report.json`
`inference-profile.json`
`memory-profile.json`
`state-diagnostics.json`
`decision.json`

每个结果必须记录：

- Source Commit。
- Protocol Hash。
- Dataset / Tokenizer Hash。
- Model Configuration。
- Seed。
- Training Tokens。
- Optimizer Updates。
- GPU / CUDA / PyTorch Version。
- Model Weight Identity。
- 是否完整训练。
- 是否完整评估。
- Independent Test 是否访问。
- 结果发布状态。

最终生成中文报告：

`RESEARCH-REPORT.zh-CN.md`

报告必须包括：

1. 原始 X4 的主要计算瓶颈。
2. 哪些重复迭代可以减少。
3. 减少迭代后的语言质量损失。
4. Persistent State 是否产生独立价值。
5. Cellular State 是否优于 Flat State。
6. Selective Cell Compute 是否有效替代 FFN。
7. 训练与推理成本是否实际降低。
8. 是否出现模型质量退化。
9. 哪些研究假设被支持、被否定或证据不足。
10. 是否值得进行更大规模训练。

必须保留完整原始数据，不只报告最好的结果。

---

# 16. Codex 执行顺序

Codex 按以下流程实施：

**Step 1 — Repository Audit**

检查 X4、EPCLM 和已有增量执行代码，确认可复用资产。

输出架构与数据身份核查报告。

**Step 2 — Freeze Baseline**

固定原始 X4 的参数、计算和评估语义。

建立正确性和 FLOPs 测试。

**Step 3 — G1 Implementation**

实现可配置迭代次数，完成 A0/A1/A2/A3。

先执行 CPU Smoke，再执行预算内的 GPU Pilot。

**Step 4 — G1 Decision**

分析质量与计算成本，冻结一个后续实验基础架构。

不得根据 Independent Test 调整选择。

**Step 5 — G2 Implementation**

实现 Flat Persistent State、Cellular Persistent State、State Read/Write 和 Chunk Runtime。

完成因果、Snapshot 和训练梯度测试。

**Step 6 — G2 Evaluation**

进行语言建模与跨 Chunk 任务比较。

明确区分 Flat State 与 Cellular 的独立贡献。

**Step 7 — G3 Implementation**

只替换一个 Stage 的 FFN。

建立 Stateless/Stateful Routed Cell 对照。

执行真实稀疏计算与性能测试。

**Step 8 — Final Evaluation**

在资源允许时对筛选后的候选执行完整开发训练与多 Seed 复核。

输出统一性能、语言质量、State 和计算量比较。

**Step 9 — Publication**

提交源码、测试、协议和结果报告。

保留历史实验不变。

如果正式训练未运行或中断，仍需交付完成的工程实现，并明确标注缺失的研究证据。

---

# 17. 最终成功标准

本次实验并不要求所有候选都优于 X4。

我们真正希望获得的是明确的架构结论。

最低成功条件：

**工程层面：**

- X4 原始基线可复现。
- 可配置重复计算深度。
- 新架构能够完成语言训练与自回归生成。
- Persistent State 可以正确读写、保存、恢复。
- 选择性 Cell 计算真正执行，且可测量成本。
- 全部必要正确性测试通过。

**研究层面：**

至少能够可靠回答：

- X4 有多少重复计算是必要的？
- State 能否在不保留完整历史 Token 的情况下提高跨 Chunk 信息利用能力？
- 多 Cell 状态是否比等规模 Flat State 更有价值？
- Cell Compute 能否以更低成本替代部分密集 FFN？
- 新架构的计算质量—成本曲线是否优于原始 X4？

如果结果支持新的 Cellular 架构，再考虑扩大模型规模、增加独立 Cell 参数、引入复杂路由和更长时间尺度计算。

如果结果不支持，应当保留最简单且表现最好的有效机制，不继续增加复杂度。

---

# 18. 最终研究命题

**English**

Can persistent computational cells replace part of the repeated Attention and FFN computation in X4, improving computational efficiency while preserving language modeling quality and enabling effective long-horizon state utilization?

**中文**

具有持续状态的计算 Cell，能否替代 X4 中部分重复的 Attention 与 FFN 计算，在保持语言建模质量的同时提高计算效率，并获得有效的长程状态利用能力？

本次实验的核心不是证明 Cellular 必然优于 Transformer，而是确定：

**Cellular Computation 是否能够成为一种具有独立计算价值、持久状态能力和可验证效率优势的语言模型计算机制。**

这是本次重构与所有实验判定的最终依据。

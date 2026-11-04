# EPCLM Core-001 — Event-Driven Persistent Cellular Computing

**实验与工程设计规范 v0.1**

- **项目**：MiniCells / EPCLM
- **日期**：2026-10-31
- **状态**：DRAFT — 待实施前技术审查与协议冻结
- **实验性质**：核心架构可行性验证
- **实施平台**：Python / PyTorch / Kaggle GPU
- **目标产物**：可训练的 Cellular Language Model Core、推理运行时、实验报告、可复现模型资产
- **历史关系**：独立于 EPCLM-001 和 EPCLM-002，不修改既有冻结结果

---

## 1. 研究目标

### 1.1 核心目标

建立一个以 Cellular State 为计算载体的事件驱动语言模型，验证以下命题：

> 一个由持续状态 Cell 组成的神经计算网络，能够通过局部通信和共享递归更新规则处理连续输入事件，并在不反复访问完整历史 Token 的情况下，形成可用于自回归语言预测的内部计算状态。

这个目标包含三个核心概念。

**State — 状态**

State 是计算系统当前的内部表示，不等同于记忆数据库。它可以表示输入条件、中间结果、历史依赖以及其他通过训练形成的潜在信息。

**Connection — 连接**

Connection 定义 Cell 之间的信息传递方式。第一版采用固定局部连接，验证多个 Cell 通过通信完成有用计算的能力。

**Event — 事件**

Event 是触发状态变化的外部输入。模型不需要在每次计算时重新读取完整上下文，而是在已有状态上继续演化。

### 1.2 需要回答的科学问题

本实验的核心问题为：

- H1：Cellular State 能否作为连续事件处理的有效计算载体？
- H2：Cell 间通信是否对模型能力提供独立的、可重复的贡献？
- H3：该架构能否直接通过 Next-Token Prediction 学习真实语言？
- H4：Cellular 状态组织相对于等预算 Flat Recurrent Model 是否具有额外价值？

另设探索性问题：

- H5：在没有新 Event 的情况下，增加内部演化次数能否改善语言预测或特定推理任务？

H1–H3 是最小核心机制的必要条件。H4 决定 Cellular 组织是否具有超越普通递归模型的价值。H5 不属于第一版核心验收。

---

## 2. 已有实验与设计依据

### 2.1 可以直接继承的结论

| 历史实验 | 结论 | Core-001 中的处理 |
|---|---|---|
| Echo | 共享局部更新规则已有工程实现基础 | 复用 Cellular 更新思想 |
| Native N1 | 简单残差更新在既有架构搜索中优于更复杂的部分更新机制 | 优先使用 Residual Update |
| Native X3/X4 | SwiGLU、现代化语言组件提供有效改善 | 复用可迁移组件与训练体系 |
| Constructive CLM-004 | 顺序、并行 Cell 组合在受控任务中获得正式支持 | 为后续多 Cell 计算提供依据 |
| Constructive CLM-005 | 学习型路由与控制在受控条件下可行 | 保留接口，不作为首版依赖 |
| Native M1 | 稀疏 Cell 能参与真实语言模型训练 | 复用训练经验 |
| Native M2/M3 | 受保护写入和增长仍未解决语言保持问题 | 首版不进行在线参数更新 |
| EPCLM-001 | 增量执行与事件运行时已有受限工作负载证据 | 复用工程正确性测试 |
| EPCLM-002 A0 | Writer/Reader 隔离可行，但学习型记忆未通过筛选 | 不继承 Event-indexed Memory 架构 |
| Experiment 006 | Path-First 工程框架和对照设计已建立 | 为未来动态执行提供接口 |

### 2.2 Reasoning NCA 的启发

Reasoning with Neural Cellular Automata 在空间推理中验证了局部连接、递归状态演化和异步更新等机制。

Core-001 主要继承：

- 分布式 Cell 状态；
- 共享局部更新规则；
- 多轮内部计算；
- 从演化状态读取结果。

但不直接宣称其迷宫、数独推理结果能够迁移到语言模型。

Core-001 与原研究最重要的区别是引入连续外部事件，并让同一个 Cellular State 跨事件保留。

### 2.3 研究边界

本实验不要求：

- 无损保存完整历史；
- 无限长度上下文；
- 运行时修改模型参数；
- Replay-free Continual Learning；
- 动态 Cell 增长；
- 自主 Cell 分裂；
- Adaptive Attention；
- Hierarchical Routing；
- Tree Search；
- 强化学习；
- 优于 Transformer 的效率。

这些机制均可作为后续独立实验，不得因其缺失而将 Core-001 自动判为失败。

---

## 3. 统一计算模型

### 3.1 双时间尺度

定义：

- `t`：Event 时间，即第几个输入事件。
- `k`：内部计算时间，即收到当前事件后第几轮 Cell 更新。

系统的主要过程：

```text
Previous State S_t
        |
        v
   Inject(event)
        |
        v
   Internal State
        |
        v
   Evolve K steps
        |
        v
     New State S_(t+1)
        |
        +------> Readout
        |
        +------> Next Event
```

状态转移：

`S_(t+1) = Evolve(Inject(S_t, e_t), K)`

输出：

`logits_(t+1) = Read(S_(t+1))`

原则上，`Evolve` 也允许在没有新 Event 的情况下继续执行。

### 3.2 状态定义

固定状态：

`S ∈ R^[B, N, D]`

其中：

- `B`：Batch Size。
- `N`：Cell 数量。
- `D`：每个 Cell 的状态维度。

每个 Cell 持有状态 `h_i`。

首版不为 Cell 分配具体语义角色。Cell 的位置标识可以固定，但其内部功能应通过训练形成。

### 3.3 State 生命周期

**事件内部：**

Cell 状态可以进行多轮更新，并在不同 Cell 之间传播。

**事件之间：**

处理完事件后，保留完整 Cellular State，作为下一事件的起点。

**训练片段之间：**

同一序列的状态可以跨 Truncated BPTT 边界传递，但切断更早的梯度图。

**独立文档之间：**

默认重置 State，避免无关样本之间的信息泄漏。

**会话之间：**

可以通过 Snapshot 保存状态，并在后续恢复。

不实现自动跨用户、跨任务共享状态。

---

## 4. 最小模型架构

### 4.1 基础配置

建议的第一轮配置：

| 参数 | 初始值 |
|---|---|
| Cell 数量 | 16 |
| State Dimension | 128 |
| Persistent State | 2,048 个数值 |
| FP32 状态大小 | 8 KiB |
| Cell 拓扑 | 一维环形网络 |
| 邻域半径 | 1 |
| 初始更新轮数 | K = 4 |
| 通信 | 固定邻域可学习聚合 |
| 更新网络 | 共享 Residual MLP / SwiGLU |
| Event 类型 | Token |
| Tokenizer | 现有 2,048 词表 |
| Output | Next-Token Logits |
| 历史 Token KV Cache | 无 |
| 动态 Cell 增长 | 无 |

参数量不预先强制等于 1.9M。先完成实际参数计数，再确定用于公平对照的统一模型规模。

### 4.2 Event Encoder

每个输入 Token 映射为：

`e_t = Embedding(x_t)`

输入经过投影后，被注入固定入口 Cell。

首版采用一至两个固定入口，不使用全局动态搜索。

例如：

`h_0 <- h_0 + W_e · e_t`

Event 只在当前事件开始时注入一次。

如果实现选择在整个事件期间保留 Event 条件，必须将该条件视为临时工作输入，而不是跨事件的历史缓存。

首选实现为**单次注入**，降低计算语义复杂度。

### 4.3 Connection

采用一维环形网络：

`Neighbors(i) = {i-1, i, i+1}`

所有索引对 Cell 数量取模。

优点：

- 通信范围固定；
- 每个 Cell 的邻居数量相同；
- 容易实现批量计算；
- 容易进行通信消融；
- 不依赖历史 Token 长度。

环形拓扑只是最小工程基线，不表示语言具有天然环形结构。

### 4.4 Local Communication

每个 Cell 从邻域获取消息：

`m_i = W_left·Norm(h_(i-1)) + W_self·Norm(h_i) + W_right·Norm(h_(i+1))`

相对方向投影参数在所有 Cell 之间共享。

首版不使用 Content-based Q/K/V Attention。

后续可以在保持其余组件不变的条件下，将 `LocalCommunication` 替换为局部 Attention。

### 4.5 Shared Update Rule

采用简单残差更新：

`delta_i = SwiGLU_MLP([Norm(h_i), m_i])`

`h_i_new = h_i + alpha · delta_i`

其中 `alpha` 是预先固定的更新比例，初始候选为 0.1，实际数值须在训练前通过数值稳定性预检冻结。

每轮更新采用同步语义：所有 Cell 根据上一轮的状态计算新状态，不允许在同一轮中因原位修改产生顺序依赖。

不引入复杂 GRU、ARU 或多个独立的 Cell Update Network。

### 4.6 Readout

Readout 从固定数量的 Cell 状态聚合得到语言预测。

首版可以使用：

`z = FixedPool(S)`

`logits = LMHead(Norm(z))`

`FixedPool` 可以是固定拼接后的线性投影，或固定可学习权重聚合，但不得依赖历史 Token/KV。

Readout 必须为无状态、无副作用操作。

不能设置 Event Embedding 到 LM Head 的直接旁路，以免模型不经过状态演化就完成预测。

### 4.7 无事件内部计算

Runtime 支持：

`S_new = Evolve(S, extra_steps)`

此时不注入新 Event。

这一能力必须从第一版存在，但不强制要求增加计算步数会提高语言质量。

额外迭代的实际收益属于 H5，须在独立实验中验证。不能仅因为输出随迭代次数变化就宣称实现了推理。

---

## 5. Core Runtime 接口

### 5.1 必需接口

```python
state = core.init_state(batch_size=1)

state = core.inject(state, token_id)

state = core.evolve(state, steps=4)

logits = core.read(state)

snapshot = runtime.snapshot(state)

state = runtime.restore(snapshot)

state = core.reset(state)
```

推荐同时提供一个组合接口：

```python
state, logits = core.step(
    state=state,
    token_id=token_id,
    evolve_steps=4,
)
```

这里建议采用函数式返回新状态的实现方式，不在模型的训练前向过程中原位覆盖状态 Tensor，以避免破坏梯度追踪。

### 5.2 推理循环

```text
Initialize State
     |
Read input token
     |
Inject
     |
Evolve K steps
     |
Read next-token logits
     |
Sample next token
     |
Repeat
```

模型生成的 Token 与外部输入的 Token 使用相同的事件处理语义。

### 5.3 Snapshot 格式

Snapshot 至少包含：

- 格式版本；
- Model ID 与权重哈希；
- State Shape；
- State Tensor；
- 数据类型与精度；
- 已处理 Event 数；
- 内部计算步数；
- 当前会话边界；
- 随机调度状态（如果未来启用随机更新）；
- 内容校验哈希。

Snapshot 不包含完整历史 Token 序列。

训练 Checkpoint 与推理 Snapshot 必须分开定义。训练 Checkpoint 还需要优化器、训练位置及随机数生成器状态。

### 5.4 执行正确性

必须测试：

- 完整逐事件执行与分段执行一致；
- 同一运行状态保存恢复后结果一致；
- 同一 Seed 和调度下结果可复现；
- 当前输出不读取未来 Event；
- `read()` 不改变 State；
- 处理某一序列不会污染其他 Batch Lane；
- Reset 后状态符合定义；
- 超过数值稳定范围时明确失败，不允许静默修复。

FP32 CPU 路径应优先实现精确的参考行为。GPU 路径在固定硬件、精度和调度下采用注册的数值容差比较。

---

## 6. 训练设计

### 6.1 数据集

首轮使用现有 TinyStories 训练体系。

复用：

- 固定的 TinyStories 数据修订；
- 现有 ByteLevel BPE Tokenizer；
- 2,048 词表；
- 既有语料准备与哈希校验机制。

新实验必须建立独立的数据清单。

训练、开发验证和正式确认数据应按照 Story ID 或规范化故事内容划分，并检查重复与交叉污染。

不能沿用已被其他实验观察过的 Confirmation Set，声称它仍是全新盲测数据。

### 6.2 训练目标

主要目标为标准自回归语言建模：

`L = CrossEntropy(Read(S_(t+1)), x_(t+1))`

模型必须仅根据已处理的输入事件预测后续 Token。

不使用人工 Cell 角色标签。

不使用中间推理轨迹监督。

不预设每个 Cell 应学习的语义。

不在第一轮加入 RL Reward。

### 6.3 梯度传播

推荐 Truncated BPTT。

初始候选：

- 训练展开长度：64 Token；
- 同一文档后续片段继续使用原有 State；
- 在 BPTT 边界切断历史梯度；
- 文档结束时明确 Reset；
- 不同 Batch Lane 独立管理 State。

必须记录每个片段的实际有效 Token 数，确保计算正确的训练 Token 预算。

这里需要特别防止：因为训练片段长度是 64，就把模型的状态也在每 64 Token 重置。

两者必须严格区分。

### 6.4 内部迭代数

第一轮主模型固定：

`K_train = 4`

使用相同训练权重，在评估时诊断：

`K_eval ∈ {2, 4, 8}`

K=2 和 K=8 属于计算深度敏感性测试，并非独立训练模型的公平对照。

只有在主实验完成后，才考虑随机计算深度训练，例如从 `{2,4,8}` 中采样。

未经相应训练，不应期待增加 K 必然提升质量。

### 6.5 优化器

优先使用现有 Native CLM 已验证的 AdamW 训练基础设施：

- Gradient Clipping；
- 学习率 Warmup；
- Cosine Decay；
- 混合精度训练；
- 非有限梯度检测；
- 完整训练 Checkpoint；
- 可恢复的数据采样顺序。

开始前必须检查多轮残差迭代是否导致 State 范数、梯度或输出出现不稳定。

任何稳定性修复都需要记录源码和配置版本。

---

## 7. 对照模型设计

本实验必须避免将“可以训练”直接当成“Cellular 结构有优势”。

### 7.1 主对照

| 模型 | 架构 | 目的 |
|---|---|---|
| C0 | 独立 Cell，无跨 Cell 通信 | 检查通信价值 |
| C1 | 固定局部连接的 EPCLM Core | 核心候选 |
| F0 | Flat Recurrent State Model | 检查分布式组织价值 |
| L0 | 无历史状态的局部上下文模型 | 检查历史状态价值 |

C0 与 C1 应尽量共享网络规模、状态预算、事件编码和 Readout 结构。

F0 的持久状态容量需要与 C1 匹配，并报告参数量及计算预算差异。

L0 使用明确限定的短期输入窗口，不允许将完整历史隐含地提供给模型。

### 7.2 历史参考模型

保留：

- T1 Modern Transformer；
- X4 Native CLM。

这两个模型首先用于历史质量和训练效率参照。

只有在模型参数、数据、训练 Token、优化器、评估集及执行预算得到合理对齐时，才能作严格的科学比较。

不应直接将既有历史 PPL 视为 Core-001 的正式同条件对照结果。

### 7.3 因果消融

对于训练好的 C1，额外执行：

- 清零 State；
- 替换为另一条序列的 State；
- 阻断跨 Cell 消息；
- 仅保留入口 Cell；
- 调整内部迭代次数；
- 保存恢复后继续执行。

这些是机制诊断。

训练好的 C1 在消融后退化，不等于它优于一个从头训练、没有对应组件的对照模型。

因此必须同时报告冻结模型消融与独立训练对照。

---

## 8. 实验阶段

### Gate 0 — 工程正确性

**不需要完整 GPU 训练。**

测试：

- Model 创建、参数计数；
- Forward / Backward；
- 状态张量形状与 Batch 隔离；
- 固定邻域通信；
- 输入 Event 对状态的因果影响；
- 多轮更新；
- Readout 无副作用；
- Snapshot / Restore；
- Reset；
- FP32 参考路径；
- 梯度正确性；
- 无未来信息泄漏；
- Checkpoint 恢复；
- 数据和协议哈希；
- 输出指标与报告生成。

所有核心测试通过后才允许正式训练。

### Gate 1 — 小规模语言学习 Pilot

采用小规模 TinyStories 训练数据。

建议按阶段增加训练量：

`250K → 1M → 2M tokens`

实际训练量应根据 GPU 吞吐量和损失曲线确定，而不是要求每次都跑完整阶段。

目的：

- 验证模型可以稳定学习；
- 测量实际训练速度；
- 检查 State/Gradient 的数值稳定性；
- 判断 C1 与 F0 是否需要终止；
- 建立合理的训练预算。

Pilot 的评估使用开发集，不触碰正式确认集。

如果训练不能稳定下降，停止扩展，进入机制诊断。

### Gate 2 — Core 语言训练

通过 Pilot 后，确定统一训练预算。

建议以既有 10M Token 体系为上限候选，但必须以实际测得的吞吐量评估 Kaggle 成本。

训练 C0、C1、F0、L0。

同一比较中的模型应使用一致的数据预算、评估规则和停止条件。

主要输出：

- 训练 NLL；
- 验证 NLL；
- PPL；
- 每 Token 实际计算成本；
- 模型参数量；
- Persistent State 字节数；
- Peak VRAM；
- 训练吞吐；
- Decode Latency。

### Gate 3 — State / Connection 因果实验

对训练完成的模型进行测试。

**State Test**

构造较早历史不同、最近输入相同的成对序列，使正确预测必须依赖历史。

比较原始 State、清零 State、错误 State 和无状态基线。

**Connection Test**

比较 C1、C0 和通信消融。

要求报告语义任务质量与总体语言 NLL，而不是仅展示热力图。

**Event Test**

同一序列分别连续处理、分段处理、保存恢复后处理。

验证状态和预测的一致性。

### Gate 4 — 独立确认

在开发阶段选定架构和阈值后，冻结：

- 模型实现；
- Source Commit；
- Tokenizer；
- 数据修订及划分；
- 训练配置；
- State 容量；
- 训练预算；
- 评估任务；
- 主指标；
- 科学验收标准；
- 统计分析方法。

随后使用至少三个未参与开发调参的训练 Seed 和独立确认数据。

需要明确：正式 Seed 不应沿用其他已完成实验的冻结确认 Seed。

### Gate 5 — Exploratory Internal Compute

这一阶段不属于 Core 的必要验收。

研究：

- 增加 Evolve 步数后的性能变化；
- 是否出现稳定收敛；
- 不同任务是否需要不同计算深度；
- 在没有新 Event 时能否产生有用计算；
- 内部迭代是否对多步推理任务产生独立收益。

如果只是 Next-Token PPL 随 K 波动，不能宣称模型形成了推理能力。

真正的内部推理验证，需要后续包含可验证答案与计算深度监督或奖励的独立实验。

---

## 9. 科学判定标准

### 9.1 H1 — 持续状态计算成立

**SUPPORTED 条件：**

- Event-State 执行正确；
- State 可保存和恢复；
- 不依赖完整历史 KV；
- 历史依赖任务能够利用此前 State；
- 清零或替换 State 造成可重复且方向合理的退化。

如果模型只依赖最新 Event，则 H1 不支持。

### 9.2 H2 — Cell 通信具有独立价值

**SUPPORTED 条件：**

- C1 对 C0 存在独立确认的质量收益；
- Cell 间通信消融产生因果退化；
- 结果跨 Seed 方向一致；
- 计算预算差异得到明确报告。

建议开发阶段参考至少 1% 的相对 NLL 改善作为实际意义阈值，但正式阈值必须在盲测前根据开发数据和统计功效审查冻结。

如果只有消融退化，而从头训练的 C0 不弱于 C1，应报告“使用了通信，但尚未证明通信提供净收益”。

### 9.3 H3 — 真实语言模型成立

**SUPPORTED 条件：**

- 真实语料 Next-Token Loss 能够稳定下降；
- 独立验证数据显著优于无历史的合理基线；
- 能够完成自回归生成；
- 模型在持续状态条件下正常处理连续语言；
- State 没有数值崩溃或无法恢复的问题。

仅仅训练 Loss 下降不够。

### 9.4 H4 — Cellular 组织具有额外价值

必须将 C1 与 F0 进行对比。

只有在相近容量和计算预算下，C1 在语言质量、历史依赖、状态利用效率或计算成本方面取得可靠收益，才能支持该命题。

若 C1 能训练，但 F0 在所有主要指标上均明显更优，则：

- H1/H3 仍可能成立；
- H4 不支持；
- 不得宣称 Cellular Organization 已展现优越性。

### 9.5 统计原则

- 同一数据样本的模型比较应使用配对设计。
- 对语言数据按 Story / Document 聚类，而不是将重叠 Token 窗口全部视为独立样本。
- 主指标报告差值、相对变化、95% 置信区间和训练 Seed 间差异。
- 采用预先指定的 Bootstrap 或等价统计方法。
- 开发结果与正式确认结果严格区分。
- 不得在观察正式结果后修改阈值并重新解释为预注册成功。

---

## 10. 工程与性能指标

除模型质量外，必须记录：

| 指标 | 说明 |
|---|---|
| Resident Parameters | 模型总参数量 |
| State Bytes | 持续状态实际占用 |
| Snapshot Bytes | 序列化后大小 |
| FLOPs / Event | 每事件实际计算量的分析估计 |
| Kernel Launch Count | CUDA Kernel 数量 |
| Decode Latency | 逐 Token 推理延迟 |
| Training Tokens/s | 实际训练吞吐 |
| Peak VRAM | 训练显存峰值 |
| State Norm | 状态范数及演化稳定性 |
| Delta Norm | 每轮 Cell 更新幅度 |
| Communication Contribution | 跨 Cell 消息的消融效应 |
| K Sensitivity | 不同迭代次数下的质量与计算量 |

不能把参数激活比例等同于实际加速，也不能把 FLOPs 下降直接解释为 Latency 下降。

既有 EPCLM-001 经验表明，增量和事件执行性能依赖具体实现、精度、设备及工作负载。

---

## 11. Inspector 与可观察性

Core 必须提供研究级状态检查能力。

最小 Inspector：

- State Norm Heatmap；
- Cell Delta Heatmap；
- Local Communication Heatmap；
- 每轮输出 NLL / Probability；
- 不同 K 的演化轨迹；
- State Reset / Swap 的预测对照；
- Snapshot/Restore 一致性报告。

Inspector 默认只读取运行时状态，不改变模型行为。

允许记录指定事件的状态轨迹，但不应该默认保存每个训练 Token 的完整状态，避免极高的日志和存储开销。

特别注意：

**Cell 活跃度和状态变化不等同于可解释语义。**

不能仅凭热力图把某个 Cell 命名为“人物记忆”或“推理 Cell”。

---

## 12. Kaggle 运行与预算

### 12.1 资源约束

参考现有 Kaggle 执行约束：

| 项目 | 预算 |
|---|---:|
| Notebook 硬上限 | 7,200 秒 |
| 单次训练和验证执行目标 | 5,400 秒 |
| 结果发布预留 | 600 秒 |
| 安全与环境余量 | 1,200 秒 |

这些是单次运行的工程规划值，不意味着所有模型必须在同一次运行中训练完成。

### 12.2 Pilot 必须先测量

在实际 GPU 上测量：

- 前向和反向耗时；
- Token/s；
- 每个 K 的成本；
- Checkpoint 保存与恢复；
- Validation 耗时；
- Snapshot 与 Inspector 开销；
- 发布耗时。

之后据此决定每次运行可完成多少训练 Token。

不得因时间不足而静默降低不同模型的训练预算，然后将结果作为公平比较。

### 12.3 断点恢复

Checkpoint 必须包括：

- 模型权重；
- Optimizer；
- Scheduler；
- GradScaler；
- Python / NumPy / PyTorch / CUDA RNG；
- 训练数据采样顺序；
- 已处理 Token 数；
- 训练流对应的 Cellular State；
- 训练片段位置；
- Model/Protocol/Data 哈希。

任何恢复必须验证 Source 与配置身份。

运行超时或被中断时，保留进度，不能把部分结果标记为完整。

---

## 13. 代码组织

建议建立独立研究单元：

```text
research/projects/cellular-language-models/
  experiments/
    epclm-core-001/
      README.md
      SPEC.md
      protocol.json
      config/
      notebook.ipynb
      implementation/
        model.py
        state.py
        communication.py
        update.py
        runtime.py
        trainer.py
        baselines.py
        evaluate.py
        inspector.py
        checkpoint.py
        publish.py
      tests/
      reports/
```

实际布局应遵循仓库现有的 Research Project 规范，复用已有公共组件，不重复构建通用训练和发布基础设施。

### 模块边界

**Model**

只负责可微计算。

**Runtime**

负责 State 生命周期、事件注入、内部演化、Readout 和 Snapshot。

**Trainer**

负责数据流、BPTT、优化器和训练恢复。

**Evaluator**

负责科学指标、对照模型和消融。

**Inspector**

负责状态诊断与可视化。

**Publisher**

负责实验身份、哈希、指标和结果发布。

所有模块使用同一套状态转移语义，避免训练和推理分别实现两套不一致的更新规则。

---

## 14. 实施顺序

### 阶段 A — 实现 Core

交付：

- `State`
- `Inject`
- `LocalCommunication`
- `SharedUpdate`
- `Evolve`
- `Readout`
- `Snapshot/Restore`
- 最小 CLI

完成 Gate 0。

### 阶段 B — 训练基础设施

交付：

- TinyStories 数据适配；
- 可恢复训练；
- Truncated BPTT；
- Validation；
- GPU Pilot；
- 基本 Inspector。

不得在训练基础设施尚不能恢复的情况下启动长时间实验。

### 阶段 C — 基础模型训练

先训练 C1 和 F0。

如果 C1 无法稳定学习，但 F0 能正常学习，优先检查：

- 注入方式；
- 残差更新稳定性；
- 梯度传播；
- Cell 通信；
- Readout；
- K 和训练截断长度。

不得直接增加十倍参数规模试图掩盖问题。

在核心模型能够学习后，训练 C0/L0，并完成主对照。

### 阶段 D — 科学分析

完成：

- H1–H4 的开发判断；
- State/Connection 消融；
- NLL/PPL 曲线；
- State 演化可视化；
- 实际成本测量；
- 架构失败模式分析。

### 阶段 E — 正式确认与发布

在冻结协议并完成独立确认后，发布：

- 研究报告；
- 模型配置；
- 训练和验证指标；
- 完整 Source / Data / Protocol Identity；
- 统计结果；
- 模型资产引用；
- 结果可视化；
- 已知限制；
- 明确的 SUPPORTED / NOT SUPPORTED / INCONCLUSIVE 分类。

---

## 15. 停止规则和失败处理

### 情形一：Core 无法学习语言

停止完整训练。

首先诊断数值稳定性、训练数据、目标对齐、梯度、注入与 Readout。

如果修复需要修改实验模型，应形成新版本，不覆盖原始实验记录。

### 情形二：State 没有历史因果作用

H1 不支持。

停止对持久语言状态优势的宣传，检查训练目标是否允许模型退化为最近 Token 预测器。

### 情形三：Cell 通信没有收益

H2 不支持。

不必废弃整个 Runtime，但不能声称 Cellular Organization 的独立价值已经得到验证。

后续可以研究 Local Attention 或 Adaptive Perception，但必须以新的实验变量进行验证。

### 情形四：模型可训练，但明显落后 Flat RNN

H3 可能支持，H4 不支持。

应评估 Cellular 结构是否提供其他实际收益。

如果没有，则需要重新审视固定拓扑而不是继续扩大模型。

### 情形五：增加 Evolve 次数没有收益

只影响 H5。

不否定基本事件驱动状态计算能力。

可以保留 `evolve()` 接口，未来在具备可验证推理目标后重新研究计算深度。

### 情形六：Kaggle 超时

结果标记为 INCOMPLETE / INCONCLUSIVE。

允许在相同实验身份下恢复，不得把未训练完成的条件解释为架构失败。

---

## 16. 不纳入首版的后续研究方向

| 方向 | 可能的新实验 |
|---|---|
| Local Content Attention | 比较固定邻域投影和局部 Q/K/V |
| Adaptive NCA | 学习动态感知范围 |
| Path-Conditioned Execution | 根据当前状态选择更新路径 |
| Hierarchical Cellular | 分层组织与跨组通信 |
| Stochastic / Asynchronous NCA | 随机更新与长迭代稳定性 |
| Cellular Reasoning | 无新 Event 条件下的可验证推理 |
| Adaptive Compute | 学习决定 Evolve 次数 |
| Cell Growth | 可控的状态容量扩展 |
| Continual Parameter Learning | 独立参数更新、安全提交与历史保持 |

其中前三项的扩展应尽量复用 Core-001 的 State、Runtime 和 Trainer，而不是建立全新的模型运行规范。

---

## 17. 最终交付清单

工程交付：

- [ ] 可训练的 EPCLM Core 模型
- [ ] 完整 State / Event / Connection / Update 抽象
- [ ] 推理与状态恢复 Runtime
- [ ] TinyStories 训练入口
- [ ] 可恢复的 Kaggle Notebook
- [ ] C0、C1、F0、L0 对照实现
- [ ] Inspector 与状态演化可视化
- [ ] 自动化单元测试及数值正确性测试
- [ ] 模型和结果发布机制

科学交付：

- [ ] H1 State Causality
- [ ] H2 Cellular Communication
- [ ] H3 Language Learning
- [ ] H4 Cellular Organization Value
- [ ] 探索性 H5 Internal Computation
- [ ] 训练质量与计算成本报告
- [ ] 多 Seed 独立确认
- [ ] 失败模式与适用边界
- [ ] 最终研究结论

所有尚未执行的门槛默认为 NOT EVALUATED，不能提前标记为 PASS。

---

## 18. 首版验收定义

**EPCLM Core-001 的最小机制成立，当且仅当：**

1. 网络可以接收连续事件，并通过共享规则更新持久的 Cellular State。
2. State 在连续事件处理中具有经过实验证实的历史因果作用。
3. Cell 间通信对所定义的计算任务具有独立的、可重复的贡献。
4. 模型能够从这些持续演化的状态中学习真实语言序列的预测任务。
5. 以上结果可以在冻结协议和独立样本上复现。

即使这些条件全部满足，也只代表：

`EPCLM_CORE_MECHANISM_SUPPORTED`

不自动代表：

`CELLULAR_SUPERIOR_TO_RNN`

`CELLULAR_SUPERIOR_TO_TRANSFORMER`

`GENERAL_REASONING_SUPPORTED`

`LIFELONG_LEARNING_SUPPORTED`

上述结论必须由各自独立的实验提供证据。

---

## 19. 最终架构原则

EPCLM Core-001 的核心可以凝练为三个操作：

```text
Inject(Event)
       |
       v
Evolve(Cellular State)
       |
       v
Read(Computed State)
```

其中：

**State 是计算载体。**

Memory、工作状态和推理中的中间表示，可以作为状态演化所学习到的功能。

**Connection 是计算协作机制。**

固定邻域只是第一版；未来可以进一步学习通信关系和计算路径。

**Event 是持续计算的输入机制。**

外部事件触发变化，而内部计算可以独立继续。历史输入不需要在每次计算时重新作为完整上下文提供。

核心目标不是把 Transformer 中的 Attention 换成另一种 Attention，也不是建立一个更复杂的记忆数据库。

我们的目标是验证：

> 一个持续存在、能够通过局部交互自主演化的 Cellular Network，是否能够成为语言模型的基本计算结构。

只要这一机制获得可靠证据，EPCLM 就拥有一个可以继续研究 Adaptive Communication、Path-Conditioned Computation、Cellular Reasoning 和长期状态演化的统一基础。

---

## 20. 参考资料

**内部研究资料**

- [Native CLM Architecture Lab](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/native-clm-v0/implementation/README.md)
- [MiniCells Evidence Map](https://github.com/ArcheLabs/mini-cells/blob/main/research/governance/MINICELLS_EVIDENCE_MAP.md)
- [Constructive CLM-004 Results](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/constructive-clm-004-model-level-multicell-computation/artifacts/RESULTS.md)
- [Constructive CLM-005 Results](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/constructive-clm-005-scaffold-removal/artifacts/RESULTS.md)
- [Experiment 006 Path-First](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/006-path-first-clm/README.md)

**外部研究**

- [Reasoning with Neural Cellular Automata — arXiv:2609.36126](https://arxiv.org/abs/2609.36126)
- [Reasoning NCA 官方代码](https://github.com/paradigms-of-intelligence/reasoning-nca)
- [2D Spatial Reasoning with Adaptive Neural Cellular Automata — arXiv:2610.08518](https://arxiv.org/abs/2610.08518)

**文档状态：DRAFT v0.1。** 本文给出了拟实施架构和开发实验方案。正式 GPU 训练前仍需完成源码、训练预算、数据分区、统计阈值与独立 Seed 的冻结登记。本设计不修改或重新解释任何历史实验的正式结论。

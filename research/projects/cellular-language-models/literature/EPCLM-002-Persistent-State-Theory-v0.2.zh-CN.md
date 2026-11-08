# EPCLM-002 — Persistent State Theory

**项目：** MiniCells / Cellular Language Models  
**文档版本：** 0.2 — Research Update  
**日期：** 2026-11-08  
**状态：** 理论研究纲领与文献／实验更新；不是实现规范，也不预设具体 Cell 网络、神经算子或 GPU Kernel。
**更新记录（2026-11-08）：** 增补 SSM、Titans Neural Memory、Local / Sliding Window Attention 的组合研究；加入 Titans Cell 可移植性的证据边界；增加约 10M 参数的混合架构消融方案。原有理论定义、历史实验记录和证据等级保留。

---

## 摘要

EPCLM-002 研究一个核心假设：

> **持久状态（Persistent State）能否取代历史 Token Attention，成为语言模型跨时间保存和访问信息的主要机制？**

这不是简单地把 Transformer 的 KV Cache 改成一个状态张量，也不是给现有 CLM 增加长期记忆模块。研究目标是重新定义序列模型如何表示历史、如何写入新信息、如何访问相关信息，以及如何在有限计算预算下持续演化。

本文件提出一个与具体实现无关的理论框架。它将问题分解为四个相互依赖的能力：

1. **State Sufficiency：** 当前状态是否保留了未来预测所需的历史信息？
2. **State Formation and Transition：** 新输入如何被吸收、合并、修正或遗忘？
3. **State Retrieval and Routing：** 如何在不扫描完整历史 Token 序列、也不扫描全部 Cell 的情况下，找到相关状态？
4. **State-based Generation：** 如何仅依赖持久状态和有限的当前工作上下文完成自回归预测？

研究不预设固定大小的状态一定足够。对任意长、任意内容的历史，固定容量状态不可能无损保存所有信息。EPCLM 必须研究的是：在明确的数据分布、未来任务、记忆预算和允许误差下，怎样构造足够充分、可更新、可寻址且计算有效的状态。

本文区分三类内容：

- **已有研究事实：** 已发表论文或公开预印本实际提出和验证的机制。
- **EPCLM 理论提案：** 本项目建议研究的抽象、定义和假设。
- **开放问题：** 目前没有得到理论证明或实验验证的部分。

---

# 1. 研究目标与边界

## 1.1 研究目标

建立一种语言序列计算范式，使模型在处理历史输入时，不必永久保留并反复检索完整的历史 Token 表示，而是将与未来任务相关的信息写入持久状态。

理想情况下，模型在时刻 \(t\) 的预测满足：

\[
P_\theta(x_{t+1}\mid x_{1:t})
\approx
P_\theta(x_{t+1}\mid S_t, W_t)
\]

其中：

- \(x_{1:t}\)：截至当前时刻的历史输入。
- \(S_t\)：跨 Token、跨输入片段持续存在的状态。
- \(W_t\)：有限的当前工作上下文，例如当前 chunk、最近少量 Token 或临时计算结果。

核心要求是：随着历史长度增长，模型不需要继续保留完整的历史 Token KV；历史信息应主要由 \(S_t\) 承担。

## 1.2 不在本阶段解决的问题

EPCLM-002 暂时不规定：

- Cell 必须是固定数量还是动态增删。
- Cell State 必须是向量、矩阵、MLP 参数、张量网格还是其他结构。
- 必须采用 NCA、SSM、RNN、图网络或其他具体算子。
- 必须彻底禁止所有 Attention。
- 必须使用某种特定 GPU Kernel 或内存布局。
- 必须立即达到大模型的语言能力或吞吐量。

这些都应由理论分析和实验决定。

## 1.3 与旧 CLM 的边界

旧 X3/X4 已有 Token-associated Hidden State、重复迭代更新、局部因果 Attention 和分阶段 KV Cache。它们证明了循环状态计算和增量执行的一些工程性质，但状态仍然依附于 Token 位置，历史信息仍通过 KV 参与后续计算。

EPCLM-002 的关键变化不是“再增加几次迭代”，而是改变历史信息的表示主体：

\[
\text{Token-indexed intermediate state}
\quad\longrightarrow\quad
\text{Token-independent persistent state}
\]

旧 CLM 保留为历史基线，不作为新理论必须继承的架构约束。

---

# 2. 首要概念澄清：替换 Attention 不等于只增加 State

## 2.1 Attention 实际承担了什么

标准因果 Self-Attention 可写为：

\[
o_t =
\sum_{i=1}^{t}
\alpha_{t,i}v_i
\]

\[
\alpha_{t,i} =
\operatorname{softmax}_{i\leq t}
\left(
\frac{q_t k_i^\top}{\sqrt d}
\right)
\]

它至少承担四类功能：

1. **历史表示保存：** Transformer 通常将历史各层的 Key/Value 保存在 KV Cache 中。
2. **内容寻址：** 当前 Query 根据 Key 选择历史位置。
3. **信息聚合：** 将多个历史 Value 组合成当前输出。
4. **上下文交互：** 让不同 Token 的表示在计算过程中相互影响。

因此，Attention 并不是单一的“记忆模块”。它同时包含历史存储形式和动态检索/聚合运算。

## 2.2 State 只解决了存储问题的一部分

如果模型将历史 KV 压缩成一个向量或矩阵状态，但仍然没有解决如何从状态中找到相关信息，那么模型只是换了一种存储格式。

反过来，如果模型将历史信息存入一百万个 Cell，但每个新 Token 都要对全部 Cell 计算相似度并执行 Softmax，那么它可能只是把 Token Attention 改成了 Cell Attention。

因此，EPCLM 的研究对象应当是一个完整机制：

\[
\boxed{
\text{Persistent Memory}
+
\text{Selective Retrieval}
+
\text{State Transition}
+
\text{Readout}
}
\]

这里的目标不是在术语上消灭 Attention，而是取消对**完整历史 Token 集合的密集动态检索**，并避免把相同的全局检索成本原样转移到 Cell 集合。

## 2.3 两个不同强度的研究主张

必须区分：

**主张 A：Historical-Attention-Free**

不再对无界历史 Token/KV 执行 Attention。模型允许当前 chunk 内部使用有限局部 Attention。

**主张 B：Attention-Free State Retrieval**

状态读取本身也不采用 Query-Key 全候选相似度和 Softmax，而是通过稀疏索引、显式地址、结构化路由或其他机制完成。

A 是 EPCLM 的最低核心目标；B 是更强的研究目标，不能预先假定一定可行或一定优于 A。

---

# 3. 形式化问题定义

## 3.1 历史、未来查询与目标

令：

\[
H_t = X_{1:t}=(x_1,\ldots,x_t)
\]

表示历史输入。

未来任务不应只限定为紧邻的下一个 Token。令：

- \(Q\)：未来查询、当前生成需求或任务条件。
- \(Y\)：需要预测的目标，可以是下一个 Token、后续片段、事实答案或任务结果。
- \(S_t=f_\theta(H_t)\)：由历史形成的持久状态。
- \(W_t\)：有限的短期工作上下文。

模型希望满足：

\[
P(Y\mid H_t,Q)
\approx
P_\theta(Y\mid S_t,W_t,Q)
\]

注意：状态在历史处理时通常不知道未来的具体 Query。因此，状态形成机制必须保留对目标查询分布有用的信息，而不是只针对一个已知问题写入答案。

## 3.2 状态的因果更新

持久状态必须能够在线演化：

\[
S_0 = s_{\mathrm{init}}
\]

\[
\boxed{
S_t = F_\theta(S_{t-1},x_t)
}
\]

对 chunk 输入 \(X_{t:t+c}\)，也可以定义：

\[
S_{t+c}=F_\theta^{(c)}(S_t,X_{t:t+c})
\]

其中 \(F_\theta^{(c)}\) 是对一个输入片段的状态转移。

该定义只规定因果关系，不规定 \(F_\theta\) 的内部结构。它可以包含写入、路由、局部通信、状态更新和状态维护。

## 3.3 Cell 化状态的抽象

如果采用 Cell 表示，持久状态可以写为：

\[
S_t=\{C_i^t\}_{i=1}^{N_t}
\]

一个 Cell 可以抽象为：

\[
C_i^t=(\mathrm{id}_i,z_i^t,\mu_i^t)
\]

其中：

- \(\mathrm{id}_i\)：稳定身份或地址。
- \(z_i^t\)：可学习的内容状态。
- \(\mu_i^t\)：可选元数据，例如类型、关系、时间、版本或生命周期信息。

不要求每个 Cell 都有固定的 Key，也不要求 Cell 数量固定。关键条件是：Cell 的身份和状态不等同于某个 Token 的位置表示。

---

# 4. 核心理论一：State Sufficiency

这是 EPCLM 最重要的数学问题。

## 4.1 预测充分性定义

对给定的历史分布和未来任务分布，如果：

\[
P(Y\mid H_t,W_t,Q)=P(Y\mid S_t,W_t,Q)
\]

则称 \(S_t\) 对目标 \(Y\) 在给定 \(Q\) 条件下是历史 \(H_t\) 的预测充分状态（predictively sufficient state）。

实际模型允许近似充分。定义任务风险：

\[
\boxed{
\mathcal{R}_{\mathrm{suff}}
=
\mathbb{E}_{H_t,W_t,Q}
\left[
D_{\mathrm{KL}}
\left(
P(Y\mid H_t,W_t,Q)
\parallel
P_\theta(Y\mid S_t,W_t,Q)
\right)
\right]
}
\]

目标是在状态预算约束下使该风险足够低。

若使用真实的条件分布 \(P(Y\mid S_t,W_t,Q)\)，则有：

\[
I(Y;H_t\mid S_t,W_t,Q)
=
\mathbb{E}
\left[
D_{\mathrm{KL}}
\left(
P(Y\mid H_t,W_t,Q)
\parallel
P(Y\mid S_t,W_t,Q)
\right)
\right]
\]

因此：

\[
\boxed{
I(Y;H_t\mid S_t,W_t,Q)=0
}
\]

是预测充分性的一个等价表达。

它的含义是：在已经知道持久状态、允许保留的有限工作上下文和未来查询后，额外查看更早的原始历史，不再为目标 \(Y\) 提供信息。若研究一个不保留任何局部上下文的更强模型，则可令 \(W_t=\varnothing\)。

这不是说状态能够重建历史，也不是说历史中每个 Token 都被保存了。它只要求状态保留对目标任务有用的信息。

## 4.2 为什么必须限定任务分布

同一段历史可能支持很多不同的未来问题。

例如：

> Alice lives in London. She moved there in 2020.

未来可能询问：

- Alice 住在哪里？
- 她什么时候搬到那里？
- 句子中出现了哪些词？
- “there” 指代什么？
- 将原文逐字复述。

一个压缩状态可能足以回答前两个问题，却无法逐字重建原文。

因此，State Sufficiency 必须明确未来查询集合 \(\mathcal Q\)：

\[
\mathcal{R}_{\mathrm{suff}}(\mathcal Q)
=
\mathbb E_{Q\sim\mathcal Q}
D_{\mathrm{KL}}
\left(
P(Y\mid H,Q)
\parallel
P_\theta(Y\mid S,Q)
\right)
\]

状态的“充分”不是绝对属性，而是相对于目标任务分布、查询分布和误差容忍度的属性。

**EPCLM 不应声称状态能无损保存所有历史；它应证明状态对预先定义的任务族足够充分。**

## 4.3 固定容量状态的理论限制

假设历史 Token 来自大小为 \(|\mathcal V|\) 的词表，长度为 \(T\)。

若要求对任意历史序列进行无损、可逆保存，至少需要：

\[
B_{\mathrm{history}}
\geq T\log_2|\mathcal V|
\]

比特。

如果状态经过有限精度编码后只有固定 \(B\) 比特，那么它最多表示 \(2^B\) 种不同状态。对未量化的连续状态，必须先指定数值精度、可区分精度或有效编码长度，才能讨论实际容量。随着 \(T\) 无限增长，所有可能的历史序列数量会超过状态可表示数量。因此，不可能对任意历史做到无损保存。

这不是某个 Cell 设计的缺陷，而是有限状态容量的基本限制。

但语言建模通常不要求无损重建全部历史。自然语言存在统计规律、冗余和任务相关性，因此有可能将大量历史压缩为较小的预测充分表示。

这正是需要实验验证的部分。

## 4.4 预测信息瓶颈

可以将状态学习定义为一个受预算约束的优化问题：

\[
\boxed{
\min_\theta
I(S_t;H_t)
+
\beta\,
\mathcal R_{\mathrm{suff}}
}
\]

其中：

- \(I(S_t;H_t)\)：状态携带的历史信息量，作为压缩程度的理论代理。
- \(\mathcal R_{\mathrm{suff}}\)：状态对未来任务的预测损失。
- \(eta\)：信息保留与任务性能之间的权衡系数。

实际实现不一定直接计算互信息。可以使用状态字节数、Cell 数量、量化精度、状态熵估计、任务损失和容量曲线作为代理指标。

更直接的工程形式是：

\[
\min_\theta
\mathcal L_{\mathrm{task}}
+
\lambda_B\,\mathrm{MemoryBytes}(S_t)
\]

并同时约束读取和更新成本。

## 4.5 固定状态与增长状态必须分别研究

EPCLM 不应预先规定所有 Cell State 都必须固定容量。

至少有两种理论路线：

**Fixed-capacity state**

\[
\mathrm{Memory}(S_t)\leq B
\]

优点是显存和推理内存有明确上限；缺点是必须进行有损压缩、选择性遗忘或状态合并。

**Expandable state**

\[
B_t \leq B_0 + g(\mathrm{relevant\ information\ in}\ H_t)
\]

状态容量可以随着真正需要保留的信息增长。它不必随着每个 Token 线性增长，但也不能声称无条件恒定内存。

Expandable state 需要解决寻址、索引、碎片、生命周期、压缩和长期存储成本。

这两条路线应当作为独立实验变量，而不是在理论阶段提前合并。

---

# 5. 核心理论二：State Formation and Transition

状态系统的关键不只是保存信息，还要能正确处理新信息。

## 5.1 状态形成的四个动作

对于新输入 \(x_t\)，系统至少需要决定：

1. **Create：** 是否需要创建新的状态实体。
2. **Write：** 将新信息写入哪些已有状态。
3. **Update：** 如何将新信息与已有信息融合。
4. **Retain / Forget：** 哪些旧信息继续保留，哪些需要衰减、压缩或移除。

这些动作可以由同一模型完成，也可以由不同模块完成。本阶段不作架构承诺。

## 5.2 抽象状态转移

令 \(E_t\) 表示当前输入产生的事件或写入请求，令 \(A_t\) 为被选中的状态实体：

\[
A_t = \mathrm{Route}_\theta(E_t,S_t,\mathcal I_t)
\]

其中 \(\mathcal I_t\) 是可选的状态索引。

对活跃 Cell：

\[
z_i^{t+1}
=
U_\theta
\left(
z_i^t,
E_t,
M_i^t,
\Delta t,
\mu_i^t
\right),
\quad i\in A_t
\]

对未激活 Cell：

\[
z_i^{t+1}=z_i^t
\]

这里的“未更新”是执行语义，不代表系统已证明路由正确。若重要信息没有进入活跃集合，状态系统可能永久丢失它。

## 5.3 新信息、重复信息与冲突信息

状态转移至少要处理三种情况：

- **重复：** 新输入重复已有事实，系统应避免无意义地不断复制。
- **补充：** 新输入补充已有状态，例如增加时间、来源或关系。
- **冲突或修正：** 新输入与旧状态不一致，例如 Alice 从 London 搬到 Paris。

对于事实型记忆，不能简单将两个表示平均。系统可能需要保留时间、来源、置信度和版本：

\[
m=(\mathrm{subject},\mathrm{predicate},\mathrm{object},
\mathrm{time},\mathrm{source},\mathrm{confidence})
\]

这只是一个可能的结构化表示例子，不是 EPCLM 对 Cell 数据结构的规定。

对自然语言的隐含语义、风格、语用和推理状态，可能需要不同于显式事实记录的表示。

## 5.4 稳定性与可塑性

状态系统需要同时具备：

- **Plasticity：** 新信息能够改变状态。
- **Stability：** 新信息不应无意破坏已有重要信息。

可将其视为一个多目标优化问题：

\[
\min
\mathcal L_{\mathrm{new}}
+
\lambda_{\mathrm{retain}}\mathcal L_{\mathrm{retention}}
+
\lambda_{\mathrm{interf}}\mathcal L_{\mathrm{interference}}
\]

其中：

- \(\mathcal L_{\mathrm{new}}\)：对新信息的学习或预测损失。
- \(\mathcal L_{\mathrm{retention}}\)：对旧任务、旧事实的保留损失。
- \(\mathcal L_{\mathrm{interference}}\)：无关写入对已有状态造成的损害。

注意，状态变化小不等于稳定性好。遇到真实修正时，模型必须能够有意改变旧状态。

---

# 6. 核心理论三：State Retrieval and Routing

这是决定 EPCLM 是否真正摆脱历史 Attention 的第二个关键问题。

## 6.1 为什么状态本身不够

假设模型将 100 万个 Cell 存储在 GPU 显存中。

若每次输入都计算 Query 与所有 Cell Key 的相似度：

\[
s_i=q^\top k_i
\]

然后执行：

\[
r=\sum_{i=1}^{N}\operatorname{softmax}(s)_i z_i
\]

那么计算仍然是对全部 Cell 的内容寻址聚合。

这不是历史 Token Attention，但它仍然是全局密集检索。若 N 很大，成本可能成为新的瓶颈。

因此，EPCLM 必须把“记忆表示”和“记忆索引”分开研究。

## 6.2 三种候选读取方式

### A. Dense content-addressed read

对全部状态进行相似度匹配。

优点：简单、可微、容易训练。

缺点：通常随 Cell 数量线性增长；本质上保留了全候选 Attention 式检索。

它适合作为正确性基线，不应自动成为最终机制。

### B. Indexed sparse retrieval

通过索引结构先找到候选 Cell，再读取少量候选：

\[
C_q=\mathrm{IndexLookup}(q,\mathcal I_t),
\quad |C_q|=K\ll N
\]

然后：

\[
r_t=\mathrm{Aggregate}(C_q)
\]

索引候选可以来自：

- 显式地址或符号键。
- 哈希或分层索引。
- 学习型路由器。
- 稀疏图邻接关系。
- 近似最近邻索引。

其理想成本可能接近：

\[
O(\mathrm{IndexCost}(N)+Kd)
\]

但不能把它无条件写成 \(O(1)\) 或 \(O(\log N)\)。具体复杂度取决于索引结构、更新频率、数据分布、候选召回率和最坏情况。

### C. Structured / graph traversal

状态实体之间维护关系或局部连接。查询先进入少量起始 Cell，再沿相关关系传播：

\[
C_0\rightarrow C_1\rightarrow\cdots\rightarrow C_h
\]

如果任务具有实体关系、事件依赖或层级结构，这可能比全局相似度检索更有效。

但图遍历也可能发生错误传播、路径爆炸或遗漏。它必须与索引检索基线比较。

## 6.3 路由正确性必须量化

稀疏路由的核心指标不只是活跃 Cell 比例，而是必要信息是否被覆盖。

定义：

\[
\mathrm{Recall@K}
=
\frac{
\text{被路由选中的必要记忆项}
}{
\text{全部必要记忆项}
}
\]

也需要测量：

- Recall@K 与任务准确率的关系。
- 路由器自身的时间、显存和能耗。
- 索引更新成本。
- 候选集合大小 K。
- 错误路由后的恢复能力。
- 高负载和分布外输入时的最坏情况。

如果路由本身扫描所有 Cell，那么稀疏更新并没有带来系统级稀疏性。

## 6.4 读取是否必须使用 Attention？

不必须，但读取不能凭空发生。

只要系统需要从多个候选信息源中选择相关信息，就必须存在某种选择机制。

它可以是显式地址、索引、规则、图遍历、学习路由或内容相似度。

若最后仍对候选 Cell 使用 Softmax 加权聚合，那么它在数学上可能仍属于 Attention 类操作，只是作用范围缩小了。

EPCLM 应当区分：

- **不再访问历史 Token 的 Attention。**
- **不再对全部 Cell 做密集 Attention。**
- **完全不用 Attention 运算。**

这三个目标强度不同。第一项是必要目标，后两项需要由实验决定。

---

# 7. 核心理论四：Event-driven Sparse State Evolution

## 7.1 基本定义

令全部状态实体为 \(\mathcal C_t\)，当前事件激活的实体为：

\[
A_t\subseteq \mathcal C_t
\]

稀疏更新率：

\[
\rho_t=\frac{|A_t|}{|\mathcal C_t|}
\]

理想情况下：

\[
\rho_t\ll 1
\]

系统只对活跃状态执行昂贵更新，未活跃状态保持不变。

## 7.2 事件不等于 Token

事件可以来自：

- 外部新输入。
- 已有状态之间的消息。
- 状态变化触发的后续处理。
- 定时或生命周期事件。
- 需要重新验证的冲突或依赖。

因此，Token 是事件的可能来源，但 Event 不应被定义为 Token 的别名。

## 7.3 事件传播与终止

若 Cell 更新会产生新事件，则需要定义：

\[
E_t\rightarrow A_t\rightarrow
\{z_i^{t+1}\}\rightarrow E_{t+1}^{\mathrm{internal}}
\]

必须防止：

- 无限事件循环。
- 重复消息风暴。
- 状态振荡。
- 事件丢失。
- 不同更新顺序导致不可控结果。

理论阶段至少要规定事件的因果顺序、去重语义、冲突处理和最大传播边界。

## 7.4 稀疏计算的系统级条件

稀疏更新只有在以下成本都受控时才有意义：

\[
C_{\mathrm{total}} =
C_{\mathrm{encode}}
+C_{\mathrm{route}}
+C_{\mathrm{read}}
+C_{\mathrm{communicate}}
+C_{\mathrm{update}}
+C_{\mathrm{writeback}}
+C_{\mathrm{index}}
\]

不能只测量 \(C_{\mathrm{update}}\)。

如果每次更新一个 Cell 都需要大量小 Kernel、随机访存或 CPU-GPU 同步，理论稀疏性可能无法转化为实际速度。

---

# 8. State-based Autoregressive Generation

## 8.1 当前输入与历史状态分离

建议在理论上区分：

- \(S_t\)：持久状态，负责跨输入片段保存历史信息。
- \(W_t\)：有限工作状态，负责当前片段内的局部计算。
- \(\theta\)：模型长期参数，保存训练得到的一般语言知识和更新规则。

这三个部分不能混为一谈。

## 8.2 生成方程

可以定义：

\[
r_t=\mathrm{Read}_\theta(S_t,W_t,q_t)
\]

\[
p_\theta(x_{t+1}\mid S_t,W_t)
=
\mathrm{Softmax}
\left(
G_\theta(r_t,W_t)
\right)
\]

生成新 Token 后：

\[
S_{t+1}=F_\theta(S_t,W_t,x_{t+1})
\]

短期工作状态可随 chunk 滑动或清空，但持久状态继续存在。

严格的 Historical-Attention-Free 测试要求：清除历史 Token 和历史 KV 后，模型仍能依靠持久状态完成后续预测。

## 8.3 不应预先排斥局部 Attention

当前 chunk 内部的局部 Attention 可能是高效的短期工作机制。

EPCLM 的核心主张是历史信息不再依赖无界 Token Attention，而不是所有层、所有阶段、所有输入都不能出现 Attention。

实验必须分别报告：

- 跨历史 Token 的读取成本。
- 当前 chunk 内部的局部计算成本。
- 持久状态读取成本。

---

# 9. 训练理论：如何让模型学会写入有用状态

持久状态不能只靠人工规则建立。写入、更新和读取必须通过训练形成协同机制。

## 9.1 联合训练目标

建议先研究以下抽象目标：

\[
\mathcal L =
\mathcal L_{\mathrm{LM}}
+\lambda_q\mathcal L_{\mathrm{query}}
+\lambda_u\mathcal L_{\mathrm{update}}
+\lambda_r\mathcal L_{\mathrm{retention}}
+\lambda_s\mathcal L_{\mathrm{sparsity}}
+\lambda_b\mathcal L_{\mathrm{budget}}
\]

各项含义：

- \(\mathcal L_{\mathrm{LM}}\)：普通自回归语言建模损失。
- \(\mathcal L_{\mathrm{query}}\)：从状态回答未来查询的损失。
- \(\mathcal L_{\mathrm{update}}\)：新信息写入、事实更新和冲突处理损失。
- \(\mathcal L_{\mathrm{retention}}\)：旧信息保留和抗干扰损失。
- \(\mathcal L_{\mathrm{sparsity}}\)：限制不必要的激活和状态更新。
- \(\mathcal L_{\mathrm{budget}}\)：限制状态容量、路由和读取成本。

这些项不是最终固定配方。必须通过消融实验判断每项是否必要。

## 9.2 训练样本必须包含“写入—隔离—读取”

普通 next-token training 可能让模型依赖当前可见的历史 Token，而没有真正学会写入持久状态。

因此，训练样本应形成明确的阶段：

1. 提供 Context。
2. 要求模型形成状态。
3. 移除原始 Context 和历史 KV。
4. 提供 Query。
5. 要求模型仅依靠状态作答。

这使训练目标与研究假设直接对应。

## 9.3 训练时的历史访问限制

如果训练时允许模型通过完整历史 Attention 作答，推理时却要求它只依赖 State，那么模型可能根本没有学会写入足够的信息。

因此，需要设计 Context Removal / Memory-only Query 训练阶段，确保模型在读取时确实无法访问原始历史。

## 9.4 状态更新的梯度路径

可研究：

- 完整跨时间反向传播。
- 分块或截断 BPTT。
- 通过辅助预测目标训练写入器。
- 测试时优化记忆状态。
- 状态快照、回放和蒸馏。

不同训练方式会改变状态容量、写入成本和长期稳定性，不能假定某一种方式最优。

---

# 10. 与已有研究的关系

本节总结截至 2026-10-30 可核验的相关工作。论文提出的机制与 EPCLM 目标并不完全相同；以下明确区分其贡献和未解决的问题。

| 研究 | 已有贡献 | 与 EPCLM 的关系及边界 |
|---|---|---|
| Mamba (2023) | 输入依赖的选择性状态空间；无需历史 KV 的递归推理 | 证明无 Self-Attention 的高效状态序列模型可行；其状态是结构化递归状态，不是显式 Cell 记忆网络 |
| Mamba-2 / SSD (ICML 2024) | 建立 SSM 与 Attention 变体之间的结构化对偶关系 | 提醒我们状态计算与 Attention 可能存在数学等价或近似关系，不能只按名称判断是否真正改变范式 |
| RWKV-7 (2025) | 更灵活的动态状态演化、矩阵状态和递归推理 | 是重要的无标准 Self-Attention 语言模型基线；不是显式持久 Cell 数据库 |
| TTT (ICML 2025) | 将隐藏状态本身定义为可在测试时更新的学习模型 | 提供“状态可通过学习规则写入”的机制先例；状态仍是模型参数化记忆 |
| Titans (NeurIPS 2025) | Surprise-driven neural long-term memory，结合短期 Attention | 证明长短期记忆组合有价值；不是完全移除 Attention 的架构 |
| MIRAS (ICLR 2026) | 从记忆结构、attentional bias、retention gate、memory learning algorithm 统一分析序列模型 | 对 EPCLM 的写入、保留、学习目标设计很有参考价值；其“attentional bias”不是简单等同于 Transformer Self-Attention |
| Nested Learning / Hope (NeurIPS 2025) | 多时间尺度优化和 Continuum Memory System | 支持多层更新频率和自修改记忆的研究方向；没有给出 EPCLM 所需的独立 Cell 寻址系统 |
| ATLAS (ICML 2026) | 通过当前与过去 Token 优化测试时记忆，改善固定记忆容量和纯在线更新限制 | 说明记忆更新不能只看最后一个 Token；需要考虑跨片段累积和更高容量 |
| Memory Caching (ICML 2026) | 缓存递归模型的状态检查点，让有效记忆容量随上下文增长 | 对固定状态容量是重要警示；但状态快照缓存会重新引入增长型历史存储 |
| GradMem (ICML 2026) | 对少量 Memory Tokens 做测试时梯度优化，随后移除原始 Context | 是 Context Removal 的直接先例；但不是动态 Cell 网络，读取仍依赖基础模型 |
| Tensor Memory (2026 preprint) | 固定容量 3D 状态张量、软写入、局部交互和门控递归；评估覆盖语言、图像和视频 | 与空间化持久状态接近，但它是附加到 Transformer block 的记忆模块，不等于完整 State-based LM |
| TextNCA (Findings of EMNLP 2026) | Token 位置作为 Cell，局部 Attention 和 GRU 更新，测试迭代共享规则 | 与旧 CLM 相关；其结果未证明迭代共享本身带来可靠 PPL 优势，也未实现跨输入持久 Cell Memory |
| DNC (Nature 2016) | 可微外部读写存储器和显式地址/内容寻址 | 证明可学习读写外部结构化记忆的可行性；内容查找仍使用 Attention-like addressing |
| Modern Hopfield Networks (ICLR 2021) | 证明现代 Hopfield 更新与 Transformer Attention 存在等价关系 | 提醒我们：只把 Attention 换个名字或换成关联记忆，可能没有真正替代检索机制 |
| Growing Neural Cellular Automata (2020) | 持久 Cell 状态、共享局部更新规则、自组织与再生 | 提供 Cellular Dynamics 的基础范式；任务是形态生成，不是语言历史记忆 |

## 10.1 最重要的研究启示

### Mamba：状态必须具有选择性

Mamba 指出，普通固定动力学的 SSM 难以处理信息密集的离散语言。输入依赖的选择机制使模型能够决定传播或遗忘哪些信息。

EPCLM 必须解决同样的问题：什么信息写入状态，什么信息被忽略？

### TTT / Titans / MIRAS：写入规则是记忆能力的一部分

记忆不是只看状态张量大小。写入目标、更新算法、遗忘机制和读取过程共同决定有效容量。

EPCLM 不能只定义 Cell State 的维度，然后期待它自然形成有效记忆。

### ATLAS / Memory Caching：固定容量的回忆瓶颈是真实问题

当任务要求精确回忆大量独立事实时，固定状态可能不足。扩大状态、保存状态快照或使用外部存储都可能提升能力，但会重新引入容量和检索成本。

EPCLM 必须把容量—准确率—计算成本曲线作为核心结果。

### GradMem：Context Removal 是必要的实验范式

必须真正删除原始上下文，而不是仅仅在模型中增加一份状态副本。否则无法证明模型依赖的是持久状态。

### TextNCA：Cellular 形式本身不构成架构优势

将 Token 位置命名为 Cell、共享更新规则或增加迭代次数，不等于实现了持久状态记忆。必须证明状态独立于 Token 序列，并且对跨片段任务有可测量的价值。

---

## 10.2 2026-11-08 补充研究：SSM、Neural Memory 与 Attention 的职责

本次讨论把研究重点从“持久 State 能否取代历史 Token Attention”进一步拆分为三个可独立验证的机制：递归状态、测试时神经记忆、局部精确交互。它们不是同一种记忆的不同名称。

| 机制 | 更准确的定义 | 主要优势 | 不能直接假定的能力 |
|---|---|---|---|
| SSM / Mamba State | 由输入驱动、逐步演化的递归隐藏状态 | 以固定规模状态持续处理长序列；避免保存无界历史 KV | 不是精确、可寻址的事实数据库；不保证无损恢复历史 |
| Titans Neural Memory | 通过测试时梯度更新的小型神经网络参数 | 可学习历史输入中的非线性关联；可按 Query 读取 | 不保证事实精确性；原论文没有证明用户级独立 Cell 可直接跨模型移植 |
| Local / Sliding Window Attention | 对有限邻近 Token 做显式内容交互 | 精确保留局部语法、语义和邻近依赖；KV 范围可受窗口限制 | 单独使用时不能直接访问窗口外 Token |
| External Knowledge | 显式保存事实、文档、来源和版本 | 可审计、可精确查询、可更新和撤销 | 需要索引、检索和上下文融合成本 |

**术语修正：** SSM State 更适合称为 *recurrent / fading temporal state*，而不是“全局知识库”。它把历史压缩进递归状态，但其有效信息量和可恢复精度受状态维度、动力学、训练目标及输入分布约束。Titans 则是通过梯度更新形成的参数化关联记忆。两者可能互补，但互补性必须通过消融证明。

### 10.2.1 与本研究最相关的混合架构

| 研究 | 已实现的组合 | 已报告的证据 | 对 EPCLM / MiniCells 的边界 |
|---|---|---|---|
| Mamba-3（ICLR 2026）[30] | 新一代 SSM：更有表达力的离散化递归、复数 State 更新、MIMO | 在 1.5B 规模报告 SISO 相比 Gated DeltaNet 下游平均准确率 +0.6 个百分点，MIMO 再增加约 1.2 个百分点；在 State Size 实验中，以约一半 State Size 达到接近 Mamba-2 的 PPL | 是本项目 SSM Backbone 的最新基线；论文没有 Titans 或 Local Attention Cell 组合实验 |
| Understanding and Enhancing Mamba-Transformer Hybrids（2025 预印本）[31] | 比较 SSM 与 Attention 的 Sequential / Parallel 融合 | 作者报告 Sequential Hybrid 在较短上下文更好，Parallel Hybrid 在较长上下文更有效；用 paraphrase-augmented continual training 改善 Recall | 支持在 MiniCells 中将顺序堆叠与并行融合分开做消融；是预印本，不是 Mamba + Titans 证据 |
| B'MOJO（NeurIPS 2024）[20] | 短期上下文中的精确记忆、权重中的结构记忆、SSM State 中的 fading memory、外部 storage 中的长期精确记忆 | 在关联回忆任务上优于所比较的 SSM 与混合模型；语言建模 PPL 与同规模 Transformer / SSM 相当，最高测试至 1.4B；报告训练速度最高约快 10%；训练至 8K 后测试至 32K | 直接支持“SSM State + 外部长期记忆”可组合，但外部记忆是异步检索的 eidetic storage，不是 Titans Neural Memory，也没有验证可训练的独立知识 Cell |
| Titans（NeurIPS 2025）[5] | MAC：把 Neural Memory 读出作为 Attention 上下文；MAG：Sliding Window Attention 与 Neural Memory 并行并通过 Gate 融合；MAL：Neural Memory 层后接 Sliding Window Attention；另有无 Attention 的 LMM 单独模型 | 论文在 170M–760M 等规模评估多个变体；报告 MAC / MAG / MAL 在其测试设置中优于 Samba 与 Gated DeltaNet-H2；MAC 在更长依赖的部分测试中优于 MAG；单独 LMM 也被作为无 Attention 序列模型测试 | 这是“Titans + Local Attention”的直接证据；不是 Mamba + Titans。LMM-only 说明 Neural Memory 可以被作为序列模型研究，但不等于在所有任务上都可替代 Attention |
| TNT（ICLR 2026）[21] | 一个跨大 Chunk 持续演化的 Global Deep Memory，加上多个并行 Local Deep Memory；Local Memory 周期性重置以打破训练依赖 | 150M 参数、10B Tokens 实验中，TNT 对 Titans 小 Chunk 基线报告最高 17.37× time-to-target-loss 加速；多 Local Memory 改善语言建模 PPL；移除 Global Memory 后 PPL 明显恶化 | Global / Local 是 Titans / TTT 类 Deep Memory 的层级训练结构，不是 Mamba SSM State；多个 Local Memory 是模型内部多尺度模块，不等于用户可独立发布、授权和 Fork 的 Cell |
| Samba（ICLR 2025）[22] | Mamba 与 Sliding Window Attention 按层混合 | 3.8B 参数、3.2T Tokens；报告 4K 训练长度下的长上下文泛化、256K Passkey 检索表现，以及长输入下相对 GQA Transformer 的吞吐优势 | 直接支持“SSM + Local Attention”；没有 Titans Neural Memory，也没有独立知识模块 |
| Griffin（2024 预印本）[23] | Gated Linear Recurrence（RG-LRU）与 Local Attention 混合 | 扩展至 14B；作者报告其在若干设置下达到或超过自有 Transformer 基线，并具有长度外推能力 | 递归机制不是 Mamba SSM；可作为“递归状态 + 局部注意力”的相邻架构先例，不应标记为 Mamba + Titans |
| RAttention（2025 预印本 / Apple 研究页面）[25] | Sliding Window Attention + Residual Linear Attention（RLA），由 RLA 处理窗口外信息 | 在 3B / 12B 规模测试；报告窗口 512 时在若干基准上匹配或超过 Full Attention / 更大 SWA 窗口，并改善 RULER 长度泛化 | RLA 是额外的线性递归路径，不是 Titans，也不是 Mamba；说明缩小局部窗口时需要一个窗口外信息通路 |
| Gated Attention（NeurIPS 2025）[24] | 在 SDPA 输出后增加 Head-specific Sigmoid Gate | 比较 30 个 15B MoE 与 1.7B Dense 变体，训练数据约 3.5T Tokens；作者报告性能、训练稳定性和长度外推改善，并缓解 Attention Sink | 该 Gate 调制 Attention 输出，不扩大 Attention 的可见窗口；论文的主要证据不是“Gated SWA + Titans”，不能把它与 Titans MAG 的 Gate 混为一谈 |
| Nemotron 3 Nano（NVIDIA 技术报告）[26] | 52 层混合结构：23 层 Mamba-2、6 层 GQA Attention、23 层 MoE | 报告 31.6B 总参数、约 3.2B 每 Token 激活参数，支持最长 1M Context；以少量 Attention 层保留全局信息交互 | 这是产业级 Mamba + 稀疏全局 Attention + MoE 实例，不是 Local Attention，也没有 Titans Neural Memory 或可分享 Cell |

### 10.2.2 关键论文结果的解释边界

**B'MOJO：** 它将 *fading memory in state* 与 *long-term eidetic memory in storage* 放在同一框架中，说明递归压缩与精确外部存储可以组合。其结果不能直接外推为“SSM State + Titans 参数记忆”已经成立。

**Titans：** 原论文已包含与我们讨论高度相关的 MAG：SWA 提供短期精确交互，Neural Memory 提供长期学习记忆，二者通过非线性 Gate 融合。MAG 的 Gate 是“Attention 分支与 Neural Memory 分支之间的融合”，而 Gated Attention 论文中的 Gate 是“Attention 输出的输入相关门控”。二者不是同一个模块。Titans 还测试了无 Attention 的 LMM，但论文结果并未证明它在所有任务上都能取代局部 Attention。

**TNT：** TNT 的 Global Memory 在大 Chunk 间持续传递状态；Local Memory 在训练阶段按 Shard 重置，以获得并行性，随后通过第二阶段小 Chunk 微调恢复高分辨率推理。其 17.37× 是相对 Titans 小 Chunk 训练配置的 time-to-target-loss 加速，不代表全面快于最优化 Transformer：论文同一张表中的 Gated Transformer + FlashAttention 仍有更短训练时间。多 Local Memory 数量增加时，语言建模 PPL 持续改善，但常识准确率并非单调增加；移除 Global Memory 会显著恶化结果。

**Samba / Griffin / Nemotron 3：** 这些工作证明了递归状态与 Attention 可以互补，但 Attention 的范围和递归单元不同。Samba 使用 Mamba + SWA；Griffin 使用 RG-LRU + Local Attention；Nemotron 3 使用 Mamba-2 与少量 GQA 全局 Attention。它们不能合并成一个笼统的“Mamba + Attention”结果。

**RAttention：** 它不是单纯缩小窗口，也不是给 SWA 加一个 Gate，而是在窗口外增加 Residual Linear Attention 通路。因此，它是我们评估“局部窗口之外的信息如何进入模型”时的重要对照。

### 10.2.3 当前证据矩阵

| 组合 | 是否已有直接研究 | 代表工作 | 当前结论 |
|---|---|---|---|
| SSM + Local Attention | 是 | Samba、Griffin（非 Mamba SSM）、Nemotron 3（少量全局 Attention） | 已有多种可行架构与规模化证据 |
| Titans Neural Memory + Local Attention | 是 | Titans MAG / MAL | 已有直接实验；MAG 是最贴近当前 MiniCells 设想的先例 |
| SSM + 外部精确记忆 | 是 | B'MOJO | 已有直接证据，但记忆不是 Titans |
| 多尺度 / Global-Local Titans Memory | 是 | TNT | 已有训练与质量实验；模块是模型内部层级，不等于可移植 Cell |
| Mamba State + Titans Neural Memory | 本次检索未发现严格直接的统一实验 | — | 是待验证组合，不能写成已有结果 |
| Mamba State + Local Attention + Titans Neural Memory | 本次检索未发现严格直接的三者消融实验 | — | 是 MiniCells 可研究的组合假设 |

## 10.3 Titans Neural Memory 的模块化与可移植性

### 10.3.1 已有机制能证明什么

Titans 的 Neural Memory 是一个可通过梯度更新的深层网络。其在线记忆不只是普通静态向量；更新过程涉及网络参数以及优化过程中的动量、遗忘 / Weight Decay 等状态。理论上可以把某个时刻的 Memory 参数保存下来，并在兼容的推理实例中恢复。

但必须区分三个不同主张：

1. **Checkpointable：** 某个 Memory 实例的参数和必要更新状态可以保存、恢复。
2. **Forkable：** 从一个 Checkpoint 复制出独立分支，之后分别学习。
3. **Portable / Shareable：** 另一个模型或用户加载该模块后，仍能正确读取并继续使用。

Titans 论文验证的是 Neural Memory 的学习与读取机制，以及不同架构变体；它没有直接验证用户级 Cell Registry、独立权限、跨用户 Fork 市场，或跨 Backbone 通用加载。

TNT 的 Global / Local Memory 数量与分辨率实验，也只证明模型内部多尺度 Memory 结构可以协作，不证明这些 Memory 可以脱离原模型独立分发。

### 10.3.2 可移植性分级

| 级别 | 目标环境 | 需要满足的条件 | 当前判断 |
|---|---|---|---|
| P1：同一 Backbone 实例间 | 完全相同的模型权重、Tokenizer、Memory 结构和接口 | 保存 Memory 权重；若继续训练，还要保存优化器状态、步数和更新超参数 | 最适合作为 MVP；仍需实测 |
| P2：同架构、不同微调版本 | 同一架构，但 Backbone 权重不同 | 验证 Key / Query / Value 表示是否兼容；必要时增加 Reader / Adapter 校准 | 未有 Titans 专属直接证据 |
| P3：不同 Backbone | 不同模型家族、维度或表示空间 | 共享 Memory Latent Space，或训练目标模型侧 Reader / Adapter | 属于独立研究问题，不应假定直接加载可行 |

同样的 Hidden Dimension 不代表相同语义空间。Titans 的 Key、Value 和 Query 通常由 Backbone 表示经过投影得到；因此 Memory 参数可能依赖原 Backbone 的表示坐标。跨模型共享不能只检查张量形状。

### 10.3.3 相邻的可移植记忆研究

| 研究 | 记忆形式 | 已验证的迁移 | 与 Titans Cell 的区别 |
|---|---|---|---|
| Frozen Memory Is Not Enough / Cross-Model Memory Transfer via Target-Side Reader Adaptation（2026 预印本）[27] | Engram 式外部可寻址哈希记忆表 | 冻结来源记忆，在目标 Backbone 侧训练轻量 Reader；作者报告目标侧 Reader 对跨模型可用性很关键 | 证明外部记忆资产 + 目标侧 Reader 适配的路线有实证支持；不是 Titans 在线梯度记忆 |
| LatentPort（2026 预印本）[28] | Hybrid 模型的 KV 与 Gated DeltaNet 递归推理状态 | 在一个架构匹配的 Qwen3.5 4B → 9B 同系列模型对上测试状态迁移；报告无需目标模型重放历史前缀 | 迁移的是推理运行状态，不是 Titans Neural Memory 权重；证据只覆盖一个方向、一个模型对和 teacher-forced continuation，不能推成通用跨模型协议 |
| Pluggable Neural Machine Translation Models via Memory-augmented Adapters（LREC-COLING 2024）[29] | 用户文本构建的多粒度外部记忆 + Adapter | 将用户特定记忆以可插拔方式接入预训练 NMT 模型，并测试风格 / 领域适配 | 是可插拔用户记忆的相邻先例，但记忆是检索增强，不是 Titans 参数化 Memory |

### 10.3.4 MiniCells 的 Cell 资产定义建议

将“可移植 Cell”作为系统设计目标，而不是现有 Titans 已有能力。一个可继续训练的 Cell 包至少应记录：

- `memory_architecture`：网络结构、层数、维度和更新规则版本。
- `memory_weights`：当前 Neural Memory 参数。
- `optimizer_state`：动量、优化器步数和必要的更新状态；仅推理使用时可以省略，但必须标记为 inference-only。
- `io_interface`：Key / Query / Value 的维度、归一化和投影协议。
- `backbone_compatibility`：Tokenizer、Backbone 架构及权重版本、Adapter 版本。
- `provenance`：知识来源、训练数据摘要、贡献者和许可。
- `checkpoint_version`：父版本、Fork 来源和时间点。
- `permissions`：read、learn、fork、export、share 等授权。

第一阶段只要求 P1：相同 Backbone 权重与接口下，Cell 可保存、加载、Fork 并独立演化。P2 / P3 不应阻塞 MVP。

## 10.4 Local Attention、Gated Attention 与长程信息通路

### 10.4.1 Sliding Window Attention 的边界

对窗口大小为 \(W\) 的因果 SWA，位置 \(t\) 直接访问的范围是 \([t-W+1,t]\)。在固定窗口下，其每层 Attention 计算与 KV 规模随序列长度近似线性增长，而不是完整历史的二次注意力。

但 SWA 本身不是 Long-range Attention：窗口外 Token 不会被该层直接访问。多层局部交互可以逐层传播信息，但这不等于模型拥有对任意历史位置的直接精确寻址能力。若任务需要跨窗口的长程信息，必须存在另一条信息通路，例如 SSM State、Titans Memory、Residual Linear Attention、显式检索或少量 Global Attention。

### 10.4.2 两种 Gate 必须分开

**Gated Attention（Qiu et al.）**：

\[
Y=\operatorname{SDPA}(Q,K,V),\qquad
Y'=Y\odot\sigma(XW_g)
\]

它控制当前 Attention 输出的强度和稀疏性。将同一 Gate 放到 SWA 的 SDPA 输出后是合理的实验假设，但原论文的主要证据不能直接视为“Gated SWA 已经验证”。它不改变 Attention Mask，也不会让窗口外 Token 自动变得可见。

**Titans MAG Gate**：并行计算 SWA 输出与 Neural Memory 输出，再用可学习非线性 Gate 融合两条分支。它控制的是“短期 Attention 与长期 Neural Memory 如何融合”，与 Gated Attention 的作用位置不同。

因此可以设计两级门控，但应先做消融：先测 Gated SWA，再测 Titans MAG，再测两种 Gate 是否互补。不要一开始把两个 Gate 叠加后将性能变化归因于其中一个。

### 10.4.3 三条长程路径的候选设计

| 路径 | 机制 | 适合承担的职责 | 主要代价 / 风险 |
|---|---|---|---|
| 递归压缩路径 | Mamba / 其他 SSM | 连续输入形成低成本全局时间状态 | 状态有损；不保证精确事实恢复 |
| 参数化关联路径 | Titans Neural Memory / 多个 Cells | 跨 Chunk、跨会话学习可复用关联 | 在线更新成本、遗忘、表示对齐和 Cell 冲突 |
| 显式窗口外路径 | RAttention 的 RLA、B'MOJO storage、RAG 或少量 Global Attention | 将窗口外信息带回当前计算 | 额外状态、检索或 Attention 成本；机制并不等价 |

一个候选融合形式是：

\[
Z_t=[X_{t-W:t},\;P_s(H_t^{SSM}),\;R_t^{Titans},\;R_t^{External}]
\]

其中 \(P_s\) 将 SSM State 投影为有限 State Tokens，\(R_t^{Titans}\) 是所选 Cell 的记忆读出，\(R_t^{External}\) 是有来源的外部事实。局部 Attention 对这些有限表示进行融合。该表达是 MiniCells 的设计提案，不是已有论文已经验证的统一结构。

### 10.4.4 对“长程注意力”的准确表述

Sliding Window + Gate 仍然是局部 Attention。若它与 Mamba、Titans 或 Residual Linear Attention 协作，整个模型可以具备长程信息处理能力，但不能把 SWA 本身称为全局或长程 Attention。更准确的术语是：**Local Attention with Recurrent / Neural / Retrieval Memory Paths**。

## 10.5 本次更新后的研究缺口

现有研究已经分别证明：

1. SSM 可以与局部或少量全局 Attention 组合（Samba、Griffin、Nemotron 3）。
2. Titans Neural Memory 可以与 Sliding Window Attention 组合（MAG / MAL）。
3. SSM 可以与异步更新的精确外部记忆组合（B'MOJO）。
4. Titans / TTT 类 Deep Memory 可以做 Global-Local 多尺度训练（TNT）。
5. 某些外部记忆表和递归推理状态可以在受限条件下跨模型迁移（Frozen Memory、LatentPort）。

但本次文献核验没有找到一项严格控制变量的工作，完整验证 **Mamba State + Local / Gated Attention + 多个独立 Titans Neural Memory Cells**，也没有找到直接证明 Titans Cell 可以作为用户级知识资产在不同 Backbone 间通用移植的实验。

因此，MiniCells 的新颖性不应写成“首次把 Mamba、Titans 和 Attention 放在一起”，而应聚焦于可检验的组合问题：三种机制是否提供互补信息；Memory 是否可被独立寻址、持久化和 Fork；相同知识模块能否在兼容接口下被不同 Agent 复用；总成本是否优于 Transformer + RAG 等强基线。

# 11. EPCLM 的理论假设与可证伪条件

EPCLM-002 不把下列假设当成事实。每一项都必须允许实验否定。

## H1 — Predictive State Sufficiency

在给定任务分布和记忆预算下，持久状态可以保留足够的历史信息，使移除原始历史 Token/KV 后的预测损失接近完整历史基线。

**反证条件：** 在合理增加状态容量、训练预算和读取能力后，状态模型仍在核心记忆任务上持续显著落后。

## H2 — Selective State Transition

新输入只需更新状态系统的一小部分，即可保持必要的预测和记忆能力。

**反证条件：** 达到目标准确率所需的活跃状态比例接近全部状态，或路由成本抵消更新节省。

## H3 — Indexed State Retrieval

通过索引或结构化路由，可以在不扫描全部 Cell 的情况下检索足够相关的信息。

**反证条件：** 为达到可接受 Recall@K，候选数量必须接近 N，或索引维护成本过高。

## H4 — State-based Autoregressive Generation

模型可以在不访问历史 Token/KV 的情况下，利用持久状态完成连续生成。

**反证条件：** Context Removal 后模型无法维持基本语言质量，或必须不断保留大量历史 Token 才能恢复性能。

## H5 — System-level Efficiency

在相同任务质量或相同质量容忍度下，状态机制能降低真实推理成本。

**反证条件：** 状态写入、路由、索引、随机访问、Kernel Launch 和数据搬运的总成本不低于强 Transformer 基线。

## H6 — Cellular Organization Adds Value

将状态组织为具有独立身份和局部关系的 Cell，比同等容量的单一向量/矩阵状态或普通记忆 Token 更有优势。

**反证条件：** 在同等内存、参数和计算预算下，Cell 结构没有可重复的质量、可扩展性或效率优势。

---

# 12. 实验规划

建议将 EPCLM-002 分成理论验证、记忆机制验证和语言模型验证三个层次。前一层未通过，不进入后一层的大规模实现。

## Experiment 002-A — Predictive State Sufficiency

**目标：** 验证历史内容是否能被有限状态压缩，同时保持未来任务所需的信息。

测试任务：

- 单事实精确记忆。
- 多个 Key-Value 绑定。
- 多实体、多关系。
- 随机距离的延迟查询。
- 长距离组合推理。
- 不同信息密度和不同干扰率。

对照组：

1. Transformer + 完整 KV。
2. Transformer + 有限压缩记忆。
3. 标准 RNN/SSM 状态。
4. 可学习持久状态候选。

关键变量：

- 状态字节数。
- 历史长度。
- 事实数量。
- 干扰 Token 数。
- Query 类型和分布。

关键输出：

- 任务准确率。
- 状态容量曲线。
- 达到固定准确率所需的状态字节数。
- Context Removal 后的性能下降。
- 记忆容量增加时的边际收益。

本实验不需要先实现复杂 Cell 网络。可以用最简单的状态表示建立容量下界和任务难度基线。

## Experiment 002-B — Write, Update and Retention

**目标：** 验证状态是否能够正确吸收新信息，而不是只在写入后静态保存。

任务：

- 重复写入同一事实。
- 对已有事实补充属性。
- 修正已有事实。
- 新旧事实冲突。
- 多来源信息。
- 大量无关信息干扰。
- 长时间未访问后的记忆保留。

关键指标：

- 新信息写入准确率。
- 旧信息保留率。
- 冲突解决准确率。
- 错误覆盖率。
- 状态膨胀速度。
- 遗忘策略对其他任务的副作用。

## Experiment 002-C — Retrieval and Sparse Routing

**目标：** 验证是否可以不扫描全部 Cell 而完成有效状态读取。

对照：

- Dense scan / dense content lookup。
- Hash / explicit address。
- Hierarchical index。
- Learned sparse routing。
- Graph traversal。

测量：

- Recall@K。
- 端到端查询准确率。
- 路由延迟。
- 索引内存。
- 索引更新成本。
- GPU 内存访问和数据搬运。
- K/N 与质量之间的关系。

注意：若候选集合内仍采用 Attention，应明确标记为“稀疏候选 Attention”，不能宣称完全 Attention-free。

## Experiment 002-D — State-based Autoregressive Language Modeling

**目标：** 将通过前述验证的状态机制集成到小型语言模型。

建议先使用现有 1.9M 级别实验规模，避免在理论机制尚未验证时投入 30M 训练。

基线：

- Transformer T1：标准 Transformer 基线。
- Native CLM X3/X4：旧机制基线。
- 新 State-based Candidate：持久状态候选。

测试：

- 标准语言建模。
- 跨 chunk 连续预测。
- 历史 KV 清除后的生成。
- 长距离实体和关系依赖。
- 状态更新后的生成质量。
- 长时间持续运行。

## Experiment 002-E — System Efficiency and Scaling

只有在质量和正确性达到预设门槛后才进行。

测量：

- Prefill / state formation。
- 单 Token state update。
- State retrieval。
- Decode latency。
- End-to-end latency。
- Peak GPU memory。
- 状态存储增长率。
- GPU DRAM traffic。
- Kernel Launch 数量。
- CPU-GPU 同步。
- 能耗或 GPU-seconds。

至少报告两种公平比较：

1. **Matched quality：** 在相近语言质量和记忆准确率下比较成本。
2. **Matched compute / memory budget：** 在相近计算和存储预算下比较质量。

不能只比较理论 FLOPs 或某个孤立 Kernel。

---

## Experiment 002-F — 约 10M 参数的混合架构消融

**目标：** 在小规模、可控的模型中，分别测量局部 Attention、Gated Attention、SSM State 与 Titans Neural Memory 的增量贡献。该实验验证机制，不以 10M 模型的通用语言能力作为目标。

### F.1 分阶段实验矩阵

**Stage 1：建立序列处理基线**

| ID | Backbone | 额外记忆 | 目的 |
|---|---|---|---|
| F1 | Full Attention Transformer | 无 | 强历史 KV 基线 |
| F2 | Sliding Window Transformer | 无 | 测量局部窗口的代价 |
| F3 | Gated Sliding Window Transformer | 无 | 测量 Head-specific Gate 对局部 Attention 的贡献 |
| F4 | Mamba-3 SISO | 无 | 测量纯 SSM State |
| F5-S | Mamba-3 与 Sliding Window Attention 按层交替 / 顺序组合 | 无 | 复现 Samba 类组合的缩小版机制基线 |
| F5-P（可选） | Mamba-3 与 Gated SWA 并行计算，再融合两条分支 | 无 | 对照 Sequential / Parallel Hybrid；检验长上下文下并行融合是否更有利 |

**Stage 2：加入 Neural Memory**

| ID | Backbone | Memory | 研究问题 |
|---|---|---|---|
| F6 | Gated SWA Transformer | Titans MAG | 复现最接近的已发表 Titans + Local Attention 组合 |
| F7 | Titans LMM-style sequence model | Titans Neural Memory | 测试无 Attention 时 Deep Memory 的能力边界 |
| F8 | Mamba-3 | Titans | 测试 SSM State 与 Titans 是否互补 |
| F9-S | Mamba-3 + Gated SWA（顺序 / 交替） | Titans | 测试三种机制的顺序组合 |
| F9-P（可选） | Mamba-3 与 Gated SWA 并行，再与 Titans Memory 融合 | Titans | 测试三路并行融合是否优于顺序堆叠 |
| F10（可选） | SWA + Residual Linear Attention | 无 Titans | 以 RAttention 类窗口外路径作为竞争性对照 |

**Stage 3：多 Cell 与可移植性**

- 在 F6 或 F9 的稳定配置上创建两个独立 Titans Cells。
- 比较单 Cell、两个独立 Cell、共享 Cell 只读和 Fork 后独立学习。
- 同一 Backbone、同一权重版本下，在两个模型实例之间导出、加载和恢复 Cell。
- 之后再测试不同微调版本的 Backbone + Reader / Adapter；跨模型家族迁移不作为第一轮门槛。

### F.2 参数与训练公平性

- 目标模型规模约 10M；记录 Backbone 静态参数、Attention 参数、Titans Memory 网络参数、运行时 Memory 参数和优化器状态，不能只报一个总参数数。
- 尽量控制总静态可训练参数、Tokenizer、训练数据、训练 Token 数、序列长度、优化器、精度、随机种子和硬件环境。
- 对 Titans 在线更新，分别报告推理态 Memory 参数、可训练态优化器状态及其持久化成本。
- 使用相同的数据顺序与评价集；至少多个随机种子；同时报告质量与系统成本。
- 10M 模型适合机制和合成记忆任务，不足以据此判断大模型级通用语言能力或商业服务性能。

### F.3 必须包含的状态清除与记忆隔离实验

1. **同会话连续：** 保留 KV / SWA 窗口、Mamba State 和 Titans Memory。
2. **跨 Chunk：** 清空局部窗口，保留 Mamba State 和 Titans Memory。
3. **跨会话：** 清空 KV / Local Window 与 Mamba State，只恢复 Titans Memory。
4. **SSM 消融：** 保留 Titans，清空 Mamba State；与完整模型比较。
5. **Titans 消融：** 保留 Mamba State，清空 Titans；与完整模型比较。
6. **双清除：** 清空两者，仅保留当前输入窗口，作为短期上下文基线。
7. **Cell Fork：** 从同一 Checkpoint 复制两个 Cell，分别写入不同知识，检查隔离性和独立演化。

这些条件用于确定信息究竟保存在 KV、SSM State 还是 Titans 参数中，避免把未清除的运行时状态误认为持久知识。

### F.4 评价任务与指标

| 任务 | 主要指标 | 验证内容 |
|---|---|---|
| Short-range language modeling | PPL / next-token loss | 局部语义与语言建模基础能力 |
| Delayed associative recall | exact accuracy、Recall@K | 窗口外关联恢复 |
| Long-range fact retrieval | exact match、source accuracy | 参数化记忆是否能精确恢复事实 |
| Cross-session recall | 清除 KV / SSM 后的准确率 | Titans 是否提供独立持久记忆 |
| Continual update / correction | 新知识写入率、旧知识保留率、冲突修正准确率 | 持续学习与遗忘 |
| Cell isolation / Fork | 跨 Cell 污染率、Fork 后行为差异 | 模块独立性 |
| Portability | 加载后性能差、目标侧适配成本 | 同 Backbone 与跨 Backbone 可移植性 |
| System cost | Recall / Learn 延迟、状态字节、优化器字节、GPU 时间 | 是否存在实际成本优势 |

必须保留 Transformer + RAG 对照。若 Titans 只在合成关联任务上有效，而 RAG 在精确事实与来源任务上更好，应如实记录任务边界，不能用单一总分掩盖差异。

### F.5 决策规则

- 若 F6 已经明显优于 F3，先研究 Titans + SWA，不急于增加 Mamba。
- 若 F8 相比 F4 和 Titans-only 有稳定增益，才保留 SSM + Titans 双路径。
- 若 F9 没有超过 F6 / F8 的质量或成本前沿，则不保留三路结构。
- 若 Titans Cell 在同 Backbone 的 Checkpoint / Fork 测试中不能隔离或恢复，暂停跨用户分享设计。
- 若不同 Backbone 迁移必须训练大量目标侧参数，Cell 的“可移植”应限定为带 Adapter 的迁移，而不是宣称通用即插即用。

# 13. 实验方法与公平性要求

## 13.1 必须控制的变量

- 相同 Tokenizer。
- 相同训练数据和数据划分。
- 相同训练 Token 数量（对应实验中）。
- 明确参数量。
- 明确状态容量。
- 明确训练和推理精度。
- 多个随机种子。
- 冻结的评价代码和数据。
- 可复现的硬件和软件环境。

## 13.2 Transformer 必须获得合理基线

不能故意限制 Transformer 的历史访问能力。

至少需要：

- 正常完整 KV Cache 基线。
- 合理的高效 Attention 实现。
- 在内存受限实验中，明确 KV 的预算。
- 在 Context Removal 实验中，允许 Transformer 使用其自身可学习的压缩记忆对照，而不是只和被人为削弱的模型比较。

## 13.3 必须区分三个效率指标

**计算效率：** FLOPs、有效更新数、算子成本。

**存储效率：** 状态字节数、KV 字节数、索引大小、峰值显存。

**系统效率：** 实际延迟、吞吐、GPU 时间、数据搬运和 Kernel Launch。

理论稀疏不等于硬件加速；状态更小也不等于整体推理更快。

---

# 14. 主要风险与失败模式

| 风险 | 说明 | 必须进行的验证 |
|---|---|---|
| 信息瓶颈 | 状态压缩丢失后续 Query 所需信息 | 容量曲线、随机事实回忆 |
| 干扰 | 新写入破坏旧记忆 | 多轮更新与保留测试 |
| 路由遗漏 | 必要 Cell 没有被激活 | Recall@K、分布外路由测试 |
| 全局扫描 | 路由器仍遍历全部 Cell | 端到端路由复杂度 |
| Cell 膨胀 | 状态数量随输入无控制增长 | 长期状态增长率 |
| 状态振荡 | Cell 消息引起持续循环更新 | 事件终止和稳定性测试 |
| 读取退化 | 状态存在但不能被有效解码 | Context Removal 查询测试 |
| 训练不匹配 | 训练依赖完整历史，推理依赖 State | 训练时强制移除历史 |
| 硬件不友好 | 稀疏访问导致随机访存和小 Kernel | 实际 GPU Profiling |
| 评估偏差 | 只在人工记忆任务上表现好 | 自然语言与合成任务共同评估 |

---

# 15. EPCLM-002 的阶段性决策门槛

建议使用以下决策顺序，而不是先规定一个架构再持续优化。

### Gate A — 理论可行性

必须完成：

- 预测充分性的明确定义。
- 固定与增长状态的容量限制。
- 状态更新和读取的因果模型。
- 检索复杂度分析。
- 可证伪假设。

未完成则不进入完整实现。

### Gate B — 记忆机制成立

必须证明：

- 清除历史 Token/KV 后仍能完成定义好的查询。
- 新信息可写入。
- 冲突和修正可处理。
- 旧信息不会被无关输入轻易破坏。

### Gate C — 稀疏读取成立

必须证明：

- 路由 Recall 达到预设要求。
- 不需要全量扫描。
- 索引和路由的系统成本可接受。

### Gate D — 语言模型成立

必须证明：

- 新模型能够完成自回归语言建模。
- 与 T1、X3/X4 的对比可复现。
- Context Removal 后的语言质量可量化。

### Gate E — 效率优势成立

必须在质量相当的条件下，观察到真实系统成本下降，才可以主张新机制具有效率优势。

---

# 16. 预期理论贡献

如果 EPCLM 成功，潜在贡献不应只表述为“用 Cell 取代 Transformer”。

更准确的贡献可能包括：

1. **Predictive Persistent State：** 将历史表示为针对未来任务充分的持久状态。
2. **State Formation and Evolution：** 学习如何把连续输入写入、融合、修正和遗忘。
3. **Selective State Retrieval：** 在不扫描完整历史的条件下访问相关状态。
4. **Event-driven Computation：** 让计算量主要由必要的状态变化和传播决定。
5. **State-based Autoregressive Modeling：** 在历史 Token/KV 被清除后，仍然通过持久状态进行生成。
6. **Capacity–Quality–Cost Characterization：** 定量刻画记忆容量、预测质量和实际计算成本之间的关系。

其中第 6 项尤其重要。即使最终发现 State-based 模型并不能在所有任务上优于 Transformer，只要我们能清楚刻画它的能力边界，也能形成有价值的研究结论。

---

# 17. 当前建议的研究决策

基于现有研究，建议采取以下决策：

| 问题 | 当前建议 |
|---|---|
| 是否继续优化旧 X3/X4 的 Prefill / Decode？ | 暂停作为主线，仅保留基线 |
| 是否必须彻底删除所有 Attention？ | 否；先消除对无界历史 Token Attention 的依赖 |
| 是否必须使用固定容量 State？ | 否；固定和增长状态分别研究 |
| 是否必须使用显式 Cell？ | 作为核心候选，但需要与普通向量/矩阵状态对照 |
| 是否立即实现复杂 Event Runtime？ | 否；先证明状态形成、读取和充分性 |
| 是否立即训练 30M？ | 否；先用小规模实验验证机制 |
| 第一项实验是什么？ | 002-A Predictive State Sufficiency |

## 17.1 2026-11-08 研究路线更新

以下决策是本次文献整理后的当前建议；它们更新研究优先级，但不覆盖前文已经记录的理论假设与历史实验。

| 问题 | 当前立场 |
|---|---|
| SSM State 是否必需？ | 否，作为可选的递归全局时间状态；必须证明它相对 Titans / Local Attention 提供增量价值 |
| Titans 是否等于精确知识库？ | 否；它是参数化关联记忆。精确事实、来源和版本仍由 External Knowledge 负责 |
| 是否必须保留 Attention？ | 不预设全局 Attention 必须存在；允许 Local Attention。Titans 原论文也测试了无 Attention 的 LMM，但混合模型有其任务优势 |
| SWA + Gated Attention 是否构成长程注意力？ | 否；Gate 不扩大窗口。必须与 SSM、Titans、RLA、检索或少量 Global Attention 等窗口外通路配合 |
| Titans 多 Memory 是否等于独立 Cell？ | 否；TNT 的多尺度 Local Memory 是模型内部模块。用户级独立 Cell 需要额外的生命周期、隔离和权限设计 |
| Cell 分享是否已被 Titans 论文证明？ | 否；先验证同 Backbone 的 Checkpoint / Restore / Fork，再研究跨 Backbone Adapter |
| 初始模型规模 | 约 10M 参数，作为机制验证规模；不要将其性能外推到成熟 LLM |
| 第一轮主实验 | F1–F9 的分阶段消融；RAG 与 Full Attention 保留为强对照 |
| 三机制统一架构 | Mamba State + Gated SWA + Titans 是待验证候选，不是已成立结论 |

### 扩展研究命题（2026-11-08）

> **Can sequence computation, recurrent temporal state, test-time neural memory, and exact external knowledge be separated into independently testable components, then recombined so that each component contributes measurable value in long-range understanding, cross-session learning, and portable knowledge reuse?**

中文：

> 能否将序列计算、递归时间状态、测试时神经记忆和精确外部知识分离为可独立验证的组件，再将它们组合起来，使每个组件都能在长程理解、跨会话学习和可移植知识复用中提供可测量的增量价值？

这项扩展命题不预设 Mamba、Titans 或 Attention 中任何一个必须存在。实验结果可以支持三路组合，也可以证明两路组合甚至单一路径更合理。

## 最终研究命题

EPCLM-002 的核心命题应当写为：

> **Can a language model replace unbounded historical-token retrieval with a learned, persistent, selectively addressable state that preserves the information required for future prediction, while reducing total memory and computation cost?**

中文：

> 语言模型能否用一种可学习、持久且可选择性寻址的状态，替代对无界历史 Token 的检索，在保留未来预测所需信息的同时，降低整体存储和计算成本？

这是一个可以被数学分析、实验验证，也可以被推翻的研究命题。

**最重要的原则：持久状态不是天然优于 KV Cache。只有当它在明确的记忆预算下保留了足够的预测信息，并且其写入、路由、读取和更新的总成本低于强基线时，EPCLM 才真正取得了架构优势。**

---

# 参考文献

以下链接指向论文原文、正式会议论文页或研究机构页面。原有文献按 2026-10-30 整理；本次新增文献与状态核验截至 2026-11-08。公开预印本与正式会议论文分别标注，不把预印本自动视为同行评审结论。

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** Albert Gu, Tri Dao. arXiv:2312.00752, 2023.  
   https://arxiv.org/abs/2312.00752

2. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.** Tri Dao, Albert Gu. ICML 2024.  
   https://proceedings.mlr.press/v235/dao24a.html

3. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** Yu Sun et al. ICML 2025.  
   https://proceedings.mlr.press/v267/sun25h.html

4. **RWKV-7 “Goose” with Expressive Dynamic State Evolution.** Bo Peng et al. arXiv:2503.14456, 2025; COLM 2025.  
   https://arxiv.org/abs/2503.14456

5. **Titans: Learning to Memorize at Test Time.** Ali Behrouz, Peilin Zhong, Vahab Mirrokni. NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html

6. **It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization.** Ali Behrouz et al. ICLR 2026.  
   https://arxiv.org/abs/2504.13173

7. **Nested Learning: The Illusion of Deep Learning Architectures.** Ali Behrouz et al. NeurIPS 2025.  
   https://papers.nips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html

8. **ATLAS: Learning to Optimally Memorize the Context at Test Time.** Ali Behrouz et al. ICML 2026.  
   https://proceedings.mlr.press/v306/behrouz26b.html

9. **Memory Caching: RNNs with Growing Memory.** Ali Behrouz et al. ICML 2026.  
   https://proceedings.mlr.press/v306/behrouz26a.html

10. **GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent.** Yuri Kuratov et al. ICML 2026.  
    https://proceedings.mlr.press/v306/kuratov26a.html

11. **Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers.** Kabir Swain et al. arXiv:2605.27686, 2026 preprint.  
    https://arxiv.org/abs/2605.27686

12. **TextNCA: Neural Cellular Automata for Language Modeling via Hierarchical Local Attention.** Avni Mittal et al. Findings of EMNLP 2026.  
    https://aclanthology.org/2026.findings-emnlp.1413/

13. **Hybrid computing using a neural network with dynamic external memory.** Alex Graves et al. Nature 538, 471–476, 2016.  
    https://doi.org/10.1038/nature20101

14. **Hopfield Networks is All You Need.** Hubert Ramsauer et al. ICLR 2021.  
    https://arxiv.org/abs/2008.02217

15. **Growing Neural Cellular Automata.** Alexander Mordvintsev et al. Distill, 2020.  
    https://distill.pub/2020/growing-ca/

16. **Compressive Transformers for Long-Range Sequence Modelling.** Jack W. Rae et al. arXiv:1911.05507, 2019.  
    https://arxiv.org/abs/1911.05507

17. **Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention.** Tsendsuren Munkhdalai et al. arXiv:2404.07143, 2024.  
    https://arxiv.org/abs/2404.07143

18. **Retentive Network: A Successor to Transformer for Large Language Models.** Yutao Sun et al. arXiv:2307.08621, 2023.  
    https://arxiv.org/abs/2307.08621

19. **Differentiable Neural Computers — Research overview.** Google DeepMind, 2016.  
    https://deepmind.google/blog/differentiable-neural-computers/


20. **B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory.** Luca Zancato et al. NeurIPS 2024.  
    https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb6281d3f2cfe2eb8b1307227d3677b3-Abstract-Conference.html

21. **TNT: Improving Chunkwise Training for Test-Time Memorization.** Zeman Li et al. ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html

22. **Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.** Liliang Ren et al. ICLR 2025.  
    https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html

23. **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.** Soham De et al. arXiv:2402.19427, 2024 preprint.  
    https://arxiv.org/abs/2402.19427

24. **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.** Zihan Qiu et al. NeurIPS 2025.  
    https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html

25. **RAttention: Towards the Minimal Sliding Window Size in Local-Global Attention Models.** Bailin Wang, Chang Lan, Chong Wang, Ruoming Pang. arXiv:2506.15545, 2025; Apple Machine Learning Research page, September 2025.  
    https://arxiv.org/abs/2506.15545  
    https://machinelearning.apple.com/research/rattention

26. **Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning.** NVIDIA. Technical Report, 2025.  
    https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf

27. **Frozen Memory Is Not Enough: Rethinking External Memory as Extraction.** Mingyuan Li, Guangsheng Yu, Xu Wang, Shaoxiong Ji. arXiv:2608.17050, 2026 preprint (version 5, 2026-10-24).  
    https://arxiv.org/abs/2608.17050

28. **LatentPort: Beyond KV Cache — Cross-Model Transfer of Recurrent Memory in Hybrid Language Models.** Simon P. Villani. arXiv:2609.25053, 2026 preprint.  
    https://arxiv.org/abs/2609.25053

29. **Pluggable Neural Machine Translation Models via Memory-augmented Adapters.** Yuzhuang Xu et al. LREC-COLING 2024.  
    https://aclanthology.org/2024.lrec-main.1120/

30. **Mamba-3: Improved Sequence Modeling using State Space Principles.** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu. ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html  
    https://arxiv.org/abs/2603.15569

31. **Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling.** Hyunji Lee et al. arXiv:2510.26912, 2025 preprint.  
    https://arxiv.org/abs/2510.26912

---

**文档结束。**
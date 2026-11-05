# EPCLM Stateful Computation & Routing Experiment 001
## v0.3 — Chunk-Parallel Execution Redesign

**项目：** MiniCells / EPCLM
**仓库：** `ArcheLabs/mini-cells`
**文档类型：** 训练前机制修订与实施规范
**状态：** PROPOSED / NOT TRAINED
**优先级：** v0.3 高于 v0.2 和 v0.1，冲突时以本文件为准
**实验时间上限：** 180 分钟
**实施对象：** Codex

---

# 1. 修订目标

## 1.1 背景

EPCLM Stateful Computation & Routing Experiment 001 的 v0.2 采用逐 Token 递归计算：

1. 当前 Token 读取历史 KV。
2. Router 选择 Cell。
3. CellCore 执行计算。
4. 更新选中 Cell 的 State。
5. 再次路由并执行第二轮 CellCore。
6. 当前 Token 的最终表示写入 KV。
7. 下一个 Token 才能继续计算。

因此，即使 M0 不包含 Persistent State，也存在严格的 Token 间递归依赖。

此前 Kaggle Warm-up 结果：

| 模型 | Training Targets/s |
|---|---:|
| T0 | 2,249.5 |
| M0 | 117.4 |
| M1 | 96.8 |
| M2 | 89.1 |

在原有执行方式下，四模型各完成 250K 有效预测目标，估计需要约 129 分钟，超过 80 分钟训练窗口。

这些速度属于 v0.2 的实测值，不能用于推断 v0.3 的最终性能。

## 1.2 本次修订的核心目标

**消除 Token 之间不必要的计算递归，允许同一个 Chunk 内的所有 Token 并行完成 Attention、Cell Routing 和 Cell Computation。**

保留：

- Shared Token Workspace。
- 动态 Cell Routing。
- K=2 的顺序计算步骤。
- Persistent Cell State。
- State-aware Routing。
- T0/M0/M1/M2 四模型。
- 完整文档训练。
- 三小时实验上限。

修改：

- Token KV 的生成时机及来源。
- State 的读取和更新时序。
- Cell Write 的聚合方式。
- 跨 Chunk 的反向传播机制。
- 训练前向与流式推理之间的等价性定义。

## 1.3 关键研究边界

v0.3 不要求与 v0.2 的数值输出等价。

这是一个新的计算机制，不是原模型的 Kernel 优化。

因此：

- v0.2 Checkpoint 不得直接作为 v0.3 的训练初始化。
- v0.2 Warm-up 吞吐不得当作 v0.3 吞吐。
- v0.2 实验结果只能作为历史执行成本参考。
- 所有 v0.3 模型必须重新执行正确性测试和测速。

---

# 2. 核心机制：两个时间尺度

v0.3 将 EPCLM 分为两个计算过程。

## 2.1 Fast Path — Token Computation

在一个 Chunk 内并行执行：

- Token Embedding。
- Causal Token Attention。
- Dynamic Cell Routing。
- Selected CellCore。
- Token Representation Transformation。
- Language Prediction。
- State Write Proposal。

此过程不要求前一个 Token 的 Cell 计算先完成。

## 2.2 Slow Path — Persistent State Evolution

在 Chunk 完成后：

- 按 Cell 聚合本 Chunk 的 Write Proposal。
- 更新被调用 Cell 的 Persistent State。
- 未被调用的 Cell 保持不变。
- 将新的 State 提供给下一个 Chunk。

因此：

\[
S_{c+1}=U(S_c,M_c)
\]

其中 \(M_c\) 为当前 Chunk 所有有效 Cell 计算产生的写入信息。

重要规则：

**当前 Chunk 的预测只能使用 Chunk 开始时的 State。当前 Chunk 产生的新 State 只能供后续 Chunk 使用。**

---

# 3. Shared Token Workspace 的重新定义

## 3.1 取消 Final-Output KV Recursion

v0.2：

\[
KV_t=W_{KV}h_t^{final}
\]

由于 \(h_t^{final}\) 依赖前面 Token 的最终表示，因此造成 Token 递归。

v0.3：

\[
K_t=W_K\phi(x_t,p_t)
\]

\[
V_t=W_V\phi(x_t,p_t)
\]

其中：

- \(x_t\) 是当前 Token 的输入表示。
- \(p_t\) 是位置相关信息。
- \(\phi\) 是独立于 Cell 输出和动态 State 的输入侧投影。
- 所有 Token 的 K/V 可以并行产生。

首轮可以令 \(\phi\) 为 Embedding 后的 RMSNorm 与线性投影，不引入额外复杂编码器。

## 3.2 Parallel Causal Attention

对于 Chunk \(c\)：

\[
X_c=[x_{c,1},\ldots,x_{c,B}]
\]

一次性生成：

\[
Q_c,\ K_c,\ V_c
\]

然后：

\[
A_c=
\operatorname{CausalMHA}
\left(
Q_c,
[K_{<c};K_c],
[V_{<c};V_c]
\right)
\]

Chunk 内使用严格 causal mask，允许当前 Token 访问自身及此前 Token 的输入侧 K/V，不允许访问未来 Token。

输出：

\[
H_c^{(0)}=X_c+W_OA_c
\]

## 3.3 Workspace 约束

1. 所有 Cellular 模型使用同一 Shared Token Workspace 定义。
2. 每个 Token 只生成一份 K/V。
3. KV 不来自 CellCore 的最终输出。
4. 不再执行 Final Token Representation → KV 的递归写入。
5. 所有 Cell 共享 Token Workspace。
6. 不为不同 Cell 创建完整独立 KV Cache。
7. 不将不同 Cell 的中间表示直接堆入 Token KV。
8. 历史 KV 仍采用原定的上下文窗口约束。
9. Token KV 与 Cell Persistent State 必须分别存储。
10. Workspace 投影在所有正式实验组中保持相同架构。

## 3.4 重要架构代价

这一修改使 Token Attention 能够并行，但也改变了 Workspace 的表达能力。

历史 Token 的 K/V 不再包含它经过 Cellular Computation 后的最终表示。

因此，不能预先假设 v0.3 一定保留 v0.2 的语言质量。

本轮必须实际测量这一改变的代价。

---

# 4. Chunk 内的 Parallel Cellular Computation

## 4.1 固定计算深度

首轮保留：

- Cell 数量：16。
- 每个 Token 两步 Cellular Computation。
- 每步 Top-1 Cell Selection。
- CellCore 参数共享。
- 不使用 Cell-to-Cell Graph。
- 不使用动态停止。
- 不增加重复 FFN。

以上设计维持与此前四模型研究相同的计算组织问题。

## 4.2 Chunk State Snapshot

令当前 Chunk 开始时的状态为：

\[
S_c=\{s_{c,1},\ldots,s_{c,16}\}
\]

整个 Chunk 内所有 Token 和两步 Router 使用同一个冻结的 State Snapshot。

这里的“冻结”指当前 Chunk 内不修改其数值，不代表切断对前面 Chunk 的梯度。

因此：

\[
S_{c,t}=S_c,\quad t\in c
\]

## 4.3 Step 1 — Parallel Routing

当前 Chunk 所有 Token 的表示：

\[
H_c^{(0)}\in\mathbb R^{B\times d}
\]

一次计算所有 Token 与 16 个 Cell 的 Router Score。

M0/M1：

\[
r_{t,i}^{(0)}
=
\frac{(W_Rh_t^{(0)})^\top W_Ee_i}{\sqrt{d_r}}
\]

M2：

\[
r_{t,i}^{(0)}
=
\frac{(W_Rh_t^{(0)})^\top
(W_Ee_i+W_Ss_{c,i})}{\sqrt{d_r}}
\]

Router 按原 v0.2 的 Categorical Policy 训练。

每个 Token 只选择一个 Cell。

## 4.4 Step 1 — Parallel CellCore

将所有 Token 选中的 Cell Identity 和 State 按 Batch 索引 Gather。

一次性执行共享 CellCore：

\[
u_t^{(0)}
=
F_\theta(h_t^{(0)},e_{i_t},s_{c,i_t})
\]

M0 不读取 State。

更新：

\[
h_t^{(1)}
=
h_t^{(0)}+W_Cu_t^{(0)}
\]

要求：

- 实际只执行被选择的 CellCore。
- 不计算全部 16 个 Cell 的候选输出。
- 各 Token 的 CellCore 可合并为批量张量运算。
- 不使用逐 Token 的 Python CellCore 调用。
- 记录实际有效 CellCore 调用数量。

## 4.5 Step 2 — Re-routing

第二步使用更新后的表示：

\[
r_{t,i}^{(1)}
=
Router(h_t^{(1)},e_i,S_c)
\]

因此，第二步可以选择与第一步不同的 Cell。

随后：

\[
u_t^{(1)}
=
F_\theta(h_t^{(1)},e_{j_t},s_{c,j_t})
\]

\[
h_t^{(2)}
=
h_t^{(1)}+W_Cu_t^{(1)}
\]

注意：

即使连续两步选择相同 Cell，第二步也读取 Chunk 开始时的同一个 State Snapshot。

v0.2 的“第二步读取第一步刚更新的 State”不再适用。

## 4.6 Language Prediction

最终：

\[
logits_t=LMHead(h_t^{(2)})
\]

所有 Token 的预测并行完成。

必须保证：

\[
logits_t=f(x_{\leq t},S_c)
\]

不能依赖当前 Chunk 内未来 Token 的输入或写入结果。

---

# 5. State Write Proposal 与聚合

## 5.1 不再即时更新 State

每次 CellCore 调用只产生：

- Token 计算结果。
- State Write Proposal。
- Cell ID。
- Router Action 信息。

但不直接修改持久 State。

## 5.2 State Write Proposal

对第 \(k\) 步选中的 Cell：

\[
w_{t,k}=W_{write}
[h_t^{(k)},u_t^{(k)},e_{i_{t,k}}]
\]

可以采用共享线性投影及轻量非线性。

不得为每个 Cell 创建一套独立 Writer。

M1/M2 使用相同的 Writer 结构。

M0 不创建或更新 Persistent State；可以计算用于诊断的选择计数，但不执行无效的 State Writer。

## 5.3 按 Cell 聚合

对于 Cell \(i\)，定义：

\[
n_i=
\sum_{t,k}\mathbf 1[i_{t,k}=i]
\]

\[
m_i=
\frac{
\sum_{t,k}\mathbf 1[i_{t,k}=i]w_{t,k}
}{
\max(n_i,1)
}
\]

聚合范围只包含当前文档、当前 Chunk 的有效 Token。

Padding、无效目标和其他文档的 Write Proposal 不得参与聚合。

如果一个 Cell 在两步计算中被同一 Token 多次选中，按两次真实调用贡献统计。

## 5.4 Chunk State Update

Chunk 完成后：

\[
s_{c+1,i}=
\begin{cases}
U_\phi(s_{c,i},m_i,n_i),&n_i>0\\
s_{c,i},&n_i=0
\end{cases}
\]

首轮沿用共享的轻量 Gated Update 或审计通过的 GRU Update。

更新器必须支持：

- 相同参数跨 Cell 共享。
- 只更新活跃 Cell。
- 未选中 Cell 状态不变。
- FP32 State Update。
- 非原地、可微的 State 替换。

## 5.5 数值与实现约束

优先使用张量化的 Scatter-Add / Segment Reduction。

禁止：

- 为每个 Token 启动一个 GRU Kernel。
- 在 Python 中逐 Token、逐 Cell 聚合。
- 在每个 Write Proposal 产生时立即修改 State。
- 跨文档聚合。
- 因 GPU Batch 排列顺序改变而改变文档的 State 所属关系。

浮点归约顺序可能影响末位精度。正确性测试应使用明确的数值容差，而不是要求不同 GPU Kernel 逐位完全一致。

---

# 6. T0 / M0 / M1 / M2 的最终定义

| 模型 | Token Attention | 动态 Cell Routing | Persistent State | State-aware Router |
|---|---|---|---|---|
| T0 | 标准 Transformer MHA | 无 | 无 | 无 |
| M0 | Parallel Shared Workspace MHA | 是 | 无 | 无 |
| M1 | 同 M0 | 是 | 有 | 无直接输入 |
| M2 | 同 M0 | 是 | 有 | 有 |

## 6.1 T0

保持标准现代 Transformer：

- Decoder-only。
- Causal MHA。
- Pre-Norm。
- RMSNorm。
- RoPE。
- SwiGLU。
- Residual。
- 标准逐层 KV Cache。

不增加 GQA、MLA 或新 Attention 算法。

T0 可以使用标准并行 Transformer Training。

必须保持四模型的数据、Tokenizer、训练 Token、优化器和评估口径一致。

## 6.2 M0

M0 是 Chunk-Parallel Cellular 计算的无状态对照。

- Shared Token Workspace。
- K=2。
- Top-1 Routing。
- Shared CellCore。
- No Persistent State。

它测试的是整体 Cellular 计算组织，而不是单独证明动态路由优于固定 Layer。

## 6.3 M1

M1 在 M0 基础上增加：

- Cell Persistent State。
- State-conditioned CellCore。
- Chunk-end State Writer。
- Chunk-end State Update。

Router 不直接读取 State。

但是，当前 Token 表示可能通过过去的 State 计算间接受影响，因此不能声称 Router 与 State 完全独立。

## 6.4 M2

M2 与 M1 相同，但 Router Score 增加 State 项：

\[
r_{t,i}
=
\frac{q_t^\top(W_Ee_i+W_Ss_{c,i})}{\sqrt{d_r}}
\]

M1 与 M2 的唯一核心机制差异应当是 Router 能否直接读取 State。

不额外增加 Cell FFN、Graph 或 Attention。

---

# 7. 训练并行与因果正确性

## 7.1 Chunk 内并行

在同一个 Chunk 中：

- Token Attention 并行。
- Router Scores 并行。
- CellCore Calls 批量执行。
- 两个 Routing Steps 保持顺序。
- State Writer 批量聚合。
- Chunk 完成后才更新 State。

因此，同一个 Chunk 内不存在 Token-by-Token Cell 状态递归。

## 7.2 Chunk 间依赖

\[
S_{c+1}=U(S_c,M_c)
\]

下一 Chunk：

\[
H_{c+1}=F(X_{c+1},KV_{\le c+1},S_{c+1})
\]

Chunk 之间仍然存在 State 依赖。

本次只要求消除 Token 级串行，不宣称整个文档完全并行。

## 7.3 防止未来信息泄漏

假设 Chunk 长度为 64。

当前 Chunk 的所有 Token 可以参与计算 State Write Proposal。

但聚合后的新 State：

\[
S_{c+1}
\]

只能在下一 Chunk 开始使用。

不得让本 Chunk 较早 Token 读取包含本 Chunk 较晚 Token 的新 State。

必须验证：

- 修改未来 Token 不改变此前位置的 Logits。
- 修改下一 Chunk 不改变当前 Chunk 的预测。
- Chunk 末尾产生的 State 可以影响下一 Chunk。
- State 不跨文档泄漏。

---

# 8. TBPTT 与跨 Chunk 梯度

## 8.1 关键问题

本次 State 只在 Chunk 结束后更新。

因此，当前 Chunk 的预测主要使用旧 State，而不是当前 Chunk 新写入的 State。

如果每个 Chunk 完成后立即 Detach State：

\[
S_{c+1}=\operatorname{Detach}(U(S_c,M_c))
\]

那么下一 Chunk 的语言损失就无法通过普通反向传播训练当前 Chunk 的 Writer。

这会严重削弱 Persistent State 学习。

**因此，v0.3 禁止采用每个 Chunk 立即 Detach State 的默认规则。**

## 8.2 梯度窗口

定义：

\[
H=\text{TBPTT Horizon in Chunks}
\]

首轮冻结：

\[
H=2
\]

即至少允许下一 Chunk 的损失反向传播到前一 Chunk 的 State Update。

如果硬件允许，可以在后续独立实验中研究更长窗口，但本轮不进行 Horizon 搜索。

## 8.3 KV 与 State 的梯度范围

采用以下规则：

- State 的计算图跨越连续两个 Chunk。
- 到达 TBPTT 窗口边界后 Detach State，保留数值。
- KV 的跨 Chunk 历史按相同预设的截断边界处理。
- Detach 不得删除 KV 或 State 的数值。
- 文档边界同时 Reset KV 和 State。
- Optimizer Step 只能在对应冻结文档组完成梯度累积后执行。
- 参数更新之后不得继续使用旧参数生成的缓存。

注意：

对于不依赖前一 Chunk Cell 输出的输入侧 KV，可以采用预计算和冻结缓存优化；但不能无意切断当前 Chunk 内本来应存在的 Workspace Projection 梯度。

## 8.4 TBPTT 的限制

H=2 只能让部分状态更新直接获得下一 Chunk 的语言损失反馈。

跨越更远 Chunk 的依赖仍然可能被截断。

实验报告必须记录：

- State 梯度窗口。
- KV 梯度窗口。
- 实际跨 Chunk 梯度是否非零。
- 有多少 State Write 能够获得后续 Chunk 的梯度。
- 是否发生显存溢出或 Detach 错误。

不能将 H=2 的结果解释为已经学会任意长度的长期记忆。

---

# 9. Router 的训练定义

## 9.1 保留 v0.2 的 Categorical Router

训练：

\[
a_{t,k}\sim
Categorical(Softmax(r_{t,k}/1.0))
\]

推理：

\[
a_{t,k}=\operatorname{Argmax}(r_{t,k})
\]

每个 Token、每个 Routing Step 只执行一个选中的 CellCore。

禁止重新引入未定义的 Straight-through。

## 9.2 Policy Gradient

保持原有 Policy Gradient 思路。

但必须考虑新的跨 Chunk State 作用。

一个路由动作可能影响：

1. 当前 Token 的预测。
2. 当前 Chunk 写入哪个 Cell。
3. 后续 Chunk 读取到什么 State。
4. 后续 Chunk 的 Router 和 Cell 计算。

因此，Router 的 Credit Assignment 不应只考虑当前 Token 的 CE。

## 9.3 首轮截断 Return

首轮建议采用：

\[
G_{t,k}
=
\ell_t+
\beta\overline{\mathcal L}_{c+1}
\]

其中：

- \(\ell_t\)：当前 Token 的 CE。
- \(\overline{\mathcal L}_{c+1}\)：下一 Chunk 有效目标的平均 CE。
- \(\beta\)：跨 Chunk 权重。

首轮取：

\[
\beta=1
\]

最后一个 Chunk 没有下一 Chunk 时，该项为零。

训练目标：

\[
\mathcal L=
\mathcal L_{CE}
+
\lambda_{PG}
\mathcal L_{PG}
\]

\[
\mathcal L_{PG}
=
\frac{1}{N_{actions}}
\sum_{t,k}
\operatorname{stopgrad}(G_{t,k}-b_k^{old})
\log\pi_\theta(a_{t,k})
\]

使用每个 Routing Step 独立的 EMA Baseline。

推荐首轮冻结：

- `temperature = 1.0`
- `ema_decay = 0.98`
- `lambda_pg = 0.1`

这些是开发实验超参数，不是经过验证的最优值。

## 9.4 梯度和 Baseline 更新

- Baseline 使用本次 Optimizer Step 之前的旧值。
- 先计算 Policy Loss。
- 成功完成 Optimizer Step 后更新 Baseline。
- 若发生 Overflow 或 Optimizer Step 失败，不推进 Baseline。
- 同时记录 CE Gradient 与 Router Gradient。
- 不得仅通过非零 Router 梯度认定路由已经学会有效分工。

## 9.5 重要限制

本次 Return 只覆盖有限的未来损失。

它不是完整文档目标的无偏 Policy Gradient 估计。

报告必须明确：

`TRUNCATED_ROUTER_CREDIT_ASSIGNMENT`

不允许宣称解决了完整长程信用分配问题。

---

# 10. Chunk Size 与并行效率

## 10.1 推荐候选

性能预检比较：

- Chunk Size 32。
- Chunk Size 64。
- Chunk Size 128。

不得直接假设 Chunk 越大越好。

更大的 Chunk 有利于 GPU 并行，但会降低 State 更新频率，并增加单次 Chunk 的工作量。

## 10.2 正式实验选择

本轮建议默认：

`chunk_size = 64`

但只有在真实 GPU Benchmark 证明它满足成本和正确性要求后才能冻结。

训练开始前：

1. 测试 32/64/128 的正确性。
2. 测量吞吐和显存。
3. 按统一规则选择一个 Chunk Size。
4. 写入 Protocol。
5. 四个模型使用相同文档划分。
6. 正式训练期间不得修改 Chunk Size。

T0 不具有 Persistent State，但必须使用相同有效训练目标和上下文评价条件。

## 10.3 不同文档并行

独立文档可以进一步形成 Batch。

要求：

- 每个文档单独维护 State。
- 每个文档单独维护历史 KV。
- Padding 不参与 State 聚合。
- 不同文档之间不得互相 Attention。
- 同一个文档内部 Chunk 顺序不变。
- 统一统计有效预测目标数。

---

# 11. 训练数据与协议

继续沿用 v0.2 的可靠文档流要求。

必须存在：

`document_index.jsonl`

记录：

- Document ID。
- Split。
- Token 范围。
- Tokenizer SHA。
- Document SHA。
- EOS 策略。
- Chunk 边界。
- 有效预测目标数。
- Loss Mask。

不能将随机 125 Token 片段直接称为完整文档。

## 11.1 统一训练预算

候选：

- 1M。
- 500K。
- 250K。

预算按真正参与 Loss 的预测目标计算。

完整文档组向下取整规则沿用 v0.2，四模型使用同一份冻结的文档顺序和有效目标预算。

训练前必须重新计算新的吞吐量。

旧 v0.2 的 117.4 / 96.8 / 89.1 Targets/s 不可直接用于新版本预算判断。

## 11.2 初始化与复现

- 四模型公共模块使用匹配的初始化策略。
- M0/M1/M2 公共参数使用相同的初始值。
- State 模块采用独立 RNG。
- Warm-up 结束后重置全部模型、优化器、Scaler、Router Baseline、RNG 和数据游标。
- 记录 source、implementation、protocol 和 dataset Hash。
- Independent Test 不参与开发筛选。

---

# 12. 正确性测试

本轮必须新增并通过：

| 测试 | 要求 |
|---|---|
| `test_chunk_parallel_causality` | 当前 Logits 不受未来 Token 影响 |
| `test_workspace_input_kv` | KV 来源为输入侧表示 |
| `test_workspace_parallel_equivalence` | 并行 Chunk Attention 与同定义参考实现等价 |
| `test_workspace_streaming_equivalence` | 流式处理与并行 Prefill 等价 |
| `test_no_final_output_kv_recursion` | 最终 Cell 输出不进入当前 Workspace KV |
| `test_state_snapshot_consistency` | Chunk 内所有 Token 读取同一 State |
| `test_no_within_chunk_state_leak` | Chunk 内未来写入不影响当前预测 |
| `test_state_aggregate_per_cell` | Write 正确按 Cell 聚合 |
| `test_inactive_state_identity` | 未选中 Cell 不更新 |
| `test_cross_chunk_state_gradient` | 下个 Chunk Loss 能训练上个 Chunk Writer |
| `test_router_gradient` | Router Policy Gradient 有效 |
| `test_sparse_cellcore_execution` | 只执行被选择的 CellCore |
| `test_chunk_partition_consistency` | 固定逻辑 Chunk 边界不受 API 分段影响 |
| `test_document_state_isolation` | 文档间 State/KV 隔离 |
| `test_padding_write_mask` | Padding 不参与 State Write |
| `test_snapshot_restore` | State、KV 和待提交 Chunk 恢复一致 |

## 12.1 Snapshot / Streaming 语义

固定逻辑 Chunk 边界必须由文档内 Token 位置决定，而不是由 API 请求大小决定。

如果 Prompt 在 Chunk 中途结束：

- 不提前提交 State。
- 保留当前 Chunk 的未提交 Write Proposal。
- 后续生成继续使用当前 Chunk 的 State Snapshot。
- 达到固定 Chunk 边界后再更新 State。

Snapshot 必须包含：

- 已提交 Cell State。
- 当前 Chunk State Snapshot。
- 当前 Chunk 未提交写入的聚合信息。
- Token KV。
- 当前文档内 Token 位置。
- Router RNG（如需复现采样路径）。

---

# 13. 评价指标

## 13.1 Language Modeling

主要指标：

- Validation NLL。
- Validation PPL。
- Per-document Paired NLL。
- 固定 Seed 的 Bootstrap CI。
- Learning Curve。

四模型使用相同：

- Validation 文档。
- 前缀。
- Chunk 划分。
- Loss Mask。
- 正式计分位置。

训练采用 Categorical Router，正式验证采用 Argmax。

如时间允许，可补充固定随机种子的 Categorical Evaluation，用于诊断训练和推理路由差异。

## 13.2 State Diagnostics

M1/M2：

- State Norm。
- State Update Magnitude。
- State Read-Off。
- State Write-Off。
- Reset Every Chunk。
- Cell Pairwise Cosine Similarity。
- State Effective Rank。
- Cell Update Frequency。

M2 额外：

- Ignore-State-Router。
- Router State Contribution。
- Cell-State Shuffle。

## 13.3 Router Diagnostics

- 每个 Cell 的选择频率。
- Router Entropy。
- Dead Cell Ratio。
- Route Transition Matrix。
- Step-1 / Step-2 相同 Cell 比例。
- 每个 Chunk 的活跃 Cell 数量。
- 每个 Cell 的 Write Count。

## 13.4 性能指标

必须统计：

- Training Targets/s。
- Forward Time。
- Backward Time。
- FLOPs / Token。
- Active FLOPs / Token。
- Attention FLOPs。
- Router FLOPs。
- CellCore FLOPs。
- State Write / Update FLOPs。
- Peak VRAM。
- KV Memory。
- Persistent State Memory。
- Decode Median / P95。
- Prefill Throughput。
- 实际 GPU Kernel Launch 和同步热点。

不能只凭解析 FLOPs 声称获得真实加速。

---

# 14. 三小时预算与执行顺序

## 14.1 总时间上限

整个 Kaggle 实验运行：

\[
T_{\max}=180\text{ minutes}
\]

允许提前退出。

唯一计时起点包括：

- 环境准备。
- 依赖检查。
- 协议与资产检查。
- Smoke Tests。
- Warm-up。
- Training。
- Evaluation。
- Publication。

## 14.2 阶段时间预算

| 阶段 | 预算 | 累计截止 |
|---|---:|---:|
| Audit / Environment | 10 分钟 | 00:10 |
| Data / Protocol Integrity | 5 分钟 | 00:15 |
| Warm-up / Chunk Benchmark | 10 分钟 | 00:25 |
| Four-model Training | 80 分钟 | 01:45 |
| Evaluation / Ablations | 45 分钟 | 02:30 |
| Reporting / Publication | 15 分钟 | 02:45 |
| Buffer | 15 分钟 | 03:00 |

正式训练前所有机制实现和必需单元测试必须已经完成并通过。

## 14.3 Warm-up

必须覆盖：

- 真正的 Chunk Parallel Attention。
- K=2 Routing。
- Batched CellCore。
- State Write Aggregation。
- State Update。
- 至少两个连续 Chunk 的反向传播。
- Optimizer Step。
- 真实 KV 长度与文档流。

不允许只测试前向或空 KV。

## 14.4 预算选择

基于四模型实测速度选择统一 Token 预算。

候选：

1. 1M。
2. 500K。
3. 250K。

选择最高且能够在 80 分钟训练窗口内完成的预算，预留至少 20% 运行余量。

若最低预算也无法完成：

- 不降低单模型要求以制造结果。
- 不让某些模型训练更多 Token。
- 记录 `TIME_BUDGET_INSUFFICIENT`。
- 发布审计、测速及失败原因。
- 不启动正式训练。

## 14.5 硬停止

- 第 105 分钟后停止新的正式训练。
- 第 165 分钟后不再启动训练或评估。
- 第 180 分钟强制结束并发布当前可验证结果。

所有未完成实验标记 `NOT_RUN` 或 `INCOMPLETE`。

---

# 15. 新机制的主要风险

## 15.1 Workspace 表达能力下降

由于 KV 不再来自 Cell 的最终输出，历史 Token 的表示可能不如多层 Transformer 丰富。

这属于架构代价，不是可以忽略的实现细节。

## 15.2 Chunk 内 State 更新延迟

State 只能在下一 Chunk 被读取。

可能削弱模型对当前 Chunk 内刚产生的信息的立即记忆和适应。

## 15.3 State 聚合丢失顺序

Mean Aggregation 对 Token 顺序不敏感。

虽然 Writer 输入包含 Token 的上下文表示，但最终聚合仍可能丢失某些顺序信息。

首轮不增加复杂 Writer。先测量是否出现明显问题。

## 15.4 State 更新信用分配不足

H=2 只覆盖有限的跨 Chunk 梯度。

必须记录跨 Chunk Writer Gradient，而不是只观察 State 数值变化。

## 15.5 Router 收敛问题

Categorical + Policy Gradient 可能具有较高方差。

同时存在 Cell 使用集中、State 更新稀疏以及长程回报难以归因的问题。

不允许把 Router Entropy 降低直接解释为功能专业化。

## 15.6 Chunk Size 影响质量

改变 Chunk Size 会改变 State 更新频率和读取时序，不是纯粹的硬件优化。

因此，所有模型必须在正式训练前冻结同一个 Chunk Size。

---

# 16. Codex 实施计划

## Phase A — 机制改造

修改 Workspace：

- K/V 从 Token 输入侧生成。
- 移除 Final-output KV Recursion。
- 支持 Chunk Parallel Causal Attention。
- 保留流式 KV 访问与缓存。

修改 Cell：

- Token 维度批量 Routing。
- Token 维度批量 CellCore。
- 固定 K=2。
- State Snapshot。

修改 State：

- Write Proposal。
- Per-cell Aggregation。
- Chunk-end Update。
- 不活跃 Cell No-op。
- FP32 Update。

## Phase B — 梯度与训练

实现：

- H=2 TBPTT。
- 跨 Chunk State Writer Gradient。
- KV/State 截断边界。
- 完整文档组 Optimizer Update。
- Token-weighted Loss。
- Policy Gradient。
- Overflow / Retry 一致性。

## Phase C — 正确性

运行第 12 节全部必需测试。

任何因果、跨 Chunk 梯度、State 生命周期或 Workspace 等价测试失败，禁止正式训练。

## Phase D — GPU 性能预检

记录 Chunk 32/64/128：

- Throughput。
- GPU VRAM。
- Forward / Backward Time。
- State Update Cost。
- Router Cost。

在正式训练前冻结 Chunk Size。

## Phase E — 四模型实验

训练：

- T0。
- M0。
- M1。
- M2。

不得增加第五组。

## Phase F — Evaluation & Publication

完成：

- PPL / NLL。
- State Ablations。
- Router Diagnostics。
- Cost Comparison。
- Research Report。
- Publication Receipt。

---

# 17. 发布产物

建议独立使用 v0.3 结果目录，避免覆盖 v0.2：

`artifacts/experiments/epclm-stateful-computation-routing-001/v0.3/`

必须包含：

- `audit.json`
- `protocol.json`
- `data-manifest.json`
- `warmup.json`
- `correctness.json`
- `training-results.json`
- `validation-results.json`
- `state-diagnostics.json`
- `routing-diagnostics.json`
- `cost-report.json`
- `decision.json`
- `RESEARCH-REPORT.zh-CN.md`
- `publication-receipt.json`

报告必须标注：

`MECHANISM_VERSION = chunk-parallel-v0.3`

并记录：

- Source Commit。
- Implementation Hash。
- Protocol Hash。
- Data / Tokenizer Hash。
- GPU Environment。
- Chunk Size。
- TBPTT Horizon。
- 实际有效训练目标数。
- 实际各阶段耗时。
- 未运行项目。
- Scientific Confirmation = false。

---

# 18. 成功标准

本次最重要的目标不是立即超过历史 44.987 PPL，而是验证一种可并行训练的 Cellular 计算机制。

## G0 — Correctness

- 因果正确。
- Chunk Parallel 与参考实现等价。
- State 正确持续与更新。
- Router 正常训练。
- 跨 Chunk Writer Gradient 非零。

## G1 — Execution Efficiency

- M0/M1/M2 的训练吞吐相对于 v0.2 有明确改善。
- 能够在 80 分钟训练窗口内完成共同最低预算。
- 真实 CellCore 计算没有退化为全部 Cell 密集计算。

## G2 — Language Modeling

- 四模型完成同协议 PPL/NLL 比较。
- 记录与 T0 的质量和成本差距。
- 不把历史 PPL 当作严格同协议结果。

## G3 — State Utility

- M1 与 M0 的差异可测量。
- M2 与 M1 的差异可测量。
- State Read / Write / Reset 消融能够检验状态贡献。
- 能区分静态 Cell Identity、动态 State 和 State-aware Routing 的作用。

---

# 19. 最终执行要求

Codex 必须先完成机制改造和正确性测试，再启动正式 Kaggle 训练。

本次不得通过以下方式改善速度：

- 降低训练 Token 预算但不更新协议。
- 删除两步 CellCore。
- 移除 Persistent State。
- 将 Router 改为固定路由。
- 禁止真实 State Update。
- 让全部 Cell 密集执行却宣称稀疏。
- 在 Chunk 内使用未来 State。
- 只训练短 Chunk 后宣称长期记忆能力。

允许：

- Chunk Parallel Attention。
- Batched Routing。
- Batched CellCore。
- Scatter / Segment Reduction。
- 预分配 KV。
- 独立文档并行。
- 合理 Kernel Fusion。
- 不改变数学定义的内存和调度优化。

---

# 20. 最终研究命题

**Can persistent stateful cellular computation be organized into parallel token-level computation and chunk-level state evolution, preserving dynamic routing while avoiding token-by-token recurrent training?**

中文：

**能否将持久化 Cellular Computation 组织为 Token 级并行计算与 Chunk 级状态演化，在保留动态 Cell Routing 和持续状态能力的同时，避免逐 Token 递归训练？**

本轮实验的首要目标，是证明这种计算组织具有正确的因果语义、可训练性以及可接受的 GPU 效率。

只有在这些条件成立之后，才有意义继续研究 EPCLM 是否能够在语言质量、持续状态利用和计算效率上超越标准 Transformer。

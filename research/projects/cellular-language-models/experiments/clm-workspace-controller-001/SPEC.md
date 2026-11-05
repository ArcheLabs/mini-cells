# CLM v0.4 实验执行规范

**项目：** MiniCells / EPCLM
**实验名称：** Shared Workspace & Dynamic Cellular Computation
**实验编号：** `clm-workspace-controller-001`
**状态：** Proposed / Awaiting Implementation
**研究性质：** 架构验证、机制消融、基线比较

---

## 1. 实验目标

### 1.1 背景

此前 EPCLM v0.3 实验已经完成 Chunk-Parallel 训练机制的工程验证，但尚未证明 Stateful Cell 和 State-aware Routing 的独立收益。

已观察到的主要问题包括：

- Router 的 Softmax Entropy 接近均匀分布的理论上限。
- 不同 Cell 是否形成有价值的计算分化尚未得到充分证明。
- Cell State 对最终预测损失的贡献较小。
- 共享 CellCore、Cell Identity 和 State 联合训练可能存在相互依赖。
- Cellular 模型与 Transformer 基线在参数规模、Attention 次数等方面存在差异。

本次实验不再以 Cell-local Persistent State 为架构基础，而是回到信息交互、计算执行与计算控制三个独立机制。

### 1.2 核心研究假设

**H1：Shared Workspace**

历史 Token 的信息可以通过统一的、因果共享的 KV Workspace 提供，不必为每个 Cell 建立独立的历史 KV。

**H2：Dynamic Cellular Computation**

在相同信息访问条件下，根据当前计算表示动态选择 Cell，能够比固定计算路径获得更好的计算质量或效率。

**H3：Computational Controller State**

一个在当前 Token 内部递归更新的高维 Controller State，可以改善后续 Cell 的选择，而不要求每个 Cell 自身拥有持久状态。

**H4：Repeated Workspace Interaction**

不同计算步骤利用更新后的 Query 再次读取共享 Workspace，可能比只进行一次历史信息交互更加有效。

### 1.3 本次不研究的内容

本次明确排除：

- Cell-local Persistent State。
- 跨 Token 的 Controller State 继承。
- 动态 STOP 与可变计算步数。
- Cell-to-Cell 固定连接图。
- 显式 Cell 能力向量。
- Computational Demand 的独立监督模型。
- 强化学习式硬路由。
- 在线持续学习和模型参数在线更新。
- GPU-PVM、JAM 或 MiniJAM 链上执行性能。

这些内容可以保留为未来研究方向，但不能进入本次主实验，避免同时引入过多无法分辨的变量。

---

## 2. 核心架构定义

CLM v0.4 由三个模块构成：

1. Shared Token Workspace
2. Computational Cells
3. Computational Controller

其中 Workspace 负责信息交互，Cell 负责非线性计算，Controller 负责计算选择及计算过程状态更新。

### 2.1 Shared Token Workspace

给定 Token 序列：

\[
X=(x_1,x_2,\ldots,x_T)
\]

首先通过一个因果 Workspace Encoder：

\[
u_t=E_{\mathrm{ws}}(x_{\leq t})
\]

生成统一的 Token 表示。

随后使用共享投影：

\[
K_t=W_Ku_t
\]

\[
V_t=W_Vu_t
\]

形成：

\[
M=\{(K_t,V_t)\}_{t=1}^{T}
\]

#### Workspace 规则

- 每个 Token 生成一组逻辑上共享的 K/V 表示。
- 多头 Attention 可以具有多个 Head 的 K/V，但不因 Cell 或计算步骤而生成独立副本。
- Workspace KV 不记录 Cell ID。
- Workspace KV 不记录 Cell 计算步骤。
- Workspace KV 不由 Cell 的中间输出生成。
- 同一个 Token 的全部 Cell 计算步骤读取相同的历史 KV 集合。
- 不同步骤可以生成不同 Query。
- Workspace Encoder 与整个模型端到端训练。
- Workspace 必须满足严格因果约束。

注意：如果 Workspace Encoder 自身采用 Causal Self-Attention，则在自回归推理中，Encoder 可能还需要内部 KV Cache。它与提供给 Cell 的 Shared Workspace KV 是两个逻辑对象。

因此，不能宣称整个模型只有一份物理 KV Cache。必须记录 Encoder Cache 与 Shared Workspace Cache 的总成本。

#### Workspace Encoder

初始实现采用：

- 1 个 Causal Transformer Block。
- Pre-LayerNorm。
- Rotary Position Embedding。
- 标准 Causal Self-Attention。
- SwiGLU FFN。
- 不使用独立 Cell 输出更新 Workspace。

这是一个有意识的工程折中：Workspace 的信息编码仍使用固定结构，但主要的后续计算由动态 Cell 系统承担。

### 2.2 Computation State

对于 Token \(t\)，定义：

\[
h_{t,k}\in\mathbb R^d
\]

为它执行第 \(k\) 次 Cellular Computation 前的表示。

初始化：

\[
h_{t,0}=u_t
\]

所有 Cell 使用相同的输入输出维度。

### 2.3 Repeated Attention Read

每次 Cell 计算前，使用当前表示生成 Query：

\[
q_{t,k}=W_Q\operatorname{LN}(h_{t,k})
\]

从共享 Workspace 读取：

\[
r_{t,k}
=
W_O\operatorname{Attention}
(q_{t,k},K_{\leq t},V_{\leq t})
\]

形成待计算表示：

\[
v_{t,k}
=
\operatorname{LN}(h_{t,k}+r_{t,k})
\]

所有计算步骤共享 Query Projection 和 Attention Output Projection 参数。

不允许为第 1 步、第 2 步分别建立独立参数化的 Attention Layer，否则容易重新引入固定层级。

**重要实验控制：**

主实验中，Query 只由当前计算表示 \(h_{t,k}\) 产生，不直接读取 Controller State。

这样可以独立检验 Controller State 对 Cell 路由的贡献。

将来可以通过单独的扩展实验：

\[
q_{t,k}=Q(h_{t,k},z_{t,k})
\]

研究 State 是否也应该参与信息检索，但不能混入本次主实验。

### 2.4 Computational Cell

设共有 \(N\) 个 Cell：

\[
\mathcal C=\{C_1,C_2,\ldots,C_N\}
\]

每个 Cell 具有独立参数：

\[
C_i(v;\theta_i)
\]

初始使用 SwiGLU FFN：

\[
C_i(v)
=
W_{o,i}
\left[
\operatorname{SiLU}(W_{g,i}v)
\odot(W_{u,i}v)
\right]
\]

所有 Cell：

- 使用相同结构。
- 拥有独立权重。
- 接收相同维度的输入。
- 产生相同维度的输出。
- 不拥有持久 State。
- 不具有固定层级。
- 不具有预定义职责。
- 不拥有固定的下一 Cell 连接。

允许同一个 Cell 在不同计算步骤被重复选择。

本次禁止通过参数共享的 CellCore + Identity 来替代独立 Cell，以减少此前实验中出现的分化不充分风险。

### 2.5 Sparse Cell Routing

Controller 为每个 Cell 输出一个 Logit：

\[
l_{t,k}=R_\phi(v_{t,k},z_{t,k})
\]

选择：

\[
S_{t,k}=\operatorname{TopK}(l_{t,k},2)
\]

只对选中 Cell 的 Logit 执行 Softmax：

\[
g_{t,k,i}
=
\operatorname{Softmax}
(l_{t,k}[S_{t,k}])_i
\]

然后：

\[
\Delta h_{t,k}
=
\sum_{i\in S_{t,k}}
g_{t,k,i}C_i(v_{t,k})
\]

最后：

\[
\boxed{
h_{t,k+1}
=
h_{t,k}+r_{t,k}+\Delta h_{t,k}
}
\]

这里 Top-2 表示同一步并行执行两个 Cell，并融合结果。

真正的顺序计算发生在相邻步骤之间。

必须明确：Top-K 的索引选择不可直接求导。梯度通过已选中的门控权重、Cell 输出与后续计算传播。未选中的 Cell 不会自动获得任务损失的反事实梯度。

### 2.6 Computational Controller State

定义：

\[
z_{t,k}\in\mathbb R^m
\]

Controller State 仅属于当前 Token 的计算过程。

初始状态：

\[
z_{t,0}=0
\]

使用零初始化可以避免在第一次路由中引入与无状态 Router 不同的初始化信息，从而更清晰地检验第二次路由是否受益于计算历史。

每次 Cell 执行后：

\[
z_{t,k+1}
=
U_\psi
\left(
z_{t,k},
h_{t,k},
\Delta h_{t,k},
r_{t,k},
g_{t,k}
\right)
\]

其中 \(g_{t,k}\) 可以表示为一个长度为 \(N\) 的稀疏门控向量，未选中 Cell 的权重为零。

初始 State Update 使用轻量 GRU 风格结构。

限制：

- Controller State 不直接进入 LM Head。
- Controller State 不直接修改主计算表示。
- Controller State 不跨 Token 继承。
- Controller 不允许内部建立完整 Token 历史存储。
- Controller 的计算成本必须独立统计。
- Controller State 必须能够被关闭以执行消融。

这使 State 的职责仅限于辅助后续计算决策。

### 2.7 计算次数

本次固定：

\[
K=2
\]

即每个 Token 执行两次 Cellular Computation。

每次独立生成路由结果，允许重复选择同一 Cell。

不实现 STOP。

---

## 3. Workspace KV 的因果语义

这是本实验最高优先级的正确性约束之一。

### 3.1 训练模式

对于完整训练序列，允许并行生成全部 Workspace 表示。

但必须保证：

\[
u_t=f(x_{\leq t})
\]

不能包含未来 Token 信息。

第 \(t\) 个位置的每次 Workspace Read 也只能访问：

\[
(K_j,V_j),\quad j\leq t
\]

当前 Token 可以访问自己的 KV，因为当前位置输入为 \(x_t\)，模型要预测的是 \(x_{t+1}\)。

### 3.2 自回归推理模式

对于新输入 Token \(x_t\)：

1. 使用 Encoder 的已有缓存生成 \(u_t\)。
2. 生成并保存当前 Token 的 Workspace K/V。
3. 初始化 \(h_{t,0}\) 和 \(z_{t,0}\)。
4. 执行两步 Cell 计算。
5. 生成下一 Token 的 Logits。
6. 采样下一个 Token 后重复。

当前 Token 的 Cell 输出不回写已有 Workspace KV。

### 3.3 必须验证的等价性

在相同权重、相同输入序列、关闭随机训练噪声的条件下：

- Full Sequence Forward
- Prefill + Cached Decode
- Token-by-Token Decode

应在声明的数值容差范围内产生一致 Logits。

同时必须测试：

- 修改未来 Token 不会改变之前位置的输出。
- 修改后一个 Token 不会改变前一个 Token 的 KV。
- Step 1 和 Step 2 读取同一份历史 KV。
- Workspace 缓存不包含 Cell ID 或 Step ID。
- Encoder 与 Workspace 的 Cache 在推理时不会发生覆盖或混用。

如果这些测试失败，本次实验不得进入正式训练。

---

## 4. 实验模型矩阵

主实验设置五组模型。

| 模型 | Workspace | 计算路径 | Controller State | Workspace Read |
|---|---|---|---|---|
| T0 | 每层独立 KV | 固定 Transformer Layers | 无 | 每层 Attention |
| C0 | 统一共享 KV | 固定 FFN 路径 | 无 | 每步读取 |
| C1 | 统一共享 KV | 动态 Top-2 Cells | 无 | 每步读取 |
| C2 | 统一共享 KV | 动态 Top-2 Cells | 有 | 每步读取 |
| C3 | 统一共享 KV | 动态 Top-2 Cells | 有 | 仅首次读取 |

### 4.1 T0 — Transformer Baseline

标准 Decoder-only Transformer。

初始采用三个 Block，以便与 CLM 的一个 Workspace Encoder Block 加两次计算步骤形成大致对应的 Attention 计算深度。

T0 每层拥有独立的 Attention 与 KV Cache。

需要建立：

- T0-P：总参数量接近 C2。
- T0-F：每 Token 激活 FLOPs 接近 C2。

如果通过同一配置能够同时满足两种要求，则可以合并。

不要求两个基线具有相同宽度或 FFN Expansion Ratio，但必须如实报告结构差异。

### 4.2 C0 — Fixed Computation

使用相同的 Workspace Encoder 和两次 Attention Read。

每步采用固定 FFN，不使用动态 Router。

需要使每步 FFN 的激活 FLOPs 尽量接近 Top-2 Cell 组合的激活成本。

C0 用于回答：

**在共享历史信息机制相同的情况下，动态选择是否有必要？**

由于 C0 与 C1 的总参数规模可能不同，需要额外报告这种差异，并在关键结果上使用参数匹配的固定路径变体进行复核。

### 4.3 C1 — Stateless Dynamic Cellular Computation

使用 8 个独立 Cell，每步 Top-2。

Router 只访问：

\[
(v_{t,k})
\]

不使用 Controller State。

为了区分递归状态收益与额外参数收益，应增加一个参数或计算量接近 C2 Controller 的 Stateless Router 变体，作为必要时的复核对照。

### 4.4 C2 — Stateful Controller

使用与 C1 相同的 Workspace、Cell 数量、FFN 结构及两步计算。

唯一主要区别是增加独立的 Controller State：

\[
z_{t,k+1}=U_\psi(\cdots)
\]

Router 使用：

\[
R_\phi(v_{t,k},z_{t,k})
\]

其中 \(z_{t,0}=0\)。

C2 是本次提出的完整候选架构。

### 4.5 C3 — Single Workspace Read

保持 C2 的 Controller、Cell、Router 和计算步数不变。

仅在第 1 步执行 Workspace Read：

\[
r_{t,0}=\operatorname{Attention}(q_{t,0},M)
\]

第 2 步复用相同结果：

\[
r_{t,1}=r_{t,0}
\]

其他计算语义不变。

C3 用于判断重复生成 Query、重新访问共享历史是否具有价值。

需要明确：C2 与 C3 的 Attention FLOPs 不同，因此必须同时比较质量和计算成本，不能只比较 PPL。

---

## 5. 初始模型配置

建议的第一版配置：

| 参数 | 建议值 |
|---|---|
| Hidden Dimension | 256 |
| Attention Heads | 4 |
| Workspace Encoder Layers | 1 |
| Workspace KV Heads | 4 |
| Cell Count | 8 |
| Active Cells per Step | 2 |
| Cell FFN Intermediate Dimension | 512 |
| Cellular Steps | 2 |
| Controller State Dimension | 256 |
| Controller Update | GRU-style |
| Position Encoding | RoPE |
| Context Length | 256 |
| Cell-local State | Disabled |
| Dynamic STOP | Disabled |
| Dropout | 初期设为 0，后续按统一协议调整 |

这些值是起始规格，不是硬性理论要求。

特别注意：

- 必须计算实际模型参数量。
- 必须测量激活 FLOPs。
- 不得简单假设相同 Hidden Dimension 就意味着公平。
- 不得为了让 C2 获胜而单独提高训练 Token 数或模型规模。
- 所有主要架构变量在正式训练前锁定。

---

## 6. 训练方案

### 6.1 复用既有实验基础

优先复用 v0.3 的：

- Tokenizer。
- 数据集及预处理。
- Train / Validation Split。
- 随机种子管理。
- Checkpoint 格式。
- 训练指标记录。
- PPL 与 NLL 计算方法。
- 训练吞吐测量。
- 实验 Manifest、Protocol 和 Audit 机制。

必须保留 v0.3 原始结果。

不要覆盖旧实验。

本次的实验编号、配置文件、Checkpoint 和报告必须独立保存。

### 6.2 训练优化器

建议默认：

- AdamW。
- 初始学习率 `3e-4`。
- Weight Decay `0.1`。
- Warmup 占总更新步数的约 3%。
- Cosine Learning Rate Decay。
- Gradient Clipping `1.0`。
- 优先 BF16；硬件不支持时使用 FP16 和适当 Loss Scaling。
- Flash / Scaled Dot Product Attention（满足正确性要求时）。

上述值是默认候选。若 v0.3 已有经过验证的优化器配置，应优先以它为共同基线，再对所有模型采用相同的超参数选择规则。

### 6.3 Router 训练

所有动态 Cell 模型：

- 从开始就使用 Top-2。
- 仅激活选中的 Cell。
- 选中门控保持可微分。
- 不使用训练时全量 Soft Mixing、推理时突然改成 Top-2 的主实验方案。
- 训练期间允许统一的有限 Logit Noise，以增加探索。
- 推理和正式评估关闭随机路由噪声。
- 不采用 REINFORCE 作为主训练算法。
- 不使用额外的 Oracle Route 标签训练主模型。

初始辅助负载均衡可以参考 Switch-style Loss，但具体系数必须预先记录，并进行相同规则的选择。

需要监控负载均衡是否导致 Router 概率长期近似完全均匀。

不得将 Router Entropy 降低本身作为训练目标或成功标准。

### 6.4 正式目标函数

主训练目标：

\[
\mathcal L=
\mathcal L_{\mathrm{LM}}
+
\alpha\mathcal L_{\mathrm{balance}}
\]

其中：

\[
\mathcal L_{\mathrm{LM}}
=
-\frac1T\sum_t
\log P(x_{t+1}\mid x_{\leq t})
\]

负载均衡损失仅适用于使用动态专家路由的模型。

需要记录：

- 主语言损失。
- 负载均衡损失。
- 总训练损失。
- Router Grad Norm。
- Cell Grad Norm。
- Controller Grad Norm。
- Workspace Grad Norm。

特别检查 C2 的 Controller State Update 是否能从最终语言损失获得非零的有效梯度。

### 6.5 训练预算

采用三阶段执行。

**阶段 A：Correctness**

少量随机输入和固定小批量。

目标是通过所有前向、反向、因果、缓存、路由正确性检查。

**阶段 B：Smoke Training**

每组执行相同 Token 预算的短训练。

检查：

- Loss 能否下降。
- 有无 NaN / Inf。
- Router 是否能够获得梯度。
- 是否存在单个 Cell 长期完全不被访问。
- Controller State 是否数值稳定。
- GPU 训练吞吐是否满足继续实验的基本条件。

**阶段 C：Main Training**

建议先以每模型每种子约 32M 训练 Token 作为规划预算，并根据既有 GPU 资源在正式执行前统一锁定。

正式实验至少使用三个种子。

推荐初始种子：

`17 / 29 / 43`

若资源不足，应统一缩减所有模型的主实验预算，不能只缩减表现不佳的模型。

所有模型使用相同训练数据顺序规则、有效 Token Batch Size 和评估间隔。

还应报告按固定训练 FLOPs 比较的结果，避免将相同 Token 数误认为相同训练成本。

---

## 7. 训练正确性测试

正式训练前必须通过以下测试。

### 7.1 Workspace 测试

- 因果 Mask 正确。
- RoPE 位置一致。
- Encoder KV 与 Shared Workspace KV 分离。
- Full Sequence 与 Cached Decode Logits 一致。
- 所有 Cell Step 读取同一份 K/V。
- Step 变化只改变 Query，不改变历史 K/V。
- Workspace KV 不携带 Cell/Step 维度。

### 7.2 Cell 测试

- 8 个 Cell 参数相互独立。
- 所有 Cell 输入输出维度一致。
- 被选中的 Cell 执行。
- 未选中的 Cell 不执行 FFN。
- Top-2 权重和为 1。
- 允许重复选择同一个 Cell。
- Sparse Dispatch 的结果与参考 Dense Gather 实现一致。
- 不同 Token 的输出不会因 Dispatch、Gather 或 Scatter 错误而混淆。

### 7.3 Router 梯度测试

使用两个输出不同的固定 Cell。

验证：

- 两者门控可得到有效梯度。
- Router Logit 改变能改变混合输出。
- 未选中的 Cell 不会错误地获得该 Token 的 Expert 计算梯度。
- Controller State Update 能通过第二步的门控和最终损失获得梯度。
- 禁用 State 后，C2 的路由行为可以按照定义退化为无状态条件。
- 不存在意外的 `.detach()` 或非预期的离散化切断梯度。

### 7.4 State 测试

- 新 Token 开始时 State 重置。
- 同一 Token 第一次计算后 State 正确更新。
- 第二次计算能够读取新的 State。
- State 不直接进入最终 LM Head。
- State 不泄漏其他 Token 的信息。
- 关闭 State 后仍可完成正确推理。
- 不使用目标标签更新推理时的 State。

---

## 8. GPU 实现要求

### 8.1 并行维度

训练时：

- Workspace Encoder 在 Token 维度并行。
- 同一个 Cell Step 中，不同 Token 的 Cell 路由并行。
- 对每个 Cell 聚集被选中的 Token，进行批量 FFN 计算。
- 完成当前 Step 后，统一进入下一计算步骤。

Cell Step 之间保持顺序依赖。

不得引入跨 Token 的 Controller State 递归依赖。

### 8.2 Sparse Dispatch

建议实现：

1. 展平 Batch × Sequence。
2. 计算 Top-2 Index。
3. 按 Expert ID 对 Token 分组。
4. 批量执行对应 Cell FFN。
5. 根据门控权重进行加权。
6. Scatter 回原 Token 顺序。

初期可以采用易于审计的 PyTorch 实现，之后根据 Profiling 决定是否优化。

不应在第一轮就引入复杂自定义 CUDA Kernel，以免将训练错误与内核问题混在一起。

### 8.3 Capacity Handling

小规模研究实验中，优先采用不丢弃 Token 的 Dispatch。

如果由于硬件限制必须使用 Expert Capacity，应明确记录：

- Capacity Factor。
- Overflow 数量。
- Dropped Token 比例。
- Overflow 处理策略。

不同模型之间不能采用不同的丢弃策略。

### 8.4 性能指标

必须分别记录：

- Workspace Encoder Time。
- Attention Read Time。
- Router Time。
- Sparse Dispatch Time。
- Cell FFN Time。
- Controller Update Time。
- Backward Time。
- Total Step Time。
- Peak Allocated GPU Memory。
- Tokens/sec。
- Active FLOPs/Token。

注意：Top-2 稀疏计算在小模型上可能受到 Dispatch Overhead 影响。

不能只根据理论 FLOPs 判断真实效率。

---

## 9. 路由与状态诊断

本次要求保留完整的机制诊断。

### 9.1 Router Diagnostics

分别记录第 1 步和第 2 步：

- Router Logit Mean。
- Router Logit Std。
- Router Probability Entropy。
- Top-1 Probability。
- Top-1 / Top-2 Gap。
- 每个 Cell 的选择次数。
- 每个 Cell 的实际激活 Token 数。
- 不同 Step 的 Cell 使用分布。
- 同一个 Token 两步选择相同 Cell 的比例。

重点比较路由行为是否与预测质量相关。

Router 分布不均匀不等于学会有意义的选择。

### 9.2 Controller State Diagnostics

针对 C2：

- State Norm。
- State Update Norm。
- Step 0 与 Step 1 的 State 差异。
- Controller State Update 的梯度。
- Controller State 对 Router Logits 的贡献。
- State Reset 消融的 NLL 变化。
- State 打乱或置零后的路由变化。

如果 State 对 Logit 有明显影响，但对最终预测质量没有改善，不能视为成功。

### 9.3 Cell Diagnostics

针对 C1/C2：

- 不同 Cell 的参数更新幅度。
- 不同 Cell 的激活次数。
- 每个 Cell 的输出 Norm。
- 相同输入下不同 Cell 的输出差异。
- 不同 Cell 的输出 Cosine Similarity。
- Cell 替换后的任务损失变化。

Cell 输出不同只是必要的诊断信息，不代表其差异有用。

---

## 10. Counterfactual Routing

这是本次机制验证的重要组成部分。

使用同一已训练 Checkpoint，不重新训练，比较不同执行策略。

至少包含：

**Normal Routing**

使用模型自身 Router。

**Random Routing**

从候选 Cell 中按预先定义的随机策略选择，同时保持相同 Top-2 数量。

**Fixed Routing**

所有 Token 使用固定的 Cell 组合。

建议至少测试数种固定组合，避免只选择一个特别差的组合。

**Shuffled Routing**

保留总体 Cell 使用频率，但打乱 Token 与 Cell 的对应关系。

这有助于区分模型收益究竟来自某些 Cell 被频繁使用，还是来自输入与 Cell 之间的正确匹配。

**Controller State Off**

针对 C2，在评估时关闭 State。

注意：对于 C2，强制改动某一步的路由后，后续 Controller State 必须使用实际强制执行的动作和计算结果更新，不能继续使用原始路径的 State。

否则反事实路径不是真实可执行的计算过程。

### 10.1 主要比较

记录每个策略的：

- NLL。
- PPL。
- Active FLOPs。
- 推理延迟。
- Cell 使用分布。

关键指标：

\[
\Delta L_{\mathrm{route}}
=
L_{\mathrm{random}}-L_{\mathrm{normal}}
\]

以及：

\[
\Delta L_{\mathrm{fixed}}
=
L_{\mathrm{fixed}}-L_{\mathrm{normal}}
\]

如果两者稳定为正，并超过实验随机波动，才说明正常路由可能具有有用的选择能力。

### 10.2 小规模 Oracle 分析

在少量固定验证样本上，允许枚举或采样候选两步 Cell 路径，测量它们的真实最终 NLL。

目的：

- 判断不同路径是否存在任务效用差异。
- 估计正常 Router 与离线最优路径的差距。
- 检查路由错误是否具有可学习的规律。

Oracle 可以利用真实预测标签评估候选路径，因此只能作为诊断参考上界。

不得将 Oracle NLL 直接当成模型可实现的推理性能。

---

## 11. Transformer 公平比较

### 11.1 必须报告的三种预算

**Parameter Budget**

总参数量，包括：

- Workspace Encoder。
- 所有 Cell 参数。
- Router。
- Controller。
- LM Head。

**Active Computation Budget**

每 Token 实际执行的主要浮点运算。

未激活的 Cell 参数不应计入激活 FLOPs，但仍计入总参数量。

**Training Budget**

训练 Token 数、更新步数、累计训练 FLOPs 与总 GPU 时间。

三者必须独立报告。

### 11.2 不允许的比较

- 用 C2 的 8 个 Cell 总参数与只有两个 FFN 的 C0 比较，却声称已完成等参数对照。
- 只匹配参数量，不报告激活 FLOPs。
- 只报告理论 FLOPs，不测量实际吞吐。
- 为某一个模型使用额外训练数据。
- 在查看正式验证集结果后，只对表现最好的模型追加训练。
- 只展示最好一个 Seed。
- 将模型质量差异直接归因于路由，而没有内部消融。

### 11.3 最终结果

至少提供：

| 模型 | Total Params | Active FLOPs/Token | Train Tokens | NLL | PPL | Tokens/sec | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0-P | | | | | | | |
| T0-F | | | | | | | |
| C0 | | | | | | | |
| C1 | | | | | | | |
| C2 | | | | | | | |
| C3 | | | | | | | |

数值必须来自实际运行，不允许占位值出现在最终正式报告中。

建议另绘制 NLL 对 Active FLOPs、NLL 对训练 FLOPs、NLL 对推理延迟的比较曲线。

---

## 12. 实验执行顺序

严格按以下顺序进行。

### Phase 0 — Repository Audit

先检查当前 MiniCells 仓库中已有的：

- EPCLM v0.3 实现。
- 实验配置。
- Checkpoint。
- Dataset / Tokenizer。
- GPU 训练脚本。
- 诊断脚本。
- CI / Test。
- 实验报告生成程序。

输出一份简要 Audit，说明哪些模块可以复用，哪些模块需要新增。

不得在未检查现有实现的情况下重新创建一套平行训练框架。

### Phase 1 — Shared Workspace

首先实现 W0 Workspace Encoder 和重复 Attention Read。

用固定 FFN 路径执行 C0。

验证因果、KV Cache、训练并行、完整序列与增量推理一致性。

**Gate：** C0 所有正确性测试通过且能够正常训练。

### Phase 2 — Dynamic Cellular Computation

实现独立 Cell、Top-2 Router、Sparse Dispatch。

完成 C1。

先不实现 Controller State。

**Gate：** Router 具有有效梯度，Cell Dispatch 正确，训练损失正常下降。

### Phase 3 — Stateful Controller

在 C1 上增加独立 Controller State。

完成 C2。

确保 State 只影响 Cell 选择，不直接影响主表示或 LM Head。

**Gate：** State Update 的梯度、状态隔离和消融机制全部通过。

### Phase 4 — Workspace Read Ablation

完成 C3。

确保除了第二次 Workspace Read 以外，C2 和 C3 的模型机制一致。

**Gate：** C3 可正常训练，性能与 FLOPs 统计准确。

### Phase 5 — Transformer Baselines

完成 T0-P 和 T0-F。

在所有模型主实验训练前，锁定基线配置与比较预算。

### Phase 6 — Main Training

执行统一正式训练协议。

所有模型使用相同数据版本与评价规则。

至少三个随机种子。

不在正式执行期间随意修改架构。

### Phase 7 — Mechanism Evaluation

执行：

- Routing Counterfactual。
- State Ablation。
- Workspace Read Ablation。
- Cell Output Diagnostics。
- FLOPs / Latency / KV Memory Comparison。

### Phase 8 — Research Decision

生成完整报告，决定是否继续研究：

- Dynamic STOP。
- State-aware Workspace Query。
- Cell 能力表示。
- 更大规模 CLM。
- 更复杂的 Workspace 更新机制。

这些决定必须以实验结果为依据。

---

## 13. 研究决策门槛

为避免再次出现“训练可以正常运行，但无法判断研究结论”的情况，本次明确设置以下判断条件。

### H1：Shared Workspace

如果 C0 相对 T0 的质量差距过大，应先调查共享历史表示的瓶颈，而不是立即将问题归因于 Cell 路由。

共享 Workspace 的收益需结合 KV 内存、计算量与语言质量评估。

### H2：Dynamic Routing

如果 C1 的正常路由不能稳定优于合理的固定与随机路径，不能宣称已验证动态 Cell 计算。

但由于 C0 与 C1 存在参数和结构差异，也必须结合匹配预算对照判断。

### H3：Controller State

如果 C2 相比 C1 没有稳定改善，并且 State 消融不产生明显影响，应暂停复杂 Controller 的扩展。

不能仅通过增加 State Dimension 来尝试挽救结果。

### H4：Repeated Attention

如果 C2 不优于 C3，或者质量收益不足以抵消额外成本，应优先检查第二次 Attention 的信息获取行为。

不应直接添加第三次或更多 Attention Read。

### 统一判断原则

优先以验证 NLL、实际计算成本和多随机种子稳定性作出判断。

在主实验前记录最小有意义改善幅度，避免根据结果事后设定成功门槛。

若差异小于实验随机波动，应报告为证据不足，而非 PASS。

---

## 14. 实验交付物

建议沿用现有实验目录规范，将本次结果放入独立版本目录。

建议结构：

`artifacts/experiments/clm-workspace-controller-001/v0.4/`

至少包含：

| 文件 | 内容 |
|---|---|
| `protocol.json` | 实验条件、模型矩阵、预算和随机种子 |
| `manifest.json` | Git SHA、配置、依赖、硬件及数据版本 |
| `model-configs/` | 每个模型的最终配置 |
| `training-metrics/` | 训练与验证曲线 |
| `routing-diagnostics.json` | 路由统计 |
| `state-diagnostics.json` | Controller State 统计 |
| `counterfactual-routing.json` | 路由反事实结果 |
| `compute-profile.json` | FLOPs、延迟、显存和吞吐 |
| `correctness-report.json` | 因果、缓存、梯度和 Dispatch 验证 |
| `decision.json` | 各研究假设的结果与决策 |
| `audit.json` | 实验复现与完整性检查 |
| `RESEARCH-REPORT.zh-CN.md` | 中文完整研究报告 |

Checkpoint 及大型 Tensor 日志可以单独存储，但必须在 Manifest 中记录可定位路径和校验信息。

### 14.1 中文研究报告内容

报告必须包含：

1. 研究背景与实验假设。
2. 与 v0.3 的架构差异。
3. 每个模型的确切计算结构。
4. 数据、训练预算和硬件。
5. 正确性测试结果。
6. 全部模型的训练与验证结果。
7. 参数、FLOPs、吞吐与显存比较。
8. Router 与 Cell 诊断。
9. Controller State 消融。
10. Workspace Read 消融。
11. Counterfactual Routing。
12. 各假设的证据等级。
13. 失败机制与局限。
14. 下一阶段建议。

不得把推断写成已经验证的事实。

---

## 15. 代码质量与仓库要求

- 复用既有 MiniCells / EPCLM 训练基础设施。
- 不覆盖 v0.3 Checkpoint 和实验结果。
- 新增模型使用可配置的统一接口。
- 不将实验参数硬编码到模型实现。
- 所有实验配置可复现。
- Checkpoint 保存 Optimizer、Scheduler、RNG State 及完整训练进度。
- 支持从中断处恢复。
- 支持独立 Evaluation，不需要重新训练。
- 支持 CPU 小规模正确性测试。
- 支持 GPU 正式实验。
- 新增必要的单元测试与回归测试。
- 在创建正式报告前执行 Diff Check、Test 和环境 Audit。
- 不将生成数据、大型 Checkpoint 或私密运行环境配置直接提交到 Git。

允许根据现有项目规范调整具体目录结构，但不能省略可复现性和实验追溯要求。

---

## 16. 对 Codex 的最终执行要求

本次任务不是单纯开发一个新的 CLM 模型，而是建立一套可以对研究假设作出明确判断的实验。

执行时遵循以下原则：

**第一，先正确，再优化。**

Workspace 因果语义、KV Cache 等价性、Cell Dispatch 和 Router 梯度必须先通过验证。

**第二，先分离变量，再比较架构。**

不能同时改变 Cell 参数结构、Attention 机制和 Router 后，将全部性能差异归因于某个机制。

**第三，不预设 CLM 一定优于 Transformer。**

本实验应允许清晰地得出否定结果，并解释原因。

**第四，严格区分可训练性和架构有效性。**

训练 Loss 下降、Router 出现偏好、Cell 输出产生差异，均不等于动态计算路径已经获得实际收益。

**第五，优先保留最简单的有效架构。**

如果 Controller State 没有独立收益，应当优先采用 Stateless Router。

如果重复 Attention 没有收益，应当优先采用更简单的 Workspace Read。

如果动态路由不能优于固定路径，应重新检查其必要性，而不是持续增加模型复杂度。

### 本次最终研究问题

**在因果共享的信息空间上，一个由独立参数 Cell 构成、采用稀疏可微路由的递归计算系统，能否学习有用的动态计算路径？独立高维 Controller State 和重复 Workspace Read 是否能够分别提供额外收益？**

本次实验的成功标准不是证明全部假设正确，而是以可复现、可解释、预算可比的实验，明确判断每个假设是否获得支持。

---

**实施优先级：**

`Repository Audit → Workspace Correctness → C0 → C1 → C2 → C3 → Transformer Baselines → Main Training → Counterfactual Diagnostics → Research Report`

只有上一阶段的必要 Gate 通过，才进入下一阶段。最终以完整报告、原始指标及可复现配置作为交付依据。

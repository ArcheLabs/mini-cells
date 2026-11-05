# Mamba-3 Cellular（M3C）系列实验设计

**版本：** 0.1-draft  
**日期：** 2026-11-05  
**状态：** PROPOSED_NOT_REGISTERED  
**正式训练授权：** NONE  
**所属项目：** MiniCells / Cellular Language Models  
**实验系列目录：** research/projects/cellular-language-models/experiments/mamba3-cellular-series/

---

## 1. 研究目标

M3C 研究一种混合计算架构：

- 主干严格采用 Mamba-3 的层级堆叠与原生递归状态机制。
- 任意 Mamba-3 Layer 在处理当前 Token 后，可以选择跳过 Cell，或调用一个无持久状态的 Cell。
- Cell 调用包含一次 Workspace Attention 读取和一次 Cell 计算。
- Cell 输出返回 Mamba 主干的 Residual Stream，后续 Mamba-3 Layer 继续处理。
- Cell Pool 是平面集合，不规定 Cell 之间的固定顺序、父子关系或连接图。
- 不把 Cell-local State 作为默认记忆机制；跨 Token 历史首先由 Mamba-3 原生 Layer State 表示。

本系列不预设 M3C 一定优于 Mamba-3。研究目标是分别验证 Workspace、Cell 计算、动态路由和状态记忆检索的独立价值，并在相同质量或相同资源预算下判断是否存在结构性优势。

### 1.1 核心研究问题

**RQ1 — Mamba-3 主干：** 能否在 MiniCells 现有训练与评估环境中，正确复现官方 Mamba-3 SISO 路径，并验证官方 MIMO 路径的可用性？

**RQ2 — Workspace：** 有界 Attention Workspace 是否能补充 Mamba-3 的递归状态，改善局部精确检索或关联回忆？其缓存与计算成本是多少？

**RQ3 — Stateless Cell：** 在 Mamba-3 Layer 之间加入无持久状态的 Cell，是否能在参数量、训练 Token 和计算预算受控的条件下改善任务能力？

**RQ4 — 动态调用：** 不同 Token、不同 Mamba 层是否确实需要不同的 Cell 计算？模型能否学会跳过无必要的 Cell 调用？

**RQ5 — 动态深度：** 同一 Mamba Layer 是否可以按需调用多个 Cell，并在达到充分计算后停止，而不依赖固定 Cell 顺序？

**RQ6 — 统一状态读取：** 能否从 Mamba-3 的 Chunk State 构造可寻址的压缩记忆，并让 Workspace 对其执行 Attention 式检索？这必须从 Mamba-3 的实际递归公式重新推导，不能假定 Mamba-2 DART 公式直接适用。

### 1.2 核心假设

| 假设 | 可证伪主张 | 失败时的决策 |
|---|---|---|
| H1 | 官方 Mamba-3 SISO 能在锁定环境中正确训练、Prefill 和 Cached Decode | 停止后续架构实验，先解决基线 |
| H2 | 有界 Workspace 在检索/状态任务上提供可重复收益，且成本可接受 | 若无收益，不保留 Workspace Attention |
| H3 | 固定调用的 Stateless Cell 比参数/计算匹配的普通 Adapter 更有用 | 若不成立，不进入动态 Cell 路由 |
| H4 | 动态 Skip/Cell 路由形成可解释、可复现的功能分工 | 若路由退化，不宣称动态计算成立 |
| H5 | 动态路由的逻辑稀疏能转化为真实 Decode 延迟、能耗或内存收益 | 若无实际收益，不宣称效率优势 |
| H6 | Mamba-3 Chunk State 可支持比单一递归读出更灵活的 State-Memory Retrieval | 若推导或实验失败，保留局部 Workspace，不强行移植 DART |

---

## 2. 架构原则

### 2.1 Mamba-3 是主干，不重新发明 SSM

M3C 使用官方 Mamba-3 模块作为每层的核心序列混合器。Mamba-3 论文的主要改进包括：

1. 更有表达力的 Exponential-Trapezoidal Discretization；
2. 复数状态更新，用于增强状态跟踪；
3. MIMO（Multi-Input, Multi-Output）形式，在保持状态规模的同时提高 Decode 阶段的计算利用率；
4. 配套的归一化、投影和训练/推理实现改进。

M3C 不修改 Mamba-3 的状态转移、离散化、复数旋转、MIMO 规则或官方 Kernel。所有 Cell 扩展都放在 Mamba-3 Block 的外部层间边界。

### 2.2 Mamba Layer 的状态归属

每个 Mamba-3 Layer 继续拥有自己的官方推理状态。不要把它简化成一个自行定义的普通向量 State。官方实现的推理路径涉及 SSM State，以及角度/旋转和相关输入缓存状态；实际结构以锁定版本的官方实现为准。

记第 \(\ell\) 层在处理 Token \(t\) 前后的官方状态包为：

\[
\Sigma_{\ell,t-1}^{M3}
\quad\longrightarrow\quad
\Sigma_{\ell,t}^{M3}
\]

Cell 不直接读取、覆盖或更新这个状态包。Cell 的输出只改变 Residual Stream，并作为后续 Mamba 层的输入。

因此，某层的 Cell 输出不会回头修改该层刚刚完成的状态更新；它会影响后续层在当前 Token 上的输入，以及后续 Token 在后续层中的状态演化。这是本系列必须保持清晰的因果语义。

### 2.3 Cell 是无持久状态的计算函数

Cell 不保存跨 Token 的 Cell-local State：

\[
c_i = F_i(u,r;\theta_i)
\]

其中：

- \(u\)：当前 Mamba 层的隐藏表示；
- \(r\)：Workspace Attention 的读取结果；
- \(\theta_i\)：Cell 自身的参数。

同一个 Cell 在不同时间被调用时，函数参数不因历史调用而发生持久变化。历史影响通过 Mamba-3 主干的输入表示传递，而不是通过 Cell 自己的长期记忆。

第一版不实现 Cell-to-Cell 固定连接、Cell 生长、Cell-local Plasticity 或跨 Token Controller State。

### 2.4 Cell Pool 是平面的，并可被任意 Mamba 层调用

所有 Mamba 层共享一个全局平面 Cell Pool：

\[
\mathcal C=\{C_1,\ldots,C_N\}
\]

任何层都可以请求调用其中任意 Cell。Cell 之间没有固定的上下游层级。

Router 必须接收 Layer Identity 或 Layer Embedding，使同一个共享 Cell Pool 可以区分不同层的调用环境。是否需要 Layer-specific Input/Output Adapter，作为单独消融，不在首个实现中同时引入。

如果共享 Cell Pool 因层间表示分布差异而失败，才比较每层独立 Cell Pool；不得一开始就把“全局共享 Cell”与“每层独立 Cell”混为同一个实验。

---

## 3. M3C 的计算结构

### 3.1 Mamba-3 主干

令 \(x_{\ell-1,t}\) 为第 \(\ell\) 层接收的 Residual Stream。Mamba-3 Block 的标准外部残差包装写为：

\[
u_{\ell,t}
=
x_{\ell-1,t}
+
\operatorname{M3}_{\ell}
\left(
\operatorname{Norm}_{\ell}(x_{\ell-1,t});
\Sigma_{\ell,t-1}^{M3}
\right)
\]

具体 Pre-Norm、Residual、Fused Norm 和参数共享方式必须与所选官方模型包装保持一致，并在 M3C-001 锁定。不得为了方便 Cell 插入而改变 Mamba-3 Block 内部语义。

### 3.2 Layer-local Shared Workspace

首个版本不增加一个完整 Transformer Encoder 作为第二条主干，也不保存无界的全历史 KV。

在每个 Mamba 层的输出表示 \(u_{\ell,t}\) 上构建有界 Workspace：

\[
k_{\ell,t}=W^K_\ell\operatorname{Norm}(u_{\ell,t})
\]

\[
v_{\ell,t}=W^V_\ell\operatorname{Norm}(u_{\ell,t})
\]

第 \(\ell\) 层在 Token \(t\) 可访问：

\[
\mathcal W_{\ell,t}
=
\{(k_{\ell,j},v_{\ell,j}) :
\max(1,t-W+1)\le j\le t\}
\]

其中 \(W\) 是固定 Workspace Window。

所有在第 \(\ell\) 层被调用的 Cell 共用同一份 \(\mathcal W_{\ell,t}\)。Workspace 不包含 Cell ID 或 Cell Step ID，也不为每个 Cell 复制 KV。

不同 Mamba 层默认使用各自的 Layer-local Workspace，因为各层隐藏表示分布可能不同。所谓 Shared，首先指同一层内所有 Cell 和调用步骤共享，而不是未经验证地把所有层的隐藏表示混成一份全局 KV。

跨层 Global Workspace 是后续消融，不是默认主实现。

### 3.3 Workspace Read

只有当调用门控决定执行 Cell 时，才执行 Workspace Attention：

\[
q_{\ell,t}
=
W^Q_\ell\operatorname{Norm}(u_{\ell,t})
\]

\[
r_{\ell,t}
=
\operatorname{Attention}
\left(
q_{\ell,t},
K_{\ell,[t-W+1:t]},
V_{\ell,[t-W+1:t]}
\right)
\]

必须使用因果约束，允许读取当前 Token 的 K/V，但禁止读取未来 Token。

K/V 写入与 Attention Read 分离：即使当前 Token 跳过 Cell，仍需按既定策略将其 K/V 写入有限 Workspace，供未来 Token 使用。该写入成本必须统计。后续可以独立测试只记录 Cell 事件的 Event Workspace，但不能混入第一版。

### 3.4 调用门控与 Cell 路由

为了让 Skip 真正节省 Workspace Read 和 Cell 计算，先执行低成本调用门控：

\[
a_{\ell,t}
=
\operatorname{InvokeGate}
(u_{\ell,t},e_\ell)
\]

其中 \(e_\ell\) 是 Layer Embedding，\(a_{\ell,t}\in\{0,1\}\) 表示是否调用 Cell。

若 \(a_{\ell,t}=0\)，直接：

\[
x_{\ell,t}=u_{\ell,t}
\]

若 \(a_{\ell,t}=1\)，才执行 Workspace Read，并选择 Cell：

\[
i_{\ell,t}
=
\operatorname{Router}
(u_{\ell,t},r_{\ell,t},e_\ell)
\]

第一版动态路由以 Top-1 为主，Skip 是独立动作。Top-2、多 Cell 并行或连续调用留到后续实验。

### 3.5 Stateless Cell 与 Residual Return

将当前隐藏表示与 Workspace Read 融合：

\[
z_{\ell,t}
=
\operatorname{Norm}
\left(
W_h u_{\ell,t}+W_r r_{\ell,t}
\right)
\]

调用选中的 Cell：

\[
c_{\ell,t}=F_{i_{\ell,t}}(z_{\ell,t};\theta_{i_{\ell,t}})
\]

将结果投影回主干：

\[
x_{\ell,t}
=
u_{\ell,t}
+
a_{\ell,t}\alpha_\ell W_o c_{\ell,t}
\]

\(x_{\ell,t}\) 进入下一 Mamba-3 Layer。

\(\alpha_\ell\) 使用小但非零的初始残差尺度，避免零门控导致 Cell 内部参数在训练初期完全没有任务梯度。精确初始化值需在 Smoke 阶段比较并冻结。

### 3.6 完整信息流

\[
\boxed{
\text{Mamba-3 Layer}
\rightarrow
\text{Invoke Gate}
\rightarrow
\begin{cases}
\text{Skip}\\
\text{Workspace Read}
\rightarrow
\text{Flat Cell Router}
\rightarrow
\text{Stateless Cell}
\end{cases}
\rightarrow
\text{Residual Return}
\rightarrow
\text{Next Mamba-3 Layer}
}
\]

这保留了 Mamba-3 的层级主干，同时允许 Cell 计算在层间按需发生。Cell 没有自己的递归时间轴，也没有固定的 Cell-to-Cell 层级。

---

## 4. Workspace 的设计边界

### 4.1 首选：有界 Layer-local Token Workspace

首个实现采用有界 Sliding Window Workspace。建议在 Smoke 阶段比较：

- W = 64
- W = 128
- W = 256

正式配置只能在看到正式评估集结果之前冻结。

每个 Mamba 层维护自己的 Workspace KV；同一层的 Cell Pool 共享这份 KV。必须统计所有层 Workspace KV 的总内存，而不是只报告单层缓存。

### 4.2 不采用无界 Token KV

M3C 不应在每层附加完整历史 Transformer KV，否则会重新引入随上下文长度线性增长的 Token Cache，削弱 Mamba-3 的主要效率优势。

Workspace 的职责是短期精确访问，而不是重复保存完整历史。

### 4.3 State-derived Workspace 是后续独立研究

DART 提出从 Mamba-2 的 Chunk State Contributions 构造 State Memory，并通过 State-Memory Attention 进行检索。但 DART 的推导基于 Mamba-2 SSD 结构。

Mamba-3 使用 Exponential-Trapezoidal Recurrence、Complex-valued State 和 MIMO。不能未经推导就把 Mamba-2 的 \(\Delta H_{[c]}\)、Key Decode 或 Value Decode 公式照搬到 Mamba-3。

M3C-006 必须先完成：

1. Mamba-3 SISO Chunk Scan 的状态贡献分解；
2. 复数状态与旋转项的存储语义；
3. MIMO Rank 对状态贡献和检索接口的影响；
4. 因果性与 Chunk Boundary 等价性证明；
5. 真实 Cache、读取成本和检索准确率测量。

若无法建立正确的 State Memory 表示，M3C 保留有界 Token Workspace，不把 DART 视为必需组件。

---

## 5. 实验系列与阶段门禁

### M3C-001 — Official Mamba-3 Baseline

**问题：** 官方 Mamba-3 能否在本项目环境中稳定运行，并成为可信基线？

范围：

- 使用官方 Mamba-3 模块，不自行重写 SSM。
- 主基线先采用 SISO。
- MIMO 作为独立配置，在 TileLang 与硬件兼容门禁通过后运行。
- 建立 Embedding、Mamba-3 Layer Stack、Final Norm 和 LM Head 的最小语言模型包装。
- 记录每层官方推理 State 的实际结构和尺寸。
- 建立 Full Sequence、Prefill、Cached Decode 的一致性测试。

初始候选配置：

| 参数 | 候选值 |
|---|---|
| d_model | 256 |
| layers | 8 |
| d_state | 128（另测 64） |
| expand | 2 |
| headdim | 64 |
| mode | SISO |
| chunk_size | 按官方实现与硬件门禁确定 |
| MIMO | 单独候选，rank 由官方配置与门禁确定 |

这只是小规模机制验证配置，不代表 1.5B 或端侧产品配置。

基线对照：

- Mamba-3 SISO；
- Mamba-3 MIMO（若可用）；
- 参数匹配 Transformer 作为参考；
- 可选 Mamba-2 作为架构演进参考，不取代 Mamba-3 主基线。

Gate 001：

- 官方源码 Commit、依赖版本和 Kernel 路径锁定；
- Forward 与 Decode 正确；
- 状态恢复正确；
- Loss、梯度和参数更新有限；
- 训练吞吐、Decode 延迟、峰值显存可测量；
- SISO/MIMO 分别报告，不把不可用路径标为通过。

### M3C-002 — Bounded Shared Workspace

**问题：** 有界 Attention Read 是否补充 Mamba-3 的递归状态？

加入 Layer-local Shared Workspace，但暂不加入 Cell。

对照：

- M3-Base：纯 Mamba-3；
- M3-WA：Mamba-3 + Workspace Attention Adapter；
- 不同 Workspace Window；
- 必要时比较 Global Workspace 与 Layer-local Workspace，但必须单独记录额外 Encoder/Projection 成本。

Workspace-only Adapter 使用独立残差门控，作为 Cell Attention 分支的控制组。

Gate 002：

- 严格因果；
- Full Sequence 与 Cached Decode Logits 一致；
- Workspace KV 总量有界且统计完整；
- 在 MQAR/检索任务上报告增益；
- 报告 PPL、Prefill、Decode、内存和实际 Kernel 成本。

若 Workspace 只增加成本而没有可重复的任务收益，不进入 Cell 主实验。

### M3C-003 — Fixed Stateless Cell Adapter

**问题：** 在不引入动态路由的条件下，Cell 计算本身是否有价值？

先选择预注册的中间层位置，固定调用一个 Stateless Cell，并使用 Workspace Read。

对照：

- M3-Base；
- M3-WA；
- M3-WA + 固定 Cell；
- 参数量或激活 FLOPs 匹配的普通 FFN/Adapter。

这一阶段不允许动态 Skip，不允许多个 Cell 路由，不允许 Cell-local State。先证明 Cell 函数在 Mamba-3 层间具有独立价值。

Gate 003：

- Cell 输出能产生稳定的任务改善，或在同等质量下减少资源成本；
- 改善不能仅由参数增加解释；
- Cell 梯度和实际参数更新可验证；
- 结果在多个 Seed 上可复现后，才进入动态路由。

### M3C-004 — Layer-agnostic Dynamic Cell Routing

**问题：** 任意 Mamba-3 层是否能根据当前 Token 和 Layer Context，决定跳过或调用平面 Cell Pool 中的任意 Cell？

设计：

- 一个跨层共享的全局平面 Cell Pool；
- 每层有独立调用门控；
- 调用门控以当前隐藏表示和 Layer Embedding 为条件；
- 调用后，Workspace Read 执行；
- Cell Router 选择一个 Cell；
- Cell 输出通过残差返回下一 Mamba 层。

主实验动作空间：

\[
\{\text{SKIP},C_1,C_2,\ldots,C_N\}
\]

Cell Pool 初始候选 N = 4、8、16。正式值在 Smoke 后冻结。

训练实现必须明确处理硬路由的梯度问题。首选先使用可微的调用门控/受控代理梯度，再使用真实稀疏 Dispatch 进行推理；如果使用 Straight-Through 或 Gumbel 方法，必须独立报告其训练—推理差异。不能把全 Cell Dense Soft Mixing 的训练 FLOPs 当成稀疏推理成本。

Gate 004：

- 路由不是随机或均匀坍缩；
- Cell 使用分布和功能差异有可重复证据；
- Skip 比例与任务难度或层位置有可解释关系；
- 真实稀疏 Dispatch 有实际收益，或至少不造成无法接受的延迟退化；
- Router、Workspace Read、Dispatch、Cell Kernel 的成本分别报告。

### M3C-005 — Adaptive Multi-Cell Computation

**问题：** 一个 Mamba 层是否能按需调用多个 Cell，直到计算足够，而不需要固定 Cell 顺序？

在 M3C-004 通过后，允许在同一个 Mamba 层内重复执行：

\[
z_{k+1}=F_{i_k}(z_k,r_{\ell,t})
\]

每次调用后 Router 可以选择另一个 Cell、重复选择同一个 Cell，或 STOP。

约束：

- 最大调用次数 Kmax 预先冻结；
- 每次调用使用当前更新后的表示重新产生 Query；
- Workspace KV 不因 Cell 调用而复制；
- Cell 不保存跨 Token State；
- 必须记录实际调用次数、停止原因和总成本；
- 不允许通过无限循环获得更高质量。

Gate 005：

- 在相同平均 Cell 调用预算下优于固定调用次数，或在相同质量下减少调用次数；
- STOP 具有可复现的因果效用；
- 实际延迟改善大于 Router 和调度开销。

### M3C-006 — Mamba-3 State Memory Workspace

**问题：** 能否将 Mamba-3 的递归状态贡献组织成可寻址的压缩记忆，并由 Attention 读取？

本阶段以 DART 的 Mamba-2 State-Memory Attention 为研究参照，但必须先推导 Mamba-3 对应的状态贡献。此阶段不是 M3C-001 至 005 的前置依赖。

Gate 006：

- State Memory 的数学定义与因果性明确；
- 能够从 Chunk State 贡献中构造 Query-dependent Retrieval；
- 与 Mamba-3 原生读出数值关系经过测试；
- 在 MQAR、NIAH、长程回忆任务上与纯 Mamba-3、Token Workspace 对照；
- 报告每 Chunk Cache、读取复杂度、Kernel 延迟和内存。

---

## 6. 数据与任务

所有数据必须有来源、版本、Tokenizer、Split 和 Hash。优先审计 MiniCells 已有训练数据和 Tokenizer；只有确认其可复用且身份完整，才沿用历史基线。不得因为历史实验存在某个 PPL 数字，就假定新模型使用了相同数据或训练预算。

### 6.1 语言建模

- 使用经过审计的项目公共训练语料；
- 固定 Tokenizer 与 Vocabulary；
- 固定 Train/Validation/Test；
- 记录 NLL 与 PPL；
- 同时报告每 Token 训练 FLOPs、训练时间和实际 Decode 成本。

### 6.2 状态跟踪

使用可控合成任务：

- Parity；
- Counter / Running Sum；
- Finite-State Transition；
- 状态重置与状态延续；
- 多事件更新后的最终状态查询。

这些任务用于测试 Mamba-3 State 的追踪能力，以及 Cell 是否真正改善复杂状态相关计算。

### 6.3 关联回忆与检索

- Multi-Query Associative Recall（MQAR）；
- Copy / Repeat；
- Needle-in-a-Haystack；
- Passkey Retrieval；
- Phonebook-style Retrieval（若数据生成协议可复现）。

报告不同历史长度、关联数量、干扰项数量下的 Exact Match 或 Recall。

### 6.4 动态计算任务

构造所需计算深度不同的任务：

- 单步变换；
- 两步组合；
- 多步组合；
- 需要跨层信息读取后才能完成的任务；
- 无需额外 Cell 的简单输入。

目标是验证 Router 是否根据真实计算需求调用 Cell，而不是只根据 Token 位置或固定层偏好调用。

### 6.5 长上下文

至少覆盖短、中、长三类长度。候选评估长度：

- 256；
- 1K；
- 4K；
- 16K；
- 64K（仅在数据与硬件预算允许时）。

不得把超过训练长度的 Zero-shot 外推与训练长度内的结果混为一谈。

---

## 7. 基线、公平性与消融

### 7.1 必须报告的基线

| 基线 | 用途 |
|---|---|
| Mamba-3 SISO | 主基线 |
| Mamba-3 MIMO | Mamba-3 内部变体 |
| Transformer | 通用质量参考 |
| Mamba-3 + Workspace | Attention 增益控制 |
| Mamba-3 + Workspace + Fixed Cell | Cell 函数增益控制 |
| Mamba-3 + Workspace + Dynamic Cell | M3C 主候选 |

Mamba-2 可作为附加架构参考，但不替代 Mamba-3 主基线。

### 7.2 参数与计算匹配

必须分别建立：

1. 参数量匹配；
2. 训练激活 FLOPs 匹配；
3. Decode 实际延迟匹配或实际延迟报告；
4. 质量匹配下的资源成本比较。

动态 Cell 的总参数量、每 Token 激活参数量和真实执行 FLOPs 是三个不同数字，必须分别报告。

### 7.3 必要消融

- 无 Workspace；
- Workspace Window 64/128/256；
- Global vs Layer-local Workspace；
- Workspace Read 在调用前/调用后的位置；
- 无 Cell；
- 固定 Cell；
- 全局共享 Cell Pool vs 每层独立 Cell Pool；
- 4/8/16 Cells；
- Top-1 vs Top-2；
- 无 Skip vs 有 Skip；
- 无 Controller vs Layer-conditioned Controller；
- 固定调用次数 vs Adaptive STOP；
- SISO vs MIMO；
- Cell Residual Gate 不同初始化；
- Workspace K/V 每 Token 写入 vs 仅 Cell 事件写入（后者属于扩展实验）。

不能在看到测试集结果后挑选最优窗口、最优层位置或最优 Cell 数。超参数选择只允许使用开发集/验证集，并在正式测试前冻结。

---

## 8. 训练协议

### 8.1 阶段划分

**Engineering / Correctness Smoke**

- 随机小输入；
- Forward / Backward；
- 因果性；
- Full Sequence 与 Cached Decode 一致性；
- State Checkpoint / Restore；
- 数值稳定；
- Kernel 可运行；
- 实际内存与吞吐。

**Development Pilot**

- 少量模型配置；
- 单 Seed；
- 每个实验臂相同有效预测 Token；
- 用于检查 Loss 是否下降、Cell 是否得到梯度、路由是否坍缩；
- 不作正式科学结论。

**Main Study**

- 至少 3 个预注册 Seed；
- 每个实验臂相同训练 Token；
- 同一数据顺序规则、Tokenizer、优化器选择流程和评估集；
- 预算根据 Pilot 实测吞吐在训练前冻结；
- 不得因为某个实验臂结果不好而单独追加训练。

### 8.2 暂定训练预算

本提案不冻结正式预算。建议先以每臂 32K–100K 有效预测目标完成工程 Smoke，再由 GPU 实测确定 Pilot。Pilot 可从每臂 1M–2M Targets 评估；Main 的每臂 Token 预算和 Seed 数必须在正式运行前单独冻结。

这不是运行授权。当前授权 Token 数为 0。

### 8.3 优化器与精度

优先复用仓库已验证的训练框架和数据管线。AdamW、学习率、Weight Decay、Warmup、Scheduler、Gradient Clipping 和精度策略必须在所有模型臂之间统一。

Mamba-3 SISO/MIMO 可能使用不同 Kernel 路径；必须记录实际执行的 Kernel 和精度，不允许用“请求 BF16”代替“实际 BF16”。

若 T4 或当前 CUDA/Triton 环境无法运行官方 Mamba-3 路径，先标记 ENGINEERING_BLOCKED；不得静默替换为未经验证的 PyTorch 近似实现并称为官方 Mamba-3。

### 8.4 MIMO 特别门禁

官方 Mamba-3 代码中的 MIMO Kernel 依赖 TileLang 路径；SISO 与 MIMO 的依赖、Kernel 和 Chunk 配置不同。

因此：

- SISO 是第一条可复现基线；
- MIMO 必须单独通过 TileLang、GPU 架构、精度和 Forward/Decode 正确性门禁；
- MIMO 不可用不阻塞 SISO 的机制研究；
- 不可将 MIMO 理论 Decode 延迟优势直接当作当前设备上的实测结果。

---

## 9. 正确性与因果门禁

在任何训练前必须通过：

1. Mamba-3 原生 Forward / Backward；
2. Mamba-3 Prefill 与 Token-by-Token Decode 等价；
3. 每层 Recurrent State 的隔离、恢复与版本检查；
4. Workspace 严格因果 Mask；
5. 当前 Token 只能访问历史和自身；
6. 修改未来 Token 不得改变之前位置的 Logits；
7. Workspace KV 不含 Cell ID / Step ID；
8. 同层所有 Cell 共享同一 Workspace KV；
9. Skip 时不执行 Workspace Read 和 Cell Kernel；
10. Cell 输出只写回 Residual Stream，不直接篡改 Mamba-3 内部 State；
11. Full Sequence、Prefill + Cached Decode、Token-by-Token Decode 的 Logits 在规定容差内一致；
12. 训练、恢复、Checkpoint 和 RNG 行为可复现。

如果 Skip 仅仅屏蔽 Cell 输出，但 Workspace Attention 仍然执行，不能报告 Workspace 计算被跳过。

---

## 10. 评价指标

### 10.1 模型质量

- Validation/Test NLL；
- PPL；
- MQAR Exact Match / Recall；
- NIAH / Passkey Retrieval Accuracy；
- State Tracking Accuracy；
- 动态计算任务成功率。

### 10.2 计算效率

- Prefill Tokens/sec；
- Decode Latency p50 / p95；
- Batch=1 与适当批量的 Decode 吞吐；
- 训练 Tokens/sec；
- 每生成 Token 的实际 Kernel 时间；
- Router 与 Dispatch 时间；
- Workspace Read 时间；
- Cell 计算时间；
- 实际与理论 FLOPs。

### 10.3 内存

- Mamba-3 每层 Recurrent State Bytes；
- Workspace KV Bytes；
- Cell 参数 Bytes；
- 临时 Activation；
- 峰值 GPU RAM；
- Checkpoint 大小。

### 10.4 动态计算

- 每 Token 的 Cell Invocation Rate；
- 每层调用率；
- Skip Ratio；
- 每 Token 平均 Cell Calls；
- 每个 Cell 的实际使用量；
- 路由稳定性；
- 不同任务的路由差异；
- Cell 替换/禁用的因果影响；
- 动态路径与固定路径的实际成本差。

Router Entropy 只是诊断量，不是成功标准。低熵可能代表有效分工，也可能代表路由坍缩；高熵可能代表负载均衡，也可能代表无差别选择。必须结合反事实替换、任务分组和实际计算收益判断。

### 10.5 端侧相关指标

如果后续进入资源受限设备验证，额外测量：

- 峰值 RSS / RAM；
- 每 Token 能耗；
- 长时间运行温度与 Throttling；
- State/Workspace Checkpoint 大小；
- 冷启动与恢复时间。

---

## 11. 实际稀疏执行要求

动态 Cell 不能只在数学上减少调用次数。

必须测量：

\[
C_{\mathrm{total}}
=
C_{\mathrm{gate}}
+
C_{\mathrm{workspace-write}}
+
C_{\mathrm{workspace-read}}
+
C_{\mathrm{route}}
+
C_{\mathrm{dispatch}}
+
C_{\mathrm{cell}}
+
C_{\mathrm{return}}
\]

实现时优先按 Cell ID 对被选中的 Token 分组，再执行批量 Cell Kernel，避免每个 Token 单独发起小 Kernel。

必须报告：

- 逻辑激活 FLOPs；
- 实际执行 FLOPs（如可测）；
- Kernel Launch 数；
- CPU/GPU 同步；
- Dispatch 与 Gather/Scatter 时间；
- 真实 Decode Latency。

如果逻辑调用率下降但真实延迟上升，结果应判为“逻辑稀疏成立、物理效率未成立”，不能宣称加速。

---

## 12. 工程实现计划

### Phase 0 — Source / Environment Audit

- 锁定 state-spaces/mamba 的不可变 Commit；
- 记录 PyTorch、CUDA、Triton、TileLang、Python 版本；
- 记录 GPU 型号、Compute Capability；
- 检查官方 Mamba3 SISO 与 MIMO Kernel；
- 记录官方模型包装和 Inference State 结构；
- 不开始训练。

### Phase 1 — Mamba-3 Baseline

- 建立最小 LM Wrapper；
- 复现 SISO；
- 完成 Forward/Backward/Decode；
- 保存 Baseline Manifest；
- 测量模型质量、内存和速度。

### Phase 2 — Workspace Adapter

- 实现 Layer-local Bounded KV；
- 实现因果 Workspace Read；
- 完成 Cache 等价性；
- 测量 Workspace 写入和读取成本。

### Phase 3 — Fixed Cell Adapter

- 建立全局 Stateless Cell Pool；
- 固定中间层或预注册层调用；
- 不启用动态路由；
- 与参数/计算匹配的普通 Adapter 对比。

### Phase 4 — Dynamic Routing

- 加入每 Token × Layer 调用门控；
- 加入平面 Cell Router；
- 真实 Skip 必须跳过 Workspace Read 和 Cell；
- 进行路由诊断和稀疏 Kernel Profiling。

### Phase 5 — Adaptive Depth

- 允许同层多次调用；
- 加入 STOP；
- 冻结最大调用次数；
- 对比固定计算深度。

### Phase 6 — State Memory Workspace

- 独立推导 Mamba-3 Chunk State Memory；
- 不复用未经证明的 Mamba-2 DART 公式；
- 完成数学、数值和因果门禁后才实现。

---

## 13. 风险与失败条件

| 风险 | 影响 | 处理 |
|---|---|---|
| 官方 Mamba-3 Kernel 与 T4/CUDA 环境不兼容 | 无法建立可靠基线 | Gate 0 阻断，不用伪实现替代 |
| MIMO TileLang 不可用 | MIMO 路径无法比较 | SISO 继续，MIMO 标记未支持 |
| Workspace KV 按层增长 | 缓存成本侵蚀优势 | 使用有界窗口，完整报告总 KV |
| Cell 共享跨层表示不稳定 | 全局 Cell Pool 无法分工 | 先测共享池，再比较 Layer Adapter |
| Router 坍缩或随机 | 动态计算没有功能意义 | 反事实与任务分组诊断 |
| Sparse Dispatch 开销过高 | 逻辑稀疏不转化为速度 | 分组 Kernel，报告真实延迟 |
| Workspace 泄漏未来 Token | 语言模型评估无效 | 因果 Gate 失败即停止 |
| DART 公式不能迁移到 Mamba-3 | State Workspace 不成立 | 保留 Local Workspace，不强行移植 |
| 只优化 PPL | 不能验证长期回忆和动态计算 | 加入状态跟踪、MQAR、长程检索 |

---

## 14. 结果状态与科学表述

每个阶段分别记录：

- NOT_STARTED；
- ENGINEERING_BLOCKED；
- SMOKE_COMPLETE；
- PILOT_COMPLETE；
- FORMAL_READY；
- FORMAL_COMPLETE；
- NOT_SUPPORTED；
- INCONCLUSIVE。

本系列当前状态为 PROPOSED_NOT_REGISTERED。

不得把：

- Mamba-3 官方论文结果；
- DART 论文结果；
- CLM v0.4 历史 Smoke；
- CLM v0.4 被阻断的 Pilot；
- EPCLM 的开发期结果；

写成 M3C 的实验结果。

正式研究结论至少需要三个 Seed、预先冻结的对照和预算、完整的结果审计。单 Seed 只作为开发证据。

---

## 15. 交付结构

正式启动后建议采用：

~~~text
mamba3-cellular-series/
  README.md
  DESIGN.zh-CN.md
  REFERENCES.md
  M3C-001/
    README.md
    protocol.json
    environment.json
    notebook.ipynb
    reports/
  M3C-002/
  M3C-003/
  M3C-004/
  M3C-005/
  M3C-006/
~~~

每个实验必须保存：

- 不可变 Source Commit；
- Protocol 与 Config Hash；
- 数据集和 Tokenizer 身份；
- Seed；
- 实际有效 Targets；
- 模型参数量；
- 训练与推理 FLOPs；
- 设备与精度；
- Checkpoint 身份；
- 原始指标；
- 结果报告；
- Audit。

当前提案不创建空的子实验目录，不创建未经批准的训练 Notebook，不运行 GPU 训练。

---

## 16. 本系列的成功标准

M3C 不以“Cell 被调用”或“PPL 略有下降”作为成功。

最小成功条件是：

1. Mamba-3 主干保持正确，官方递归状态语义不被破坏；
2. Workspace Attention 的读取收益与缓存成本可以独立测量；
3. Stateless Cell 相比匹配 Adapter 有可重复的增益；
4. 动态路由能够根据输入和层位置选择调用或跳过；
5. 逻辑稀疏能在实际执行中转化为可测量的成本改善，或者在相同成本下带来更高任务质量；
6. 所有结论在独立 Seed 和完整审计中复现。

若 Cell 没有带来额外价值，M3C 应保留 Mamba-3 + Workspace 的结果并停止扩展 Cell，而不是为了维持研究路线继续增加复杂度。

---

## 17. 初始研究决策

第一步只授权 **M3C-001 的源码、环境与正确性审计设计**。当前不授权 GPU 训练、Checkpoint 生成或正式实验。

M3C-001 的首要输出不是一个新的模型，而是确认官方 Mamba-3 能否成为可靠的 MiniCells 实验底座。

完成该门禁后，再决定是否进入 Workspace、Cell 和动态计算阶段。

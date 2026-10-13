# MiniCells Experiment 006 — Path-First Native CLM

**实施规格 / Codex 任务文档 · Draft v0.1 · 2026-10-13**

> 目标：实现并检验第一个真正以**自主计算路径**为主要计算机制的原生语言模型，而非细粒度 MoE 或固定 Transformer 的局部 FFN Graph。
>
> 执行要求：实现全部代码、实验对照、CPU Gate、审计记录、可复现 Kaggle 入口及结果分析工具；**不得启动正式 GPU 训练，直到协议和资产正式注册并明确授权**。不修改 Experiment 005 的冻结协议、代码或历史结果。

## 0. 任务性质与最终交付

本任务不是讨论稿，而是工程实施规范。Codex 应在 `ArcheLabs/mini-cells` 中创建独立的 Experiment 006 研究单元，完成可运行、可测、可复现的实现，进行静态检查和 CPU 预检，提交并推送独立分支，回报 commit、CI、Gate 状态与仍未冻结事项。无需再进行独立的合成组合能力科研阶段（原提议的 006-A）；**允许仅为验证梯度、因果性和模型运行正确性构造小尺寸单元测试**，不得将这种测试宣传为 Path-First 的科学成果。

科学主张分为三个独立层次，不得混用：

1. **有效语言模型**：Path-First 能否直接从 next-token loss 训练并形成有竞争力的语言能力。
2. **路径的独立贡献**：相比只看状态的动态路由，利用前驱/Connection 的路径控制能否改善**质量—实际计算量**权衡。
3. **按需计算**：STOP 是否能在质量约束下减少平均必要计算。此主张只有 R3 才能验证，R0–R2 不得预先宣称。

不将「更多 Cell」「图发生变化」「路径可视化复杂」「参数激活率低」「训练成功」单独作为 CLM 价值的证明。

## 1. 仓库证据与约束（只读引用）

以下是已经在仓库检查到的历史证据，不得重写其含义：

- `research/06-native-clm/experiments/constructive-clm-004-model-level-multicell-computation/`：受控线性残差算子世界中，顺序/并行 Cell 组合及路径顺序恢复得到正式支持；**不等于语言级路径学习成功**。
- `research/06-native-clm/experiments/constructive-clm-005-scaffold-removal/`：受控世界中的学习型 Router/控制器得到支持；不是任意复杂语言计算的证据。
- `src/minicells/native_clm_v0.py` 和 `research/06-native-clm/hypothesis/M1_CLOSURE.md`：约 12.15M、8 个 Cell、Top-2、单个 Cellular Layer 的 next-token 训练通过；它保留固定六层 Transformer 且 Cell 并行混合，**不是 Path-First**。
- `src/minicells/sparse_cellular_textnca.py`、`src/minicells/overcomplete_cellular_textnca.py`：已有循环状态更新与条件计算；前者的稀疏质量门槛和后者的密集 Scaffold 交接均出现负面实验结果。尤其应避免「局部拟合良好，整条轨迹失真」。
- `research/08-hybrid-clm/experiments/005-compositional-cell-graph/`：G0 在固定 Transformer 层内走四跳，硬图限制、Top-2 backup 梯度、外部 Graph refresh；开发阶段效果不佳。005 的正式历史、资产、种子、确认 holdout 和冻结评估规则保持原状。中止记录不得冒充完成确认实验的科学结论。
- `research/audits/CLM_CAPABILITY_CEILING.md`：历史实验尚未支持自治持续语言学习、自然 Cell 本体或可保证安全的局部写入。006 不承诺解决这些问题。

**强制保护范围**：不得修改或重新解释 003/004/005、Native M1/M2/M3 或 Constructive CLM 的冻结协议、正式结果和 consumed seeds；不得读取/使用 005 预留的 confirmation holdout 进行 006 开发；不得利用 005 confirmation 种子充当 006 untouched confirmation。

## 2. 核心架构原则

1. **Path 为一等公民**：下一操作由已执行计算得到的当前状态、前驱操作及可用因果上下文共同决定；路径随每个 Token 变化，可复用、重复调用相同 Cell。
2. **不依赖固定六层 Transformer 主干**：允许最小输入表示/可复用因果上下文读取接口；主要计算必须由共享模块池的递归执行构成，不得通过保留大部分固定 Transformer 后加少量 Cell 来获取语言质量。
3. **Connection 初期是可学习的转移先验，而非不可绕开的布尔图**：全部模块可选。若实验以后需做稀疏图，应另起受控实验，不纳入 006 第一版。
4. **路径选择和 Cell 计算必须共享最终语言模型目标**：不得把强制前驱+候选的局部 probe NLL 当成主要 Connection 学习目标。
5. **路径质量与实际计算共同评估**：记录全部常驻参数、每 Token 真正触及的不同参数、重复 Cell 调用、实际 FLOPs、路径长度、上下文读取次数、路由器开销、时间及精度；不把独立访问参数等同于 RAM 使用。
6. **先测固定步数路径，再测 STOP**：第一阶段 R0/R1/R2 使用相同最大执行步数；R3 只能在这三者完成开发审阅后作为第二阶段启用。
7. **无端侧工程优化范围**：不做 GGUF、量化、闪存分页、预取、手机适配、CUDA 专用算子、Cell 生长、局部保护写入、证明/证书或 JAM 部署。

## 3. 形式化定义：Path-First LM v0

### 3.1 Token、状态和只读因果记忆

给定输入 Token `x_1 ... x_N`，每个位置 `n` 先构造 `z_n = E(x_n) + Pos(n)`。从 `z` 生成只读、与路径无关的键值记忆 `K_n = W_K z_n`、`V_n = W_V z_n`。这部分可以包括必要的轻量 LayerNorm/投影，但不能暗中加入完整的深层 Transformer 主干。

每个位置的计算状态初始为 `h_{n,0} = Init(z_n)`，前驱身份 `i_{n,0} = START`。上下文读取模块 `CTX_READ` 的查询来自当前 `h_{n,t}`，仅可读取 `K_{<=n}, V_{<=n}`：

\[
\mathrm{Read}(h_{n,t}, M_{\le n}) = W_O\,\mathrm{Attention}(W_Q h_{n,t},K_{\le n},V_{\le n}).
\]

**明确此版本的研究边界**：上下文记忆是只读且与路径无关的输入记忆。它保证预填充和增量解码有统一、可审计的因果语义，允许位置间路由独立训练，但暂不研究将前面 Token 的已完成动态路径状态写回后续 Token 的记忆。后者是将来扩展，不得在 v0 偷偷实现或宣称已证明。

### 3.2 计算模块与更新

共享模块池：`Cell[0 .. C-1]`（同一输入/输出隐藏维度的可学习残差变换）和特殊可重复调用的 `CTX_READ`。每个 Cell 可简单采用 `FFN(d, d_cell, d)`；不需要为了“粒度”而把传统 FFN 人为等分。定义操作集合：

\[
\mathcal A = \{C_1,\ldots,C_C,\mathrm{CTX\_READ}\}.
\]

对于所选操作 `a_{n,t}`：

\[
h_{n,t+1}=h_{n,t}+\alpha_a F_a(\mathrm{LN}(h_{n,t}),M_{\le n}),\qquad i_{n,t+1}=a_{n,t}.
\]

`Cell` 仅作用于当前状态，`CTX_READ` 读取静态因果记忆。`alpha_a` 可固定为与最大步数有关的稳定小值（例如 `1/sqrt(T_max)`，作为待冻结提议），或采用有界可学习门；**所有 R0/R1/R2 使用同一种状态更新和相同初始化**。同一模块允许在一条路径中多次出现，不得禁止循环。

前 `R0/R1/R2` 统一 `T_fixed` 次决策（建议先提议 6 次，最终以注册配置为准）。`h_{n,T}` 经共享 `LN + tied LM head` 输出 Token logits。不能再在路径之后叠加未经计费的额外 Transformer Block。

### 3.3 三种路由的唯一差别

令 `t` 是当前步数，`a` 是候选操作，`i` 是前驱操作：

- **R0 — Static Recurrent Path**：`q_0(a|t)=softmax(b_t(a))`。可学习每步的全局选择，但不看 Token 内容；推理中每个位置采用相同确定性模块序列。与其他臂共享同一模块池和计算语义。
- **R1 — State-Conditioned Path**：`q_1(a|h_t,t)=softmax(g_phi(h_t,t)_a)`。每步可按当前状态选择操作，不输入前驱身份或边先验。
- **R2 — Connection-Conditioned Path**：`q_2(a|h_t,i_t,t)=softmax(g_phi(h_t,t)_a+e_phi(i_t,a))`。`e_phi` 可用低秩 source/target Embedding 内积。全候选可选，无布尔 adjacency 或前期锁边。

必须保证 R1、R2 的状态打分网络等价；R2 相对 R1 只增加前驱/Connection 信号。总参数差异应小于 1%，差额必须报告，不准用无法执行的“假参数”填充配平。

必须报告一个 R2 **Connection-removal** 干预：在完全相同的已训练权重和状态执行方式下，将 `e_phi(i,a)` 置零/打乱，检验前驱信号是否确实贡献 NLL 改善。还必须在相同模块使用多重集合下打乱执行顺序，测量顺序贡献。前两种干预分别评价转移先验和顺序，不能将任何非交换性本身当作有用连接的证明。

### 3.4 第二阶段 R3 — Adaptive Path

在完成 R0/R1/R2 开发审阅后，将 `STOP` 加入 R2 操作空间，设 `1 <= T(x) <= T_max`。STOP 不运行 Cell，使用停止时的状态输出。训练目标加入真实执行成本（包括 CTX_READ、Controller、重复调用）：

\[
\mathcal J=\mathbb E_{P\sim q_\phi}\Big[\mathcal L_{LM}(P)+\lambda\,\mathrm{Cost}(P)\Big].
\]

R3 以**质量—计算 Pareto 曲线**为判断依据，不能只报告平均路径长度减少；保留 `T_max` 内强制 STOP 的稳定边界。`lambda` 的候选范围及选择方式必须先在开发协议中冻结，不可事后根据 confirmation 挑选最佳。

## 4. 路径级训练：不再重复 G0 的局部近似错误

### 4.1 默认训练方法：On-policy hard trajectory + score-function gradient

R1/R2 在训练时依策略 `q_phi` 逐步**采样并真实执行一条硬路径**；不在前向将多个 Cell 的输出混合冒充所选择的路径。每个被执行模块和输出 Head 通过标准反向传播接受 LM 梯度；Controller 的离散选择使用整个 rollout 对应的 policy-gradient 信号。

此 v0 的静态因果记忆**不受其他 Token 的路径行动影响**，因此可以把该位置的 next-token NLL `ell_n(P_n)`（加路径成本）作为该 Token 路径的回报/损失：

\[
J_n=\ell_n(P_n)+\lambda\operatorname{Cost}(P_n).
\]

控制器的无偏 score-function 梯度（在所定义的随机路径目标及适当可微性条件下）为：

\[
\nabla_\phi\mathbb E[J_n] \supseteq \mathbb E\Big[\sum_t (J_n-b(s_{n,t}))\nabla_\phi\log q_\phi(a_{n,t}\mid s_{n,t})\Big]
\]

上式刻画离散行动概率的 score-function 项；如 `J_n` 还通过连续计算显式依赖控制器参数，须另加对应的路径内导数。状态依赖 Baseline `b(s)` 在 Actor loss 中必须 `stop_gradient`，不能泄漏标签进入 inference Router。实现时应把 `J` 对 Cell/记忆/Head 的普通路径梯度与 Controller 的 score-function 项正确分开，避免重复计梯度。可采用 Actor-Critic/移动平均 Baseline、熵正则和逐步 reward-to-go 降低方差，但这些改动必须在开发配置登记；训练与推理分别记录 `sampled` 与 `greedy` 指标。

**重要限制**：如果未来允许 Token A 的已路由隐藏状态进入 Token B 的因果记忆，上述“每位置 NLL 回报”不再自动有效，必须改为包含未来受影响 Token 的 reward-to-go 或完整序列目标。本任务的代码和文档必须将该假设显式断言。

### 4.2 不允许使用的默认捷径

- 不得把 005 的 `top2_backup_gradient` 当作 R2 的唯一路径梯度来源。
- 不得以强制单条边、少量固定窗口上的 NLL 改善作为自然路径优化的主要监督。
- 不得只在密集教师轨迹上训练局部 Cell、然后未经 closed-loop 验证就称已实现自洽路径学习。
- 不得在训练时全部 Cell 加权执行，推理时突然 Top-1/硬路径，却未给出独立 train–inference gap 测试。

### 4.3 必备数值稳定性与诊断

- Controller、Cell、输入记忆分别记录梯度范数、非有限率、更新幅度；独立记录 Actor loss、Critic loss、CE、熵和路径成本。
- 记录采样与贪心 PPL、轨迹分布差距、模块使用熵、START/前驱条件分布、重复调用率、`CTX_READ` 频率和路径多样性。
- 监控路由塌陷（所有 Token 选择同一路径）、无效梯度（Controller 无法获得梯度）、Cell 饥饿、上下文读取完全不使用、过度重复无效操作。
- 增加极小离散控制器的精确期望梯度对照**单元测试**（例如 2 个动作、2 步，枚举所有路径），检验 PG 估计方向/数值与正确目标相符。这只是数学/代码 Gate，不是另做 006-A 合成科研。
- 明确训练时每 Token 轨迹采样、批次权重、损失归一方式；避免除以随机路径长度后悄然改变优化目标。

## 5. 公平基线与训练矩阵

**主实验第一阶段（必须完成）**：

| Arm | 配置 | 唯一变化 | 主要科学问题 |
|---|---|---|---|
| R0 | Static Recurrent Path | 每步全局固定/可学习动作，不依赖输入状态 | 单纯递归计算是否足够？ |
| R1 | State-Conditioned Path | 当前状态决定下一操作 | 动态模块选择是否有价值？ |
| R2 | Connection-Conditioned Path | 在 R1 基础上加入前驱/转移先验 | 路径历史是否产生**独立增益**？ |
| M0 | Matched MoE Baseline | 标准 Top-k MoE 语言模型，重新在相同资产下训练 | Path-First 是否比现实强基线有潜力？ |

`M0` 既有 005 结果仅可作历史上下文，**不能直接当作 006 完整对照**；若参数规模、Tokenizer、训练预算、上下文长度、优化器或评估集不同，须在 006 相同设置下重训标准 MoE。若资源有限，先完成 R0/R1/R2 的开发 Gate，再启动 M0；最终任何对 MoE 的优越性主张必须有 M0 公平基线。

R0/R1/R2 使用同一只读因果记忆、初始化策略、模块池、输出 Head、训练 tokens、上下文长度、优化器和可执行操作集合。允许模型路由导致每 Token 操作比例不同，但必须报告并基于**实际 FLOPs** 做质量—成本对比，尤其将 `CTX_READ` 次数和成本单独归档。R2 的优势不能靠执行更多、更昂贵的上下文操作换取而不说明。

建议先做 12–20M 规模的**开发可行性配置**（参数规模最终需由 Codex 的实测计数确定），再按已有 005 的约 61M 规模建立公平放大配置。不能因为单种子小模型结果好就直接宣布 30B 可行，也不能因 v0 静态输入记忆带来的质量不足就直接否定所有动态路径架构。

R3 仅在 Gate 3 审阅后实施，使用同一 Cell Pool、Router、ContextRead 和真实成本定义，改变 STOP 与可变步数功能；必须和预算匹配的固定步数 R2 比较。

## 6. 数据、种子与评估协议

1. 初步建议沿用 TinyStories next-token 训练体系和已验证的资产下载/校验辅助模块，但创建 006 自己的**独立协议、资产清单、seed namespace 和 holdout 身份**。不得读取 005 的预留 confirmation holdout，也不能把从开发曲线见过的数据当作新确认集。
2. 数据完整性：来源固定 revision、原始 Parquet/文本 SHA256、Tokenizer 哈希、训练/开发/确认故事级不重叠验证、重复故事检测、所有可用窗口身份；若数据集不够划分独立确认，停止注册并重新选择来源，禁止复用泄漏集。
3. 每个 Arm 相同训练 tokens、micro/global batch、学习率日程、随机初始化方案和优化器实现；初始化共享权重按参数名/作用匹配，避免由模型构造顺序带来结构间随机偏差。
4. 开发阶段可先完成单个 seed 的短跑/全跑，结果仅用于机制诊断；正式确认预留**至少 3 个未经观察的种子**。代码必须禁止在 confirmation 授权前使用其数据和种子。
5. 在看到 confirmation 结果前冻结配置、数据身份、门槛、统计方法、最优 checkpoint 选择规则以及消融顺序。所有更新必须形成显式协议修订和新的源码 SHA，保留旧证据。
6. 所有结果按**故事分组的配对 Bootstrap**分析 NLL/PPL 差异；不得将同一故事内重叠窗口当作独立观察。记录 Bootstrap 重复数、随机种子、CI、样本量。
7. 质量：报告 CE/NLL、PPL、训练曲线、样例质量及 train vs greedy path 差距；记录 total params、实际触及不同权重、总 GEMM FLOPs、非 GEMM、上下文 Attention 成本、Controller 成本与实际路径调用。实测延迟记录但不得把当前 PyTorch 小 Kernel 性能自动解释为数学模型的计算效率。
8. Decode：采用真实因果 Prefix 测试，验证 `full forward` 与逐 Token `prefill + decode` logits 在冻结容差内一致；必须验证修改未来 Token 不改变先前 Token 输出。随机路径测试须固定并重放动作序列，避免把随机选择差异误判为 Decode 错误。

### 候选科学门槛（需注册前再审查）

- **语言训练可行性**：R0/R1/R2 全部完成训练且 next-token NLL 稳定下降，无非有限或重度路由塌陷；仅支持“可训练”。
- **状态路径收益**：R1 相对 R0 在相同/更低实际计算预算下有可重复 PPL 改善。
- **Connection 独立收益**：R2 相对 R1 在**等 FLOPs 或成本不增加的条件下**，配对 Bootstrap 的 PPL ratio 95% CI 上界 < 1；三个确认种子方向一致。开发参考阈值可提议改善至少 1%，但**数值必须在正式注册时明确冻结**。
- **路径因果性**：在 R2 已训练权重上取消/打乱 edge prior、相同 Cell 多重集合改变顺序，应产生方向合理、可重复的质量差异；但该消融不能单独代替 R2 vs R1 的整体因果对照。必须报告 ROUTER 内容条件性及实际路径分布，排除退化成静态序列。
- **对 MoE 的架构竞争力**：R2 还需对 006 内独立训练 M0 报告 PPL—真实 FLOPs Pareto，若质量仍明显落后，结论必须写为 `PATH_MECHANISM_SIGNAL_ONLY`，不能声称优于 MoE。
- **STOP 有效**（仅 R3）：与预算相当的固定深度 R2 对照，同等可接受 PPL 下平均真实 FLOPs 显著降低；不同难度输入的成本分配可检验，禁止将“总执行次数小于最大次数”单独称为成功。

完整数值阈值、候选规模、GPU 预算在实验注册前冻结；本草案绝不通过叙述提前替代正式注册。

## 7. Gate 和执行授权

- **Gate 0 — CPU 工程预检**：所有 Arm 创建、参数计数、forward/backward、单步梯度、控制器 PG 小规模枚举验证、无未来泄漏、重复 Cell、样本固定路径复现、decode 对齐、Checkpoint 中断恢复、资产哈希门禁、报表解析及单元测试通过。可以执行，不消耗 GPU。
- **Gate 1 — 协议注册**：资产、Seed、独立开发/确认集、操作集合、模型规模、训练 budget、R2 阈值、统计、Profile/发布格式均冻结；代码 `REGISTERED`。Codex 不得擅自注册“科学支持”，必须保留用户审核入口。
- **Gate 2 — GPU 开发**：用户明确授权后，短跑 R0/R1/R2 → 完整开发 seed R0/R1/R2 → 补训 M0；汇总质量—FLOPs 曲线和核心消融。开发阶段不能读取 confirmation。
- **Gate 3 — 研究审阅**：检查 R2 是否真正优于 R1、有没有静态路由塌陷、实际成本是否正常、上下文依赖是否有效，以及是否存在训练/推理路径分布鸿沟。负面结果可决定停止，不得隐瞒。
- **Gate 4 — 形式确认**：只有 Gate 3 明确批准后，使用原先预留的至少 3 个 untouched confirmation seeds/独立 holdout 完整执行及配对分析。不能“看完结果再定阈值”。
- **Gate 5 — R3**：固定深度路径收益得到可信证据后单独启动，独立注册 STOP/成本规则；不得混进 R0–R2 的首次正式注册。

若 M0、R0、R1、R2 某臂异常中止，应产生 `INCOMPLETE`，而不是自动用现有已完成臂得出正式科学结论。

## 8. 工程目录与接口

推荐路径（以仓库现状自适应，不要复制过时的兼容路径）：

```text
research/08-hybrid-clm/experiments/006-path-first-clm/
  README.md
  PROTOCOL_DRAFT.md
  protocol.json                # PROPOSED_NOT_REGISTERED -> REGISTERED
  config/proposed.json
  registration/README.md      # 注册前写明 NOT REGISTERED
  implementation/exp006/
    __init__.py
    config.py
    data.py
    memory.py                 # 静态、因果、只读上下文
    cells.py                  # 共享 Cell + ContextRead
    controller.py             # R0/R1/R2，R3 留接口
    model.py                  # 真正的路径执行器
    training.py               # on-policy hard trajectory / critic
    accounting.py             # 成本、FLOPs、触及权重
    profiling.py              # batch1 与缓存对齐
    checkpoint.py             # 恢复及随机状态
    evaluation.py             # NLL/PPL、干预、CI
    report.py                 # 分臂、分 seed、最终判定
    assets.py                 # 哈希及故事分组门禁
    publish.py                # 追加式、不可变结果档案
  notebook.ipynb              # Kaggle 的手动 GPU 入口
  AUDIT.md
scripts/research/
  run_experiment_006.py
  analyze_experiment_006.py
  publish_experiment_006.py
tests/research/path_first_006/
  test_model.py
  test_causal_and_decode.py
  test_policy_gradient.py
  test_routing_controls.py
  test_checkpoint_assets.py
  test_accounting_report.py
.github/workflows/  # 仅添加 CPU lint/test/preflight
```

CLI 至少支持：

```bash
python scripts/research/run_experiment_006.py --mode preflight
python scripts/research/run_experiment_006.py --mode dry-run
python scripts/research/run_experiment_006.py --mode train --arm R1 --seed <registered-dev-seed>
python scripts/research/run_experiment_006.py --mode profile --arm R1 --seed <registered-dev-seed>
python scripts/research/analyze_experiment_006.py --seed <registered-dev-seed>
```

未注册时 `train/profile/publish` 不得绕开 Gate；是否允许非正式 CPU toy 参数测试由明确的 `--mode preflight` 执行，不可重用真实 confirmation 信息。Notebook 默认采用固定完整源 SHA、注册资产哈希、现有安全的 Kaggle Secret/Git publish 机制；不在 Notebook 或日志中暴露 HF/GitHub token。不存在可执行的正式注册时，Notebook 需给出明确阻断信息。

### 结果 Schema 要求

每个 `arm/seed` 归档至少包含：`identity`（source SHA/config/protocol/data hash/seed/role）、`status`、`model_config`、`checkpoint_sha`、`learning_curve`、`development_eval` 或 `confirmation_holdout`、`routing_diagnostics`、`compute_accounting`、`decode_profile`、`interventions`、`runtime_versions`。失败/中断必须存档阶段、原因和已完成检查。大型模型权重与原始数据放独立安全存储，Git 只存不可变元数据、统计和内容哈希。

## 9. 必须实现的自动验收

- [ ] R0/R1/R2 共享同一个模块池、ContextRead、状态更新和 Head；R2 才读取前驱 ID/Connection。
- [ ] 硬路径确实被执行；前向每步只调用被选择模块，不偷偷执行所有 Cell。
- [ ] `CTX_READ` 严格 causal；修改未来输入不能影响较早位置 logits。
- [ ] ContextRead 可以被重复选择；Cell 可以被重复执行，不因某个位置未访问产生非法梯度。
- [ ] PG estimator 在可枚举微型路径上的数值/方向测试通过；Controller 梯度非零且稳定。
- [ ] 全部 Arm 的实际 FLOPs/Controller/ContextRead/重复调用可审计；参数容量差异 <1% 或明确阻断。
- [ ] 配对故事群 Bootstrap 正确，身份不一致不能计算科学结论。
- [ ] Connection-removal、edge-prior shuffle、same-multiset order permutation 的指标分开存档。
- [ ] Full forward vs incremental decode 在确定性动作重放下对齐。
- [ ] Checkpoint 恢复保留模型/优化器/随机数/路径采样器/训练进度与数据顺序身份。
- [ ] 005 的 source/protocol/结果文件未被修改，旧 CI 回归不受影响。
- [ ] 未正式注册时，GPU 训练入口明确拒绝执行。
- [ ] GitHub Actions 完成 CPU 测试并输出可核验记录。

## 10. 实施步骤（Codex 依次完成）

1. **Read-only 调查**：核对本说明列出的 Native v0、CLM v2、Constructive 004/005、Experiment 005、历史审计及仓库现行目录。先输出架构差异和计划复用清单，不得修改旧研究。
2. **独立目录与 Proposal**：建立 006、配置 Schema、注册门禁和文档；采用新 experiment id 与 seed namespace，结果发布位置独立。
3. **最小模型**：实现因果静态记忆、Cell Pool、CTX_READ、R0/R1/R2 路径控制器、真正的串行状态执行、可重放 trace 和标准 LM Head。R3 的 STOP 接口可以预留，但不得影响初版比较。
4. **训练与诊断**：实现硬路径 on-policy 训练、PG/Baseline、稳定机制、成本计数、干预与完整路径 telemetry。
5. **Baseline 与评价**：实现或严格适配同训练条件的 M0、故事级 NLL/PPL、paired CI、Decode、实际 FLOPs/延迟以及结果的可审计 schema。
6. **工程质量**：添加 CPU 测试、故障注入/断点恢复、完整数据哈希检查及 CI，执行 Gate 0。
7. **Kaggle 与发布**：复用现有安全发布链路，固定 SHA 与手动执行方式；训练仍处于 Gate 1 阻断状态。
8. **交付**：提交并推送，报告分支、完整 SHA、主要文件、CPU 测试清单、Gate 0 结果、参数计数、未冻结协议参数、是否触碰 005、是否运行 GPU（必须为否）。

## 11. 不得误报的科研状态

- `ENGINEERING_READY`：实现、CPU Gate 通过，**不代表模型有效**。
- `DEVELOPMENT_SIGNAL`：开发集有路径收益，**不代表跨种子科学支持**。
- `PATH_MECHANISM_SUPPORTED`：只有预注册确认种子 + holdout、R2/R1 的成本公平比较、路径消融及质量门槛全部满足。
- `COMPUTE_ADAPTIVITY_SUPPORTED`：另需 R3 的停止机制、同质量计算成本收益和独立确认。
- `SYSTEM_SUPERIORITY_SUPPORTED`：另需对标准 MoE 的同预算真实系统对照；若只是 FLOPs 更少而真实延迟更差，不能宣称已更快。
- `NOT_SUPPORTED` / `INCOMPLETE`：如实登记，不应修改已观察种子或门槛将结果改成支持。

**最终验收标准不是生成一张可视化路径图，而是交付一个可训练、可审计、会因计算状态而改变路径、并且能明确检验前驱连接是否带来独立语言建模收益的真实语言模型。**

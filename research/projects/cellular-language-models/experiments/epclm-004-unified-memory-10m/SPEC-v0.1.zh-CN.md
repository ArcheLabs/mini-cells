# EPCLM-003/004 --- Unified Memory Architecture & 10M Experimental Specification

**项目：** MiniCells / EPCLM\
**文档编号：** EPCLM-003/004\
**版本：** v0.1\
**日期：** 2026-11-08\
**状态：** Codex 实施规格（Research MVP）\
**关联文档：** EPCLM-002 --- Persistent State Theory v0.2\
**目标：** 在约 10M 静态参数规模下，比较 SSM State、Sliding Window
Attention、B'MOJO 式选择性精确存储和 Titans Neural Memory
的作用，并验证它们是否具有可测量的互补性。

------------------------------------------------------------------------

## 0. 文档用途与执行原则

本文件是下一阶段实验的唯一实施规格。它同时包含：

1.  三篇核心论文的统一机制对照；
2.  MiniCells 当前研究假设；
3.  10M 参数实验的模型矩阵和资源预算；
4.  模块接口、数据契约和状态生命周期；
5.  训练、评估、消融和复现要求；
6.  Codex 可分阶段执行的工程任务和验收标准。

本文件不是对三篇论文的完整复现承诺。10M
模型与原论文的模型规模、数据量和训练预算差距很大。本项目的目标是**机制验证**，不能把
10M 实验结果外推为大模型性能结论。

### 0.1 证据等级

全文使用以下标记：

-   **\[PAPER\]**：原论文明确提出或报告的内容。
-   **\[PROJECT-HYPOTHESIS\]**：MiniCells 当前提出、尚待验证的研究假设。
-   **\[MVP-CHOICE\]**：为了让实验可实施而作出的工程选择，不代表论文原始设计。
-   **\[OPEN\]**：当前没有充分证据，需要实验或进一步研究。

任何实验报告都必须沿用这套区分。不得将项目假设写成已被论文证明的事实。

### 0.2 总体研究问题

> 不同记忆机制能否协作，形成一个持久、可组合、可独立演化的语言模型记忆系统，并在相同静态参数和可比计算预算下，改善跨
> Chunk、跨 Session 的信息保留与调用？

具体分为四个问题：

1.  SSM 是否提供了 Titans Neural Memory 之外有价值的递归时间状态？
2.  Titans Neural Memory 是否比普通 SSM State
    或外部精确存储更擅长学习和恢复关联？
3.  Sliding Window Attention 是否足以承担短期精确 Token 交互？
4.  SSM、Titans 和 SWA 组合后是否产生互补收益，还是存在明显冗余？

------------------------------------------------------------------------

# 1. 核心论文及证据边界

## 1.1 Samba：Mamba + Sliding Window Attention

**论文：** *Samba: Simple Hybrid State Space Models for Efficient
Unlimited Context Language Modeling*\
**会议：** ICLR 2025\
**论文：**
https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html\
**官方代码：** https://github.com/microsoft/Samba

**\[PAPER\] 核心机制**

Samba 将 Mamba 选择性状态空间层与 Sliding Window
Attention（SWA）按层组合。官方代码将其概括为：

`Mamba + MLP + Sliding Window Attention + MLP`

其目标是同时获得：

-   Mamba 对历史序列的递归压缩；
-   SWA 对近期 Token 的精确交互；
-   随序列长度近似线性增长的推理路径。

**\[PAPER\] 论文报告**

-   最大模型约 3.8B 参数；
-   训练数据约 3.2T Tokens；
-   以 4K 长度训练后，报告最长至 1M 上下文的零样本 PPL 长度泛化；
-   在特定微调设置下，256K Passkey Retrieval 达到完美召回；
-   对 128K 输入，报告相对 GQA Transformer 的吞吐优势。

这些结果属于论文的大规模实验，不代表 10M
模型能够复现相同长度泛化或性能。

**对 MiniCells 的意义**

Samba 是验证"SSM State + 局部精确 Attention"的基线。它没有 Titans Neural
Memory，也没有用户级独立知识模块。

**本项目不做的事**

第一轮不重新实现或优化 Mamba Kernel。使用官方 Samba 代码所依赖的 Mamba
实现，固定 Git commit 和依赖版本。不得在未记录的情况下将 Mamba-1 替换为
Mamba-2 或 Mamba-3。

## 1.2 B'MOJO：Fading State + Innovation-selected Eidetic Storage

**论文：** *B'MOJO: Hybrid State Space Realizations of Foundation Models
with Eidetic and Fading Memory*\
**会议：** NeurIPS 2024\
**论文：**
https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb6281d3f2cfe2eb8b1307227d3677b3-Abstract-Conference.html\
**PDF：**
https://proceedings.neurips.cc/paper_files/paper/2024/file/eb6281d3f2cfe2eb8b1307227d3677b3-Paper-Conference.pdf

**\[PAPER\] 四类记忆**

B'MOJO 将信息保存方式区分为：

1.  Short-term eidetic memory：当前局部上下文；
2.  Permanent structural memory：模型权重；
3.  Fading memory：SSM State；
4.  Long-term eidetic memory：外部 Storage 中选择性保存的历史 Token。

**\[PAPER\] Innovation Selection**

B'MOJO 通过 SSM 输出的预测误差评估新信息是否难以由当前 fading state
解释。如果误差足够大，就将相应输入 Token 加入 eidetic memory。

概念流程：

``` text
input token
   ├──> SSM state update ──> fading output
   │                           │
   │                           └──> prediction error / innovation
   │                                      │
   │                                      └──> select token for eidetic storage
   │
   └──> local input window

local input window + fading output + selected eidetic tokens
   └──> Sliding Window Attention
```

原论文使用固定预测器实现 Innovation Selection，并讨论了 running weighted
average / causal convolution 形式。B'MOJO 的 eidetic memory
可以随数据增长，直到资源限制；它不是固定大小的 Titans MLP。

**\[PAPER\] B'MOJO-F**

B'MOJO-F 不使用 Innovation Selection，因此没有长期 eidetic Token
Storage。它保留 fading state 与局部 Attention 的组合。

**\[PAPER\] 论文报告**

-   在 Multi-Query Associative Recall 等 transductive inference
    任务中，B'MOJO 优于论文比较的 SSM 和 Hybrid 模型；
-   语言建模 PPL 与同规模 Transformer / SSM 相当，报告规模最高到 1.4B；
-   训练速度最高报告约 10% 优势；
-   在最长 8K 训练长度的设置下，测试到 32K。

**对 MiniCells 的意义**

B'MOJO 提供了一个关键对照：当 SSM State
对某些历史信息压缩不足时，系统可以选择性保留原始历史 Token。

它证明的是"SSM + 选择性精确外部记忆"的组合，不是"SSM + Titans Neural
Memory"。

**\[MVP-CHOICE\] 有限容量适配**

原论文的 eidetic storage
可增长至硬件限制。本项目为了可比较和可控资源，使用固定容量 Top-K
存储。这是 B'MOJO-inspired adaptation，不应标记为完整 B'MOJO 复现。

## 1.3 Titans：Test-time Neural Memory 与 MAG

**论文：** *Titans: Learning to Memorize at Test Time*\
**会议：** NeurIPS 2025\
**论文：**
https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html\
**PDF：**
https://papers.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf

**\[PAPER\] Neural Memory**

Titans 使用小型深层 MLP 作为 Neural
Memory。其参数在测试时通过梯度更新，以学习 Key-Value 关联。

简化目标：

\[ `\mathcal`{=tex}L\_{`\mathrm{mem}`{=tex}} =
\|M\_{`\theta`{=tex}}(k_t)-v_t\|\_2\^2 \]

其中：

-   (k_t=x_tW_K)
-   (v_t=x_tW_V)
-   (q_t=x_tW_Q)
-   读取结果为 (M\_{`\theta`{=tex}}(q_t))

记忆更新包括：

-   momentary surprise：当前输入产生的梯度；
-   past surprise：由动量保留的历史更新信号；
-   adaptive forgetting：根据输入调节遗忘；
-   inner-loop：更新 Neural Memory 的动态参数；
-   outer-loop：训练 Backbone、投影、Gate 和记忆初始化等静态参数。

简化更新形式：

\[ S_t=`\eta`{=tex}*t
S*{t-1}-`\theta`{=tex}*t`\nabla`{=tex}*`\theta`{=tex}`\mathcal`{=tex}L\_{`\mathrm{mem}`{=tex}}
\]

\[ M_t=(1-`\alpha`{=tex}*t)M*{t-1}+S_t \]

实现时应遵循原论文更完整的张量形状和逐层更新规则，不得把上述简式误当成完整实现。

**\[PAPER\] MAG**

Memory as a Gate（MAG）并行使用：

-   SWA：短期精确交互；
-   Titans Neural Memory：长程参数化记忆；
-   Gate：融合两条分支。

MAG 的 Gate 与 Gated Attention 论文中的 Attention Output Gate 不同。

**\[PAPER\] 其他 Titans 变体**

-   MAC：Memory as a Context；
-   MAG：Memory as a Gate；
-   MAL：Memory as a Layer；
-   LMM：仅 Neural Memory 的序列模型对照。

**\[PAPER\] 论文实验边界**

Titans 原论文的语言模型规模从 170M 起，远大于本项目的
10M。论文的训练配置还包括 32K Vocabulary、4K 训练长度、SWA 窗口
2K、Memory chunk size 16 等设置。不能把 10M 模型视为论文结果复现。

**对 MiniCells 的意义**

Titans 是本项目验证"独立、可持续学习的参数化记忆"最直接的基础机制。

但原论文没有直接证明用户级 Cell Registry、独立授权、跨用户 Fork 或跨
Backbone 通用移植。

## 1.4 相关辅助研究

### TNT

**论文：** *TNT: Improving Chunkwise Training for Test-Time
Memorization*\
**会议：** ICLR 2026\
**论文：**
https://proceedings.iclr.cc/paper_files/paper/2026/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html\
**PDF：**
https://openreview.net/pdf/424813e8424a5a7c3af990a764d389ec5b389490.pdf

TNT 研究 Deep Memory 的 chunkwise 训练效率。其 Global Memory 跨较大
Chunk 传递状态；多个 Local Memory
在训练阶段周期性重置，以打破长程串行依赖；第二阶段再使用较小 Chunk 微调
Local Memory。

TNT 的 Global / Local Memory 是 Titans / TTT 类 Deep Memory
的层级训练设计，不是 Mamba SSM
State。它不属于第一轮核心模型矩阵，但可以作为后续优化 Titans
训练吞吐的参考。

TNT 还提出 Q-K Projection，处理 Memory 以 Key 域训练、却以 Query
域读取造成的输入域偏差。该机制应作为 Titans Retrieval
的后续独立消融，不要在首个 MAG 实现中未经记录地加入。

### Gated Attention

**论文：** *Gated Attention for Large Language Models: Non-linearity,
Sparsity, and Attention-Sink-Free*\
**会议：** NeurIPS 2025\
**论文：**
https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html\
**代码：** https://github.com/qiuzh20/gatedattention

该论文的有效方案是在 SDPA 输出后增加 head-specific sigmoid gate：

\[ Y' = Y `\odot`{=tex}`\sigma`{=tex}(XW_g) \]

该 Gate 调节 Attention 自身输出，不会扩大 SWA
的可见窗口，也不会自动提供窗口外长程访问。

第一轮不将它与 MAG Gate 混为一谈。先完成普通 SWA 和 MAG，再把 Gated
Attention 作为单独消融。

------------------------------------------------------------------------

# 2. 统一机制模型

## 2.1 五种信息保存机制

  ----------------------------------------------------------------------------------------------------------
  机制        保存对象                 写入方式               读取方式    主要限制
  ----------- ------------------------ ---------------------- ----------- ----------------------------------
  Full        历史 Token 的 K/V 表示   追加                   Query-Key   KV 随上下文增长
  Attention /                                                 相似度      
  KV                                                                      

  SWA / Local 最近 W 个 Token 的 K/V   滑动追加、淘汰         局部        窗口外不能直接访问
  KV                                                          Attention   

  SSM State   固定结构的递归隐藏状态   输入依赖状态转移       状态输出    有损压缩，精确恢复能力有限

  B'MOJO      被选中的历史 Token 表示  Innovation Selection   通过        容量和检索预算受限
  Eidetic                                                     Attention   
  Store                                                       读取        

  Titans      动态神经网络参数         梯度、动量、遗忘更新   MLP         可能干扰、遗忘、域偏移和更新成本
  Neural                                                      前向查询    
  Memory                                                                  
  ----------------------------------------------------------------------------------------------------------

## 2.2 不允许混淆的状态

实现必须将以下对象分开命名和保存：

-   `model_parameters`：训练得到的静态模型权重；
-   `mamba_state`：SSM 的递归状态；
-   `kv_cache`：局部 Attention 的 Key/Value；
-   `titans_fast_weights`：当前 Cell 的动态 Neural Memory 参数；
-   `titans_momentum`：Titans 更新所需的动量状态；
-   `eidetic_store`：B'MOJO-inspired 选择性 Token 存储；
-   `optimizer_state`：静态模型训练优化器状态；
-   `session_metadata`：Session、用户、任务和版本信息。

这些对象的生命周期、持久化和资源统计必须分别实现。

## 2.3 复杂度与资源口径

不应只报告静态参数量。每个模型必须同时报告：

1.  静态模型参数量；
2.  每个并发 Stream 的动态状态字节数；
3.  每个 Titans Cell 的 Fast Weights 和 Momentum 字节数；
4.  KV Cache 字节数；
5.  Eidetic Store 字节数；
6.  Checkpoint 大小；
7.  单 Token 和单 Chunk 的读取、更新延迟；
8.  峰值训练显存和推理显存。

SWA 的局部窗口可限制 KV Cache；Mamba
的状态通常随序列长度保持固定结构；Titans 的 Fast Weights 不随 Token
数线性增长，但其参数副本、优化器状态和更新计算必须计入；B'MOJO-inspired
Store 在本 MVP 中采用固定 Top-K。

------------------------------------------------------------------------

# 3. 研究假设与可证伪条件

## H1 --- Local Interaction Hypothesis

**假设：** 在 10M 规模下，SWA 比纯 Mamba 更擅长需要局部精确 Token
交互的任务。

**验证：** 比较 SWA-only、Mamba-only、Samba-lite
在局部语言建模、Induction Head、Selective Copy
和短距离关联任务上的表现。

**否证条件：** 在相同静态参数和可比训练预算下，SWA
对目标任务没有稳定增益，或其增益不足以抵消 KV 资源成本。

## H2 --- SSM Temporal State Hypothesis

**假设：** Mamba State 能以较低、近似固定的运行时状态成本提供跨 Chunk
的连续历史压缩。

**验证：** 比较 Mamba-only 与 SWA-only
在超出局部窗口的任务中的准确率、状态字节数和吞吐。

**否证条件：** Mamba State 在目标任务上没有可测量的跨 Chunk
增益，或其状态成本和质量劣势不具竞争力。

## H3 --- Titans Parametric Memory Hypothesis

**假设：** Titans Neural Memory 能够通过在线梯度更新学习历史 Key-Value
关联，在跨 Chunk 或跨 Session 任务中提供超出 SWA 的增量价值。

**验证：** 比较 SWA-only 与 Titans MAG；清除 KV、局部上下文和 SSM State
后，仅加载 Titans Cell，测试知识恢复。

**否证条件：** Titans
只在保留历史上下文时有效；清空上下文后不能恢复；或更新成本、遗忘和干扰超过其质量收益。

## H4 --- SSM + Titans Complementarity Hypothesis

**假设：** Mamba State 与 Titans Neural Memory
表示不同类型的历史信息，组合后优于任意单独机制。

**验证：** 比较 Samba-lite、Titans MAG 和 Mamba + Titans + SWA
三路模型。

**否证条件：** 三路模型没有稳定超过最强两路模型的质量---资源 Pareto
前沿，或其中一个分支可以删除而不降低目标任务性能。

## H5 --- Innovation Storage Hypothesis

**假设：** 对 SSM
难以预测的信息进行选择性精确保留，可以改善延迟精确召回。

**验证：** 比较 B'MOJO-F-inspired 与 B'MOJO-Innovation-inspired。

**否证条件：** 固定预算下 Innovation Store 不提高 Exact
Match，或收益完全来自增加的动态存储容量。

## H6 --- Cell Persistence and Fork Hypothesis

**假设：** 相同 Backbone 和相同 Memory Interface 下，Titans Cell
可以保存、恢复和 Fork，且分支可以独立演化。

**验证：** Checkpoint/Restore 和双 Fork 隔离测试。

**否证条件：** 无法在相同静态权重下恢复近似一致的读取结果，或修改一个
Fork 会影响另一个 Fork。

------------------------------------------------------------------------

# 4. MVP 范围和明确不做的事

## 4.1 本轮必须完成

1.  统一 Tokenizer、训练数据和评估协议；
2.  10M 静态参数预算和自动参数统计；
3.  Full Attention、SWA、Mamba、Samba-lite 基线；
4.  Titans Neural Memory 的参考实现；
5.  Titans MAG-lite；
6.  Mamba + SWA + Titans 的三路候选架构；
7.  B'MOJO-inspired Innovation Store；
8.  合成关联记忆、跨 Chunk、跨 Session 和持续学习测试；
9.  独立的 Runtime State / Memory Cell Checkpoint；
10. 完整 Run Manifest 和可复现实验记录。

## 4.2 本轮明确不做

-   不训练 100M--1B 以上模型；
-   不追求生产级 LLM 助手能力；
-   不实现 Teacher Agent；
-   不实现分布式 Cell Network；
-   不实现跨 Backbone 通用 Cell 迁移；
-   不实现多用户权限市场；
-   不实现完整 TNT 两阶段训练优化；
-   不实现复杂 Memory Router；
-   不同时引入多层级 Titans Cell；
-   不把 Mamba-3 作为第一轮必选项；
-   不把 Gated Attention 与 MAG Gate 同时默认开启；
-   不宣称完整复现 B'MOJO，除非按原论文实现其 State
    Realization、Innovation Selection 和读取路径。

------------------------------------------------------------------------

# 5. 实验矩阵

## 5.1 第一轮核心矩阵

  ----------------------------------------------------------------------------
  ID              模型                        机制             优先级
  --------------- --------------------------- ---------------- ---------------
  E0              Full Attention              全局因果         必须
                                              Attention        

  E1              SWA                         局部 Attention   必须

  E2              Mamba-only                  递归 SSM State   必须

  E3              Samba-lite                  Mamba + SWA      必须
                                              层级组合         

  E4              Titans MAG-lite             SWA + Titans     必须
                                              Memory + MAG     
                                              Gate             

  E5              Mamba-Titans-SWA            Mamba State +    必须
                                              Titans + SWA     
                                              三路融合         

  E6              BMOJO-F-inspired            SSM State +      第二阶段
                                              SWA，Attention   
                                              可读取 State     
                                              Summary          

  E7              BMOJO-Innovation-inspired   E6 + 固定预算    第二阶段
                                              Innovation Store 

  E8              Gated-SWA                   SWA +            独立消融
                                              Head-specific    
                                              SDPA Output Gate 
  ----------------------------------------------------------------------------

第一阶段只实现 E0--E5。E6--E8 必须等核心训练和评估管线稳定后再加入。

## 5.2 模型定义

### E0 --- Full Attention

标准 Decoder-only Transformer，因果全局 Attention。

用途：短上下文质量上界和 KV 成本基线。

### E1 --- SWA

与 E0 使用相同 Transformer Block，区别仅为 Attention Mask 限制到最近 W
个 Token。

用途：分离"全局可见范围"与"局部计算"的影响。

### E2 --- Mamba-only

使用 Samba 官方仓库对应的 Mamba 实现和固定版本。

用途：纯递归状态基线。

### E3 --- Samba-lite

在小模型中按层交替组合 Mamba 和 SWA，并保留 MLP / Norm / Residual。

**注意：** 这是 10M 规模的 Samba-inspired
implementation，不是原论文模型复现。架构层级比例、宽度和深度需在参数预算下确定，并在配置中显式记录。

### E4 --- Titans MAG-lite

使用一个独立 Titans Neural Memory 模块与 SWA 分支并行，再用 MAG Gate
融合。

**MVP 限定：** 初始版本允许只设置一个 Memory-enabled block 或一个共享
Memory 模块，以降低实现成本。必须标注为
`mag_lite_single_memory`，不得称为完整 Titans MAG 复现。后续可扩展为
per-layer Memory。

### E5 --- Mamba-Titans-SWA

三条并行路径：

``` text
                         ┌──> SWA branch ───────────┐
input hidden h_t ────────┼──> Mamba branch ─────────┼──> 3-way gated fusion
                         └──> Titans Memory branch ┘
```

-   SWA：当前局部 Token 交互；
-   Mamba：递归时间状态；
-   Titans：在线更新的参数化关联记忆；
-   Fusion：融合三个分支。

这是 MiniCells 的新架构假设，不是 Samba、B'MOJO 或 Titans
论文中已经验证的模型。

Fusion 的第一版采用可学习三路 Gate：

\[ \[g_A,g_S,g_M\] =
`\operatorname{softmax}`{=tex}(W_g\[h_t;a_t;s_t;m_t\]) \]

\[ z_t = g_A`\odot`{=tex}a_t + g_S`\odot`{=tex}P_S(s_t) +
g_M`\odot`{=tex}P_M(m_t) \]

其中 (a_t,s_t,m_t) 分别是 SWA、SSM、Titans 输出，(P_S,P_M)
是必要的维度投影。

随后进行 Residual + Norm + MLP。实现必须保证各分支输出维度一致，并记录
Gate 的输入、输出和均值分布。

**消融：** 关闭 SSM、关闭 Titans、关闭
SWA，分别退化为对应两路或单路配置。

### E6 --- BMOJO-F-inspired

使用 SSM 递归状态和 SWA，并让 Attention 能访问固定数量的 State Summary
Tokens。

它不是原论文 B'MOJO-F 的严格复现，除非实现了论文中的状态空间 realization
和相应的 interleaved attention 路径。

### E7 --- BMOJO-Innovation-inspired

在 E6 基础上加入 Innovation Selection：

1.  使用 SSM 输出 (y_t)；
2.  使用固定预测器估计 (`\hat`{=tex}y_t)；
3.  计算 innovation score；
4.  当分数足够高时，将对应输入表示写入固定容量 Store；
5.  Store 满时淘汰最低分条目；
6.  Attention 同时访问近期窗口、State Summary 和 Store 中的 Token。

原论文 Store 可增长；本项目固定容量是工程适配，必须标注。

------------------------------------------------------------------------

# 6. 10M 参数预算和模型配置

## 6.1 预算定义

本项目的"10M 参数"指**静态可训练参数**：

包括：

-   Token Embedding / LM Head；
-   Backbone 所有静态权重；
-   Attention、Mamba、MLP、Norm；
-   Titans Memory 的可训练初始化参数；
-   Key / Value / Query Projection；
-   Gate 和输出 Projection；
-   B'MOJO-inspired 的可训练部分（如果有）。

不计入静态参数、但必须单独报告：

-   Titans Fast Weights 的动态副本；
-   Titans Momentum；
-   Mamba Runtime State；
-   KV Cache；
-   Eidetic Store；
-   训练 Optimizer State；
-   临时 Activation 和 Gradient。

**不得把动态 Titans Memory 参数排除在资源统计之外。**

## 6.2 参数目标

-   目标静态参数：10,000,000；
-   第一轮允许范围：9,700,000--10,300,000；
-   最终对照组尽量控制在 ±1%；
-   若某种架构无法达到该预算，必须报告实际参数量，并用资源归一化曲线比较，不得虚报。

## 6.3 初始配置候选

建议先使用：

-   `d_model = 256`；
-   `vocab_size = 8192`；
-   输入输出 Embedding 权重共享；
-   RoPE 或对应 Mamba 位置机制；
-   RMSNorm；
-   SwiGLU / 与模型架构匹配的 MLP；
-   无额外可学习绝对位置 Embedding，除非模型实现确实需要。

这些是候选值，不是固定最优配置。Tokenizer
训练完成后，词表大小必须固定，所有实验共用同一个 Tokenizer。

## 6.4 自动参数预算脚本

Codex 必须实现：

``` bash
python scripts/count_parameters.py --config configs/model_x.yaml
python scripts/search_config.py --target-params 10000000 --tolerance 0.03
```

`search_config.py` 应搜索：

-   d_model；
-   block count；
-   MLP expansion；
-   Mamba expansion / state size；
-   Attention head count；
-   Titans memory depth / hidden expansion；
-   Memory-enabled block count。

输出每个候选模型的：

-   总静态参数；
-   分模块参数；
-   Embedding 参数；
-   Titans 静态初始化参数；
-   每个 Cell 动态 Fast Weight 参数量；
-   预计运行时状态大小。

脚本必须通过真实实例化模型统计参数，不得只依赖公式估算。

## 6.5 两种公平性报告

必须同时提供：

**Iso-parameter：** 静态参数量相近。

**Iso-resource：** 在相同 Token Budget、训练时间或推理显存下比较质量。

只满足参数量相近，不代表 FLOPs、显存和运行成本相近。

------------------------------------------------------------------------

# 7. 模块接口和数据契约

以下为逻辑接口。Codex
可以根据现有代码调整语言或具体类名，但不得改变语义。

## 7.1 Backbone Interface

``` python
class Backbone:
    def init_runtime_state(self, batch_size: int):
        ...

    def forward_chunk(
        self,
        input_ids,
        runtime_state,
        memory_state=None,
        use_cache=True,
    ):
        # Returns:
        # hidden_states, logits, next_runtime_state, optional_aux
        ...
```

Backbone 不应直接管理 Cell 文件、用户权限或 Library。

## 7.2 RuntimeState

``` python
@dataclass
class RuntimeState:
    model_revision: str
    sequence_position: int
    mamba_state: object | None
    kv_cache: object | None
    local_window_buffer: object | None
```

必须支持初始化、跨 Chunk 继续、显式 reset、保存和恢复，并与 Titans Cell
独立管理。

## 7.3 TitansMemoryState

``` python
@dataclass
class TitansMemoryState:
    architecture_id: str
    memory_weights: object
    momentum: object
    update_step: int
    cell_id: str
    revision: str
```

必须区分：

-   `meta_parameters`：静态模型参数；
-   `fast_weights`：当前 Cell 的动态权重；
-   `momentum`：动态更新状态；
-   `read_only`：是否允许更新。

## 7.4 MemoryCell Interface

``` python
class MemoryCell:
    def create(self, config): ...
    def load(self, checkpoint): ...
    def recall(self, query, state): ...
    def learn(self, keys, values, state): ...
    def checkpoint(self, state): ...
    def restore(self, checkpoint): ...
    def fork(self, checkpoint): ...
```

第一阶段只要求单 Cell 和两个 Fork，不实现权限市场。

## 7.5 EideticStore Interface

``` python
class EideticStore:
    def insert(self, token_repr, score, position, metadata): ...
    def retrieve(self, query, top_k): ...
    def evict_lowest_score(self): ...
    def checkpoint(self): ...
    def restore(self, checkpoint): ...
```

固定容量必须在配置中声明。Store 不允许静默无限增长。

## 7.6 Fusion Interface

``` python
class MemoryFusion:
    def forward(
        self,
        local_attention_output,
        ssm_output=None,
        neural_memory_output=None,
    ):
        ...
```

Gate
必须输出可记录的权重。测试中需要验证关闭某分支后结果符合对应消融配置。

------------------------------------------------------------------------

# 8. Titans Neural Memory 的实现要求

## 8.1 内层学习目标

按照 Titans 的 Key-Value Associative Memory 目标：

\[ k_t = W_K h_t,`\quad`{=tex} v_t = W_V h_t,`\quad`{=tex} q_t = W_Q h_t
\]

\[ `\mathcal`{=tex}L\_{`\mathrm{mem}`{=tex},t} =
\|M\_{`\theta`{=tex}\_t}(k_t)-v_t\|\_2\^2 \]

读取：

\[ r_t=M\_{`\theta`{=tex}\_t}(q_t) \]

Memory 网络采用论文中的深层 MLP 思路。第一版优先使用 2-layer
MLP、GELU、Residual 和 LayerNorm。所有维度、初始化和更新超参数写入
config。

## 8.2 更新机制

实现以下组件：

-   momentary surprise；
-   past surprise / momentum；
-   adaptive forgetting / weight decay；
-   per-channel 或按论文实现的学习率、动量和遗忘 Gate；
-   chunkwise update；
-   Memory Checkpoint。

不要用普通 AdamW 替代 Titans 内层更新。AdamW 是静态 Backbone
外层训练优化器；Titans Fast Weight 更新应遵循论文机制。

## 8.3 训练模式

至少实现三种模式：

1.  `meta_train`：更新静态 Backbone 和 Memory
    初始化参数，同时通过内层更新学习记忆策略；
2.  `online_adapt`：静态模型权重冻结，只更新当前 Cell Fast Weights；
3.  `read_only`：不更新 Fast Weights，只执行 Recall。

需要明确 `model.train()/eval()` 与 Memory 是否更新是两个独立开关。

## 8.4 更新顺序和因果性

自回归任务必须避免未来信息泄漏。

建议采用清晰的读写顺序：

1.  使用当前可见输入计算 hidden；
2.  使用此前 Memory State 进行当前读取；
3.  计算当前输出 / next-token logits；
4.  当前输入对应的 Key-Value 对进入 Memory 更新；
5.  将更新后的 Memory State 用于后续位置。

如果为了复现论文而使用不同的同 Chunk 更新顺序，必须写入配置并通过严格的
causal leakage 测试。

## 8.5 Chunkwise 更新

-   初始 `memory_update_chunk_size = 16`；
-   第二档 `32`；
-   先实现可验证的参考路径；
-   再优化批量矩阵计算；
-   每个 Chunk 结束必须有明确的 Memory State；
-   训练和推理 Chunk Size 必须一致或进行专门的 mismatch 实验。

Titans 论文和 TNT 都指出 Chunk Size 对训练效率和推理质量有影响。不要把
Chunk Size 当成无关实现参数。

## 8.6 梯度正确性

先实现小规模 reference path：

-   对内层 Memory Loss 求梯度；
-   更新 Fast Weights；
-   验证外层 Loss 对静态初始化参数和投影参数存在合理梯度；
-   对照有限差分或简化解析例子；
-   检查梯度非 NaN、非 Inf。

如果采用截断梯度或一阶近似，必须在 Run Manifest
中明确标记，不能声称是完整二阶元学习。

------------------------------------------------------------------------

# 9. B'MOJO-inspired Innovation Store 实现要求

此模块属于第二阶段。

## 9.1 Innovation Score

根据 SSM fading output 计算：

\[ `\epsilon`{=tex}\_t =
`\operatorname{Error}`{=tex}(`\hat`{=tex}y_t,y_t) \]

第一版使用固定、无可学习参数的 causal predictor。可采用论文中描述的
running weighted average / causal convolution 形式。

实现应将 predictor 类型、历史跨度和误差度量写入配置。

## 9.2 固定容量策略

\[MVP-CHOICE\]

-   初始容量：64 Token；
-   第二档容量：128 Token；
-   每条记录保存 Token 表示、位置、Innovation Score；
-   未满时直接插入；
-   满时，仅当新 Score 高于当前最低 Score 时替换最低项；
-   Store 不得超过容量；
-   训练和评估均记录 Store 命中率、替换率和平均年龄。

这是一种受限资源下的 B'MOJO-inspired
adaptation，不等于原论文允许增长至硬件上限的完整 Store。

## 9.3 读取

Attention 读取：

-   当前 Local Window；
-   SSM State Summary；
-   Eidetic Store 中的候选 Token。

对比实验必须包含：

-   无 Store；
-   随机 Store；
-   FIFO Store；
-   Innovation Top-K Store。

这样才能确认收益来自 Innovation Selection，而不是单纯多保存了一些
Token。

------------------------------------------------------------------------

# 10. 数据集与任务

## 10.1 任务优先级

**第一优先级：合成记忆任务。**\
它们能够精确控制历史信息、干扰、延迟和更新规则，适合 10M 模型机制研究。

**第二优先级：自然语言建模。**\
用于检查记忆模块是否破坏基础语言建模能力，不作为唯一成功标准。

## 10.2 合成任务

### A. Multi-Query Associative Recall（MQAR）

生成若干 Key-Value 对：

``` text
K1 -> V1
K2 -> V2
...
Kn -> Vn
```

中间插入随机干扰 Token，最后给出一个或多个 Query Key，要求输出对应
Value。

控制变量：

-   Key-Value 数量：8 / 16 / 32 / 64；
-   序列长度：256 / 512 / 2K / 8K / 16K；
-   Query 数量：1 / 8 / 32；
-   干扰比例；
-   Key 相似度；
-   Value 是否重复。

训练和测试使用不同随机种子，避免记住固定映射。

### B. Delayed Retrieval

在较早位置写入事实，经过可控数量的干扰 Token 后提问。

长度分档：小于窗口、等于窗口、2 倍窗口、8 倍窗口、16 倍窗口。

### C. Selective Copy / Induction

测试局部模式识别、重复结构、复制和延迟模式匹配。

### D. Update and Correction

示例：

``` text
Session 1: Alice -> London
Session 2: Alice -> Paris
Query: Alice lives where?
Expected: Paris
```

额外测试：

-   旧事实是否被错误保留；
-   无关事实是否导致覆盖；
-   多次更新后的最终版本；
-   旧版本是否仍可通过时间条件查询。

### E. Cross-session Recall

流程：

1.  Session A 输入知识；
2.  允许 Titans Cell 在线更新；
3.  保存 Titans Cell；
4.  完全清除 KV Cache；
5.  清除 Local Window；
6.  清除 Mamba State（另设 Carry State 对照）；
7.  重新创建模型 Runtime；
8.  只加载静态模型参数和持久 Cell；
9.  Session B 提问；
10. 测量 Exact Match 和延迟。

必须分别报告：

-   仅 Titans 持久；
-   仅 Mamba State 持久；
-   Mamba State + Titans 持久；
-   两者均不持久。

这组实验用于判断两种长期状态是否提供不同价值。

## 10.3 自然语言数据

建议使用公开、可复现的小型语料作为语言建模对照。优先考虑
WikiText-103，并固定数据版本、Tokenizer 和预处理脚本。

Tokenizer 建议：

-   SentencePiece BPE；
-   初始词表 8,192；
-   只使用训练集训练 Tokenizer；
-   保存 Tokenizer 文件及 SHA256；
-   所有模型共享同一 Tokenizer。

如果实际语料或现有仓库已有合适
Tokenizer，应先检查并记录，不得在不同模型间使用不同 Tokenizer。

## 10.4 数据切分和泄漏控制

-   Train / Validation / Test 使用不同生成种子；
-   不允许同一 Key-Value 样本跨集合重复；
-   测试集必须包含训练时未见的 Key、Value 和组合；
-   Cross-session 测试中，Session B 不得重复提供 Session A 的事实；
-   每个 Batch 内不同 Episode 的 Runtime State 和 Titans Memory
    必须隔离。

------------------------------------------------------------------------

# 11. 训练协议

## 11.1 分阶段 Token Budget

  -----------------------------------------------------------------------
  阶段                模型数量           训练预算           随机种子
  -------------- ------------------ ------------------ ------------------
  Smoke Test           E0--E5        0.1M--1M Tokens           1

  Screening            E0--E5           20M Tokens             1

  Confirmation      保留 4--5 个       100M Tokens           至少 3

  Cell              最佳 1--2 个     按 Episode 计量         至少 3
  Persistence       Titans 配置                        
  -----------------------------------------------------------------------

这些是实验预算目标，不是训练时间承诺。资源不足时可降低 Token
Budget，但必须记录实际值。

## 11.2 外层优化器

建议统一使用 AdamW：

-   β1 = 0.9；
-   β2 = 0.95；
-   weight decay = 0.1；
-   gradient clipping = 1.0；
-   cosine decay；
-   warmup 约为总训练步数的 1%--2%。

Learning Rate 先在基础 Transformer / SWA 基线上做小范围
Pilot，再冻结一套共同的主要设置。若不同架构确实需要不同学习率，必须公开调参范围和选择过程。

## 11.3 Titans 内层优化

Titans 的学习率、Momentum、Forget Gate 和 Chunk Size
独立记录。不得将内层更新超参数和外层 AdamW 混为一谈。

## 11.4 精度

-   支持 FP32 reference；
-   GPU 支持时优先 BF16；
-   NVIDIA T4 等不支持 BF16 的设备使用 FP16 AMP + GradScaler；
-   Titans Fast Weight 的更新可根据数值稳定性使用 FP32；
-   记录实际精度策略。

## 11.5 Chunk 和 Window

建议初始值：

-   SWA window：256；
-   备选：128、512；
-   Titans update chunk：16；
-   备选：32；
-   训练序列：2K；
-   长度测试：2K、4K、8K、16K。

这些是 10M MVP 的工程起点，不是原论文最优参数。

------------------------------------------------------------------------

# 12. 评估指标

## 12.1 任务质量

  指标                        用途
  --------------------------- ------------------------
  Validation PPL              基础语言建模
  Exact Match                 精确 Key-Value 召回
  Recall@K                    记忆检索候选覆盖
  Delayed Recall Accuracy     延迟回忆
  Cross-session Exact Match   持久知识恢复
  Correction Accuracy         新事实覆盖旧事实
  Retention Accuracy          持续学习后旧知识保留
  Interference Score          无关学习对旧知识的影响
  Induction / Copy Accuracy   局部序列能力

## 12.2 资源成本

必须记录：

-   静态参数量；
-   Tokens/sec；
-   Training wall-clock；
-   Peak GPU Memory；
-   Inference latency；
-   Per-token latency；
-   Per-chunk update latency；
-   Titans Fast Weight 字节数；
-   Momentum 字节数；
-   Mamba State 字节数；
-   KV Cache 字节数；
-   Eidetic Store 字节数；
-   Checkpoint size；
-   Save / Restore latency。

## 12.3 质量---成本曲线

不得只比较单一准确率。至少提供：

1.  Exact Match vs Total Runtime Memory Bytes；
2.  Exact Match vs Latency；
3.  PPL vs Training FLOPs 或训练时间；
4.  Cross-session Recall vs Titans Cell Size；
5.  Retention vs Update Count。

如果一个架构质量更高，但需要更多动态参数、更新 FLOPs
或延迟，必须明确报告。

------------------------------------------------------------------------

# 13. Checkpoint、Restore 和 Fork 规格

## 13.1 文件结构

建议使用以下布局：

``` text
artifacts/
  runs/
    <run_id>/
      config.yaml
      manifest.json
      model_static.pt
      tokenizer.model
      metrics.jsonl
      eval/
      checkpoints/
        step_<n>/
          model_static.pt
          optimizer.pt
          runtime_state.pt
          titans_cell.pt
          eidetic_store.pt
          metadata.json
```

实际项目可根据现有仓库调整，但必须保持文件职责清楚。

## 13.2 Cell Manifest

每个 Titans Cell 至少记录：

``` json
{
  "cell_id": "cell-...",
  "revision": 1,
  "parent_revision": null,
  "architecture_id": "titans-mag-lite-v1",
  "backbone_id": "...",
  "backbone_revision": "...",
  "tokenizer_sha256": "...",
  "memory_interface_version": 1,
  "memory_weights_file": "fast_weights.pt",
  "momentum_file": "momentum.pt",
  "update_step": 0,
  "mode": "trainable",
  "created_at": "...",
  "provenance": []
}
```

Codex 不需要实现用户权限系统，但 Manifest 必须预留 provenance、revision
和 compatibility 字段。

## 13.3 Checkpoint 语义

-   `model_static.pt`：静态模型参数；
-   `runtime_state.pt`：Mamba State、KV Cache、Local Window；
-   `titans_cell.pt`：Fast Weights、Momentum、Update Step；
-   `eidetic_store.pt`：选中 Token、Score、Position；
-   `optimizer.pt`：外层训练优化器。

不得把 Runtime State 和 Cell 权重合并成一个无法独立管理的 Blob。

## 13.4 Fork

Fork 操作必须：

1.  从同一 Checkpoint 复制 Fast Weights 和 Momentum；
2.  生成新的 Cell ID；
3.  保留 Parent Revision；
4.  之后更新完全独立；
5.  修改 Fork A 不得改变 Fork B；
6.  可分别保存和恢复。

第一版不实现 Merge。两个 Fast Weight
参数不允许直接平均后声称完成知识合并。

------------------------------------------------------------------------

# 14. 正确性和验收测试

## 14.1 基础模型测试

-   [ ] 所有模型输入输出 Shape 正确；
-   [ ] 因果 Mask 正确；
-   [ ] 修改未来 Token 不得改变过去位置 logits；
-   [ ] 训练 Loss 有限；
-   [ ] 梯度无 NaN / Inf；
-   [ ] 参数统计脚本与 `sum(p.numel())` 一致；
-   [ ] 所有核心模型静态参数处于目标范围。

## 14.2 State 生命周期测试

-   [ ] Mamba State 可跨 Chunk 延续；
-   [ ] Mamba State Reset 后与全新初始化一致；
-   [ ] KV Cache 可保存和恢复；
-   [ ] KV Cache 不跨独立 Episode 泄漏；
-   [ ] Runtime State 与 Titans Cell 可独立保存。

## 14.3 Titans 测试

-   [ ] Memory Fast Weights 会随 Learn 更新；
-   [ ] Read-only 模式不更新；
-   [ ] Momentum 正确保存；
-   [ ] Forget Gate 处于合法数值范围；
-   [ ] Checkpoint / Restore 后读取结果一致或在数值容差内；
-   [ ] Memory 更新只使用允许的历史输入；
-   [ ] 清空 KV 和 Mamba State 后，持久 Titans Cell 仍能恢复已学习知识。

## 14.4 B'MOJO-inspired 测试

-   [ ] Innovation Score 可复现；
-   [ ] Store 未满时插入；
-   [ ] Store 满时按规则淘汰；
-   [ ] Store 永不超过容量；
-   [ ] Random / FIFO / Innovation Top-K 对照配置可运行；
-   [ ] Store Checkpoint / Restore 后记录一致。

## 14.5 Fork 隔离测试

-   [ ] A、B 从同一 Checkpoint Fork；
-   [ ] A 学习新事实；
-   [ ] B 不学习；
-   [ ] A 的更新不改变 B；
-   [ ] B 的读取结果保持在容差内；
-   [ ] 两者均能独立保存和恢复。

## 14.6 端到端 Cross-session 测试

-   [ ] Session A 输入知识；
-   [ ] 保存 Titans Cell；
-   [ ] 清除 KV；
-   [ ] 清除 Local Window；
-   [ ] 清除 Mamba State；
-   [ ] 重建 Runtime；
-   [ ] 加载 Titans Cell；
-   [ ] Session B 进行 Recall；
-   [ ] 记录 Exact Match、延迟和资源使用。

这是本项目最重要的验收路径之一。

------------------------------------------------------------------------

# 15. Codex 实施里程碑

## M0 --- Repository and Environment Audit

Codex 首先检查现有 MiniCells 仓库：

-   当前训练框架；
-   已有模型实现；
-   数据管线；
-   Checkpoint 格式；
-   现有 Cell / State 代码；
-   GPU 和 CUDA 环境；
-   依赖版本；
-   测试与 CI。

**不得在审计前重写现有训练框架。**

交付：

-   `docs/EPCLM-004-IMPLEMENTATION-PLAN.md`
-   `configs/environment.yaml`
-   当前架构和复用组件报告。

验收：所有新依赖、版本、GPU 能力和需要修改的既有模块均有记录。

## M1 --- Experiment Harness

实现：

-   统一 Tokenizer 接口；
-   合成 MQAR Generator；
-   WikiText-103 数据加载；
-   统一 Training Loop；
-   Evaluation Harness；
-   Parameter Counter；
-   Run Manifest；
-   Metrics JSONL。

验收：

-   E0 至少完成一次短训练；
-   数据集切分可复现；
-   参数统计准确；
-   相同配置和 Seed 可以重复运行。

## M2 --- Baseline Architectures

按顺序实现：

1.  E0 Full Attention；
2.  E1 SWA；
3.  E2 Mamba-only；
4.  E3 Samba-lite。

优先复用官方 Samba 的模型代码和依赖，固定
commit。若环境不支持，先提供正确性 Reference
Implementation，不要先写优化 Kernel。

验收：四个模型通过基础测试和相同合成任务。

## M3 --- Titans MAG-lite

实现：

-   Neural Memory MLP；
-   Key / Value / Query Projection；
-   Inner-loop update；
-   Momentum；
-   Adaptive Forgetting；
-   MAG Gate；
-   Memory Checkpoint / Restore；
-   Read-only / Online Adaptation 两种模式。

先通过小张量梯度测试，再运行完整训练。

验收：E4 可训练，且跨 Session 只加载 Cell 时能够执行 Recall。

## M4 --- Mamba + Titans

实现 E5 三路分支：

-   SWA；
-   Mamba；
-   Titans；
-   3-way Fusion。

必须实现关闭单一分支的消融配置。

验收：E5 与 E3、E4 在同一 Evaluation Harness 中比较。

## M5 --- B'MOJO-inspired Store

实现 E6 / E7：

-   State Summary；
-   Innovation Predictor；
-   Top-K Eidetic Store；
-   Store Attention Read；
-   FIFO / Random / Innovation 对照。

验收：容量、淘汰、读取、Checkpoint 均通过测试。

## M6 --- Formal Screening and Confirmation

-   先完成 20M Token Screening；
-   选择 4--5 个候选；
-   对最终候选完成 100M Token 和至少 3 个 Seed；
-   输出质量---成本曲线；
-   不得只提交最佳 Seed。

验收：结果文件可由 Run Manifest 追溯。

## M7 --- Cell Persistence and Fork

实现双 Fork 独立演化和跨 Session 恢复。

验收：第 14 节所有 Cell 测试通过。

------------------------------------------------------------------------

# 16. Codex 执行纪律

1.  每个里程碑独立提交，不要一次性实现所有架构。
2.  先写测试，再写模块。
3.  每次变更运行相关单元测试。
4.  不得用随机输出或静态 Mock 伪装 Recall 成功。
5.  不得把模型权重记忆误当成当前 Session 的 Titans Fast Weight 学习。
6.  不得把 Mamba State 和 Titans Cell 混存。
7.  不得将 B'MOJO-inspired 实现标记为原论文完整复现。
8.  不得把 MAG Gate 与 Gated Attention 合并为一个不可消融的 Gate。
9.  所有实验必须记录 Commit SHA、Config Hash、Tokenizer
    Hash、Seed、Token Budget 和硬件信息。
10. 任何论文机制偏离都必须记录在 `implementation_deviations.md`。
11. 若现有仓库已经有可复用组件，优先复用并补测试，不得为了符合本文件而无理由重写。
12. 若遇到资源限制，先缩小 Batch / Token
    Budget，不得悄悄降低参数量或改变模型定义。

------------------------------------------------------------------------

# 17. 风险登记与结果解释

  ------------------------------------------------------------------------------------------------
  风险                   表现                       处理方式
  ---------------------- -------------------------- ----------------------------------------------
  10M 容量不足           所有模型 Recall 都低       先检查任务难度、Tokenizer
                                                    和训练收敛；不要立即否定机制

  Titans 更新不稳定      Loss 爆炸、Fast Weights    FP32 Fast Weights、梯度裁剪、降低内层学习率
                         发散                       

  Titans 有记忆但不会用  Cell 可恢复，模型不利用    检查 Query / Key / Value 表示对齐和 Gate
                         Recall                     

  SSM 与 Titans 冗余     E5 不超过 E3 或 E4         分析两路表示相似度和 Gate
                                                    权重，考虑删除一条路径

  Innovation Store       Top-K 与 Random / FIFO     检查创新度预测器、容量和任务是否真正需要精确
  无增益                 相近                       Token

  PPL 改善但 Recall 变差 语言建模指标掩盖记忆缺陷   以目标任务指标单独判定，不用 PPL 替代 Recall

  Cell 跨 Session 失败   清除 Runtime 后知识丢失    检查 Fast Weights 是否真的被保存，而非仅保存
                                                    meta-parameters

  Fork 污染              一个分支修改另一个分支     检查 Tensor aliasing、浅拷贝和共享 Optimizer
                                                    State

  参数公平但计算不公平   模型 Token/s 差异很大      同时报告 Iso-parameter 与 Iso-resource

  小模型无法稳定收敛     Seed 间方差大              增加训练预算和 Seeds，降低结论强度
  ------------------------------------------------------------------------------------------------

## 17.1 结果决策规则

-   若 E4 优于 E1，且跨 Session Recall 成立：继续研究 Titans Cell。
-   若 E3 优于 E1，且资源成本可接受：保留 SSM + SWA 路线。
-   若 E5 稳定优于 E3 和 E4 的 Pareto 前沿：支持 SSM + Titans 互补假设。
-   若 E5 不优于最强两路模型：不要保留三路复杂度。
-   若 E7 优于 E6 且优于 Random / FIFO：支持 Innovation Selection
    的增量价值。
-   若 E4 / E5 能跨 Session 保存并 Fork：进入 Cell Portability 的同
    Backbone 级别研究。
-   若所有模型都不能在跨 Session 任务上成功：先检查任务构造、Memory
    更新和读写因果性，不得直接声称 Neural Memory 不可行。

------------------------------------------------------------------------

# 18. 最终交付清单

本轮实验结束后，必须具备：

-   [ ] 统一实验代码；
-   [ ] 固定 Tokenizer；
-   [ ] 固定训练 / 验证 / 测试数据版本；
-   [ ] E0--E5 模型配置；
-   [ ] E6--E8 后续配置或明确延期记录；
-   [ ] 参数统计报告；
-   [ ] 静态参数与动态状态资源报告；
-   [ ] 20M Screening 结果；
-   [ ] 100M Confirmation 结果（若资源允许）；
-   [ ] 至少 3 Seed 的最终结果；
-   [ ] Cross-session Recall 结果；
-   [ ] Titans Checkpoint / Restore 测试；
-   [ ] Fork 隔离测试；
-   [ ] 所有失败和偏离记录；
-   [ ] 最终架构决策记录（ADR）。

------------------------------------------------------------------------

# 19. 参考文献与实现资源

## 核心论文

1.  Zancato et al. **B'MOJO: Hybrid State Space Realizations of
    Foundation Models with Eidetic and Fading Memory.** NeurIPS 2024.\
    https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb6281d3f2cfe2eb8b1307227d3677b3-Abstract-Conference.html\
    PDF:
    https://proceedings.neurips.cc/paper_files/paper/2024/file/eb6281d3f2cfe2eb8b1307227d3677b3-Paper-Conference.pdf

2.  Behrouz, Zhong, Mirrokni. **Titans: Learning to Memorize at Test
    Time.** NeurIPS 2025.\
    https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html\
    PDF:
    https://papers.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf

3.  Ren et al. **Samba: Simple Hybrid State Space Models for Efficient
    Unlimited Context Language Modeling.** ICLR 2025.\
    https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html\
    Official code: https://github.com/microsoft/Samba

## 辅助论文

4.  Li et al. **TNT: Improving Chunkwise Training for Test-Time
    Memorization.** ICLR 2026.\
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html\
    PDF:
    https://openreview.net/pdf/424813e8424a5a7c3af990a764d389ec5b389490.pdf

5.  Qiu et al. **Gated Attention for Large Language Models:
    Non-linearity, Sparsity, and Attention-Sink-Free.** NeurIPS 2025.\
    https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html\
    Code: https://github.com/qiuzh20/gatedattention

## 实现资源

6.  Mamba official repository: https://github.com/state-spaces/mamba\
7.  Samba official repository: https://github.com/microsoft/Samba

**实现注意：** Titans 和 B'MOJO 的论文链接是机制定义的首要依据。第三方
Titans
代码可以用于交叉检查，但必须记录其与论文的差异，不得把未经核验的第三方实现当作官方复现。

------------------------------------------------------------------------

# 20. 给 Codex 的第一条实施任务

不要让 Codex 一开始就实现 E0--E8 全部模型。

先将本文件交给 Codex，并使用下面的任务说明：

``` text
请阅读 EPCLM-003/004 — Unified Memory Architecture & 10M Experimental Specification。

本次只执行 M0：Repository and Environment Audit。

要求：
1. 检查现有仓库结构、模型实现、训练框架、数据管线、Checkpoint 和测试。
2. 检查 Python、PyTorch、CUDA、GPU、Mamba 相关依赖和可用性。
3. 识别哪些已有模块可以复用，哪些需要新增。
4. 核对官方 Samba 仓库与当前环境的兼容方式，先不要安装或升级大批依赖。
5. 给出 10M 参数预算的初步可行性分析。
6. 输出 implementation plan、environment manifest 和 architecture reuse report。
7. 不要训练模型，不要重写已有架构，不要开始实现 Titans。
8. 完成后报告阻塞项、风险和 M1 的最小实施范围。

所有结论必须有实际文件、命令输出或测试依据。不要推测代码已经存在。
```

M0 完成并审查后，再开始 M1。这样可以避免 Codex 在不了解现有 MiniCells
实现的情况下，另起一套重复的训练框架。

------------------------------------------------------------------------

## 文档结论

本轮研究不预设最终模型一定是 Mamba + Titans + Attention。

我们要验证的是：

-   SWA 是否提供不可替代的短期精确交互；
-   SSM 是否提供有价值的递归时间状态；
-   Titans 是否提供可持续学习的参数化记忆；
-   B'MOJO 式选择性存储是否弥补精确回忆缺口；
-   这些机制是否具有真正的互补性；
-   Titans Memory 能否成为独立持久化、恢复和 Fork 的 Cell。

**最终架构由实验结果决定，不由模块数量决定。**

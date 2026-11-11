# M3C 参考资料与研究边界

本索引区分原始论文、官方实现和相邻研究。引用外部论文不表示 MiniCells 已复现其结果，也不表示相关机制已适配 Mamba-3。

相关理论综述：[EPCLM-002 Persistent State Theory v0.2 中文原文](../../literature/EPCLM-002-Persistent-State-Theory-v0.2.zh-CN.md) / [English translation](../../literature/EPCLM-002-Persistent-State-Theory-v0.2.en.md)。该文扩展讨论 SSM、Titans、局部 Attention 与可移植记忆；其中实验安排为研究提案，不构成 M3C 训练授权。另见[文献与理论索引](../../literature/README.md)。

## A. Mamba 与 State Space Model

### A1. Mamba — Linear-Time Sequence Modeling with Selective State Spaces

- Albert Gu, Tri Dao, 2023.
- Paper: https://arxiv.org/abs/2312.00752
- Code: https://github.com/state-spaces/mamba
- 相关性：Selective SSM、输入依赖的状态更新、硬件感知实现。
- M3C 用途：作为 Mamba 系列的基础背景；M3C 不重新实现 Mamba-1。

### A2. Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality

- Tri Dao, Albert Gu, 2024.
- Paper: https://arxiv.org/abs/2405.21060
- 相关性：Mamba-2 / SSD 的结构化状态空间对偶、Chunked Scan、递归形式与结构化因果 Attention 的联系。
- M3C 用途：理解 State 与 Attention 的数学联系。
- 边界：SSD 的具体公式依赖其参数化条件，不能直接推断标准 Softmax Attention 与有限 State 普遍等价。

### A3. Mamba-3: Improved Sequence Modeling using State Space Principles

- Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu, 2026.
- ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html
- arXiv: https://arxiv.org/abs/2603.15569
- 相关性：M3C 的主干架构依据。重点包括 Exponential-Trapezoidal Discretization、Complex-valued State Update、MIMO，以及推理效率导向的实现。
- M3C 用途：必须以论文与官方代码共同作为实现依据；在 M3C-001 锁定官方代码 Commit。
- 边界：论文中的 1.5B 规模结果不能直接外推为 MiniCells 小模型或 T4 上的结果。

### A4. Official Mamba Implementation

- Repository: https://github.com/state-spaces/mamba
- Mamba-3 module: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
- 相关性：官方 Mamba3 模块、SISO/MIMO 配置、Triton/TileLang Kernel 与 Inference State 管理。
- M3C 用途：优先复用官方模块，不自行重写递归核心。
- 工程注意：官方代码当前 MIMO 路径包含 TileLang Kernel 依赖；必须实际检查目标硬件和依赖版本。文档中的 main 分支 URL 是导航链接，正式实验必须锁定不可变 Commit。

## B. State 与 Attention 的混合/统一读取

### B1. Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

- Liliang Ren et al., ICLR 2025.
- Microsoft Research: https://www.microsoft.com/en-us/research/publication/samba-simple-hybrid-state-space-models-for-efficient-unlimited-context-language-modeling/
- Code: https://github.com/microsoft/Samba
- 相关性：按 Layer 交错 Mamba 与 Sliding Window Attention。
- M3C 用途：作为有界 Workspace Attention 的混合架构参考。
- 边界：Samba 的 Attention 与 Mamba 是层级交错结构，不是 M3C 的平面 Cell 调用。

### B2. Hymba: A Hybrid-head Architecture for Small Language Models

- Xin Dong et al., ICLR 2025.
- Paper: https://arxiv.org/abs/2411.13676
- Official code: https://github.com/NVlabs/hymba
- NVIDIA Research: https://research.nvidia.com/publication/2025-04_hymba-hybrid-head-architecture-small-language-models
- 相关性：在 Hybrid Head 中结合 Attention 与 SSM，分别利用精确回忆和上下文压缩。
- M3C 用途：小模型、KV Cache、局部/全局 Attention 的设计参考。
- 边界：Hymba 的 Hybrid-head 结构与 M3C 的 Mamba Layer + 可选 Cell 分支不同。

### B3. DART: Decoded Attention over Recurrent States for Efficient Long-Context Sequence Modeling

- Yixiao Qian et al., 2026 arXiv preprint.
- Paper: https://arxiv.org/abs/2608.02032
- 相关性：从 Mamba-2 Chunk State Contributions 解码 Key/Value，执行 State-Memory Attention（SMA），以压缩状态支持更细粒度的历史读取。
- M3C 用途：M3C-006 的直接研究参照；帮助设计共享压缩记忆与 Attention 式检索。
- 边界：DART 当前推导基于 Mamba-2 SSD。Mamba-3 的复数状态、Exponential-Trapezoidal Recurrence 和 MIMO 需要重新推导，不能直接移植公式或声称已兼容。

### B4. Titans: Learning to Memorize at Test Time

- Venue: NeurIPS 2025. [Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) · [Full paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)
- Relevance: Online neural memory, surprise momentum, adaptive forgetting, and integration with attention. Section 4.5 and Table 4 report the component ablations.
- MiniCells follow-up: [P4 memory controls, forgetting ablations, and Memory Policy proposal](../p4-peer-titans-collaboration/docs/TITANS-MEMORY-POLICY.md). The proposal records differences between the paper and the frozen M3 Memory implementation.

## C. 动态计算与路由

### C1. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models

- Paper: https://arxiv.org/abs/2404.02258
- 相关性：根据 Token 动态分配 Transformer 层计算量。
- M3C 用途：动态 Skip / Cell Invocation 的强制对照；用于区分“动态调用”本身与“平面 Cell Pool”带来的增益。
- 边界：Mixture-of-Depths 保留 Transformer 层级骨架，不等同于 M3C 的跨 Mamba 层共享 Cell Pool。

### C2. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- Paper: https://arxiv.org/abs/2101.03961
- 相关性：稀疏专家路由、负载均衡与稀疏执行。
- M3C 用途：Router 训练与实际 Dispatch 成本的参考。
- 边界：MoE 的专家通常位于固定层中；M3C 还要决定是否调用 Cell 以及在哪个 Mamba 层调用。

## D. MiniCells 内部历史资料

### D1. CLM Workspace Controller-001 / v0.4

- Spec: https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/clm-workspace-controller-001/SPEC.md
- README: https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/clm-workspace-controller-001/README.md
- Smoke report: https://github.com/ArcheLabs/mini-cells/blob/main/artifacts/experiments/clm-workspace-controller-001/v0.4/RESEARCH-REPORT.zh-CN.md
- 相关性：Shared Workspace、独立 Stateless Cells、Controller State、固定两步和反事实路由诊断。
- 边界：历史 Smoke 不构成正式科学确认；双 T4 2M Pilot 报告为 PILOT_BLOCKED，未开始训练。M3C 不覆盖或重写这些历史状态。

### D2. EPCLM-001 — Incremental and Event-driven Execution

- README: https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/epclm-001-incremental-execution/README.md
- 相关性：增量计算、缓存复用和物理执行成本的验证纪律。
- M3C 用途：参考如何区分逻辑更新、物理更新、实际吞吐与缓存成本。

### D3. EPCLM-002-A0 — Persistent State Recall

- README: https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/README.md
- Screening: https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/A0-v0.2-screening.md
- 相关性：持久 State、精确回忆、Raw Context Removal 与 Flat Recurrent State 对照。
- M3C 用途：后续验证 Mamba State 是否足以支持定义明确的长程任务。
- 边界：该研究是独立的状态回忆机制实验，不能用作 M3C 的结果。

---

## 资料使用规则

1. 优先引用原始论文与官方实现。
2. 对预印本明确标注 arXiv preprint，不把预印本当作已同行评审的定论。
3. 官方代码的 main 分支链接只用于导航；正式实验必须记录 Commit SHA。
4. 外部论文结果只用于提出假设和设计对照，不可写入 MiniCells 实验结果。
5. 若 Mamba-3 官方代码更新，M3C-001 必须记录具体差异；不得静默跟随 main。

# MiniCells Experiment 005 — Learned Cell Connections and Compositional Efficiency

**中文名称：学习型 Cell 连接的组合效率验证**
**版本：设计草案 v0.1（2026-10-12）**
**状态：PROPOSED / NOT REGISTERED / UNRUN**
**仓库：** `ArcheLabs/mini-cells`
**用途：** 用于评审与后续 Codex 实施；本文不是已冻结的科研协议，也不包含新的实验结果。

## 0. 执行摘要

本轮研究的工程目标是未来让**总参数规模为 7B–30B 的预训练 MoE**在手机上以可接受的质量损失和实际推理成本运行。005 **不直接声称**已完成 7B–30B 迁移或移动端验证；它只检验前置的、可被证伪的结构假设：

> **在近似相等的总参数量下，细粒度 Cell 之间学习到的条件连接与顺序函数复合，是否能补偿小 Cell 单体能力的不足，使整个模型以少于传统 MoE 的独立激活权重和真实计算量实现接近的质量？**

必须将以下命题独立判断：

- **B1 — 函数组合价值：** 相同细粒度 Cell、相同执行次数下，顺序复合能否优于并行聚合？
- **B2 — 学习连接价值（主命题）：** 同样允许顺序复合、同样的 Cell 参数和执行预算下，学习得到的稀疏前驱→后继连接，是否优于随机连接以及没有连接限制的状态感知路由？
- **S — 系统价值：** 相对于 8-Expert Top-2 MoE，在可控质量损失下，独立激活权重、全模型 FLOPs **和真实推理延迟**是否同时改善？

005 的主科学目标是 **B2**。B1 是必要机制对照，S 是迁移到大型预训练 MoE 的门槛。**不允许**因 Cell 数增加、拓扑图看起来复杂或局部 PPL 改善就宣告成功。

## 1. 003 / 004 的已知证据与限制

### 1.1 冻结历史结果（seed 42，仅用于产生 005 假设）

| 004 实验臂 | 总参数 | PPL ↓ | GEMM FLOPs/token ↓ | T4 resident-batch 前向延迟 ↓ |
|---|---:|---:|---:|---:|
| `fixed-cell` | 60,894,720 | 12.2870 | 123.171M | 23.15 ms |
| `global-router` | 60,919,296 | 12.3420 | 47.722M | 26.31 ms |
| `soft-cell` | 61,115,904 | 12.1822 | 338.020M | 88.48 ms |
| `hard-cell` | 61,115,904 | 14.8248 | 43.834M | 65.75 ms |
| `graph-cell` | 61,116,336 | 13.0706 | 55.558M | 78.04 ms |

来源：
- https://github.com/ArcheLabs/mini-cells/blob/main/research/artifacts/004-cell-graph/a59402238b686ff4981786f307d4a3f110698d61/seed-42/reference/metrics.json
- https://github.com/ArcheLabs/mini-cells/blob/main/research/artifacts/004-cell-graph/6312cefae33c354670d2350b315ac20d1e488a51/seed-42/graph/metrics.json

003 的 `soft-cell`→`hard-cell` 降低了实际 GEMM FLOPs，但 PPL 从 12.1822 恶化到 14.8248；且 003 的 `fixed/global-router` 未完成注册的 10M token，因此整体为 `DEVELOPMENT_INCOMPLETE`。004 将 Graph Prior 和 Top-2 Backup Gradient **同时引入**：相比 hard-cell 质量改善，无法据此判断是哪一个机制导致改善。004 的 Graph PPL=13.0706，未过 PPL≤13.0；平均路径熵 Graph≈3.292、Hard≈1.605，未过 Graph<Hard；Graph 的先验接近均匀分布。**004 不是 Graph 结构胜出的证据。**

004 的 seed42 `reference` 与 `graph` 记录在不同的 source commit 目录；虽已检查差异只涉及结果/registry 且配置、定义和数据哈希匹配，仍不满足原协议严格的 source-identity 门槛。不得合并伪造已通过结果。

### 1.2 当前实验没有测到的内容

- 8 个 Expert 对 8 个 Cell：颗粒度**相同**，未验证“多而小”的结构优势。
- 004 中顺序路径重复访问 Cell，较少**独立权重**访问并不意味着较少**执行次数**。
- 已有延迟是双 T4 上**全序列 resident-batch 前向**；不是单 token KV-cache decode，也不是手机实测。
- 未证明 Graph Prior 的独立因果收益、连接可解释性、规模外推、手机可用性。

本实验**绝不重写** 003/004 历史正式状态。

## 2. 理论模型与数学问题

### 2.1 模型和参数预算

复用 004 的 causal Transformer 外壳：6 blocks、hidden width `d=512`、8 heads、seq length=125、vocab=8192、TinyStories 数据身份、10M training tokens/arm（最终是否继续沿用，在冻结 005 协议前决定；本草案建议沿用）。每个 Block 的共享 Attention、Embedding 和输出层保持对照可比。

固定每层 FFN **参数池容量**：

\[
P_{\mathrm{FFN,pool}}=2dH=2\times512\times8192=8{,}388{,}608.
\]

基线 MoE：`N=8`，`hidden=1024`，Top-2；细粒度 Cell：`N=32`，`hidden=256`。在 FFN 主权重上完全等容量；总模型参数因 Router/连接元数据不同允许 **≤1%** 偏差，并须逐参数列示、计入全部开销，不可以隐形增加计算网络。

Cell 是共享状态空间上的一个**残差计算层级**，而不是完整独立 Transformer，也不是静态语义知识单元：

\[
C_i(h)=h+\gamma F_i(\operatorname{Norm}(h)),\quad F_i(h)=W_{i,\mathrm{down}}\,\sigma(W_{i,\mathrm{up}}h).
\]

四步路径：

\[
h_0=x,\quad h_{t+1}=C_{i_t}(h_t),\quad t=0,1,2,3.
\]

各顺序臂使用完全相同的 `Norm`/`gamma` 方案；`gamma` 初值建议为 1/4，是否允许训练应在协议冻结前确定。共享 Attention 每个主干 Block 仍只执行一次。005 不尝试跨 Transformer Blocks 任意跳转（那是更强的下一阶段命题）。

### 2.2 连接的意义

连接 `i→j` **不是参数相似度**，而是在当前状态下“Cell i 执行后允许/倾向于执行 Cell j”的条件计算关系。目标是检验这种约束是否改善函数复合的泛化或搜索成本，而不是预先假定其有语义。

图路由得分（建议）：

\[
s_{ij}(h)=q_\theta(h,e_i)^\top k_j/\sqrt r+a_{ij},\qquad
P(j\mid h,i)=\operatorname{softmax}_{j\in\mathcal N(i)}s_{ij}(h).
\]

- `r=32` 为地址/路由低维宽度（可在 pilot 前调整并冻结）。
- `q` 为整个 Block 共享的轻量投影，`e_i/k_j` 为 Cell 的低维地址；允许 `q` 同时使用当前状态和当前 Cell 身份。
- `\mathcal N(i)` 为每个源 Cell 的 **8 个**候选后继。首次进入使用统一 START 控制节点，其自身没有 FFN 计算参数；所有顺序臂的 START 逻辑等价。
- 优先使用**因子化** `a_{ij}=u_i^\top v_j`，不延续 004 的每个 Cell 输出 N+1 个全连接 contextual head；不允许 O(N²d) 的路由权重爆炸。
- 图连接可重复访问同一 Cell，但记录重复执行和独立权重访问的差异；主实验固定 4 次 Cell 执行，**禁用 STOP**，先排除 003 的提前停止混淆。

理论上，无边限制但同时知道 `(h,i)` 的强 Flat Sequential Router 可以模拟许多 Graph 路径。因此 Graph 的待验证优势是**受限候选空间下更好的训练归纳偏置、结构复用和实际成本**，不能凭“有连接”声称严格更高的函数表达上限。

## 3. 六个主要实验臂

| ID | 容量 | 形式 | 每 Block 激活计算单元 | 候选路由空间 | 科学角色 |
|---|---|---|---|---|---|
| `M0` | 8×1024 | 原生并行 MoE Top-2 | 2 大 Expert | 全局 8 | 最终模型基线 |
| `P0` | 32×256 | 并行 Top-4 小 Cell | 4 小 Cell | 全局 32 | 控制颗粒度，测 B1 |
| `S0` | 32×256 | 状态感知 Flat Sequential | 4 步 | 全局 32，允许 `(h,i)` | 强无图序列基线 |
| `R0` | 32×256 | 随机**固定**稀疏 Graph | 4 步 | 每源 8 个固定候选 | 测连接约束本身 |
| `G0` | 32×256 | **学习**稀疏 Graph | 4 步 | 每源 8 个可学习候选 | 主实验臂，测 B2 |
| `G0-no-edge` | 相同 G0 checkpoint | 后验打乱/替换连接 | 4 步 | 同出度的扰动图 | 无需重训的因果诊断，**不是独立训练臂** |

说明：

1. `P0` 每步多个分支在相同输入上计算并聚合；`S0/R0/G0` 每步的输入来自前一步的输出。**B1 要求 P0 和 S0 在 FFN 访问次数、参数池和训练预算下公平比较。**
2. `S0` 的 Router 必须能访问当前隐状态 **和**前驱 Cell 身份，避免将它人为弱化。
3. `R0` 与 `G0` 要有相同出度、相同路由打分器、同一初始化/训练策略和可用计算预算；区别只在邻接边是否根据学习信号更新。
4. `M0` 的原生 Top-2 MoE 必须完整计入 Router；不可与未收敛的基线比较。
5. **重要附加控制（若预检表明必要，需在正式冻结前引入）：** `S0-shortlist`：状态感知全局 Top-8 候选再执行一步，不含持久的 `i→j` 邻接；用于判断 G0 的优势究竟是稀疏 Top-8 候选，还是 learned connections。没有该控制则对“学习连接独立优于低成本候选检索”的结论须降级。
6. 所有模型训练条件相同；不能让 G0 多训练 token 或使用更多标注数据。备选梯度训练额外计算单独报告。

## 4. Graph 的可学习性：明确的实现契约

仅对当前已有边进行分类、但不探索新边，无法证明拓扑确实被学习。G0 必须有**可审计的边替换机制**。

推荐研究实现（最终实施前进行小规模预检）：

- 每个源节点保留 8 条出边；初始图来自确定性种子，保证全节点可达性，并保证每个节点有至少一条可探索路线。
- 训练时使用共享状态打分器进行候选内 hard 选择；一次正式前向恰好执行一个选定 Cell。Top-2 backup 仅用于反向训练近似，推理完全关闭。
- 每 `R=500` optimizer steps 执行一次**训练集专用** edge-refresh；针对少量训练态，采样候选边以外的后继，用同一质量代理比较下一步的候选价值，再以固定比例替换低价值边。代理可使用冻结或当前模型的 teacher-forced next-token NLL，须事先冻结定义；所有额外探测 FLOPs 与访问的权重在**训练账本**中记录。
- 图学习控制用训练子集训练边评分器 `u_i,v_j`；正式验证、报告集不能反向参与选边。
- `R0` 的边始终固定；如果 G0 的 edge-refresh 额外使用监督/计算，则 R0/S0 必须获得匹配的学习数据和训练预算，或至少增加 matched-training-compute 的敏感性对照。**否则不能把增益单纯归于图结构。**
- 输出图完整快照、每次增删边、触发信号、边分数、平均入度/出度、被访问边和训练中未被访问的节点。
- 学到的候选图是实质的拓扑约束，而不是类似 004 的“全连接 softmax 加一个几乎均匀的先验”。

**验收前置：** 构造一个人工两步函数组合任务，让可学习图对顺序敏感，预检 G0 能优于随机固定连接；验证边变化不是随机抖动。该预检是单元/机制测试，不能冒充语言模型证据。

## 5. 训练协议

### 5.1 数据与预算

- 与 004 对照时沿用已固定的 `roneneldan/TinyStories` revision、tokenizer、50,000 train stories、seq length=125 和相同训练窗口算法；本草案默认每个完整训练臂 **10,000,000 token**、总 5,000 steps（batch 16×seq 125）。
- 以 `M0/P0/S0/R0/G0` 五个**需要训练**的主臂计，单种子总预算 50M tokens；`G0-no-edge` 只评估，不重新训练。
- `G0` 的边提案/探索使用训练集派生的专用 probe 数据，不能接触验证集；probe 文件、采样 ID 和哈希必须固定。
- 004 已经看过的 TinyStories 验证故事只能做**开发回归**；正式确认需预留从未用于调参的独立 holdout 集合，确认其与训练/probe 不重叠并保存内容哈希。
- 开发/确认种子需在冻协议前登记并做 repo-wide collision 检查；不可将已观察开发种子升级为 untouched formal seeds。

### 5.2 Hard-first 训练，不再把全量 Soft 混合当最终推理路径

- 在主实验中，`S0/R0/G0` 从一开始就采用固定四步的 hard forward，避免 003 的 Soft→Hard 行为切换以及巨大的 Soft 路径计算。
- 路由训练统一使用 004 的 **Top-2 Backup Gradient** 作为起点：被选和备选计算用于训练 surrogate，推理只执行选中分支；其计算成本和显存独立记录。
- 必须通过 autograd 检验：hard forward 输出与不启用 backup 的推理相同（数值误差内）、未选训练备选外的 Cell 无额外计算、图梯度有效。
- 路由 temperature、load-balance/entropy 正则、学习率和初始化在开发结束后冻结；不得依照确认集结果改参数。
- 限制训练臂的 batch、数据与总训练 token 一致；额外报告 GPU-hours、训练 FLOPs、前向 Cell 执行和备选 Cell 执行，避免“推理省算力、训练无限烧算力”的隐性比较。

### 5.3 残差幅度与对照有效性

- `P0` 用 `x + Σ w_j F_j(Norm(x))`；`S0/R0/G0` 用四步 `h←h+γF_i(Norm(h))`。
- 序列模型中的 `γ` 需要明确初始化策略并冻结。建议 `γ=1/4` 起步，可用共同的学习型 scalar，但训练范围和正则统一。
- `START` 不消耗一个 Cell 计算名额，第一次选择也要计入路由开销；不要像 004 一样让 Cell0 成为无条件必算入口，从而改变可用自由度。
- 每次 Cell 复用必须更新状态，重复访问的 FLOPs 不得去重。

## 6. 实际激活参数、计算量与推理时间：必须实现的度量

将指标分成**不同物理含义**，严禁互相替代。

### 6.1 独立访问权重：`unique_weight_params/token`

对每个 token，在完整前向中将每个实际读取的权重张量/可区分行片段计数一次：

\[
P_{\mathrm{unique}}(x)=P_{\mathrm{shared,touched}}(x)+\sum_{i\in\mathrm{UniqueVisited}(x)}P(C_i)+P_{\mathrm{controller,touched}}(x).
\]

- 同一 Cell 再次执行：独立权重只计一次，但重算仍计入操作量。
- 共享 Attention、输入/输出权重、正常执行的归一化和 Router 完整纳入（带权重绑定的 output projection 需按唯一权重存储计数）。
- 对 embedding 行与投影权重读法做精确区分：输出词表矩阵投影可能访问全部词表权重；不能把所有 embedding 行简单假定为只触达输入 token 行。
- 汇报 mean / p50 / p95 / p99；同时给出 `unique_weight_params / model_total_params`。
- 若运行时可以复用权重缓存，该值是**模型权重工作集上界/代理**，**不是**实测 RAM 驻留量或 DRAM bytes。

### 6.2 计算量：`executed_gemm_flops/token` 与 `total_forward_cost`

延续 004 的 observed GEMM 计数（含 attention math 路径），并增补：
- 逐层 Cell/GEMM FLOPs、共享 attention、Router/候选评分、所有非 matmul 操作与控制开销（如果后者无法以 FLOPs 可靠定义，分别报告事件、时间和内存流量）。
- `executed_cell_visits/token` 与 `unique_cells/token`；二者不同。
- `train_backup_cell_visits/token` 单列，不可纳入推理节省。
- `P0` 实际 Top-4 计数不能错误地把未选中 Cell 算入。

### 6.3 真实延迟、权重流量与缓存

- 必测 batch=1、实际逐 token 自回归 decode；另测 prefill 和不同 batch。
- 必须有**真正的 KV-cache decode 或等价的合法增量执行**，并用 full-forward logits 做一致性测试；如果当前 003/004 runner 不支持增量解码，应在 005 的工程 gate 中先实现，**不得把 resident-batch timing 冒充 decode**。
- 计入路由、index/gather/scatter、kernel launch、同步和必要内存传输；分开显示 CUDA kernel time 与 host elapsed time。
- 测试 cold / warm cache，明确模型是否常驻 GPU 和量化/权重 dtype；如无法得到可信 DRAM bytes，则记录 profiler 支持的内存活动，报告为 proxy，不编造真实流量。
- 记录 latency p50/p95、tokens/sec、峰值设备内存、活跃权重工作集与执行 kernel 数量。
- 先在 T4 的隔离单 worker 环境完成可比 runtime 测试，手机端测试属于后续目标而非 005 已完成声明。

### 6.4 基线的理论预算校验

- `M0` FFN 主权重每层 `8×2×512×1024 = 8,388,608`；Top-2 对每个 token 执行 `2×2×512×1024 = 2,097,152` 个 FFN 权重对应的计算，FFN GEMM 约 `4,194,304 FLOPs/token/layer`。
- `G0` 每层 `32×2×512×256 = 8,388,608`；四步执行 `4×2×512×256 = 1,048,576` 个 FFN 访问-参数次数，对应约 `2,097,152 FFN GEMM FLOPs/token/layer`，仅为 M0 FFN 的一半；这不是全模型 FLOPs 减半。
- **结构性预测**（未测）：按旧共享主干约 10.56M 和不含新增 Router 计算，M0 独立激活参数约 23.17M，若 G0 每层最多四个不同 Cell，则约 16.85M + 新 Router 开销，理论上可减少约 27%。若 G0 训练后实际质量明显更差，这个理论优势毫无意义。

## 7. 质量评价与实验分析

- 主指标：同数据、同 tokenizer、同验证 token 的 held-out NLL、PPL；PPL 用 token-weighted NLL 计算，提供验证样本 paired bootstrap CI。
- 辅助指标：训练学习曲线、重复路径率、跳转频率、各层使用率、每条连接的触发量、强制步数下的路径多样性、边打乱后的质量变化、复杂输入上的失败案例。
- 不以原始节点编号跨随机种子比较 Graph：Cell 的标签可置换；需要结构不变量、功能对齐或计算行为比较。
- 需报告 `G0` 与 `R0`、`S0` 在每个种子的差值，以及共同采样的置信区间；n=3 种子只能提供有限稳健性证据，不能伪称普适统计结论。
- `G0-no-edge` 必须使用多个固定扰动种子，保持出度、节点数和参数不变；该后验干预只是机制诊断，不能替代独立的 `R0` 训练对照。

## 8. 预建议的 Go / No-Go 门槛（实施前冻结，不作为事后可改数字）

### 8.1 B1 — 顺序组合价值

在**匹配的 Cell 参数和计算预算**下，`S0` 相比 `P0` 的 held-out PPL 应具有稳定改善（建议开发结束后冻结“相对改善 ≥1%”作为确认阈值）；否则只能说顺序结构可训练，不能声称组合效率优势。

### 8.2 B2 — 学习连接价值（主命题）

必须同时满足：

1. `G0` 比同出度、同训练预算的 `R0` 表现更好；建议阈值为 PPL 相对改善 ≥1%，每个确认种子均同向。
2. 相对强 `S0`，`G0` 在**匹配质量**下具有低于 S0 的实际路由或整体推理开销，或在匹配端到端开销下具有更低 PPL；不能仅凭 G0 低于 R0 就声称超越无图路由。
3. 学习图相比固定图存在可验证的非随机边替换；删除/打乱重要边引发可复现的质量变化，并且未发生所有路径归一或大面积不用节点的塌缩。
4. 若 `S0-shortlist` 加入正式矩阵，则还必须报告相对它的结果；否则 B2 只能表述为“超越随机拓扑、未证明超越强全局候选选择”。

### 8.3 S — 面向后续 MoE 迁移的系统门槛

以本种子同训练预算的 `M0` 为基准，**同时**：

- `G0 PPL ≤ 1.03 × M0 PPL`；
- `G0 unique_weight_params/token ≤ 0.75 × M0`；
- `G0 全模型 observed GEMM FLOPs/token ≤ 0.80 × M0`；
- 在完成公平的 decode/runtime 优化后，`G0` 的 batch=1 p50 decode latency 不高于 `M0`，p95 不发生预注册意义上的显著恶化。

`B2` 通过但 `S` 失败 → Graph 连接有科研价值，但不足以支持手机部署目标；`S` 通过但 `B2` 失败 → 小单元稀疏机制可能有效，但不能归功于 Learned Graph。不得混成一种“总体通过”。

**上述数字属于拟注册值**；可在正式确认运行前、仅使用 pilot 数据修订一次并留下 amend 记录。一旦冻结，禁止事后因未过线而调整。

## 9. 预检 / 开发 / 确认的执行顺序

### Gate 0 — 基础设施正确性（无 GPU 完整训练）

- 模型计算图与 shape tests：M0/P0/S0/R0/G0 输出尺寸一致；参数差 ≤1%。
- 控制器至少包含 `(h,i)`；START 处理公平；4-step 无 STOP；重入同 Cell 的 residual 结果正确。
- 确认 Graph 的 candidate constraint 在**推理真实生效**，而不是输出全连接概率。
- Hard forward 与 eval 数值等价；反向 backup 梯度没有跨越禁用路径；训练/推理 FLOPs 核算不同。
- FLOPs、unique weight 计数的玩具解析例子对照（含重复访问、权重绑定）。
- 测试版本、配置、输入/输出哈希的全链路归档与失败时的 resume/retry。

### Gate 1 — 小型合成组合任务（机制预检）

- 设计顺序敏感的合成函数组合，其中 `A→B` 与 `B→A` 导致不同目标；训练/验证输入独立。
- 在相同执行预算下检验 learned adjacency 可以被训练、固定随机邻接有可识别劣势。
- 此结果只是调试路由机制，不可提升为语言模型或 MoE 的科学证据。

### Gate 2 — 单开发 seed 的 TinyStories 全臂试验

- 先单 seed、五个完整训练臂，每臂达到 10M token；分 session 运行只为资源调度，**不能改变数据、source/config identity**。
- 如发现梯度爆炸、图孤立、100% 重复路径、数据错误、超额路由参数等，记录为失败/诊断，修改必须产生新的版本哈希；不能原位覆盖结果。
- 先画完整 Pareto 图：PPL–独立激活参数、PPL–实际 GEMM、PPL–decode latency。

### Gate 3 — 冻结确认协议

- 冻结训练代码 SHA、精确超参、数据 manifest、held-out 切分、学习图边刷新规则、总参数容差、对照和所有通过门槛。
- 对全仓种子做冲突检查，注册完全未观察的 3 个确认 seeds；训练/验证哈希及 metadata 同一版本。
- 确认集评测只能一次性按协议进入决定流程；不允许根据确认结果再挑最佳 Graph 或调整门槛。

### Gate 4 — 多 seed 确认与停止判定

- 每 seed 所有需要训练的主臂、统一输入和完整质量/性能测量结束后才判定该 seed；缺 arm、OOM、校验失败均 `INCOMPLETE`，不是 `PASS`。
- 三个确认 seeds 全部通过对应门槛，才能声明本次样本下 `B1_SUPPORTED` / `B2_SUPPORTED` / `SYSTEM_GATE_SUPPORTED`；明确这些结论仍局限于从零训练的 61M 语言模型。
- 任一核心 gate 失败：原样保留 `NOT_SUPPORTED` 结论及诊断，不许默默扩预算；下一版独立注册。

## 10. 资源安排与执行治理

- 复用 004 已有的 Kaggle 手工启动、离线 pinned 数据资产、双 T4 independent worker、按 arm checkpoint 及 append-only 发布流程。手动运行不设应用内部固定墙钟截断，但云平台自身资源限制仍可能造成中断。
- notebook 只作为薄入口，不放主要训练算法；新 Graph、仪表和分析以可单测 Python 模块实现。
- T4 同时运行两个 arm 可用于训练，但最终可比 latency 测量必须在**隔离的单 worker**里顺序执行，以消除共享 CPU/GPU 影响。
- 长任务每个 arm 定期保存可恢复 checkpoint：model、optimizer、scheduler、RNG、图结构、数据窗口游标、已消费 tokens、当前源码/输入身份；重新启动后不能重复计入训练 token。
- 每次会话输出 `DEVELOPMENT_SESSION_COMPLETED` 不等于整体科学通过；最终 `decision.json` 由独立分析程序根据完整矩阵计算。
- 保护数据/secret：GitHub token 仅由 Kaggle Secret 读取，不可写入 notebook、日志、manifest 或 commit。无需云工作持有签名密钥。

## 11. 代码结构与 Codex 实施范围

建议**新增**而非修改 003/004 的科学历史路径：

```text
research/experiments/005-compositional-cell-graph/
  README.md
  PROTOCOL_DRAFT.md             # 本文及后续审定版本
  protocol.json                # 冻结后机器可读契约
  manual-execution.md
research/configs/005-compositional-cell-graph.json
research/notebooks/03-routing-and-growth/
  experiment-005-compositional-cell-graph-kaggle.ipynb
src/minicells/experiments/exp005/
  cell.py                      # 32×256 residual Cell bank
  routing.py                   # state-aware global / random / learned graph
  topology.py                  # adjacency proposal, refresh, logging
  models.py                    # M0/P0/S0/R0/G0 adapters
  training.py                  # hard forward, shared backup surrogate
  accounting.py                # unique weight, repeated compute, FLOPs
  profiling.py                 # isolated profiling and decode parity
  report.py                    # strict paired-arm / multi-seed gates
scripts/research/
  run_experiment_005.py
  analyze_experiment_005.py
tests/research/exp005/
  test_parameter_parity.py
  test_graph_constraint.py
  test_gradient_and_forward.py
  test_execution_accounting.py
  test_checkpoint_resume.py
  test_data_and_provenance.py
  test_protocol_decision.py
research/artifacts/005-compositional-cell-graph/
  <source-commit>/<seed>/<arm-or-session>/...
```

此为**建议目录**，实施时可根据仓库约定调整，但不能改写旧结果或现有已冻结协议。

### 最小可执行 CLI 契约

```bash
# 预检（无需完整训练）
python scripts/research/run_experiment_005.py --mode preflight --config research/configs/005-compositional-cell-graph.json

# 单臂开发训练（示意；实际参数以 --help 为准）
python scripts/research/run_experiment_005.py --mode train --arm G0 --seed <registered-development-seed> --source-commit <full-sha>

# 最后分析完整矩阵
python scripts/research/analyze_experiment_005.py --archive research/artifacts/005-compositional-cell-graph/<source-commit>
```

这些 CLI 是**开发规范，不是已存在命令**。Codex 应先实现 `--help`、参数验证和 dry-run，再连接 GPU 训练。

## 12. 不得回避的失败模式与解释边界

1. **质量靠多执行 Cell 才改善**：即便 PPL 好，若超出等预算计算，B2 不成立。
2. **纯路由优化代替拓扑学习**：若 G0 相比 S0 或 S0-shortlist 无额外益处，最多说明状态感知顺序路由有益。
3. **训练/推理不一致**：含大量 soft mixture 的训练不能用 hard-only FLOPs 报告替代训练成本；需对 hard-only eval 做独立验证。
4. **Graph 成为几乎全连接**：候选边必须硬限制在 8 且实际裁剪；记录探索阶段额外成本。
5. **Graph 退化成重复执行入口 Cell**：记录 node utilization、repeat rate、边多样性、训练稳定性；不能靠低路径熵当证据。
6. **小 Cell 高 kernel overhead**：即便 FFN FLOPs 减半，batch=1 latency 可能更差，系统 gate 就应失败。
7. **从零训练到预训练迁移**：005 的 61M 阳性不能推出 7B–30B MoE 的可转换性，更不意味着训练后能力保持；后续必须做 pretrained checkpoint transfer/knowledge distillation。
8. **共享状态与完整 Cell 层级的区别**：本研究复用共享 Attention 骨架，只证明 FFN residual Cell 顺序组合；未来完整 Attention+FFN Cell 图及 KV 管理需另立实验。
9. **统计与样本泄漏**：开发用的 TinyStories 验证集不能被重命名为 untouched confirmation。
10. **显存、RAM 与权重访问混淆**：unique touched weights ≠ model file size ≠ peak RAM ≠ DRAM bytes；每个指标独立测量。

## 13. 期望交付的科研报告

每一次完整实验必须有：

- 原始模型/数据/protocol/checkpoint SHA、环境、随机种子、训练 token 和精确参数统计；
- 每臂的收敛曲线和验证 PPL；
- 每层独立激活参数/重复执行/GEMM FLOPs/路由开销/推理延迟；
- Learned Graph 可视化及 edge-refresh history、random/shuffle 对照；
- B1、B2、System 三个**分别给出的**机器判定与失败原因；
- Pareto 图：`PPL vs Unique Active Weights`、`PPL vs FLOPs`、`PPL vs Decode Latency`；
- 对原生 Top-2 MoE 的优势与劣势，以及明确的 **What this does NOT prove**。

### 后续研究路径（005 之后，不在本次承诺范围）

1. 若 B2 和系统 gate 支持，再检验 64/128/更大 Cell 图的候选检索与拓扑扩展性。
2. 迁移到真实预训练 7B MoE，研究如何保留原生能力，并在同质量下减少激活工作集。
3. 迁移到 30B 并接入真实手机运行时，独立验证量化、KV cache、存储/RAM 与热功耗。

---

**最终验收原则：** 005 的第一责任不是证明 Graph Cell 正确，而是检验连接是否真的使小 Cell 的组合**比公平的无图路由更高效**。若随机图、Flat Sequential 或原生 MoE 更好，应原样发布负面结果，不为 Graph 制造有利对照。

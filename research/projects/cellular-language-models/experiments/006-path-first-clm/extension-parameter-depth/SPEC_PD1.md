# CLM × MoE：小参数—长自适应计算链实验设计（PD-001）

**文档类型**：Codex 可执行研究与工程规范  
**版本**：Proposal v0.1 · 2026-10-14  
**目标仓库**：`ArcheLabs/mini-cells`  
**建议归属**：`research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/`  
**初始状态**：`PROPOSED_NOT_REGISTERED`；允许工程实现、单元测试与 CPU 预检，**不授权正式 GPU 训练、confirmation 数据访问或结果发布**。

> **核心科学问题**：CLM 能否用明显少于 MoE 的常驻参数，通过更长、由隐藏状态控制的串行计算链，在相同语言建模任务上获得与 MoE 接近的效用？如果失败，是容量、上下文机制还是 Cell–Connection 联合训练的信用分配与路径退化造成的？

## 0. 给 Codex 的执行指令

1. 在当前仓库**先做只读核查**，确认 Experiment 006 及其 Gate、冻结历史、模型/训练接口；以本规范建立**追加式扩展**，不能重写已有 006 原始草案、结果、注册和审计。若现有 006 仍未注册，不得通过新目录绕开门禁。
2. 实现完整的模型、MoE 对照调用、路径级联合训练、STOP、质量/成本测量、诊断、统计、checkpoint、CI 和 Kaggle 手动运行入口；所有开发采用非科研 seed 的 CPU fixtures。不得为了好看而伪造正式数值。
3. 核心计算必须是实际执行的**单条硬路径**，允许 Cell 自连接；不拆开预训练 Cell/Connection，不使用局部 probe/外部边刷新作为主训练目标，不把并行 Top-k 专家混合冒充串行路径。
4. 保留 MoE 作为第一比较基准；没有对同数据从头训练、经过身份核验的 MoE，不得报告“达到/超过 MoE”。
5. Gate 0 完成后，提交独立分支，报告完整 commit SHA、工程验收、仍待注册的参数和是否触发 GPU（预期：否）。

## 1. 与已有研究的关系——不要重复已有工作

仓库现有 `006-path-first-clm` 已经实现下列**工程机制**：M0（六层缓存注意力的 Top-2 MoE），R0 静态步路由，R1 隐状态路由，R2 隐状态+前驱 Connection 先验；全部 006 当前候选均约 16.5M 参数，较大候选约 61M。R 系列采用只读因果输入 K/V、硬路径、整轨迹 score-function/critic，当前固定六步且明确拒绝 R3/STOP。现有 README 标为 `PROPOSED_NOT_REGISTERED`，不是正式科学结果。

相关仓库主文件：
- `research/projects/cellular-language-models/experiments/006-path-first-clm/{README.md,PROTOCOL_DRAFT.md,AUDIT.md,config/proposed.json,registration/README.md}`
- `.../implementation/exp006/{model.py,controller.py,training.py,baseline.py,accounting.py,evaluation.py,checkpoint.py}`
- `tests/research/path_first_006/`

**本扩展与原 006 的区别**：原 006 主要问“路径与前驱 Connection 有没有独立价值”；PD-001 首先问“**更小参数 + 更长路径是否可与 MoE 质量接近**”，并研究两者在**不分开训练**条件下的共同适应。新增独立 `experiment_id=006-PD1`、运行身份、配置和报告，不将原 R2 的六步结果改名为本实验结果。历史 003/004/005、Native M1/M2/M3、consumed/reserved seeds、005 confirmation 资产及历史结论全部只读保护。

本阶段明确**不做**：Cell 增长、持续学习、模型迁移、MPT 并行验证、PVM/JAM、移动端优化、额外专门的合成组合能力科研阶段。可以建立极小可枚举动作测试，用于检验概率梯度、梯度流与因果正确性，但不能将其宣传为科学验证结果。

## 2. 研究假设与必须区分的结论

**H1 — 参数替代假设（主要）**：在同等训练数据/Tokenizer、相同自回归语言建模目标下，存在参数数目明显小于 M0 MoE 的 PD-CLM，其最终贪心/确定性解码 NLL 对 MoE 满足预注册的非劣效界限，即增加推理计算能够一定程度上补偿较小参数量。

**H2 — 长路径有效性**：对于相同的 CLM 参数规模与训练协议，较长的可用计算预算能使任务 NLL 改善；必须区分“训练时已经允许更多步”与“只在推理时突然增加步数”。

**H3 — 自适应性**：学习型 STOP 可以按隐藏状态选择不同步数，在质量受控时优于固定最大步数的成本；不是只要 `mean(T) < T_max` 就成立。

**H4 — 联合训练机制**：在 Cells 与 Connection **始终共同更新**的前提下，改进整路径信用分配/探索能比现有单轨迹 Actor–Critic 提供更可靠的路径学习，减少早期路由塌陷/饥饿；不能以“单独训好 Cell 再训路由”作为证据。

**H5 — Connection 价值（次要）**：前驱相关连接先验是否在状态路由之外提供增益。只看路由图漂亮、路径多样或自连接比例高并不能证明 H5。

**不允许偷换结论**：本实验的 TinyStories next-token 指标只能支持“小规模语言建模质量非劣效”；不能由 PPL 接近直接声称与大模型有相同推理、知识或通用语言能力。

## 3. 比较体系：以 MoE 为锚点，严格区分三种公平性

### 3.1 主基准 M0：真正的标准 MoE

复用原 006 `M0` 的六层 causal Transformer+Top-2 MoE 架构，**在 PD-001 自己冻结的资产与训练配置上从头训练**。原 006 候选参数为 **16,483,968**；最终以新注册配置实例化实测数为准。不得引用 005 旧 M0 结果充当本实验的对照。

M0 必须保留普通注意力与它的原生训练优化；不允许人为降低 MoE 的专家容量、专家利用率或上下文能力，只为给 CLM 制造优势。

### 3.2 核心 PD-CLM 训练臂（名称建议）

| 臂 | 参数目标（相对 M0 常驻参数） | 路径预算/是否停止 | 目的 |
|---|---:|---|---|
| M0 | 100% | 六层 MoE，原生路由 | **主基准** |
| P100-F16 | 100% ±5% | 固定 16 次动作 | 验证路径架构本身，不主张节省参数 |
| P50-F16 | 50% ±5% | 固定 16 次动作 | 小参数、长度对照 |
| P50-F32 | 50% ±5% | 固定 32 次动作 | 固定参数下的计算深度增益 |
| P50-A32 | 50% ±5% | `0…32` 次实际动作、可 STOP | **主候选：自适应** |
| P25-F32 | 25% ±5% | 固定 32 次动作 | 更强参数压缩对照 |
| P25-A64 | 25% ±5% | `0…64` 次实际动作、可 STOP | **更激进主候选** |

此表为**待冻结建议**，非现有仓库已有正式训练。先由配置生成器真正实例化并打印总参数，不能通过“注册但不使用的假参数”实现配平。总参数包括 Token/Position Embedding、Cell、Connection Router、Context Read、Critic、STOP、Head（如果权重绑定只计一次）。每个参数档位的所有后续臂应尽量复用同一种宽度/Cell Pool，以控制架构变化；确需调整宽度才能达到 25% 时，必须将这一容量与表示维度变化作为混杂因素写入报告。

**核心变量**是“相对 M0 参数数目 × 实际路径长度 × 质量”，不是不断添加更多 Cell。默认允许全部动作自连接，Cell 重复执行的 FLOPs **每次都计入**。

### 3.3 必须有的同参数诊断对照

- `P50-SELF32`：与 P50-A32 尽量使用完全相同的可训练 Cell Pool 和上下文接口，只允许**当前 Cell 自连接、CTX_READ 或 STOP**；可在每个 Token 的初始阶段选择一次起始 Cell。整模型仍从随机初始化端到端共同训练。它仅检验跨 Cell 转移是否必要，不能直接当作公平系统基准，因为动作空间变化会影响训练难度。
- `P50-STATE32`：不使用前驱 edge prior，仅根据当前隐藏状态选择下一个 Cell 或 STOP；Cell、Router 仍然联合训练。其与 `P50-A32` 的状态打分器保持等价；用于判断“显式 Connection 是否提供超出状态信息的价值”。
- `P50-A32-J1`：与主候选完全相同架构和参数，只更换整路径信用分配方式（见第 5 节）；不预训练任何一个组件。

**执行预算控制**：第一轮优先完成 `M0`, `P50-F16`, `P50-F32`, `P50-A32` 的短程诊断；其它配置仅在开发评审通过后完整训练。确认试验之前冻结最终主比较候选及全套对照，不得根据 confirmation 结果增删臂。

### 3.4 三个互补的比较轴

1. **相同训练 Token 数**：所有臂同一语料、Tokenizer、训练样本顺序规则、等量目标 Token；用于检验“数据相同下参数和推理深度的替代”。这是主轴，不意味着训练 FLOPs 相同。
2. **推理质量—实际计算 Pareto**：比较 NLL/PPL 与**完整前向推理实际 FLOPs**，包括重复 Cell、Context Read、Router、输出 Head。单看平均步数没有意义；这是主结论的必备第二轴。
3. **真实硬件成本**：相同 CPU/T4、相同精度、相同 Token 长度、分别在冷/热缓存和 batch=1 与小批量测试峰值权重/运行内存、decode 延迟、吞吐。低参数不自动等于低内存，更不自动等于高速度。

训练总 FLOPs、峰值训练显存、训练墙钟时间独立报告，不能只按相同 Token 数宣称“计算公平”。同等训练 FLOPs 的比较可作为后续附加分析，需预先指定采样 checkpoint，不能补跑后选择对自己有利的结果。

## 4. PD-CLM 架构：状态、Cell、Connection、Self-Loop、STOP

### 4.1 不得隐藏固定 Transformer 主干

沿用 006 的可审计 `static_input_readonly`：

\[
z_n=E(x_n)+Pos(n),\quad h_{n,0}=Init(z_n),\quad M_{\le n}=\{K_j,V_j\}_{j\le n}.
\]

K/V 只从输入 Token/位置生成。`CTX_READ` 使用当前状态查询因果只读 K/V；它是计算动作之一，能重复执行。不能在该接口之后悄悄附加若干未计费 Transformer 层，也不能让未来 Token、标签、目标难度评分进入 Router。

这一设计相较于标准 MoE 的深层动态上下文注意力存在表达能力差异，**不能在 CLM 失败时直接断言是动态连接失败**。届时优先以因果干预和同接口诊断定位，必要时另注册“相同上下文读取机制”对照；不能事后偷偷改 006 语义。

### 4.2 实际串行状态演化

设 `C_i` 是同一隐藏维度上的残差 Cell；操作空间

\[
\mathcal A=\{C_1,\dots,C_C,CTX\_READ,STOP\}.
\]

不应新增独立的 `SELF` 虚假操作。对于上一步执行的 Cell `i_t`，再次选择 `C_{i_t}` 就是**自连接**；它真实执行一次并更新状态，而不是恒等跳过。

\[
h_{t+1}=h_t+\alpha F_{a_t}(LN(h_t),M_{\le n}),\quad a_t\ne STOP.
\]

可学习 Connection:

\[
q_\phi(a_t\mid h_t,i_{t-1},t)=\operatorname{softmax}\big(g_\phi(h_t,t)_a+e_\phi(i_{t-1},a)\big).
\]

所有跨 Cell 转移均可选，前驱先验可学习，不引入硬布尔邻接、离线刷新或“备份专家”混合。Cell 与路由器均由同一 LM 任务共同训练。R1-style `STATE` 对照把 `e` 置零且从头训练，R2-style `A32` 同时学习它。

### 4.3 STOP 与有界执行

- 自适应臂 `T_min=0`，`T_max=32/64`（按臂定义）；**允许首步直接 STOP**，相当于零次 Cell/CTX_READ 动作（但 embedding、路由判断和 LM Head 仍有实际成本）。`STOP` 是可选离散动作，发生时**不执行 Cell/CTX_READ**，直接使用当前 `h_t` 做 tied LM 输出。
- 在 `T_max` 达到时**强制终止**，不是一次额外采样路由；记 `forced_stop=true`，不产生虚构的 `log π(STOP)`。
- 固定臂只能在恰好 `T_fixed` 次实际动作后输出，不能暗中使用提前停止。
- 训练可采用成本系数渐进启用，但任何阶段均同时更新 Cell 与 Connection；成本日程在正式注册前冻结，不能根据 confirmation 调参。
- 不将步数除到终态 CE 上；按有效预测 Token 归一损失，路径长度只通过显式成本项产生影响。

### 4.4 长路径的训练数值边界

64 步必须保留真实跨步信用分配，不允许用随意的 `detach(h_t)` 截断 BPTT 后仍声称等价端到端训练。出现 OOM 先减少**微批量**并梯度累积，使有效 batch 的 Token 数保持一致；可使用 activation checkpointing，但应验证前后 logits/梯度与随机路由重放一致。记录梯度范数、状态范数、Cell 更新幅度与稳定性。`T=128/256` 仅为后续可选扩展，不纳入首次正式比较。

## 5. 主要研究难点：Cell–Connection 始终联合训练

### 5.1 训练目标与梯度定义

对 Token 位置 `n` 的采样完整路径 `P_n=(a_1,\ldots,a_T,STOP)`，定义

\[
J_n(P_n)=\ell_{LM}(h_T,y_n)+\lambda\,Cost(P_n).
\]

优化期望 `E_{P~qφ}[J_n]`。所选择 Cell、输入记忆、输出 Head 对真实执行链做标准反传；Router/Connection 对离散采样使用 score-function 估计（PG）；**路由动作依赖隐藏状态的连续梯度不能无意截断**。

基本估计：

\[
L_{actor}=\frac1N\sum_n\sum_{t\in\text{sampled decisions}}(J_n-\mathrm{sg}[b(s_{n,t})])\log q_\phi(a_{n,t}\mid s_{n,t}).
\]

这里采用损失最小化记号；baseline/critic 的状态输入必须在 critic loss 中断开对 Cell 的梯度，Actor 优势数值整体 `detach`，并确保 `log q` 对形成状态的连续计算具有正确依赖。Critic 用终端损失回归。终端任务 CE 直接反传；不能把 Actor 项中的 `J` 未 detach 而导致重复求导。对 `forced_stop` 不计决策项。熵正则可以预注册用于探索，但不强迫所有 Cell 均匀利用。

**注意**：对于当前静态只读记忆，一个 Token 的路由不会影响其他 Token 的上下文，因此可使用每 Token 终端 NLL 作为路径回报。一旦引入跨 Token 的路径生成记忆，必须重新推导包含后续 Token 损失的信用分配；本实验禁止这种改动。

### 5.2 J0：现有 006 的单轨迹联合训练（基础方案）

每个 Token 采样一条真正硬路径，执行到 STOP/上界，更新 Cell/Connection/STOP/输出层；Critic 用 detached state 预测终端 NLL。沿用可验证的 `Actor–Critic + trajectory CE` 实现，首先修正任何数值/梯度/STOP mask 缺陷。**没有 Cell-only 或 Router-only 预训练阶段。**

### 5.3 J1：同输入多轨迹、整路径相对评价（主要算法对照）

为研究联合训练中“路径评估依赖正在变化的 Cell”的问题，针对同一 Token 在**相同当前权重**下独立采样 `K=2` 条完整路径：

1. 对每条路径真实执行、计算 `J^(k)`；所有被访问 Cell 都正常收到任务梯度。
2. Router 更新采用 `J^(k) - b^(-k)`，其中 `b^(-k)` 是**其它独立路径**在该 Token 上的平均终端损失（`K=2` 时就是另一条路径），它在当前路径的 Actor loss 中必须 `detach`。
3. Cell/Head 的任务损失对 K 条路径取平均；Actor 梯度对 K 条完整路径取平均，不因为路径更长而除以步数。保持与 J0 一样的权重初始化、总参数、训练 Token 数和停止规则。
4. **严格记录实际 2 倍 rollout/训练计算费用**。比较 J0/J1 时，既展示同 Token 结果，也展示等训练 FLOPs 的结果，不把更贵的训练免费处理。

J1 是验证“更好的完整路径相对信用分配是否改善协同训练”的**候选方法**，不是已被证明有效的算法。它只改变联合目标的估计和方差，不把 Cell 与 Connection 拆开训练。可保留单轨迹 critic 作为辅助诊断，但不能混淆不同 baseline 来源。J1 的 leave-one-out baseline 是否满足预期无偏性，应在小型精确枚举单元测试中检验。

### 5.4 训练与共适应诊断（强制）

必须逐 step / eval 保存：
- `ce`, `actor_loss`, `critic_loss`, `entropy`, `cost_lambda`, `train_FLOPs`, `grad_norm_cell`, `grad_norm_connection`, `grad_norm_read`, `grad_norm_head`, nonfinite/update norm；
- 每 Token 长度 `mean/p50/p90/p99/max`、强制停止比例、STOP 概率、采样与贪心 NLL/PPL、不同路径的回报方差；
- 自连接比率、跨 Cell 转移频率、CTX_READ 频率、Cell 使用不均衡/饥饿、路径唯一性、按 `prev→next` 的条件分布；
- 梯度随路径位置的范数衰减、长路径中的状态范数/变化量、重复调用后的真实 `||Δh||`；
- 训练前/后以及固定 checkpoint 的相同输入路径重放、路由漂移、CE 对路由决策的敏感性。

`self-loop` 不能因为多次访问同一 Cell 就默认认定冗余；诊断“无效循环”需结合可复现的**提前停止/替代决策后完整重执行**，测量终端损失变化。路径越长不等于有效计算越多。

### 5.5 干预：辨别“更长”“连接”和“计算实际价值”

只在**冻结模型权重**后执行，保持每个条件的资产、Token、采样种子、计算统计一致：
1. `stop_early`: 对可完成路径在指定步 `t` 强制终止，观察 NLL 与节省 FLOPs；固定臂只能作为诊断，不能改变模型训练语义。
2. `extra_steps`: 在已允许预算内增加实际动作，采用注册的路由策略，不允许训练外任意增至 256 后声称可外推。
3. `edge_zero` / `edge_shuffle`: 对 Connection 先验置零/打乱，**重新滚动**下游状态与路径；不能在改边后沿用原动作序列来估计真实路由策略质量。
4. `order_permutation`: 保留一个路径的 Cell 调用多重集合，重新排列动作并真实执行，评价顺序价值；只证明顺序的因果性，不单独证明 Connection 有价值。
5. `self_loop_replacement`: 选定自连接位置，以其它合法动作替换并重新生成后缀；报告 NLL 差值和新增成本。不能只看局部 one-step loss。
6. `context_zero_replay`: 用固定动作序列移除 CTX_READ 的信息，隔离上下文依赖；需与 route replan 结果分开标注。

所有干预都报告 `NLL_after - NLL_before`、`ΔFLOPs`、配对样本数和统计区间。**策略干预（重新路由）**与**固定动作重放（仅评价转移变换）**的因果意义不同，必须存两个单独字段。

## 6. 数据、训练配置与开发/确认隔离

**默认复用 006 草案候选，但重新冻结 PD-001 身份**：`roneneldan/TinyStories` 固定 revision、训练范围 `[0,50000)`、开发范围 `[50000,51024)`、候选 confirmation `[60000,61024)`，词表 8192、最大序列长度 125、只从训练故事训练 Tokenizer。此范围必须先经过 raw hash、story hash、跨 split 去重/查重、有效窗口数量和任何 005 资产/留存集冲突审计；未通过则**阻断注册**而不是悄悄换窗口。

建议首轮：每臂 10M `next-token target`，AdamW 与原 006 初始候选一致（`lr=3e-4`，`warmup=200`，weight decay=0.1）；**上述仅为建议，训练时一定要从正式注册 JSON 读值**。保持相同目标 Token 数（考虑动态停止后输出 Token 不变）、相同 batch 的有效 Token 数、Tokenizer、随机训练顺序协议与 checkpoint 规则 `last_completed_step`。长路径 OOM 时只改微批量/累积而不改有效训练 Token。

开始前必须验证 10M Token 是否足以训练出**有信息量**的 MoE 锚点；如果 M0 自身训练未稳定，结论为 `BASELINE_UNDERTRAINED`，不能以它证明小 CLM 非劣效。需要增加 Token 数/更大语料时必须**另行修订注册协议**，对所有臂保持一致，不得只给某一方增加训练数据。

建议阶段：
- **Gate 0 工程预检**：全臂构造、自动参数搜索、单步 CPU 前反传、数学梯度枚举、因果/decode 重放、checkpoint 和结果 Schema，无科研训练。
- **Gate 1 预注册**：冻结源 SHA、所有臂/候选、真实参数、预算、损失权重/日程、Tokenizer/资产哈希、开发与确认 seeds（确认至少三组从未观察的 seeds）、主要指标与统计、GPU 授权。
- **Gate 2 GPU 开发**：先同时运行 M0 与核心三臂短程诊断，再按预注册协议扩展训练；开发集可用于一次候选决策，保留完整负面证据。每个分支使用独立 run ID。
- **Gate 3 开发审阅**：判定是否确有长度→质量收益、J1 是否改善稳定性、资源/成本是否有实质可行性；如果没有就停，不强推 confirmation。
- **Gate 4 untouched confirmation**：按 Gate 1 规则在至少三组未观察的训练 seed 上从头跑冻结候选，confirmation 语料**仅在最终 checkpoint 哈希固定之后**加载评估；不得微调门槛/选择更好 checkpoint。
- **Gate 5 研究判断与归档**：输出支持、不支持或不确定；保留失败运行、峰值资源、实际计算和原始统计身份。

Kaggle 环境如沿用 006 两张 T4，允许两个**互不共享状态**的 GPU worker 并发训练不同 Arm；Decode profiler 等两 worker 全部结束后再串行运行，避免测量干扰。Notebook 默认只做 CPU preflight，不自动启动 GPU、不自动读取 secrets，Git 只保存小型元数据/内容哈希；权重与原始语料不提交。

## 7. 评价与预注册判定（建议阈值，非当前正式承诺）

### 7.1 主量化指标

对每个故事/Token 的 next-token NLL：

\[
\Delta = NLL_{CLM}-NLL_{M0}=\log\left(\frac{PPL_{CLM}}{PPL_{M0}}\right).
\]

建议把“接近 MoE”的操作性定义暂定为 **PPL 比值不超过 1.02**（非劣效 margin `δ=ln(1.02)`；正式注册前可审查修改，但**不能看确认结果后改**）。正式结论应采用**单侧 95% 置信上界**检验：

\[
CI^{upper}_{95\%}(\Delta) < \ln(1.02).
\]

这里的 2% 是本研究提出的可操作性公差，而非学界通用的“等同能力”标准。仅仅 `p>0.05` 或两者差异“不显著”**不能**当作非劣效成功。确认同时要求至少 3 个独立训练 seed 的**非劣效结果具有合理一致性**；不能错误地要求每个 seed 都比 MoE 更低 NLL（非劣效允许轻微劣势）。如出现明显 seed 间异质性则记录分歧，不用一个平均数掩盖。

以 **按故事分组的配对 Bootstrap** 计算区间（候选 5000 次，固定统计 seed），不能把同一故事重叠窗口当作独立样本。训练 seed 的不确定性与故事重采样的不确定性应分别列报；如进行汇总层级 Bootstrap，必须同时保留单 seed 表。冻结确认样本量和抽样规则，先检查检测 2% PPL margin 的统计功效；不足时**在 confirmation 前**扩大独立故事范围，而不是看结果后加样本。

### 7.2 硬件与成本指标

必须同时提供：
- **容量**：所有常驻模型权重数、保存权重文件字节数、运行峰值 CPU RSS / GPU allocated+reserved、缓存占用；所有臂相同数值精度与序列窗口。
- **推理计算**：`observed GEMM FLOPs/token`、Attention/非 GEMM 已实现可核算 FLOPs、完整推理 FLOPs/token（若不能完整核算应明确 `INCOMPLETE_ACCOUNTING`）、每 Token 实际 Cell/CTX_READ/Router 调用次数。
- **实际性能**：prefill 与 cached token-by-token decode 分开测试；batch=1 和预注册小批量、CPU 和 T4、相同精度/线程设置、warmup、重复次数，报告 p50/p95 延迟、吞吐、峰值内存。
- **训练成本**：累计训练 FLOPs、墙钟时间、GPU-hours、最大显存；J1 两轨迹必须按实际开销计入。

低参数但 10–100 倍推理 FLOPs 的情形，只能说 `MEMORY_CAPACITY_TRADEOFF`，不能报告 `SYSTEM_SUPERIORITY`。

### 7.3 长路径与自适应性的有效性判据

- 在 **相同 50% 参数**前提下，比较 `F16`→`F32`：不能以单次 seed 波动当作“深度换参数”证据，应显示配对 NLL 改善与实际 FLOPs 上升。
- 比较 `F32`→`A32`：是否在预先规定的质量容忍带内减少实际 FLOPs；`mean(T)` 降低但 FLOPs 或 PPL 显著变差不算通过。
- 提前停止/额外计算干预：至少应观察到**对一部分输入延长路径改善了终端 NLL**；纯粹重复但最终无影响不是有效计算。
- 按难度分桶分析 `T` 分布：可使用被冻结 M0 的真实目标 Token NLL **仅作事后诊断标签**，绝不输入 Router/训练目标；简单/困难分组与计算量相关并不自动证明因果自适应，需结合停止干预。
- 若 J0 出现路由塌陷、长路径梯度无效、严重 Cell 饥饿或 STOP 提前塌陷，应详细保留日志并和 J1 联合训练对比，而不是回退到组件分离训练。

### 7.4 结论状态（务必按证据命名）

`ENGINEERING_READY` / `DEVELOPMENT_SIGNAL` / `CLM_MOE_NONINFERIORITY_SUPPORTED` / `PARAMETER_DEPTH_TRADEOFF_SUPPORTED` / `ADAPTIVE_COMPUTE_SUPPORTED` / `PATH_COADAPTATION_SIGNAL` / `MEMORY_CAPACITY_TRADEOFF` / `NOT_SUPPORTED` / `INCONCLUSIVE` / `INCOMPLETE`。

以上状态**不自动互相蕴含**。对 MoE 非劣效未成立时，允许独立报告“较长路径在 CLM 内部改善质量”；反过来，PPL 接近但路径没有因果收益时，不得声称自适应长路径机制得到验证。

## 8. 自动测试与不可省略的反作弊 Gate

1. **硬路径执行**：对单 Token 记录 module hooks：每步仅选中动作实际运行，self-loop 每次都执行，未选 Cell 不运行；没有隐藏并行专家计算。
2. **Self-loop 默认连接**：`C_i→C_i` 合法，执行后状态真实变化、重复动作真实计费；无重复上限（只受全局 `T_max` 限制）。
3. **STOP 正确性**：`T_min=0/T_max`、初始 STOP、强制终止、已停止位置后续不更新、STOP 不执行 Cell/CTX_READ、不追加伪造 logprob、所有位置恰好输出一个 logits。
4. **梯度数值正确性**：2 动作×2 步的精确轨迹枚举，与 J0/J1 score-function 的**大量样本平均梯度**对照；测试 loss/Actor advantage/critic detach、真实状态依赖的策略梯度、梯度裁剪、合法 mask。
5. **多轨迹无信息泄漏**：J1 的 leave-one-out baseline 不依赖本轨迹动作，Critic 输入不接触目标 Token 标签；多个轨迹不共享可变隐藏状态或 path RNG 状态。
6. **路径/训练可复现**：固定输入+seed+权重 → 同动作、长度、STOP 与 logits；checkpoint 中断恢复与不断训在冻结容差内匹配（包括 Python/NumPy/Torch/CUDA、单独 path RNG、数据顺序、optimizer、mixed-precision scaler）。
7. **因果正确性**：修改未来 Token 不改变过去 logits；`full forward` 与 `prefill+decode` 在固定动作重放下对齐；STOPlength mask 对齐。
8. **无参数偷藏**：自动列出 Cell/Connection/Context/Embedding/Head/Critic 的可训练和常驻总参数。每个压缩臂达到注册比例 ±5%，否则阻断进入正式训练。
9. **成本不免费**：同 Cell 重复 10 次，计算调用/FLOPs 需增加对应次数；上下文 Read、路由、output Head、后续重执行及 J1 的 K rollout 均计入相应成本口径。
10. **资产与授权门禁**：005 保留的 holdout 与历史种子不可碰；未注册时所有 `train/profile/publish/confirmation` 入口 fail closed；确认 holdout 只能在 checkpoint 冻结后读取。
11. **统计审计**：故事分组 Bootstrap、单侧非劣效检验、seed 分层、缺失/失败臂处理、schema、负面结果归档正确；不能把不同 data SHA 或 config SHA 的指标并排当公平比较。
12. **工程可运行**：`ruff`、`pytest`、完整 CPU gate、最小 CUDA synthetic micro-preflight（若用户单独授权）通过；训练前的 Notebook 所有密钥保持脱敏。

## 9. 工程结构、CLI 与输出

只追加 PD 扩展；能复用 006 `exp006` 且语义不变的代码就复用，**不要复制一整套失去审计能力的训练栈**。建议新增：

```text
research/projects/cellular-language-models/experiments/006-path-first-clm/
  extension-parameter-depth/
    README.md
    SPEC_PD1.md                 # 本文档
    protocol.json              # PROPOSED_NOT_REGISTERED
    config/proposed.json
    registration/README.md
    implementation/pd001/
      models.py                # PDCLM + STOP / 固定 / SELF / STATE
      router.py                # 可学习前驱与停止动作
      losses.py                # J0/J1 联合训练与精确枚举对照
      configs.py               # 参数自动搜索与注册校验
      runner.py                # 共享资产和训练恢复工具
      analysis.py              # 非劣效、长度、干预、CI
      profile.py               # 真 FLOPs 和硬件开销
      report.py                # 明确证据状态
    notebook.ipynb             # 手动 Kaggle 入口；默认阻断 GPU
    reports/                   # 小型审计元数据，非权重
  tests/research/path_first_pd001/
```

CLI 可以选择一致的 `run_pd001.py` 路径，但至少暴露以下模式：

```bash
python .../run_pd001.py --mode audit
python .../run_pd001.py --mode preflight
python .../run_pd001.py --mode dry-run --arm P50-A32
python .../run_pd001.py --mode params --all-arms
python .../run_pd001.py --mode train --arm P50-A32 --seed REGISTERED_SEED --config REGISTERED_JSON --source-commit FULL_SHA --assets BUNDLE --output RUN
python .../run_pd001.py --mode evaluate --run RUN --assets BUNDLE
python .../run_pd001.py --mode profile --run RUN --device cpu
python .../run_pd001.py --mode report --manifest REGISTERED_MANIFEST
```

实际脚本位置与 imports 可以遵循仓库约定调整，但命令应可复制运行。每次结果至少保存 `experiment_id/arm/source_sha/config_sha/protocol_sha/asset_sha/tokenizer_sha/seed/role/status/checkpoint_sha`，及 `quality`, `cost`, `routing`, `stability`, `interventions`, `statistics`, `device`, `runtime`, `authorization`。训练失败必须留下 `INCOMPLETE` 与 checkpoint，可安全恢复；正式结果追加不可覆盖。

## 10. Codex 实施顺序与最终交付

**P0 · 只读核查**：输出 006 的实际 HEAD、配置状态、可复用模块、需要修改的模型接口、历史受保护文件清单、当前 Gate 0 结果；若本规范与现有冻结协议冲突，只新增 PD-001 隔离层，先记录冲突。

**P1 · 模型**：实现自连接+STOP 真硬路径、固定/自适应、相同状态处理与 causal input-only K/V；实现总参数配比自动检查及 M0 从头训练入口。

**P2 · 联合训练**：实现 J0/J1 的完整轨迹、无偏性数值检验、Critic、显式计算成本、可复现日志。**绝不拆开训练组件**。

**P3 · 对照与指标**：实现质量/PPL、固定深度曲线、PPL 非劣效统计、硬件成本、干预、Router collapse/Cell 饥饿/梯度诊断、同故事成对评估。

**P4 · 稳定与安全**：资产门禁、seed 冲突检查、可重复/断点续训、CPU 预检、CI、Kaggle 手动 notebook；全部未授权 GPU 入口拒绝。

**P5 · 交付**：commit 并推送独立分支，报告：`branch`, `full_sha`, `source_sha`, `files_changed`, `pytest/ruff/preflight`, `actual_parameter_counts`, `gate_state`, `unregistered_fields`, `protected_files_diff`, `gpu_runs=0`。未做正式训练应写 `NO_SCIENTIFIC_RESULT_YET`，不得自行批准 Gate 1 或冒称 H1–H5 获得支持。

---

## 附：研究设计依据（不是既有 CLM 结论）

- Switch Transformers（MoE 稀疏路由与负载问题），JMLR 2022：<https://www.jmlr.org/papers/v23/21-0998.html>
- PonderNet（可学习停止与动态计算），2021：<https://arxiv.org/abs/2107.05407>
- Mixture-of-Depths（token 级按需计算对照视角），2024：<https://arxiv.org/abs/2404.02258>
- Scaling up Test-Time Compute with Latent Reasoning（循环深度语言模型），NeurIPS 2025：<https://proceedings.nips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html>

这些论文说明动态路由/可变深度/循环计算已有相关研究，**不**替代本仓库所需的、关于 CLM 是否能以更少参数达到 MoE 语言建模质量的直接证据。

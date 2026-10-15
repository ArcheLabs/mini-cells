# FAST-001 — CLM × MoE 低成本联合训练探索实验

**文档类型**：科研实验协议 + Codex 工程实施规范  
**版本**：v1.0（开发探索提案，2026-10-15）  
**项目**：MiniCells / Cellular Language Models  
**仓库**：[`ArcheLabs/mini-cells`](https://github.com/ArcheLabs/mini-cells)  
**建议目录**：`research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/exploration-fast-001/`  
**实验标识**：`006-PD1-FAST-001`（不得覆盖 `006-PD1` 的历史运行身份、配置、指标和 checkpoint）  
**状态**：`ENGINEERING_PROPOSED`。本文件授权 Codex 实现、静态检查、CPU 单元测试与小规模合成预检；**不自行授权 GPU 训练、旧 checkpoint 覆盖、confirmation 数据使用或对外宣称科研成功**。真实 GPU 开发训练须由用户在检查预算和代码后手动启动。

> **一句话目标**：先以很小的模型、相同数据预算和较短训练时间，识别现有 Cell–Connection 端到端训练的有效性与失败机制；再决定是否增加计算深度、训练预算或提出新的学习算法。MoE 是能力锚点，不把“F16 一定优于 F8”当作预设真理。

---

## 0. Codex 执行总指令

1. **只读检查**当前 `mini-cells` 仓库，特别是 PD-001 `models.py`、`router.py`、`losses.py`、`training.py`、`engine.py`、`performance.py`、`PERFORMANCE_REPAIR.md`、`checkpoint.py`、`evaluation.py`、`orchestration.py`。对照本文件记录兼容性/差异；不得直接改动历史科学档案。
2. 在 PD-001 下新增 **FAST-001 独立入口、独立 config、独立 run/seed namespace、独立资产清单和独立结果目录**。优先组合复用现有执行器、MoE、J0 loss、轻量性能模式及 checkpoint，不复制整套旧代码，也不得让新工程偷偷改变原 PD-001 的默认行为。
3. 只实现本阶段**三组真实模型**：`M0-TINY`、`CLM-S8`、`CLM-S16`。如需在元信息中保留未来 STOP/J1 扩展，可以使用未启用的 schema 字段；本次不得训练额外科学臂。
4. 确保两个 CLM 在**每一步都通过单一共享优化器联合更新** Cell、Connection Router、Context Read、Head；不允许 Cell-only/Router-only 预训练，不允许全 Cell 混合代替硬路径，不允许禁止自连接。
5. 完整实现可复现的短程训练、两张 GPU 的任务调度、离线质量/路径诊断、轻量性能基准、断点恢复、自动测试、实验报告与清晰的阶段门禁。所有真实测量与科学判断必须来自运行日志，不得虚构。
6. 提交独立分支并尽可能推送；回报 commit SHA、模型实际参数数、CPU Gate、基准时间和阻断事项。**没有用户进一步授权时，止步于工程预检和可运行 Notebook，不启动真实 GPU 训练。**

## 1. 科学目标与研究边界

### 1.1 我们真正要证明什么

长期假设：

\[
\text{CLM}(P_{\mathrm{small}},T_{\mathrm{long}})\quad\text{可在效用上接近}\quad\text{MoE}(P_{\mathrm{large}},T_{\mathrm{short}}),\qquad P_{\mathrm{small}}<P_{\mathrm{large}}.
\]

这里的“小参数”首先指**模型常驻总参数**更少；“更长”指**真实顺序状态变换次数**更多，而非同时执行多个专家。单个 Cell 的参数自然更少，但它与模型总参数不是同一概念；报告必须分开列示：常驻总参数、Cell 参数、Router/Context 参数、每条路径触及的不同参数、同一参数的重复执行次数、实际计算与耗时。

本阶段不是要证明 CLM 的所有理论主张。小模型的短训练只能提供**研发筛选信号**，不能证明与大型 LLM 或 MoE 具有等同的通用能力。

### 1.2 本阶段优先回答的三个问题

- **Q1：可训练性**。CLM 在低参数和较长硬路径下，能否稳定降低未见开发样本的 next-token NLL，Cell 和 Connection 是否都得到实际训练？
- **Q2：计算链的增量价值**。在**相同 CLM 总参数、Cell Pool、数据和优化协议**下，16 步与 8 步相比，是否体现可复现的语言建模增益，还是出现路径训练退化？
- **Q3：与 MoE 的差距**。一个约为 MoE 总参数一半的 CLM，经过更多计算，能够达到多接近的开发集质量？差距随训练如何变化，训练时间代价是多少？

**重要：Q2 是机制诊断，不是将“更多步必然更好”作为科研目标。** 即使 S16 不优于 S8，也不能自动否定 Q3：有可能当前算法不善于优化更长路径；需要路径因果诊断定位。

### 1.3 明确不做的研究

首轮不研究：自适应 STOP、J1 双轨迹、Cell 生长、外部教师、局部连接预训练、MPT 并行、连续学习、安全本地写入、JAM/PVM 部署、专门 CUDA Kernel、量化、异步推理。后续可以以一个变量一次的方式引入；避免同时改变模型、训练算法和数据。

## 2. 唯一实验矩阵：三组模型

| Arm | 目标常驻参数 | 推理计算结构 | 本阶段作用 |
|---|---:|---|---|
| `M0-TINY` | 约 **2.0M** | 小型标准因果 Transformer + Top-2 MoE，保持 6 层 | 质量锚点 |
| `CLM-S8` | 约 **1.0M** | 8 次硬路径动作，Cell/CTX_READ 可重复 | 小参数较短路径 |
| `CLM-S16` | 与 `CLM-S8` **完全相同参数数量及结构** | 16 次硬路径动作，Cell/CTX_READ 可重复 | 长路径机制诊断 |

**参数量是调参搜索目标，不是已经验证的实际值。** Codex 必须在 `meta`/CPU 设备上真实实例化后打印参数明细。总参数包含嵌入、位置、输入初始化、K/V、Context Read、所有 Cell、Connection、Critic、归一化、Head；共享权重只计一次。不得用不执行的“填充参数”凑数。

### 2.1 参数配平的强约束

- `M0-TINY` 与 CLM 的总参数比例目标为 `CLM / M0 = 0.50 ± 0.05`，即约 45%–55%，不是每个 Cell 的比例。
- `CLM-S8` 与 `CLM-S16` 必须使用完全相同的**可训练参数形状、参数数量和初始化策略**。步数嵌入表按同一个最大容量（例如 16 或 32）预分配；不能因为预算由 8 变 16 而新增可训练参数。
- 两个 CLM 的隐藏宽度、Cell 个数、Cell 中间宽度、Context Read、Router、Critic、残差比例 `alpha` 和输入/输出接口保持一致；只改变 `max_actions`。
- MoE 保持独立的、合理的小型 Top-2 MoE 实现；不能为了让 CLM 变好而人为削弱 MoE。允许搜索 MoE expert hidden width 达成参数目标，但不得更改基线的学习目标。
- **建议起始搜索空间**：共享 `vocab_size=4096`、`max_seq_len=64`；两侧 `width=128`、`heads=4`；MoE `layers=6`、`experts=4`、`top_k=2`；CLM `cell_count=6`，其 Cell hidden width 与 MoE expert hidden width 分别由脚本求解。上述均为候选，实际结构以自动参数配平后的**冻结 config**为准；若目标无法在保留公平架构的情况下实现，应报告失败而不是强行配平。
- 使用同一个 tokenizer、训练与开发数据切分、输入上下文长度、token-level CE/NLL 指标。若隐藏维度不得不不同，须标记为结构混杂项；首选保持一致。

### 2.2 唯一主要实验变量

- `M0-TINY` vs `CLM-S16`：不同架构、CLM 总参数更少且路径更长，属于**系统级比较**；不能独自识别究竟是参数、上下文机制还是优化器造成差距。
- `CLM-S8` vs `CLM-S16`：同参数、同架构、同上下文接口、不同允许执行步数，属于**计算深度机制对照**。
- 不引入第四组固定路由或随机路由的**训练臂**。必要的路由干预在**同一已训练模型的离线评估**中执行，不新增 GPU 长程训练。

## 3. CLM 数学定义与不可改变的语义

给定 Token `x_n`：

\[
z_n=E(x_n)+Pos(n),\qquad h_{n,0}=Init(z_n).
\]

只读因果上下文：`M_<=n = K/V(z_<=n)`，由输入表示产生；`CTX_READ` 根据当前 `h` 查询过去及当前 token，绝不读取未来 token。沿用现有 `static_input_readonly`，不暗中叠加固定 Transformer 主干。要在报告中承认这一 Context Read 与 MoE 深层 Attention 存在结构差异。

动作空间：

\[
\mathcal A=\{C_0,\ldots,C_{k-1},\mathrm{CTX\_READ}\}.
\]

Connection 路由器依据当前状态、前驱动作和步数，给每个动作一个概率。每个 token 每一步**实际采样一个动作并执行**：

\[
a_t\sim\pi_\phi(\cdot\mid h_t,a_{t-1},t),\quad
h_{t+1}=h_t+\alpha F_{a_t}(\mathrm{LN}(h_t),M_{\le n}).
\]

- `C_i -> C_i`（自连接）完全合法，可以连续出现；不同 `h_t` 代表真实不同的计算输入。
- 相同 Cell 可重复使用，**重复执行每次计费**，不能把相同权重调用合并成“一次计算”。
- 计算预算分别是 **8** 或 **16** 个动作；首轮**不启用 STOP**。`CTX_READ` 占用一步；不将其当作免费计算。
- 每条路径终点经过统一的可训练归一化与共享 LM Head 输出 logits；不存在固定深层后处理。
- 全程保持严格自回归因果性和 cached decode/full forward 一致性；随机动作对齐时使用 replay。
- 残差缩放 `alpha` 对 S8/S16 固定为相同数值，可从现有 PD 的 `0.125` 开始，**不得使用 `1/sqrt(T)` 导致两组隐式改变运算子**；后续调整属于新假设。

## 4. 训练算法：只采用联合训练 J0

所有模型从随机初始化训练，两个 CLM 共享相同的训练机制而非共享训练中的权重。Cell、Connection、Context Read、嵌入与 Head 均由同一训练步骤联合更新。

对于已采样完整路径 \(P\)，任务损失是每个目标 token 的真实 CE：

\[
\ell(P)=\mathrm{CE}(\mathrm{Head}(h_T),y).
\]

Cell 和 Head 等连续参数从真实执行路径的计算图反向传播；Connection 的离散路径选择采用现有 J0 整轨迹 score-function + value baseline；critic 的输入在自身训练支路中 detach：

\[
\mathcal L_{J0}=\ell(P)
+\sum_t(\operatorname{stopgrad}[\ell(P)-b_t])\log \pi_\phi(a_t\mid h_t,a_{t-1},t)
+\beta \mathcal L_{\rm critic}-\eta H.
\]

上式为训练代理损失示意；实现须沿用/校验仓库中 `pd001.losses.joint_loss` 的实际路径依赖梯度处理，避免因重构出现 actor 梯度重复计数、baseline 梯度泄漏或无偏性错误。特别注意：**`actor_loss` 可为负且数值很大，不能以其绝对值下降作为路由学习成功的证据。**

- 不冻结任何 Cell 或 Connection；不切换单独训练。
- 不加入为缩短执行路径而设计的 STOP/cost penalty。两 CLM 固定次数下可令 `cost_lambda=0`，成本只用于离线记录。
- 允许优化器共用超参，例如 AdamW、`lr=3e-4`、warmup 50 Step、gradient clip 1.0、weight decay 0.1、统一精度 FP32。这些值是开发起点，须在第一轮开始前冻结。
- 所有 arm 的**目标 token 预算相同**；注意 J0 对 CLM 的 Controller 优化并不代表训练 FLOPs 与 MoE 相同，完整报告训练 FLOPs 和 wall time。
- 初次对照不要启用 J1、多路径采样、混合精度、路径正则或新路由算法，避免无法归因。

### 4.1 S8 与 S16 的初始化控制

两 CLM 可使用相同全模型初始权重（相同权重张量及初始化种子，单独训练进程），相同训练样本选择顺序；路由 action RNG 记录且独立于数据采样器。两组的实际动作序列不必相同，也不应强行对齐。训练损失与梯度需要按 token/轨迹正确归一，不得除以路径长度造成目标变化。

## 5. 快速实验的资源与数据预算

### 5.1 推荐默认 profile（待基准冻结）

| 配置项 | FAST-001 提案 | 说明 |
|---|---:|---|
| 模型数 | **3** | `M0-TINY`, `CLM-S8`, `CLM-S16` |
| Tokenizer | 4096 vocab | 三组完全共享；训练数据生成自己的 tokenizer |
| Sequence length | 64 | 相同有效因果窗口 |
| Effective batch | 8 | 相同每 Step 目标数 |
| Microbatch | 先测 1 和 2 | 最终值根据同 GPU 基准/显存确定并冻结；默认先用 1 |
| 主开发训练 | **500 optimizer Steps** | 每组 `500 × 8 × 64 = 256,000` 目标 token |
| 极短 smoke | 50 Steps | 只用于验证训练与数值，不报告科研胜负 |
| 中途评价 | 100、250、500 Steps | 开发集独立评估；诊断单独计时 |
| 短评估窗口 | 128 个 64-token 窗口 | 仅开发集，不碰 confirmation |
| 终点评估 | 256 个冻结窗口 | 至少按 story 分组，尽可能不同 story |
| 随机种子 | 1 个探索 seed | 若观察到有价值信号，再复验 2 个新开发 seed |
| 精度 | FP32 | 先不引入 AMP 数值变化 |
| 诊断 | 轻量每 100 Step；深度分析仅评估阶段 | 不在每 Step 保留完整路径梯度 |
| Checkpoint | 每 100 Step + 安全暂停前 | 必须支持跨 Kaggle 会话保存/恢复 |

以上 256K token **很少，只能提供快速学习信号，不能衡量模型最终能力上限**。如果三个模型均欠训练，不能把负结果写为 CLM 失败；可在审阅后扩展到 `1000–2000` Steps（512K–1.024M tokens），但这属于明确记录的新开发预算。禁止看过结果后用扩展轮的成绩与没有同样训练预算的旧 M0 比较。

### 5.2 数据集与分组

推荐延用已经审计过来源的 `roneneldan/TinyStories`，但创建 FAST-001 自己的 tokenizer/窗口文件/manifest；不得因为调小 vocab 就继续使用原 8192-vocab 的资产冒充兼容数据。候选原始行范围：训练 `[0,8192)`，开发 `[50000,50512)`；**必须通过源数据 revision/hash、story ID、去重/交叉去重、窗口构造检查后才可确认**。如果这与历史冻结资产或 confirmation 保留范围发生冲突，则遵循已有治理规则重新选择、登记开发范围，不静默覆盖。

- Tokenizer 仅使用训练故事拟合，所有 arm 相同；保留 vocab/vocab hash/语料 hash。
- 开发数据按**故事**隔离，训练/开发不得有相同 story 或近似重复内容；整个序列窗口不得跨越故事边界。目标标签严格右移一位。
- 每组使用**相同且冻结**的开发窗口索引；采样顺序/窗口生成器与模型路径 RNG 分离。
- 过去 Experiment 005/006 和 PD-001 的 confirmation 资产、保留 seed 不得读取或转换成本实验开发数据。
- 探索性多轮调参会重复查看开发集，因此**最终只能对未来全新 sealed holdout**进行确认，不能把开发集显著性写成正式确认。

### 5.3 时间预算门槛（先测量，后承诺）

不凭参数减少推断实际 GPU 加速。**初始工程目标**是在当前 Kaggle T4 上，让三组 500 Step 的首次开发运行整体尽量控制在约 **30 分钟**；这是*待实测的目标*，不是运行时长保证。若预估无法满足，停止启动长程工作并在同一配置下先交付 profiling/瓶颈表；不要为了赶时间自动更改模型、训练 Token 或损失。

在 GPU Pilot 中，先分别对 M0/S8/S16 进行 3–5 Step 预热、10–20 Step 计时，记录每个 optimizer step 的中位值、p90 和峰值显存（不含初始化/缓存/完整评估）。据此重新计算完整三个 arm 的 wall-time 区间，并让用户选择是否执行正式开发训练。

## 6. 两张 GPU 并行调度

优先让两组耗时较长的 CLM 同时开始；较快完成的 GPU 再执行 M0：

```text
GPU 0                         GPU 1
  CLM-S16                        CLM-S8
     |                              |
     |                           完成 S8
     |                              |
     |                           M0-TINY
     |                              |
     |                         完成 M0
  完成 S16

所有 worker 退出后：串行 evaluate/profile/intervene/report
```

允许按**实测预测时间**调整 CLM/S8/S16 的 GPU 归属；不得因设备速度不同据此宣称某架构更快。质量训练允许并行，**性能对照必须在同一张 T4、相同精度、相同 batch/sequence、无其他任务时串行测量**。

- 每个 `arm/seed` 各有输出目录、独立 checkpoint、独立日志、状态；失败不删除其他 arm 进度。
- 先验 GPU 0/1 是否都是 T4、各自剩余显存、当前负载；不足则串行或仅执行 smoke。
- 严禁 GPU worker 同时进行大规模 profile 导致互相干扰。
- 无需把全部训练封装成不可恢复的单次 Run All。允许独立 arm 训练/断点恢复/单独生成汇总。
- Notebook 接近会话配额应提前触发安全暂停，在 checkpoint 成功持久化后退出，而非盲目等待系统终止。

## 7. 性能工程：复用 light profile，禁止再犯旧问题

已检查仓库存在 `pd001-light-v1`：训练路径已经分离轻量损失与完整诊断（`engine.py`、`performance.py`），`PERFORMANCE_REPAIR.md` 说明此前持续保留梯度、逐 Cell `.item()/float()`、前向 hooks、`TorchDispatchMode`、每 Step 两次 Probe 均造成额外调度/同步风险。**本实验必须以最新经过校验的 light 路径为基线**，不能复制旧版取消实验的训练循环。

正常训练：

- `capture=False`，仅周期诊断需要临时采集状态及路径。
- 不在每 Step clone 所有参数、记录完整梯度演化、构建完整路径图。
- 不常驻 `TrainingGemmCounter` / `TorchDispatchMode` / 每层 Forward Hook。
- 不在所有 optimizer step 调用完整 greedy+sampled Probe。
- 多个标量 loss/grad norm 尽可能一次 CPU 转换；必须保留溢出/NaN 的失败保护。
- 数据制作/压缩、日志序列化、保存 Checkpoint 和离线 Profile 不计入纯训练 Step 速度；**也要单独记录端到端总墙钟时间**。

但以下核心执行开销仍在：`torch.where/nonzero` 变长路由、Python 中按 Cell 分组、动态 `index_copy`、小矩阵 GEMM、串行 Router 与 Cell 相互等待。首轮**不因优化速度而改写硬路径执行语义**。在有同 GPU Profiler 证据后，可以另起工程修复分支，使用动作 replay + 梯度对照证明等价。

## 8. 质量与路径机制评价

### 8.1 必须输出的质量数据

- `train_ce`：只作为优化收敛诊断；不作为主结论。
- `dev_nll_greedy` 和 `dev_ppl`：**主要观察指标**。固定同一窗口，计算按目标 token 聚合的 NLL，再用 `exp(NLL)` 得到 PPL。不得先对每个 window 取 PPL 后平均。
- `dev_nll_sampled`、`sampled_minus_greedy`：区别训练采样与推理贪心路由的分布差异。
- 每 100/250/500 step 的冻结开发曲线；训练 Token、训练 FLOPs（有测量时）、实际 wall time、显存。
- `M0-TINY` 必须显示基本学习能力：训练/开发 NLL 均有明显下降且无数值故障；否则标记 `UNDERPOWERED_BASELINE`，不能用其弱表现论证 CLM 成功。

### 8.2 CLM 路径诊断（默认周期汇总，不每步高开销测量）

- `cell_usage` / `unused_cell_fraction`：排查饥饿与永久闲置。
- `self_loop_ratio`、`cross_cell_transition_count`、`ctx_read_frequency`：分析 Cell、上下文读取的实际使用。
- `router_entropy_per_decision`（优先按有效动作/位置汇总）、路径多样性、常见前驱转移；不要把“路由更均匀”自动视为质量更好。
- Cell、Connection、Context 的梯度范数、非有限梯度比例、更新幅度，只在抽样诊断 Step 统计。
- `state_norm`, `delta_norm` 随路径位置的曲线，用于识别早期固定点、振荡、梯度衰减/增长。
- 自连接调用次数、每 Token 唯一触及权重、重复 GEMM 工作量、总 FLOPs/延迟，严格区分独特权重与执行工作。

### 8.3 轻量离线路径因果干预：不新增训练臂

仅对已训练 CLM 的小规模冻结开发窗口操作；固定模型权重，重新执行干预后的后续状态。每项记录相同 token 的 NLL 差值和成本：

1. **Edge-zero / edge-shuffle**：移除或置乱前驱 Connection 先验，后续步骤重新规划，评估显式前驱信息作用。
2. **Same-multiset order permutation**：固定多重集合、置乱 Cell 执行顺序并重新真实执行，评估顺序是否重要；确认 replay 路径合法。
3. **Self-loop replacement**：替换某一次真实连续自连接，保持干预前缀一致，之后重算路径；报告仅在存在重复执行的样本子集上成立。
4. **Context-zero with replay**：保持动作序列不变，将 CTX_READ 输出置零，检验语言能力是否真正使用上下文；同样的干预不等价于删除路由动作。
5. **Truncated path readout**：在固定模型权重下检验第 4/8/12/16 步（只取合法范围）的终端 NLL、状态变化和计算成本。**需要明确这并不是训练过的 STOP，也不是公平的不同深度模型对照**，只作为“额外状态变换是否持续带来短期效用”的诊断。

干预差值接近 0 不代表连接不重要（可能存在替代路径、干预不充分），差值明显不为 0 也不意味着“优于 MoE”。每个报告必须说明干预语义、适用样本数、采样种子和限制。

## 9. 核心对比与解释规则

### 9.1 主指标

定义三个 arm 在**同一冻结开发窗口、同训练目标 Token 数**下的 NLL：

\[
L_M = NLL(M0\text{-}TINY),\quad L_8 = NLL(CLM\text{-}S8),\quad L_{16}=NLL(CLM\text{-}S16).
\]

报告：

- **系统级差距**：`Δ_system = L_16 - L_M`；小于 0 为开发集上 CLM 优于 MoE，接近 0 为接近，但不得直接宣布非劣效。
- **路径预算效应**：`Δ_depth = L_16 - L_8`；小于 0 表示本次 S16 比 S8 好。
- **参数比**：`P_CLM/P_MOE`；**必须报告实际测量**。
- **效用—计算代价**：验证 NLL 与实际 forward FLOPs、吞吐、decode latency、权重常驻字节、峰值 GPU 内存的并列结果。

### 9.2 统计与可解释边界

- 第 1 轮 **1 个开发 seed** 用于找信号，仅展示差值与不确定性，**不提供“显著支持”科研结论**。
- 验证窗口按 story 分组进行 paired bootstrap（建议 2,000 次；首轮也可 1,000 次以省开销，先冻结）。不能将同一故事内重叠窗口当作独立样本。
- 如果观察到方向性改善，再扩展到另 **2 个未使用的开发 seed**，保留每个 seed 的符号与分布。确认阶段必须另有 untouched seed/holdout 与事先冻结门槛。
- 对象是 TinyStories 的有限训练规模；不能声称已证明复杂推理、持续学习或替代大型 LLM。
- 同 Token 预算不等于同训练 FLOPs；同总参数不等于同架构或同上下文表达力；不能把训练时间作为纯数学效率。

### 9.3 决策矩阵

| 观测 | 临时结论 | 下一步 |
|---|---|---|
| M0 与 CLM 都不稳定/无明显 dev NLL 下降 | `BASELINE_UNDERTRAINED` / `INCONCLUSIVE` | 调整学习设置或提高统一 token 预算，不判架构 |
| M0 有效，S8/S16 都无效 | `JOINT_LEARNING_BOTTLENECK_CANDIDATE` | 重点分析 credit assignment、上下文读取、Cell 使用 |
| S16 明显优于 S8，仍落后 MoE | `DEPTH_SIGNAL_ONLY` | 保持小参数，重点优化完整路径联合训练 |
| S16 与 S8 持平 | `NO_DEPTH_SIGNAL_YET` | 检查 Router 是否使用长路径、状态是否近固定点、优化稳定性 |
| S16 比 S8 更差 | `LONG_PATH_OPTIMIZATION_RISK` | 分析梯度与状态演化、终端信用分配，不直接否定长期假设 |
| S16 接近甚至好于 M0 | `PROMISING_DEVELOPMENT_SIGNAL` | 追加开发种子、增大训练预算、重做公平性能/最终 holdout |

**本轮的成功定义**是：三臂稳定、可复现地完成一次公平的快速开发对照，并能通过诊断区分“容量不足、上下文不足、路由/信用分配、无效重复、工程性能”中的主要嫌疑；并不要求 CLM 必须胜出。

## 10. 具体执行 Gate

### Gate 0 — 阅读与模型工程（无 GPU）

- 审计现有 PD-001 并产出 `ARCHITECTURE_AUDIT.md`：实际复用函数、改动点、历史保护范围。
- 建立独立实验 ID、配置文件、模型派生 config/参数搜索脚本、资产契约、三臂执行入口、结果 schema。
- CPU/meta 构造三臂，列出全参数明细和比例，满足 `CLM-S8` / `CLM-S16` 参数完全相等、CLM 对 MoE 约 50%。
- 跑单元测试、静态检查和测试用小数据，证明所有模型能 forward/backward，Cell/Router 梯度存在；此结果不得记为真实语言实验。

### Gate 1 — 数据与正确性（无正式 GPU 科研训练）

- 固定来源、Tokenizer、有效样本和窗口 hash、随机种子、数据源 revision，确认没有与历史保留集冲突；完成故事级去重/窗口边界校验。
- 完成完整路径真实执行、重复自连接、无未来泄漏、动作 replay/full vs incremental decode 对齐、actor/critic 梯度数值 Gate、Checkpoint 重启一致性测试。
- light 与 instrumented 在固定种子短程上的 loss/梯度等价验证；性能监测不得改变训练目标或 RNG。

### Gate 2 — GPU 性能 Pilot（用户手动授权后）

- 同一 GPU 依次运行三臂，每臂 3–5 warmup + 10–20 实测 optimizer steps；单独测峰值显存和全程 end-to-end overhead。
- 记录轻量训练成本、动态索引开销、Profiler 关键栈，计算真实 500-step 预算及 Kaggle 会话配额风险。
- 未经过硬件 Pilot，不得声称 30 分钟完成；如果超预算，**先提交报告并由用户选择**：继续 500-step、改为 250-step 明确降级探索、还是另行修复执行器。不能自动悄悄降预算。

### Gate 3 — 一轮快速开发训练（用户手动授权后）

- 三臂统一随机 seed namespace、数据和训练 budget；双 GPU 按第 6 节调度。
- 每 100 Step 和暂停边界可靠保存 checkpoint；输出独立日志；执行步骤可用 `--arms`, `--resume`, `--max-session-seconds`。
- 100/250/500 对齐评估 + 终点离线路径干预 + 独立串行硬件 Profile。
- 报告所有完成/失败臂；如任何核心 arm 未完成，不发布正式三臂对照结论。

### Gate 4 — 开发评审与下一轮决策

- 根据 `Δ_system`, `Δ_depth`、NLL 曲线、路径因果、稳定性和 wall time 形成 `DECISION.md`。
- 若有积极信号：同架构扩到两个新开发 seed，或统一扩到 1000–2000 steps。
- 若无积极信号：优先定位训练机制，在独立版本中尝试一次只改动**一个假设**（如 J1 双轨迹相对 baseline、反事实后缀信用分配、状态演化稳定机制、STOP），以三臂基础数据作为参照。
- **不得**在本阶段自动打开旧 PD-001 的长期训练、确认集或正式科研胜出状态。

## 11. 工程结构与 CLI

推荐追加式目录：

```text
.../006-path-first-clm/extension-parameter-depth/
  exploration-fast-001/
    README.md
    SPEC_FAST001.md                  # 本文及冻结协议快照
    ARCHITECTURE_AUDIT.md
    config/proposed.json
    config/frozen-development.json   # 用户批准后生成
    implementation/fast001/
      config.py                      # 三臂/搜索/身份/预算校验
      entry.py                       # CLI
      data.py                        # 独立轻量资产与哈希
      train.py                       # 复用 pd001-light，不复制 objective
      schedule.py                    # 双 GPU 并发/恢复
      evaluate.py                    # 冻结窗口质量与路由诊断
      interventions.py               # 小样本真实后缀重算
      report.py                      # 比较 + 判定 + provenance
      resume.py                      # portable checkpoint/校验
    notebook.ipynb                   # 默认 CPU 预检；GPU 明确开关
    reports/                          # 仅小体积 JSON、MD，不放语料/权重
  
tests/research/path_first_fast001/
  .github/workflows/                 # 追加轻量 CPU Gate
```

CLI 可按仓库的既有风格实现，至少具有以下等价能力；实际命令以 `README.md` 为准：

```bash
# 只读审计 + CPU/meta 参数配平
python .../exploration-fast-001/run_fast001.py --mode audit
python .../exploration-fast-001/run_fast001.py --mode params --all-arms
python .../exploration-fast-001/run_fast001.py --mode preflight

# 用户明确授权后，固定配置执行同设备短 benchmark
python .../exploration-fast-001/run_fast001.py --mode benchmark --arms M0-TINY CLM-S8 CLM-S16 --device cuda:0

# 用户明确授权后，开启可恢复开发训练；不含 confirmation
python .../exploration-fast-001/run_fast001.py --mode train-matrix --arms CLM-S16 CLM-S8 M0-TINY --devices cuda:0 cuda:1 --config config/frozen-development.json

# 完整训练结束后，先在单 GPU 串行评估与诊断
python .../exploration-fast-001/run_fast001.py --mode evaluate --all-arms
python .../exploration-fast-001/run_fast001.py --mode intervene --arms CLM-S8 CLM-S16
python .../exploration-fast-001/run_fast001.py --mode report
```

要求：在非 Kaggle / 无 GPU 环境也可以完成 CPU Gate，不能默认一键启动三臂 500-Step；默认 Notebook 打开应是 `preflight`，GPU 训练采用用户可见的明确动作。复用仓库已有自动注册/身份校验时，新的 FAST-001 开发 profile **不得绕开旧 confirmation 封印**。

## 12. Checkpoint 与可靠恢复

- 每臂保存：source SHA、config/protocol/hash、arm、seed、Tokenizer hash、asset manifest、global step、model、AdamW、scheduler、Python/NumPy/Torch/CUDA RNG、独立 path generator、数据采样状态、当前诊断和训练 history。
- 保存原子化；Checkpoint 完整 SHA256 可核验；完成的臂只读复用，**不能自动覆盖**。
- Kaggle Output 需要真正保存为可跨会话取回的持久文件包；新会话先验证来源和散列，再允许 resume。不能假设 `/kaggle/working` 自然跨会话持久。
- 安全暂停只在完整 optimizer step 后发生。`PAUSED_RESUMABLE` 与 `DEVELOPMENT_COMPLETED`、`INCOMPLETE` 不可混淆。
- 不将完整权重、原始 TinyStories、Token、敏感环境变量放入 Git。Git 仅保存小规模的实验配置快照、指标、结果索引和校验值。
- 旧 `006-PD1` 的真实运行（M0 4300 Step / P50-F16 600 Step 的历史日志）仅作为**工程速度问题的历史参考**，不与缩小后的 FAST-001 三臂结果直接合并比较。

## 13. 必须自动通过的验收测试

1. 三臂实际构造完成；`M0-TINY` 为真正 causal Top-2 MoE，非静态 FFN 假基线；两个 CLM 含真实状态路由、前驱 Connection、CTX_READ、同一 Cell 的连续自连接。
2. `CLM-S8`、`CLM-S16` 的所有 `named_parameters()` 名称/shape 一一对应且总数完全相等；含相同最大 step embedding capacity；模型初始哈希可用同 seed 对齐。
3. CLM 总参数比例在冻结容差内；所有常驻、Cell、Connection、Critic、Context 参数统计可审计。
4. 非选中 Cell **不在前向执行**；选中 Cell 的重复执行次数与 trace 一致；`CTX_READ` 成本每次计入；没有 dense expert fallback。
5. 训练损失正确归一至 target tokens，非 path length；Cell/Connection/Context/Head 在被访问时有有效梯度；未被选中的 Cell 本 Step 可以无梯度，不能误判为训练错误。
6. Actor/Critic 数据依赖明确，critic baseline detach；微型枚举测试、有限差分或既有等价测试确认路径策略梯度方向与数学期望一致。
7. 改变未来 token 不影响过去位置 logits；`prefill + decode` 与 full forward 在确定性 action replay 下按容差对齐。
8. 同配置 train->save->restore->继续能重放相同 RNG 行为/动作和训练数据索引；跨会话移植核验 hash；人为中断可恢复。
9. `light` 不常驻完整梯度捕获、逐步全 Cell hooks/Dispatch、每步 Probe；profile/诊断和优化目标隔离。短程同种子等价性测试通过；profiler 真实测试性能与显存。
10. 每轮 500-Step 预算、相同 tokenizer/数据/目标 token、相同开发窗口和故事分组正确；结果缺臂时不产生“成功”的报告。
11. 禁止旧实验历史目录、原 006 Frozen Protocol、005 reserved confirmation、旧已完成 metrics 被修改；CI diff 有审计。
12. 所有研究报告具备完整 provenance；开发 signal 与确认结果状态严格区分。CPU 预检通过不得打印 `CLM_SUPERIOR`。

## 14. 报告格式（最小内容契约）

`report.json` / `REPORT.md` 至少包含：

```json
{
  "experiment_id": "006-PD1-FAST-001",
  "status": "ENGINEERING_READY_OR_DEVELOPMENT_COMPLETED_OR_INCOMPLETE",
  "source_sha": "...",
  "config_sha256": "...",
  "asset_manifest_sha256": "...",
  "tokenizer_sha256": "...",
  "development_seed": "...",
  "arms": {
    "M0-TINY": {"total_params": null, "final_dev_nll": null, "train_tokens": null, "wall_seconds": null},
    "CLM-S8": {"total_params": null, "final_dev_nll": null, "train_tokens": null, "wall_seconds": null},
    "CLM-S16": {"total_params": null, "final_dev_nll": null, "train_tokens": null, "wall_seconds": null}
  },
  "contrasts": {
    "system_delta_nll_s16_minus_m0": null,
    "depth_delta_nll_s16_minus_s8": null,
    "paired_story_bootstrap": null
  },
  "path_diagnostics": {},
  "cost_and_latency": {},
  "research_claim": "EXPLORATORY_ONLY",
  "notes": []
}
```

此处 `null` 是规范占位，Codex 不得填入未经测量的数值；真实运行中需补充完整曲线、训练起点/末点、可观察样本数、CPU/GPU/precision、耗时统计、错误与恢复状态以及所有离线干预。建议至少给出以下一张统一结果表：

| Arm | Params | Train tokens | Dev NLL | PPL | GPU train wall | Forward FLOPs/token | Decode ms/token | Peak mem | Self-loop % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0-TINY | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | N/A |
| CLM-S8 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| CLM-S16 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

## 15. Codex 实施顺序与交付清单

**P0：只读核查 + 设计差异**  
列出 PD-001 与 FAST-001 的参数、路径和数据差异；确认 `pd001-light-v1` 性能修复基线。保留旧仓库所有历史结果与 frozen gate。

**P1：模型缩放与对照**  
实现/实例化三臂；搜索真实参数；验证 CLM-S8/S16 参数 shape 完全一致与 M0 的相对比例。禁止用虚假空参数凑数。

**P2：数据资产与快速训练**  
统一 tokenizer 和故事级切分；复用 J0/PDCLM/M0、light trainer，实现 50-Step smoke、500-Step dev profile、独立 seed/identity、断点恢复与三臂双 GPU 调度。

**P3：评估与诊断**  
冻结 dev windows，完成 NLL/PPL、S16-S8、S16-M0、状态/梯度/自连接和极少量离线路径干预；生成自动化报告，不把可训练性误判为路径有效性。

**P4：性能与质量 Gate**  
CPU 单元测试、CI、同 GPU 的 10–20-Step pilot、Kaggle portable checkpoint 演练、科学历史 diff，交付来源 SHA、配置 hash 和预计时间。未经用户明确允许不自动训练 500 Step。

**P5：整理与推送**  
新分支提交代码、配置样例、README、Notebook、测试和不含秘密的审计记录；回报 commit、CI、CPU Gate、实际参数数、运行脚本、GPU 是否使用（默认否）、仍待用户决定的事项。

---

## 16. 成功与停止条件

**工程完成**：所有 Gate 0/1 测试通过、三臂可运行、实验数据与 checkpoint 可核验、结果 schema 完整、历史研究不受影响。  
**本轮开发完成**：三组在相同训练 token 下完成、得到冻结 dev NLL、独立路径诊断、可复现运行记录和成本报告。  
**机制积极信号**：至少一个 CLM 在开发集稳定学习，且 S16 的有效路径行为/质量改善有可解释证据；不要求超过 MoE。  
**正式科学支持**：本阶段**不授予**。需未来单独预注册、未看过的 confirmation 数据/随机种子、足够训练预算与独立复现。

**最终原则**：我们优先以较小的开销发现训练机制是否有用，不以一次短跑的胜负决定 CLM 长期方向；不为了提升性能而改回密集计算；不把研究仪器开销混同于模型本身成本；不把路径变长等同于路径更有效。

---

### 源码参考（仓库内现有实现，Codex 必须以工作树最新实际版本复核）

- [PD-001 说明与研究边界](https://github.com/ArcheLabs/mini-cells/tree/main/research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth)
- [性能修复与残余瓶颈](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/PERFORMANCE_REPAIR.md)
- [轻量训练 engine](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/implementation/pd001/engine.py)
- [PD-CLM 模型实现](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/implementation/pd001/models.py)
- [联合训练损失](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/implementation/pd001/losses.py)
- [现有参数计数记录](https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/reports/parameters.json)

**备注**：本文件是待实施的研究设计，而非已经运行的实验报告；文中建议参数和加速目标不代表已经实测或被仓库接受。

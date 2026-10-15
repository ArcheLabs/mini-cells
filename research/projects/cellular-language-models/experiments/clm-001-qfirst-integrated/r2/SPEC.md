# CLM-001-R2 — Stabilized Relative Q-First Cellular Learning
## MiniCells 完整 CLM 第二轮：实施规范、工程修复、训练协议与判定门禁

**版本**：实施草案 v1.0 · 2026-10-15  
**项目**：`ArcheLabs/mini-cells` / Cellular Language Models  
**基于**：CLM-001-QFIRST 第一轮源码 `8b7cc5344d0b35f58681286101e93e39ff1fc41d` 与 Kaggle `minicells (1).log`；实施前须对比仓库最新 HEAD，不能假定 HEAD 仍等于该 SHA。  
**拟建目录**：`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r2/`（可依仓库规范调整，须与 R1 不可变结果物理隔离）  
**实验 ID**：`CLM-001-R2`；**目标**：完整 CLM 下，相对 Q-first 能否更可靠地识别 Cell/CTX_READ/STOP 的条件计算收益，并改善实际路径选择？  
**研究属性**：新的单种子开发/探索性实验；**不是**可直接发表的多种子确认、MoE 对标或完整自治生长证据。  
**实施权限边界**：本文件用于交给 Codex 实施。先实现、审计、CPU 验证及 GPU Pilot；真实 500-step Kaggle 开发训练的授权、资源限制与发布权限按实际执行指令核验。不得自动启动 Confirmation，不得覆盖 R1 输出，不得在代码或日志中输出 Secrets。

---

## 0. 执行摘要：本轮必须做什么

本轮不是重新发明 CLM，也不是在 Q-first 外面新增一个 Attention 主干。**保留完整体系**：可重复执行的 Cells、统一 Connection、显式 `CTX_READ`、显式 `STOP`、不可变候选估值、Shadow Cell/Connection/Path 的提议/试用/评价/接纳或拒绝，以及统一的资源账本。

**唯一主要训练机制改进**：从「拟合完整路径绝对 cost-to-go」改为「以立即 STOP 为基准的**相对 cost-to-go**，辅助置信度约束的候选排序」。同时修正状态采样、梯度干扰、Q 估计和工程安全；这些修正整体构成 `QF-R2` 方法包，不能只凭 QF-R2 与 J0-R2 的比较声称某个单点改动独自奏效。

**本轮绝不做的事**：

- 不增加 Connection 的免费 Context Summary、不预先强制 CTX_READ、不让 Cell 直接读取历史 token。原因是 R2 首先应在**不改变信息通道**的条件下改进价值学习；这也保持与 J0-R2 的输入契约相同。可检测这种限制是否导致状态混叠，但不能掩盖它。
- 不预设 Cell 语义专家标签；不靠权重正交损失强制“专业化”；不增加第二个 Cell 内部 Exit Controller。自连接与跨 Cell 连接都属于 Connection 动作。
- 不用已知目标标签直接选择正式训练/推理动作；标签只进入 Oracle 与梯度更新，不进入在线策略输入。
- 不更换 TinyStories 分割、Tokenizer、训练 Token 规模并把结果宣称与 R1 可直接单变量比较。由于训练方法与工程修复一起变化，R1→R2 是探索性的整体版本对照。
- 不为“演示成功”强制让 Shadow 晋升；晋升必须有独立 Admission 证据。

### 0.1 R2 相对 R1 的变更表

| 模块 | R1 现状 | R2 要求 | 优先级 |
|---|---|---|---|
| Q 目标 | 绝对 `CE_terminal + λ×remaining_cost` | `ΔQ(s,a)=Q(s,a)-Q(s,STOP)`；STOP 分数为 0 | P0 |
| 排序学习 | 只用单动作 Huber | 相对值 Huber + **可识别的**动作对排序损失 | P0 |
| 连续参数梯度 | Q/critic 与 Cell CE 全局同一梯度裁剪 | 连续计算组与 Connection 组**分别**归一化/裁剪，账本分开 | P0 |
| QF critic | 训练了不参与动作选择的辅助 critic | QF 不用此 critic 更新/影响 Cell 梯度；可保留常驻参数以匹配 | P0 |
| 采样状态 | 先随机 token、随机 depth，停止后可能抽不到 | 从**真实访问状态**按深度分层抽样；记录覆盖 | P0 |
| 候选后缀 | 每动作固定 2 条 | 先 2 条，**仅模糊状态**可增加至 4/8；全程限额与成对随机数 | P1 |
| 早期路由 | 固定 ε=0.15 | 两臂使用同一预先冻结的探索调度，保留探索下限 | P1 |
| 工程 Shadow | `PROBATION` 梯度非有限导致整个 worker 中断 | 候选级失败隔离+数值故障包；Accepted/Oracle 根故障仍立即中止 | P0 |
| 检查点 | Admission 在第 400 步更新与周期性保存之间执行 | 训练步提交与候选事件**分离为两个事务**，重启不得重复更新 | P0 |
| GPU Pilot | 未阻止第 400 步 Shadow 私有训练失败 | GPU preflight 覆盖全部三类候选的真实 forward/backward | P0 |
| 报告回传 | 失败时发布只有顶层 `INCOMPLETE` | 各臂独立聚合、部分成功可追溯、完整失败分类 | P0 |
| 上下文 | Cell 本身仅见 `h`，独立静态输入 KV，CTX 为可选动作 | **保持不变**；新增因果干预/覆盖诊断 | P0 |
| MoE 对照 | 先前 FAST-001 M0 每层有 Attention | R2 仅 QF/J0 同构内部对照；历史 M0 不作为单变量外部因果对照 | 必守 |

---

## 1. 已观察事实、风险与不可混淆的结论

### 1.1 来自 R1 日志的事实

- Kaggle 使用源码 `8b7cc534...`，CPU `74 passed`，两臂使用 T4 x2 并行执行；初始模型总参数均为 **1,075,492**。
- **QF-FULL 成功提交第 400 步的正式更新**，然后进入第 400 步的 Shadow Admission；在 `Nursery.probation -> Candidate Trainer.update` 中报告 `OracleUntrusted("nonfinite gradient; no update committed")`，QF worker 退出码 1。**不能断言第一个异常的具体算子、参数名称或 NaN/Inf 类型；原始日志没有给出**。
- **J0-FULL 完成 500 步**；其最终开发 `greedy NLL ≈ 4.7043`，`sampled NLL ≈ 4.6417`，`uniform NLL ≈ 4.6411`。J0 greedy 主动 STOP 率与 CTX 调用率均为 0，全部达到预算上限；随机采样 STOP 较常见。这是“路由效率没有得到证明”的迹象，不是确定的理论失败。
- 相同步号 `351–400` 的平均**在线训练 CE**：QF **5.3927**、J0 **4.8767**；对应平均归一化路径成本 QF **0.5293**、J0 **0.4473**。训练 CE 并非独立开发集 NLL；不能据此确认泛化差异。当前证据**不支持**“Q-first 已优于 J0”。
- GitHub 结果分支 `results/clm001-qfirst-8b7cc5344d0b-a0b43988` 已回传 `report.json`，状态为 `INCOMPLETE`、错误 `Development worker failed; retained checkpoints can resume: [1, 0]`；主要为顶层失败元数据，不能视为完整双臂科学结果。
- 本轮新实验必须使用独立 source/config/seed/run id 与结果目录；旧 R1 的源码与日志不可回写，旧 checkpoint 无需冒险跨代码版本加载继续训练。

### 1.2 来自 R1 源码的已确认实现问题

1. `training.py` 的 QF 路径，先把 live CE 反传，再把 `q_loss + 0.5×critic_loss` 反传，之后对**所有模块参数调用一次全局 `clip_grad_norm_(..., 1.0)`**。因此 Q/critic 的大梯度**可能缩小** Cell/embedding/head 的有效更新。这是机制性风险，不等于已经测出了具体梯度数值；R2 必须用组间梯度范数验证。
2. R1 QF 的辅助 critic 并不用于 `distribution(q,...)` 决策，却参与上述反传，可能消耗梯度预算。R2 移除 **QF 上这个不必要的 critic 梯度**，不影响 J0 必需的 critic。
3. `Trainer.update` 的 probe 先对全部 token 均匀选择位置，再随机选 `depth∈[0,B)`；若 token 已 STOP，选定深度不会产生 probe。价值标签对深度的覆盖会偏斜，必须改采已访问的状态。
4. `q_loss` 对入选动作采用 `sum(loss/inclusion) / sum(1/inclusion)` 的**自归一化逆概率加权**。当入选概率变化时，这是自归一化估计量，不等于对全部合法动作平均损失的严格无偏 Horvitz–Thompson (HT) 估计。R2 必须清楚指定目标测度和采样概率，并单独测试估计量性质。
5. `cli.development` 在第 400 步完成训练 update 后执行 Shadow Admission，再执行该步的常规 checkpoint 判断；一旦中间失败，就可能只剩第 350 步可恢复 checkpoint。R2 须引入提交点与事件状态机。
6. `publication.py` 只搜集顶层白名单文件，不主动读取 `QF-FULL/`、`J0-FULL/` 目录的已完成或中断指标；顶层失败报告会丢失足够多的科学证据。R2 必须实现跨 worker 的容错聚合。
7. R1 自动模式没有冻结 Admission 数值阈值，只能测量与拒绝，不会合法晋升。R2 必须**预先冻结阈值或明确测量模式**；不能通过事后看到结果决定门槛。

### 1.3 MoE / CTX 信息流审计结果（R2 不更改）

- FAST-001 `M0` 使用 `MoELM → Block`，**每一层**先执行 `scaled_dot_product_attention`，再执行 MoE 专家 FFN；Expert 本身不直接查询历史 KV。
- CLM 的 Cell 也仅执行当前状态变换；`CTX_READ` 是通过 Connection 决策触发的单独动作，读取 `InputMemory` 的**静态输入 K/V**。这些与 M0 各层动态 K/V 的表达能力并不等价。
- 本轮 QF 与 J0 使用**相同的** Memory/CTX 动作契约。不给其中一臂添加额外的跨 token 观察通道。若发现 Q 无法预测首次 CTX 的收益，应记录为**部分可观测/状态混叠**的负结果，不要在同次试验中偷偷补 Attention。

### 1.4 科学结论边界

R2 能测试：相对价值监督、候选排序、动态路由/STOP、Shadow 候选安全治理是否比既有内部基线更有效。**不能**仅凭这一轮证明：所有 Cell 自发专业化、大规模语言质量可扩展、胜过 6 层标准 MoE，或 Shadow 生命周期长期持续改善。原实验 019 的 `UTILITY_ORACLE_INCONSISTENT` 警示仍有效。

---

## 2. 不改变架构的统一执行语义

### 2.1 输入、状态、动作

对位置 `n`：

\[
 z_n=E(x_n)+P(n),\quad M=\mathrm{InputMemory}(z_{\le n}),\quad h_{n,0}=\mathrm{Init}(z_n).
\]

一次计算决策可视状态：`o_t=(h_t, previous_action, depth, remaining_budget, legal_mask, registry_version)`。完整环境状态实际还含因果 Memory `M`；**当前 Connection 不直接见 M**，这意味着在没有 CTX_READ 之前，Q 是在部分观察下拟合条件期望。这点在公式与报告中要区分：

\[
 Q^{\pi}(s,a),\quad s=(h,M,\ldots)\qquad\text{vs}\qquad
 \widehat Q_\psi(o,a),\quad o=(h,\ldots).
\]

后者可以学习在可观测信息下的**条件平均价值**，但可能无法辨别两个历史上下文不同、当前 `h` 相同的状态。R2 用诊断证明是否存在该限制，**不增加免费的 `ContextSummary`**。

接受动作空间：

\[
\mathcal A(o_t)=\{C_0,\ldots,C_{5},\mathrm{CTX\_READ},\mathrm{STOP}\}\cap\mathrm{legal}(o_t).
\]

预留 Shadow Cell 处于禁用状态，直到事务化接受。`C_i` 可以连续多次调用，也可与其他 Cell 交替。

### 2.2 执行与终止

| 动作 | 执行 | 成本 | 状态变化 |
|---|---|---|---|
| `Cell_i` | `h ← h + α·Cell_i(LN(h))` | `1/B` | 当前 Token 隐藏状态更新 |
| `CTX_READ` | `h ← h + α·CTX(LN(h), M)` | `1/B` | 使用因果输入 K/V 更新 `h` |
| `STOP` | 直接从 `Head(h)` 预测 | 0 | 结束，不再执行任何 Cell |
| `FORCED_END` | 在 B 次非 STOP 动作预算耗尽后输出 | 0 个**新增**动作 | 是预算边界，不是模型学会 STOP |

建议沿用 `B=16`，主动 STOP 至少需 `1` 次非 STOP 操作； `STOP` 不是 Cell 内部的 Exit Network。训练与评估必须保持调用次数、预算、合法动作、输出头、因果边界相同。缓存 decode 的固定轨迹 replay 与 full forward 一致性必须单测。

### 2.3 Shadow 候选与正式动作的边界

Shadow Cell 是候选算子，Shadow Connection 是候选转移偏好/条件，Shadow Path 是多步动作序列；三者都是同一提议/评价/接纳系统的一部分，但**多步 Shadow Path 不可按一个零成本动作计算**。未接受的候选不会进入正式推理的 legal action mask；私有训练不会更改 Accepted 模型的参数、优化器、RNG 与 Registry。

---

## 3. 核心创新：以 STOP 为锚点的相对 cost-to-go

### 3.1 目标定义

在相同的冻结模型快照 `θ_k`、冻结后续策略 `π_k`、同一 token 的目标 `y`、同一状态 `s_t` 和预算下，定义**每个候选动作的实际后续期望损失**：

\[
Q_k^{\pi}(s,a)=\mathbb E_{\omega}\big[\ell(\mathrm{Head}(h_T),y)+\lambda C_{t:T}(a,\omega)\mid s_t=s,a_t=a,\theta_k,\pi_k\big].
\]

其中 `ω` 是候选动作之后的策略随机数；本轮训练标签是**给定真实样本目标 y 的 Monte Carlo 估计**，不是无需标签的真实部署价值。最终部署策略只能看到输入/隐藏状态。

STOP 的直接样本基准：

\[
B_k(s;y)=\ell(\mathrm{Head}(h_t),y),\qquad
Q_k(s,\mathrm{STOP})=B_k(s;y).
\]

对任意非 STOP 候选：

\[
\boxed{\Delta Q_k(s,a)=Q_k^{\pi}(s,a)-B_k(s;y)}.
\]

成本越低越好，因此 `ΔQ<0` 表示与立即停止相比，额外计算预计更有价值；`ΔQ>0` 表示继续计算预计不划算。

**STOP 可能暂时不合法**：当 `depth < minimum_actions` 时，它仍可用作仅供训练的数学基准 `B_k(s;y)`，但策略动作掩码始终禁止此时执行 STOP。必须在日志中区分 `STOP_reference_illegal` 与 `STOP_legal`；禁止把前者计作 STOP 策略监督的准确率。

### 3.2 Connection 参数化（不增加参数）

沿用 R1 `Connection.scorer` 的动作数输出 `raw_cost(o,a)`；通过**每个状态减去自身 STOP 原始输出**，构造相对动作成本：

```python
raw_cost, critic = controller_raw(o)
stop_raw = raw_cost[..., STOP_ID:STOP_ID + 1]
delta = raw_cost - stop_raw        # delta[STOP_ID] == 0 in exact arithmetic
legal_delta = delta.masked_fill(~legal_mask, float('inf'))
a_greedy = legal_delta.argmin(-1)
```

无需另建 Context Summary、Actor Head 或新增 Cell Exit 参数。注意：对**同一个状态全部动作减去同一常数**不会改变 softmax/argmin 的偏好，R2 的改进并非这一代数运算本身，而是：

1. 用真实 STOP 即时 CE 作**差值监督**（而非绝对 CE）；
2. 使用成对差值进行排序学习；
3. 改善监督覆盖与 Q 梯度隔离；
4. 有计划地让 Connection 利用这种相对值学习。

R2 需要测试 `delta[..., STOP_ID]` 精确为 0（容许浮点零）、所有合法动作可比较、非法动作永不可采样，并确认逻辑上 `argmin(raw)==argmin(delta)` 在没有平局时、同一合法 mask 下成立；**平局时必须对原始与相对输出应用同一显式 tie-break 规则**。**不能把“减 STOP 后瞬间改善路由”当成理论假设。**

### 3.3 实际采样策略与 STOP 语义

\[
\pi_{\psi,t}(a\mid o)=(1-\epsilon_t)\operatorname{Softmax}_{a\in legal}\left(-\Delta\hat Q_{\psi}(o,a)/\tau\right)+\epsilon_t\operatorname{Uniform}(legal).
\]

`ε_t` 按**预先冻结的共同时间表**改变，不能用开发集结果按臂临时改变。Greedy 决策包含 STOP=0，因此只有其他合法动作的预测相对价值为负时才继续计算（忽略极小浮点误差/平局的明确规则）。推荐平局显式首选 STOP；平局策略必须事先冻结并纳入所有策略评估，而不是事后只对 QF 调整。

J0-R2 保留相同动作合法性和同样的 `ε_t,τ` 探索预算，但其 `raw_cost` 是**通过原有路径级 actor/critic 学到的评分**，不接受反事实差值/排序回归。这样两臂在输入/资源上匹配，主要区别是 Connection 的训练方法。

### 3.4 相对 Q 值的优势与局限

- 优势：不要求 Q 同时拟合每个 token 的绝对语言难度；同一状态中动作比较可更直接；STOP 基准无后缀采样噪声。
- 不保证：`ΔQ` 无方差、长后缀的 MC 方差消失、由小模型预测的排序一定准确、所有 Cell 会自发专业化。
- 限制：用当前训练目标 `y` 构造 `B(s;y)`，并不能给推理端免费的目标答案；`B` 只能作为**训练标签的中心化项**。
- 统计原则：不同候选价值差距若小于 MC 估计不确定性，应标记 **UNIDENTIFIABLE/TIE** 而非伪造“唯一最优动作”。

### 3.5 边际值训练损失

每个实际探针状态 `o` 可选合法非 STOP 动作子集 `S(o)`，其选择条件包含强制实际动作/可用 STOP 与概率采样的其他动作。对于被采样的非 STOP 动作 `a`：

\[
\mathcal L_{\mathrm{rel}}(o,a)=\operatorname{Huber}\left(\Delta\hat Q_\psi(o,a),\widehat{\Delta Q}_k(s,a)\right).
\]

**目标采样测度先明确**：为使稀疏探针学习“全部合法非 STOP 动作上的均匀误差”，可采用 Horvitz–Thompson：

\[
\widehat{\mathcal L}_{\mathrm{rel}}(o)=\frac{1}{|A_{\mathrm{nonstop}}(o)|}
\sum_{a\in S(o)\cap A_{\mathrm{nonstop}}(o)}\frac{\mathcal L_{\mathrm{rel}}(o,a)}{p_{\mathrm{inc}}(a\mid o)}.
\]

`p_inc` 是动作进入**候选集合**的真实边际包含概率，与模型选择正式动作的行为概率不同。所有包含概率需要随标签保存；若候选由确定性 top-Q/置信策略选取且无法计算有效包含概率，必须保留一条明确定义的均匀探索子样本用于无偏估计，或把剩余估计量明确标记为有偏训练启发式。不得继续把自归一化 IPW 称为严格无偏估计。候选动作集合去重且不可将 `p=0` 的动作放入权重计算。

如果使用“必选实际动作+必选 STOP+随机 extras”，应在 CPU 枚举小世界检查 inclusion 公式；`STOP` 自身没有 `L_rel` 回归项（预测固定为零）。采样状态时也存在状态抽样设计分布；本轮优先明确其分层权重，而不是默认所有状态等权代表所有在线访问事件。

### 3.6 成对偏好损失（仅可靠差值进入）

对同一状态下的两动作 `a,b`，估计共享 CRN 后缀样本的配对差：

\[
\widehat d_{a,b}=\frac1m\sum_{j=1}^m\big[L(a,\omega_j)-L(b,\omega_j)\big].
\]

只在 `(i)` 样本数量达到预设下限（建议 `m≥4`）且 `(ii)` `|d|` 大于预冻结的数值容差与不确定性带宽时，将较优一方作为监督。可采用：

\[
\mathcal L_{\mathrm{rank}}=w_{ab}\,\operatorname{softplus}\Big(\operatorname{sign}(-\widehat d_{a,b})\cdot[\Delta\hat Q(o,a)-\Delta\hat Q(o,b)]/\tau_r\Big).
\]

例如 `d<0` 表示 a 更好，则应惩罚 `pred(a)>pred(b)`。`w_ab` 由置信区间与 gap 给出，有上限；易混淆的动作对 `w=0`，并记录被跳过比例。不可以仅凭两条有噪声后缀就给出强标签。若候选包含概率不等，应对 pairwise 采样偏差审计；未经审计时排序损失仅是**有偏辅助优化**，不能当成真实全动作排序的无偏估计。

总 Connection 训练损失建议：

\[
\boxed{\mathcal L_{QF}=\mathcal L_{\mathrm{rel}}+\beta\,\mathcal L_{\mathrm{rank}}},\qquad \beta=0.1\;\text{（开发候选，需在 Pilot 前冻结）}.
\]

`β` 不得根据最终 Development 指标反向调参。记录单独的 `loss_rel`、`loss_rank`、有效排序对数、有效探针深度/动作与 Q 校准。没有有效偏好对的步骤，仅计算相对值回归。

---

## 4. 反事实估值 Oracle 与探索预算

### 4.1 主 Oracle：完整后缀重新规划

对第 t 步**动作执行之前**的真实访问状态（完整 `h, causally accessible M, previous, depth, registry`）：

1. 在 `θ_k,π_k` 的**同一个不可变 Snapshot** 上固定候选集合。
2. 对每个合法 `Cell_i/CTX_READ`：执行候选动作，**从新的隐藏状态重新调用后续 Connection**，直到 STOP 或预算耗尽；严禁复用原路径后缀的 hidden/output。
3. 所有候选都使用对齐的 `U(depth, sample_index)` 基础均匀数流；相同随机数不等于相同动作序列。
4. 记录每个候选的 `CE_end`、自当前点算起的非 STOP 调用次数、`λC`、STOP 或强制结束、trace digest、候选与策略快照 hash、token 目标 hash、动作包含概率。
5. 即时 STOP loss `B(s;y)` **单独计算一次**并缓存于本探针的只读标签结构。
6. 将 `mean(CE+λC)-B(s;y)` 作为相对值标签；保持样本方差、成对差的方差、后缀采样数和是否可识别。
7. Snapshot 不一致、随机状态污染、未来 Token 泄漏或标签 hash 错误一律 `ORACLE_UNTRUSTED`，不能由 Shadow 的候选局部异常吞掉。

**受控诊断**可以固定后续**动作序列**测试单步算子替换效应，但不可把它与主 Oracle 的“固定后续策略”目标混合报告。

### 4.2 改为从已访问状态中抽样

当前 R1 的 token × 随机 depth 抽样易丢失后期状态。R2 使用**流式已访问状态 reservoir**：在真实 rollout 过程中对每个尚存活的 `(token,depth)` 产生不带 target 的探针候选记录；按深度桶保留固定大小（在 CPU 上可见的记录不应持有所有激活计算图）。从中选择**最多 2 个状态 / 优化 Step**，并分配“浅层+实际可达的较深层”两类配额；实际缺席的深度明确记录 `coverage_missing`。

**防样本选择偏差**：若需要估计全体真实访问状态的平均量，应另用均匀访问事件采样或记录每个桶的包含概率及权重。本轮“按深度分层的训练采样”可用于改善覆盖，但与全体状态自然分布不等价。准确 Q 报告需使用独立预先固定的状态分布，不可只挑容易区分的样本。

建议深度桶 `0–3`, `4–7`, `8–11`, `12–15`；每训练 Step 最多 2 个探针，采用确定的交替分层配额；若所需桶无活状态，允许回退到其他已访问桶并记录。保证 probe 捕获的 `h` 与其源 token 的 `M`、目标位置一一对应。

### 4.3 候选选择与探索覆盖

合法动作数约为 7 Cell/CTX + STOP（预留 Cell 不合法）。对每个状态：

- 实际动作必须加入评分；如果 STOP 合法，必算其**即时基准**（不是再开一个 MC suffix）；另外从其余合法非 STOP 动作中均匀抽取最多 2 个。
- 保留至少一种纯随机动作选择机制，不能全部由当前 Q 的偏好决定；这样才能发现早期被错误低估的 Cell。
- 增加候选类型与 CTX/STOP 的分层计数，避免“收集了很多标签，却从未比较过 CONTEXT 与其他动作”。
- 在正式验证之前，暂不加入完全依赖 Q 预测不确定性的定向候选选择，避免 inclusion 概率难审计；如要加入，必须明确混合分布、保底覆盖与概率审计。本轮首选**随机候选+统计上的自适应后缀重采样**。

### 4.4 预算受限的后缀自适应采样

为避免成本爆炸：默认每个非 STOP 候选 **2 条 CRN 后缀**，在候选对排序不确定且值得区分时增至 **4 或 8 条**。建议自适应基于预注册的配对差 `|d|` 和其标准误，而非真实目标是否有利于 QF。必须记录新增采样策略的规则、总 operator calls、token-step、标签样本数。

为使不同候选配对，**同一个探针下的所有非 STOP 候选使用相同样本数 m 和对齐的 m 条随机流**；否则不能直接称其为 CRN 配对差。

Pilot 用实测总时长冻结一个 `max_counterfactual_operator_calls_per_step`，达到上限时停止**后缀加采样**而非静默削减正式训练 Token；记录 `BUDGET_TRUNCATED_LABEL`，未充分比较的 pair 不参与强排序训练。

### 4.5 不可变性与验证

- 训练标签必须绑定 `source SHA + config hash + data/tokenizer hash + source-model hash + suffix-policy hash + registry version + probe-state hash + target hash + seed/hash`。
- 每个优化 Step 的 label 仅在该 Snapshot 下有效，不允许未经版本化的长期 Replay Buffer。
- 冻结模型短世界中必须用**枚举真值**验证 MC 标签、STOP 差值、成本计数、forced end、后缀重新规划、共同随机数。
- 未来 token 修改不得影响当前位置 logits/决策（在相同 RNG/uniform 输入下）。
- **训练评估隔离**：Train 产生训练目标；Calibration 只检验并作预先声明的决策门禁；Admission 专供候选接纳；Development 不参与调参；Confirmation 不读不训练。

---

## 5. 联合训练算法、稳定性与公平性

### 5.1 主要方法：连续 Cell 梯度与 Connection Q 梯度分组

**不改变端到端任务**，但明确近似优化的梯度通路：

- **连续计算参数** `θ`: embedding、position、init、Cells、CTX_READ、LayerNorm、Head；仅在**实际硬路径**执行的 CE 反向传播。候选反事实 Oracle `no_grad()`，绝不能通过替代路径隐式修改 Accepted。
- **Connection 参数** `ψ`: 对 Detached probe hidden 拟合相对值与可信偏好；QF 不用 J0 actor-gradient，也不通过 Q 回归训练隐藏状态。
- 对 QF，**辅助 critic 的训练不参与正式学习**（可保留等容量模块和参数审计，优化器不更新）；对 J0 则保留必要的 critic。
- 必须为**连续计算组**与**Connection/critic 组**分别统计原始梯度范数、单独裁剪阈值、有效更新范数；不得由 `Q-loss` 通过全局梯度裁剪压低 Cell/Head 的 CE 更新。没有梯度的模块仍需审计为冻结/不活动，而非误称“已训练”。
- 在 QF 中如必要采用分组 Optimizer 或独立 optimizer；两者必须在 Accepted 模型级别共同成功或共同回滚，保证参数/AdamW/随机状态一致。J0 也采用相同的分组裁剪与 checkpoint 语义以控制工程因素。

示例边界伪代码（不是完整实现）：

```python
step_snapshot = frozen_snapshot(accepted_model, route_policy)
trajectory = accepted_model.forward_hard(train_x, policy=shared_epsilon_schedule)

# 1) Accepted 连续任务损失；离散动作本身不通过 CE 反传
loss_ce = terminal_ce(trajectory.logits, train_y)
backprop_to_continuous_group(loss_ce)

# 2) 只在冻结快照评分; 目标 y 只能进入候选评分与任务 CE
probes = sample_actually_visited_states(trajectory, per_step_max=2)
labels = paired_full_suffix_oracle(step_snapshot, probes, train_y,
                                   use_stop_anchor=True, no_grad=True)
loss_q = relative_q_huber(detach(probes), detach(labels))
loss_rank = reliable_pairs_only(detach(probes), detach(labels))
backprop_to_connection_group(loss_q + beta * loss_rank)

# 3) 分组 finite+clip、全状态备份；任意正式更新失败 => 全事务回滚
assert_finite(per_group_losses, per_group_grads, labels)
clip_each_parameter_group_separately()
atomic_accepted_optimizer_step()
commit_train_step_with_checkpoint_identity()

# 4) 生命周期事件在已提交的主训练 checkpoint 之后单独处理
if shadow_event_due():
    execute_shadow_event_transaction()
```

这仍不是对离散策略目标的无偏联合策略梯度；是用反事实监督改善动作选择的 **Q-first 近似策略改进**。报告不得将其称为数学上已证明的收敛算法。

### 5.2 J0-R2 内部对照

两臂相同：模型结构、参数初始种子、`h` 与 `M` 的输入路径、CTX 的只读静态 Memory、STOP 与 forced end、6+1 Cell 容量、有效 Token 数、每次调用的成本、FFN/Attention 算子、分组裁剪基础设施、CPU/GPU Pilot、数据划分、Shadow 候选提议与评价预算、评估 RNG 基础设计。

不同：

- `QF-R2`: 正式 Connection 通过 `L_rel + β L_rank` 训练；可在 QF 训练路径中使用 `argmin relative-Q / epsilon-Boltzmann`。
- `J0-R2`: 用 R1 的 score-function `actor + critic + entropy`，同一 `λC`；不通过候选标签直接训练它的 Router。其候选估值可作为独立诊断/Shadow 治理。

**公平声明限制**：QF 的反事实标签属于额外计算监督，J0 的 actor 利用每条正式路径作为更新信号。即使两臂调用同样的探针，**有效训练监督量仍不同**。必须分别报告实时总计算与部署计算：不能只按模型参数/正式路径 FLOPs 断言 QF 训练更高效。

### 5.3 共享探索调度与目标稳定性

建议候选 `ε(step)`（**两臂同一固定函数**；初次 GPU Pilot 后冻结）：

| Step | ε |
|---|---:|
| 1–100 | 0.40 |
| 101–300 | 0.25 |
| 301–500 | 0.15 |

`temperature=1.0`、`cost_lambda=0.01` 可作为 R1 延续候选；它们不一定是最佳值，必须在 Pilot 前冻结并注明为何选择。**不要**根据 R2 Development NLL 临时修改。可报告按后验 λ 重新打分的成本曲线，但不能将那种重新打分当成真实新 λ 策略执行。

为控制 Q/Cell 的相互反馈：

- 同一训练 Step 的 Oracle 后续策略固定于 Step 开始的 Snapshot；绝不混合 `θ_{k+1}` 与 `π_k`。
- 如使用慢更新 `target_Q` 构造**后缀策略**，必须明确该策略与正式 `π_live` 的区别，记录两个策略 hash；基于某一冻结策略训练的标签不得伪称另一策略的 Q。**R2 核心版本建议先不用 EMA target**，以减少未验证目标漂移机制；使用逐步快照即可。
- 必须报告 `Q_prediction_delta`, `Q_gap`, `Q_mae`, paired ranking 以及 Calibration regret 的时间曲线；只看 Q Huber 下降无研究价值。

### 5.4 计算成本与 STOP

保持 `cost_unit = nonSTOP_calls / 16`、候选 `λ=0.01`。每增加一项真实计算操作，目标成本增加 `0.01/16=0.000625`，**这个系数很小**；STOP 是否有效取决于继续计算的边际 NLL 改善而不是绝对 NLL。需报告：

- 主动 STOP 比例与深度分布；forced end 比例独立统计；
- mean、p50、p90 非 STOP 调用次数，CTX 调用次数；
- 各深度立即 STOP 的真实样本收益与继续计算收益配对对照；
- 最终的 `NLL vs 真实有效调用`（至少 greedy、sampled、uniform、constant）以及真实 latency、GPU bytes、候选探索 wall；
- **STOP collapse**：全部立即 STOP 或全部强制 16 步，都要区别于“每种状态恰好有相同最优行动”的可能性，结合独立 STOP 反事实判断。

---

---

## 6. Shadow 生命周期与已暴露的数值故障：必须先修复

### 6.1 正确的故障分类与状态机

R1 原有状态：`PROPOSED → PROBATION → EVALUATED → PROMOTED/REJECTED/EXPIRED`。R2 新增 `FAILED` 终态，**只用于已确认局限于候选私有分支的故障**：

```text
             PROPOSED
                │
                ▼
             PROBATION ─────────────► FAILED (candidate-local)
                │                          │
                ▼                          └── preserve Accepted; report negative evidence
             EVALUATED
                │
         ┌──────┴────────┐
         ▼               ▼
      PROMOTED        REJECTED
```

必须区分以下错误严重度：

| 错误 | 分类 | 训练行为 | 结果声明 |
|---|---|---|---|
| Shadow 候选参数/梯度 NaN/Inf，Accepted 的字节级 witness 保持一致、Oracle 正常 | `CANDIDATE_NUMERICAL_FAILURE` | 丢弃该私有分支、标 `FAILED`，不影响其他候选与后续正式训练 | 完整主训练可继续，但生命周期失败必须计入结论 |
| Shadow 候选自身出现禁止动作/无法完成预测；Accepted 未受污染 | `CANDIDATE_CONTRACT_FAILURE` | `FAILED` 并保存最小诊断；其他候选可以继续 | 同上 |
| Accepted 模型参数/梯度非有限 | `ACCEPTED_NUMERICAL_FAILURE` | **立即停止该 arm**；回滚正式事务 | 本 arm `FAILED`，禁止伪装完成 |
| Oracle Snapshot/同版本校验失败、未来信息泄漏 | `ORACLE_UNTRUSTED` | **整个对应训练路径 Gate 阻断** | 不得继续用该 Oracle 训练 |
| Accepted 版本 witness 或原子 Checkpoint 不一致 | `INTEGRITY_FAILURE` | **停训，保留原文件** | 结果必须标 `INCOMPLETE` |
| 远端发布失败 | `PUBLICATION_FAILURE` | 不重新训练；保留本地结果并支持 retry publish | 训练结果与发布状态分别记录 |

`except Exception: continue` **禁止**。必须捕获少数受控的候选局部异常，保存异常类型与原生 traceback，然后核验父模型参数、Optimizer、Registry、RNG、缓存无污染，才能继续。若父模型状态发生任何未经授权的变化，立即升级为 `INTEGRITY_FAILURE`。

### 6.2 非有限梯度定位要求

在每个私有 Shadow Step 的 backward/clip/optimizer 前后分别检查：

1. 总 loss 及分量 `CE / loss_rel / loss_rank / J0 actor / J0 critic / entropy` 是否为 finite；
2. `grads` 中首个非有限元素所属 `parameter_name`、shape、dtype、`nan_count/posinf_count/neginf_count`；
3. 各参数组 grad L2 norm、max abs、剪裁前后 norm；对**存在 `None` 梯度**的 Cell 单列说明；
4. 候选类型 `CELL/CONNECTION/PATH`、候选 Registry 与版本、合法动作 mask、路径动作摘要、步数、CTX/STOP 数、初始化来源；
5. 发生在 `forward / backward / optimizer_step / evaluate / admission / persist` 哪个子阶段；
6. 模型/Optimizer/Candidate 的哈希、训练 Step、私有 Step、分配给候选的独立 RNG 哈希、冻结数据批次 digest；
7. 对确实发生的非有限异常，在**候选私有沙箱**上附带 `torch.autograd.detect_anomaly` 的可选复现路径；不得默认在全量训练每步打开异常检测而大幅降低速度。

诊断数据须可通过 `stage, candidate_type, branch_snapshot_sha, private_step` 索引。**禁止**上传原始训练文本、目标 token 内容、权重、AdamW 动量、私钥或任何 Kaggle Secret 到 GitHub。最小故障快照（如明确授权且空间允许）可留 Kaggle Output，GitHub 只发布汇总统计与 hash。

### 6.3 Candidate 的训练边界和优化器状态

`Nursery.propose/rebase/probation` 须明确哪些参数可训练：

- **Shadow Cell**：目标 reserve slot（例如 `cells.6.*`）和其对应的 Connection 参数；旧 Cell/Head/Embedding/Memory 只读。需要验证已激活 reserved Cell 实际参加前向与获得有限梯度；若未被访问，不可以把“零梯度”包装成学会了新技能。
- **Shadow Connection**：只训练候选转移/Connection 相关参数；未修改正式 Cell 的权重。
- **Shadow Path**：候选计划是路径提议而非自动新建大型权重；如果其私有训练允许更新 Connection，需要说明固定前缀计划下真实可学习的参数范围与前缀逻辑的一致性。
- R1 的 `rebase` 会在 Admission 时重新基于 Accepted 模型创建候选副本；R2 必须将 Proposal Blueprint 与当前候选参数快照显式区分，保存 `proposed_parent_sha`, `rebased_parent_sha`，不能宣称 Step 100 创建的候选训练了 300 步。
- optimizer 状态：候选可以从父优化器 deepcopy，但**新启用槽及发生拓扑变化的参数需明确初始化/重置 Adam 矩**；不得让历史旧参数的动量在没有对应意义的新候选上生效。CPU 单测须证明未训练参数仍字节一致。
- 模型/所有候选分别使用独立随机数子流；`rebase` 不得因为每次新建临时 Nursery，而把所有候选重置为同一个隐含私有 seed；seed derivation 应绑定 `(experiment, arm, candidate_uuid, phase, parent_version)`。

### 6.4 Shadow 评价与合法接纳

建议维持 R1 预算：最多 `1` 个 Shadow Cell Birth 与 `1` 个 Connection/Path 提议组；Step 100/200 提议、Step 400 Admission（需以里程碑驱动的事件 ID 保证 exactly-once）。对每个候选：

1. 创建/隔离 → 候选私有训练 → `PROBATION` 结果检查；
2. 计算**同一父版本、同一数据、同一随机种子、相同成本规则**的 Accepted 与 Candidate 性能；
3. Admission 使用独立的 `admission` 角色数据按 story 聚类；
4. 必须检查旧能力回归、因果性、合法动作与 STOP、最大深度约束；
5. 若阈值尚未冻结，则只做 Measurement/Reject，不允许 Promote；若阈值冻结，必须通过独立的增益及回归 Gate 才可 Promote；
6. Promote 必须原子提交模型、Optimizer、Registry、RNG、Q 输出行映射、版本和 journal，失败完整回滚；任何晋升必须使旧 Snapshot 标签失效。

**推荐本轮保守 Admission 候选阈值（进入开发前冻结，可经 Pilot/独立 Calibration 讨论后调整）：**

- `minimum_admission_stories >= 32`；
- paired story bootstrap（≥1000 replicates）下 `upper95(Δ[NLL+λC]) < -0.005`；
- `upper95(ΔNLL) ≤ 0`；
- 结构契约、因果性、算子数与容量账本全部 PASS；
- 失败候选永不晋升，接受比例可为零，不能降低门槛“确保有增长”。

这是一组**提议中的探索性阈值**，不是历史实验验证出的最优标准。应在 `r2.lock.json` 中锁定，在开发运行结束后不得更改并重判同次结果。Admission 无候选通过不算框架故障：只说明这批候选未证明有效。要测试原子晋升行为，在 CPU **注入已知必通过的合成候选**完成，不得向实际模型强行晋升无益候选。

### 6.5 第 400 步事务性修复：训练更新与候选事件拆分

R1 第 400 步在 Training Commit 后、下一 checkpoint 之前进入 Shadow，因此失败时可能只剩第 350 步。R2 必须拆为：

```text
main update #400
  ├─ finite check + optimizer atomic commit
  ├─ checkpoint-step-000400.pt  (immutable)
  └─ latest.json -> step400      [主训练已经完成]

shadow event: admission-000400
  ├─ admission-event.json = STARTED (event id + parent hash)
  ├─ candidate #1 probation+evaluation
  ├─ candidate #2 probation+evaluation
  ├─ candidate #3 probation+evaluation
  ├─ each individual status persisted atomically
  └─ admission-event.json = COMPLETED / PARTIAL_FAILURE

main update #401
```

重复启动时，**不能重复第 400 步主训练**；如果 Step 400 已保存、Admission 事件 `STARTED`，则只恢复未完成的候选事件。事件提交必须与候选训练状态一致。若加载到 `Step 350` 说明新检查点未提交，可正常重放 Step 351–400，但必须使用相同输入/RNG/版本标识。Checkpoint 兼容性与移动文件原子更新单测必需。

此外，在 Step 100/200 提议时也要引入事件事务：已经创建的候选不能被重启重复创建或重复消耗 Birth 预算。每种事件有 `idempotency_key=(run_identity, step, event_type, registry_version)`。

---

## 7. 工程任务分解：交给 Codex 的具体改动

建议保留 R1 原目录不变，新建 `r2` 实现与配置，并复用只读模块；若决定在同一个 Python package 内用版本开关实现，必须由**独立入口、固定协议和测试**保证 R1 可复现，禁止静默改变 R1 行为。以下以新增 `r2/` 为例，文件名仅为建议，Codex 可根据实际仓库树结构等价拆分。

| 任务 | 建议位置/涉及原文件 | Codex 具体工作 | 必须的验收 |
|---|---|---|---|
| R2 配置/入口 | `r2/config/proposed.json`, `r2/run.py`；原 `common.py` | 新 `experiment_id`、seed namespace、Source/Manifest/λ/ε 版本锁与清晰运行模式 | R1 identity 不可误用，未授权模式拒绝运行 |
| Relative Connection | `r2/.../q_connection.py`; 原 `q_connection.py` | 输出相对 STOP 的 Delta、所有动作合法 mask、正确梯度 | STOP 零、masked argmin、与 raw 排序等价单测（对平局应用统一规则） |
| Relative Label | `r2/.../oracle.py`; 原 `oracle.py` | 加单状态 `stop_loss`、差值目标、CRN 配对方差、识别性标签 | 枚举短世界正确、未来 token 不泄漏 |
| Probe Sampler | `r2/.../rollout.py`, `training.py` | 仅从已访问状态分层抽样，保留 target-free 提议过程和包含概率 | 深度覆盖、active-only、无标签参与选择 |
| Q 回归与排序 | `r2/.../training.py` | 正确 HT 权重；鲁棒相对 Q Huber；可信动作对排序 | HT 小世界枚举/MC 校验、样本权重/方向正确 |
| 梯度隔离 | `r2/.../training.py`, `optimizer.py` | 连续参数与 Connection 分组 finite/clip；QF 无用 critic 梯度取消；J0 保留 actor/critic | Q 梯度大时 Cell 更新不缩小；参数哈希与回滚正确 |
| Shadow 安全 | `r2/.../shadows.py`, `admission.py` | typed failure、父 witness、候选不同 seed、候选独立 AdamW 与升级事务 | 3 类候选均能成功及故障注入，Accepted 不变 |
| Checkpoint / 事件 | `r2/.../checkpoint.py`, `automatic.py` | 主训练提交/Shadow 事件分离，exactly-once 恢复、不同 arm 独立状态 | Kill/restart before/after Step 400 全通过 |
| 双臂评估 | `r2/.../evaluation.py`, `report.py` | dev NLL 与成本、Q regret/排序、STOP/CTX/自连接、路由干预 | 相同样本、分组 bootstrap、明确定义政策 |
| 容错发布 | `r2/.../publication.py`, `automatic.py` | arm 独立指标、顶层部分汇总、白名单/hash、只读 token 隐私 | 单 arm 失败仍能安全上报另一 arm 的完成指标 |
| 测试/CI | `tests/research/clm001_r2/`, `.github/workflows/...` | CPU synthetic 正确性和 property-based 测试 | 不依赖 Kaggle Secret、无 GPU 即可 PASS |
| 文档/审计 | `r2/README.md`, `SPEC.md`, `ARCHITECTURE_AUDIT.md`, `CHANGELOG.md` | 列清相对 R1 的变更、限制与科学门禁 | source SHA 可追溯，结果不得过度宣称 |

**旧文件作为参考的已知锚点**：

```text
research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/
  implementation/clm001/{training,q_connection,oracle,sampling,rollout,shadows,
                        admission,checkpoint,automatic,cli,evaluation,report,publication}.py
  config/proposed.json
  ARCHITECTURE_AUDIT.md
  ORACLE_TRUST_GATE.md
  SHADOW_ISOLATION_AUDIT.md
  PERFORMANCE_BUDGET.md
```

当前仓库该提交只保证具备上列 R1 路径；R2 新文件尚未实现，Codex 不应在施工日志中声称已存在。若仓库已有更新，应先列出 `git diff` 风险、重用已有修复，不可覆盖别人已提交的同类功能。

---

## 8. CPU 正确性、故障注入与 GPU Pilot：运行门禁

### 8.1 CPU Gate：必须包含的单测

**A. 价值数学与动作语义**

- `STOP` 即时损失对任意合法/非法 STOP 状态的**参考标签**与直接 Head CE 完全一致；非法 STOP 仍不可被 policy 采样。
- 对非 STOP `Cell/CTX` 的 `ΔQ = full_suffix_loss - immediate_stop_loss`，与短世界完整枚举 oracle 对齐（区分 Monte Carlo 容差和精确等式）。
- 完全相等的动作有 `ΔQ` 对；恒等/零更新算子的正确差值；恶化动作的 `ΔQ>0`；改善动作 `<0`。
- `raw_q - raw_q_STOP` 的 STOP 恰为 0、合法 argmin/softmax 与减常数前决策等价；已停 token 无后续动作；最大预算 forced end 不伪造 STOP。
- CTX memory 因果测试：修改未来 token，不改变此前 logits；历史信息只可通过 CTX 引入 hidden；Cell/Router 不直接读取 Memory。
- 同一隐藏状态下不同历史 Memory 的**条件价值可能不同**；这是诊断用人工构造例子，不能据此强制添加新上下文编码器。

**B. 反事实与采样统计**

- 重新规划的后缀可因 `h'` 不同而改变后续动作；不得固定旧 suffix action trace 冒充 Q。
- 基础随机数按 sample×absolute-depth 对齐，跨候选改变概率而不共享 RNG 消费状态。
- 以枚举包含概率的玩具候选集合检验 HT 加权平均期望与全合法动作平均 loss 一致；若使用 self-normalized 版本必须明确它不是无偏估计。
- 有限 `m=2,4,8` 样本方差、成对差与可靠排序判断正确；差异小于不确定性时排序标签跳过。
- 对深度桶只抽取已访问状态；在全部早停的合成输入中明确记录后层覆盖缺失而非伪造状态。

**C. 梯度独立性**

- 给 Connection 注入极大但 finite 的 Q 梯度后，连续 Cell/Head 的 CE 梯度组裁剪不受影响（比较同一输入 CE-only 基准）。
- QF 的无用 critic 不参与优化；J0 critic 正常训练；参数总量和 **trainable/active** 分开报告。
- 反事实标签产生不能反传到 Accepted；如果选择所有候选但没有正式执行，未选 Cell 不获得额外任务梯度。
- 任何非有限 accepted 梯度/参数更新：完整回滚 Model、Optimizer、Step、Registry、RNG、journal，并使该 arm 停止。

**D. Shadow 及恢复**

- `cell/connection/path` 各自一例正常 probation + Admission，恰好一次事务；至少一例合成“已知净改善”候选能走过安全晋升，另外一例强制拒绝。
- 三类候选分别注入 `nan` 或 `inf` 私有梯度：只标候选 `FAILED`，父模型模型/Optimizer/RNG/Registry/State 字节不变，其他候选与正式训练可以继续。
- Snapshot 或 Accepted 模型一致性损坏：不能被候选异常捕获后继续。
- 检查点在 `Step 400 training committed / before admission / during candidate / after admission` 四点模拟断电并恢复；绝不重复 Token 更新、Shadow Birth、候选试用或晋升。
- Python/NumPy/CPU Torch/CUDA RNG 需考虑设备正确性（历史曾有 CUDA RNG state 恢复到非 CPU byte tensor 的工程修复）；GPU Pilot 中实测跨重启轨迹等价。

**E. 工程/发布**

- 在子 Worker 完成与失败的各种组合下，自动 top-level 报告始终可生成；超大对象必须**预先设计聚合与路径白名单**，不能发布截断 JSON 伪装完整。
- 不准将裸训练样本、目标 token、模型权重或 Secrets 上传至 GitHub。顶层失败状态与每 arm 训练完成状态分开。

### 8.2 GPU Pilot 不能只测试普通梯度

**必须在真实 GPU 上测试 Shadow Cell 激活后的私有 backward**，因为上轮正是在 Step 400 的私有训练失败，而非普通主训练。建议 Pilot 顺序：

1. 单步正常 QF/J0 forward/backward/optimizer 及模型有限性；
2. 单个真实 Frozen Snapshot，至少一个 `Cell`、一个 `CTX`、一个合法 `STOP` 的后缀 CRN 标签；
3. QF RelativeQ 梯度组、J0 actor/critic 梯度组测试，记录完整 grad norm；
4. 在独立 candidate copy 上分别运行 Shadow Cell、Connection、Path 的真实私有 train step（建议至少 2 步）；验证 `finite`, `requires_grad`, accepted witness；
5. 在候选上人工注入非有限梯度，验证**确实能继续主训练**；人工注入 Accepted/Oracle 污染，验证**确实强制失败**；
6. checkpoint save/restore（同源同配置）与事务性事件恢复；
7. 测量正式路径成本、每个探针额外后缀调用量、两个 Worker 共存的峰值 VRAM、wall time、P50/P90 step time；
8. 只有内存/正确性/总运行预算都符合实际 Kaggle 配额，才启动完整两臂 500 Step。

**禁止**为了跑过 Pilot 静默把 Shadow 训练禁用、降低探针数或减少 500 Step。若必要降预算，生成新版本 Config 与新 identity，并在运行前说明变更；原 proposal 仍保留。

### 8.3 不重跑 R1 的合理性

本轮无需为确定性复现 R1 的第 400 步故障而重新运行完整的 400 步：日志已定位到 candidate private backward 的非有限梯度，但未确定首个坏算子。R2 用**新源码中的定位日志 + 故障注入 + 真实 Shadow GPU Pilot**提高防护质量；若又出现同类错误，新运行即可给出首个坏参数的可审计证据。禁止在文档中写“已经查明 R1 NaN 的根因”。

---

## 9. R2 正式实验设计与数据锁

### 9.1 两臂、共同设置与资源预算

| 项目 | 统一候选配置 | 说明 |
|---|---:|---|
| Arms | `QF-R2`, `J0-R2` | 不增加新 MoE 完整训练臂 |
| Seed | `151001201`（候选） | 与 R1 不同，两个训练臂共享初始化；训练前锁定 |
| Resident Params | 目标仍为 `1,075,492` | 如保留 Q/critic 模块并不增建 Summary，必须在审计中给出真实值 |
| Width / Heads | `128 / 4` | 沿用 R1 |
| 初始正式 Cell | `6` | 允许自连接 |
| Shadow Cell reserve | `1` | 常驻参数计入总量；未激活不得算实际有效 Cell |
| CTX | `static_input_readonly` | 相同静态因果 KV，显式 CTX_READ 动作 |
| STOP | `minimum_actions=1` | 预算末端 forced end 与主动 STOP 分开 |
| 最大非 STOP 计算数 | `16` | 真实调度与成本上限 |
| `α` | `0.125` | 不变 |
| Corpus | TinyStories，同一固定 revision | 保留 R1 输入资源定义；运行前记录实际文件 hash |
| Tokenizer | `4096` vocab | 复用相同 Tokenizer 与 Tokenizer hash |
| 训练步数 | `500` | 每臂 `256,000` 目标 Token |
| Batch×Sequence | `effective 8 × 64` | microbatch=1 |
| AdamW | 原始 3e-4 基准、warmup 50 | 连续/Connection 分组梯度裁剪；每组 LR 分别记录 |
| 计算惩罚 | `λ = 0.01`（候选冻结） | `calls/16`；不是实际 FLOPs |
| ε 时间表 | 0.40 / 0.25 / 0.15 | QF/J0 两臂一致 |
| Train probing | 最多每步 2 状态 | 已访问状态分层，目标预算上限锁定 |
| 候选非 STOP 附加动作 | 最多 2 | 实际动作必选，STOP 作为直接基准 |
| Suffix samples | `2→4→8` | 同探针对齐 CRN；总预算硬上限 |
| Shadow | Step 100/200 propose，400 admission | 候选最大 3；每候选 private_train_steps=2 |
| GPU | Kaggle T4 x2（候选） | 每臂独占 GPU，Pilot 必须实测 |
| 运行精度 | Float32 | 与 R1 对齐，便于数值诊断 |

推荐保持 R1 分割（以仓库配置为权威，不根据输出结果更改）：

```json
{
  "dataset": "roneneldan/TinyStories",
  "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
  "train_range": [0, 8192],
  "calibration_range": [50000, 50128],
  "admission_range": [50128, 50256],
  "development_range": [50256, 50512],
  "reserved_boundary": 60000,
  "tokenizer_vocab": 4096
}
```

**注意**：以上 `range` 为原始 story 角色区间，不代表每一轮评估都遍历该范围中的全部样本。R2 要明示从这些范围产生评估窗口的冻结 seed、窗口长度、分组 story ID，并在结果中报告精确窗口数及其 hash。训练资产可以跨轮共享，但绝不能由新候选伪装成旧任务相同 checkpoint；数据、tokenizer 的内容 SHA 应单独冻结。

### 9.2 三个不同的效果视角，避免 Shadow 晋升干扰机制结论

为避免“第 400 步一方接纳新 Cell，另一方未接纳，最后比较错把容量差异当作 Q 优势”，设定：

1. **主要方法对照（Pre-admission）**：Step 400 **主训练提交后、Shadow Admission 之前**，在相同的 6 accepted Cells、相同 token 预算下评估 QF-R2 与 J0-R2。这是本轮较纯粹的 Q 学习 vs J0 学习方法对照。这里的 Development 结果严格只读，不参与 Admission 与 LR/β/ε 选择。
2. **完整体系结局（Post-lifecycle）**：Step 500 最终评估。此时两臂可能因不同 Shadow 接纳结果而拥有不同**活跃计算容量**；可报告完整体系优劣，但必须记录真实活跃参数与候选差异，不能假装是纯学习损失单变量比较。
3. **方法稳定性诊断**：可在 Calibration 上按 Step 100/200/300/400/500 记录 Q ranking、Regret 与 STOP；这些诊断不允许读取或更新 Development/Confirmation 参数。

**禁止把 Step 400 与 Step 500 指标混在一起给出单一“QF 胜负”。** 主对照固定在 Pre-admission，长期系统状态另报。

### 9.3 主要评价策略

两臂都至少报告：

- `greedy`：严格合法的最小预测成本决策（J0 为其 learned score 的最优动作），含统一 tie rule；
- `sampled`：预冻结 ε/τ 下真实采样；
- `soft`：无 uniform mixture 的 softmax；
- `uniform`：只在合法动作中均匀采样；
- `constant`：同一个预定义动作优先、合法时回退的冻结规则；
- `context_zero`、`state_permuted`、`previous_permuted`、`cell_swapped`：机制诊断。须报告具体干预在动作选择前还是执行阶段生效，且按**重新执行的最终路径**计 NLL；不能把只修改中间 Tensor 的旧结果当最终推理效果。

输出**独立** `mean_nll`, `ppl`, `mean_cost`, `mean_actions`, `stop_rate`, `forced_end_rate`, `ctx_read_fraction`, `self_loop_fraction`, `train_wall`, `oracle_wall`, `shadow_wall`, `synchronized_inference_latency`, `estimated_FLOPs`，区分 host enqueue timing 与 GPU synchronized timing。全部指标按 `same windows + same role` 可配对比较；不同模型抽样轨迹可用同一种子 CRN 但不能声称路径相同。

### 9.4 Q 专项评估：必须比“Q loss 降低”更严格

R2 建议将 Calibration 的反事实 Q 状态数由 R1 的 `8` 提至 `24`（在 Pilot 可接受的预算内冻结），每个状态对**所有合法非 STOP 动作**取 `m=8` 统一 CRN 后缀。若预算不够，减少**独立状态数**但须在锁文件中明确，不能删掉部分动作并声称完整 Q-ranking。建议记录：

- `pred_delta` 与 `oracle_delta` 的 MAE、Bias、分 Cell/CTX/STOP 类型统计；
- 在动作价值可识别状态中的 Spearman 与 pairwise order accuracy；
- `Qf policy regret`：相同快照下，预测最优动作相对于充分采样 Oracle 最优动作的平均成本差；并与 Uniform/Constant 排序策略比；
- Oracle 最优与次优动作的 `best_gap` 及配对差不确定性，低于分辨率的状态列为 `LOW_ACTION_IDENTIFIABILITY`；
- `STOP-too-early/too-late`，仅在 STOP **合法**且可以统计区分的状态上计算；
- CTX 是否有可测正向效用以及前后路径是否确实需要再次调用 CTX；
- 路由偏好是否只与 Cell ID、previous action、depth 相关，而不随隐藏状态有效变化；
- 训练候选的相对值拟合误差与独立 Calibration 的误差**分别**展示；不得只报告训练 loss。

**对 J0 的 Q regret 指标**也应由它的独立真实路径 Oracle 评估，但不能将两个不同模型/不同后缀策略的绝对 Q 数字混为同一个函数做数值泛化比较。比较更稳妥的是各自**相对同模型 Uniform** 的 regret 改善，以及 paired Development `NLL + λC`。

### 9.5 初始状态与 CTX 的专项诊断（禁止补充免费上下文）

R2 增加一个冻结的 Context-Need Probe：在同一 token/相同位置但不同合法前缀（保证构造样例确实只在历史信息上不同）检查，尚未执行 CTX 前 `h`/Connection 观察是否相同或近似相同，而 `CTX_READ` 真实效用是否不同。进一步检查 **完成一次 CTX 后**，Connection 的选择能否随新状态区分。

指标可包括：`first_ctx_depth`, `ctx_before_stop`, `ctx_seen_at_least_once`, `ctx_value_delta_before_first_ctx`, `ctx_value_delta_after_ctx`, `probe_state_alias_count`。必须保证人工探针遵守原始上下文长度和目标构造，并与正式 TinyStories 评估分开。若该问题成立，报告为 **PARTIAL_OBSERVABILITY_LIMITATION**，它不等于 Q 算法错误，也不允许在 R2 中私自加固定 Attention 作弊。

### 9.6 功能分化只能通过条件因果价值验证

仅据 `Cell ID`、参数范数、相异权重、使用频率或 Router entropy，**不得宣称“专业化已出现”**。支持性诊断：

- 针对各 Cell 的 `ΔQ(s,Cell_i)` 在不同状态族上的条件分布；
- 真实候选替换改变最终 NLL 的方向、幅度及置信区间；
- 符合收益排序的 Cell 调用是否增加、是否仍有未被调用但有价值的 Cell；
- Context 参与前后，Cell 的条件收益是否有显著改变；
- 由相同初始种子但不同训练方式形成的分化仅作探索性比较。

**本轮不增加对 Shadow 路径的任务梯度回传**，仅 Shadow 候选在私有 Probation 内可训练；否则相对 Q 方法、额外可微候选训练、收益晋升三项同时改变，无法识别改进来源。这部分作为 R3 可能的研究扩展记录。

---

## 10. 判定指标、统计协议与失败状态

### 10.1 预冻结目标与指标优先级

采用 `U = NLL + λ×mean_cost` 作为 **primary cost-aware objective**，并与 NLL 原始值并列报告，避免通过更短路径但质量严重恶化取得假胜利。

| 领域 | Primary / Gate | Secondary |
|---|---|---|
| 数值安全 | Accepted Oracle+Checkpoint 一致性 PASS | Candidate-local FAILED 比例 |
| Q 学习 | 独立可识别状态的 rank/regret 相对于随机排序有改善 | Huber MAE、Bias、gap 分布 |
| 路由效果 | Step 400 paired `ΔU = U_QF−U_J0`、QF vs self-Uniform | Greedy/sample/soft 策略差异 |
| STOP | STOP 检测误差、与动态计算预算的实际收益 | stop-depth 分布、forced-end rate |
| Context | 非零有效读取与干预响应/效用归因 | CTX 访问频率、首读深度 |
| Shadow | 独立 Admission + 候选隔离 + 事务回滚 | 提议/失败/拒绝/晋升计数 |
| 计算 | 训练真实 Token + 探索调用 + GPU wall 全账本 | latency/FLOPs/peak memory |

**建议先冻结成以下开发信号标准**（均为开发探索性的可操作门槛，不等于多种子科学确认）：

- 基础门禁：CPU/真实 GPU Pilot 通过；两臂 Step 400 Pre-admission 和 Step 500 都有完整报告；所有 Accepted 模型参数 finite 且结果身份一致。
- Q 排序信号：Calibration 在 `identifiable` 的状态中，平均 Spearman >0；同时 Q 选择的 average regret **严格低于同模型的 Uniform 选择平均 regret**；按 story bootstrap 的 95% 区间用于辅助评估（若区间跨 0，报告 `WEAK_OR_INCONCLUSIVE_Q_SIGNAL`，不宣称可靠优势）。
- 实际路径优势：Step 400 `U_QF−U_J0` 在 paired story bootstrap 中 `upper95 < 0`，且 `U_QF−U_uniform(QF)` 也 `upper95 < 0`；如未满足，记 `Q_SIGNAL_NO_POLICY_GAIN` 或 `NO_Q_ADVANTAGE`，不以 Q Loss 下降代替。
- STOP：没有不解释的全 STOP/全 forced end 退化；必须与 Oracle 的 STOP 最优判断相符、至少取得可识别状态上的决策收益。仅有 STOP 使用率>0 **不算通过**。
- Shadow：有完整候选生命周期及父模型未污染的证据；允许零晋升，但此时不得宣称“能力增长成功”。若候选发生受控失败，主模型可完成，但状态必须附带 `SHADOW_PARTIAL_FAILURE`。

**若希望用固定数值门槛批准扩展预算**，建议在启动前再冻结 `min_identifiable_states`, `max_stop_error_rate`, `min_q_regret_gain`, `max_diagnostic_missing_fraction` 等具体门槛。对 24 个 Calibration 状态强设 0.6/0.7 的排序率没有充分统计基础；宁可明确探索性 inconclusive，也不能为了产生 PASS 字段随意预设虚高/虚低门槛。

### 10.2 统计计算

- 训练 CE 是在线目标，只看趋势，**不应用**来代替 Development NLL 的泛化对照。
- 同角色固定故事分组，对每个 story 的均值应用 bootstrap，至少 2000 次重复，提供 `paired ΔNLL`, `paired Δcost`, `paired ΔU` 的均值与双侧 95% CI；比较时对方差与差值范围做 sanity check。
- 非常小的数值差异要报告实际效应量；不因单一偶然分组得到微弱负数就宣布明显优势。
- 反事实 rank/regret 的 Oracle 本身是有限 MC 估计，应提供 MC 配对误差与可识别状态比例；把 Oracle 的估计噪声误当 Q 泛化失败同样不科学。
- 本轮只有 1 个开发 seed，**故事级置信区间不等于跨种子重复性**；任何“成功”均标记 `EXPLORATORY_DEV_SIGNAL`，后续多 seed confirmation 另行注册。
- 中途 checkpoint、Calibration 与 Admission 不允许读取未触及的正式 Confirmation。没有显式审批，不能自动解封 Confirmation。

### 10.3 状态枚举，不得将异常隐藏在 COMPLETED 中

建议包含以下独立字段：

```text
training_status:       COMPLETE | INCOMPLETE | FAILED
oracle_status:         TRUSTED | UNTRUSTED | UNIDENTIFIABLE
q_signal_status:       POSITIVE_DEV | WEAK_DEV | NO_SIGNAL | UNREVIEWED
policy_status:         ADVANTAGE_DEV | NO_GAIN | STOP_COLLAPSE | UNREVIEWED
shadow_status:         ALL_EVALUATED | PARTIAL_FAILURE | ALL_REJECTED | PROMOTED | NOT_RUN
publication_status:    PUSHED | LOCAL_ONLY | FAILED
scientific_decision:   false
formal_confirmation:   false
```

主运行 `training_status=COMPLETE` 且 `shadow_status=PARTIAL_FAILURE` **可以同时成立**；但 `integrated_system_fully_validated=false`。如果只是 J0 完成而 QF 失败，输出 `PARTIAL` 结果并保留 J0 数据，不能将整个报告只浓缩成一行失败 JSON。

---

## 11. 结果产物与 GitHub 回传要求

### 11.1 Kaggle Output：完整可恢复资料

建议目录（示意）：

```text
CLM-001-R2/<source12>/<run-id>/
  identity.json
  r2.lock.json                       # 冻结 source/config/assets/protocol hashes
  asset-manifest.json
  runtime-state.json
  preflight/
    cpu-gate.json
    gpu-pilot.json
    cost-projection.json
  arms/
    QF-R2/
      latest.json
      checkpoint-step-000100.pt
      checkpoint-step-000200.pt
      checkpoint-step-000300.pt
      checkpoint-step-000400.pt
      checkpoint-step-000500.pt
      metrics.json
      pre-admission-eval.json
      q-calibration.json
      candidate-events.json
      diagnostics/
    J0-R2/
      ...
  combined-report.json
  failure-index.json
  portable-manifest.json
  publication-state.json
```

Checkpoint 的保存间隔可保持 50 Step；上表仅示例重要节点，**R2 必须保证 Step 400 的 Pre-Admission Checkpoint 是独立已提交文件**。任何文件大小限制不应通过静默删除重要科学字段“解决”；分拆详细原始记录并通过索引/hash 链接。

### 11.2 GitHub：最小完整的聚合结果

由于 GitHub 只应接收允许发布的无敏感聚合物，本轮新增**安全可脱敏**的白名单，至少包含：

```text
run-summary.json
arm-QF-R2-metrics-summary.json
arm-J0-R2-metrics-summary.json
pre-admission-contrast.json
final-system-contrast.json
q-calibration-summary.json
stop-ctx-path-summary.json
shadow-candidate-summary.json
failure-index.json
resource-ledger.json
publication.json
```

每个 arm 在完成指定结果阶段时可以独立写入本地汇总。顶层 aggregator **无论子进程退出码是否为 0**，都应尝试从现存且 hash 可验证的证据生成 `combined-report`；如果 GitHub 授权有效，失败时仍按已发布白名单推送聚合文件。每个文件的 `complete`, `source_sha`, `config_hash`, `arm`, `last_committed_step`, `scope` 必须自描述。

**发布失败不能引起再次训练**；`publish-only`/`retry-publish` 应为可幂等操作，前提是同 run-id、同不可变汇总 hash。任何部分失败字段必须原样保留，不能在重试时把 `FAILED` 变成 `COMPLETED`。

### 11.3 日志最小汇总行

正常训练行建议：

```json
{
  "event": "train_step",
  "arm": "QF-R2",
  "step": 123,
  "ce_live": 5.75,
  "loss_rel": 0.14,
  "loss_rank": 0.02,
  "rank_pairs_valid": 2,
  "mean_actions": 8.5,
  "ctx_fraction": 0.13,
  "stop_rate": 0.42,
  "forced_end_rate": 0.12,
  "grad_norm_continuous_preclip": 0.85,
  "grad_norm_connection_preclip": 2.10,
  "oracle_states": 2,
  "oracle_suffix_calls": 85,
  "oracle_seconds": 0.11,
  "source_model_sha": "..."
}
```

以上都是**字段与虚构示例数值**，不得作为真实运行结果直接写入正式报告。候选失败行另记 `candidate_type/private_step/failure_code/first_nonfinite_param/accepted_witness_ok/next_action`，保证后续能在没有云端 checkpoint 的情况下定位问题。

### 11.4 完整资源账本

报告至少区分：

- `live_training_tokens`: `500×8×64 = 256000` / arm；
- `shadow_private_training_tokens`：单列，不与主训练 Token 混合冒充更多数据效率；
- `counterfactual_operator_calls` 和 `counterfactual_token_steps`；
- accepted forward/backward 逻辑 GEMM FLOPs、CTX attention FLOPs、Shadow 子图耗时；
- 每 arm wall time、两 GPU 并行的 makespan、CPU preflight/GPU Pilot/HF 下载/发布时间；
- 同步 GPU 测量与 host enqueue 计时不可混报；
- resident/active/trainable 参数、GPU 峰值、模型推理延迟（含 STOP 实际动作次数与 Controller 额外调用）。

---

## 12. 推荐施工顺序：不要求重跑 R1，但不得跳过验证

### Phase A — 仓库现状与 R1 只读基线审计（无 GPU）

- 获取当前 `ArcheLabs/mini-cells` 指定目标分支最新 SHA 与 R1 基线 commit `8b7cc534...` 对比，整理已修复问题、可复用代码和任何 merge 风险；
- 确认历史 FAST-001 M0 每层 Attention、R1 CLM 静态输入 Memory/可选 CTX、R1 失败点与 GitHub 不完整汇总；
- 创建独立 R2 Protocol、运行 identity、入口与配置（初期 `status=PROPOSED`）；
- **Gate A**：源码路径、角色数据、固定信息通道、状态机及资源预算审计完成。

### Phase B — Relative-Q 与样本统计的 CPU 工程实现

- 先写精确 STOP 基准/差值 Oracle，再实现模型 RelativeQ 输出和 masked action；
- 实现 candidate inclusion、从真实访问状态选 probe、CRN 配对及分层覆盖计数；
- 实现 Huber + 可靠排序对、无效 pair 跳过、按组 clip 及 QF critic 取消；
- 用枚举小世界证明数值和方向正确；
- **Gate B**：CPU 全部 PASS，旧 R1 入口行为保持不变。

### Phase C — Shadow 故障保护与事务恢复

- Candidate typed failure、首坏梯度定位、独立 RNG、Accepted witness 与原子回滚；
- 训练 Step commit 与 Admission 事件 exactly-once；
- 三类 Shadow 的正常路径及异常注入；
- 两臂部分结果聚合与白名单发布；
- **Gate C**：正常及异常各类路径通过，尤其是 `Step 400` 事务模拟测试。

### Phase D — GPU Pilot 与小样本真实 Smoke

- 同源同锁配置，T4 x2 设备检测；真实 QF/J0 前向与反向、冻结快照 Oracle、Shadow 三类候选真实私有训练；
- 真实 Checkpoint/CUDA RNG 恢复、最大峰值显存与后缀预算压力检查；
- 如 GPU 预算不符合，停止并生成**版本化的**新建议配置，不减少已冻结正式实验的训练 Step；
- **Gate D**：输出 `GPU_PILOT_PASS`；否则不得启动完整训练。

### Phase E — 新运行 ID 的 500 Step 双臂开发训练

- `QF-R2` 与 `J0-R2` 分 GPU 并行；维护独立日志和 checkpoint；
- Step 400 **先持久化主训练和 pre-admission 评估**，再执行 Shadow Admission 事务；
- Shadow 候选级有限数值故障可失败隔离；Accepted/Oracle/持久化故障必须停训；
- Step 500 后各臂独立评估；若其中一臂失败，报告部分结果而非丢弃另一臂；
- **Gate E**：完成并输出 `DEVELOPMENT_COMPLETE` 或带有准确子状态的 `PARTIAL`。

### Phase F — 研究报告与后续 Go/No-Go

- 统计 Pre-admission 方法对照与 Post-lifecycle 完整系统表现，严格区分；
- 提供 Q ranking/regret、STOP、CTX、Cell 条件效用、Shadow、成本及故障证据；
- 标准决策：`QF_DEV_SIGNAL / Q_SIGNAL_NO_POLICY_GAIN / Q_UNIDENTIFIABLE / STOP_COLLAPSE / ENGINEERING_INCOMPLETE`；
- 根据结果确定是否需要专门研究 Context State Aliasing、候选路径辅助 Cell 梯度、目标网络/更新频率或更大数据规模；
- 即使最优表现出现，也只得出**开发级观察**，后续新 seed 的 confirmation 需单独批准。

---

## 13. 交给 Codex 的操作指令（可直接使用）

> 请在 `ArcheLabs/mini-cells` 中按照附件《CLM-001-R2 — Stabilized Relative Q-First Cellular Learning》实施**第二轮独立实验**。
>
> **先读取并审计** R1 commit `8b7cc5344d0b35f58681286101e93e39ff1fc41d` 的 `clm-001-qfirst-integrated/implementation/clm001/`、R1 `config/proposed.json`、历史 006 M0 MoE/PDCLM 的上下文实现，以及本协议关于失败处理和统计声明的门禁。先输出 R1→R2 的差异列表，特别是 QF 用绝对 cost-to-go、共用梯度裁剪、无用 critic 梯度、随机 depth 抽样、Shadow 私有梯度非有限、Step 400 事务与部分报告未上传等问题。
>
> **实现范围**：R2 从 STOP 锚定的 RelativeQ 开始，增加配对候选排序、真实访问状态分层探针、明确包含概率、受限 CRN 后缀自适应采样、两组独立梯度 clip、固定共同 ε 时间表、显式 STOP/CTX 因果测试、Shadow typed failure 与原子事件恢复、容错双臂聚合回传。
>
> **不得改变**：Cell 自身无直接跨 token 读取、Connection 无额外免费 Context Summary、CTX_READ 显式动作、Cell 自连接合法、STOP 唯一退出机制、B=16、同构 QF/J0 对照、R1 原有数据分割与 Token 预算、Shadow 未接受前不进入正式合法动作集；不要增加新的 MoE 全训练臂，也不要先预训练固定功能 Cell。
>
> **工程优先级**：先通过精确短世界 Oracle 与 CPU 属性测试，再通过真实 GPU Shadow 三类候选的 forward/backward Smoke、故障注入和 Step 400 checkpoint/restart 事务测试。任何 Candidate-private 故障均须有父模型 witness；Accepted/Oracle/Checkpoint 根故障仍然 fail-closed。只发布白名单的脱敏元数据，不发布 Secrets/原始目标/参数。
>
> **结果**：实现完成时提供源分支/commit、修改文件、pytest 数、GPU Pilot 测量、冻结配置 hash、运行方法和预期 GitHub 结果路径。没有实际执行 500-step 就明确标记 `NOT_TRAINED`；任何执行中的部分失败须将两臂已有证据归档。不得编造优于 J0/MoE、Cell 专业化或 Shadow 成功晋升结论。
>
> **工作模式**：可以直接实现、测试与提交源码。启动完整 Kaggle 500-step 开发训练前，核验本次用户指令是否明确授权对应 GPU/Secrets 使用；如未授权，停在 Pilot 与冻结配置，提供可执行单命令入口和测量报告，不要擅自启动长任务。

---

## 附录 A：推荐冻结配置的增量字段示例

下列为建议配置片段，**尚未经过运行验证，也不是仓库现有文件**。Codex 应与 R1 config schema 合并，完整实现 `validate()`，并生成 hash/lock。所有参数与 RunID 在 GPU Pilot 审查后冻结。

```json
{
  "experiment_id": "CLM-001-R2",
  "arms": ["QF-R2", "J0-R2"],
  "seed": 151001201,
  "model": {
    "width": 128,
    "heads": 4,
    "accepted_cells": 6,
    "reserve_slots": 1,
    "budget": 16,
    "minimum_actions": 1,
    "memory_mode": "static_input_readonly",
    "connection_extra_context_summary": false,
    "cell_direct_memory_access": false
  },
  "policy": {
    "value_convention": "relative_cost_to_stop_lower_is_better",
    "lambda_cost": 0.01,
    "temperature": 1.0,
    "epsilon_schedule": [[1, 0.40], [101, 0.25], [301, 0.15]],
    "stop_tie_rule": "prefer_STOP_when_legal"
  },
  "oracle": {
    "probe_source": "stratified_actually_visited",
    "probes_per_step_max": 2,
    "extra_actions": 2,
    "suffix_samples_initial": 2,
    "suffix_samples_allowed": [2, 4, 8],
    "crn": "absolute_depth_inverse_cdf",
    "label": "cost_to_go_minus_immediate_stop",
    "regression_weighting": "horvitz_thompson_inclusion",
    "pairwise_min_samples": 4
  },
  "training": {
    "steps": 500,
    "effective_batch": 8,
    "sequence_length": 64,
    "target_tokens": 256000,
    "learning_rate_continuous": 0.0003,
    "learning_rate_connection": 0.0003,
    "gradient_clip": "per_parameter_group",
    "gradient_clip_continuous": 1.0,
    "gradient_clip_connection": 1.0,
    "relative_q_huber": true,
    "rank_loss_weight": 0.1,
    "qf_auxiliary_critic_training": false,
    "j0_actor_critic_training": true
  },
  "shadow": {
    "max_cell_births": 1,
    "max_connection_path_groups": 1,
    "cell_proposal_step": 100,
    "connection_path_step": 200,
    "admission_step": 400,
    "private_train_steps": 2,
    "candidate_failure_isolation": true,
    "admission_requires_precommitted_checkpoint": true,
    "minimum_admission_stories": 32,
    "min_cost_aware_improvement": 0.005,
    "max_upper95_nll_regression": 0.0
  },
  "evaluation": {
    "pre_admission_step": 400,
    "final_step": 500,
    "q_oracle_calibration_states": 24,
    "q_oracle_suffix_samples": 8,
    "bootstrap_replicates": 2000,
    "never_train_on_development_or_confirmation": true
  },
  "publication": {
    "partial_arm_results": true,
    "whitelist_only": true,
    "publish_weights": false,
    "publish_raw_oracle_targets": false
  },
  "scientific_decision": false
}
```

**需要 Codex 补全且不得猜测为正式冻结**：R1 其余配置的具体常数、候选探针预算上限、分层抽样概率、可靠偏好对的数值容差/置信阈值、模型恰当的初始化、真实 Pilot 性能阈值、Admission 事件 replay 策略及 GitHub 路径命名。`max_upper95_nll_regression` 为 R2 新配置字段，须正确映射到现有 `admission.py` 语义，不能混用 `nll_regression_tolerance` 的含义。所有补全数值必须进入锁定配置并作为最终结果 identity 的一部分。

---

## 附录 B：验收检查清单

```text
[ ] 不覆写 R1 源码、结果、已有 checkpoint
[ ] QF/J0 状态信息完全一致，无免费 Context Summary
[ ] CTX_READ 是合法可选动作；Cell 自连接合法；STOP 无双重退出系统
[ ] STOP Baseline exact；相对 Q 与原始成本符号正确
[ ] 真实访问状态抽样，无停后“伪状态”
[ ] 候选包含概率正确；HT 与 pairwise 采样契约经小世界验证
[ ] Short-world Oracle 枚举对齐、完整后缀重新规划、CRN 真实配对
[ ] QF/Cell 分组梯度裁剪；无用 QF critic 梯度不干扰
[ ] Shadow 三类候选在真实 GPU 上均经过 backward
[ ] Shadow candidate-local 错误不会破坏 Accepted 或整个训练
[ ] Accepted/Oracle/Checkpoint 错误仍 fail-closed
[ ] 第 400 步主更新先提交 checkpoint；Admission 事务 exactly-once
[ ] 恢复时 Token 顺序、模型/优化器/RNG 与版本一致
[ ] R2 500-step 数据角色/Seed/Hash/资源预算冻结
[ ] Step 400 pre-admission 与 Step 500 post-lifecycle 分开评价
[ ] Q 的 held-out 排序/Regret 与 STOP/CTX 专项诊断齐全
[ ] Shadow 数量/失败/拒绝/晋升/回滚真实计数
[ ] 任意单臂失败保留另一臂的完整可发布摘要
[ ] GitHub 只回传脱敏汇总，不泄露权重/Secret/目标标签
[ ] 结果 `scientific_decision=false`，不误称 MoE 对照胜利
```

---

## 附录 C：参考源码和证据定位

- R1 源码基线：`ArcheLabs/mini-cells@8b7cc5344d0b35f58681286101e93e39ff1fc41d`；`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/`。
- R1 Kaggle log：本会话附件 `minicells (1).log`，失败部分在文件内容约第 **809–831** 行；完整文件约 7,633 行。
- R1 GitHub 已回传结果分支：`results/clm001-qfirst-8b7cc5344d0b-a0b43988`；`report.json` 为 `INCOMPLETE`。
- 先前设计依据：本会话附件 `CLM_001_QFirst_Integrated_Training_Protocol.md`（R1 的理论/实施协议）；R2 只用于新增实验，不倒写旧协议。
- MoE 上下文访问的源码依据：`research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/baseline.py` 的 `Block.forward`；CLM 的静态输入 Memory：`exp006/memory.py`，真实显式 CTX：`clm001/model.py` 与 `rollout.py`。
- 特别提醒：R2 的全部改进目前只是**已知问题驱动的理论/工程方案**；未经新实验数据，不得写成已验证有效。

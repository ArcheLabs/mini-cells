# CLM-001 — Q-First Integrated Cellular Learning
## 基于候选计算价值函数的统一 CLM 理论、实验协议与 Codex 实施规范

**版本**：v1.0 / 2026-10-15  
**性质**：理论与开发实验提案；**不是已经验证的科学结论**  
**研究项目**：MiniCells / Cellular Language Models  
**仓库**：`ArcheLabs/mini-cells`  
**建议新目录**：`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/`  
**实验标识**：`CLM-001-QFIRST`（与 `006-PD1-FAST-001` 完全隔离）  
**当前授权范围**：编写代码、静态审计、CPU 测试、合成任务正确性检查、性能/资产预检。**本文不代表用户已授权真实 GPU 训练、正式 confirmation、覆盖旧结果、自动长期监控或发布能力声明。** 后续实施与 GPU 运行须分别经用户明确批准。

> **核心问题**：从随机初始化开始训练的 CLM，能否在一个统一系统中学习候选计算的条件价值，用该价值选择 Cell、自连接、跨 Cell 连接和 STOP，并通过隔离的 Shadow 候选探索形成有实际收益的计算结构？  
> **第一轮成功标准**不是超过 MoE，而是：候选价值标签可信、Connection 的价值排序具有独立预测力、STOP 体现质量—成本权衡、实际路径优于内部无效价值对照、候选结构的接纳安全且可验证。

---

## 0. 结论先行：唯一的设计选择

1. **Q-first**：Connection 的首要可学习对象是 `Q(s,a)`——在当前计算状态下选择候选动作后，按明确的后续策略完成计算的**预期最终损失加计算成本**；**值越小越好**。本协议使用 *cost-to-go* 约定，不混用“回报越大越好”的 Q 符号。
2. **统一动作空间**：`{Cell_0 … Cell_{K-1}, CTX_READ, STOP}`。自连接是再次选择同一个 Cell；跨连接是选其他 Cell；单个 Cell 不再配专用 Exit Controller。
3. **Shadow 是候选的治理状态，不是另一套计算规则**：Shadow Cell 是候选算子，Shadow Connection 是候选条件转移，Shadow Path 是多步候选计划；共同使用反事实评价与版本化接纳框架，但有不同参数和执行语义。
4. **连续计算与离散学习分工**：正式执行路径上的 Cell/Context/Head 由真实任务 CE 反向传播；Connection 的 Q 头由同权重快照上生成的、可审计的候选损失标签监督学习。训练时可用答案形成监督标签，**真实路由选择及推理输入绝不直接读取答案**。
5. **STOP 与其他动作公平比较**：STOP 立即输出、后续成本为 0；继续执行须承担路径成本。达到最大预算时的强制结束不是模型主动选择 STOP，不计为 STOP 学习成功。
6. **承认旧研究失败边界**：尤其是 Experiment 019 的 `UTILITY_ORACLE_INCONSISTENT`、历史的 Cell/Connection 耦合、读地址干扰与 Shadow 隔离尚未证明端到端语言收益。**先验证价值标签，再相信价值学习。**
7. **不以 MoE 为首轮主要胜负指标**。首轮采用一个完整 CLM 与一个**同结构、同预算、相同候选和生命周期协议**的内部学习方法对照；MoE 可作后续独立外部质量锚点。

---

## 1. 证据基础、已有结论及其边界

### 1.1 FAST-001 的实测基线

本次协议基于已上传的 `FAST-001 report.json`（source SHA `1d393e7622821433712841c7ba64d7de689e7dc4`、开发 seed `151000101`），不从这些结果自动推导普遍定律：

| Arm | Params | 500-Step 训练 Token | 最终 dev greedy NLL | sampled NLL | 实际训练秒 |
|---|---:|---:|---:|---:|---:|
| `M0-TINY` | 2,013,440 | 256,000 | 4.4833 | 4.4833 | 128.57 |
| `CLM-S8` | 1,011,000 | 256,000 | 4.7045 | 4.6193 | 373.36 |
| `CLM-S16` | 1,011,000 | 256,000 | 约 4.7243 | 约 4.6209 | 约 735 |

结论仅限：约 1M 参数 CLM 能降低 NLL；增加固定路径长度未观察到收益；Greedy 与随机路径之间存在明显质量差；不能用训练能力证明连接策略已经学会有效计算。报告属于 `EXPLORATORY_ONLY`、一个 seed，不能宣称正式复现或统计普遍性。

### 1.2 应复用而非重做的历史工作

| 研究线索 | 与本协议的关系 | 边界 |
|---|---|---|
| Constructive CLM-004/005 | 受控任务中多 Cell 组合和可学习控制存在性证据 | 不可外推为自然语言大模型成功 |
| Growing CLM / Experiment 016 | 连接、剪枝、分裂、结构因果测量与生命周期原语 | 语言保持未通过整体要求 |
| Experiment 019 | 条件候选边际效用、反事实/有限差分 Oracle 及负例设计 | 已有 `UTILITY_ORACLE_INCONSISTENT`，不得沿用未核验目标 |
| CLM-0.3c | 反事实增生/分裂判定的经验与失败 | 没有验证出可靠分裂效用信号 |
| Shadow Cell Validation 001/v2 | copy-on-write、条件表达、隔离候选、版本接纳 | 控制实验并非本次完整动态语言模型的成功先例 |
| 006 / PD-001 / FAST-001 | 真实硬路径、同 Cell 重复执行、因果读、原子检查点、light 性能模式 | 原 J0 信用分配并未形成可靠路径优势 |

**工程约束**：旧已冻结协议、已有 decision、种子、checkpoint、报告只读；新实验独立 source/config/data/seed/run identity。历史 `CLM_CAPABILITY_CEILING.md` 的 No-Go 声明仍然适用。

### 1.3 科学命题的三个层次

- **数学定义成立**：我们能定义完整、因果的计算过程和预期成本，并给 Q 提供可测、受控的训练标签。这在工程正确性 Gate 中检验。
- **训练机制有效**：Q 从候选反事实中学到可泛化的相对排序，训练后的路径与 STOP 优于内部对照。这需要实验，不能由数学定义直接保证。
- **长期系统目标**：功能分化、可靠结构增生、参数效率和持续学习。这些不是第一轮开发实验可以确认的结论。

---

## 2. 为什么 Q-first 可能打破 Cell–Connection 耦合

联合训练的依赖关系是：Cell 的训练状态分布由 Connection 决定；Connection 能否学会偏好又取决于 Cell 的相对效用。若初始路径近均匀且不同 Cell 获得相似经验，系统可能停留在“不同动作价值近似相等”的低分化状态。

只使用单条随机轨迹的末端 NLL 来更新每一步动作，会有较大的信用分配噪声。Q-first 的改变**不是人为要求 Cell 不同**，而是让 Connection 先通过**同一状态、替换一个候选动作、完整重算后续路径**得到更接近动作特异的监督。

**重要限制**：若不同 Cell 在目标状态中真的功能相同，则所有有效候选的真实 Q 仍近乎一致。Q-first 不能凭空创造功能差异。它能提供更可靠的**发现差异、保留有用差异和按差异重新分配训练经验**的机制；能否自发形成持续、可泛化的专业化，必须另由因果干预检验。

---

## 3. 统一模型的形式化定义

### 3.1 每个 token 的因果状态

给定输入前缀 `x_≤n`、需要预测的 `y_n=x_{n+1}`：

\[
 z_n=E(x_n)+P(n),\quad M_{\le n}=\mathrm{ImmutableKV}(z_{\le n}),\quad h_{n,0}=\mathrm{Init}(z_n).
\]

`M_≤n` 只从输入 token 产生，**绝不可读取未来 token 或目标标签**。除显式 `CTX_READ` 外，Cell 不得偷用后续 token。第一轮不插入大型固定 Transformer 骨干。允许评估强制 Context Read 的机制诊断，但不能默默加入免费上下文操作。

某一步的充分决策状态建议为：

\[
 s_t=(h_t,a_{t-1},t,\, B-t,\, m_t),
\]

其中 `B` 是最大计算预算，`m_t` 是动作合法性掩码/候选版本。必要的上下文摘要可由隐藏状态体现，不允许让 Q 额外访问目标 y 或未来 `x`。如果状态不足以预测效用，这是待测缺陷而非可忽略实现细节。

### 3.2 动作空间与真实执行

\[
 \mathcal A(s_t)=\{C_0,\dots,C_{K-1},\mathrm{CTX\_READ},\mathrm{STOP}\}\cap\mathrm{legal}(s_t).
\]

若 `a_t=C_i`：

\[
 h_{t+1}=h_t+\alpha C_i(\mathrm{LN}(h_t)).
\]

若 `a_t=CTX_READ`：

\[
 h_{t+1}=h_t+\alpha\mathrm{ContextRead}(\mathrm{LN}(h_t),M_{\le n}).
\]

若 `a_t=STOP`：直接输出 `logits=Head(h_t)`，不执行 Cell，不产生伪后续动作。重复选择同一 Cell 是真实的下一次计算；每次执行都计入计算成本。

- **STOP 合法性**：默认至少执行 1 个真实非 STOP 动作后才能主动 STOP（避免未训练的退化空路径）；此最小值是固定预注册的实验控制，而不是单独 Cell Exit 机制。
- **预算 B=16**：达到预算后 `FORCED_TERMINATION`，执行 Head；日志必须与主动 STOP 分开统计。
- `CTX_READ` 必须计入真实路径时间和逻辑计算量；禁止把它设置为免费 Transformer 后处理。
- 路由在当前 token 的每一步执行；输出 logits 对上下文未来变化保持因果不变；缓存 decode 与 full forward 在固定动作 replay 下必须对齐。

### 3.3 统一候选集合并不意味着相同生命周期

- **Accepted action**：正式模型可选的已接受 Cell/CTX_READ/STOP，按 Q 策略调用。
- **Shadow Cell**：预留的或 copy-on-write 创建的候选算子；在接受前不进入正式推理动作集。
- **Shadow Connection**：对候选源、目标及状态条件的转移提议；形式可以是路由先验或可学习的状态条件许可，而非简单布尔边。
- **Shadow Path**：一段候选动作序列（可含自连接、CTX_READ、STOP），是**多步计划/轨迹**，不能把它当成与单步 Cell 相同的 primitive 而漏算其中每一步成本。
- 候选的存在、训练、评分、拒绝和接纳都受统一的 `Candidate Registry` 管理。Shadow 候选的分支计算或学习**不得污染 Accepted Model 的在线执行结果**。

---

## 4. Q 的严格定义与 STOP 的特殊地位

### 4.1 采用“成本越低越好”的 cost-to-go

固定一个已接受模型参数快照 `θ_k`、冻结后续策略 `π_k`、固定预算及成本协议。定义理论目标：

\[
 Q^{\pi_k}_{\theta_k}(s,a)
 =\mathbb E_{y,\omega}\!\left[
 \ell\big(\mathrm{Head}(h_{\mathrm{end}}),y\big)
 +\lambda\sum_{u=t}^{T-1}c(a_u)
 \mid s_t=s,a_t=a,\pi_k \text{ controls suffix}
 \right].
\]

这里期望来自条件目标分布与未来路径随机性。**Q 不表示某个样本的已知答案，也不是一个动作的固定全局质量**。实际训练只有有限样本的 Monte-Carlo 目标 `q̂(s,a;y,ω)`；需要预测其条件期望，并保留估计方差。

- **STOP**：`q̂(s,STOP;y)=CE(Head(h_t),y)`，该状态上的即停目标可直接计算，不需要后缀抽样；STOP 的未来执行成本严格为零。
- **Cell / CTX_READ**：先执行该动作，再用同一个冻结后续策略完成余下路径，重新计算最终 CE 与剩余成本。
- **Forced Termination**：预算耗尽时为终止条件，不是 `π` 主动选中 STOP 的数据点。
- **成本约定**：默认每次 Cell/CTX_READ 真实执行计 1 个 `cost_unit`（可在后续版本用已实测算力校准不同算子）；完整路径成本可除以 B 归一化，使不同配置尺度一致。训练目标 `λ` 及成本单位必须在开发训练前冻结；同时独立报告 FLOPs、实际 GPU wall、采样和 Shadow 评估开销。

### 4.2 Bellman 关系与非平稳性

固定 `θ_k,π_k` 且给定足够状态信息时，理论上有

\[
 Q(s,\mathrm{STOP})=\mathbb E_y[\ell(\mathrm{Head}(h),y)\mid s],
\]

\[
 Q(s,a)=\lambda c(a)+\mathbb E_{s'\mid s,a}[V^{\pi_k}(s')],\quad
 V^{\pi_k}(s')=\sum_{a'}\pi_k(a'\mid s')Q(s',a').
\]

如果采用贪心续行策略，`V` 的定义应同步改成对应确定性续行，不能一边给策略抽样后缀、一边把标签说成贪心最优价值。**本轮以真实完成后缀的 Monte-Carlo 目标为主；Bellman/TD 仅作为后续低成本近似选项，不在第一次训练引入未审计的 bootstrap 目标。**

由于 `θ` 和 `π` 同时更新，真实 Q 目标会漂移。禁止将长期旧样本不加版本区分地混入当前 Q 标签。候选数据必须携带模型快照 hash、policy hash、action registry version、候选后缀 RNG 和最大步数。

### 4.3 Connection 的参数化

第一轮**不再另设独立 Actor Head**。Connection 自身就是全动作的成本预测器：

\[
 \widehat{Q}_{\psi}(s,a)\in\mathbb R.
\]

决策分布完全由 Q 输出构造：

\[
 \pi_\psi(a\mid s)
 =(1-\epsilon)\,\mathrm{softmax}_{a\in\mathrm{legal}}
 \left(-\frac{\widehat Q_\psi(s,a)}{\tau}\right)
 +\epsilon\,\mathrm{Uniform}(\mathrm{legal}).
\]

- 保留当前 `Router` 的状态输入、前驱 ID、step embedding 及边先验作为 Q 的可选输入；输出解释由 logit 变成**成本数值**，不得把旧 logit 强行解释为已校准 Q。
- `ε>0` 是显式探索覆盖保障；不能因 logits 微差导致只有一个动作永远可见。推理可使用 `argmin Q` 或软策略，但两者均须真实评估，不能混写结果。
- 使用 `detach(h_t)` 训练 Q 头，防止辅助 Q 回归把连续 Cell 状态推向拟合缓存标签而非语言目标。Cell/Context/Head 仍由正式执行轨迹的 CE 反向传播。这属于**同一训练循环中的联合学习**，不是保证全部参数对 Q 目标同时可微。
- Q 的预测值、损失、校准和排序必须报告**整体及分动作类型**指标；只降 Q-MSE 不构成路径选择成功。

---

## 5. 反事实候选 Oracle：先通过正确性 Gate 才准信任

### 5.1 从同一状态分叉的干预协议

对一个训练状态 `s_t`、同样的 `θ_k`、同样的有效 y：

1. 保存此步之前的完整 `h_t`、因果 memory、前驱动作、剩余预算、候选可用性、模型快照 ID。
2. 选择**实际动作 + STOP（合法时） + 至多 1–2 个不同候选动作**；禁止按照真实 y 在线挑选要正式执行的动作。候选覆盖随机采样和基于 Q 不确定性/缺口的选择，但保存 proposal probability。
3. **分别**执行每个候选一次，再从新状态重新运行相同的冻结后续策略 `π_k`；完整重算下游 hidden state、路由、Context Read 和 STOP。
4. 对每个候选记录 `CE + λ × remaining_cost`、真实动作轨迹、强制终止、候选有效性、RNG 和版本号。
5. 使用 `torch.no_grad()` / 独立快照生成 Q 的监督标签；此时 **不更新 Accepted Model**，不通过探索分支偷偷训练未选中 Cell。
6. 合法候选集合随状态及生命周期变化；非法/未接纳 Shadow 不出现在正式推理分布中，其探索标签只用于 Candidate 评估或明确隔离的 Shadow 数据集。

**重要区别**：固定同一后缀**动作序列**与固定同一后缀**策略**是两种不同实验。主 Oracle 固定后者、重新决策；前者仅用于机制诊断。所有真实成本按实际动作计费。

### 5.2 公共随机数降低比较噪声，但不能改变语义

比较两候选时，对后续策略使用对齐的**基础随机数流**（common random numbers）；即便每个分支的动作概率不同，也通过对同一均匀变量做 inverse-CDF 抽样形成合法耦合。不得简单复用动作 ID 或持有旧 `torch.Generator` 导致一条路径隐式消费另一条路径的 RNG。

STOP 无后缀，因此可直接求真实样本损失。Cell 候选的多个后缀样本必须保留均值和方差；单次结果只能算 Monte-Carlo 样本，不能称为准确 Oracle。

### 5.3 Oracle 正确性/一致性 Gate（禁止用历史失败机制代替验证）

在合成可枚举小世界中，使用**所有合法动作、所有短后缀**构造精确参考值。至少检查：

- `STOP` 即时 CE 恒等；预算极限行为和强制结束正确。
- 对同一快照的 `Q(s,a)`，枚举期望与足够 Monte-Carlo 采样均值在统计容差内一致；若不一致，首先排查 suffix policy、cost accounting、teacher forcing 与 RNG。
- `CTX_READ` 从不读取未来；重放与独立 full rollout 给出一致 logits/动作/成本。
- 有限差分的候选结构微小扰动与重新执行后的损失方向一致（连续插值仅用在可微候选，不对离散动作滥用导数）。
- 对短世界的成对动作，候选 Q 的顺序能预测真正的优胜动作；若近似同分，标记 `LOW_ACTION_IDENTIFIABILITY`，**不强迫路由分化**。
- 不依赖同一批样本同时选候选、估计 Q 精度和宣布性能改善；评估按 story 拆分，并核查版本和 provenance。

**Gate 失败 => `ORACLE_UNTRUSTED`，终止正式 Q-first 训练。** 合成 Gate 通过也不等于真实语言语境中价值估计有用。真实模型先跑冻结快照的短期标签校验与候选重复测量，再进入完整开发训练。

---

## 6. Q-first 联合训练算法：可直接实现的第一版

### 6.1 主目标、价值目标与隔离边界

正式模型的训练目标为

\[
 J(\theta,\psi)=\mathbb E_{\tau\sim\pi_\psi}
 \left[\mathrm{CE}(h_T,y)+\lambda C(\tau)\right].
\]

训练采取**明确区分梯度来源的近似优化**：

- **连续模型更新**：仅对**实际正式轨迹**的 CE 求梯度，更新 Embedding、Cell、Context Read、Head 和输入状态网络。代价中与离散动作有关的部分通过策略学习体现；不可把其普通张量数值当成可通过 Cell 权重直接微分的成本。
- **Connection/Q 更新**：在隔离分支上生成 `q̂`；最小化 `Huber(Qψ(detach(s),a), stopgrad(q̂))`，并按候选覆盖概率明确采样/加权。可引入候选间 pairwise ranking 辅助损失，但**第一版默认仅 Q 回归**，以免混入第二种学习机制。
- **正式策略优化的解释**：因为 `π` 是 `Qψ` 的函数，通过拟合真值后的策略改进**间接**优化任务目标；这不是对 `J` 的完全端到端无偏梯度估计。报告不得声称因此获得理论收敛保证。
- **Shadow 更新**：只在它自己版本化的 candidate branch 上训练候选 Cell / connection 参数；它的梯度、AdamW 状态不能流入 Accepted Model，直到完整候选经验证事务化接纳。

建议保留 `θ` 与 `ψ` **独立 optimizer parameter groups**（可使用一个协调的优化步骤）；分别记录 LR、梯度、回滚状态和更新 hash。所有正式参数在同一个持续训练过程中一起演化，不进行独立 Cell 预训练。

### 6.2 训练循环的规范伪代码

```python
for optimizer_step in range(1, MAX_STEPS + 1):
    batch = next_frozen_schedule()            # no future token leakage
    snapshot = make_readonly_version(theta, psi, accepted_registry)

    # A. 正式模型路径：仅使用 Q(state, action) 和独立 RNG 作决策
    trajectory = rollout_hard_actions(
        accepted_model, batch.inputs,
        q_policy=psi, epsilon=eps, tau=tau,
        max_nonstop_actions=16, allow_self=True,
        stop_is_real_action=True,
    )
    ce_live = terminal_language_ce(trajectory.logits, batch.targets)
    backprop_continuous_params_from_live_trajectory(ce_live)

    # B. 反事实评分：使用步开始时冻结的模型/策略快照
    states = sample_probe_states(trajectory, budget=PROBES_PER_STEP)
    candidate_set = propose_actions(
        states, include_chosen=True, include_stop_if_legal=True,
        extra_actions=EXTRA_ACTIONS, independent_rng=True,
    )
    q_labels = complete_counterfactual_suffixes(
        snapshot, states, candidate_set,
        true_targets=batch.targets,         # ONLY inside this scoring branch
        suffix_policy=snapshot.policy,
        paired_randomness=True, no_grad=True,
    )
    loss_q = robust_regression(
        q_model=psi, features=detach(states),
        labels=detach(q_labels), proposal_weights=...,
    )
    backprop_q_head_only(loss_q)

    clip_and_step_accepted_optimizers_atomically()

    # C. Shadow 候选独立处理，不与正式 optimizer 同步提交
    if due_for_shadow_proposal(optimizer_step):
        propose_or_advance_shadow_from_frozen_checkpoint()
    if due_for_shadow_admission(optimizer_step):
        evaluate_shadow_vs_same_version_accepted_model()
        atomic_promote_or_reject_after_independent_gate()

    save_metrics_and_checkpoint_if_due()
```

**执行顺序细节**：实际代码中必须避免在 `optimizer.step()` 或释放计算图前改写 live model；快照应在同一 optimizer step 的参数更新**之前**锁定；候选评分使用保留的不可变状态。如果显存不够，先完成 live 反向传播再用冻结权重载入副本进行评分，但不能用更新后的 `θ_{k+1}` 冒充 `θ_k`。`π_k` 同理。

### 6.3 Q 训练漂移与估值偏差处理

- 第一版不使用跨许多更新步骤的离线 replay buffer。允许**同快照短缓存**，一旦模型/结构版本改变即失效。
- 至少报告不同动作的 Q 预测偏差、MAE、Spearman 排序、top-1 选择准确率、相对 regret、最优 STOP 识别率。
- 训练数据 Q 低误差但 held-out 排序失败 => `Q_GENERALIZATION_FAILURE`。
- Q 预测选择的动作在采样支持之外 => `Q_EXTRAPOLATION_RISK`，保持 `ε>0` 并禁止在未覆盖动作上自动接纳候选。
- 如果 Q 值偏好只因自连接/Cell ID 而对输入不敏感，需要固定状态/替换状态、打乱上一动作和动作 ID 的反事实诊断。
- 初始 `Q` 可用近似相等输出/小方差初始化，初期使用均匀混合探索覆盖；不能用训练标签直接决定 live action。

### 6.4 成本与探索预算

建议初始配置（**候选数值，须经 GPU Pilot、正确性 Gate 后冻结**）：

| 变量 | 起始候选 | 意义 |
|---|---:|---|
| 初始正式 Cell 数 | 6 | 与 FAST-001 接近 |
| 预留 Shadow Cell slots | 1 | 不立即扩大常驻参数上限 |
| 最大非 STOP 动作 | 16 | 允许自连接与 CTX_READ |
| 主动 STOP 最小非 STOP 动作数 | 1 | 避免 0 步退化；真实 STOP 仍参与价值学习 |
| 探索 ε | 0.15 | 采样覆盖，记录随步数变化；首轮不自动调参 |
| Boltzmann 温度 τ | 1.0（Q 以 nats 计） | 需经 Q 标签尺度审计 |
| 每训练 step 的反事实 probe 状态 | 最多 2 个 *token-step* | 不能为每个 token 每步展开全部候选 |
| 每 probe 的动作数 | 实际动作 + STOP（合法时） + 至多 2 其他动作 | 去重，并记录 proposal 概率 |
| 候选动作后缀数 | 2 | 合成枚举 Gate 不受此上限限制 |
| 完整后缀执行 | 从新状态重新决策 | 不复用旧状态/旧动作作为主 Oracle |
| 成本 c(Cell/CTX) | 每个真实动作 1/B | 统一逻辑预算；硬件成本另报 |
| 成本 λ | **冻结前由开发 calibration/Pilot 选择** | 不允许看最终 holdout 后选 λ；记录 λ=0 的离线曲线对照 |
| 主训练精度 | FP32 | 保留历史可信计算语义 |
| 训练 steps | 先 50-step smoke，完整开发 500 | 500×8×64=256k Token |
| batch / context | effective 8 / seq 64 | 复用 FAST-001 开发 profile |
| 轻量采样 | 不在每 step 全量探针/Dispatch/grad capture | 优先采用 `pd001-light-v1` |

此配置下，候选后缀可能使计算成本显著超过 FAST-001。**没有同设备实测，不承诺 30 分钟总时长，也不默认为候选分支计算免费。**

---

## 7. Shadow Cell、Shadow Connection、Shadow Path 统一生命周期

### 7.1 两个世界的边界

- `ACCEPTED`：用于正式训练路径和推理的模型版本；永久记录 Cell ID、Edge/Policy 版本、STOP 成本语义。
- `SHADOW`：独立候选版本；状态、权重、优化器、缓存、后缀 RNG 与已接受状态逻辑隔离。
- `PROPOSED -> PROBATION -> EVALUATED -> PROMOTED | REJECTED | EXPIRED` 是统一生命周期；所有状态转换都写入不可变 journal。

### 7.2 候选类型与可执行定义

**Shadow Connection**：对已经存在的候选 `i→j` 或条件转移 `g_e(h)` 提议新路由先验/资格，不改变 Cell 权重。新边在影子分支中得到访问机会，再通过完整后缀效用与接受集回归进行评价。**允许 `i=j` 自连接**，`i→STOP` 是可表达的终止选择。注意：当前第一版 Accepted Router 仍可在所有合法 Cell 间选择；影子连接代表**条件偏好/路径策略变更**，而非错误地宣称新增一条原先被禁止的布尔边。

**Shadow Path**：具体的候选动作前缀或完整计算轨迹，可能由序列 `{Cell_i, Cell_i, CTX_READ, Cell_j, STOP}` 构成。它本身不应新增常驻参数。可把高收益候选作为 Q 的诊断数据、可能的候选先验，但**不能把含答案的最优路径硬编码为推理决策**。

**Shadow Cell**：从某 Accepted Cell 复制或从预留槽初始化。优先选择可证明 birth-time 函数不变的 copy-on-write：`C_shadow ≡ C_parent` 在出生点，但 Shadow 不自动成为活跃路径。候选训练只发生在独立私有分支；成熟后必须通过独立 admission 校验，才能把 Cell 与其路由依赖作为**同一事务**接纳。仅仅复制一个 Cell 不保证功能分化，需检验其实际条件边际效用。

### 7.3 统一接纳协议

1. **Proposal**：由训练数据/状态及模型快照产生；允许标签参与训练候选，但标记候选来源，禁止泄漏到推理。
2. **Probation**：Shadow 与 Accepted 保持分离；对相同训练目标预算开展受控学习/计算。
3. **Paired evaluation**：将候选和精确同版本 Accepted 比较，冻结评估样本、seed、计算预算及后续策略；按 story 分组报告 `ΔNLL`、成本、旧功能回归及不确定性。
4. **Admission**：只有预先冻结的门槛通过才可接纳。**候选本身的局部边际收益只负责提议，不负责最终决定。**
5. **Atomic promotion**：Cell、Connection（及依赖边）、Q 头扩容或行重映射、optimizer 状态、action registry、checkpoint 指纹**全部作为一次事务**提交。若失败，完整回滚；禁止半接纳。
6. **Reset / versioning**：接纳改变策略或参数时，必须失效旧快照的 Q 标签，恢复覆盖探索并重新校准 STOP。

**具体门槛建议**：在独立 admission 数据上，候选**配对** token NLL 不劣于 Accepted、额外计算成本按冻结 λ 纳入、关键回归合同通过；声称积极收益须有预注册的最小改善幅度及分组置信区间。只有“未观察到退步”不足以接纳会增加容量或复杂度的候选。第一轮在没有足够故事数时允许拒绝全部 Shadow；不得为演示生命周期强制接纳。

### 7.4 容量公平与主动变化

预留一个 Shadow Cell slot，**两个训练臂都计入常驻总参数量**。实际生效与未激活参数另行计数，必须区分。禁止一个训练臂隐式增生参数而另一个训练臂维持原规模，再声称训练方法更优。

*本轮仅允许最多一次 Shadow Cell 的候选出生和一次成组 Shadow Connection/Path proposal。* 真实接纳次数可能为 0。未来如需自主多代分裂或合并，另行注册扩展协议；本轮不能自动开放无界生长。

---

## 8. 统一体系的内部对照：两臂而非 MoE 主导

对照**不是**“有 Cells vs 没 Cells”，也不是“无 STOP vs 有 STOP”；两臂都具有完整 CLM 计算体系。

| Arm | Cell/CTX/STOP/自连接 | Shadow 候选/接纳协议 | Connection 学习 | 主要问题 |
|---|---|---|---|---|
| **`QF-FULL`** | 相同 | 相同 | 候选全后缀 cost-to-go 的 Q 回归，按 Q 选路 | Q-first 能否形成有效计算组织 |
| **`J0-FULL`** | 相同 | 相同 | 旧 J0 的整路径 score-function + critic，在同一 λ 成本目标下 | 相对于既有联合训练，有没有可靠的增量价值 |

控制约束：相同正式/预留参数预算、Cell 初始化、Context/Head、数据顺序、训练 Token、训练 step、最大步数、成本约定、可接纳候选类型与候选评估预算。`J0-FULL` 也执行同样的候选探索/评分和 Shadow 治理，但是**不将候选结果用于直接训练正式 Router**；否则比较不是 Q-first vs J0。其额外评分为测量/候选治理开销，必须单列有效训练 FLOPs 与探索计算，不能用相同 wall time 假装相同模型 FLOPs。J0 的 Critic 额外参数要在两臂中明确计入或用同规模辅助价值头配平，避免偷换总参数。

可只对 `QF-FULL` 建立 `QF-SHUFFLED-LABEL` 的**冻结 checkpoint 诊断**（对同一 batch 的候选价值与动作身份打乱、观察选择和 NLL）；首轮不增加第三个完整训练臂。若 QF 表现积极，再使用新的开发 seed 复验；不得单 seed 宣称机制成立。

**为什么不把 Shadow 全部关闭来做对照？** 因为那会同时改变两个假设。这里首先比较路径学习原理；Shadow 生命周期是否带来独立收益，可在后续冻结 Q-first 机制后再单独检验。

---

## 9. 数据协议、防泄漏及评估

### 9.1 数据源与身份

首轮沿用 FAST-001 已审计的 TinyStories 训练资产体系，保持 tokenizer、序列 64、有效 batch 8、500 steps 与同类型 dev 评价；新实验要生成自身冻结的 `asset_manifest` 与 `config_sha256`。可读取获授权的**开发数据**，但不得读取或重写旧确认 holdout、历史已消耗正式种子。

按故事（不是随机 token）划分：

- **Train**：联合训练与候选 Oracle 标签、Shadow probation。
- **Q-calibration**：调试标签一致性/成本尺度/价值排序；不能用作正式性能证明。
- **Admission**：冻结候选接纳的独立开发验证集，不用于候选训练。
- **Dev evaluation**：冻结比较两臂的独立窗口，用于研究决策；重复迭代后属于已被观察的开发数据。
- **Confirmation**：未来独立预注册、从未用于调参与生命周期选择的 holdout，本轮关闭。

可复用 FAST-001 的原始 train/dev 来源**但不得将曾经查看过的 eval 窗口伪装为新的确认集**；拆分、跨故事重复检测、tokenizer hash 和窗口 ID 要持久化。禁止候选训练访问 Admission/Dev 的目标 y。

### 9.2 一致预算

- 两臂 `optimizer steps × effective_batch × seq = 500 × 8 × 64 = 256,000` **正式任务训练目标 token**。
- 另行记录 `counterfactual_rollout_tokens`、`shadow_candidate_train_tokens`、`oracle_labelled_actions`、`total_gpu_seconds` 和 `observed/proxy FLOPs`。**不能说“相同正式训练 token = 相同总计算”。**
- 仅在评估中启用 Greedy / Sampled / ε 策略，报告各自质量，不能只选最好的一种做结论。
- 在相同输入和预算下比较 `STOP_rate`、forced_stop_rate、平均/分位实际路径长度、NLL-计算成本 Pareto、CTX 使用、self-loop 占比、独立 Cell 受益状态分布。

### 9.3 主要评价（主次要明确）

**首要机制指标（不以 MoE NLL 排名）**：

1. **Action-value ranking validity**：独立样本上，预测的 `Q(s,a)-Q(s,b)` 与重新算完后缀的真实配对 cost-to-go 差值具有正相关，top-1 regret 比随机/常数预测器更低；报告 bootstrap CI 与跨 seed 稳定性。
2. **STOP rationality**：当 Oracle 判定 STOP 优于继续时，模型主动 STOP 的比例是否提高；当继续明显更优时，是否继续；同时报告停止错误的两类成本。
3. **On-policy realized utility**：按照**无标签**学到的路由策略实际运行，其独立 dev `NLL + λC` 优于同预算内部对照及自模型的常数/均匀路由，且不能仅依赖训练时 label-search Oracle。
4. **Composition and conditional specialization**：`Cell_i` 的效用随状态条件变化；替换路径/冻结 Cell/打乱条件 connection 是否造成可复现的性能变化；没有结构因果证据不得把 Cell ID 分布解释为知识专业化。
5. **Shadow safety and utility**：所有接纳均通过 holdout admission、差异记录与原子回滚；没有接纳不等于实验失败，但不能声称候选结构提升。

**次要指标**：dev NLL/PPL、常驻/有效参数、运行时 FLOPs、缓存 decode 延迟、训练整体 wall/探索开销、GPU 利用率、Q calibration、路径多样性、值函数漂移、对照成本差。

### 9.4 明确的不可接受解释

- Q-MSE 下降 ≠ Connection 学会正确路径。
- 路由熵下降 ≠ 学会有效计算。
- 自连接较多 ≠ 病态，也 ≠ 真实迭代能力；必须用反事实干预验证。
- STOP 较早 ≠ 计算效率优越；必须同时满足质量约束。
- Shadow 被接纳 ≠ 模型获得新知识；需独立验证。
- 有效训练 token 相同 ≠ 计算公平；要报告候选探索 FLOPs。
- 256k Token、小模型、一个开发 seed ≠ CLM 的一般能力或持续学习结论。

---

## 10. 实验门禁：必须依次通过

### Gate 0 — 源码审计与协议锁定（CPU，无 GPU）

- 对 `ArcheLabs/mini-cells` 最新工作树做只读核查，确定 FAST-001、`pd001-light-v1`、`PDCLM`、Router、J0、ContextRead、Checkpoint、Shadow 候选代码哪些可复用，哪些不能复用。产出 `ARCHITECTURE_AUDIT.md` 与所有源码文件 hash。
- 额外核查 Experiment 019 的 `UTILITY_ORACLE_INCONSISTENT` 真实判断与失败条件；本次新的冻结 Oracle 不能默默借用旧未通过验证的实现。
- 确定 `QF-FULL`/`J0-FULL` 实际参数数量、reserve slots、STOP 编码、初始化差异、双方比较范围；新的工作树不能修改旧注册历史。

### Gate 1 — 算法正确性与理论 Oracle（CPU + 极小合成数据）

- 枚举短预算、少 Cell、CTX_READ、STOP、自连接、强制终止的全部合法路径，证实 action-cost、Bellman 与 Monte-Carlo 定义一致。
- 检验 replay、梯度、无未来泄漏、即停 CE 恒等、不同候选真实全后缀重算、CRN 合法性、版本隔离。
- 验证 Q 头只由候选标签更新，正式 Cell 由真实任务 CE 更新；合成 Q 监督不使用 future token 特征。
- 小世界中构造状态 A 更适合 Cell 0、状态 B 更适合 Cell 1，Q 必须从输入状态学会正确区分；同时放入“两个 Cell 真同质”的零效用对照，Q **不得虚构专业化收益**。
- 故意注入错误候选后缀策略、无效标签、fake STOP、shadow-to-accepted 参数污染，测试 Gate fail-closed。

### Gate 2 — 真实模型冻结快照的 Oracle Pilot（GPU，须授权）

- 使用 FAST-001 合法开发 checkpoint 或全新 1M 开发模型，跑**有上限的候选对照**；只能用于工程/机制评估，不能混入 CLM-001 正式训练身份。
- 每组 Q-value 对比同一个模型快照、同一后续策略和分离的 target；对候选顺序/收益重复测量，报告估计方差和置信区间。
- 在同一张 T4 逐项测每步 `rollout`, `backward`, `Q oracle`, `shadow`, `admission`, `checkpoint` 的真实 wall time、peak memory、同步点。与历史 FAST-001 373/735 秒仅作工程参照，不当作相同硬件条件下新算法速度保证。
- 如 Oracle 错、显存不足或预计总成本过高：停止完整训练，仅交付 profiling 和修复，不自动降低科学预算。

### Gate 3 — 完整体系首次开发训练（GPU，**再次取得用户明确授权**）

- 同时从随机初始化启动 `QF-FULL` 与 `J0-FULL`；默认两张 T4 一张一臂，两个完整训练并发，候选诊断/正式评估避免同卡争用。
- 50 Step 数值稳定 smoke；开发训练总上限 500 Step；checkpoint 建议每 50/100 Step，按事件安全暂停。
- 每 100 Step 在隔离 Dev 片段离线记录 NLL、Q 排序、STOP 错误及计算长度；密集候选全后缀的独立分析单独排程。全部候选分支、shadow、Q、Accepted 优化器状态和 RNG 需可恢复。
- 不允许为了“看见一个 Shadow 成功”临时调 admission 标准；真实结果可能是 0 个 promotion。

### Gate 4 — 结果审查、机制判断与后续决策

| 状态 | 解释 | 下一步 |
|---|---|---|
| `ORACLE_UNTRUSTED` | 后缀、成本或标签不一致 | 修 Oracle；**不启动 Q-first** |
| `Q_UNIDENTIFIABLE` | 候选真实价值几乎无可分辨差异 | 研究 Cell 功能差异/输入上下文/候选覆盖，不强行训 Q |
| `Q_GENERALIZATION_FAILURE` | Train Q 拟合好，但独立排序差 | 研究表征/过拟合/样本覆盖，不扩大规模 |
| `STOP_COLLAPSE` | STOP 退化过早或永不到达，偏离成本—质量目标 | 复核 λ、STOP 监督、预算门槛，独立新协议再试 |
| `Q_SIGNAL_NO_POLICY_GAIN` | Q 排序有信息，但真实路由无收益 | 研究误差传播、分布转移、温度/探索和多步组合 |
| `NO_SHADOW_PROMOTION` | 没有达到接纳门槛的候选 | 记录为阴性候选信号，不降低门槛凑成功 |
| `INTEGRATED_DEV_SIGNAL` | Q 排序、STOP、路径策略在开发集有相互一致的正证据 | 新开发 seed 复验；再考虑注册独立 confirmation |
| `ENGINEERING_BLOCKED` | 运行时间、显存、checkpoint、RNG 或模型隔离失败 | 先修工程，不输出科学结论 |

**预先避免单点成功**：至少要求真实 `Q` 排序领先随机基线、on-policy cost-aware dev 指标领先匹配内部对照且 STOP 不出现代价失控，才能在**开发层级**写 `INTEGRATED_DEV_SIGNAL`。没有跨 seed/confirmation 时不能写“CLM 完整理论已验证”。具体数值门槛/CI 方法必须在正式开发运行前冻结，不能根据结果事后挑选。

---

## 11. 实施结构、接口与 Codex 任务

建议**只新增目录，不改旧源码默认行为**：

```text
research/projects/cellular-language-models/experiments/
  clm-001-qfirst-integrated/
    README.md
    SPEC.md                          # 本文的 source-bound 冻结副本
    ARCHITECTURE_AUDIT.md
    THEORY_AND_LIMITS.md
    config/proposed.json
    config/development-frozen.json   # 用户审核后冻结
    protocol.json
    implementation/clm001/
      model.py                       # reuse or wrap PDCLM Cell/Context/Head
      q_connection.py                # cost-to-go for Cell/CTX_READ/STOP
      actions.py                     # single action registry / legality / budgets
      rollout.py                     # hard path + STOP + replay
      oracle.py                      # same-snapshot full-suffix candidates
      sampling.py                    # CRN, proposal probs, coverage
      training.py                    # continuous CE + detached Q regression
      shadows.py                     # shadow candidate registry + COW isolation
      admission.py                   # paired admission and atomic transaction
      evaluation.py                  # Q ranking, STOP, policy and causal probes
      performance.py                 # light training; measured overhead
      checkpoint.py                  # accepted + shadow + optimizers + RNG
      report.py                      # machine-readable outcomes / provenance
      cli.py
    notebook.ipynb                   # preflight-only default; GPU explicit
    reports/                        # only small metadata, no sensitive data

tests/research/clm001_qfirst/
  test_oracle_enumeration.py
  test_stop_budget.py
  test_counterfactual_reexecution.py
  test_crn_pairing.py
  test_context_causality.py
  test_value_no_label_leak.py
  test_q_detach_and_live_grad.py
  test_shadow_isolation.py
  test_atomic_promotion_rollback.py
  test_j0_matched_control.py
  test_resume_identity.py
  test_report_contract.py
```

CLI 要至少支持：

```bash
python .../clm-001-qfirst-integrated/run.py --mode audit
python .../clm-001-qfirst-integrated/run.py --mode preflight --device cpu
python .../clm-001-qfirst-integrated/run.py --mode oracle-enumerate --toy
python .../clm-001-qfirst-integrated/run.py --mode params --all-arms

# 这些需要用户后续明确许可；不得在 Codex 常规实施中自动执行
python .../clm-001-qfirst-integrated/run.py --mode gpu-pilot --device cuda:0 --max-steps 20
python .../clm-001-qfirst-integrated/run.py --mode train-dev --arms QF-FULL J0-FULL --devices cuda:0 cuda:1 --config config/development-frozen.json
python .../clm-001-qfirst-integrated/run.py --mode evaluate --run-dir OUTPUT
python .../clm-001-qfirst-integrated/run.py --mode report --run-dir OUTPUT
```

**实施顺序**：P0 仓库审计和理论映射 → P1 动作/STOP/因果性 → P2 真正的 Q Oracle 和正确性 → P3 Q Connection 与连续 CE 联合训练 → P4 Shadow 候选 Registry 与接纳事务 → P5 匹配 J0 内部对照与评估 → P6 Checkpoint/CPU Gate/CI → P7 用户审核 GPU Pilot → P8 用户审核开发训练。Codex 不能以“完整体系比较重要”为由跳过 P2–P4 的正确性测试。

---

## 12. 数据产物与可复现性契约

### 12.1 每个 optimizer step 的最小日志

`step`, `arm`, `model_version`, `policy_version`, `seed`, `train_token_count`, `live_ce`, `path_cost`, `q_loss`, `q_label_count`, `q_mae`, `q_pairwise_accuracy`, `epsilon`, `temperature`, `lambda_cost`, `actual_stop_rate`, `forced_end_rate`, `mean_steps`, `ctx_reads`, `self_loops`, `cell_visits`, `train_seconds`, `oracle_seconds`, `shadow_seconds`, `peak_gpu_bytes`, `checkpoint_sha256`。

**未采样的 Q 诊断必须为 `null`，不得填 0 冒充估计通过。** `STOP` 目标和 `FORCED_TERMINATION` 必须不同编码。

### 12.2 每个候选样本的 provenance

`candidate_id`, `candidate_type`, `source_state_hash`, `source_model_sha`, `source_policy_sha`, `source_action`, `alternative_action`, `available_actions`, `target_token_hash`, `rollout_seeds`, `suffix_policy_kind`, `q_sample_values`, `q_mean`, `q_var`, `remaining_cost`, `completion_trace_hash`, `proposal_probability`, `candidate_status`。不得将原始敏感 Token/训练语料、私钥、Kaggle secrets 推送到 Git。

### 12.3 Checkpoint 与跨会话恢复

- Accepted model、完整 Q 头、Cell/CTX/Head、全部 optimizer/scheduler、全局步数、Torch/CUDA/NumPy/Python/候选 RNG、当前 epsilon/τ/λ、数据采样位置、action registry、活跃 Shadow 分支及其 optimizer 和事务日志完整保存。
- 同时保存唯一 source SHA、config/protocol hash、Tokenizer/assets hash、每个有效 candidate checkpoint 和文件 SHA256。
- 原子保存、断电恢复、暂停与完成状态分离；保存版本不可原地覆盖历史已完成结果。
- Kaggle `/kaggle/working` **不是跨会话自动持久化保障**；必须先将 Output 正式保存并在新会话重新导入验证恢复。未通过 portable checkpoint gate 不运行 500 Step。
- 旧 FAST-001 checkpoint 与新 Q-first 训练 Checkpoint **不可直接接续**，因为学习目标与状态身份不同；只允许只读机制诊断。

### 12.4 最终机器可读报告格式

```json
{
  "experiment_id": "CLM-001-QFIRST",
  "claim_scope": "DEVELOPMENT_ONLY",
  "status": "ENGINEERING_READY | ORACLE_UNTRUSTED | DEVELOPMENT_COMPLETED | INCOMPLETE",
  "source_sha": null,
  "config_sha256": null,
  "asset_manifest_sha256": null,
  "tokenizer_sha256": null,
  "seed": null,
  "arms": {
    "QF-FULL": {
      "params_resident": null,
      "params_active": null,
      "train_tokens": null,
      "dev_nll_greedy": null,
      "dev_nll_sampled": null,
      "q_rank_spearman": null,
      "q_regret": null,
      "stop_accuracy_vs_oracle": null,
      "actual_stop_rate": null,
      "forced_end_rate": null,
      "mean_executed_actions": null,
      "shadow_proposed": null,
      "shadow_promoted": null,
      "shadow_rolled_back": null,
      "training_wall_seconds": null,
      "exploration_wall_seconds": null
    },
    "J0-FULL": {"same_schema": true}
  },
  "paired_internal_contrasts": {},
  "oracle_gate": {},
  "action_conditional_utility": {},
  "stop_nll_compute_pareto": {},
  "shadow_journal_sha256": null,
  "scientific_decision": false,
  "notes": []
}
```

以上 `null` 是未测占位。`J0-FULL` 实际输出必须展开成相同 schema，不能保留 `same_schema` 作为最终实验数据。所有数值和正面/负面结论须来自真实执行；系统不得自动填充估计值。

---

## 13. 严格验收测试：工程完成的定义

1. **真实计算**：每 token 每步只执行一个被选中的动作；重复 Cell 执行真实多次；`STOP` 与强制结束区分；candidate 只能在候选分支出现。
2. **上下文因果**：改变未来输入不能影响当前/过去 logits；无未声明免费 attention，CTX_READ 正确读取已有前缀。
3. **参数契约**：两主臂参数/初始化/Shadow 预留容量及累计训练目标可审计；任何不等配平显式标记，不伪称同参。
4. **Q 语义**：Q 是 `cost-to-go`、值越小越好；STOP target 是立即 CE；Cell/CTX target 是完整后缀 CE+剩余成本。
5. **后缀一致性**：同快照同策略、多动作独立重算；短世界精确枚举与 MC 统计一致；任意故意打乱后缀协议必须拒绝。
6. **随机数隔离**：live action RNG、candidate proposal RNG、suffix RNG、shadow training RNG 均独立；推理结果不取决于是否运行候选诊断。
7. **无标签推理**：在线路由/STOP/Shadow 接纳推理路径不访问 train target、dev label 或实验结果；只有隔离的 train Oracle 生成器能访问训练 y。
8. **梯度契约**：正式 CE 到已选 Cell/CTX/Head；Q 回归只到 Connection Q 头；拒绝候选不能污染 Accepted 参数/优化器状态。
9. **条件分化测试**：功能异质的合成 Cell 排序可学习；功能同质的 Cell 不应被报告为自发专业化。
10. **接纳/回滚**：同版本配对验证，原子 commit、失败回滚；候选策略改变会使旧 Q label 缓存失效。
11. **断点恢复**：完整状态及 RNG 跨会话恢复后轨迹可复现（CUDA 容忍浮点舍入，动作及 RNG 应严格一致）。
12. **性能隔离**：`pd001-light-v1` 的全量诊断和 Dispatch 不得重新进入每步训练热路径；真实反事实开销单独测量。
13. **科学边界**：只用开发 seed/数据写 `EXPLORATORY_ONLY`；旧正式数据冻结不变；No-Go 历史声明不被改写。
14. **零结果也可交付**：Oracle 不可信、无 Q 优势、STOP 退化、Shadow 接纳为零，均输出完整可信报告并按状态终止，不强行证明假设。

---

## 14. 下一步实施授权与完成物

**本文件交给 Codex 的当前任务**：只做 Gate 0 和 Gate 1 所需的理论映射、兼容性审计、源码实现、单元测试、合成枚举、代码审查、Notebook/CLI、文档、CI、Checkpoint 演练及提交/推送；后续 GPU Pilot 和完整 500-step 训练均由用户明确下达新的运行指令。不要把文档中的“建议配置”当成无需批准的长期 GPU 使用许可。

交付时必须报告：

- 新分支/commit、对旧研究树的只读差异检查；
- 完整 `config/proposed.json`、参数清单、动作/STOP 语义、Q Oracle 单元验证结果；
- `ORACLE_TRUST_GATE.md`（枚举/随机一致性、候选前缀冻结与标签隔离）；
- `SHADOW_ISOLATION_AUDIT.md`（候选/接纳/回滚的事务正确性）；
- `PERFORMANCE_BUDGET.md`（非 GPU 的估计必须标成估计，GPU 时间留待 Pilot）；
- 最小可用 `notebook.ipynb`、README、标准 report schema、portable checkpoint 演练；
- 明确的下一步 **GPU Pilot 操作步骤**与需要用户核准的冻结配置；
- 尚未解决的风险与不能声称的科学结论。

### 最终研究原则

> **Q-first 不是将训练标签泄漏进路由，而是在训练阶段用可验证的反事实计算构造监督，让只观察当前状态的 Connection 学习条件价值；Cell 和 Connection 仍在一个持续演化的模型中共同学习。STOP 是同一策略的真实动作，Shadow 是有边界的候选计算；任何新增结构只有通过完整模型评价，才可能成为 Accepted Model 的一部分。**
>
> 数学定义完整不等于训练机制会成功。CLM-001 必须能够明确失败，而且应先用最小成本失败得有信息量。

---

## 参考与 provenance

### 仓库内研究资产（Codex 以实际 source SHA 再核验）

- `research/governance/audits/CLM_CAPABILITY_CEILING.md` — 科学支持边界及 No-Go。
- `research/governance/audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md` — 旧生命周期、候选隔离与回滚原语。
- `research/projects/cellular-computing/context/02-self-organization/hypothesis/sources/EXPERIMENT_019.md` — 条件候选效用定义与 Oracle Gate。
- `research/projects/cellular-computing/experiments/019-proposal-utility-discovery-stable/artifacts/RESULTS.md` — `UTILITY_ORACLE_INCONSISTENT`。
- `research/projects/cellular-language-models/context/06-native-clm/hypothesis/SHADOW_CELL_VALIDATION_001.md` — Shadow copy-on-write 定义。
- `research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/implementation/pd001/{models,router,losses,engine,performance}.py` — 真实硬路径、J0 与 light trainer。
- 上传结果 `report.json` — FAST-001 单种子开发实验结果，不是 confirmation 结果。

### 公开理论参照（不等于直接证实本方案）

- Sutton, Precup & Singh (1999), *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*. DOI `10.1016/S0004-3702(99)00052-1`：复合动作与终止条件的理论语境。本实验选择显式自连接 + 统一 STOP，并不复制 Option 内部终止器。
- Banino, Balaguer & Blundell (2021), *PonderNet: Learning to Ponder*, arXiv `2107.05407`：学习计算深度与终止的经验背景，并非本 Q-first 架构的理论保证。
- Fujimoto, Meger & Precup (2019), *Off-Policy Deep Reinforcement Learning without Exploration*, ICML：警示行为策略外动作价值的外推误差；因此本协议强制候选覆盖与分布报告。
- van Hasselt (2010), *Double Q-Learning*, NeurIPS：说明对估计 Q 做 `argmax/min` 可能放大估值误差；本轮首先使用真实后缀 MC 标注和覆盖采样，不把双 Q 作为未检验捷径。

**文件结束。**

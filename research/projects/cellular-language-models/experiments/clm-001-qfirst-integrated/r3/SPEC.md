# CLM-001-R3 — Competitive Credit Assignment & Specialization Emergence
## MiniCells 完整 CLM：竞争性信用分配、Cell 功能分化与自强化机制研究／Codex 实施协议

> 状态：**PROPOSED / DESIGN ONLY / NOT IMPLEMENTED / NOT TRAINED**  
> 日期：2026-10-15  
> 研究类型：在 CLM-001 既有架构内，检验新的训练机制；不宣称新算法已证实有效。  
> 代码仓库：`ArcheLabs/mini-cells`。建议实施分支：`codex/clm001-r3-competitive-credit`。  
> 源码基线：`4fcb95490c2ca68affbab6881135f2e85e5bea87`；R2 历史结果分支：`results/clm001-r2-4fcb95490c2c-c541c76b`。  
> 术语：`Cell`（连续计算算子），`Connection`（单一决策机制），`CTX_READ`（显式上下文读取），`STOP`（终止动作），`Shadow`（候选生命周期）。

---

## 0. 执行摘要与不可变研究边界

### 0.1 要回答的根本问题

CLM-001-R2 工程上完成两组各 500 Step 训练，但尚未证明 Connection 具有可靠的动作价值排序能力，也未证明 Cell 形成可利用的条件功能分化。下一轮不以“再把 Q 调得更准”为唯一目标，而检验下面的完整因果链：

```text
Cell 存在初始计算差异
          ↓
训练中的竞争性信用信号识别差异
          ↓
Connection 对不同状态形成可验证的条件偏好
          ↓
条件分配改变各 Cell 的学习状态分布
          ↓
Cell 产生可重复、可干预的条件功能优势
          ↓
后续 Connection 对这些优势持续强化
```

**主要假设 H1**：为硬路径 CLM 增加一个**有偏但可测试的局部竞争性梯度通道**，在不改变真实前向执行的条件下，相比 R2 的 Q-only 方法，能改善 Connection 的条件选择和相应实际任务效用。

**更强假设 H2**：H1 的收益是通过“路由条件分化 → 学习机会改变 → Cell 条件优势增强”的反馈实现的，而非仅仅来自辅助损失、额外候选计算、整体 NLL 更快下降或路由坍缩。

H1 与 H2 必须分别判定。若只观察到质量改善而无法证实反馈链，只能得出训练方法改善，不能声称已经解决自主演化或功能分化。

### 0.2 保留原有完整 CLM 架构

正式前向推理只能执行 **一个选中动作**：一个 Cell、一次 CTX_READ，或者 STOP。允许 Cell 自连接、重复调用、序列组合；预算和 STOP 语义不变。Memory 保持基于初始输入的因果、只读 K/V。Cell 不直接访问 Memory，Connection 不免费获得 Context Summary；不引入固定 Transformer Attention Backbone；不将混合专家输出当成正式推理路径。

运行时：

\[
a_t\in\{C_0,\ldots,C_5,\mathrm{CTX},\mathrm{STOP}\},\qquad
h_{t+1}=\begin{cases}
h_t+\alpha C_{a_t}(\operatorname{LN}(h_t)),&a_t\in C\\
h_t+\alpha\operatorname{ContextRead}(\operatorname{LN}(h_t),M),&a_t=\mathrm{CTX}\\
h_t,&a_t=\mathrm{STOP}\text{（结束，不再继续）}.
\end{cases}
\]

保留 6 个 accepted Cell、1 个 reserved Shadow Cell、预算 16、独立缓存/检查点/数据窗和现有 Shadow 状态机。除非经过独立研究批准，不改变这些结构边界。

### 0.3 创新与对照应当分开

本轮**核心改动只发生在训练信用分配**。不能同时加入 Context Summary、改 Cell 宽度、引入新 Attention、更换 Tokenizer、改变数据域或扩大预算，否则机制归因会失效。

正式主对照必须保证：相同源代码算子、初始参数（按名字初始化）、训练 Token、相同样本顺序、运行预算、STOP/CTX 机制、Shadow Admission 政策；仅竞争性梯度是否存在不同。额外候选计算要单独计量并加入计算匹配控制。

### 0.4 已知事实（不是新预测）

读取 R2 历史聚合报告与实现所得：

| 项目 | R2 已有证据 |
|---|---|
| QF-R2、J0-R2 | 均完成 500 Step / 256,000 Token；真实 GPU Pilot、故障注入及断点恢复通过 |
| QF Greedy 与自身 Uniform | Greedy NLL 4.73393，Uniform 4.71355；Greedy 较差且成本较高 |
| QF vs J0 Greedy | QF NLL 4.73393，J0 4.78097；但不能归因为 Q 排序成功 |
| QF Q-calibration | Spearman ≈ -0.0623；Pairwise Accuracy ≈ 0.463；17 个有效状态，11 个可辨识状态 |
| CTX | QF Greedy 中 Context 动作占比 ≈ 0.00135%（近乎不读上下文） |
| Shadow | 两组各 3 个候选均拒绝；无有效晋升证据 |
| 工程 | 完整回传、候选数值隔离、Checkpoint 已可执行；应继承而非重新实现一套 |

注意 R2 的校准代码主要测量窗口首 Token，存在上下文覆盖不足；本轮必须修正。上述 Development 数据为单 Seed、有限 MC 的探索证据，不能当成确认性结果。

### 0.5 明确禁止的结论

不得把以下现象单独写为“Cell 已分化”：不同 Cell 权重不相等、调用次数接近均衡、路由熵降低、不同数据类别路由不同、路由梯度非零、某一次 NLL 下降、某个 Shadow 被接受。功能分化必须有冻结模型下的**配对反事实干预**与**状态条件交叉优势**。

---

## 1. 外部研究依据与本方案的创新界限

1. **Switch Transformer（JMLR 2022）**：稀疏专家路由、可微门控、辅助负载均衡、训练稳定性；证明可用连续可微权重优化部分路由参数，但并未保证形成互补的专家功能。来源：https://www.jmlr.org/papers/v23/21-0998.html
2. **StableMoE（ACL 2022）**：路由目标在训练过程频繁切换会损害专家样本效率；分阶段冻结路线是一种干预。其结论支持测量路由波动，但不等于必须先固定连接。来源：https://aclanthology.org/2022.acl-long.489/
3. **SparseMixer / Sparse Backpropagation for MoE Training（2023）**：稀疏离散路由丢失部分路由梯度项，设计梯度估计可改善训练；与本轮寻找“硬路径下的竞争性信用通道”直接相关，但其估计器不能直接无证明迁移到 CLM 的序列动作。来源：https://arxiv.org/abs/2310.00811
4. **GRIN: GRadient-INformed MoE（2024）**：研究离散稀疏路由的梯度估计；提醒我们不应把所有 Sparse MoE 都简化为可微 Top-K 权重训练。来源：https://arxiv.org/abs/2409.12136
5. **From Sparse to Soft Mixtures of Experts（ICLR 2024）**：不同的可微分专家分配方式可改善优化，但其架构采用不同的 token 组合方式，并不是 CLM 的硬序列计算。来源：https://proceedings.iclr.cc/paper_files/paper/2024/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html

**文献支持的是训练动机，不是 CLM 方法已经有效。** 以下“局部竞争性 Straight-Through 代理”“因果自强化指标”“混合局部／长程 Q 信号”的具体协议均属本轮待检验的研究提案。

---

## 2. 基础因果模型：分化机制必须具备什么

### 2.1 三种不同的对象

- **参数差异**：\(\theta_i\neq\theta_j\)。随机初始化天然成立，没有科学价值。
- **路由差异**：\(P(a=i\mid s\in G_1)\neq P(a=i\mid s\in G_2)\)。它可能只是识别了词面格式，也可能是因果有益的。
- **功能分化**：相同状态下交换 Cell、固定后缀／成本后，可测得一组状态适合 Cell i、另一组适合 Cell j，并且该结构跨评估窗口和随机 Seed 稳定。

本轮的因变量必须是第三种；第二种是可能的中介变量，第一种仅作完整性检查。

### 2.2 自强化需要两个方向的实证证据

对于一组预先定义且不依赖本轮目标标签的固定评估状态 \(G\)，令：

\[
A_{ij}(G,t)=\mathbb E_{s\in G}[U_t(s,j)-U_t(s,i)]
\]

其中 \(U\) 为配对的最终 CE + 规范化执行成本；\(A_{ij}>0\) 表示在组 G 选择 i 优于 j。另定义路由偏好：

\[
R_{ij}(G,t)=\mathbb E_{s\in G}[\log \pi_t(i|s)-\log\pi_t(j|s)]
\]

完整正反馈要求：（a）先出现可辨识的正 \(A_{ij}\)，随后 \(R_{ij}\) 朝相同方向变化；（b）在匹配参数/数据、固定评估状态下，\(A_{ij}\) 进一步扩大；（c）对其他组出现有意义的不同优势，不能是所有组都坍缩到同一 Cell。

**警告**：\(A\) 与 \(R\) 在同一时间点相关，不足以证明因果反馈。必须引入路由冻结/打乱/等频控制，以区分共同趋势。

### 2.3 可能失败的四个不同层次

F1：候选 Cell 根本没有可辨识的条件收益差异。F2：差异存在，但代理梯度无法正确指向收益更好的动作。F3：代理能学习选择，但 Cell 训练仍不能产生稳定分工。F4：局部竞争有效，但完整长路径、CTX 或 STOP 改善有限。必须逐项报告，不能统称“模型没学会”。

---

## 3. 主要机制：硬路径 + 局部竞争梯度（LCG）

### 3.1 为什么不直接把 CLM 变成 Soft MoE

我们的架构要验证 **硬执行、顺序组合、自连接、动态停止** 的可学习性。将每一步改为 \(\sum_i p_i C_i(h)\) 的真实混合前向，会改变计算语义；这样即使学会，也无法证明硬路径 CLM 有效。

因此，提出 **Local Competitive Gradient（LCG）**：前向仍执行选中 Cell；在少量训练位置，额外读取其他 Cell 的输出，仅作为反向传播的辅助训练代理，**不写入正式模型状态**。R2 的真实硬路径和预测 Head 不变。

### 3.2 单个决策点的定义

在一个已真实访问、选中 Cell \(a\) 的决策点：

\[
\delta_i(s)=\alpha C_i(\operatorname{LN}(h)),\quad
h_{\mathrm{hard}}=h+\delta_a(s)
\]

使用当前 Connection 的 Cell-logits \(z_i(s)\)，在候选集合 \(K_s\subseteq\{0,\ldots,5\}\) 上形成条件分布：

\[
p_i^{K}(s)=\frac{\exp(z_i(s)/\tau_c)}{\sum_{j\in K_s}\exp(z_j(s)/\tau_c)}.
\]

候选集合至少包含实际选中 Cell，加上 \(k-1\) 个合法且不同的 Cell。候选来源预先冻结，含均匀探索配额，**不得只包含当前模型认为最好的 Cell**。记录 inclusion probability；不把子集内 softmax 解释成对全部动作的无偏梯度。

只用于梯度的连续代理：

\[
\bar\delta=\sum_{i\in K_s}p_i^K\operatorname{sg}(\delta_i),
\qquad
\boxed{h_{\mathrm{ST}}=h_{\mathrm{hard}}+\bar\delta-\operatorname{sg}(\bar\delta)}.
\]

`sg` 为 Stop Gradient。构造前向值：

\[
\operatorname{value}(h_{\mathrm{ST}})=\operatorname{value}(h_{\mathrm{hard}}).
\]

因此，实际模型输出、被调用算子、CTX 记忆、STOP、路径长度完全保持硬路径语义。

但从语言损失到 Router logit 的代理梯度为：

\[
\boxed{
\frac{\widehat{\partial L}}{\partial z_i}
=\frac{p_i^K}{\tau_c}
\left\langle g,\operatorname{sg}(\delta_i)-\bar\delta\right\rangle},
\quad g=\frac{\partial L}{\partial h_{\mathrm{hard}}}.
\]

含义：相对于候选均值，哪个 Cell 的输出更顺应当前真实任务的下降方向，就为该 Cell 的路由倾向提供训练信号。

### 3.3 这种梯度的性质：**有偏估计，绝非完整硬动作策略梯度**

上述梯度是围绕选中路径和局部状态的一阶代理。它不含切换动作后改变未来状态、后续选择与停止行为产生的全部影响，也不保证与真实完整路径效用同号。它**不是** \(\nabla_\theta E_{a\sim\pi_\theta}[L]\) 的无偏估计，也不等价于 MoE 原始连续前向。

尤其当：

- 不同 Cell 引起很大的隐藏状态变化；
- 后续 Connection 因状态变化改选不同路径；
- 候选存在复杂的先后互补关系；
- 激活函数局部曲率高，或 STOP 被过早触发；

局部一阶代理就可能产生系统性错误。因此本轮必须配备 **真值方向审计**（第 6 节）；不允许仅凭代理梯度存在就认为方法合理。

### 3.4 推荐工程实现：两种等价意图，不可混用

**实现方式 A（默认，较易审计）**：先执行完全真实的硬路径 CE backward，在选中决策后的隐藏态缓存 \(g=\partial L/\partial h_{\mathrm{hard}}\)，再使用已停止梯度的 \(h,\delta_i,g\) 构造一个标量竞争代理：

\[
\boxed{
L_{\mathrm{LCG}}(s)=\sum_{i\in K_s}p_i^K(s)
\underbrace{\operatorname{sg}\!\left[\langle g,\delta_i-\delta_a\rangle\right]}_{d_i^{\mathrm{local}}}
}
\]

对 \(L_{\mathrm{LCG}}\) 只向 Connection 的路由参数反传，不向 Cell、InputMemory 或 embedding 反传。它与上面 ST 节点的 logit 局部梯度一致（在对应的相同 g、相同候选集合和权重下）。优点是前向图语义简单，便于精准测量代理信号与实际有限差分的一致性。

**实现方式 B（研究性备选）**：在真实 forward 图中嵌入 \(h_{\mathrm{ST}}\) 节点，通过 CE 梯度反传。默认不得开启；若使用，需实现严格的 forward-bitwise parity、Cell 参数梯度不污染、停止后无意外梯度传播以及重放一致性测试。**不能把 A+B 相加**，否则同一个竞争信号重复计入。

**为何默认 A**：当前 R2 已经有清晰的硬路径 forward/backward、独立梯度组和数值事务。A 仅在被选中的小样本状态上新增独立 Router loss，更容易定位有效性；若 A 本身错误，不应通过更复杂的 B 掩盖。

### 3.5 避免候选计算污染 Cell 梯度

- 真实硬执行路径：Cell 接受原有语言模型 CE 梯度，保持 R2 连续参数训练机制。
- 候选 Cell 的辅助 forward：在 `no_grad` 下计算 \(\delta_i\)；不对未执行 Cell 反向传播。这保证候选模型没有“免费获得额外监督”，主实验仅改变 Connection 的信用通道。
- Competition 只更新 `controller` 相关 logits 参数；其输入 \(h\) 和梯度 \(g\) 均 detach。
- Q 回归损失仍独立执行，梯度按照既有 R2 组裁剪，并分别记录 Q 与 LCG 的梯度范数及二者夹角。
- 候选辅助计算发生于训练，计入训练成本；不构成推理中真实激活 Cell。

### 3.6 注意：仅限 Cell 竞争，CTX/STOP 不能被错误等价

**主实验 LCG 只在已执行 Cell 的决策节点、只对其他 Cell 做比较。** 不给 `CTX_READ` 和 `STOP` 造一个“零值假 Cell”，也不把 CTX 视为普通 FFN：它读取 Memory，具有不同信息语义。STOP 终止后缀，是选项终止而不是 \(\delta=0\) 的普通一步操作。

CTX 与 STOP **继续使用现有相对 Q 全后缀 Oracle**，且由同一个正式 Connection 输出动作评分；LCG 对 Cell 子集加局部竞争约束，不改变独立合法动作集合。此选择让本轮先验证最核心的 Cell 分化机制，同时保留完整体系。

若 LCG 让 CTX/STOP 退化，必须如实报告 `PARTIAL_CREDIT_FAILURE`；**不允许事后追加免费 Context 或修改 STOP 上限掩盖问题**。针对 CTX/STOP 的可微分信用扩展作为独立后续实验。

### 3.7 候选计算是否应该包括 Cell 自连接？

自连接只是合法的路径重复选择，没有新算子类型。若当前 `previous==i`，候选里仍可含 i；不施加自连接惩罚。通过条件价值和路径效用判断它是否有益。额外日志测量 `(previous_cell, candidate_cell)` 的对照优势和相同输入的 route dwell time。

---

## 4. 与 Q-first 的关系：短程竞争与长程价值分工

### 4.1 长期价值仍然保留

R2 的 STOP-relative Q：

\[
\Delta Q^{\pi}(s,a)=E_{\pi}[L_{\mathrm{terminal}}+\lambda C\mid s,a]-L_{\mathrm{STOP}}(s)
\]

继续监督 CTX、STOP 和候选动作的长程表现，允许识别局部近似忽略的有益多步组合。

训练总损失在同一结构下定义为：

\[
\boxed{
L=L_{\mathrm{hard\ CE}}+\beta_Q L_{\mathrm{relQ}}+
\beta_{\mathrm{rank}}L_{\mathrm{rank}}+
\beta_C L_{\mathrm{LCG}}+
\beta_B L_{\mathrm{balance}}
}
\]

但这里各项不能未经尺度审计直接相加；实际应把 CE、Q/Rank、LCG 的梯度分别记账、应用可控倍率，记录冲突；若 LCG 与 Q 长期强烈反向，不能只增加 LCG 权重强压 Q。

### 4.2 一个可追踪的低成本实现选择

为减少结构混杂，**主对照先复用 R2 Connection 评分张量**，不新增额外 Policy Head，不改变推理参数总量。设原始成本 \(q_i\) 越低越好，则 LCG 的“偏好 Logit”取 \(z_i=-q_i\)，只在 Cell 候选内 Softmax。R2 的 Q supervised 和 LCG 分别以不同梯度分量作用于同一组 Controller 参数。

这允许严格测试：**相同模型架构，增加局部竞争监督是否有效**。

若出现持续的梯度冲突，再在独立 R3b 试验中分离 `policy_logits` 与 `q_values` 两个 Head；不得在主实验一开始就同时增加容量与改变可识别性。

### 4.3 潜在梯度冲突及门禁

对同一批选中状态分别计算：

\[
g_Q=\nabla_\theta L_Q,\quad
g_C=\nabla_\theta L_{LCG},\quad
\cos(g_Q,g_C)=\frac{\langle g_Q,g_C\rangle}{\|g_Q\|\|g_C\|+\varepsilon}.
\]

连续记录均值、分位数、负冲突比例；对某一步的不同损失只能从相同参数快照估计，不能在两者间悄悄 `optimizer.step()`。任何梯度保护（如 PCGrad）属于**新的实验因素**，本轮不要默认开启。

### 4.4 局部竞争并不能直接替代长程 Q

如果执行 C1 的局部计算看似不佳，但 `C1→C2` 的组合很好，LCG 可能偏向错误动作。必须用完整后缀配对审计区分这种互补关系（第 6 节）。若 LCG 的单步排序经常与真实长程效用反向，结论应是“局部代理不适于此结构”，不是调大权重重试。

---

## 5. Cell 分化的条件：竞争、稳定性和探索要平衡

### 5.1 不强迫 Cell 输出彼此不同

禁止以 `||C_i(h)-C_j(h)||` 最大化、固定正交、按故事/数学标签硬编码专家作为主实验训练目标。这只会制造形式差异，不能证明有用。路由信用必须来自真实 CE/Cost 的方向，且有负样本和因果替换验证。

### 5.2 防止坍缩，但不能假装均衡即专业化

可在主实验加入小权重、两组都相同的**训练机会防坍缩约束**：

\[
L_{bal}=N\sum_{i=1}^{N}\operatorname{sg}(f_i)\bar p_i
\]

其中 \(f_i\) 为训练窗口内 Cell 被执行的比例，\(\bar p_i\) 为模型对 Cell 类动作的平均概率。因为 CLM 会重复某个 Cell、自连接、随时 STOP，计算单位必须是**实际合法且执行 Cell 的 decision event**，而不是 Token；不把 CTX、STOP 分摊为专家容量。该项仅为防止极端负载坍缩，不代表功能分化；建议前期弱权重，作为全臂共享配置锁。

**主实验更保守的选择**：保留 R2 既有探索调度（\(\epsilon\)：0.40→0.25→0.15），先增加测量而不改变负载损失。若使用新的 balance，则所有主臂必须同时使用，不能把收益错误归功于 LCG。

### 5.3 什么时候稳定 Connection？

不把“先固定 Connection”作为主研究处理。只额外运行一个**机制消融**：冻结 Router 的时间窗口 vs 持续更新，验证 StableMoE 所讨论的路由波动是否确实是本场景的主要混杂因素。冻结策略需要保留条件多样性，不能使用对所有 Token 完全恒定的同一路径。

### 5.4 路由稳定性指标

对固定 token/window、统一模型快照，记录：`action_flip_rate`、`JS(π_t,π_{t-1})`、`per-cell exposure`, `transition_matrix`, `self_loop_rate`、`group_conditioned_action_prob`；必须区分“参数变化但函数稳定”和“相同状态的实选动作变化”。由于隐藏态本身随 Cell 更新而变化，要同时提供 `frozen_state_probes`（固定旧 h，纯 Router）与 `live_recomputed_probes`（重计算当前 h）两组指标，避免混淆。

---

## 6. LCG 核心科学门禁：代理梯度与真实反事实方向必须一致吗？

这项研究的**前置关键门禁**比 500-Step 完整训练更重要。不能先跑完再问。

### 6.1 对一个固定状态的真值审计

从不带目标泄漏的 held-out Calibration 中，选择已真实访问的 \(h\)、合法 Cell 候选 \(i,j\)、同一实际选中动作 \(a\)。在冻结模型、冻结 Memory 和固定后缀策略/采样 RNG 条件下，计算：

1. 已执行 Cell a 后真实完整路径的 CE + cost。
2. 强制第一动作替换为 i；其后 **replay 相同合法的动作序列**，得到 \(L_i^{\mathrm{fixed\ suffix}}\)。
3. 强制第一动作替换为 j，得到 \(L_j^{\mathrm{fixed\ suffix}}\)。
4. 用实际执行路径上的 downstream gradient 计算局部预测：

\[
\widehat\Delta^{\mathrm{linear}}_{i,j}=\langle g,\delta_i-\delta_j\rangle
\]

5. 比较：

\[
\Delta^{\mathrm{true}}_{i,j}=L_i^{\mathrm{fixed\ suffix}}-L_j^{\mathrm{fixed\ suffix}}
\]

固定后缀行动只适用于 Cell 对 Cell 的替换；后缀动作合法性必须检查。对于允许后续动态重规划的真实部署价值，另报告 `resampled suffix`，不能与固定后缀真值混称。

### 6.2 记录两个不同的相关性

`local_agreement` = 局部一阶 \(\widehat\Delta\) 与固定后缀 \(\Delta^{true}\) 的符号一致率／Spearman。

`long_horizon_agreement` = 局部一阶结果与独立 Monte Carlo 重规划后缀的候选收益的符号一致率／Spearman。

理论上第一个可以较好，但第二个很差；这表明 LCG 有局部合理性，但无法代理全局路径价值。两者应分开判断。

### 6.3 噪声控制与正负对照

- 仅评估有限 MC 差异置信范围可辨识的样本作为主有效性汇总，同时报告全部样本和被排除比例，避免事后筛选夸大效果。
- 与随机置换候选输出的零假设、只选相同 Cell 的零差异、解析线性 Cell 玩具模型对照。
- 不用单个真实目标 Token 的一次 CE 差异冒充已知条件期望；固定后缀的单标签差异应说明解释范围。
- 记录绝对 \(\|\delta_i-\delta_j\|\)、\(\|g\|\)、代理 margin、真实 loss gap、在不同深度上的可靠性。

### 6.4 判定门槛是研究提议，必须在看真实结果前冻结

建议先以 CPU 合成任务验证数学和接口正确性；GPU 真实数据 Pilot 的 go/no-go 以以下**预注册探索性**阈值为参考：

- 固定后缀可辨识候选对至少 100 对，来自至少 32 个独立故事和不同深度；否则标记 `NOT_IDENTIFIABLE`，不以低样本结论申请 500 Step。
- `local_agreement` 的故事簇 Bootstrap 95% CI 下界大于 0.50，且相对随机对照有改善；并报告各深度失败模式。
- `long_horizon_agreement` 不设必须 >0.5 的硬门槛，而是作为是否保留 Q 长程监督的诊断；如果它显著差于随机，不应直接部署只用局部梯度的 Connection。
- 如不达上述阈值，仅做机制分析与负结果报告；不要通过临时调整超参数来制造过线结果。

这些阈值并非外部研究公认常数；它们是为避免无限调参而提出的本次开发门禁，最终需在训练前冻结到 config。

---

## 7. 实验组设计：把“额外计算”“直接竞争”“长期 Q”拆开

### 7.1 三个核心组（同容量、同 Token、相同 Seed 初始化）

| Arm | 主要训练信号 | 研究作用 | 正式推理 |
|---|---|---|---|
| `Q-R3` | R2 的 CE + relative-Q + Rank；**没有 LCG** | 重新训练的同条件 Q-first 主基线；不得只引用历史 R2 | 相同硬 Connection |
| `QC-R3` | 与 Q-R3 完全相同，**额外增加 LCG** | **核心干预组**，只检验竞争信用是否带来价值 | 同 Q-R3，绝无额外候选计算 |
| `QS-R3` | Q-R3 的损失 + 同等候选辅助 forward，附加梯度经标签打乱 / 零均值 sham 控制 | 检查额外计算、辅助噪声或附加优化本身的影响 | 同 Q-R3 |

`QS-R3` 为重要负对照：只有在 QC 超过 Q 且超过 QS 时，才可以更有理由将收益归于**方向正确的竞争信用**。Sham 的随机打乱应该在配对状态块内、保持梯度范数或采用零竞争梯度双版本；必须明确 sham 不是新的有利训练算法，并避免通过无意义的大噪声把基线人为做差。

### 7.2 辅助拆解组（不要求首轮全部 500 Step）

| Arm | 内容 | 用途 |
|---|---|---|
| `C-R3-DIAG` | LCG-only，保持同一个硬路径、相同 STOP/CTX 监督边界；屏蔽 Cell 部分 Q 梯度 | 检查局部信号在没有 Cell-Q 直接拟合时是否能驱动分化；不是最公平的主要 NLL 对照 |
| `FIX-R3-DIAG` | 固定、均衡的条件路由，Cell 从零训练 | 检验稳定分配自身是否足以产生分化 |
| `FULL-PREFERENCE-DIAG` | 小样本固定后缀候选损失直接训练偏好 logits，不使用局部线性代理 | 局部梯度上限／失配诊断，成本单独计算 |

可在 50~100 Step、小型合成数据或冻结 R2 Checkpoint 做；结果只作机制诊断，不能与主对照互相替代。

### 7.3 数据与模型冻结

使用 R2 相同的 `roneneldan/TinyStories` revision、Tokenizer SHA、数据窗、有效 batch、序列长度 64、width 128、六个正式 Cell 和单个 reserved Cell。Train / Calibration / Admission / Development 互不重叠；Confirmation 数据保持封闭。固定 main/candidate 两套模型随机初始化的命名参数一致性。

不允许为了表现更好把 R3 换成混合故事/数学而与历史 TinyStories 单域比较。多领域实验作为**第二阶段独立研究**（第 11 节）。

### 7.4 随机性与计算预算

建议预注册 Seed：`151001301`, `151001302`, `151001303`。第一批仅 Seed 1 的三主臂 exploratory development；看到 Pilot 和首轮指标后，若符合预冻结 Go 规则，才运行 Seed 2、3。无论如何，不能将单 Seed 的 story bootstrap 当作随机种子重复。

R3 起点建议每组 500 Step、8×64 有效 batch、256,000 Token、16 次最大计算预算；同一时间顺序数据。LCG 每步在实际执行的 Cell 节点采样 **32** 个独立事件，候选集合 `K=3`（实际 Cell + 2 个不同 Cell）；若不足则如实记录，禁止重复样本填数。优先在 4 个深度桶、`ctx_before ∈ {0,1}`、不同 Cell 中分层抽样。记录实际 inclusion。Q 的 2 个 full-suffix oracle 状态可暂与 R2 完全一致。

LCG 初始建议：\(\tau_c=1\)（只作为可复现起点，不代表合理最终温度），\(\beta_C=0.1\)（**仅供 Pilot 梯度尺度检查，不得未经 Pilot 直接冻结为正式值**）；根据 `grad_ratio` 选择一次且预先冻结，目标使新增 Router 梯度不过度压倒 Q 梯度，例如中位数比约 0.25–1.0，作为工程保护区间而非科学最优值。

### 7.5 训练效率公平需要两种结论

- **等数据/等 Step 结果**：三个 Arm 使用完全相同的训练样本与更新次数，比较“每 Token 的有效学习”。QC/ QS 多出的候选 forward 成本必须报告。
- **等训练计算预算结果**：以实际 GPU operator calls、同步用时、训练 FLOPs（明确包含反事实和候选计算）进行 Compute-Pareto 比较；允许 Q-R3 使用等量额外 Oracle 采样或更多更新，但该比较必须作为单独标签、独立设置，不得与等 Step 结果混淆。

对于同一模型内 `greedy / sampled / uniform / constant` 的测试，比较的是部署策略，不是同分布训练算法；必须同时报告 NLL 与每 Token 真实激活计算成本。

---

## 8. 核心实现算法：最小侵入、保留 R2 的完整工程安全边界

### 8.1 推荐模块结构（独立 R3 命名空间）

```text
research/projects/cellular-language-models/experiments/
  clm-001-qfirst-integrated/
    r2/                                 # 只读历史
    r3/
      README.md
      SPEC.md                             # 本文的仓库版，记录 SHA/变更
      config/proposed.json
      implementation/clm001_r3/
        competitive_credit.py             # 新增：局部代理数学及候选选择
        probing.py                        # 新增：事件捕获、分层采样、下游梯度采集
        training.py                       # R2 派生，增加独立 router surrogate 梯度
        rollout.py                        # 保留硬语义，只暴露指定的梯度观察点
        audit.py                          # frozen local-vs-exact-suffix 校验
        specialization.py                 # 分化、稳定性、自强化时间序列
        evaluation.py                     # 修复 token0 校准偏差
        oracle.py                         # 尽量复用 R2 immutable Snapshot
        lifecycle.py / events.py          # 复用 R2 事务式候选生命周期
        checkpoint.py                     # 独立 R3 identity + 续训
        runner.py / cli.py / publication.py
      notebook.ipynb                      # 一次 Kaggle 运行及预检
      reports/                            # CPU 校验／源代码审计

tests/research/clm001_r3/
.github/workflows/clm001-r3-cpu.yml
```

**禁止覆盖 R2 的源代码、checkpoint、結果分支、科学结论或修改 R2 发布事实**。采用独立 namespace、新 config hash、新 source SHA 和新 run identity。检查点不跨版本盲恢复。

### 8.2 Training Step 的明确执行顺序

```python
# 逻辑伪代码；具体 API 由 Codex 与现有 R2 接口对齐。
# 同一 run identity；样本和 RNG 不得从未来数据或 Oracle 标签泄漏。

accepted_snapshot = freeze_read_only_snapshot(model)
backup = complete_transaction_backup(model, optimizer, all_rng, nursery, step)

try:
    optimizer.zero_grad(set_to_none=True)

    # A. 完整真实硬路径；前向结果和 R2 语义一致。
    out, selected_events = model.hard_rollout(
        train_batch, collect_gradient_events=32,
        event_sampling_seed=separate_rng,
    )
    ce = terminal_cross_entropy(out, targets)
    ce.backward()  # 仅真实选中的 Cell 获得任务梯度

    # B. 使用已经传播到执行后状态的 downstream dL/dh。
    events = select_target_free_events(selected_events)
    local_records = []
    for event in events:
        assert event.action_type == 'CELL'
        g = event.post_state.grad.detach()
        h = event.pre_state.detach()
        K, inclusion = sample_competing_cells(event.action, seed_rng)
        with torch.no_grad():
            deltas = [cell_delta(model, h, a) for a in K]
        local_records.append((h, g, K, deltas, inclusion, event))

    # C. 只有 QC 启用 LCG；候选输出与下游梯度被 detach，
    #    scorer 输入也 detach；不能把候选 CE 直接梯度传给未执行的 Cell。
    if arm == 'QC-R3':
        credit = local_competition_loss(model.connection, local_records)
        (beta_c * credit).backward()

    # D. Q loss 与 R2 一样在 immutable snapshot 下做两点后缀探测。
    q_records = immutable_oracle_labels(accepted_snapshot, ...)
    q_loss, ranking = relative_q_loss(q_records, ...)
    (beta_q * q_loss + beta_rank * ranking).backward()

    # E. 梯度来源独立审计；group clip/finite/rollback 沿用 R2。
    validate_finite_and_trainable_grads(model)
    clip_groups_continuous_connection_independently()
    optimizer.step()
    verify_finite_and_invariants()
    commit_step_and_version()

    # F. Checkpoint 先于任何 Shadow 生命周期事件；生命周期每单位事务提交。
    save_if_milestone_or_event_pending()
    process_lifecycle_events_with_isolated_private_failure()
except Exception:
    restore_entire_transaction(backup)
    raise
```

**特别注意**：此伪代码隐含一个实施陷阱。R2 当前微批次 CE.backward 后再处理 Oracle；如果要准确获取 `post_state.grad`，必须在真实图中捕获每个选择事件**实际执行后**且参与最终预测的 Tensor，并在 `.backward()` 前 `retain_grad()` 或添加受控 Hook。完成 backward 后，该图可能释放，但 LCG 要用 detach 的常量重新执行 Router 前向，不能引用释放后的图。任何重算必须基于相同参数快照，且不能把第一个 optimizer step 提前。

另外，如果同时处理 Q 与 LCG，两个损失需要对相同预更新 Router 参数求导；必须分别记录梯度但不能先 step。可用 `torch.autograd.grad` 或在同一参数快照上执行独立 loss backward；实现需处理 PyTorch 图释放与梯度累积问题。

### 8.3 问题：在全序列硬训练中，到底何处读取 g？

`g=∂CE/∂h_after` 必须对应“某个 Token 当前动作后、该 Token 的后续硬路径及最终 logits”的完整反向影响。不能简单使用 `∂CE/∂h_before`，不能复用另一个 Token 的 g。若纯粹需要局部实验，也可在某个冻结 Probe 上 replay 完整真实后缀并精确获取梯度；两者应由合成单测证明一致。

### 8.4 处理候选集合采样的偏差

子集 Softmax 代理是明确的**候选子集目标**。K=3 时不声称估计了全六 Cell 的无偏策略梯度。记录候选 inclusion；对固定 Probe 审计，可穷举 6 个 Cell 计算参考全动作梯度，将子集估计与参考方向比较。若子集系统性忽略少数优质 Cell，改进候选选取要成为下一轮独立消融，不能悄悄修改结果。

### 8.5 数值稳定与尺度

- 真实 Continuous CE 梯度和 Connection 梯度分别裁剪，保留 R2 `finite_named()` 逐参数诊断。
- LCG 的梯度范数单独记录；记录 `cos(g_C,g_Q)`、`grad_norm` 分位数、比例异常次数。
- 对空候选、同一动作重复、无 `g`、NaN/Inf、选中 Shadow 未接受、CTX/STOP 混入 Cell 竞争立即失败。
- 对代理 `softmax` 的温度作 logit-scale 审计，不能由于 Q 差值小而让门控永远近似均匀；但温度本身不能在看到开发结果后临时挑选。
- 任何正式 accepted 权重/Oracle 快照污染、主梯度非有限或 Checkpoint 损坏仍 fail-closed；Shadow 私有候选故障按 R2 局部隔离。

---

## 9. 功能分化与自强化：必须测量的指标

### 9.1 首要指标：真实条件交叉优势（Conditional Crossover）

对于冻结模型、固定 Eval 状态分组 \(G_A,G_B\) 和任意 Cell 对 \(i,j\)，计算配对重放固定后缀的差值：

\[
D_{ij}(G)=\mathbb E_{s\in G}[L(s,j)-L(s,i)]
\]

若 \(D_{ij}(G_A)>0\) 且 \(D_{ij}(G_B)<0\)，并且两者的簇 Bootstrap 置信区间分别不跨 0，称为**条件交叉优势候选**。再要求该现象在另一个独立窗口或 Seed 上方向稳定，才可报告“跨样本验证的分化证据”。

为了避免大量 Cell 对 × 状态组的多重比较，需预冻结少量组/对比（例如按初始固定特征聚类的 2–4 个组），或进行多重检验校正并注明探索性。

### 9.2 路由偏好是否与收益相符

测量同一 Probe 下：`oracle_best_cell`, `router_top_cell`, `paired_regret`, `group_conditional_cell_probs`, `policy_tie_rate`；按 depth、是否已 CTX、前一个 Cell、Token 位置分桶。报告选择偏好和实际优势的配对准确率，而非只看路由熵。

### 9.3 真实自强化的纵向指标

固定 Epoch/Step 快照：建议 0、50、100、200、300、400（Shadow Admission 前）、500。对相同静态 Probe 执行：

- \(A_{ij}(G,t)\) 与 \(R_{ij}(G,t)\) 的轨迹图。
- `advantage_leads_route`: 先前 Cell 优势能否预测后续路由偏好增加（与固定／打乱对照比较）。
- `route_leads_advantage`: 先前选择频率增加是否伴随后续该组实际收益扩大（不能以相关代替因果）。
- `exposure_histogram`: 每个 Cell 在不同组、不同深度获得多少真实梯度样本。
- `routing_fluctuation`: 同一固定状态在模型检查点间选择是否大幅波动。

**最强机制证据**来自事先登记的小规模路由干预：在同一模型快照克隆两份，一份保留条件偏好，另一份按组内边际调用次数匹配但打乱具体 Cell 分配，再分别训练相同数据的一小段，比较功能交叉优势是否持续扩大。该干预应从相同参数开始、严格计算匹配，若开销不足则标为 `CAUSAL_FEEDBACK_NOT_TESTED`。

### 9.4 语言质量与质量–成本 Pareto

主测：Development story-cluster paired `NLL`, `PPL`, `actual_STOP`, `forced_end`, `CTX_call_fraction`, `context_read_before_stop`, `self_loop`, `avg_path_length`, `normalized_operator_cost`, `training_FLOPs`, `inference_FLOPs`, `latency`。

必须比较 QC、Q、QS 各自的 `greedy/sampled/uniform/constant`，并给出**同一模型的 Greedy vs 自身 Uniform**。不允许只报告 QC vs Q Greedy，而隐藏自身随机路由更好的可能性。

### 9.5 长程 Cell 组合指标

对于长度为 2 的有序 Cell 对：比较 `i→j` 与 `j→i`、`i→i` 和同计算预算的替代路径的冻结、固定后缀 CE。记录 `pair_synergy`, `ordered_transition_value`, `conditional_context_dependency`。避免用单步优劣替代有序组合；可能出现**单 Cell 没有条件差异，但有序双步路径有显著差异**，这属于值得报告的 CLM 特有结果。

---

## 10. 校准与统计：修复 R2 中的测量缺陷

### 10.1 修复 Calibration 固定首 Token 偏差

R2 代码中固定 `(token_index=0, depth)` 的探测，在输入窗口内几乎没有可访问的历史 Token。R3 改为预冻结的 `(story_id, window_id, token_position, depth)` 网格：

- `token_position` 覆盖序列开头、前 1/4、中位段、后 1/4；至少 4 个桶。
- `depth` 覆盖 0–3、4–7、8–11、12–15；若状态已 STOP，标记覆盖缺失，不重新抽取到“成功出现”然后隐藏失败率。
- 额外分 `CTX_seen` / `CTX_unseen`，从真实执行历史推导，不能由目标标签决定。
- 若某个 Token 处于窗口第一位置且存在 past cache，必须明确记录是否有可用 past Memory，不得将窗口0默认为拥有更早历史。

### 10.2 统计单位

训练数据分组与 Development/Calibration/Admission/Confirmation 完全分离。置信区间优先用原始 `story_id` 做簇重采样，避免把一个故事的多个 Token 视为独立样本。**三个随机 Seed 的效果差异是外层复现单位**，不能用单 Seed 的 128 个故事 CI 代替。

### 10.3 主分析与预注册终点

**Primary Endpoint P1**：QC 相对 Q 的 `same-model greedy minus own uniform` cost-aware utility 改善，且 QC 不存在全 Cell 单路由坍缩。这里“改善”要求与 Q 的相应差值成对比较（差中之差），并报告 NLL 与 cost 分量；避免被单独质量变化误导。

**Primary Mechanistic Endpoint P2**：QC 的 `conditional_cross-over advantage` 和 `router-choice agreement` 同时高于 Q；组内 matched-shuffle 干预能削弱这种效果（若未做干预，限于关联结论）。

**Supportive Endpoint S1**：LCG gradient 与冻结后缀反事实方向一致率显著优于随机；`Spearman`、`pairwise_accuracy` 和 `policy_regret` 不退化。

**Supportive Endpoint S2**：CTX/STOP 不退化，且 Q 长程收益并未被 LCG 的短程偏好系统性破坏。

### 10.4 阈值冻结，禁止事后报捷

建议单 Seed 开发门槛（**研究提议，非统计学定论**）：

- QC 在 Greedy vs 自身 Uniform 的 utility 差值较 Q 的对应差值改善，且 bootstrap CI 的方向一致；若开发阶段证据不够，只标 `INCONCLUSIVE`。
- QC `router-choice agreement` 高于 Q 与 QS 的观察差异，且置信区间/种子汇总不显示稳定反方向；如仅一个 Seed，不声称正式统计确认。
- 至少出现 2 组、2 个不同 Cell 的有条件交叉优势候选，并经过独立审计集重新测量。
- 没有严重 `CTX_COLLAPSE`, `STOP_COLLAPSE`, `ALL_ONE_CELL_COLLAPSE`, `SHADOW_INTEGRITY_FAILURE`。

正式认可 H1 至少需要 3 Seeds 的效果方向一致，以及独立未用于调参的评估数据；认可 H2 还需要纵向/干预的因果链证据。本轮默认只是 Development 探索，`scientific_decision=false`。

---

## 11. 故事、数学与数据多样性的附属研究（禁止混入主结论）

先回答竞争信用能否在 R2 TinyStories 条件下推动功能分化。后续再以**独立实验 ID**研究数据域：

| 任务组 | 数据 | 排除的混杂因素 |
|---|---|---|
| `Synthetic-Compute` | 复制/检索、简单多步计算，统一词表和模板长度 | 避免只根据字母/数字词面实现表面路由 |
| `Story-only` | R2 TinyStories | 历史可比性、低成本观察 |
| `Mixed-Story-Math` | 固定比例、相同 Token 预算、清晰标注来源 | 测量混合任务是否产生更清晰条件交叉优势 |
| `Story→Math` | 固定训练总量与阶段分割，配合 replay | 检查持续学习、路由再组织和遗忘 |

要同时报告领域样本数、梯度余弦、任务正确率或 CE、路由互信息和**功能性交叉优势**。互信息高只说明路由识别领域，并不说明它学会不同的计算机制。任何多领域收益不能代替本轮主实验的条件信用判断。

---

## 12. Shadow 生命周期：让稳定演化与竞争学习不冲突

### 12.1 R3 主实验 Shadow 政策

保持 R2 时间表（Cell step100、Connection + Path step200、Admission step400）。主实验统一 Admission 条件、私有训练预算、候选评价窗口、失败隔离。**主机制评估首选 Step 400 admission 前的模型快照**，确保正式容量没有因候选晋升发生变化。Step 500 则观察完整系统结果。

R2 所有候选均拒绝，因此下一轮不要因急于看到“生长”而放宽接纳阈值。若新竞争机制使某组 Shadow 晋升，而其他组没有，必须在报告中注明结构容量混杂，首要机制比较仍以 Step400 pre-admission 为准。

### 12.2 Shadow Connection 的下一阶段研究位置

R3 不默认采用“每次 LCG 更新只修改 Shadow Connection、定期晋升”的机制；这样会同时改变训练频率和信用通道。可在 R3b 增加慢更新、候选实验、晋升门禁，验证连接稳定性是否进一步促进分化。

### 12.3 R2 工程故障保护全部继承

保留：`NumericalFailure` 逐参数定位、`OracleUntrusted` fail-closed、候选本地 `FAILED/REJECTED`、accepted model witness、完整 RNG 事务回滚、Step400 主训练先 Checkpoint 后 Admission、`publish-only` 不训练、独立 Worker partial metrics。R3 新增代理梯度异常必须作为新失败类型写入故障包，并标记是否污染 accepted；不得吞掉正式主模型失败。

---

## 13. 运行和资源预算

### 13.1 Kaggle T4×2 的调度

建议仍用已有 Kaggle Notebook 的输入输出模式。先跑 CPU Gate → 单卡 synthetic/CUDA Gradient Pilot → 双卡并发 3-arm 轮次 → 分析。T4×2 只能同时运行两个 worker；第三个按确定顺序排队，不得将 GPU0/1 不同硬件波动混作模型差异；多个 Seed 之间轮换 GPU 分配顺序并记录卡名。

R2 每臂训练约 19–21 分钟（具体以该轮资源账本为准）；LCG 每步增加至多 32×2 个未选中 Cell 输出计算，以及梯度收集、审计开销。**没有真实 GPU Pilot 前，不作 R3 固定运行时估算**。预检时记录同步 `p50/p90_step_seconds`、模型显存峰值、附加候选执行量和完成 500 Step 的安全系数预测。

### 13.2 时间/资源超限处理

- 不因任意假定时间上限截断训练并把结果标为 COMPLETE；若需要分段，必须准确续训同一 identity。
- GPU Pilot 若显示超出当前资源能力，先降低 LCG 的抽样数或候选数，但这会形成**新的冻结配置与所有主臂一致的新 run identity**；不能在运行中悄悄调整。
- 训练成本对照与推理成本对照分开；LCG 的额外候选计算只出现在训练。

### 13.3 CPU/GPU 预检

CPU：数学解析测试、finite-difference、硬路径前向完全相同、Cell-only 限定、不泄漏目标、负对照无有效方向、故障注入、R2 回归、Checkpoint 中断恢复、候选签名与源 SHA、统计分组无泄漏。

GPU：真实模型反向 Hook 与 `post_state.grad` 验证、至少一个真实 `LCG.backward`、Q+LCG 混合梯度、Shadow 三候选 probation、数值异常隔离、主模型回滚、重启 continuation parity、两卡资源投影。

绝不能用 CPU 的模拟时间、空梯度或假冒的 GPU Pass 填报告。

---

## 14. 执行分期与门禁（先证明代理可用，再验证系统收益）

### Phase A — 冻结研究设计（仅 CPU，不训练）

审计 R2 结果、R2 模块、Tokenizer、数据窗和 R2 Checkpoint identity。生成 `r3/ARCHITECTURE_AUDIT.md`, `r3/BASELINE_EVIDENCE.json`；冻结主对照变量、候选选择规则、随机种子、统计终点；若 R2 checkpoints 不在 GitHub，不能假设可从 GitHub 下载。开发运行配置中 `development_training=false`。

### Phase B — 局部竞争梯度与正确性（CPU）

实现小型玩具 Cell/Router 的解析梯度，检验 LCG 的符号、数值、全候选 K=N 极限；确定外部 Tensor `g` 与后缀真实梯度一致。确保前向 `h_st` 与 `h_hard` 逐元素完全一致，A 方式无额外候选梯度写入 Cell。所有 gate 必须在生成 GPU Notebook 前通过。

### Phase C — 冻结后缀方向审计（CPU 小规模 + 真实 GPU Pilot）

覆盖各 Token 位置、depth、CTX 前后；导出局部代理与固定后缀完整反事实的配对结果；检查 100 个可辨识对/32 故事、95% CI 和偏差。测试伪梯度不作弊、真实 CUDA backward、候选失败隔离和 restart。若方向显著失败，不进入完整 R3-H 主训练，保留科学负结果。

### Phase D — Seed1 三主臂 500 Step 开发训练

`Q-R3`, `QC-R3`, `QS-R3` 新模型从零独立训练，不能用 R2 后期状态热启动来声称随机初始化有效；可额外单独做冻结 R2 Checkpoint 机制对照。Admission 前与最终视角各自报告；实时 checkpoint 每50步，诊断对照在 0/50/100/200/300/400/500。

### Phase E — 归因审计

优先回答：`局部代理是否同号` → `Router 条件选择是否改善` → `真实反事实功能分化是否存在` → `分化是否经由训练机会自强化` → `语言质量与成本是否获益`。任何前一关键环节不成立，后续改善不得被上纲为“自强化机制成功”。

### Phase F — 多 Seed 重复和后续模型设计决策

仅当开发门禁通过，执行剩余预注册 Seeds；锁定分析代码后汇总。失败时根据 F1–F4 类型选择后续方向：增强 Cell 异质训练信号、改进局部梯度估计、设计独立路径竞争、或改善 CTX/STOP 信用，但不得在当前 run 身份下改配置重试。

---

## 15. Codex 工程任务清单（依赖序）

| ID | 任务 | 代码区 | 验收证据 |
|---|---|---|---|
| R3-01 | R2 基线与原始结果只读审计 | `r3/ARCHITECTURE_AUDIT.md` | 引用 SHA、仅支持已核验事实 |
| R3-02 | 独立 schema/配置与授权 | `r3/config/proposed.json` | R2 identity 不复用、Confirmation 禁止 |
| R3-03 | 真实 hard path 上捕获 pre/post-state gradient | `probing.py`, `rollout.py` | 前向 parity、梯度逐状态测试 |
| R3-04 | K=3 Cell 局部竞争输出与 Router loss | `competitive_credit.py` | finite-diff、无候选 Cell 梯度 |
| R3-05 | Q/LCG 两种梯度审计与分组裁剪 | `training.py` | loss scale、cos、NaN fail-closed |
| R3-06 | 冻结后缀 Oracle 一阶信号审计 | `audit.py` | 可辨识数／CI／负对照 |
| R3-07 | Q、QC、QS 的真正同条件控制 | `config`, `runner` | 参数名/初始化哈希一致 |
| R3-08 | CTX、STOP 与 hard-path 完整性审计 | `model.py`, `evaluation.py` | STOP 禁额外操作、CTX 不免费 |
| R3-09 | 偏移 Token/深度分层校准与所有行为对比 | `evaluation.py` | token0 偏差消除、覆盖缺失率 |
| R3-10 | 功能分化与自强化时间轴 | `specialization.py` | 固定/动态 state 分别报告 |
| R3-11 | Shadow 事务回归与三类故障测试 | `events.py`, `checkpoint.py` | 400前主 checkpoint、主模型 witness |
| R3-12 | 同 Token/同计算预算资源计量 | `profiling.py` | CPU/GPU operator calls、实际 FLOPs |
| R3-13 | 独立 Worker、部分报告、metadata-only GitHub publish | `publication.py` | 失败仍保留各自报告，不上传密钥 |
| R3-14 | Notebook/CI/完整文档 | `notebook.ipynb`, workflow, `SPEC.md` | Run All + 合规 Gate + 原 R2 回归 |

---

## 16. 发布、日志与科学结果格式

### 16.1 GitHub 结果：必需文件

```text
results/clm001-r3-<source12>-<run12>/
  .../r3/artifacts/<full-source-sha>/<run-id>/
    cpu-gate.json
    gpu-pilot-summary.json
    parameter-audit.json
    arm-Q-R3-metrics-summary.json
    arm-QC-R3-metrics-summary.json
    arm-QS-R3-metrics-summary.json
    competition-gradient-audit.json
    local-vs-suffix-agreement.json
    specialization-summary.json
    feedback-longitudinal-summary.json
    q-calibration-summary.json
    stop-ctx-path-summary.json
    pre-admission-contrast.json
    final-system-contrast.json
    resource-ledger.json
    shadow-candidate-summary.json
    failure-index.json
    combined-report.json
    publication.json
```

如果只有 Pilot，必须明确 `NOT_TRAINED`；如果只完成一组，必须 `PARTIAL`，不可伪造两组平均。

### 16.2 Kaggle Output：不可上传到公开 GitHub 的完整证据

保留每个 Arm 的 weights / optimizer / RNG checkpoint、事务日志、局部竞争逐事件配对诊断、完整 Oracle 标签、每一步实际路径、原始 token/目标、用户秘钥之外的完整追踪、OOM/梯度错误栈。严格区分可公开的 aggregate metadata 和私有/体量较大的原始研究资产。GitHub 只允许白名单小型汇总 JSON，且必须附哈希与源身份。

### 16.3 建议 `train_step` 新字段

```json
{
  "step": 200,
  "arm": "QC-R3",
  "live_ce": null,
  "q_loss": null,
  "rank_loss": null,
  "lcg_loss": null,
  "lcg_events": 0,
  "lcg_candidate_cell_calls": 0,
  "lcg_depth_bucket_histogram": {},
  "lcg_grad_norm": null,
  "q_grad_norm": null,
  "lcg_q_cosine": null,
  "cell_exposure_by_group": {},
  "cell_router_entropy": null,
  "actual_stop_rate": null,
  "forced_end_rate": null,
  "ctx_reads": 0,
  "train_flops_estimate": null,
  "oracle_operator_calls": 0,
  "source_sha": "<locked>",
  "config_hash": "<locked>",
  "scientific_decision": false
}
```

上述 null 为 **schema 示例，不是实际指标**。凡缺失指标必须保留 `null` 或 `NOT_AVAILABLE`，不准将其替换成 0。

### 16.4 结果状态机

`DESIGN_ONLY` → `CPU_PASS` → `GRADIENT_PILOT_PASS` → `LOCAL_ALIGNMENT_PASS/INCONCLUSIVE/FAIL` → `DEV_TRAIN_COMPLETE/PARTIAL` → `MECHANISM_SIGNAL/NO_SIGNAL` → `MULTISEED_REPLICATED/NOT_REPLICATED`；保持独立 `engineering_status` 和 `scientific_decision`。最终单轮的正式确认默认 `false`。

---

## 17. 实验结果与下一步的决策树

| 实验观察 | 可支持的结论 | 下一步 |
|---|---|---|
| Cell 的冻结候选收益本身无法辨识 | 能力差异弱／数据设置不足；不是“Q 不会学”的充分证据 | 更聚焦的计算任务／单元差异与监督实验 |
| 局部代理方向 ≈ 随机；但完整后缀价值差异明显 | 一阶局部近似不适合当前长程硬路径 | 对照真实 counterfactual policy-gradient / path preference，不继续调 LCG 权重 |
| 代理方向可信，Router 条件选择改善，但 Cell 无交叉优势 | 有短程训练收益，未形成互补分工 | 研究训练机会分配、稳定路由、候选路径学习 |
| Cell 已出现交叉优势但无纵向强化 | “已有专长”可识别，但未证实 self-reinforcing learning | 固定/打乱路由的因果训练干预 |
| QC 比 Q 和 QS 都有更好路径收益、干预证实优势扩大 | 支持竞争信用促进自强化分化的探索性结论 | 三 Seed 复现与扩展 CTX/STOP 机制 |
| QC 语言质量好但 Greedy 仍不如自身 Uniform | 训练收益不等于策略学习成功 | 优先检查 Q 与 LCG 冲突和路径 policy improvement |
| 代理导致 CTX/STOP 大幅退化 | 局部信用与全局资源选择失衡 | 将 CTX/STOP 作为独立长程动作诊断，不伪造完备成功 |

---

## 18. 可直接交给 Codex 的执行指令

> 在 `ArcheLabs/mini-cells` 新建隔离分支 `codex/clm001-r3-competitive-credit`，以 source SHA `4fcb95490c2ca68affbab6881135f2e85e5bea87` 的 R2 实现、结果与工程事务边界为只读基线，实现本协议 **CLM-001-R3**，不要修改 R1/R2 历史数据。优先完成 Phase A–C：CPU 解析梯度验证、真实硬路径前向 parity、候选 Cell-only 辅助竞争梯度、Oracle 固定后缀真值对齐、原始 R2 工程故障回归和真实 CUDA Pilot。不要未经门禁直接运行 500 Step。
>
> 核心训练方法为：保持 R2 真实硬执行，选定少量实际 Cell 决策点，额外计算 K=3 个候选 Cell 输出，用硬路径下游 CE 梯度构造 detached local first-order credit，训练 Connection；Q long-horizon loss 同时保留。主对照 `Q-R3 / QC-R3 / QS-R3`，所有模型从零初始化、同参数/数据/评估规则，额外训练计算独立记账。任何候选 Cell 不得从辅助前向得到梯度，CTX/STOP 不得被伪装为 Cell，正式推理不得执行辅助候选。保存完整 Cell 条件功能、路由与反事实收益随训练变化的证据。
>
> 真实 Probe 上通过局部梯度—固定后缀反事实方向审计之后，才可冻结 `development_training=true` 进行单 Seed 的 500 Step 三臂开发运行；确认阶段永不自动开放。所有失败必须有原生诊断和安全边界，Checkpoints 保留在 Kaggle Output，GitHub 只回传白名单汇总与出处身份。运行完成后请提交可复现分析，**分别**回答：梯度可信否、路由有改进否、Cell 已分化否、是否构成因果正反馈、真实质量/计算成本是否改善。禁止将运行完成视为科学成功。

---

## 附录 A：建议初始配置补丁（供 Codex 冻结前审查）

```json
{
  "experiment_id": "CLM-001-R3",
  "status": "PROPOSED",
  "source_parent": "4fcb95490c2ca68affbab6881135f2e85e5bea87",
  "arms": ["Q-R3", "QC-R3", "QS-R3"],
  "seed_namespace": "CLM-001-R3/development/v1",
  "seeds": [151001301, 151001302, 151001303],
  "inherit_r2_model": true,
  "model": {
    "accepted_cells": 6,
    "reserve_slots": 1,
    "budget": 16,
    "memory_mode": "static_input_readonly",
    "cell_direct_memory_access": false,
    "connection_extra_context_summary": false
  },
  "competition": {
    "enabled_arms": ["QC-R3", "QS-R3"],
    "method": "detached_local_first_order_credit",
    "default_implementation": "post_hard_backward_router_surrogate",
    "sampled_cell_events_per_step": 32,
    "candidate_cells_per_event": 3,
    "cell_only": true,
    "candidate_includes_executed_cell": true,
    "candidate_sampling": "uniform_without_replacement_with_logged_inclusion",
    "state_buckets": ["depth_quartile", "ctx_seen"],
    "router_input_detached": true,
    "candidate_output_detached": true,
    "actual_path_forward_unchanged": true,
    "tau_competition": 1.0,
    "beta_competition": null,
    "beta_competition_status": "FREEZE_AFTER_GRADIENT_PILOT_BEFORE_DEV"
  },
  "training": {
    "steps": 500,
    "effective_batch": 8,
    "sequence_length": 64,
    "tokens_per_arm": 256000,
    "precision": "float32",
    "checkpoint_every": 50
  },
  "evaluation": {
    "snapshots": [0, 50, 100, 200, 300, 400, 500],
    "fix_r2_token0_bias": true,
    "story_cluster_bootstrap": true,
    "min_identifiable_pairs_for_pilot": 100,
    "min_independent_stories_for_pilot": 32
  },
  "gates": {
    "cpu": false,
    "gradient_pilot": false,
    "local_alignment": false,
    "development_training": false,
    "confirmation": false
  },
  "publication": {
    "github_metadata_only": true,
    "publish_partial_arms": true,
    "upload_checkpoints_to_github": false
  },
  "scientific_decision": false,
  "formal_confirmation": false
}
```

注：这里只列**新增/关键约束**；执行配置必须显式包含并验证 R2 其余数据身份、优化器参数、Oracle、Shadow 和资源字段。`beta_competition:null` 是一个必须在 GPU Pilot 后、正式训练前冻结的待定量，不允许被程序默认为 0 或偷偷填入。任何正式训练配置变化都产生新的 run ID。

## 附录 B：数学检查与预期不变量

**B1. Router 竞争方向**：若 Cell i 的局部增量 \(\delta_i\) 更符合损失下降方向，\(\langle g,\delta_i-\bar\delta\rangle<0\)，梯度下降应增加对应选择概率。构造 2-Cell 一维线性玩具例，检查解析梯度符号。

**B2. Hard-forward identity**：全部 Cell/CTX/STOP 的 `actions`, `hidden`, `logits`, `cost` 在 LCG 开关前后前向完全相等（随机流固定；候选额外采样使用独立 RNG），反向只允许 Router 梯度出现预期差异。注意浮点 forward 相等可以要求 bitwise；PyTorch 不同内核/融合时需判别是否来自实现差异。

**B3. `sg` 边界**：未选中候选 `Cell.weight.grad is None`（如它未在任何真实硬路径中被调用）；多个真实路径调用同一 Cell 时，应比较“开启竞争 vs 关闭竞争”的梯度差，证明 LCG 没有额外贡献。

**B4. STOP**：STOP 在合法状态下没有 Cell 更新，不增加 cost；后缀不执行。执行预算耗尽的 `forced_end` 与主动 STOP 区分。

**B5. Oracle**：同一 frozen identity、合法动作、相同 Memory/目标/CRN，绝不跨模型更新复用旧标签。

**B6. 信号伪造**：如果用随机打乱的候选输出，QS 不能无条件被写成“功能差异真实”；任何能制造漂亮路由模式的损失都必须经真实反事实效用校验。

**B7. 负载**：同条件 Cell 使用率接近均衡不是分化；负载坍缩一定独立报告。

**B8. 干预**：功能测量对不同 Cell 使用相同冻结状态和可比较的 suffix 与 budget；打乱后缀或偷换 Context 则试验无效。

**B9. 可恢复性**：Step400 的正式训练先提交、再处理候选；事件 idempotence、RNG 字节状态和 optimizer 恢复一致；故障不导致重复计数。

**B10. 科学边界**：3 Seeds 的跨随机性稳定性、Confirmation 和跨架构 MoE 对照是不同层次，不因某一次开发结果阳性而自动升格。

## 附录 C：证据出处、术语与阅读建议

### C.1 项目一手资料

- R2 源码基线：https://github.com/ArcheLabs/mini-cells/tree/4fcb95490c2ca68affbab6881135f2e85e5bea87/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r2
- R2 汇总结果：https://github.com/ArcheLabs/mini-cells/tree/results/clm001-r2-4fcb95490c2c-c541c76b/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r2/artifacts/4fcb95490c2ca68affbab6881135f2e85e5bea87/c541c76bc2f1
- R2 关键模块：`r2/implementation/clm001_r2/{training,rollout,evaluation,oracle,runner,checkpoint,events,publication}.py`。
- 既有实施协议：`CLM_001_R2_实施与实验协议.md`，本文件不修改或取代 R2 历史结论。

### C.2 外部论文（目的与可迁移范围）

- Fedus, Zoph & Shazeer (2022). *Switch Transformers*. JMLR, 23(120). 学习负载均衡与稀疏专家优化： https://www.jmlr.org/papers/v23/21-0998.html
- Dai et al. (2022). *StableMoE*. ACL. 学习路由波动导致的样本效率问题： https://aclanthology.org/2022.acl-long.489/
- Liu, Gao & Chen (2023). *Sparse Backpropagation for MoE Training*. 稀疏路由梯度估计问题： https://arxiv.org/abs/2310.00811
- Liu et al. (2024). *GRIN: GRadient-INformed MoE*. 离散专家选择的梯度学习： https://arxiv.org/abs/2409.12136
- Puigcerver et al. (2024). *From Sparse to Soft Mixtures of Experts*. ICLR. 可微专家分配，但架构与 CLM 不同： https://proceedings.iclr.cc/paper_files/paper/2024/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html

### C.3 可执行结论的条件

本研究设计不是“证明 CLM 具有分化机制”的论证文本，而是一份能够**否证**该主张的研究协议。只有当代理信用方向可靠、正式硬推理收益提升、条件功能优势出现并经受训练机会干预、跨独立种子/数据复现，才能逐步升级结论。更强的跨 MoE 性能和跨领域持续学习结论需要另外设置有严格资源匹配的实验。

---

*End of CLM-001-R3 proposal. No code changes, GPU training, GitHub writes or scientific confirmations are implied by this document.*

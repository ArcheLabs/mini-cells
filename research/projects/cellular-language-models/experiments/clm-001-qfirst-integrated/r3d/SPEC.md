# MiniCells R3d — Long-Horizon Credit & Joint Learning

## 长期信用分配与 Cell–Connection 联合训练实验

**文档版本：** v1.0  
**实验 ID：** `CLM-001-R3d-LHCJL-001`  
**目标仓库：** `ArcheLabs/mini-cells`  
**实验性质：** 独立探索性 Development 实验  
**执行模式：** 一次性实现、单次启动、自动完成所有训练与评估  
**科学结论：** 实验完成后自动生成，不预设成功  
**Confirmation：** 保持 SEALED

---

# 1. 实验背景

## 1.1 已有研究结论

MiniCells 已经完成 R3b、R3c 等多个阶段的研究。

R3b 采用硬路由以及 Q/LCG 等信用学习方式。实验表明，局部信用信号可能与最终执行路径的收益不一致，导致 Router 无法稳定改善质量。

R3c 移除了动态 STOP、CTX 路由竞争及原有 Q/LCG，改为固定四阶段、共享六个 Cell、每阶段 Top-2 可微门控。

R3c 三个 Seed 的最终 Development NLL：

| Seed | A 固定均衡 | B 随机均衡 | C 可微门控 | D StopGrad |
|---|---:|---:|---:|---:|
| 151001301 | 4.5760 | 4.6024 | 4.5639 | 4.5493 |
| 151001302 | 4.5708 | 4.5952 | 4.5736 | 4.5574 |
| 151001303 | 4.5649 | 4.5940 | 4.5735 | 4.5460 |

核心结论：

1. 可微 Top-2 门控的工程路径已经成立。
2. C 组并未稳定优于固定均衡路由。
3. D 组在三个 Seed 中均优于 C。
4. C 组存在明显的 Cell 梯度覆盖不足。
5. R3c 最终未实际启用非零负载均衡系数。
6. 当前尚未证明 Cell 已经形成可利用的专业化能力。

因此，R3c 不支持「普通 CE 门控梯度能够稳定改善 Cell–Connection 联合训练」这一假设。

但它也没有否定多 Cell 联合训练的可能性。

## 1.2 新实验的基本方向

R3d 不再尝试通过普通 Top-2 Softmax 梯度直接学习离散的 Cell 选择。

改为：

- Cell 沿实际执行路径接受最终 CE 梯度。
- Connection 根据候选动作的长期反事实收益进行训练。
- 训练期间允许少量反事实路径计算。
- 推理时只执行 Connection 选中的路径。
- 通过控制 Connection 更新频率，减轻路由与 Cell 能力相互变化造成的训练不稳定。

R3d 重点验证训练机制，而非发明新的网络架构。

---

# 2. 科学目标

## 2.1 核心科学问题

在相同 Cell 架构、相同训练数据和相同推理计算预算下：

**通过最终 CE 定义的长期动作价值训练 Connection，能否使其有效选择 Cell，并在 Cell 持续训练时保持正向收益？**

## 2.2 主要假设

### H1：Connection 能够学习有用的长期选择

冻结已经训练好的 Cell 和其他模型参数。

比较：

- 原始冻结 Router。
- 使用长期反事实收益训练的新 Connection。

要求学习后的 Connection 在实际闭环推理中改善最终 NLL。

### H2：长期信用优于即时信用

比较：

- 利用候选 Cell 执行后即时 CE 训练的 Connection。
- 利用候选 Cell 执行后完整剩余路径的最终 CE 训练的 Connection。

检验多步信用分配是否提供独立收益。

### H3：联合训练能够维持或扩大路由收益

允许 Cell 和 Connection 共同更新。

要求学习 Connection 的联合训练系统，在相同 Cell 更新预算下，优于冻结 Router 的继续训练基线。

### H4：训练不会导致严重的 Cell 曝光失衡

联合训练期间，应能够控制长期零曝光现象。

这是一项训练稳定性假设，不意味着最终推理必须均匀选择所有 Cell。

## 2.3 非目标

R3d 不研究：

- 新型 Cell 内部结构。
- 动态 STOP。
- 小世界拓扑。
- Stage 专属 Cell。
- Cell 生命周期与持续学习。
- 模型参数扩张或收缩。
- 新的 Context 路由。
- 跨 Stage 新型网络连接。
- 模型量化、MPT 或硬件推理优化。

上述方向均不允许在本实验中临时加入。

---

# 3. 基础架构

## 3.1 继承 R3c

使用 R3c 已有计算结构：

| 参数 | 设置 |
|---|---|
| Hidden width | 128 |
| Cell 数量 | 6 |
| Cell hidden | 248 |
| Stage 数量 | 4 |
| 每阶段激活 | 2 Cells |
| 总 Cell 调用 | 8 次 / Token |
| Context Read | 1 次 / Token |
| Cell 参数共享 | 是 |
| 动态 STOP | 关闭 |
| 初始 Alpha | 0.125 |
| Sequence length | 64 |
| Vocabulary size | 4096 |

实际配置及历史资产身份以 R3c 冻结运行记录为准，不得因为上述摘要与已保存配置存在差异而静默改写运行配置。

继续使用原始 Embedding、Context、Memory、Normalization、Cell 和最终 LM Head。

不得修改 R3c 冻结产物。

## 3.2 Cell 执行方程

每个 Stage：

\[
h_{t+1}
=
h_t+
\alpha
\sum_{i\in S_t}
g_{t,i}E_i(\operatorname{Norm}(h_t))
\]

其中：

\[
|S_t|=2
\]

四个 Stage 连续更新：

\[
h_0\rightarrow h_1\rightarrow h_2
\rightarrow h_3\rightarrow h_4
\]

每个 Stage 使用更新后的状态重新选择 Cell。

最终损失：

\[
L_{\mathrm{CE}}
=
\operatorname{CE}(\operatorname{Head}(h_4),y)
\]

## 3.3 离散动作空间

六个 Cell 中选择两个不同 Cell，共有：

\[
\binom{6}{2}=15
\]

种无序 Cell 对。

R3d 将每个 Cell 对视为一个离散动作：

\[
a_t=(i,j),\qquad i<j
\]

Connection 的职责是根据当前状态，从 15 个候选动作中选择一个。

本轮不改变每阶段执行两个 Cell 的预算。

这保证 R3d 相比 R3c 的主要变化是选择训练机制，而不是激活计算数量。

## 3.4 新 Connection 与原始 D 组保持初始等价

为了避免更换 Router 架构造成额外变量，保留 D 组已经初始化并保存的原始 Gate。

设原始 Gate 输出六个 Cell logit：

\[
z_1,\ldots,z_6
\]

对于候选 Cell 对 \((i,j)\)，定义基础分数：

\[
b_{ij}=z_i+z_j
\]

增加一个轻量的可训练决策修正：

\[
q_{ij}
=
b_{ij}+\Delta_\phi(h_t,t)_{ij}
\]

其中 \(\Delta_\phi\) 输出 15 个修正值。

初始化时：

\[
\Delta_\phi=0
\]

于是：

\[
\arg\max_{i<j}q_{ij}
=
\operatorname{Top2}(z)
\]

在相同 tie-breaking 规则下，新 Connection 的初始 Cell 对选择与 D 组一致。

选中一对 Cell 后，继续使用冻结的原始 Gate logits 计算该对 Cell 的相对混合权重：

\[
g_i=
\frac{\exp(z_i/\tau)}
{\exp(z_i/\tau)+\exp(z_j/\tau)}
\]

\[
g_j=1-g_i
\]

混合权重在 CE 训练路径中保持 StopGrad。

原始 Gate 参数全程冻结；新 Connection 仅学习修正项。

这样能够最大限度保持与 D 组相同的初始计算语义。

如果完全等价的初始化无法通过最小一致性检查，必须停止运行并报告实现错误，不得继续产生科学结果。

---

# 4. 统一优化目标

定义真实执行路径：

\[
a_{0:T-1}\sim\pi_\phi
\]

联合目标：

\[
J(\theta,\phi)=
\mathbb E_{(x,y)}
\mathbb E_{\pi_\phi}
\left[
L_{\mathrm{CE}}(x,y;\theta,a_{0:T-1})
\right]
\]

本轮固定：

\[
T=4
\]

每一步固定执行两个 Cell，因此不再添加额外的计算成本惩罚。

该目标是 Cell 与 Connection 共同追求的最终任务目标。

具体训练使用不同的近似优化途径。

## 4.1 Cell 学习

对实际执行路径进行反向传播：

\[
\nabla_\theta L_{\mathrm{CE}}
\]

仅对实际执行的 Cell 计算 CE 梯度。

不要求计算全部六个 Cell。

本轮联合训练允许 Cell 参数更新，其他公共模型参数保持冻结，以隔离 Cell–Connection 的相互作用。

这种方法是固定采样路径下的条件梯度近似；由于 Cell 参数会影响后续路由状态，不应将其声明为完整随机路径目标的无偏梯度。

## 4.2 Connection 学习

定义当前状态 \(h_t\)、候选动作 \(a\) 的长期动作价值：

\[
Q_t(h_t,a)
=
\mathbb E[
L_{\mathrm{CE}}(h_4)
\mid h_t,a_t=a
]
\]

其中：

- 当前 Stage 强制执行动作 \(a\)。
- 后续 Stage 根据更新后的状态重新路由。
- 使用最终 CE 评价候选动作。
- 候选动作比较期间冻结 Cell 和 Connection 参数。

这里的 Q 表示成本，数值越低越好。

不使用原 R3b 的 Q/LCG 实现或标签。

## 4.3 长期价值与即时价值

长期价值：

\[
Q_{\mathrm{long}}(h_t,a)
=
L_{\mathrm{CE}}(h_4^{(a)})
\]

即时价值：

\[
Q_{\mathrm{short}}(h_t,a)
=
L_{\mathrm{CE}}(\operatorname{Head}(h_{t+1}^{(a)}),y)
\]

长期价值执行完整剩余计算链。

即时价值只评价执行当前动作之后的状态。

这两项使用相同的基础 Cell、Head 和数据。

比较两者用于检验：多步信用分配是否确实必要。

---

# 5. 训练期反事实估计

## 5.1 基本原则

R3d 不枚举全部计算路径。

只在训练批次中的少量状态上比较候选下一步动作。

每次选择：

- 当前 Connection 的实际首选动作。
- 原始冻结 D 组的首选动作。
- 一个由独立 RNG 采样的探索动作。

去重后得到最多三个候选动作。

## 5.2 采样数量

默认每次策略更新选择 8 个 Token–Stage 状态进行反事实评估。

从当前训练批次真实访问的状态中采样。

Stage 采样覆盖四个阶段，使用独立随机流，避免改变数据顺序或模型随机性。

每个候选动作都：

1. 从相同的已保存状态出发。
2. 执行不同的当前 Cell 对。
3. 使用同一个冻结参数快照。
4. 在后续阶段重新执行当前 Connection。
5. 计算同一目标 Token 的最终 CE。

不得在选择候选动作之前读取目标标签作为 Connection 输入。

目标标签仅用于训练期反事实评分与损失计算。

不得以固定原路径后缀替代重新路由后缀。

## 5.3 不单独训练 Critic

为了减少不必要的工程复杂度，本轮不引入独立的 Critic 神经网络。

直接使用有限反事实执行得到的最终 CE 训练 Connection。

例如，对于两个候选动作 \(a,b\)，有：

\[
Q_a,\ Q_b
\]

定义偏好概率：

\[
p(a\succ b)
=
\sigma\left(\frac{Q_b-Q_a}{\tau_v}\right)
\]

Connection 对这两个动作预测：

\[
\hat p(a\succ b)
=
\sigma(q_a-q_b)
\]

通过二元交叉熵优化两者的一致性。

未执行、未评分的候选不作为具有已知价值的动作处理。

训练目标应清楚标记为 sampled-candidate preference optimization，而不是声称获得全部 15 个动作的精确 Q 值。

该设计减少额外参数和长期滞后估值，但会增加训练期反事实计算成本。

## 5.4 防止短期价值偏差

长期组始终使用完整剩余路径的最终 CE。

即时组使用执行当前动作后的中间 CE。

两者使用一致的 Connection 容量、优化器和训练预算。

不得为了使长期组获胜而给予额外训练 Token 或更多策略更新次数。

---

# 6. 实验对照矩阵

本次实验包含两个连续训练区间，但由同一运行程序自动执行，不设置中途人工审批或额外训练授权。

## 6.1 区间 A：冻结模型能力，验证 Connection

从各 Seed 的 R3c D Step 500 Checkpoint 出发。

冻结全部已有模型参数，包括 Cell、Embedding、Context、Memory、Head 和原始 Gate。

只训练新增 Connection 修正参数。

| Arm | Cell | Connection | 学习目标 |
|---|---|---|---|
| F0 | 冻结 | 原始 D 冻结 | 基础参照 |
| L | 冻结 | 训练 | 长期反事实价值 |
| M | 冻结 | 训练 | 即时反事实价值 |

F0 不需要重复训练，只需完成相同数据窗口上的闭环评估。

L/M 使用相同的训练 Token 数、Connection 参数容量和策略更新预算。

二者各自从同一个零修正 Connection 初始化状态开始。

## 6.2 区间 B：联合训练验证

从相同的原始 D Step 500 模型参数重新初始化三个 Arm，不直接继承区间 A 的任何 Cell 更新，因为 A 中 Cell 没有更新。

| Arm | Cell | Connection | 目的 |
|---|---|---|---|
| F | 更新 | 原始 D 冻结 | Cell-only 基线 |
| P | 更新 | A 中学到的 L，冻结 | 验证已有路由收益能否保持 |
| J | 更新 | A 中学到的 L，继续训练 | 检验持续联合训练 |

三个 Arm 使用：

- 相同初始 Cell 参数。
- 相同 Train 数据顺序。
- 相同 Token 预算。
- 相同 Cell 优化器设置。
- 相同实际推理 Cell 执行数量。

J 使用交替更新：

1. 在当前 Connection 下执行真实训练批次。
2. 用最终 CE 更新 Cell。
3. 使用本次 Cell 更新前保存的参数快照与训练状态计算反事实价值。
4. 用这些价值目标更新 Connection。
5. 下一批次使用更新后的 Cell 和 Connection。

反事实估值使用的 Cell 快照和策略版本必须记录清楚，禁止混用更新前后的中间状态与参数。

为控制策略变化，采用固定的策略更新频率与更新幅度限制。具体数值写入冻结配置，运行期间不得根据 Development NLL 自动调整。

## 6.3 对照解释

主要比较：

- L vs F0：长期路由学习是否有收益？
- L vs M：长期信用是否比即时信用更有价值？
- J vs F：联合学习是否优于冻结路由的 Cell 训练？
- J vs P：持续更新 Connection 是否比仅使用预先训练好的 Connection 更有价值？

此外，P vs F 可以判断冻结后的已学习 Connection 是否具备独立收益。

本实验不因区间 A 缺乏正向收益就中止区间 B。所有预先定义的 Arm 均完整执行，以保留真实负结果及其解释。

但如果出现数据污染、状态损坏、非有限梯度或预算违约，应立即停止受影响的运行，而不是继续生成无效指标。

---

# 7. 训练曝光与路由稳定性

## 7.1 问题

R3c C 组出现大量至少一个 Cell 零梯度的 Step。

R3d 不应允许 Connection 在训练初期完全抛弃某些尚未成熟的 Cell。

## 7.2 探索策略

联合训练时使用：

\[
\pi_{\mathrm{train}}
=
(1-\epsilon)\pi_\phi
+
\epsilon\pi_{\mathrm{explore}}
\]

其中：

- \(\pi_\phi\) 为 Connection 策略。
- \(\pi_{\mathrm{explore}}\) 为覆盖合法 Cell 对的探索分布。
- \(\epsilon\) 随预先冻结的训练进度衰减。

探索分布不依赖真实目标标签。

建议冻结如下初始候选设置，允许 Codex 在实现前依据已有配置统一确定，但不得依据 R3d Development 结果选择：

- 初始探索率：0.15。
- 最终探索率：0.02。
- 线性衰减。
- 推理探索率：0。

这些数值只是探索性实验的固定配置，不是已验证的理论最优值。

## 7.3 训练稳定性指标

每次参数更新保留轻量聚合统计：

- 各 Cell 总选择次数。
- 各 Stage 的 Cell 曝光分布。
- 每个 Cell 的 CE 梯度范数。
- 至少一个 Cell 零梯度的 Step 比例。
- Connection 相对原始 D 路由的动作变化比例。
- Cell 与 Connection 的训练损失。

不要求额外生成每个 Token 的完整路由轨迹或逐层大规模探针。

这些摘要直接从正常训练过程中已有的数据统计，不增加额外模型前向。

## 7.4 不强制推理均衡

训练曝光不是推理目标。

R3d 不应为了让所有 Cell 在推理时具有相同选择率，而强制修改最终 Connection。

最终只关心：

- Cell 是否有充分训练机会。
- 不同 Cell 是否存在可利用的能力差异。
- Connection 是否改善最终质量。
- 推理预算是否保持一致。

---

# 8. 固定训练预算

## 8.1 基础设置

采用 R3c 的原始训练数据、Tokenizer、模型设置和优化器基线。

训练 Seed：

```text
151001301
151001302
151001303
```

建议固定：

| 项目 | 配置 |
|---|---:|
| 区间 A，L 更新 | 200 Step / Seed |
| 区间 A，M 更新 | 200 Step / Seed |
| 区间 B，F 更新 | 500 Step / Seed |
| 区间 B，P 更新 | 500 Step / Seed |
| 区间 B，J 更新 | 500 Step / Seed |
| 每 Step LM 训练 Token | 512 |
| 每次 Connection 更新的反事实状态 | 8 |
| 每个状态最多候选 Cell 对 | 3 |
| Seed 数 | 3 |

不包含 F0 的零更新评估，本轮计划共 5,700 次优化器更新，其中区间 A 的 1,200 次为 Connection 更新，区间 B 的 4,500 次包含 Cell 更新；J 的策略更新在 Cell 更新之外另行计数。

实际训练期 FLOPs 会因反事实展开而高于普通 Cell-only 训练，必须单独记录。这不能被描述为同等训练算力，只能称作相同主训练 Token 和 Cell 更新预算。

## 8.2 优化器

Cell 沿用 R3c 相同的 AdamW 设置及学习率日程。

Connection 使用独立 AdamW 参数组。

建议 Connection 初始学习率与 R3c Gate 学习率一致，实际设定写入冻结配置。

长期偏好温度、最大策略更新幅度及探索率均在运行前固定。

禁止在看到 Development 结果后自动搜索最优超参数。

## 8.3 数据规则

使用现有 Train、Calibration、Development 划分。

- Train：模型与 Connection 参数更新。
- Calibration：冻结配置所需的预先规定校准及运行质量检查。
- Development：全部 Arm 的最终科学评价。
- Confirmation：不访问。

R3c 的 Development 已用于历史研究决策，因此 R3d 在相同窗口上的结果继续标记为探索性，不得升级为独立确认结果。

三个 Seed 不能将每个 Seed 内的 Story 或 Token 当作独立训练重复。

---

# 9. 一次性自动执行

## 9.1 执行要求

Codex 应一次性完成：

1. 新建 R3d 独立实验目录。
2. 复用 R3c 已验证的模型及数据实现。
3. 实现新的 Connection 和反事实价值学习。
4. 实现 F0/L/M/F/P/J 对照。
5. 实现统一训练及评估 Runner。
6. 加载历史 Checkpoint。
7. 执行全部三个 Seed。
8. 自动生成对照结果和科学总结。
9. 保存必要的完整训练状态及运行清单。

**不得在以上步骤之间要求人工确认。**

不需要独立的 Phase 0/1/2/3 人工放行机制。

不需要另外执行长时间 GPU Pilot 或 Smoke 训练。

如果依赖和训练数据已经存在，直接复用，不要重复下载或重新生成。

## 9.2 单命令执行

提供一个仓库级入口：

```bash
python scripts/run_r3d.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/config/frozen.json \
  --run-all
```

该入口由本次实施创建，不是假定仓库目前已经存在。

Kaggle Notebook 只调用这个统一入口，不要在 Notebook 中重复实现训练逻辑。

环境变量负责提供已配置的私有数据和模型存储访问权限。

不得在 Notebook 或源码中写入密钥。

## 9.3 一次性运行的边界

若环境中已有正确的 Checkpoint 和数据，程序直接加载。

若不在本地，则可通过已经存在的私有存储引用自动恢复。

R3c 的 `actual_kaggle_persistence_verified=false` 不得改写成 true。

本次仅需在使用前验证恢复对象的 SHA、模型身份、Seed、配置和张量形状，并准确报告恢复结果。若无法取得所需 D Checkpoint，应以明确错误退出，不得用随机初始化模型替代。

不需要为此另外创建独立的测试性训练 Session。

---

# 10. 最小必要正确性检查

用户要求精简流程，因此 R3d 不应重新执行整套 R3c 工程测试或生成大量独立诊断。

只保留以下会直接影响科学有效性的检查。

## 10.1 启动时

- R3c D Checkpoint 身份匹配。
- Tokenizer 与数据哈希匹配。
- 三组 Seed 的初始参数可追溯。
- 新 Connection 的零修正输出与 D 路由一致。
- 真实执行的 Cell 对始终有两个不同 Cell。
- 没有访问 Confirmation。

## 10.2 训练时

- Loss 与梯度有限。
- 被执行 Cell 有正确的梯度路径。
- 冻结参数没有被优化器修改。
- Connection 更新使用的是 Train 中的反事实评分。
- 相同对照组使用相同主训练数据顺序。
- 当前动作不同后，后续路由确实重新根据更新后的状态决策。

## 10.3 结果保存时

- Checkpoint 完整写入。
- 文件 SHA 匹配。
- 结果 Manifest 与实际运行一致。
- 所有计划 Arm 和 Seed 均有明确完成或失败状态。
- 私有 Checkpoint 不进入公开 GitHub。

上述检查应直接集成到正常运行中，不额外创建完整的测试阶段。

仅在出现检查失败时终止相关运行。

---

# 11. 评价方法

## 11.1 核心评价

统一使用最终 Development NLL。

所有 Arm：

- 使用相同的评估窗口。
- 使用真实闭环 Connection。
- 不允许读取目标标签决定推理动作。
- 每阶段执行两个 Cell。
- 总共执行四个 Stage。
- 固定执行一次 Context Read。

推理时：

- 关闭探索。
- 不计算反事实分支。
- 不保留训练期的动作价值。
- 不使用即时或长期信用监督。

## 11.2 主要比较

每个 Seed 独立报告：

\[
\Delta_{L-F0}
=
NLL(L)-NLL(F0)
\]

\[
\Delta_{L-M}
=
NLL(L)-NLL(M)
\]

\[
\Delta_{J-F}
=
NLL(J)-NLL(F)
\]

\[
\Delta_{J-P}
=
NLL(J)-NLL(P)
\]

负值表示前者具有更低的 NLL。

另外报告：

\[
\Delta_{P-F}=NLL(P)-NLL(F)
\]

作为已有 Connection 在 Cell 更新过程中的收益诊断。

## 11.3 统计

必须报告：

- 各 Seed 的独立 NLL。
- 三 Seed 平均 NLL。
- 各主要比较的配对 NLL 差异。
- Story 聚类配对 Bootstrap 区间。
- Seed 间差异。
- 全部主要比较的预设统计口径。

主要比较的多重比较调整采用预先冻结的方法，例如 Holm。

三个 Seed 只能支持探索性重复证据，不能因为 Story 数量较多就宣称具有大量独立训练重复。

## 11.4 主要判断

R3d 的理想结果为：

1. L 优于 F0。
2. L 优于 M。
3. J 优于 F。
4. J 不劣于 P，并最好有进一步改善。
5. 没有严重的持续性 Cell 零曝光。
6. 真实推理 FLOPs 和 Cell 调用预算保持一致。

如果仅 L 优于 F0，而 J 不优于 F，则支持 Connection 可以独立学习，但仍不能证明联合训练机制有效。

如果 L 与 M 表现相同，则不能声称长期信用优于即时信用。

如果 J 优于 F，但不优于 P，则可能说明提前训练好的 Connection 有价值，却没有证明持续更新 Connection 带来额外收益。

---

# 12. 计算效率与资源指标

本轮不以硬件效率突破作为主要科学成功条件，但仍须确认新 Connection 没有引入无法接受的推理开销。

报告：

| 指标 | 要求 |
|---|---|
| 模型总参数 | 记录 |
| 新 Connection 参数 | 记录 |
| 实际 Cell 调用 | 8 / Token |
| Context 调用 | 1 / Token |
| 前向算子 FLOPs | 与 F 比较 |
| Token/s | 相同硬件下比较 |
| 峰值 GPU 显存 | 相同评估设置 |
| 训练期反事实额外 FLOPs | 单独记录 |
| 训练总计算成本 | 与各基线分别报告 |

不要将不完整的 MatMul FLOPs 估计称为完整硬件 FLOPs。

不得将推理阶段没有部署的反事实评估器计入部署参数；本轮没有单独部署的 Critic。

最终如果新 Connection 使推理成本显著增长，必须在报告中明确这一代价。

---

# 13. 结果保存

建议新目录：

```text
research/projects/cellular-language-models/
  experiments/
    clm-001-qfirst-integrated/
      r3d/
        SPEC.md
        README.md
        config/
          frozen.json
          releases.lock
        implementation/
          minicells_r3d/
            connection.py
            counterfactual.py
            training.py
            evaluation.py
            runner.py
        notebooks/
          kaggle-development.ipynb
        reports/
          RUN_MANIFEST.json
          METRICS.json
          ROUTING_SUMMARY.json
          TRAINING_COST.json
          SCIENTIFIC-FINDINGS.md
```

实际目录可以遵循仓库已有包组织方式进行调整。

不得复制 R3c 中未修改的大量通用代码。

Checkpoint 应包含：

- 模型参数。
- Connection 参数。
- Optimizer 状态。
- 训练进度。
- 所有相关 RNG 状态。
- 数据游标。
- Experiment、Arm、Seed 和源码身份。

所有文件保存应支持原子写入和幂等恢复。

如训练因计算环境中断，恢复后继续剩余工作，不得从头重复已经确认完成的参数更新。

对于跨 Session 持久化，若本次仅完成即时上传和回读，应准确标记为即时校验，不得声称独立 Session 恢复验证已经完成。

---

# 14. 自动生成科学报告

实验结束后自动生成 `SCIENTIFIC-FINDINGS.md`。

报告必须包含：

## 14.1 实验完成状态

- 全部 Seed 是否完成。
- 全部 Arm 是否完成。
- 实际更新次数。
- 数据量。
- 计算资源。
- Checkpoint 保存与恢复状态。

## 14.2 主要科学结论

逐项判断 H1、H2、H3、H4：

- SUPPORTED_EXPLORATORY
- NOT_SUPPORTED
- INCONCLUSIVE
- INVALID_RUN

不能将工程运行成功等同于科学假设成立。

## 14.3 主要表格

- F0/L/M 的 NLL 比较。
- F/P/J 的 NLL 比较。
- Cell 曝光与梯度覆盖比较。
- Connection 路由分布。
- 训练计算成本。
- 推理成本。

## 14.4 失败原因分析

当假设不成立时，报告应区分：

- 是否存在条件动作价值差异。
- Connection 是否成功学习候选动作排序。
- Connection 的实际闭环 NLL 是否改善。
- Cell 能力更新是否导致路由价值变化。
- 是否出现持续性 Cell 曝光不足。
- 长期信用是否确实优于即时信用。

不因失败而自动追加新的消融训练。

---

# 15. 科学结论门槛

## 15.1 H1 成立条件

冻结 Cell 时，L 相对于 F0 在三个 Seed 上均有正向改善，并且平均收益达到预先声明的实践意义阈值。

若不满足，只能报告该配置下 Connection 长期选择未得到稳定支持。

## 15.2 H2 成立条件

L 相对 M 在相同训练预算下产生可重复的正向收益。

若 L 与 M 没有明显差异，不能声称长期信用带来了独立增益。

## 15.3 H3 成立条件

联合训练 J 相对于冻结 Router 基线 F 取得可重复质量改善，并报告 J 与 P 的结果，区分已学习路由的收益和持续路由更新的额外贡献。

## 15.4 H4 成立条件

Cell 训练曝光和梯度覆盖没有表现出持续性严重恶化，同时模型最终质量得到改善。

不得只凭训练曝光更加均衡就宣布科学成功。

实际数值阈值须在第一次正式训练前写入 `frozen.json`，可以采用预先声明的工程意义门槛；不能根据本轮 Development 数据重新挑选有利阈值。

即使通过全部探索性门槛，结论仍然属于 Development 级别，不解除 Confirmation 封存。

---

# 16. 工程限制

Codex 必须遵守：

1. 不修改 R3、R3b、R3c 的冻结实验结果。
2. 不重训历史 R3c。
3. 不增加新的 Cell 架构。
4. 不重新引入 STOP、CTX 竞争或 LCG。
5. 不恢复普通可微 Top-2 CE Gate 训练。
6. 不引入额外的独立 Critic。
7. 不新增大量 GPU Smoke 和预训练探针。
8. 不进行自动超参数搜索。
9. 不使用 Development 调整训练设置。
10. 不访问 Confirmation。
11. 不上传私有模型或原始私有数据到公开 GitHub。
12. 不将训练中断、数据缺失或无法验证的结果伪称为完成。
13. 不设置中途人工放行。
14. 不因负结果自动重训。

当前项目已有共同的数据、Tokenizer、GPU 和 Checkpoint 基础设施，应优先复用。

---

# 17. Codex 最终执行指令

请在 `ArcheLabs/mini-cells` 仓库中实现并执行 `CLM-001-R3d-LHCJL-001` 实验。

本实验采用 R3c 的固定四阶段、共享六 Cell、每阶段 Top-2 结构，以 R3c D 组 Step 500 Checkpoint 为初始化基础。

实现新的轻量 Connection，通过对 15 种 Cell 对进行状态依赖的离散选择来控制路由。Connection 使用训练期有限反事实路径的最终 CE 形成长期动作偏好监督，Cell 沿实际执行路径接受 CE 梯度。

实现完整 F0/L/M/F/P/J 对照：

- F0：原始冻结 D。
- L：冻结 Cell，训练长期价值 Connection。
- M：冻结 Cell，训练即时价值 Connection。
- F：冻结 Router，继续训练 Cell。
- P：冻结预训练长期 Connection，继续训练 Cell。
- J：长期 Connection 与 Cell 交替联合训练。

运行全部三个冻结 Seed，使用固定的训练预算及相同评估协议。

**执行要求：**

- 一次性完成全部代码。
- 创建一个统一运行入口。
- 完成实现后直接启动正式实验。
- 不执行额外的多阶段 GPU Smoke。
- 不设置中途人工批准或确认。
- 只保留防止无效训练、数据错误及结果损坏的最小检查。
- 正常运行不因科学结果为负而停止。
- 如果环境中存在可安全恢复的运行进度，则自动恢复。
- 实验完成后自动保存 Checkpoint、运行指标、Manifest 和科学结论。
- 私有模型及训练产物保存到既有私有存储，公开仓库仅提交允许公开的代码、配置和汇总。
- 不解除 Confirmation。
- 在训练资源或 Checkpoint 不可用时准确报告阻塞，不以伪造结果、随机模型或未验证数据替代。

最终返回：

1. 实际 Git Commit 和运行源码 SHA。
2. 实验运行状态。
3. 全部六组对照的结果。
4. H1–H4 的科学判断。
5. 训练和推理资源成本。
6. Checkpoint 与报告的真实保存位置。
7. 若未完成，提供明确的失败原因及可恢复状态。

**不得仅完成方案或代码后停下等待人工授权。正常条件下应在同一次执行中完成全部实验。**

---

# 18. 本轮研究的核心判断

R3d 的目标不是证明某个新型 Router 的工程实现可行，也不是单纯取得更低的 NLL。

真正需要回答：

**Cell 的计算能力能否产生有意义的长期动作价值，而 Connection 能否学习利用这种价值，并在 Cell 能力继续变化的情况下维持有效的计算组织？**

只有这项机制得到支持，MiniCells 才有充分依据进一步研究：

- 更自由的 Cell 顺序组合。
- 动态 STOP。
- 自发形成的计算组织。
- 更大规模的 Cell 网络。
- 更少参数的高效推理。

本轮不应提前引入这些复杂度。

**R3d 的唯一核心使命，是建立或否定一种可复现的 Cell–Connection 长期信用联合训练机制。**

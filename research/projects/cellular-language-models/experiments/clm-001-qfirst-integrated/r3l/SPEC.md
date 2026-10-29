# MiniCells R3l — Connection Reorganization Potential

**连接组织的可改进空间、候选生成能力与长期学习收益验证**

**实验 ID：** `CLM-001-R3l-Connection-Reorganization-Potential-001`  
**版本：** v1.0  
**仓库：** `ArcheLabs/mini-cells`  
**实验性质：** 固定 Cell 组织搜索 + 长期配对训练 + 结构潜力诊断  
**建议环境：** Kaggle T4 × 2  
**Confirmation：** `SEALED`  
**研究范围：** `EXPLORATORY_DEVELOPMENT`  
**运行方式：** 一次启动，分阶段执行，支持持久化恢复

---

# 1. 研究背景

## 1.1 R3h 至 R3k 的研究进展

MiniCells 已经完成以下阶段：

| 实验 | 主要结论 |
|---|---|
| R3h | 稳定硬路由与训练—推理对齐改善 Cell 学习质量 |
| R3i | Connection 可以受约束更新，Cell 能产生局部适应，但整体质量没有额外改善 |
| R3j | 短期 Connection 收益不能可靠预测长期收益；直接路由和 Cell 适应可能方向相反 |
| R3k | 五项早期预测指标均未获得支持；48 个候选中没有达到实际收益门槛的长期更新 |

R3k 的关键结果：

| 指标 | 结果 |
|---|---:|
| 历史 Connection 候选 | 48 |
| +100 Step 正向 | 22 |
| +100 Step 负向 | 26 |
| +100 Step 平均收益 | −0.000064 NLL |
| 最大正向收益 | +0.001825 NLL |
| 达到 +0.005 NLL 改善门槛 | 0 |
| 获得支持的早期预测指标 | 0/5 |

R3k 的五种早期指标包括：

- S-Cal：历史 Calibration 收益。
- S-Direct：即时路由收益。
- S-K10：独立 Probe 上的十步总收益。
- S-Cell：十步 Cell 适应收益。
- S-Slope：早期收益变化。

这些结果说明，当前方案的问题并不仅仅是 Connection 选择门槛不够准确。

**现有 Proposal 产生的候选，本身也没有表现出稳定、足够大的长期质量收益。**

## 1.2 R3l 的研究动机

前几轮研究始终隐含一个假设：

> 当前六个 Cell 已经具备足够不同的计算能力，因此，只要找到更好的 Connection，模型质量就应该得到改善。

但这个假设还没有得到直接验证。

可能存在三种情况：

1. 当前 Cell 已具备可互补的能力，但是 R3i 的 Proposal 无法发现有价值的组织。
2. 当前 Cell 存在某种组织潜力，但必须经过额外训练才能释放。
3. 在当前模型规模、任务和训练阶段下，改变 Connection 的可用收益本来就很小。

R3l 需要区分这三种可能性。

---

# 2. 核心科学问题

本轮回答四个问题。

### Q1：是否存在更好的合法 Connection？

在保持 Cell 参数冻结的情况下，使用比 R3i 更充分的搜索，能否找到具有明显质量改善的 Connection？

### Q2：R3i 的候选生成是否构成主要瓶颈？

在相同起点、相同合法动作空间和相同 Trust Region 下：

- 历史 R3i Proposal；
- 随机参数方向；
- 结构化参数方向；
- 经过独立数据筛选的候选；

哪类候选更有可能产生真实的质量收益？

### Q3：更好的组织是否需要持续 Cell 训练？

即使某个 Connection 在当前 Cell 参数下没有明显收益，经过后续 100 Step Cell 训练后，是否能产生更好的模型质量？

### Q4：当前六个 Cell 的计算路径是否存在尚未利用的能力空间？

在固定 Cell 参数、固定计算预算的条件下，是否存在其他合法执行路径，能够显著降低损失？

这里必须严格区分：

- Connection 参数可实现的路径。
- 人为强制指定的路径。
- 使用目标标签选择路径的 Oracle。

这三者不具有相同的可部署性。

---

# 3. 核心实验原则

## 3.1 不改变 Cell 架构

保持：

- 六个 Cell。
- 原始 Cell 参数结构。
- 原始 Gate。
- 原始 Embedding、Context、Memory 和 Output Head。
- 原始训练目标。
- 原始损失定义。
- 原始 Tokenizer。

不允许 Cell 出生、死亡、复制、合并。

不增加额外 Expert。

## 3.2 不改变合法计算预算

A 架构：

- 四个计算阶段。
- 每阶段选择两个 Cell。
- 每阶段有 15 种合法 Cell Pair。
- 每 Token 共执行八次 Cell。

B 架构：

- 八步顺序执行。
- 每一步选择六个 Cell 中的一个。
- 包含前驱 Cell 信息。
- 允许自连接。
- 每 Token 共执行八次 Cell。

所有主要质量比较必须保持八次 Cell 调用预算。

## 3.3 不引入新学习算法

R3l 不是新一代 Connection Optimizer。

本轮只研究已有结构中是否存在未被利用的计算组织。

因此：

- 不训练新的 Value Network。
- 不引入强化学习。
- 不改变 R3h 的路由机制。
- 不引入 Balance Loss。
- 不使用新的 Cell 训练目标。
- 不改变 Connection 的参数表达结构。

允许在私有诊断分支中产生和筛选候选 Connection，但不将其提交为正式 R3i 训练结果。

---

# 4. 实验身份与历史来源

## 4.1 R3i 作为模型起点

固定历史 R3i：

`CLM-001-R3i-Stable-CoAdaptation-001`

Source SHA：

`80e49b071790c8bfb58dead19e6037259551ed61`

Config SHA-256：

`72de3cfb262a93ec6bb24b29b9f6c18d1d40030c76245cfda9a504852b30e630`

Run ID：

`ed08463119358da0f81b733786ab24bd89ac69c0f6b290dafc0f26311fe328a9`

历史 Checkpoint：

`libingjiang/minicells-research-checkpoints`

HF Revision：

`2fd527bf4958979f16da1fbc88ef5000bb70cc62`

Manifest SHA-256：

`25930ac73f54cb06cc1a61670e97aad7847f52a1aacca836d8595b64f7820a15`

必须验证完整 Manifest 与实际 Checkpoint。

## 4.2 R3k 作为历史证据

R3k 结果 Commit：

`622af501e805b422393455927d122e492449ced4`

R3k Source SHA：

`879a1348c9dfee1d0c0f33af9643fc66497b57fa`

R3k Run ID：

`cf4774dd2af77d9c062831f7cdef64caea24d1ab43c753e29315ca5a0494e758`

R3k 数据只能用于：

- 历史结论记录。
- 当前实验动机。
- 历史 Proposal 指标对照。
- 验证既有结果一致性。

不得使用 R3k 长期 Holdout 的结果选择 R3l 新候选。

---

# 5. 实验起点

## 5.1 固定 12 个主要 Anchor

选择 R3i 主训练 Step 300 的正式状态。

三个 Seed：

- `151001301`
- `151001302`
- `151001303`

四个 Arm：

- `A-CH`
- `A-CT`
- `B-CH`
- `B-CT`

共：

**3 Seed × 4 Arm = 12 个 Anchor**

每个 Anchor 固定：

- 当前 Cell 参数。
- 当前 Connection 参数。
- Cell Optimizer。
- Scheduler。
- Main Step。
- Data Cursor。
- RNG。
- H/T 模式。
- Temperature。
- ε 调度。

必须恢复的是 Step 300 Proposal 执行之前的正式模型状态。

不能恢复 Step 310 的适应后状态充当 Step 300 起点。

## 5.2 为什么选择 Step 300？

原因：

1. Cell 已完成一定训练。
2. 某些轨迹可能已经接受过 Connection 更新。
3. 仍存在 100 Step 的后续训练窗口。
4. 可以和同一 Step 的历史 Proposal 对照。
5. 所有实验都从相同全局训练阶段开始。

这不会证明结果可以推广到训练初期或更大模型。

## 5.3 辅助历史观察

可以对 Step 100 的 12 个 Anchor 进行只读分析。

但不在 Step 100 重复完整 Connection 搜索和长期训练。

其作用仅为检查：

- 初始组织使用情况。
- Cell 暴露分布。
- 路由多样性。
- Step 300 是否已出现明显组织变化。

Step 100 结果不参与主要判定。

---

# 6. R3l 的三层组织空间

本轮最重要的设计，是把三种不同的组织空间分开。

## Level 1：现有 Connection 参数空间

这是主要科学比较。

A 的 Connection 是针对 15 个 Cell Pair 的修正模型。

B 的 Connection 是具有前驱 Cell 和执行步信息的六动作修正模型。

所有候选必须通过修改真实 Connection 参数产生。

不能使用强制动作绕过 Connection。

其执行必须仍然符合：

`hard_action(raw, baseline, route_switch_margin)`

以及现有 H/T 训练语义。

这一层研究：

**当前 Connection 参数表达能力中，是否已经存在更好的组织？**

## Level 2：受预算约束的参数搜索空间

在相同 Connection 模型中，扩大候选方向的覆盖。

包括：

- 历史 Proposal。
- 随机参数扰动。
- 结构化扰动。
- 在独立数据上筛选后的最优候选。

所有候选仍须通过 Trust Region。

这一层研究：

**R3i 原来的 Proposal 是否找不到实际上存在的更好参数配置？**

## Level 3：带标签的强制路径诊断

在少量固定样本上，允许显式指定合法 Cell 动作，并使用目标 Token 的损失搜索执行路径。

但必须明确：

- 它读取真实目标标签。
- 它不受当前 Connection 参数表达能力约束。
- 它不一定能被单个可部署 Connection 实现。
- 它不代表正常推理。
- 有限 Beam Search 不能证明全局最优。

这一层只能用于检查：

**同一组冻结 Cell 是否在其他合法执行路径上表现出尚未利用的能力。**

它是诊断性路径机会估计，不是正式模型的质量上界。

---

# 7. 数据边界

本轮必须使用固定数据分区。

原始数据集：

`roneneldan/TinyStories`

Revision：

`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`

Tokenizer SHA-256：

`612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`

## 7.1 既有数据

| 角色 | Story Range |
|---|---|
| Train | [0, 8192) |
| R3k Predictor Probe | [8192, 8448) |
| R3k Value Holdout | [8448, 8960) |
| Historical Calibration | [50000, 50128) |
| Historical Admission | [50128, 50256) |
| Historical Development | [50256, 50512) |

不得将这些数据当作 R3l 的全新独立评价数据。

## 7.2 R3l 新数据

建议冻结：

| 角色 | Story Range | 用途 |
|---|---|---|
| `R3L_SEARCH` | [8960, 9216) | 搜索候选 |
| `R3L_SELECT` | [9216, 9472) | 从预筛候选中选择 |
| `R3L_REPORT` | [9472, 9984) | 正式探索性质量评价 |
| `R3L_PATH_DIAGNOSTIC` | [9984, 10048) | 标签感知的路径 Oracle 诊断 |

必须在实验运行前验证：

- 数据集 Revision。
- 真实 Story 数量。
- Story 内容哈希。
- 索引不重叠。
- 与此前数据的精确重复。
- 与已有 Confirmation 定义不冲突。

数据范围只是预先提出的分区；如果与既有研究的真实数据清单冲突，必须停止并报告，不能静默替换为其他范围。

排除重复样本只能采用事前固定规则。

## 7.3 数据访问原则

`SEARCH` 可以用于候选优化和质量筛选。

`SELECT` 只能用于最终候选选择。

`REPORT` 只能用于冻结候选之后的评价。

`PATH_DIAGNOSTIC` 只能用于路径 Oracle。

任何用于选择组织的数据，都不能再次被当成独立的泛化证据。

Confirmation 始终 `SEALED`。

---

# 8. Trust Region

保持 R3i 的现有约束：

| 指标 | 上限 |
|---|---:|
| Policy KL | 0.01 |
| Hard Action Change | 10% |
| Full Path Change | 20% |

同时保留：

- Route Switch Margin。
- 数值稳定性门槛。
- Hard Action 规则。
- A 的合法 Pair。
- B 的合法下一节点。
- 原始 Admission Windows。

必须使用当前代码真实实现的判定函数，而不是仅凭上述三个数值重新实现一套简化门禁。

## 8.1 原始历史 Proposal

必须逐 Anchor 恢复 R3i 在 Step 300 实际生成的候选。

记录：

- Original Connection。
- Candidate Connection。
- Parameter Delta。
- Chosen Scale。
- Trust Metrics。
- Candidate Fingerprint。

不得重新优化生成所谓历史 Proposal。

## 8.2 新候选

每个新候选必须：

1. 从相同正式 Connection 出发。
2. 生成完整参数方向。
3. 将参数方向确定性缩放。
4. 按冻结的尺度扫描协议进行 Admission。
5. 检查合法性和数值稳定性。
6. 保存完整候选和拒绝原因。

如果没有可用的非零合法尺度，标记：

`NO_FEASIBLE_NONZERO_UPDATE`

不能将零变化候选计入有效组织搜索结果。

## 8.3 无效变化

如果候选参数改变，但在固定评价样本上：

- Hard Action 没有变化；
- 完整路径没有变化；

则标记：

`PARAMETER_CHANGE_WITHOUT_EFFECTIVE_REORGANIZATION`

这一结果属于搜索效率诊断。

不能算作发现新的计算组织。

---

# 9. Stage A：候选生成与冻结 Cell 搜索

## 9.1 搜索矩阵

对每个 Step 300 Anchor，产生以下候选。

| 家族 | 每 Anchor 数量 | 定义 |
|---|---:|---|
| Baseline | 1 | 原始 Connection |
| Historical | 1 | R3i 的真实 Proposal |
| Reverse | 1 | 历史 Proposal 参数方向取反 |
| Random | 8 | 固定种子随机扰动 |
| Structured | 4 | 结构化参数扰动 |
| **合计** | **15** | 包含 Baseline |

每个 Anchor 最多 14 个非 Baseline 候选。

总计：

**12 × 14 = 168 个非 Baseline 候选。**

失败候选也必须保留记录。

不得根据实际 NLL 动态增加随机搜索次数。

## 9.2 Random 候选

生成八个独立的确定性随机参数方向。

要求：

- 完整覆盖 Connection 的可训练参数。
- 使用专用随机流。
- 不读取目标标签。
- 不依赖 REPORT。
- 方向归一化。
- 保留原始浮点精度。
- 通过 Trust Region 后才进入质量评价。

对不同参数块分别进行数值规范化，防止单一权重张量支配全部扰动。

Random 候选是搜索对照，不是新的通用学习算法。

## 9.3 Structured 候选

四个结构化扰动需要适配 A/B 的 Connection 参数含义。

A：

- 围绕四阶段的阶段特征。
- 控制少量 Cell Pair 的分数偏移。
- 保持其他阶段影响尽可能低。
- 不修改 Cell Pair 注册表。

B：

- 围绕八个执行步和前驱 Cell 特征。
- 使用低秩或局部参数扰动。
- 不改变节点数量。
- 不删除任何合法转移。

四个结构化方向的具体构造和随机种子必须在开始评价前冻结。

可以使用低秩扰动，但不能额外引入新的运行时参数层。

## 9.4 Historical 与 Reverse

Historical：

直接使用 R3i 保存的原始 Proposal Delta 与合法尺度。

Reverse：

从同一起点应用相反参数方向：

`original_connection - scale * historical_delta`

Reverse 必须独立执行 Trust Region。

其意义是检测：

**R3i 的优化方向是否确实比其反方向更有价值。**

不得假定取反后自然合法。

---

# 10. Stage A 的搜索协议

## 10.1 冻结完整候选清单

在读取 SEARCH 的质量评价结果之前生成：

`CANDIDATE_REGISTRY.json`

记录：

- Anchor ID。
- Candidate ID。
- 方向生成规则。
- RNG Seed。
- 参数哈希。
- Trust Region 结果。
- 候选家族。
- 有效路径变化。
- 计算预算。

不得依据质量结果新增方向或扩展搜索规模。

## 10.2 SEARCH 阶段

固定 Cell 参数。

对全部有效候选执行闭环、确定性路由评价。

必须使用实际模型完整 Forward：

- 不是固定中间状态上的局部分数。
- 不是使用目标标签替换 Router。
- 不是仅比较原始 Logit Delta。
- 必须重新执行候选 Connection 的完整计算路径。

SEARCH 固定使用 32 个按事前规则选择的评价窗口。

记录每个候选的：

- NLL。
- 相对于 Baseline 的 NLL 改善。
- 路径变化。
- Cell 使用差异。
- 实际计算成本。

## 10.3 候选预筛

每个 Anchor 从有效候选中选择 SEARCH NLL 最低的前四个。

必须包括所有候选家族，不因为候选来自 Historical 而特殊加分。

若有效候选少于四个，则全部进入下一步。

不允许提前查看 SELECT 或 REPORT。

## 10.4 SELECT 阶段

在独立的 `R3L_SELECT` 上重新评价 SEARCH 前四名。

使用固定 128 个评价窗口。

选取 SELECT NLL 最低的候选：

`SEARCH_BEST`

这一候选是**在有限、预注册候选集合中，经两层数据选择的最优候选**。

它不是全局最优 Connection。

## 10.5 Random 对照

另从有效 Random 候选中，根据固定哈希顺序选择一个：

`RANDOM_CONTROL`

不能选择其中在 SEARCH 或 SELECT 上表现最好的一个当作随机对照。

若没有任何有效 Random 候选，记录该 Anchor 缺少 Random 对照，不能换用其他家族冒充。

## 10.6 冻结最终四组

每个 Anchor：

1. `BASELINE`
2. `HISTORICAL`
3. `SEARCH_BEST`
4. `RANDOM_CONTROL`

形成主要评价矩阵。

如果 SEARCH_BEST 与 HISTORICAL 或 RANDOM_CONTROL 是同一参数候选，不要创建不同参数版本。

但逻辑比较身份仍然保留，并记录结果相关性和共享模型。

不能把同一个实际 Checkpoint 重复当作独立训练结果。

---

# 11. Stage B：冻结 Cell 下的真实组织价值

此阶段不更新任何 Cell 参数。

所有四组使用相同 Anchor Cell 状态。

在 `R3L_REPORT` 上评价：

- Baseline。
- Historical。
- Search Best。
- Random Control。

主要指标：

\[
G_{\mathrm{frozen}}(c)
=
L_{\mathrm{REPORT}}(\theta_s,\phi_s)
-
L_{\mathrm{REPORT}}(\theta_s,\phi_c)
\]

其中：

- \(G>0\)：候选组织更好。
- \(G<0\)：候选组织更差。

## 11.1 三项主要比较

### F1：Historical vs Baseline

验证历史 Proposal 在新独立数据上的即时实际价值。

### F2：Search Best vs Baseline

验证扩大搜索是否能发现真正具有泛化收益的 Connection。

### F3：Search Best vs Historical

判断扩展搜索是否优于 R3i 的原始 Proposal。

同时报告 Search Best vs Random Control。

## 11.2 实际改善门槛

保持：

\[
G_{\mathrm{frozen}}\ge0.005
\]

作为实际质量改善标准。

同时提供：

- 原始 NLL 差异。
- Story 配对差异。
- Bootstrap 区间。
- 每 Seed 结果。
- 每架构结果。

不允许因为最优候选改善不到 0.005 就修改门槛。

## 11.3 搜索偏差控制

SEARCH_BEST 已经被 SEARCH 和 SELECT 选择。

因此：

- SEARCH 的最优值具有选择偏差。
- SELECT 的结果仍然是选择依据。
- REPORT 才是冻结候选的探索性泛化评价。

不能直接把 SEARCH 中的最佳 NLL 当作科学结论。

---

# 12. Stage C：长期 Cell 适应验证

本阶段回答：

**冻结 Cell 时没有优势的新组织，是否经过持续训练后产生新的质量收益？**

## 12.1 固定四组

每个 Anchor 的：

- BASELINE。
- HISTORICAL。
- SEARCH_BEST。
- RANDOM_CONTROL。

从相同 Step 300 的 Cell 参数、Optimizer 和训练状态出发。

Connection 在分支开始前设置为对应组织。

之后训练中永久冻结。

## 12.2 训练预算

每条分支：

100 Step。

每 Step：

512 Token。

总逻辑预算：

\[
12\times4\times100=4800
\]

次 Cell 更新。

训练 Token：

\[
\boxed{2,457,600}
\]

如果多个逻辑 Arm 实际指向完全相同的候选和相同训练状态，可以复用同一条物理训练轨迹，节省计算。

但必须在报告中区分：

- 逻辑 Arm 数量。
- 实际执行分支数。
- 实际新增 Cell 更新。
- 实际训练 Token。

不能重复计算同一模型的训练成本。

## 12.3 训练方式

保留 R3i 的：

- AdamW。
- LR。
- Weight Decay。
- Gradient Clip。
- Global Step。
- Data Cursor。
- Scheduler。
- H/T 路由模式。
- Temperature。
- ε 调度。
- 原始外生随机流协议。

不得重新 Warmup。

不得重新训练 Connection。

不得执行 Step 400 的原始 Proposal 调度。

每条分支训练至 Step 400，评价发生在 Step 400 Proposal 执行之前。

## 12.4 同一随机轨迹

四组使用配对的：

- Train Token。
- 数据顺序。
- 外生随机数。
- Step。
- Microbatch。
- 优化器配置。

Connection 不同导致的路由变化属于实验干预，不能强制它们选择相同 Cell。

## 12.5 评价时点

从 Step 300 出发：

| 相对 Step | 全局 Step | 目的 |
|---|---:|---|
| T=0 | 300 | 冻结 Cell 下的组织差异 |
| T=10 | 310 | 初期 Cell 适应 |
| T=50 | 350 | 中期 Cell 适应 |
| T=100 | 400 | 长期组织收益 |

所有时点使用统一 REPORT 评价规则。

不得根据中途评价改变训练策略或提前停止。

## 12.6 主要长期效果

对于候选 \(c\)：

\[
V_T(c)=
L(\theta^{BASE}_T,\phi_{BASE})
-
L(\theta^c_T,\phi_c)
\]

比较：

- Historical vs Baseline。
- Search Best vs Baseline。
- Random Control vs Baseline。

主要终点：

\[
V_{100}
\]

次要终点：

\[
V_{10},\quad V_{50}
\]

---

# 13. Cell–Connection 长期归因

延续 R3j/R3k 的四格反事实方法。

对于 Baseline 和某个候选：

| Cell 参数 | Baseline Connection | Candidate Connection |
|---|---|---|
| Baseline Cell | \(L_{OO}\) | \(L_{ON}\) |
| Candidate Cell | \(L_{NO}\) | \(L_{NN}\) |

定义：

\[
V=L_{OO}-L_{NN}
\]

直接路由价值：

\[
R=L_{OO}-L_{ON}
\]

Cell 适应价值：

\[
C=L_{ON}-L_{NN}
\]

满足：

\[
\boxed{V=R+C}
\]

额外记录非加性交互：

\[
I=L_{NN}-L_{NO}-L_{ON}+L_{OO}
\]

交互项单独报告，不得与 R、C 再次相加。

需要观察：

- 当前执行效果是否为正。
- 后续 Cell 适应是否为正。
- 是否存在直接效果负、适应效果正的案例。
- 不同搜索家族是否产生不同的长期学习效应。
- A 和 B 是否具有不同的组织学习模式。

---

# 14. Stage D：标签感知路径诊断

这是一个完全独立的诊断实验。

目的是判断：

**冻结的六个 Cell 是否在其他合法执行路径上具有尚未利用的计算能力。**

## 14.1 路径动作空间

A：

每阶段 15 个 Pair，四阶段。

完整动作组合数量：

\[
15^4=50,625
\]

B：

每步六个节点，八步。

完整动作组合数量：

\[
6^8=1,679,616
\]

因此，不应在大量 Token 上做完整路径穷举。

## 14.2 使用有限 Beam Search

预注册：

- 每个架构选取一个固定 Seed 的 Step 300 Anchor。
- H/T 各一个，总计四个 Anchor。
- 每 Anchor 16 个 Token 位置。
- Beam Width = 8。
- 完整执行深度：A 为四阶段，B 为八步。

仅从 `R3L_PATH_DIAGNOSTIC` 读取固定样本。

Beam 扩展必须符合真实 Cell 执行语义。

不允许使用静态中间状态拼接出实际不可执行的路径。

## 14.3 Oracle 的信息权限

路径搜索可以读取真实目标 Token 的 Loss。

这是有意设计的标签感知诊断。

但必须标注：

`TARGET_AWARE_NON_DEPLOYABLE_DIAGNOSTIC`

它不代表可部署 Router。

不能把这类结果当成模型在没有标签时能够达到的质量。

## 14.4 评价定义

对每个固定 Token，比较：

- 当前 Baseline Path。
- 随机合法路径。
- Beam Search 搜索到的路径。

记录：

- NLL 差异。
- 路径差异。
- Cell 调用次数。
- 路径搜索计算成本。
- 是否超过当前 Connection 的能力。

## 14.5 如何解释 Oracle？

如果 Beam Oracle 明显优于 Baseline：

说明现有 Cell 在目标标签可见、允许选择合法路径的条件下，存在可观测的路径机会。

但不能说明：

- 这些路径可以被 Connection 参数表达。
- 无标签 Router 能找到它们。
- 这些路径可以泛化。
- 现有模型能够部署该 Oracle。

如果 Beam Oracle 没有改善：

也不能证明没有更好的组织，因为 Beam Search 是有限搜索。

**本轮不声称获得严格的全局组织质量上界。**

真正的数学上界需要证明搜索覆盖了指定空间的所有候选，并明确评价数据和可实现性范围；当前有限搜索不具备这个性质。

---

# 15. 搜索空间与能力瓶颈的判定

R3l 不能只给一个 PASS/FAIL。

必须分层报告。

## 15.1 有效重组空间

统计：

- 候选中实际改变参数的数量。
- 通过 Trust Region 的数量。
- 改变 Hard Action 的数量。
- 改变完整路径的数量。
- 不同候选路径的多样性。
- 参数不同但执行路径相同的比例。

如果大量候选没有改变执行路径，说明搜索首先受到有效重组空间或扰动尺度的限制。

## 15.2 冻结 Cell 组织收益

若 SEARCH_BEST 在独立 REPORT 上优于 Historical 和 Baseline，说明扩大搜索能够发现更好的可实现组织。

若所有候选都没有明显优势，只能说明**在本次有限候选集合和 Trust Region 下**没有检测到足够大的收益。

不能声称不存在任何更好的 Connection。

## 15.3 长期训练收益

如果 SEARCH_BEST 的 T=0 收益有限，但 T=100 表现显著更好，应进一步检查 Cell 适应贡献。

如果 T=0 和 T=100 均没有优势，则当前搜索方案尚未发现有价值的长期组织。

## 15.4 路径表达能力

如果标签感知 Beam Oracle 有很大优势，而可实现 Connection 搜索没有优势：

说明问题可能涉及：

- 参数化 Connection 的表达限制。
- 输入可观测信息不足。
- 候选搜索能力不足。
- 标签感知路径选择与正常推理之间的信息差异。

不能仅根据这一对比认定 Connection 表达能力不足。

---

# 16. 统计与科学判断

## 16.1 独立性

三个 Seed 是主要独立重复来源。

12 个 Anchor 中：

- 同一 Seed 的不同 Arm 共享部分历史来源。
- 不同 Connection 候选共享相同 Cell 起点。
- 多条续训分支使用配对训练数据。

因此，候选数量不能作为独立 Seed 数。

## 16.2 主要终点

冻结：

1. SEARCH_BEST 相对 Baseline 的 T=0 REPORT 改善。
2. SEARCH_BEST 相对 Historical 的 T=0 REPORT 改善。
3. SEARCH_BEST 相对 Baseline 的 T=100 REPORT 改善。

以上是三项主要比较。

对 A、B 分架构报告，并对多重比较做 Holm 调整。

## 16.3 实际改善门槛

使用：

\[
\Delta NLL\ge0.005
\]

作为实际改善门槛。

对于多 Seed 的正向科学信号，建议同时要求：

- 至少两个 Seed 改善方向一致。
- 平均改善达到门槛。
- 改善不只来自一个极端候选。
- 统计区间与效果大小完整报告。

即使满足这些探索性条件，也不能称为独立 Confirmation。

## 16.4 Bootstrap

使用 Story-cluster 配对 Bootstrap：

`bootstrap_replicates = 2000`

固定种子：

`bootstrap_seed = 2026102304`

并按 Seed / Arm 轨迹汇总依赖关系。

不得将 Story Bootstrap 当成训练 Seed 的替代品。

## 16.5 负面结果

如果没有候选获得明显收益：

- 实验仍然有效。
- 保留全部候选。
- 保留搜索预算。
- 保留无效路径比例。
- 保留所有负面 NLL。

不能不断增加随机候选数量，直到出现正向结果。

---

# 17. 计算预算

## 17.1 搜索阶段

主要 Anchor：

12。

每 Anchor：

- 14 个非 Baseline 候选。
- 最多四个进入 SELECT。
- 最终四个逻辑比较组。

候选总量：

168 个非 Baseline 候选。

SEARCH：

固定 32 个窗口/Anchor/Candidate。

SELECT：

固定 128 个窗口/Anchor/入围 Candidate。

所有 Trust Region 扫描、搜索和评价必须记录真实前向工作量。

不预先声称它们对应固定 GPU FLOPs。

## 17.2 长期训练

| 指标 | 上限 |
|---|---:|
| Anchor | 12 |
| 逻辑训练组/Anchor | 4 |
| 训练跨度 | 100 Step |
| 每 Step Token | 512 |
| 最大新增 Cell 更新 | 4,800 |
| 最大新增 Train Token | 2,457,600 |
| Connection Optimizer 更新 | 0 |

当逻辑组共享物理状态时，实际计算量可以低于上限。

## 17.3 Oracle 诊断

四个 Anchor。

每 Anchor：

16 个 Token 位置。

Beam Width：

8。

完整合法执行深度保持不变。

Oracle 工作量必须单独统计。

如果资源不足，允许将 Oracle 诊断标记为未执行，但不得静默缩小主实验搜索矩阵或训练预算。

---

# 18. 工程实现

建议目录：

`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3l/`

结构：

    r3l/
      SPEC.md
      README.md

      config/
        frozen.json

      implementation/
        minicells_r3l/
          parent_restore.py
          data_boundaries.py
          candidate_registry.py
          candidate_directions.py
          admission.py
          search.py
          selection.py
          frozen_evaluation.py
          continuation.py
          attribution.py
          path_oracle.py
          statistics.py
          storage.py
          reports.py
          runner.py

      notebooks/
        kaggle-reorganization-potential.ipynb

      reports/
        INPUT_VERIFICATION.json
        DATA_BOUNDARY_PROOF.json
        ANCHOR_REGISTRY.json
        CANDIDATE_REGISTRY.json
        TRUST_REGION_RESULTS.json
        SEARCH_RESULTS.json
        SELECT_RESULTS.json
        FROZEN_FINALISTS.json
        FROZEN_CELL_OPPORTUNITY.json
        LONG_HORIZON_RESULTS.json
        CELL_CONNECTION_ATTRIBUTION.json
        PATH_ORACLE_DIAGNOSTIC.json
        COMPUTE_COST.json
        SCIENTIFIC-FINDINGS.md
        RUN_MANIFEST.json

## 18.1 复用代码

优先复用：

- R3i/R3j 的历史 Checkpoint 恢复。
- R3f-v2 的 Trust Region 扫描。
- R3e 的 Connection 模型。
- R3h 的稳定路由。
- R3j/R3k 的 Cell 配对续训。
- R3j/R3k 的四格归因。
- 现有数据加载。
- 现有 HF 私有 Store。
- 现有 GitHub 报告发布器。

不得重新实现另一套不兼容的 Cell Forward。

## 18.2 Connection 更新的安全边界

候选搜索在私有分支完成。

正式 Anchor 不得被候选评价修改。

Cell 训练期间：

- Connection 不参与 Optimizer。
- Connection Gradient 必须为空。
- Connection Fingerprint 保持不变。
- 每条分支保留独立身份。

恢复时需要明确区分：

- `historical_official_state`
- `experimental_connection_candidate`
- `counterfactual_training_branch`

不得将实验候选伪装为 R3i 已正式接受的 Connection Commit。

---

# 19. 自动执行与持久化

## Phase A：历史恢复

1. 读取固定 R3i Manifest。
2. 验证 Checkpoint。
3. 恢复 12 个 Step 300 Anchor。
4. 读取对应历史 Proposal。
5. 验证数据边界。
6. 保存 Anchor Registry。

## Phase B：候选生成

1. 生成固定候选。
2. 保存所有方向参数。
3. 执行 Trust Region。
4. 保存拒绝原因。
5. 保存 Candidate Registry。

## Phase C：Frozen Cell Search

1. SEARCH 全部有效候选。
2. 每 Anchor 选择前四名。
3. SELECT 重新评价。
4. 冻结 SEARCH_BEST。
5. 冻结 RANDOM_CONTROL。
6. 生成 FROZEN_FINALISTS。

## Phase D：长期训练

1. 加载最终四组。
2. 从相同 Anchor 复制训练状态。
3. 执行配对 Cell 训练。
4. 保存 T=10、50、100。
5. 校验 Connection 冻结。
6. 完成训练屏障。

## Phase E：终端评价

1. 统一读取 REPORT。
2. 评价 T=0、10、50、100。
3. 计算直接收益与长期收益。
4. 执行四格归因。
5. 汇总统计。
6. 生成科学结论。

## Phase F：路径诊断

运行独立 Beam Oracle。

该阶段不得修改正式候选选择或长期训练结果。

## Phase G：发布

- 完整 Checkpoint 保存至私有 HF。
- GitHub 只发布聚合报告。
- 验证 SHA。
- 允许断点恢复。
- 发布失败不重新训练。
- 不自动启动下一实验。

---

# 20. 强制测试

## 20.1 模型一致性

- 六个 Cell 不变。
- A 的 15 Pair 不变。
- B 的六节点动作不变。
- 执行深度不变。
- 每 Token Cell 调用预算不变。
- Forward 语义不变。
- 训练路由 H/T 不变。

## 20.2 Checkpoint

- 历史 Anchor 完整恢复。
- Model Hash 正确。
- Optimizer Hash 正确。
- RNG 正确。
- Data Cursor 正确。
- 历史 Proposal 一致。
- 跨 Session 恢复可验证。

## 20.3 搜索

- 全部候选生成在评价前冻结。
- Random Seed 固定。
- Trust Region 正确。
- Invalid Candidate 不参与质量排名。
- SEARCH 和 SELECT 数据隔离。
- REPORT 未用于候选选择。
- RANDOM_CONTROL 不按质量挑选。

## 20.4 长期训练

- 相同训练数据。
- 相同全局 Scheduler。
- 不重置 Warmup。
- 不执行新 Proposal。
- Connection 始终冻结。
- Cell 更新预算正确。
- 相同逻辑候选的物理复用得到记录。

## 20.5 统计

- 四格归因成立。
- 数值结果重复评价一致。
- Bootstrap 固定。
- 所有负面案例完整保留。
- 多重比较处理正确。
- 不把候选数当作独立 Seed。

## 20.6 Oracle

- 强制路径合法。
- 完整执行真实 Cell 运算。
- 无非法未来状态拼接。
- 不参与正式 Candidate Selection。
- 不读取 Confirmation。
- 明确标记不可部署。

---

# 21. 科学结果分类

R3l 需要产生四个独立的结果类别。

## 21.1 Connection 参数空间的可改进性

- `REALIZABLE_CONNECTION_OPPORTUNITY_OBSERVED`
- `NO_DETECTABLE_REALIZABLE_CONNECTION_OPPORTUNITY`
- `INCONCLUSIVE`

## 21.2 历史 Proposal 的候选生成能力

- `SEARCH_OUTPERFORMS_HISTORICAL_PROPOSAL`
- `NO_SEARCH_ADVANTAGE`
- `INCONCLUSIVE`

## 21.3 长期 Cell 适应收益

- `LONG_HORIZON_REORGANIZATION_GAIN`
- `NO_LONG_HORIZON_GAIN`
- `INCONCLUSIVE`

## 21.4 Cell 路径潜力诊断

- `TARGET_AWARE_PATH_OPPORTUNITY_OBSERVED`
- `NO_DETECTABLE_BEAM_PATH_OPPORTUNITY`
- `DIAGNOSTIC_NOT_COMPLETED`

这些结果不应合并为一个模糊的实验成功标志。

---

# 22. 下一阶段决策规则

根据 R3l 的结果决定后续方向。

### 情况 A：搜索发现明显更好的可实现 Connection

如果 SEARCH_BEST 在 REPORT 上稳定超过 Baseline 和 Historical：

说明 R3i 原始候选生成可能是主要限制之一。

下一阶段优先研究：

**如何在不进行大量标签筛选的条件下，学习产生这种 Connection。**

但本轮不自动实现新 Optimizer。

### 情况 B：冻结 Cell 收益很小，长期训练收益明显

说明组织的价值主要通过后续 Cell 学习释放。

下一阶段优先研究：

**对未来学习价值进行信用分配，而不是当前执行收益优化。**

### 情况 C：参数搜索无益，但标签感知路径诊断明显有益

说明合法计算路径可能包含可利用能力，但目前尚未证明这些能力能够被现有 Connection 表达或无标签发现。

下一阶段需要进一步区分：

- Connection 表达限制。
- 信息可得性限制。
- 搜索算法限制。
- Oracle 使用标签带来的额外优势。

### 情况 D：两者都没有明显收益

这将是一个重要的负面科学信号。

下一步不应继续反复调整 Connection Optimizer。

应研究：

- Cell 之间的实际能力差异。
- Cell 是否形成互补性。
- 当前六 Cell 容量是否足够。
- 当前任务是否需要复杂组织。
- 模型和数据规模是否不足。
- 固定计算预算是否限制了潜在收益。

这时应返回 Cell 能力与组织需求的基础研究。

---

# 23. 完成门禁

必须满足：

- 12 个主要 Anchor 恢复有效。
- 数据隔离验证完成。
- 全部预注册候选完成生成或明确失败。
- Trust Region 完整记录。
- SEARCH 与 SELECT 完成。
- 正式候选在 REPORT 前冻结。
- 冻结 Cell 评价完成。
- 长期训练完成。
- 四格归因完成。
- 所有有效案例得到评价。
- 路径诊断完成或明确标记独立未完成。
- 科学结论与工程结果分开报告。
- Confirmation 始终 SEALED。
- 结果可追溯到固定 Git Source 和 HF Checkpoint。
- GitHub 发布结果。
- 不自动启动下一轮训练。

任何模型身份错误、数据泄漏、候选选择污染或训练不一致，都必须独立标记工程失败。

不得解释为科学负结果。

---

# 24. Codex 实施指令

请在 `ArcheLabs/mini-cells` 中实现：

`CLM-001-R3l-Connection-Reorganization-Potential-001`

这是对当前 MiniCells 六 Cell 组织空间的受控研究，不是新的生产 Connection Optimizer。

必须：

1. 复用 R3i 的真实 Step 300 Checkpoint。
2. 固定三个 Seed、四个 Arm，共 12 个主要 Anchor。
3. 保持六 Cell 与 A/B 的真实执行语义。
4. 保持每 Token 八次 Cell 调用。
5. 保持原始 H/T 路由机制。
6. 生成 Historical、Reverse、Random、Structured 候选。
7. 对所有候选应用原始 Trust Region。
8. 使用独立 SEARCH/SELECT 筛选候选。
9. 使用未参与选择的 REPORT 评价冻结 Cell 收益。
10. 将 BASELINE、HISTORICAL、SEARCH_BEST、RANDOM_CONTROL 进行 100 Step 配对续训。
11. 评价 T=0、10、50、100。
12. 完成 Cell–Connection 四格归因。
13. 对固定的小规模样本实施独立标签感知 Beam 路径诊断。
14. 严格区分可实现 Connection 搜索和不可部署 Oracle。
15. 保留全部候选、搜索预算、拒绝原因和完整数值结果。
16. 不将有限搜索结果称为全局最优或严格质量上界。
17. 不读取 Confirmation。
18. 私有保存完整 Checkpoint。
19. GitHub 发布聚合报告。
20. 不自动进入下一阶段。

最终必须回答：

**Q1：在当前六 Cell 参数固定的条件下，是否能够通过现有 Connection 表达结构找到明显更好的计算组织？**

**Q2：如果更好的 Connection 存在，R3i 原有的 Proposal 是否因为搜索能力不足而没有找到它？**

**Q3：Connection 的价值主要来自即时改变执行路径，还是来自改变 Cell 在未来 100 Step 的学习过程？**

**Q4：如果现有 Connection 搜索无法改善质量，合法 Cell 路径中是否仍然存在可测量的潜在计算能力？**

本轮的意义不取决于结果为正。

**R3l 的最终目的是判断：当前 MiniCells 的下一项重要研究投入，应该放在 Connection 搜索、长期组织学习，还是 Cell 本身的能力与分工形成机制上。**

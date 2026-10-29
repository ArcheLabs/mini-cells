# MiniCells R3m — Stability–Reorganization Boundary

**稳定组织、学习压力与必要重组的因果验证**

| 项目 | 定义 |
|---|---|
| Experiment ID | `CLM-001-R3m-Stability-Reorganization-Boundary-001` |
| 版本 | v1.0 |
| 仓库 | `ArcheLabs/mini-cells` |
| 实验性质 | 机制研究 / 配对训练 / 阶段性门禁 |
| 主要架构 | B-H：八步顺序 Cell Graph，稳定硬路由 |
| 次要架构 | A-H：四阶段 Top-2，机制复核 |
| Cell 数量 | 6 |
| 执行预算 | 每 Token 8 次 Cell 调用 |
| 建议运行环境 | Kaggle T4 × 2，按任务分片 |
| 科学范围 | `EXPLORATORY_DEVELOPMENT` |
| Confirmation | `SEALED` |
| 自动进入下一阶段 | 禁止 |

---

# 1. 研究背景

## 1.1 已有研究结论

MiniCells 的历史研究已经提供了以下证据。

**R3h：稳定路由具有训练价值**

与冻结参考训练路由相比：

- A-H：Development NLL 平均降低约 0.0269。
- B-H：Development NLL 平均降低约 0.0417。
- 三个 Seed 的改善方向一致。

这说明使 Cell 的训练路由更接近确定性推理路由，可以改善训练后的模型质量。

但 R3h 并未证明 Connection 自主学会了最优的 Cell 选择。

**R3i–R3k：动态 Connection 更新没有稳定的长期收益**

- R3i 完成 48 次 Connection Proposal。
- 8 次通过短期 Calibration 质量接受。
- R3j 发现短期收益可能在较长训练后反转。
- R3k 完成全部 48 个候选的长期评价。
- 五种早期价值指标均未获得预测支持。
- 没有候选达到预设的 0.005 NLL 实际改善门槛。

**R3l：扩大 Connection 参数搜索也没有明显改善**

- 扩展搜索未明显优于历史 Proposal。
- 100 Step 后的质量收益仍很小。
- 标签感知路径诊断发现了额外计算机会。
- 但没有证明这些路径可以被正常、无目标标签的 Router 利用。

**历史 022 与 023b：在明确的能力冲突环境下，组织分化有过正向证据**

- 022 在 Story/Arithmetic 的受控环境中观察到较强的功能分化信号。
- 023b 在临时分支试用中，能够拒绝无必要生长并接受具有正向价值的第一项能力分化。
- 后续连续能力生长仍不稳定。

## 1.2 一个尚未验证的共同假设

这些结果提示：

MiniCells 的组织变化可能并不是一个需要持续发生的优化过程。

在没有明显新学习需求时，稳定组织可能更有利于 Cell 累积能力。

只有当原有能力结构无法有效吸收新的学习需求时，重组才可能产生长期收益。

但该解释尚未由当前六 Cell 语言模型直接证实。

---

# 2. 核心科学问题

R3m 回答四个问题。

**Q1：稳定性是否具有独立的学习价值？**

在训练 Token、Cell 容量、Cell 更新预算和执行成本相同时，保持稳定路由是否比频繁调整路由更有利于能力形成？

**Q2：新任务是否真的产生组织需求？**

当 Story 训练逐渐引入 Arithmetic 时，是否出现可检测的学习冲突、能力干扰或现有组织容量不足？

**Q3：组织调整能否产生持续、可归因的收益？**

在新需求出现后，一次受控重组是否能够改善 Arithmetic 学习，同时保留 Story 能力，并在后续训练中保持收益？

**Q4：模型能否在不知道任务身份与切换时间的情况下，识别何时值得重组？**

如果存在真实重组机会，使用训练过程中可观测的压力信号触发 Connection 更新，是否优于固定时间更新或随机时间更新？

Q4 只有在前面的重组机会门禁通过后才进入正式验证。

---

# 3. 理论假设

## H1：稳定组织促进能力形成

令：

\[
M_t=(\theta_t,\phi_t)
\]

其中：

- \(\theta_t\)：Cell 参数。
- \(\phi_t\)：Connection 参数。

在任务分布不发生实质变化的条件下，假设保持适当稳定的 \(\phi_t\)，有助于 Cell 形成持续的学习分工。

需要检验：

\[
L_{\mathrm{Stable}}
<
L_{\mathrm{Perturbed}}
\]

但仅有质量差异还不够。

还必须考虑：

- 每个 Cell 的使用次数。
- 每个 Cell 接收的梯度。
- 训练路径与推理路径的一致性。
- 训练期间的数据分布变化。

否则不能将结果单独归因于组织稳定性。

## H2：学习需求变化可能使原有组织不再合适

定义现有组织对新任务的适应能力：

\[
A(\theta_t,\phi_t,\mathcal D_{\mathrm{new}})
\]

当原有组织能够有效学习新任务时，重组不一定有额外价值。

当出现持续的学习冲突、明显干扰或无法吸收的新能力需求时，重组可能变得有益。

这个假设不要求任务发生变化就必须重组。

## H3：重组的价值必须包含未来学习

令：

\[
V_T=
L(\theta^{S}_T,\phi^{S})
-
L(\theta^{R}_T,\phi^{R})
\]

其中：

- S：保持原组织。
- R：发生一次受控重组。
- T：从相同起点继续训练的步数。

如果：

\[
V_0\approx0
\]

但：

\[
V_{400}>0
\]

且在额外巩固训练后仍然保持正值，就支持组织变化通过后续 Cell 学习释放收益的解释。

但需要进一步使用四格归因确认 Cell 与 Connection 的贡献。

## H4：按需重组优于无条件重组

如果学习压力确实包含组织需求信息，那么根据该压力决定何时提出重组，应当比单纯周期性提出重组更有效。

同时，在没有新组织需求的环境中，该机制应当减少不必要的更新。

---

# 4. 实验范围与架构

## 4.1 主要研究 B-H

R3m 第一阶段主要使用现有 B-H 架构。

保留：

- 六个共享 Cell。
- 八步顺序执行。
- 每一步从六个 Cell 中选择一个。
- 当前计算状态与合法前驱信息。
- 原始 Connection 参数结构。
- 原始 Cell 内部计算。
- 原始 Tokenizer、Embedding 和 LM Head。
- 稳定硬路由。

每 Token 仍为八次 Cell 调用。

选择 B-H 的理由：

1. R3h 中 B 的路由对齐收益明显。
2. R3l 中 B 的长期重组结果存在微弱正向信号。
3. 顺序计算更直接地体现 Cell 之间的组合关系。

这只是实验优先级，不代表 B 已经被证明优于 A。

## 4.2 A-H 的地位

A-H 作为预先定义的次要机制复核架构。

不能将 A 和 B 的原始 NLL 直接混为一种模型的独立重复。

如果 B-H 第一阶段发现正向重组价值，应单独复核 A-H。

如果 B-H 没有发现价值，仍应报告 B-H 的限制，不自动用 A-H 搜索正面结论。

## 4.3 禁止改变的内容

本轮不得：

- 增加 Cell。
- 改变 Cell 宽度。
- 增加计算步数。
- 引入新的 Transformer Backbone。
- 修改语言模型输出目标。
- 改变 Attention 或 Memory 结构。
- 改变每 Token 的 Cell 执行预算。
- 添加额外 Task ID Embedding。
- 把任务标签直接提供给 Connection。
- 在评价时读取目标 Token 选择路径。

---

# 5. 两级实验设计

R3m 分为两个主要研究阶段。

## R3m-P：Reorganization Opportunity Pilot

研究：

**在当前六 Cell 架构中，是否存在真实、持续的组织重组收益？**

它是下一阶段的科学门禁。

## R3m-M：Autonomous Boundary Mechanism

研究：

**如果组织重组确实有价值，模型能否根据学习压力判断何时需要重组？**

只有 R3m-P 达到预注册门禁，才允许进入 R3m-M。

R3m-M 使用新的训练 Seed 和独立评价数据，不直接把 Pilot 的结果作为正式证明。

二者使用不同 Run ID、独立 Manifest 和明确的数据访问记录。

---

# 6. 任务与训练环境

## 6.1 基础语言任务：Story

使用现有 TinyStories 训练体系。

保持：

- 原始固定数据集 Revision。
- 4096 Vocabulary。
- 最大 Sequence Length 64。
- 相同 Story 训练分布。
- 相同训练数据读取规则。

历史训练来源：

`roneneldan/TinyStories`

实际 Source Revision、Tokenizer SHA-256、Story Range 和数据内容哈希必须从已验证的 R3h Manifest 读取。

不能只通过数据集名称假设数据相同。

## 6.2 新任务：Arithmetic

使用确定性生成的简单算术数据。

第一版建议限制为：

- 两位数加法。
- 包含进位和不进位。
- 固定的规范答案。
- 多种等价自然语言表达模板。

示例：

`What is 27 plus 18? Answer: 45.`

需要严格控制：

- 训练和评价的操作数组合。
- 模板分布。
- 数值范围。
- 答案长度。
- 进位比例。
- Tokenizer 可表达性。

同一算术问题的不同文字模板不能同时跨越训练与评价边界。

必须按照数学表达式的规范身份去重，而不只是比较字符串是否相同。

## 6.3 Arithmetic 评价指标

同时报告：

1. 完整序列 NLL。
2. 答案 Token NLL。
3. 完整答案 Exact Match。
4. 进位与不进位的分层结果。

**Arithmetic 的主要能力指标是答案 Token NLL 和 Exact Match，而不是模板文字的整体 NLL。**

如果模型只学会回答模板，却没有学会更准确地生成数值答案，不认为形成了算术能力。

## 6.4 任务学习可行性门禁

在正式研究重组之前，先验证：

当前模型在相同计算预算下能否从训练数据中学到非平凡的 Arithmetic 规律。

至少要求：

- Arithmetic 答案质量优于固定的无训练对照。
- 优势不只存在于训练过的具体操作数组合。
- 与仅学习答案频率的简单基线进行比较。
- 未见操作数组合上具有可测量泛化。
- 没有数据身份污染。

如果 Arithmetic 本身无法有效学习，则该任务暂时不足以验证组织重组。

状态：

`TASK_LEARNABILITY_GATE_FAILED`

这不是 Connection 理论被证伪。

---

# 7. 两种环境

## 7.1 Environment N：No New Demand

Phase I：

Story。

Phase II：

继续 Story。

Phase III：

继续 Story。

允许改变 Story 的样本顺序，但保持来源分布、训练 Token 和评价规则一致。

目的：

检验在没有预先指定的新能力需求时，组织更新是否会产生无益干扰。

不得将轻微训练噪声误判为任务切换。

## 7.2 Environment C：Capability Demand

Phase I：

Story。

Phase II：

逐渐引入 Arithmetic。

建议在 400 Step 中按照固定课程：

| 相对区间 | Story | Arithmetic |
|---|---:|---:|
| 0–100 | 90% | 10% |
| 100–200 | 70% | 30% |
| 200–400 | 50% | 50% |

按固定全局 Step 与 Token 配比实现，不使用当前质量动态调整。

Phase III：

继续 50% Story + 50% Arithmetic。

目的：

研究原有组织在新的能力学习需求下是否需要调整。

## 7.3 重要限制

**Environment C 不预先等于高组织需求。**

在正式解释结果前，还需要证明：

- Arithmetic 对原有模型构成了新学习压力。
- 新能力可以被当前容量学习。
- 训练过程中是否出现可测量干扰。
- 原始组织是否出现适应不足。

只有实际检测到相关机制，才能把它称为能力冲突环境。

---

# 8. 数据隔离

R3m 必须建立新的数据身份体系。

## 8.1 Story 数据

历史 Story 训练可复用原始 Train Range。

建议新建：

| 角色 | 建议 Story Range |
|---|---|
| `R3M_STORY_TRIGGER_CALIBRATION` | [10048, 10304) |
| `R3M_STORY_PILOT_REPORT` | [10304, 10560) |
| `R3M_STORY_MAIN_REPORT` | [10560, 11072) |

这些范围只是待核验的候选分配。

Codex 必须先查询仓库真实的 Dataset Registry、历史使用清单和正式 Confirmation 定义。

如果有任何重叠或冲突，停止准备并要求修正实验注册，不得静默选择其他区间。

Trigger Calibration 使用历史训练允许的数据角色，不得自动把上述评价 Story 用于压力阈值拟合。

## 8.2 Arithmetic 数据

生成四份完全独立的集合：

- `ARITH_TRAIN`
- `ARITH_TRIGGER_CALIBRATION`
- `ARITH_PILOT_REPORT`
- `ARITH_MAIN_REPORT`

必须按操作数、答案和模板族进行去重和分组隔离。

其中 Trigger Calibration 只用于开发阶段的规则冻结。

Pilot Report 用于 R3m-P。

Main Report 用于 R3m-M。

Main Report 在正式策略、阈值和停止规则冻结前不得读取。

## 8.3 数据级门禁

必须记录：

- Generator Source SHA。
- Generator 参数。
- Generator Seed。
- 原始问题数。
- 有效问题数。
- 重复排除数。
- 数据内容哈希。
- Tokenizer 身份。
- 分组泄漏检测。
- 各类答案的 Token 长度分布。

禁止使用 Pilot 或 Main Report 的结果修改题目难度、模板或数值范围。

Confirmation 始终保持 SEALED。

---

# 9. Phase I：共同能力形成

所有正式策略从相同的 Step 300 模型分支。

## 9.1 Pilot Checkpoint

优先复用 R3h 的三个 B-H Seed：

- `151001301`
- `151001302`
- `151001303`

从真实 Step 300 Checkpoint 恢复：

- Model。
- Cell Optimizer。
- Connection。
- Scheduler State。
- RNG。
- Data Cursor。
- Training Router。
- Temperature。
- ε 状态。

必须验证全部状态身份。

如果真实 Step 300 Checkpoint 不可恢复，不允许使用其他 Step 重新标记，也不能从已知质量结果反向挑选新起点。

可以重新执行相同历史 Phase I，但必须记录为新的基线生成流程，并完成一致性验证。

## 9.2 正式实验 Seed

R3m-M 使用全新训练 Seed：

- `151001401`
- `151001402`
- `151001403`
- `151001404`
- `151001405`

需要从相同初始化规则执行新的 300 Step Story 预训练。

每个 Seed 只训练一份共同 Phase I，再复制给所有环境和策略。

## 9.3 优化器与调度

本轮主训练继续 600 Step。

不能简单把 R3h 原本的 500 Step Scheduler 当作 900 Step 调度直接使用。

必须：

1. 恢复 Step 300 当前学习率。
2. 为后续 600 Step 定义固定、连续的新调度。
3. 不重新执行 Warmup。
4. 所有策略使用完全相同的调度。
5. 固定路由探索 ε 的后续时间表。
6. 记录调度来源和每 Step 的精确取值。

建议后续学习率从 Step 300 的实际值平滑衰减到其 20%，但该规则必须在 Pilot 开始前冻结，不得根据实验效果调节。

因此，R3m 的后续训练结果不能直接与历史 R3h 的 Step 500 NLL 当作同一训练计划进行数值归因。

---

# 10. R3m-P：重组机会存在性实验

这是最关键的第一阶段。

## 10.1 研究目标

不研究自动触发。

先验证：

**在新能力需求出现时，对真实 Connection 进行一次受控调整，是否能产生持续的训练收益？**

## 10.2 四组实验

每个 Seed、每种环境使用四组。

| Arm | Connection 策略 | 作用 |
|---|---|---|
| P0 — Stable | 始终冻结组织 | 主对照 |
| P1 — Single Reorganization | 在固定时点进行一次受约束的候选重组与试用 | 测试重组潜力 |
| P2 — Sham Intervention | 同样执行 Shadow 流程，但使用原组织 | 排除额外试训造成的收益 |
| P3 — Perturbed Stability | 施加预先固定的训练期组织扰动，并控制使用量 | 检查组织连续性 |

P1 的调整时间由协议预先指定。

例如：

Phase II 开始后第 100 Step。

**不能根据已知任务切换后的质量结果选择最佳时间。**

P2 保持相同的实际训练更新数和 Ghost Shadow 计算预算。

P3 应在训练中保持合法执行路径和相同 Cell 调用预算，同时将训练—推理对齐差异单独测量。它不是稳定性因果作用的唯一证据。

## 10.3 P1 候选规则

优先复用 R3i 的 Q-first Proposal 机制以及原 Trust Region。

原约束：

| 指标 | 上限 |
|---|---:|
| Policy KL | 0.01 |
| Hard Action Change | 0.10 |
| Full Path Change | 0.20 |

候选方向、Scale Scan、局部优化步数及接受规则必须预先冻结。

Proposal 只能使用：

- 历史训练状态。
- 当前已发生的训练数据。
- 当前 Cell 和 Connection。
- Train-only Calibration。

不能读取 Pilot Report。

## 10.4 必须验证干预有效

必须证明新候选：

- 不是零参数变化。
- 实际改变了合法 Cell 路径。
- 不只是浮点数差异。
- 在真实闭环执行中改变了路径。
- 没有超出固定计算预算。
- 没有破坏 Optimizer 状态。

如果 P1 从未产生有效路由变化，则输出：

`NO_EFFECTIVE_REORGANIZATION_INTERVENTION`

此时不能得出“重组没有价值”的科学结论。

这只能说明当前实验没有成功实施有效的重组干预。

## 10.5 额外的能力机会诊断

允许设置一个单独的、只使用 Train-only Search/Select 的探索性上界对照，尝试找到比 P1 更强的合法 Connection。

但其结果必须独立标记：

`TRAIN_SELECTED_CONNECTION_DIAGNOSTIC`

它不能参与 P1 的预注册主要比较。

即使搜索后发现较好的组织，也不能用该候选替换原本冻结的 P1。

否则会再次混淆算法能力与选择数据偏差。

---

# 11. Pilot 训练与观察

## 11.1 训练阶段

| 阶段 | Step | 内容 |
|---|---:|---|
| Phase I | 0–300 | 共同 Story 基础训练 |
| Phase II | 300–700 | No Demand 或 Capability Demand |
| Phase III | 700–900 | 固定最终组织，继续巩固 |

Pilot 复用 Step 300 的历史起点。

Phase II / III 共增加 600 Step。

所有组：

- 相同有效训练 Token。
- 相同主训练更新数。
- 相同 Optimizer 超参数。
- 相同数据顺序。
- 相同外生随机流协议。

P1 的 Shadow Adaptation 必须与真实训练时间一致。

不能额外训练 10 Step 后再继续主训练 600 Step，从而暗中增加训练预算。

## 11.2 评价时点

保存：

- T=0：Phase II 初始。
- T=100：第一次可重组时点。
- T=110：完成短期适应。
- T=200。
- T=400：Phase II 结束。
- T=600：Phase III 结束。

评价在所有分支到达相同训练屏障后执行。

不得根据中途评价决定是否提前停止某个候选。

## 11.3 训练预算

3 Seed × 2 Environment × 4 Arm × 600 Step：

\[
14,400
\]

次逻辑主训练更新。

每步 512 Token：

\[
7,372,800
\]

逻辑主训练 Token。

重复的物理轨迹允许哈希验证后复用，但必须单独报告实际计算量。

额外 Proposal 和 Shadow 计算另行统计。

---

# 12. Pilot 的三个科学门禁

## Gate P0：任务确实可学习

要求 Arithmetic 在当前模型规模和固定预算下表现出非平凡泛化。

否则：

`TASK_NOT_LEARNABLE_AT_CURRENT_SCALE`

不得将这一结果解释为 Connection 失败。

## Gate P1：重组干预真实有效

要求至少两个 Seed：

- 产生非零合法 Connection 改变。
- 在指定 Train-only Probe 上改变至少 5% 的 Hard Action。
- Cell 训练状态与原分支保持有效配对。
- 未破坏计算与数值约束。

5% 是本轮建议的干预有效性门槛，需在实验前冻结。

如果不能满足：

`REORGANIZATION_INTERVENTION_INSUFFICIENT`

此时不应进入自动触发机制研究。

## Gate P2：存在持续的重组价值

在 Capability Demand 下：

主要比较：

P1 vs P0。

必须同时检查：

1. Arithmetic 答案 Token NLL 改善。
2. Arithmetic Exact Match。
3. Story 保留质量。
4. Phase III 结束后的持久性。
5. 是否优于 Sham Intervention。
6. 是否存在 Cell–Connection 因果归因。

建议预注册的探索性效果门槛：

- Arithmetic Answer NLL 平均改善至少 0.005。
- Story NLL 相对 P0 的额外退化不超过 0.01。
- Phase III 结束仍保留至少一半 Phase II 结束时的正向收益。
- 至少两个 Seed 的主要收益方向为正。
- P1 相对于 P2 存在可测量优势。

Exact Match 必须完整报告，不得在其显著恶化时仅凭答案 Token NLL 宣称能力提升。

以上都是探索性工程决策阈值，并不是统计确认。

必须另行报告不确定性，且不能把三个 Seed 当作足够的独立科学确认。

## Pilot 结果分类

- `REORGANIZATION_OPPORTUNITY_OBSERVED`
- `NO_DETECTABLE_REORGANIZATION_OPPORTUNITY`
- `TASK_LEARNABILITY_INSUFFICIENT`
- `INTERVENTION_INSUFFICIENT`
- `INCONCLUSIVE`

只有第一个状态允许建议进入正式 R3m-M。

需要独立审阅后才能真正执行下一级研究。

---

# 13. R3m-M：自主组织边界实验

R3m-M 只在 Pilot 发现有效重组机会后实施。

主要目标：

**比较不同 Connection 更新时间决策，在稳定环境与变化环境中的长期训练结果。**

## 13.1 五种策略

| 策略 | 机制 |
|---|---|
| S0 — Stable | 始终不修改 Connection |
| S1 — Periodic | 固定周期提出更新 |
| S2 — Pressure Triggered | 学习压力达到阈值时提出更新 |
| S3 — Perturbed | 在预定义时段扰动组织连续性 |
| S4 — Random Scheduled | 使用预先冻结的随机更新时间 |

S1、S2、S4：

使用完全相同的 Proposal 算法、Trust Region 和 Shadow 接受协议。

主要区别只能是：

**何时启动一次 Connection Proposal。**

S3 属于独立稳定性诊断对照，不应与其他三种策略的正常更新门禁混淆。

---

# 14. 学习压力的定义

## 14.1 不使用 Task ID

压力检测器不能读取：

- `STORY` 或 `ARITHMETIC` 任务标签。
- 环境变量 N/C。
- 真实任务切换时间。
- 未来训练样本。
- 任何 Report/Confirmation 数据。

只能读取当前和历史已发生的训练信息。

## 14.2 三类候选压力信号

### A. Prequential Loss Surprise

对当前训练 Token，先预测再更新。

使用预测损失与历史 EMA 的差值：

\[
P_L(t)=
\frac{L_t-\mu_L(t-1)}
{\sigma_L(t-1)+\epsilon}
\]

只使用截至当前时刻之前的历史统计。

### B. Gradient Conflict

对连续训练窗口中共享 Cell 参数的梯度进行比较。

例如：

\[
P_G(t)=1-
\cos(g_t,\bar g_{t-1})
\]

其中 \(\bar g_{t-1}\) 只包含历史梯度。

不能使用 Task ID 对梯度提前分组。

### C. Routing Instability

测量当前输入下：

- Hard Action 分布变化。
- Full Path 分布变化。
- Cell 使用分布变化。
- 训练路由熵变化。

不能把路由熵变化本身直接视为质量改善或冲突证明。

## 14.3 冻结组合规则

第一版采用简单、低复杂度的压力指标：

\[
P_t=
\max(P_L(t),P_G(t))
\]

Routing Instability 暂时作为独立诊断，不参与触发阈值。

Loss 和 Gradient 分数必须先通过 Train-only Calibration 标准化。

不得依据 Main Report 拟合指标权重。

## 14.4 触发条件

建议：

- 连续两个监测窗口超过阈值。
- 每 20 Step 监测一次。
- 两次正式重组之间至少间隔 100 Step。
- Phase II 最多允许两次正式 Proposal。
- 没有达到条件则不更新。

实际阈值使用 Pilot 的 Train-only 数据和独立开发记录事先冻结。

不得为了确保每个 Seed 都触发而降低阈值。

无触发是有效研究结果。

---

# 15. 更新预算公平性

这是 R3m-M 必须解决的重要工程问题。

S2 可能因为学习压力大而启动更多 Proposal。

如果给它更多搜索和 Shadow 预算，就不能把收益归因于更好的触发时机。

因此：

## 15.1 固定最大提案数量

所有策略每个 Seed 的最大 Proposal 次数：

2。

S1：

在预注册的两个固定时间点提案。

S4：

在预注册的两个伪随机时间点提案。

S2：

根据压力触发，最多两次。

S0：

不修改组织。

## 15.2 Ghost Shadow Budget

所有策略都预留相同的额外计算预算。

没有真实 Proposal 时，执行不改变主训练状态的对照计算并明确记录。

必须区分：

- 实际主训练 Cell 更新。
- Shadow Cell 更新。
- 被丢弃的 Shadow 更新。
- Connection Proposal 优化。
- 评价计算。

Ghost 计算不应被当作真实训练贡献。

如果预算仍无法完全匹配，则必须提供质量—总计算成本的比较，不可宣称严格等算力。

## 15.3 不按最终接受次数匹配

接受次数本身属于策略结果。

不能因为 S2 接受了一次更新，就事后强迫 S1 或 S4 也必须接受一次。

这样会改变被比较的策略。

应匹配的是预先分配的计算机会和主训练预算，而不是事后结果。

---

# 16. R3m-M 训练预算

五个新 Seed。

两种环境。

五种策略。

共：

\[
5\times2\times5=50
\]

条逻辑训练分支。

每条从 Phase I 结束后继续 600 Step。

主训练更新：

\[
50\times600=30,000
\]

主训练 Token：

\[
\boxed{15,360,000}
\]

额外需要五个共享 Phase I 模型：

\[
5\times300=1,500
\]

次基础训练更新。

Phase I 新增训练 Token：

\[
768,000
\]

因此，在无物理轨迹复用的情况下：

**R3m-M 最大基础 + 主训练逻辑 Token 为 16,128,000。**

该预算不含 Proposal/Shadow/评价。

这是多 Session 实验规模，不能假设一次 Kaggle T4 × 2 Session 必然完成。

必须实施可恢复的任务分片与 Checkpoint 发布。

如果 Pilot 未通过，不执行这部分预算。

---

# 17. 主要质量评价

## 17.1 两种能力分别评价

Story：

\[
L_S
\]

Arithmetic Answer：

\[
L_A
\]

不得只报告二者的混合总体训练 CE。

因为模型可能通过更擅长其中一个任务而掩盖对另一个任务的损害。

## 17.2 旧能力保留

以 S0 为对照：

\[
D_{\mathrm{old}}=
L_{S2,S}-L_{S0,S}
\]

要求在规定容忍范围内。

同时以 Phase II 初始的共同 Checkpoint 为参考，报告绝对遗忘或改善：

\[
F_{\mathrm{old}}=
L_{\mathrm{final},S}
-
L_{\mathrm{anchor},S}
\]

避免仅因两个策略都出现严重遗忘，而错误地认为旧能力得到保留。

## 17.3 新能力收益

\[
G_{\mathrm{new}}=
L_{S0,A}-L_{S2,A}
\]

正值表示 S2 的算术答案 NLL 更低。

同时报告 Exact Match 和对不同算术子类型的表现。

## 17.4 稳定—变化交互

建立固定综合效用：

\[
U=
-\frac12
\left(
\frac{L_S}{L_{S,\mathrm{anchor}}}
+
\frac{L_A}{L_{A,\mathrm{anchor}}}
\right)
\]

归一化基准是相同 Seed 的 Phase I 末期模型评价值。

即使 Anchor 尚未训练 Arithmetic，仍必须记录其 Arithmetic 评价值，并验证其数值有效。

计算：

\[
\Delta_{\mathrm{boundary}}=
(U_{S2}-U_{S0})_C
-
(U_{S2}-U_{S0})_N
\]

主要观察：

- S2 是否在有新需求的环境中更有优势。
- S2 是否在无新需求的环境中避免不必要的重组。
- 这种优势是否在 Phase III 结束后仍存在。

该交互指标必须与 Story 保留、Arithmetic 能力和计算成本一起解释。

不能单靠综合效用掩盖某一能力的明显退化。

---

# 18. 持续收益与组织归因

在 Phase II 和 Phase III 结束后，使用四格归因。

同一 Seed、同一评价数据。

| Cell | Stable Connection | Reorganized Connection |
|---|---|---|
| Stable Cell | OO | ON |
| Reorganized Cell | NO | NN |

定义：

\[
V=L_{OO}-L_{NN}
\]

\[
R=L_{OO}-L_{ON}
\]

\[
C=L_{ON}-L_{NN}
\]

满足：

\[
V=R+C
\]

同时记录另一种交叉路径和非加性交互。

分别对 Story、Arithmetic 计算。

必须避免将跨 Cell 状态的交叉组合自然视为训练实际经历过的模型。

交叉组合用于机制干预，不代表具有正常训练轨迹的可部署模型。

## 18.1 必须记录的变化

- Cell 参数 L2 差异。
- 每 Cell 梯度累计。
- Cell 使用分布。
- 每 Cell 所处理的任务后验分布。
- Cell Ablation 效果。
- Connection 回退效果。
- 被接受的更新数量。
- 每次更新后的作用范围。
- Phase III 收益保持比例。

任务后验分布只能用于事后研究分析。

不得反馈给压力检测器或 Router。

---

# 19. 功能分化测试

组织重组改善 NLL 不一定意味着形成了不同能力。

因此需要独立功能性测试。

## 19.1 Cell Ablation

逐个关闭合法 Cell 的贡献，并使用相同的替代执行规则与固定计算成本记录质量变化。

分别测量：

- Story 质量损失。
- Arithmetic 质量损失。

如果关闭不同 Cell 对不同能力造成不同影响，可以作为功能分化的探索性证据。

但必须避免把 Cell 使用频率差异误判为能力特异性。

## 19.2 Functional Specificity

对于 Cell \(i\)：

\[
S_i=
D_{i,\mathrm{arith}}
-
D_{i,\mathrm{story}}
\]

其中 \(D\) 是受控干预造成的质量损失，需做尺度归一化。

要求与：

- Cell 使用频率匹配对照。
- Cell 随机身份打乱对照。
- Stable 组织对照。

进行比较。

## 19.3 能力保持

Phase III 必须重新进行功能干预测试。

如果 Phase II 出现的分化在 Phase III 消失，不认为已经形成稳定的能力组织。

---

# 20. 统计计划

## 20.1 Pilot

三个历史 Seed。

Pilot 只能提供探索性门禁，不用于正式统计确认。

完整报告：

- 均值。
- 每 Seed 原始结果。
- Story-cluster 配对 Bootstrap。
- 数值重复误差。
- 最大、最小效果。
- 失败案例。

## 20.2 Main

五个新的独立训练 Seed。

保持：

- 所有策略配对相同初始状态。
- 相同训练数据。
- 相同外生随机流。
- 相同 Step 预算。

主要比较：

1. Capability Demand 中 S2 vs S0。
2. Capability Demand 中 S2 vs S1。
3. Capability Demand 中 S2 vs S4。
4. Stability Boundary 交互效应。
5. No Demand 中 Stable vs Perturbed。

预注册主终点、方向和多重比较家族。

使用 Holm 调整。

Seed 是训练重复单位。

Story Bootstrap 只量化固定模型下的评价样本不确定性。

不得把 512 个 Story 当成 512 个独立训练重复。

## 20.3 实际改善

保持 0.005 NLL 作为预注册的主要实际改善参考门槛。

Arithmetic Answer NLL、Story NLL 和标准化综合效用必须分别定义阈值，不能混用。

对于功能分化、触发误报率和总计算成本，应另行冻结门槛。

不得在结果出现后重新定义科学成功标准。

---

# 21. 必须区分的失败类型

R3m 不允许用一个 `FAILED` 代替所有研究问题。

至少包含：

| 结果 | 科学含义 |
|---|---|
| `TASK_NOT_LEARNABLE` | 当前模型或任务设置不能支持能力学习实验 |
| `NO_EFFECTIVE_REORGANIZATION` | 现有干预没有实际改变组织 |
| `NO_ORGANIZATION_DEMAND_SIGNAL` | 未观察到足够的组织需求证据 |
| `NO_REORGANIZATION_UTILITY` | 实施了有效重组，但没有质量收益 |
| `REORGANIZATION_UTILITY_OBSERVED` | 观察到持续、受控的重组收益 |
| `PRESSURE_TRIGGER_NOT_PREDICTIVE` | 压力信号无法区分值得重组的时机 |
| `BOUNDARY_SIGNAL_EXPLORATORY` | 按需重组在适当环境中产生探索性优势 |
| `INCONCLUSIVE` | 证据不足以支持正面或负面结论 |
| `ENGINEERING_INVALID` | Checkpoint、数据或执行语义失效 |

没有有效候选与有效候选无收益，必须严格区分。

---

# 22. 工程目录

建议：

    research/projects/
      cellular-language-models/
        experiments/
          clm-001-qfirst-integrated/
            r3m/
              SPEC.md
              README.md

              config/
                pilot.frozen.json
                main.proposed.json

              implementation/
                minicells_r3m/
                  parent_restore.py
                  data_identity.py
                  arithmetic_data.py
                  task_learnability.py
                  curriculum.py
                  model_adapter.py
                  pressure_monitor.py
                  proposal_adapter.py
                  shadow_transaction.py
                  intervention.py
                  training.py
                  evaluation.py
                  attribution.py
                  specialization.py
                  statistics.py
                  checkpointing.py
                  reports.py
                  runner.py

              notebooks/
                kaggle-pilot.ipynb
                kaggle-main.ipynb

              reports/
                pilot/
                main/

## 22.1 复用要求

优先复用：

- R3h 的模型、Trainer 与路由对齐。
- R3i 的 Proposal 与 Trust Region。
- R3j 的 Shadow Adaptation。
- R3k 的配对长期评价。
- R3l 的模型执行与候选干预。
- 022 的受控学习冲突分析思路。
- 023b 的临时试用与接受/拒绝思想。
- 现有 HF Checkpoint Store。
- 现有 GitHub 报告发布流程。

历史 022/023b 使用不同的实验架构和数据体系，不能直接复制其模型参数或评价结果作为 R3m 的正式证据。

---

# 23. Checkpoint 与跨 Session 恢复

所有 Checkpoint 必须保存：

- Experiment ID。
- Source SHA。
- Config SHA-256。
- Run ID。
- Dataset Hash。
- Model State。
- Cell Optimizer State。
- Connection State。
- Training Step。
- LR Scheduler。
- Data Cursor。
- CPU RNG。
- CUDA RNG。
- Route RNG。
- Curriculum State。
- Pressure Detector State。
- Proposal Transaction State。
- Accepted/Rejection History。

如果发生 Shadow 试训，必须保存 Old/New 双分支状态与相应接纳决策。

被拒绝的分支不能污染正式模型。

## 23.1 跨 Session 门禁

R3l 的 HF 引用目前标记：

`cross_session_verified=false`

R3m 不能依赖未经独立恢复确认的新生成模型状态来证明跨 Session 可恢复性。

正式训练前必须：

1. 在独立 Session 下载。
2. 校验 Manifest。
3. 校验模型和 Optimizer。
4. 恢复 Trainer。
5. 执行固定的下一 Step。
6. 与预保存的恢复参考结果比较。

若失败：

`CROSS_SESSION_RESTORE_INVALID`

不得继续正式 GPU 科学训练。

---

# 24. 运行方式

由于 R3m-M 计算规模较大，不应假设一次 Kaggle Session 能完成全部实验。

建议采用明确的任务分片：

- Pilot：按 Seed × Environment 分片。
- Main：按 Seed × Environment × Strategy 分片。
- 每个分片最多使用一张 T4。
- 两张 GPU 可并发执行两个独立分片。
- 每个分片完成后持久化完整 Checkpoint。
- 聚合评价等待所有要求的分片通过训练门禁。

训练期间不得访问正式 Report 数据进行超参数调整。

完整结果屏障通过后才执行正式比较。

Checkpoint 发布失败允许单独重试发布，不允许因为发布失败而重新训练并替换已完成结果。

---

# 25. 必须交付的报告

## Pilot

- `INPUT_VERIFICATION.json`
- `DATA_BOUNDARY_PROOF.json`
- `TASK_LEARNABILITY.json`
- `PILOT_MODEL_IDENTITY.json`
- `PILOT_INTERVENTIONS.json`
- `PILOT_LEARNING_CURVES.json`
- `PILOT_STORY_RETENTION.json`
- `PILOT_ARITHMETIC_ACQUISITION.json`
- `PILOT_LONG_HORIZON.json`
- `PILOT_ATTRIBUTION.json`
- `PILOT_COMPUTE_COST.json`
- `PILOT_SCIENTIFIC-FINDINGS.md`
- `PILOT_RUN_MANIFEST.json`

## Main

- `MAIN_FROZEN_PROTOCOL.json`
- `PRESSURE_TRIGGER_EVENTS.json`
- `CONNECTION_TRANSACTIONS.json`
- `ENVIRONMENT_INTERACTION.json`
- `LONG_HORIZON_RETENTION.json`
- `CAPABILITY_SPECIALIZATION.json`
- `CELL_CONNECTION_ATTRIBUTION.json`
- `COMPUTE_COST.json`
- `STATISTICAL_ANALYSIS.json`
- `SCIENTIFIC-FINDINGS.md`
- `RUN_MANIFEST.json`

完整模型保存在私有 HF。

GitHub 仅发布可公开的身份、数值和科学报告。

必须保留所有负面案例。

---

# 26. 科学完成门禁

## Pilot

只有当以下条件均满足，才能声明 Pilot 工程完成：

- 三个 Seed 的模型身份有效。
- 两种环境数据身份有效。
- Arithmetic 可学习性已检测。
- 所有主要策略完成相同训练预算。
- 有效 Connection 干预得到确认或明确失败。
- Story 与 Arithmetic 分离评价完成。
- Phase III 持久性评价完成。
- 四格归因完成。
- 数值重复性通过。
- 没有提前使用 Pilot Report 进行模型选择。
- Confirmation 始终 SEALED。

Pilot 工程完成不等于理论得到支持。

## Main

必须额外满足：

- Pilot 的进入门禁通过并经过审阅。
- 五个新 Seed。
- 正式配置冻结。
- 压力检测器冻结。
- 触发预算冻结。
- 所有主要策略完成。
- 环境交互分析完成。
- 旧能力保留评价完成。
- 新能力形成评价完成。
- 功能分化干预完成。
- 计算成本配对分析完成。
- 独立 Main Report 无泄漏。
- 没有自动调整研究阈值。
- 没有启动后续实验。

---

# 27. R3m 的科学决策

最终必须分别回答：

### Q1：稳定的组织是否具有持续学习价值？

结果：

- `STABILITY_VALUE_SUPPORTED_EXPLORATORY`
- `STABILITY_VALUE_NOT_SUPPORTED`
- `INCONCLUSIVE`

### Q2：新任务是否产生了可测量的组织需求？

结果：

- `ORGANIZATION_DEMAND_OBSERVED`
- `NO_DETECTABLE_ORGANIZATION_DEMAND`
- `INCONCLUSIVE`

### Q3：重组是否产生持续的额外质量收益？

结果：

- `PERSISTENT_REORGANIZATION_GAIN`
- `NO_PERSISTENT_REORGANIZATION_GAIN`
- `INCONCLUSIVE`

### Q4：是否能自主识别重组时机？

结果：

- `AUTONOMOUS_BOUNDARY_SIGNAL_EXPLORATORY`
- `NO_AUTONOMOUS_BOUNDARY_SIGNAL`
- `NOT_TESTED`
- `INCONCLUSIVE`

以上四项必须独立判定。

不能因为一个指标通过，就自动升级其他三个科学结论。

---

# 28. 给 Codex 的完整执行要求

请在 `ArcheLabs/mini-cells` 中实施：

`CLM-001-R3m-Stability-Reorganization-Boundary-001`

本轮重点不是提高 Connection Proposal 的数量或成功率，而是验证稳定组织与必要重组之间是否存在真实、持续的学习价值分界。

必须执行：

1. 先实施完整 R3m-P，不直接启动 R3m-M。
2. 优先复用可验证的 R3h B-H Step 300 Checkpoint。
3. 保留现有六 Cell、八步顺序执行和模型容量。
4. 新建严格隔离的 Story/Arithmetic 数据。
5. 验证 Arithmetic 在当前模型中具有可学习性。
6. 设置 No Demand 与 Capability Demand 两种环境。
7. 设置 Stable、Single Reorganization、Sham 和 Perturbed Stability 对照。
8. 保留原 Connection Trust Region。
9. 实施真实、可验证的 Connection 干预。
10. 匹配实际训练 Token、Cell 更新数和执行预算。
11. 完成 Phase II 与 Phase III 训练。
12. 评价 Story 保留和 Arithmetic 能力。
13. 完成 Cell–Connection 四格归因。
14. 检查 Cell 功能分化与有效使用。
15. 独立报告任务可学习性、干预有效性和重组价值。
16. 严格保留负面结果。
17. 通过独立 Session 验证 Checkpoint 恢复。
18. GitHub 发布完整聚合结果。
19. 不读取 Confirmation。
20. 不自动进入 R3m-M。

同时完成 R3m-M 的可执行工程准备，包括：

- 五个新的训练 Seed。
- 五种组织策略。
- 基于历史训练信息的压力检测器。
- 固定 Proposal 和 Shadow 预算。
- 独立 Main Report。
- 任务分片与断点恢复。
- 主要科学统计规则。

但只有在 Pilot 获得 `REORGANIZATION_OPPORTUNITY_OBSERVED`、研究门禁经过审阅并完成 Main 配置冻结后，才允许启动 Main 科学训练。

**最终必须明确回答：**

1. 当前六 Cell 架构是否存在真实的新能力重组需求？
2. 稳定组织是否在无新需求时更有利于持续学习？
3. 受控 Connection 调整能否在新需求出现后产生持续、可归因的收益？
4. 是否有充分理由进一步研究自主识别重组时机？

如果第一阶段没有发现真实重组机会，应停止进一步的触发器研究，而不是增加搜索规模来寻找正面结果。

这不是整个 MiniCells 理论被证伪，而是说明当前架构、任务和训练预算尚未建立可利用的重组价值。

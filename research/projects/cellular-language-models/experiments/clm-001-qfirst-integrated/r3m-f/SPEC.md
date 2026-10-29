# MiniCells R3m-F — Arithmetic Learnability & Cell Specialization

**算术可学习性、能力形成与 Cell 专业分化实验**

版本：v1.0 — Proposed

仓库：`ArcheLabs/mini-cells`

实验 ID：`CLM-001-R3m-F-Arithmetic-Specialization-001`

实验类型：`EXPLORATORY_DEVELOPMENT`

前序实验：R3m-E、022、023b

## 0. 核心目标

本实验验证以下假设：

**不同学习需求能够使 Cell 形成任务相关的功能差异，但这种局部分化未必立即表现为完整能力。随着能力学习逐渐成熟，Cell 的专业分工可能变得更加清晰。**

R3m-F 将重点回答：

1. 当前六 Cell 模型能否真正学习简单算术，而不只是学习算术文本的统计分布？
2. Story + Arithmetic 是否比 Story-only 更容易形成特定 Cell 的功能分化？
3. Cell 的任务特异性是否伴随实际算术能力的提高？
4. 如果模型无法学习算术，限制来自训练设置、表示能力，还是 Cell 架构本身？

本轮**不研究 Connection 重组**。

---

# 1. 实验依据

## 1.1 Experiment 022

此前的 Emergent Trait Bifurcation 实验观察到：

- Story 与 Arithmetic 之间存在可辨别的任务相关功能差异。
- 任务身份区分指标均值约为 0.158。
- 严格容量对照约为 0.001。
- 平均 Routing Purity 约为 89.8%。
- 三个重复实验均达到预设的相关机制门槛。

但该实验使用的是最小 TextNCA 表型架构，并非当前六 Cell 语言模型。

## 1.2 Experiment 023b

Probationary Trait Genesis 观察到：

- Story-only：3/3 拒绝新增 Trait。
- Duplicated Story：3/3 拒绝新增 Trait。
- Story + Arithmetic：3/3 接受有价值的临时 Trait 分化。
- Weak Arithmetic：0/3 接受。

这表明学习需求的性质可能影响专业分化的形成与接纳。

但它仍不等于证明当前语言模型能够形成成熟的算术 Cell。

## 1.3 R3m-E

R3m-E 已完成：

- 3 Seed。
- 4 Arm。
- 12/12 任务。
- 每组 600 Step 延续训练。

主要结果：

| 指标 | 结果 |
|---|---|
| Arithmetic Answer NLL | 明显下降 |
| Arithmetic Exact Match | 0% |
| Cell 1 对 Arithmetic 的功能敏感性 | 三个 Seed 均出现正向变化 |
| Connection 更新接纳 | 1/3 |
| 长期重组收益 | 未观察到 |
| 算术可学习性门禁 | 未通过 |

尤其值得关注：

三个 Seed 中，Cell 1 的使用频率匹配替换干预，对 Arithmetic Answer NLL 的影响相对于 Story-only 明显增加。

但这不能单独证明 Cell 1 学会了算术。

## 1.4 下一步判断

**应当先证明模型能够获得可验证的新能力，再研究这种能力如何在 Cell 之间形成和分布。**

如果算术始终无法被学会，继续调整 Connection 可能只是在优化一种尚不存在的能力。

---

# 2. 科学问题

## Q1：Arithmetic Learnability

在冻结 Connection、保持原始六 Cell 架构的条件下，当前模型是否能够掌握一位数加法？

要求使用未见操作数组合验证。

不得仅以训练损失下降或 Teacher-forced NLL 改善判定成功。

## Q2：Functional Differentiation

在混合训练后，不同 Cell 是否承担不同的任务功能？

需要区分：

- 路由频率变化。
- 参数梯度变化。
- 对任务预测结果的因果贡献。
- 稳定、可重复的专业功能。

## Q3：Specialization Maturation

如果 Arithmetic Exact Match 逐渐提高，Cell 的算术特异性是否同时增强？

需要记录能力和功能分化的共同演化，而不仅比较最终 Checkpoint。

## Q4：Learning Bottleneck

若简单算术无法学会，需要区分：

- 数据生成或评价实现错误。
- Tokenizer 与答案表示问题。
- 训练 Token 数量不足。
- 学习率或优化器不合适。
- 固定 Embedding/LM Head 的表达限制。
- Cell 模型本身的计算或表示能力限制。

不得将所有负面结果直接归因于 Cell 架构。

---

# 3. 模型与训练边界

沿用现有六 Cell B-H 语言模型。

| 项目 | 配置 |
|---|---|
| Cell 数量 | 6 |
| 执行机制 | 8 步顺序 Cell 调用 |
| Connection | 全程冻结 |
| Tokenizer | 原始 4096 Vocabulary |
| Sequence Length | 64 |
| Token / Step | 512 |
| Optimizer | 原始 Cell AdamW |
| 模型来源 | R3h B-H Step 300 |
| Seed | 151001301、151001302、151001303 |

必须保持：

- Cell 参数结构不变。
- Connection 参数结构不变。
- 每 Token 的 Cell 调用次数不变。
- Embedding 和 LM Head 保持原有冻结规则。
- 不增加模型参数。
- 不增加 Task ID Embedding。
- 不使用 Task Label 进行路由选择。
- 不引入外部预训练模型。

Task Label 可以用于训练数据编排和事后的分析分组，但不能输入 Connection。

本轮不允许因为结果不理想而改变 Cell 容量或动态增加训练步数。

---

# 4. 任务设计

为了减少 R3m-E 中两位数加法过难造成的不确定性，采用由易到难的任务体系。

## 4.1 Task C — Numeric Copy

数字复制任务。

示例：

`Repeat the number 17. Answer: 17.`

作用：

- 验证数字 Token 是否可以正确输出。
- 验证答案 Mask。
- 验证自回归生成。
- 验证模型能否完成简单的数值条件生成。
- 建立与算术任务相近的答案长度和数字 Token 分布。

此任务不用于证明算术能力。

尽可能让 Copy 与 Arithmetic 的答案数值分布匹配，以减少答案词频差异。

## 4.2 Task A1 — One-Digit Addition

一位数加法。

操作数：

`0–9`

示例：

`What is 3 plus 4? Answer: 7.`

`What is 8 plus 7? Answer: 15.`

按交换律规范化：

`(min(a,b), max(a,b))`

共有 55 个规范化加法事实。

训练与开发集按照事实身份分组，不能将 `3+4` 和 `4+3` 分别放入 Train 与 Dev。

建议约 80% 用于训练、20% 用于开发，具体整数数量和分层分配在数据生成前冻结。

必须覆盖：

- 结果小于 10。
- 结果大于或等于 10。
- 含零操作数。
- 相同操作数。
- 不同答案频率。

一个事实的不同文字模板仍属于同一数据组。

## 4.3 Task A2 — Two-Digit Addition Without Carry

两位数无进位加法。

例如：

`23 + 14 = 37`

要求个位和十位均不发生向更高位的进位。

这与仅检查个位是否进位不同。

用于判断模型能否从较简单的加法泛化到更长的数字表达。

## 4.4 Task A3 — Two-Digit Addition With Carry

两位数有进位加法。

例如：

`27 + 18 = 45`

`78 + 65 = 143`

需要明确区分：

- 个位向十位进位。
- 十位向百位进位。
- 两种进位同时存在。

A2 和 A3 暂不允许进入第一轮 GPU 训练。

**第一轮只运行 Copy 与 A1。**

只有 A1 形成可靠学习信号后，才单独审阅并冻结 A2/A3 的后续协议。

---

# 5. 先执行算术训练能力预检

在完整实验之前，先执行一个成本较低的真实模型预检。

## 5.1 Tiny Overfit Test

使用一个 Seed，从真实 R3h Step 300 Checkpoint 恢复。

选择预先固定的 8 个 A1 加法事实。

仅使用这 8 个事实进行训练。

建议最大预算：

`200 Step × 512 Token`

同时进行：

- 模型预测。
- Answer Token NLL。
- Autoregressive Exact Match。
- 梯度有效性检查。
- 参数更新检查。
- Checkpoint 恢复检查。

目的：

**验证当前训练机制至少具有记住少量简单算术映射的能力。**

建议将训练事实准确率达到 75% 作为预检通过的最低工程参考值，执行前冻结，并报告每道题的实际结果。

这不是科学泛化门槛。

如果 200 Step 后几乎无法正确回答任何训练事实：

`TINY_OVERFIT_FAILED`

暂停完整实验，检查数据、Mask、Decoder、Optimizer 与模型表示能力。

不得在相同预检身份下无限增加训练时长直到成功。

## 5.2 答案评价完整性

必须确认：

1. 答案 Token Mask 精确覆盖答案本身。
2. 不将已知答案 Token 错误暴露在自回归生成输入中。
3. 正确处理 EOS。
4. 完整答案 Exact Match 基于实际生成文本。
5. 不使用 Teacher Forcing 的答案预测冒充自由生成结果。
6. 正确处理多 Token 数字。
7. 记录首个答案 Token 的准确率。
8. 记录整段答案的准确率。
9. 分别报告一位数和两位数答案。

保存每道开发题的预测答案，以便检查错误模式。

---

# 6. 正式实验组

完整 Pilot 使用三个 Arm。

| Arm | 数据 | Connection | 目标 |
|---|---|---|---|
| F0 — Story Only | 100% Story | Stable | 原能力基线 |
| F1 — Story + Copy | Story + Numeric Copy | Stable | 数字输出能力对照 |
| F2 — Story + A1 | Story + One-Digit Addition | Stable | 核心算术能力实验 |

三个 Arm 均从相同 Seed 的 R3h B-H Step 300 Checkpoint 开始。

同一 Seed 的全部 Arm：

- 模型参数相同。
- Optimizer 初始状态相同。
- Connection 相同。
- 训练调度相同。
- 主训练 Token 总量相同。

F1 与 F2 的 Story Token 比例必须相同。

Copy 与 A1 的答案长度、数值频率和训练预算尽可能匹配；残余差异必须报告。

F0 只作为 Story-only 参照，不能当作等 Arithmetic 数据预算的对照。

## 6.1 训练比例

建议：

| Arm | Story | 新任务 |
|---|---:|---:|
| F0 | 100% | 0% |
| F1 | 50% | 50% Copy |
| F2 | 50% | 50% Arithmetic |

固定每个任务的有效 Token 分配。

不能用样本条数代替 Token 预算。

---

# 7. 训练预算

建议完整 Pilot 使用：

- 3 Seed。
- 3 Arm。
- 每个 Arm 900 Step。
- 每 Step 512 Token。

总逻辑主训练：

`3 × 3 × 900 = 8,100 updates`

`8,100 × 512 = 4,147,200 tokens`

不包括共享的历史 Step 300 训练和 Tiny Overfit 预检。

这是一个独立、冻结的探索性预算。

不得根据前几个评价结果决定给某个 Arm 追加训练更新。

如果在预算结束时仍没有学习信号，如实报告，不自动扩大预算。

---

# 8. 评价时点

从 Step 300 共同 Checkpoint 开始计算相对训练步数。

| 相对 Step | 全局 Step | 目的 |
|---|---:|---|
| 0 | 300 | 基线 |
| 100 | 400 | 初期学习 |
| 300 | 600 | 中期学习 |
| 600 | 900 | 后期学习 |
| 900 | 1200 | 最终结果 |

所有 Arm 使用相同评价时点。

评价只使用开发数据，不更新参数。

不得利用开发结果动态决定训练时长、学习率或数据难度。

---

# 9. 能力评价

## 9.1 Copy

报告：

- Answer NLL。
- 首个数字 Token 准确率。
- 完整数字 Exact Match。
- 不同数字长度的正确率。

如果 F1 无法完成 Copy，应优先调查输入表示、Decoder、Embedding/Head 与训练机制。

不应直接认为 F2 失败意味着模型不能计算加法。

## 9.2 Arithmetic

报告：

- Answer Token NLL。
- Full Sequence NLL。
- Autoregressive Exact Match。
- 训练事实准确率。
- 未见事实准确率。
- 结果小于 10 的准确率。
- 结果大于或等于 10 的准确率。
- 不同模板下的表现。
- 与答案频率基线的差距。

必须以规范化算术事实作为独立评价单位。

同一道题生成多个模板，不等于多个独立测试样本。

## 9.3 学习判定

对于 A1，建议：

`ARITHMETIC_LEARNABILITY_OBSERVED`

需要满足：

1. 未见事实的 Exact Match 明确优于预先登记的答案频率基线。
2. 至少两个 Seed 出现正向泛化信号。
3. 训练事实正确率高于初始模型。
4. Answer NLL 改善与实际答案准确率方向一致。
5. 不是只在训练模板上取得正确答案。

可以在执行前登记例如“未见事实准确率至少 50%”的探索性参考门槛，但必须保留实际分母、置信区间和基线表现。

三个 Seed 和少量一位数事实不足以构成统计确认。

如果 Copy 成功而 Arithmetic 不成功：

`COPY_LEARNABLE_ARITHMETIC_NOT_ESTABLISHED`

如果两个任务均无法学习：

`BASIC_LEARNABILITY_BOTTLENECK`

---

# 10. Cell 专业分化测量

这是本轮的第二个核心目标。

所有专业分化分析都必须与真实能力学习结果一起解释。

## 10.1 Cell 使用分布

对 Story、Copy、Arithmetic 分别统计：

- 六个 Cell 的使用频率。
- 八个执行阶段中的 Cell 分布。
- 不同任务的 Cell 使用差异。
- 不同训练时点的变化。
- 不同 Seed 的稳定性。

不能仅凭路由使用差异宣布专业分化。

## 10.2 Cell 梯度

在冻结的 Train-only Probe 上测量：

- 每个 Cell 的 Story 梯度。
- Copy 梯度。
- Arithmetic 梯度。
- 梯度范数。
- 完整有符号梯度的方向相似性。
- 梯度冲突与变化。

不允许通过开发集的目标答案调节训练 Router。

梯度冲突不能单独证明能力分化。

## 10.3 使用频率匹配的 Cell 干预

复用 R3m-E 已验证的合法 Cell Replacement 方法。

对每个 Cell 分别执行：

- Usage-matched replacement。
- Fixed-ID replacement。
- Random-ID replacement。

所有干预只用于事后机制分析。

记录相对于未干预模型的：

- Story NLL 变化。
- Copy Answer NLL 变化。
- Arithmetic Answer NLL 变化。

同时保持合法执行路径和相同 Cell 调用预算。

## 10.4 专业能力指标

对同一任务，定义 Cell i 的替换影响：

\[
\Delta_i^A =
L_A(\text{replace Cell }i)-L_A(\text{original})
\]

\[
\Delta_i^S =
L_S(\text{replace Cell }i)-L_S(\text{original})
\]

正值表示替换后性能恶化，该 Cell 对当前任务有正向贡献。

由于 Answer NLL 与 Story NLL 的基线不同，跨任务比较需要预先固定归一化方式。

定义归一化任务特异性：

\[
F_i =
\widetilde{\Delta_i^A}
-
\widetilde{\Delta_i^S}
\]

但仅有 \(F_i>0\) 仍不足以证明专业化。

还必须比较：

- F0 Story-only。
- F1 Story + Copy。
- F2 Story + Arithmetic。
- 同一 Cell 随训练时点的变化。
- 相同使用频率的对照干预。

重点检查：

**F2 中的 Cell 特异性是否超出 F1 中单纯数字处理产生的特异性。**

---

# 11. Cell 1 专项假设

R3m-E 已观察到 Cell 1 在三个 Seed 中对 Arithmetic 答案预测更敏感。

本轮提前登记一个定向探索性假设：

**H1：在学会 A1 加法的过程中，Cell 1 的 Arithmetic 特异性相对于 Story-only 和 Copy 对照进一步提高。**

必须报告 Cell 1 在：

- Step 300。
- Step 400。
- Step 600。
- Step 900。
- Step 1200。

的功能影响。

同时报告其他五个 Cell。

不得只筛选 Cell 1 的有利结果。

**Cell 1 是根据既往 R3m-E 结果选定的，因此本轮仍属于后续探索性复核，而不是完全独立的确认性检验。**

更强的结论需要新的独立 Seed、数据和冻结协议。

---

# 12. 能力形成与专业分化的关系

本轮最重要的不只是最终指标，而是随训练发展产生的变化。

对每个 Seed 记录：

\[
A(t)=\text{Arithmetic Exact Match}
\]

\[
F_i(t)=\text{Cell }i\text{ 的任务特异性}
\]

分析：

1. 是否先出现 Cell 功能差异，再出现 Arithmetic 正确率提高？
2. Arithmetic 正确率提高时，是否有特定 Cell 的贡献增强？
3. Copy 学习是否产生相同的 Cell 变化？
4. Story-only 是否也自然产生相同趋势？
5. 所观察到的关系是否在不同 Seed 上一致？

这种分析属于探索性关联证据。

只有在同一任务、同一训练时点执行干预并观察相应性能变化，才能支持局部功能贡献；不能仅凭时间先后证明 Cell 分化导致了算术能力形成。

---

# 13. 结果分类

## Outcome A：Copy 和 Arithmetic 都失败

`BASIC_LEARNABILITY_BOTTLENECK`

不能评价专业分化理论。

下一步应检查：

- 数字表示。
- Decoder。
- 答案 Mask。
- 冻结 Embedding/LM Head。
- 优化器。
- 当前架构的表达限制。

允许另立独立诊断实验检查冻结组件的影响，但不得在本实验中偷偷解冻参数。

## Outcome B：Copy 成功，Arithmetic 失败

`ARITHMETIC_LEARNING_NOT_ESTABLISHED`

说明模型具备至少一种数值条件输出能力，但尚未学会算术关系。

下一步可研究课程设计、两位数之前的更小能力单元，以及固定表示对计算规则学习的限制。

## Outcome C：Arithmetic 成功，但无专业分化

`LEARNABLE_WITHOUT_DETECTED_SPECIALIZATION`

说明当前模型能够学会任务，但没有观察到明确的 Cell 专业组织。

这会对“新能力必然导致 Cell 分化”的强假设构成负面证据。

## Outcome D：Arithmetic 成功且 Cell 分化

`FUNCTIONAL_SPECIALIZATION_SIGNAL`

说明新任务学习与可干预的 Cell 功能分化同时出现。

如果该信号在 Copy 对照中不存在，且具备跨 Seed 一致性，则进一步支持任务相关能力分工的解释。

但仍不能直接证明 Connection 重组具有价值。

## Outcome E：出现功能差异，但 Arithmetic 仍未学会

`PARTIAL_SPECIALIZATION_WITHOUT_COMPETENCE`

这可能与 R3m-E 的结果一致：

Cell 已对任务产生不同反应，但整体模型尚不具备正确解决任务的能力。

不能将这种情况报告为成功的算术专业化。

---

# 14. 后续 A2/A3 门禁

只有 A1 学习获得正向证据，才考虑：

`R3m-F2 — Two-Digit Arithmetic Progression`

建议按顺序：

1. A1：一位数加法。
2. A2：两位数无进位加法。
3. A3：两位数有进位加法。

每个阶段均使用单独冻结的训练及开发数据身份。

不得在同一次运行中根据开发成绩任意调整阶段长度。

跨阶段需要报告：

- 已有能力保留。
- 新能力 Exact Match。
- Cell 功能角色变化。
- 专业分化是否稳定。
- 是否有新的 Cell 参与新计算需求。

如果多个能力阶段均表现出稳定分化，才重新讨论 Connection 重组与自主组织形成。

---

# 15. 工程要求

## 15.1 复用既有代码

优先复用：

- R3h B-H Model。
- R3h Checkpoint Restore。
- R3m-E Trainer。
- R3m-E Data Scheduler。
- Arithmetic Generator。
- Evaluation Pipeline。
- Cell Replacement。
- Gradient Conflict。
- Artifact Store。
- Run Manifest。
- GitHub Publication。

不重新实现已有的模型与持久化机制。

## 15.2 新目录

建议：

    research/projects/cellular-language-models/
      experiments/
        clm-001-qfirst-integrated/
          r3m-f/
            README.md
            SPEC.md

            config/
              exploratory.frozen.json

            implementation/
              arithmetic_tasks.py
              task_audit.py
              learnability.py
              specialization.py
              runner.py
              report.py

            notebooks/
              kaggle-learnability.ipynb

            reports/
              published/

所有结果使用新的 Experiment ID 和 Run ID。

不得覆盖 R3m-E 结果。

## 15.3 数据边界

允许使用：

- 原始 R3h Train。
- 明确标记为 reused 的历史 Story Development。
- 新生成并冻结的 Copy Train/Dev。
- 新生成并冻结的 A1 Arithmetic Train/Dev。

禁止读取：

- R3m 原有 512 条 Main 候选。
- R3m 未注册的 Trigger/Pilot 候选。
- Confirmation。
- 其他 sealed 测试数据。

原 R3m 的 Scientific Registration 继续保持 `BLOCKED`。

## 15.4 CPU 与 GPU 门禁

先完成 CPU 测试：

- Arithmetic Fact Split。
- Copy 数据正确性。
- Answer Mask。
- 自回归评分。
- Checkpoint 一致性。
- RNG 恢复。
- 训练配对。
- Cell 干预。
- 专业特异性计算。
- 数据访问边界。

之后：

1. 运行 Tiny Overfit 工程预检。
2. 检查最小模型训练及恢复。
3. 独立审阅预检结果。
4. 工程门禁通过并获得运行授权后，执行完整三 Arm × 三 Seed Pilot。

不得自动执行后续两位数实验。

---

# 16. 报告与交付

最终交付：

- `EXPERIMENT_MANIFEST.json`
- `MODEL_LINEAGE.json`
- `DATA_AUDIT.json`
- `TINY_OVERFIT_REPORT.json`
- `COPY_CAPABILITY.json`
- `ARITHMETIC_CAPABILITY.json`
- `TRAINING_CURVES.json`
- `STORY_RETENTION.json`
- `CELL_EXPOSURE.json`
- `GRADIENT_CONFLICT.json`
- `CELL_INTERVENTION.json`
- `SPECIALIZATION_TRAJECTORY.json`
- `LEARNABILITY_DECISION.json`
- `COMPUTE_COST.json`
- `SCIENTIFIC-FINDINGS.md`

科学报告必须回答：

1. 模型能否完成少量算术事实的记忆？
2. 能否在未见事实中正确进行一位数加法？
3. Copy 对照是否成功？
4. Arithmetic 能力是否超出了数字输出模式学习？
5. 是否观察到 Cell 专业分化？
6. Cell 1 是否复现了 R3m-E 中的功能敏感性变化？
7. 专业分化是否随能力学习同步增强？
8. 下一步是否值得进入两位数加法研究？
9. 是否已经具备重新研究 Connection 重组的前提？

无论结果正负，都必须完整报告所有三个 Seed。

禁止只展示最佳 Checkpoint 或最佳 Cell。

---

# 17. 最终实施指令

请在 `ArcheLabs/mini-cells` 中建立独立的 R3m-F 探索性实验。

**第一目标：使当前六 Cell 语言模型在不修改 Connection 的情况下，真正学会简单加法。**

**第二目标：研究这种能力是否伴随真实、可干预的 Cell 专业分化。**

执行顺序：

1. 冻结协议、数据生成器和实验配置。
2. 验证 Copy/Arithmetic 的评价实现。
3. 运行 8 个训练事实的 Tiny Overfit Test。
4. 如果基本训练能力门禁通过，执行三个 Seed × 三个 Arm × 900 Step。
5. 评价 Arithmetic Exact Match、Story 保留与 Cell 任务特异性。
6. 重点研究 Cell 1，但必须完整报告六个 Cell。
7. 比较 Story-only、Story+Copy、Story+Arithmetic。
8. 使用合法替换干预检查真实功能贡献。
9. 分析能力形成与 Cell 分化的训练轨迹。
10. 生成完整科学报告并停止，不自动扩展任务难度或 GPU 预算。

保留原 R3m、R3m-E 的科学状态和结果，不修改历史数据注册，不访问 Confirmation。

本实验不以降低总体 NLL 为最终成功标准，而以**未见算术事实的实际正确率，以及伴随这种能力出现的可验证 Cell 功能分工**作为研究重点。

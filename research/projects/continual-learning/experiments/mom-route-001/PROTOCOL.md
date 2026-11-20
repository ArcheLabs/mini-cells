# MOM-ROUTE-001：Routed Memory Feasibility

**项目：** MiniCells
**实验类型：** 架构可行性验证
**状态：** PROPOSED
**模型：** MoM-Gated-DeltaNet-340M
**路由基线：** MA-001 Hybrid Raw
**训练：** 无
**GPU：** Kaggle T4（优先使用现有环境）

---

## 1. 实验目的

本实验只回答一个核心问题：

**我们能否通过外部 Addressing 将自然语言查询路由到一个或多个正确的逻辑 Memory Cell，再由冻结的 MoM 原生记忆机制利用这些内容完成回答？**

进一步验证一个关键假设：

**读取路由不必绝对准确。如果错误 Cell 被一并读取，MoM 是否能够识别并忽略无关或错误信息，而不显著影响最终输出？**

这是 MiniCells 当前架构可行性最重要的验证。

## 2. 架构约束

本实验正式采用以下架构：

```text
              User Input
                   |
                   v
          External Addressing
             (MA-001)
                   |
          Retrieve Top Candidates
                   |
             One / Multiple
             Logical Cells
                   |
                   v
             Memory Context
                   |
                   v
          MoM Native Memory
          (Frozen Backbone)
                   |
                   v
             User Query
                   |
                   v
             Final Output
```

### 2.1 明确取消 Titans

本实验不包含：

- Titans；
- Titans Fast Weights；
- Titans Outer Training；
- Titans Adapter；
- Titans Fusion；
- Titans Checkpoint；
- Titans Multi-Cell Runtime。

MA-001.4 作为历史工程成果保留，不再作为本实验的依赖。

### 2.2 两种 Router 必须区分

**External Addressing Router**

负责从自然语言查询中找到对应的外部记忆记录及逻辑 Cell。

这是 MA-001 已经研究的路由机制。

**MoM Native Router**

负责 MoM 内部 Token 到原生线性记忆状态的分配。

使用预训练权重，不修改、不训练，也不人为指定某个外部 Cell 对应哪个内部 Memory Bank。

两者不应混为一谈。

### 2.3 Memory Cell 的定义

本实验中的 Cell 是外部逻辑知识集合，不是 Titans 实例，也不要求与 MoM 内部的线性记忆状态一一对应。

写入时可以明确指定：

```text
Cell: MiniJAM
Cell: JamScript
Cell: MiniCells
```

读取时不向 Router 提供这些类别标签。

---

## 3. 实验核心假设

验证三个假设：

**H1：Addressing 可行**

自然语言查询能够找到一个或多个相关 Cell，不要求用户明确指定类别。

**H2：MoM 能够利用外部记忆**

冻结的 MoM 能够通过原生记忆状态吸收检索到的内容，并在后续查询中利用它们。

**H3：错误路由具有一定容错性**

即使读取了一个额外的错误 Cell，也不会必然导致最终答案错误。

其中 H2 和 H3 是本次最重要的研究对象。

---

## 4. 实验数据

使用三个逻辑 Cell：

| Cell | 内容 |
|---|---|
| MiniJAM | 网络架构、运行时、RPC |
| JamScript | 编译器、语言、Service Runtime |
| MiniCells | 模型、训练、数据集 |

每个 Cell 准备少量确定性事实，并为每个事实设置唯一的测试值。

使用合成标识符，例如：

```text
MiniJAM:
Test Gateway Alias = veka-27

JamScript:
Compiler Marker = sena-41

MiniCells:
Dataset Snapshot ID = voro-63
```

这些标识符应通过 Tokenizer 检查，尽量使用相同 Token 数量，避免答案长度影响评分。

数据分为：

| 查询类型 | 数量 | 目的 |
|---|---:|---|
| 单 Cell 查询 | 9 | 验证单个记忆寻址 |
| 多 Cell 查询 | 3 | 验证一次查询涉及多个 Cell |
| 弱匹配查询 | 2 | 验证同主题但缺少目标事实 |
| 无匹配查询 | 1 | 验证没有相关记忆的情况 |
| **总计** | **15** | |

查询不得直接包含 Cell ID 或类别名称。

多 Cell 查询必须确实需要两个 Cell 的事实才能完整回答。

弱匹配查询应与已有记录主题相近，但要求的具体关系或属性并不存在。

---

## 5. 路由实现

直接复用 MA-001 的冻结版本：

- Hybrid Raw；
- 原有 BM25；
- 原有 Dense E5；
- 原有融合规则；
- 不重新调整阈值；
- 不使用未经验证的 NLI 判定。

每条查询返回 Top 16 候选记录。

保留每条记录的：

- Record ID；
- Cell ID；
- 原始正文；
- 原始排序位置；
- 融合分数。

不要提前删除弱匹配记录。我们正是要验证 MoM 能否处理这些候选内容。

本轮不重新研究 MA-001 的索引算法。

---

## 6. 核心实验：四种记忆条件

每条有效查询使用以下四种条件。

### A. No Memory

不提供任何外部记忆。

建立基线，确认模型是否能够仅凭自身预测正确答案。

### B. Oracle Memory

只提供人工标注的正确记录。

这是理想寻址条件。

目的：验证 MoM 是否能够利用正确的外部记忆。

如果这一条件都不能通过，就没有必要继续研究路由。

### C. Actual Routed Memory

使用 MA-001 实际返回的 Top 16 候选记录。

不人工修正，不删除错误候选。

这是我们真正需要验证的架构路径。

### D. Fault-Injection Memory

在 Oracle Memory 的基础上，额外加入一个人工指定的错误记录。

错误记录优先选择：

- 同一主题；
- 相似术语；
- 不同事件；
- 不同主体；
- 或者具有相似描述但缺少所需关系的记录。

目的：验证 MoM 是否能够区分正确事实与错误干扰。

特别注意，不要只使用完全无关的随机记录。那种测试过于简单。

---

## 7. MoM 执行方式

使用冻结的 MoM-Gated-DeltaNet-340M。

不进行任何训练。

每个实验条件独立创建一个新的原生 Cache：

1. 将对应条件下的 Memory Context 输入 MoM。
2. 通过正常前向更新 MoM 原生递归状态。
3. 不保留外部记忆正文作为后续查询输入。
4. 使用相同的 Query 继续执行。
5. 评估正确答案与干扰答案的概率。

这里使用同一进程内的正常 Cache 连续执行。

**不进行磁盘 Checkpoint、跨进程恢复或缓存迁移测试。** 这些不是本次要解决的架构风险。

### 为什么不直接使用自由生成？

MoM 340M 是 Base 模型，不应要求它具备未经验证的指令跟随能力。

采用候选答案的 Log Probability 进行评价。

定义：

\[
M = \log P(A_{\text{correct}})
-\log P(A_{\text{distractor}})
\]

其中：

- \(M>0\)：模型更倾向于正确答案；
- \(M<0\)：模型更倾向于错误答案。

答案分数按 Token 数量归一化。

所有条件使用完全相同的提示模板、Query 和候选答案。

---

## 8. 评价指标

只记录以下指标：

| 指标 | 说明 |
|---|---|
| Oracle Accuracy | 正确记忆条件下的正确答案比例 |
| Routed Accuracy | 实际路由条件下的正确答案比例 |
| Route Coverage | 正确记录是否进入 Top 16 |
| Multi-Cell Coverage | 多 Cell 查询是否找齐全部必要 Cell |
| Wrong-Cell Flip Rate | 加入错误 Cell 后，正确答案变为错误答案的比例 |
| Memory Gain | 有记忆与无记忆之间的答案分数变化 |
| Weak-Match False Route | 弱匹配查询是否带入错误记忆 |

特别关注：

\[
\text{WrongCellFlipRate}
=
\frac{\text{加入错误 Cell 后答案由正确变错误的次数}}
{\text{Oracle 条件下正确的次数}}
\]

这是本实验最重要的容错指标之一。

---

## 9. GO / NO-GO

本轮是小规模可行性实验，不代表统计意义上的泛化能力。

### GO

同时满足：

1. Oracle 条件下，MoM 能够利用正确记忆，至少 10/12 个有效查询得到正向答案分差。
2. 实际路由条件下，至少 9/12 个有效查询保持正确答案优势。
3. 加入一个错误 Cell 后，答案翻转不超过 2/12。
4. 三个多 Cell 查询中，至少两个找齐全部必要 Cell。

这意味着：

- 外部寻址可以与 MoM 原生记忆组合；
- 不要求路由绝对完美；
- 少量错误候选不会必然破坏最终输出。

### PARTIAL PASS

Oracle 条件通过，但实际路由或错误 Cell 容错未通过。

说明 MoM 能利用正确记忆，但当前 Addressing 仍是瓶颈。

根据失败类型决定是否需要 MA-003，而不是直接扩大模型或开始训练。

### NO-GO

Oracle 条件也无法稳定利用正确记忆。

说明在当前冻结模型和接入方式下，MoM 原生记忆尚未证明能够承担预期任务。

停止本轮集成，不启动 Titans，也不立即增加训练预算。

### BLOCKED

模型、依赖或环境问题导致没有获得有效实验结果。

---

## 10. 执行要求

严格控制实验成本：

1. 复用 MA-001 的代码和冻结索引实现。
2. 只创建少量测试事实和查询。
3. 只使用 MoM 340M。
4. 不训练模型。
5. 不运行 Titans。
6. 不执行额外的缓存恢复测试。
7. 不重新调参。
8. 一次 Kaggle 运行完成全部实验条件。

失败时必须保留完整的逐查询结果，不允许因为某个断言失败而提前终止整个实验。

---

## 11. 最终决策

本实验通过后，下一步不是训练更大的 MoM，也不是重新引入 Titans。

而是根据实验结果确定：

1. 外部 Addressing 是否足以提供可用的候选记忆；
2. MoM 是否能够利用这些候选内容；
3. 错误 Cell 是否会显著污染 MoM 的原生记忆状态；
4. 是否需要进一步改进 MA-003 的寻址与证据过滤机制。

**本实验的最终目的，是证明 MiniCells 可以使用外部显式寻址 + MoM 原生线性记忆构成可运行的知识记忆架构，而不依赖 Titans 或新的大规模外层训练。**
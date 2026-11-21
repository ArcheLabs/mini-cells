# MoM 双层记忆架构可行性验证

**项目：** MiniCells
**目标仓库：** `ArcheLabs/mini-cells`
**模型：** MoM-Gated-DeltaNet-340M
**实验阶段：** MOM-ROUTE-002、MOM-STATE-001
**训练：** 不训练，冻结全部模型权重
**执行顺序：** 先完成 MOM-ROUTE-002，再执行 MOM-STATE-001

---

# 1. 研究目标

本次研究不再验证 MoM 是否能够利用外部提供的正确文本。

MOM-ROUTE-001 已经初步证明了这一点。

本次需要验证的是：

**能否将 MA-001 的外部多 Cell 记忆寻址机制，与 MoM 的原生递归记忆机制组合，形成一个能够独立寻址、持续写入、查询和恢复的记忆系统？**

我们需要验证两个独立但相互关联的能力：

1. 外部 Memory Cells 能否通过 MA-001 进行显式写入、自然语言寻址和多 Cell 读取，并将实际读取结果交给 MoM。
2. MoM 能否利用自己的原生 Router 和递归状态，持续吸收这些知识，在不重复提供原始知识文本的情况下完成后续查询。

本次不研究模型规模扩展，不进行模型训练，也不重新引入 Titans。

# 2. 必须保持的架构边界

整个系统由两套独立的路由机制组成。

## 2.1 外部 Memory Cells

外部 Cell 是 MiniCells 定义的逻辑知识单元。

例如：

- MiniJAM
- JamScript
- MiniCells
- Locus
- AI Research

外部 Cell 负责：

- 知识归属。
- 记录写入。
- Cell 身份和版本。
- 自然语言查询寻址。
- 多 Cell 选择。
- 原始知识读取。

外部路由完全复用 MA-001。

写入路由：

`Explicit Target > Session Binding > Inbox`

读取路由：

`Query → Hybrid Retrieval → Cell Aggregation → Threshold → Selected Cell IDs`

## 2.2 MoM 原生记忆

MoM 使用自身的内部路由机制。

- MoM Native Router：决定 Token 如何分配给内部 Memory Experts。
- Recurrent Memory State：保存递归记忆状态。
- `past_key_values`：承载模型运行所需的可变状态。
- MoM Backbone：执行后续预测。

不修改 MoM 原生 Router，不人为指定外部 Cell 与内部 Memory Expert 的对应关系。

**外部 Cell 和 MoM 内部 Memory Expert 是两个不同的抽象层级，不要求一一映射。**

## 2.3 完整架构

```text
                    User Input
                         |
                         v
               MA-001 External Router
                         |
                  Selected Cell IDs
                         |
                         v
                External Cell Storage
                  /      |      \
                 A       B       C
                  \      |      /
                         |
                   Retrieved Records
                         |
                         v
                MoM Native Memory
                         |
                 MoM Native Router
                         |
                Recurrent Memory State
                         |
                         v
                    Query
                         |
                         v
                     Answer
```

本次必须分别记录两套 Router 的结果，不得把它们合并为一个路由指标。

# 3. 现有代码与实验基线

先检查并复用以下现有实现：

| 文件 | 用途 |
|---|---|
| `experiments/ma-001/PROTOCOL.md` | 外部 Cell 写入和读取规范 |
| `experiments/ma-001/addressing.py` | Registry、Ranker |
| `experiments/ma-001/scripts/run_experiment.py` | MA-001 实验与冻结阈值 |
| `experiments/ma-001/artifacts/enriched-cpu/frozen.json` | 冻结路由阈值 |
| `experiments/mom-route-001/run.py` | MoM 集成与答案评分 |
| `experiments/mom-route-001/PROTOCOL.md` | MOM-ROUTE-001 原始协议 |
| `experiments/mom-int-001/preflight.py` | MoM 原生 Cache 保存和恢复验证 |

特别注意：

MOM-ROUTE-001 的 Top-16 Record Memory 是原始实验协议定义的实验条件，并非 MA-001 真正的 Cell Router。

新实验必须修正这一点。

不要修改或覆盖 MOM-ROUTE-001 的历史结果。

建议创建两个独立目录：

```text
research/projects/continual-learning/experiments/

    mom-route-002/
        PROTOCOL.md
        run.py
        fixtures/
        tests/
        artifacts/

    mom-state-001/
        PROTOCOL.md
        run.py
        fixtures/
        tests/
        artifacts/
```

优先复用现有模块，不重复实现 Ranker、MoM Adapter 和 Cache 序列化。

---

# 第一阶段：MOM-ROUTE-002

# 4. 实验目的

验证：

**MA-001 是否能够真正选择一个或多个外部 Cell，并且只有被选择的 Cell 内容进入 MoM 原生记忆状态。**

本实验重点是外部路由和 MoM 原生记忆的端到端组合。

不再使用全局 Top-16 记录作为实际 Routed Memory。

## 4.1 外部写入

复用 MA-001 Registry。

使用确定性的显式目标写入：

```python
registry.write(
    event_id,
    raw,
    descriptor,
    explicit=["projects/minijam"]
)
```

写入目标通过 MA-001 Write Router 确定。

本实验使用三个主要 Cell：

| Cell | 内容 |
|---|---|
| MiniJAM | Gateway、RPC、网络运行时 |
| JamScript | Compiler、Service Runtime |
| MiniCells | Dataset、Training、Model |

每条记录必须保留：

- Record ID
- Cell ID
- Raw Content
- Summary
- Entities
- Topics
- Aliases
- Version

不要将每条记录临时创建为一个独立 Cell。

## 4.2 外部读取

直接使用 MA-001 的原有实现：

```python
ranker = Ranker(records, raw=True)

scores = ranker.scores(
    query,
    "E3-Raw",
    dense_scores
)

selected_cells = ranker.route(
    scores,
    frozen_threshold
)
```

使用现有冻结阈值：

```text
E3-Raw = 0.031746031746031744
```

以仓库中的 `frozen.json` 为权威来源，并记录其 SHA-256。

不得：

- 重新训练 Embedding。
- 修改 RRF。
- 重新拟合 Threshold。
- 通过修改 Cell ID 绕过 Cell 聚合。
- 使用全局 Top-16 代替 Cell Router。

MA-001 的 `Ranker.scores()` 已经按照 Cell ID 聚合分数，使用同一个 Cell 内最高记录分数。

`Ranker.route()` 负责返回最多三个满足阈值的 Cell。

这两个函数必须构成实际读取路径。

## 4.3 实际读取

外部 Router 返回：

```python
selected_cells = ["minijam", "jamscript"]
```

之后：

```python
selected_records = [
    record
    for record in records
    if record["cell_id"] in selected_cells
]
```

本实验规模较小，读取选中 Cell 内的全部有效记录。

不允许重新从整个知识库进行全局 Top-K 检索。

必须记录：

```json
{
  "query": "...",
  "selected_cell_ids": [
    "minijam",
    "jamscript"
  ],
  "selected_record_ids": [],
  "excluded_cell_ids": [
    "minicells"
  ]
}
```

这样才能证明外部 Cell 的选择真正控制了知识读取范围。

## 4.4 MoM 写入和查询

继续使用 MOM-ROUTE-001 的真实模型加载方式。

每个实验条件独立创建新的 Cache：

```python
_, cache = forward(memory_tokens)

hidden, _ = forward(
    query_tokens,
    cache
)
```

必须保证：

- Memory Context 先输入。
- MoM 原生 Router 正常运行。
- 原生 Recurrent State 被更新。
- Query 不重新提供原始知识正文。
- Query 继承之前的 Cache。
- 所有模型权重冻结。

Memory Context 应保留 Cell 来源：

```text
[Cell: minijam]
[Record: record_001]
The current Gateway Alias is veka-27.

[Cell: jamscript]
[Record: record_004]
The current Compiler Marker is sena-41.
```

Oracle 和 Routed 两组必须使用相同的格式。

---

# 5. 实验条件

每条有效查询至少执行以下条件：

| Arm | 输入 |
|---|---|
| No Memory | 空记忆 |
| Oracle | 只包含正确证据记录 |
| Routed | MA-001 实际选中的 Cell 中的记录 |
| Legacy Top-16 | 旧版全局 Top-16，仅作历史对照 |

另外，对原来五个答案翻转案例执行冲突顺序实验：

| Arm | 写入顺序 |
|---|---|
| Correct Only | 只有正确记录 |
| Wrong Only | 只有错误记录 |
| Correct → Wrong | 正确记录先写入 |
| Wrong → Correct | 错误记录先写入 |

五组冲突保持原来的事实和答案，不重新设计成更容易的测试。

目的：区分内容冲突和写入顺序的影响。

Legacy Top-16 不参与 Routed PASS 判定。

---

# 6. 测试数据和查询

尽可能复用 MOM-ROUTE-001 的九组 FACTS、18 条记录和 15 条查询。

保持原有数据及答案，不通过修改数据迎合新路由结果。

查询分组：

| 类型 | 数量 |
|---|---:|
| 单 Cell 查询 | 9 |
| 双 Cell 查询 | 3 |
| 信息不完整查询 | 2 |
| 完全无匹配查询 | 1 |
| 合计 | 15 |

其中三个双 Cell 查询必须确实需要两个不同 Cell 的知识。

必须保留以下负向查询：

```text
Who approved the current compiler marker?

What is the password for the dataset snapshot?

What is the boiling point of mercury?
```

特别注意：

负向查询必须全部纳入路由统计。

此前 `weak_threshold_routes` 只统计了两条弱匹配查询，没有统计 Mercury 查询。

本次必须单独记录：

- Weak Match False Route
- No Match False Route
- Evidence Missing
- Selected Cell Count

不要将 Cell 相似度超过阈值解释为记录中确实存在答案。

本阶段不实现复杂 NLI 或新的 Evidence Classifier。先测量路由本身的能力及其边界。

负向查询没有正确答案时，不进行虚构的答案准确率评分。

---

# 7. MoM 答案评价

继续使用 MOM-ROUTE-001 的候选答案 Log Probability 方法。

定义：

```text
Margin =
    Correct Answer Mean Log Probability
    -
    Distractor Mean Log Probability
```

判断：

- Margin > 0：正确答案具有预测优势。
- Margin < 0：错误答案具有预测优势。

保持相同的 Tokenizer、Prompt Template、Query 和答案候选。

所有评分均记录逐查询结果。

不要只保留最终平均分。

---

# 8. MOM-ROUTE-002 指标

必须输出：

| 指标 | 含义 |
|---|---|
| Oracle Accuracy | 正确记忆条件下的预测优势 |
| Routed Accuracy | 真实 Cell 路由后的预测优势 |
| Cell Recall | 必要 Cell 是否被找到 |
| Cell Precision | 选择的 Cell 中相关 Cell 的比例 |
| Multi-Cell Exact Coverage | 多 Cell 查询是否找到全部必要 Cell |
| Retrieval Leakage | 是否读取未选择 Cell 的记录 |
| Legacy Top-16 Accuracy | 旧实现对照 |
| Wrong-Cell Flip Rate | 错误事实导致的答案翻转 |
| Weak-Match False Route | 弱匹配错误路由 |
| No-Match False Route | 完全无匹配错误路由 |
| Memory Gain | 相对 No Memory 的分数提升 |

所有指标必须按 Query 类型分别报告。

## 8.1 验收标准

**GO：**

1. Oracle 至少 10/12。
2. Routed 至少 9/12。
3. 三个 Multi-Cell 查询至少两个找到全部必要 Cell。
4. 五组冲突案例中，错误写入导致的翻转不超过两组。
5. 没有发生未选中 Cell 的记录读取。

**PARTIAL_PASS：**

Oracle 通过，但 Routed 或多 Cell 覆盖率未通过。

这说明 MoM 可以利用正确记忆，但外部寻址仍是瓶颈。

**NO_GO：**

Oracle 低于 10/12。

当前冻结 MoM 在本实验条件下未能稳定利用正确记忆。

**BLOCKED：**

模型、依赖、环境或数据错误导致没有获得有效结果。

负向查询的错误路由必须单独报告。由于样本很少，不得仅凭 1 条 Mercury 查询声称已经证明通用拒绝能力。

---

# 第二阶段：MOM-STATE-001

# 9. 实验目的

第一阶段验证的是：

```text
External Cell
    → MA-001 Router
    → MoM Cache
    → Query
```

但每次查询都重新创建 Cache。

第二阶段要验证：

**MoM 能否在同一个原生递归状态中持续接收来自多个外部 Cell 的知识，并在后续多轮查询中使用这些知识，而不必重新提供原始知识文本？**

本阶段使用一个共享的 MoM Native Cache。

不将每个外部 Cell 强行映射到某个 MoM Memory Expert。

外部 Cell 仍然独立保存在 MA-001 Registry 中。

MoM 的内部 Router 自主决定 Token 的内部记忆分配。

这是一个共享状态的持续记忆实验，不应被解释为已经实现独立的 MoM Cell 状态。

---

# 10. 顺序写入实验

使用三个外部 Cell：

```text
Cell A: MiniJAM
Cell B: JamScript
Cell C: MiniCells
```

通过 MA-001 Write Router 将记录写入对应 Cell。

之后，将这些记录依次输入同一个 MoM Cache。

## 10.1 写入顺序

第一次：

```text
Write Cell A
    |
    v
Update MoM Cache
    |
    v
Query A
```

第二次：

```text
Continue Existing Cache
    |
    v
Write Cell B
    |
    v
Update MoM Cache
    |
    v
Query A
Query B
```

第三次：

```text
Continue Existing Cache
    |
    v
Write Cell C
    |
    v
Update MoM Cache
    |
    v
Query A
Query B
Query C
Query A + B
Query A + B + C
```

必须保证：

- 不重新初始化 MoM Cache。
- 不重新输入 A 的原始知识。
- 不重新输入 B 的原始知识。
- 每次写入都继承上一次返回的 Cache。
- 保留所有实际写入顺序和状态版本。

## 10.2 写入格式

每次写入都带有明确的外部来源信息：

```text
[Memory Write]

[Cell ID: minijam]
[Record ID: record_001]
[Version: 1]

Gateway Alias = veka-27.
```

Cell ID 是知识来源的标识，不是内部 Memory Expert ID。

不要人工指定 MoM 内部 Memory Expert。

---

# 11. 外部路由与持久化状态的结合

当所有 Cell 已经写入共享 MoM 状态后，提出新 Query。

先通过 MA-001 External Router 获得：

```text
Selected Cells:
- minijam
- jamscript
```

然后在 Query 中提供所选 Cell 的标识：

```text
[Selected Cells: minijam, jamscript]

Question:
What are the current Gateway Alias and Compiler Marker?
```

这里要特别注意：

共享 MoM Cache 已经包含多个 Cell 的历史知识。

外部 Router 返回的 Cell ID 是查询时提供给 MoM 的外部寻址信息，不是对内部 Cache 的硬性屏蔽。

本阶段要验证：

**MoM 能否结合外部选中的 Cell 身份，利用其内部原生 Router，从共享递归状态中正确使用对应知识。**

如果出现跨 Cell 知识混淆，必须记录为实际架构限制。

不能通过在 Query 中重新拼接原始事实来掩盖失败。

---

# 12. 必须执行的持续记忆测试

## 测试 A：旧知识保留

写入 A 后，查询 A。

再写入 B、C。

不重新提供 A 的原始记录，再次查询 A。

要求：

A 的正确答案优势不能因为 B、C 的写入而消失。

## 测试 B：新知识吸收

写入 B、C 后，分别查询 B、C。

要求：

新知识可以通过已有 Cache 使用。

## 测试 C：跨 Cell 组合

分别查询：

- A + B
- A + C
- B + C
- A + B + C

要求：

模型能够利用多个 Cell 的知识完成同一次回答评分。

## 测试 D：跨 Cell 冲突

在两个不同 Cell 中写入同一属性的不同值：

```text
Cell A:
Gateway Alias = veka-27

Cell B:
Gateway Alias = nalo-27
```

然后使用 MA-001 分别选择 A 和 B：

```text
Selected Cell: A
Query: What is the Gateway Alias?
Expected: veka-27
```

```text
Selected Cell: B
Query: What is the Gateway Alias?
Expected: nalo-27
```

再查询两个 Cell：

```text
Selected Cells: A, B
Query: What are the two Gateway Aliases?
Expected: veka-27, nalo-27
```

这是本阶段的重要测试。

它可以验证外部寻址信息是否足以帮助 MoM 在共享状态中区分不同知识来源。

如果无法做到，必须报告共享状态的寻址限制。

## 测试 E：持续干扰

在写入 A、B、C 后，执行一组预先固定的无关知识写入。

建议增加 10 条与原始事实无关的记录。

然后重新查询 A、B、C。

记录：

- 旧知识召回率。
- 正确答案 Margin。
- 新知识召回率。
- 冲突翻转。
- 状态大小。
- 写入耗时。

## 测试 F：知识更新

对 Cell A 的一个属性写入新版本：

```text
Version 1:
Gateway Alias = veka-27

Version 2:
Gateway Alias = nalo-27
```

分别测试：

- 当前版本查询。
- 历史版本查询。
- 版本更新后的旧知识干扰。

必须在写入文本中明确提供版本和有效性语义。

本实验验证的是 MoM 对结构化版本信息的利用，不代表它已经拥有自动事实仲裁能力。

---

# 13. Query 必须是非破坏性的

这是第二阶段的重要工程要求。

MoM 在执行 Query 时可能继续更新递归状态。

因此，不能直接使用查询后的 Cache 覆盖持久记忆状态。

建议采用：

```python
persistent_cache = current_cache

query_cache = clone_cache(
    persistent_cache
)

result = forward(
    query_tokens,
    query_cache
)

# 不提交 query_cache
# persistent_cache 保持不变
```

要求：

1. 写入操作更新持久 Cache。
2. 查询操作使用 Cache 副本。
3. 查询结束后丢弃查询副本。
4. 查询前后持久 Cache 的状态 Hash 必须一致。

如果当前 MoM Cache API 会原地修改状态，必须先进行完整 Clone。

不能仅复制 Python 对象引用。

---

# 14. 跨进程状态保存与恢复

复用：

`experiments/mom-int-001/preflight.py`

其中已有真实 MoM Cache 的保存、恢复和连续执行对照。

本阶段将它扩展到真实知识写入场景。

保存内容必须包括：

- 每一层的 recurrent_state。
- 每一层的 conv_state。
- 所有其他可变 Cache 字段。
- Cache 的 Token Position / seen_tokens。
- MoM 模型版本。
- Tokenizer 版本。
- 状态 Schema Version。
- 已写入的外部 Cell 版本和记录清单。
- 所有状态文件的 SHA-256。

禁止只保存 fast weights 或部分 Recurrent State。

## 14.1 执行流程

Process A：

1. 加载冻结 MoM 340M。
2. 初始化空 Cache。
3. 写入 A。
4. 写入 B。
5. 写入 C。
6. 执行全部查询。
7. 保存完整 Cache。
8. 保存状态 Hash。
9. 结束进程。

Process B：

1. 启动新的 Python 进程。
2. 加载相同模型版本。
3. 恢复完整 Cache。
4. 校验全部状态。
5. 不重新执行任何原始知识写入。
6. 执行相同 Query。
7. 比较恢复前后的答案分数。
8. 再执行一次新的知识写入。
9. 验证新旧知识是否都能继续使用。

要求：

- 不依赖原进程对象。
- 不重新构造原始 Memory Context。
- 不重新执行历史写入。
- 不使用 Mock Cache。
- 必须是实际 MoM 模型状态。

恢复后的 Cache Tensor 应进行逐层、逐字段比较。

答案 Log Probability 应与连续运行结果进行比较。若存在数值非确定性，必须明确记录误差和容差，不得静默放宽。

---

# 15. MOM-STATE-001 指标

| 指标 | 验证内容 |
|---|---|
| Sequential Write Success | 多次写入是否连续成功 |
| Old Memory Retention | 新知识写入后旧知识是否保留 |
| New Memory Recall | 新写入知识是否可用 |
| Multi-Cell Recall | 多 Cell 组合查询 |
| Cross-Cell Isolation | 相似事实是否发生混淆 |
| Update Accuracy | 新旧版本是否能够区分 |
| Interference Resistance | 无关写入后的知识保持 |
| Read State Integrity | 查询是否改变持久状态 |
| Restore State Integrity | 恢复前后状态是否一致 |
| Post-Restore Continuation | 恢复后能否继续写入和查询 |

## 15.1 验收标准

**PASS：**

1. A、B、C 顺序写入均成功。
2. B、C 写入后，A 的原有知识仍可召回。
3. 三个 Cell 的新知识均能够召回。
4. 多 Cell 查询至少达到与 MOM-ROUTE-002 Oracle 基线相当的正确答案优势。
5. 跨 Cell 冲突测试能够区分不同 Cell 的事实。
6. 查询不改变持久 Cache。
7. 全部状态可以在新进程中恢复。
8. 恢复后答案分数保持在预定数值容差内。
9. 恢复后可以继续写入新知识并使用旧知识。

**PARTIAL_PASS：**

顺序写入和状态恢复成功，但出现旧知识丢失、跨 Cell 混淆或多 Cell 组合失败。

必须明确定位是：

- 内部记忆容量问题。
- MoM 原生路由问题。
- 共享状态干扰问题。
- 外部 Cell 寻址信息不足。
- Cache 保存和恢复问题。

**NO_GO：**

MoM 无法在当前架构下可靠地持续利用已经写入的知识。

**BLOCKED：**

环境或模型问题导致没有完成有效测试。

注意：本实验只能证明受控实验长度内的持续状态能力，不能直接声称已经证明无限期长期记忆。

---

# 16. 两阶段之间的 Gate

执行顺序：

```text
MOM-ROUTE-002
       |
       v
Oracle Memory PASS?
       |
   +---+---+
   |       |
  YES      NO
   |       |
   v       v
MOM-STATE  STOP
-001       Investigate
   |
   v
Sequential Write
   |
   v
Cross-Cell Recall
   |
   v
Fresh Process Restore
   |
   v
Final Decision
```

MOM-ROUTE-002 的 Oracle 未通过时，不应继续把外部路由问题与模型记忆问题混在一起。

如果 Oracle 通过但 Routed 未通过，可以继续使用 Oracle 条件执行 MOM-STATE-001，以独立判断 MoM 持续记忆能力；但不得宣称外部路由集成已经通过。

---

# 17. 资源与工程约束

两个实验共同遵守：

1. 只使用 MoM 340M。
2. 模型权重完全冻结。
3. 不训练 Backbone。
4. 不训练 MoM Router。
5. 不训练 MA-001 Router。
6. 不重新引入 Titans。
7. 不扩大到 1.3B。
8. 不增加新的大型模型。
9. 不修改原始 MA-001 冻结阈值。
10. 不修改 MOM-ROUTE-001 历史结果。
11. 复用 MOM-INT-001 的模型加载和 Cache Adapter。
12. 保留所有逐查询、逐写入和状态级证据。

如果现有接口不支持某项操作，应明确报告缺失接口及原因，不允许使用 Mock 或文本拼接伪装成功。

---

# 18. Codex 最终交付要求

Codex 应当完成：

**MOM-ROUTE-002**

- 完整实验协议。
- 真实 Cell Router 接入。
- 正确的 Cell-level Retrieval。
- 真实 MoM Native Cache 写入与读取。
- 逐查询结果。
- 路由指标。
- 冲突写入顺序对照。
- GO / PARTIAL_PASS / NO_GO / BLOCKED 决策。

**MOM-STATE-001**

- 顺序写入实现。
- 持久 Cache 管理。
- 非破坏性查询。
- 多 Cell 冲突与组合实验。
- Cache 保存和恢复。
- Fresh Process 验证。
- 恢复后继续写入。
- 完整状态 Hash 和逐查询报告。

统一生成：

```text
artifacts/
    manifest.json
    environment.json
    model_manifest.json

    route_metrics.json
    route_queries.jsonl

    state_write_trace.jsonl
    state_query_results.jsonl
    state_hashes.json
    restore_comparison.json

    decision.json
    RESULTS.md
```

每个实验使用独立的输出目录，记录：

- Git Commit SHA。
- MoM 源码 Revision。
- Model Weight Revision。
- Tokenizer Revision。
- E5 Revision。
- 依赖版本。
- 数据集 Hash。
- 冻结阈值 Hash。
- GPU 型号。
- 执行时间。

Codex 必须先运行全部单元测试和环境检查，再启动 GPU 实验。

不要因为某一条 Query 失败而提前终止整个实验。必须尽可能保留全部有效结果。

---

# 19. 最终研究问题

完成这两个实验后，我们应该能够回答：

1. MA-001 能否真正寻址多个外部知识 Cell？
2. 选中的 Cell 是否能够实际控制进入 MoM 的知识范围？
3. MoM 原生 Router 是否能够利用这些知识？
4. 多个外部 Cell 的知识能否共同参与一次 MoM 推理？
5. MoM 能否在同一个原生递归状态中持续接收新知识？
6. 新知识是否会破坏旧知识？
7. 外部 Cell 的身份能否帮助 MoM 区分相互冲突的事实？
8. 保存并恢复原生状态后，能否继续使用此前获得的知识？

本次研究的最终目标：

**验证 MA-001 External Addressing + MoM Native Memory 能否构成一个可寻址、可持续写入、可查询、可恢复的双层记忆系统。**

不以模型规模或训练量作为成功标准。

只有当这套架构在冻结的 MoM 340M 上通过上述实验，我们才有依据继续研究更大规模的知识容量、复杂推理和长期持续学习。
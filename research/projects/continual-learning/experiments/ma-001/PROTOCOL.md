# MiniCells Memory Addressing Feasibility Experiment

**实验编号：** MA-001
**实验名称：** Memory Address Index & Semantic Routing
**版本：** v0.1
**项目：** MiniCells
**目标仓库：** `ArcheLabs/mini-cells`
**实验类型：** 系统架构可行性实验
**运行环境：** CPU 优先，无需 GPU
**状态：** READY FOR IMPLEMENTATION

---

# 1. 实验目的

本实验验证 MiniCells 的独立神经记忆寻址机制。

我们计划将多个独立 Titans Cell 组织成一个可以跨 Session 使用的记忆系统。

系统必须满足：

1. 写入时，根据显性的项目、专业领域等类别，确定信息应该进入哪些 Cell。
2. 写入后，为每条记忆建立可检索的 Address Descriptor。
3. 读取时，用户只需要提出自然语言问题，不需要指定项目、专业或 Cell ID。
4. 系统通过语义寻址，自动找到一个或多个相关 Cell。
5. 一个 Query 可以同时命中多个不同 Session 创建的 Cell。
6. Cell 的身份独立于 Session，不能因为 Session 结束而丢失地址关系。

本实验只验证外部 Memory Addressing，不验证 Titans 的实际记忆学习能力。

**核心假设：**

通过显性分类写入、事件级地址描述、混合语义检索和 Cell ID 聚合，可以实现跨 Session 的多 Cell 自主寻址。

# 2. 实验范围与排除项

## 2.1 本次必须实现

- Cell Registry
- Category Registry
- Session Binding
- Category-based Write Router
- Address Descriptor
- Address Index
- Semantic Read Router
- Multi-Cell Candidate Selection
- 跨 Session 寻址
- 地址索引持久化
- 基础版本一致性检查

## 2.2 本次不实现

以下内容明确排除：

- Titans 训练或模型架构修改。
- MoM-GatedDeltaNet-340M / 1.3B。
- Qwen Backbone 加载。
- Titans Fast Weights。
- MoM 内部 Router。
- 神经记忆内容的实际 Recall。
- Memory Fusion 的神经网络实现。
- Router 的机器学习训练。
- Web UI 或在线服务。

使用 Mock Cell 模拟独立记忆对象。

Mock Cell 只需要拥有 Cell ID、类别、版本和状态哈希，不需要真正的神经网络参数。

这样可以隔离 Addressing 问题，避免将记忆能力、模型能力和路由能力混在一起。

特别注意：

Address Descriptor 不是完整知识副本，也不是 RAG 的答案来源。

它只负责描述某条记忆可能涉及什么内容，以及应该访问哪个 Cell。

实际知识仍然应该由 Titans 提供。

# 3. 系统架构

实现以下结构：

```text
                     Natural Language Query
                                |
                                v
                       Query Encoder
                                |
                                v
                    Memory Address Index
                                |
                  +-------------+-------------+
                  |             |             |
                  v             v             v
               Address A     Address B     Address C
                  |             |             |
                  v             v             v
               Cell A        Cell B        Cell C
                  |             |             |
                  +-------------+-------------+
                                |
                                v
                       Candidate Cell IDs
```

写入流程：

```text
                  New Information
                         |
                         v
                  Category Router
                         |
              +----------+----------+
              |                     |
              v                     v
          Target Cell A          Target Cell B
              |                     |
              v                     v
         Mock Cell Update      Mock Cell Update
              |                     |
              +----------+----------+
                         |
                         v
                  Address Builder
                         |
                         v
                   Address Index
```

注意：写入路由和读取路由是两个完全不同的机制。

- Write Router：显性分类驱动。
- Read Router：自然语言语义驱动。

读取不得要求用户指定 Category。

# 4. Cell Registry

建立统一的 Cell Registry。

每个 Cell 至少具有以下字段：

```json
{
  "cell_id": "cell_minijam",
  "category_id": "projects/minijam",
  "description": "MiniJAM architecture, development and decisions",
  "created_by_session": "session_a",
  "current_version": 1,
  "state_hash": "...",
  "status": "active",
  "created_at": "...",
  "updated_at": "..."
}
```

要求：

1. Cell ID 全局唯一。
2. Cell ID 不依赖 Session ID。
3. 一个 Category 可以映射到一个或多个 Cell。
4. Cell 可以被不同 Session 加载。
5. Session 结束不能删除 Cell。
6. Registry 必须持久化。
7. 删除 Cell 必须有明确的 Tombstone 状态，不能让旧地址继续指向已删除对象。

本次实验初始采用一个 Category 对应一个主要 Cell，以降低复杂度。

数据结构必须允许以后扩展为一个 Category 对应多个 Cell。

# 5. Write Router

写入路由必须是确定性的。

优先级：

1. 用户明确指定的目标 Category / Cell。
2. 当前 Session 的 Category Binding。
3. 如果两者都不存在，进入 Inbox，不进行猜测。

规则：

```python
def resolve_write_targets(
    explicit_targets,
    session_binding,
):
    if explicit_targets:
        return resolve_categories(explicit_targets)

    if session_binding:
        return resolve_categories(session_binding)

    return INBOX
```

禁止使用 Embedding 相似度自动决定知识归属。

本次不实现 LLM 分类器。

## 5.1 多类别写入

例如一条信息同时涉及 MiniJAM 和 AI Research：

```json
{
  "event_id": "event_001",
  "target_categories": [
    "projects/minijam",
    "domains/ai_research"
  ]
}
```

应当分别更新两个 Cell。

两条 Address Record 使用同一个 Canonical Event ID，但具有不同的 Cell ID。

## 5.2 写入验收

必须覆盖：

- Explicit Target。
- Session Default。
- Explicit Target Override。
- Multi-Category Write。
- Unknown Category。
- No Session Binding。
- Inbox Fallback。

所有确定性路由测试必须 100% 通过。

# 6. Address Descriptor

这是本次实验的核心数据结构。

每次向 Cell 写入信息，都应生成一个 Address Descriptor。

建议字段：

```json
{
  "address_id": "addr_001",
  "canonical_event_id": "event_001",
  "cell_id": "cell_minijam",
  "category_id": "projects/minijam",

  "summary": "MiniJAM selected GPU-PVM as a major research direction.",

  "entities": [
    "MiniJAM",
    "GPU-PVM"
  ],

  "topics": [
    "GPU computation",
    "project roadmap",
    "architecture decision"
  ],

  "aliases": [
    "GPU parallel computation",
    "GPU execution"
  ],

  "cell_version_at_write": 1,
  "descriptor_schema_version": 1,
  "active": true
}
```

重要要求：

- Descriptor 必须足够短。
- 不保存完整原始对话。
- 不保存完整答案。
- 必须包含 Cell ID。
- 必须记录创建时的 Cell Version。
- 支持一个 Event 对应多个 Cell。
- 支持未来更新、失效和重建。

本次使用人工编写的 Gold Descriptors，隔离检索算法的能力。

同时保留原始 Event Text，供 Raw Text 检索对照使用。

本次不引入 LLM 自动摘要，以免将摘要生成质量和寻址质量混为一谈。

# 7. Read Router

这是本次最重要的实现。

用户输入自然语言 Query。

系统必须自动找到相关 Cell，不允许用户指定 Category。

## 7.1 检索方式

采用 Hybrid Retrieval，包含三个检索通道。

**A. Dense Semantic Retrieval**

使用多语言 Embedding。

优先复用仓库已有的 Embedding 实现；如果没有，使用：

`intfloat/multilingual-e5-small`

要求：

- 固定模型 Revision。
- Query 使用 `query:` 前缀。
- Descriptor 使用 `passage:` 前缀。
- 使用归一化向量。
- 使用 Cosine Similarity。
- Embedding 结果持久化，避免每次重新计算。

**B. Lexical Retrieval**

使用 BM25 或现有仓库中等价的关键词检索实现。

索引字段：

- Summary
- Entities
- Topics
- Aliases

Category ID 不能作为默认搜索字段。

**C. Entity Matching**

提取 Query 中的实体，与 Descriptor 中的 Entities 和 Aliases 进行匹配。

支持预定义别名。

例如：

- GPU PVM / GPU-PVM
- Neural Memory / 神经记忆
- JAM / JAM Protocol

## 7.2 Hybrid Ranking

使用 Reciprocal Rank Fusion：

\[
RRF(d)=
\sum_j \frac{w_j}{k+rank_j(d)}
\]

其中：

- Dense Retrieval：一个排序通道。
- BM25：一个排序通道。
- Entity Matching：一个排序通道。
- 初始权重相同。
- 初始 \(k=60\)。

不要直接将 Cosine Similarity 和 BM25 分数相加，因为它们不在同一个数值尺度。

## 7.3 Cell 聚合

检索首先发生在 Address Record 层级。

随后根据 Cell ID 聚合：

```python
cell_score[cell_id] = max(
    scores_of_all_matching_records_in_cell
)
```

使用最高匹配分数，避免拥有更多 Address Records 的 Cell 获得不合理的数量优势。

之后：

1. 按 Cell Score 排序。
2. 去除重复 Cell ID。
3. 选择最多 3 个候选 Cell。
4. 使用在 Development Set 上确定的置信度阈值。
5. 没有候选达到阈值时返回 ABSTAIN。

阈值必须在 Development Set 上确定，然后冻结。

## 7.4 非显式读取要求

必须保证：

- Query 不要求包含 Category。
- Query 不要求包含 Cell 名称。
- Session Binding 不作为读取的硬性限制。
- Category 不能作为硬过滤条件。
- 同一个 Query 可以返回多个不同 Category 的 Cell。
- 不相关 Query 必须允许返回 ABSTAIN。

权限过滤属于独立的 Access Control，不属于本次语义路由。

# 8. 实验数据

建立可重复的合成测试数据集。

使用 10 个独立 Cell：

| Cell | Category |
|---|---|
| A | projects/minijam |
| B | projects/minicells |
| C | projects/jamscript |
| D | projects/locus |
| E | domains/jam_protocol |
| F | domains/ai_memory |
| G | domains/linear_models |
| H | domains/llm_training |
| I | domains/finance |
| J | shared/architecture_decisions |

每个 Cell 创建 12 条不同的 Memory Events。

总计：

- 10 Cells。
- 120 Events。
- 120 Event Address Records，单类别事件。
- 额外增加多类别投影记录。

事件内容必须具有真实的语义区别，同时设计有意相似的 Hard Negatives。

例如：

- Titans Neural Memory 与 TTT-MLP。
- DeltaNet 与 Gated DeltaNet。
- MiniJAM 网络架构与 JAM 协议架构。
- Locus Ownership 与一般账户所有权。
- Qwen 训练与 Neural Memory 在线更新。

不能只使用明显不同的关键词，否则无法测试真正的语义寻址。

## 8.1 Query Dataset

建立 80 条人工编写的 Query：

| 类型 | 数量 |
|---|---:|
| 单 Cell 直接查询 | 16 |
| 单 Cell 语义改写查询 | 16 |
| 多 Cell 查询 | 16 |
| Hard Negative 查询 | 12 |
| 无匹配查询 | 8 |
| 信息不足、应该拒绝路由的查询 | 12 |
| 合计 | 80 |

多 Cell 查询中，至少一半必须跨 Category。

每条 Query 必须包含：

```json
{
  "query_id": "query_001",
  "query": "我们之前为什么改变了神经记忆的更新方式？",
  "relevant_cell_ids": [
    "cell_minicells",
    "cell_ai_memory"
  ],
  "expected_action": "route",
  "query_type": "cross_category"
}
```

对于无匹配或信息不足的 Query：

```json
{
  "query_id": "query_080",
  "query": "昨天会议中提到的具体预算是多少？",
  "relevant_cell_ids": [],
  "expected_action": "abstain",
  "query_type": "no_match"
}
```

Query 需要人工编写，不能直接复制 Descriptor 的原句。

特别是语义改写组，不能直接出现 Category 名称。

## 8.2 数据划分

固定：

- Development：24 Queries。
- Holdout：56 Queries。

按照 Query 类型分层划分。

Holdout 不得参与阈值、权重和参数调优。

# 9. 必须执行的实验组

比较四种寻址方式：

| 实验组 | 检索方式 |
|---|---|
| E0 | BM25 Only |
| E1 | Dense Only |
| E2 | Entity Matching Only |
| E3 | Hybrid RRF |

另外，E3 必须执行两种索引内容对照：

- E3-Raw：使用原始 Event Text。
- E3-Descriptor：使用 Summary + Entities + Topics + Aliases。

这样可以判断：

1. 纯语义检索是否足够。
2. 关键词是否能补充语义检索。
3. 实体匹配是否能够解决技术名词和专有名称问题。
4. 结构化 Descriptor 是否比直接索引原始内容更有效。

不要增加训练组，不要训练 Router。

# 10. 跨 Session 核心场景

必须完整实现以下场景：

**Session A**

创建：

- Cell a
- Cell b
- Cell c

分别写入不同类别的内容。

**Session B**

创建：

- Cell d
- Cell f

写入不同类别的内容。

**Session C**

不绑定任何 Category。

提出一个需要读取 a、b、f 中知识的自然语言 Query。

预期：

```json
{
  "session_id": "session_c",
  "query": "...",
  "expected_cell_ids": [
    "cell_a",
    "cell_b",
    "cell_f"
  ],
  "returned_cell_ids": [
    "cell_a",
    "cell_b",
    "cell_f"
  ]
}
```

必须证明：

- C 不需要知道 a、b、f 来自哪个 Session。
- C 不需要提供 Category。
- 不需要加载 c、d。
- 不能因为 a、b、f 的创建 Session 不同而无法查询。
- 返回结果必须只包含命中的 Cell ID，不返回 Descriptor 中的答案内容。

# 11. 持久化与一致性

本次不实现真实 Titans Checkpoint。

但是必须验证 Address Index 自身的持久化和一致性。

使用 SQLite 或仓库已有的轻量持久化实现。

必须测试：

1. 新进程能够重新加载 Registry。
2. 新进程能够重新加载 Address Index。
3. Cell ID 不发生变化。
4. Address Record 不发生丢失。
5. 删除 Cell 后旧地址不再参与候选。
6. 更新 Cell Version 后，新地址记录能够正确关联。
7. 多类别写入失败时，不产生部分地址记录。
8. 注入写入失败时，Cell Mock Version 和 Address Index 必须一起回滚。

使用 Failure Injection 测试事务一致性。

不要在本实验中实现完整的神经网络 Checkpoint 或跨机器迁移。

# 12. 评估指标

## 12.1 Write Routing

| 指标 | 目标 |
|---|---:|
| Write Assignment Accuracy | 100% |
| Multi-Write Exact Match | 100% |
| Inbox Fallback Accuracy | 100% |

## 12.2 Read Routing

主要指标：

\[
Recall@K=
\frac{\text{Top K 中找到的相关 Cell 数}}
{\text{全部相关 Cell 数}}
\]

必须报告：

- Recall@1
- Recall@3
- Recall@5
- MRR
- Precision@3
- Multi-Cell AllRelevant@3
- Cross-Category Recall@3

另外报告：

- No-Match False Positive Rate。
- Abstention Accuracy。
- 平均候选 Cell 数。
- CPU 检索延迟。

## 12.3 初步通过标准

| 指标 | 通过标准 |
|---|---:|
| Write Assignment Accuracy | 100% |
| Overall Recall@3 | >= 90% |
| Cross-Category Recall@3 | >= 85% |
| Multi-Cell AllRelevant@3 | >= 80% |
| No-Match False Positive Rate | <= 10% |

Hybrid 应当与 Dense Only、BM25 Only 进行比较，不能仅报告 Hybrid 自身结果。

如果 Hybrid 在整体或关键子集上明显不如最佳单通道，必须分析原因，不能直接判定通过。

以上阈值是本次工程可行性 Gate，不代表生产环境的最终性能要求。

# 13. 实现建议

先检查当前仓库的实验目录、依赖、数据结构和测试框架。

优先复用已有实现，不要重写 Titans、Memory Runtime 或训练框架。

建议采用独立实验目录；如果仓库已有统一的 Experiment 目录规范，应当沿用。

参考目录：

```text
research/experiments/memory-addressing-v0.1/
    README.md
    config.yaml

    fixtures/
        cells.jsonl
        events.jsonl
        queries.dev.jsonl
        queries.holdout.jsonl

    src/
        registry.py
        category_router.py
        descriptor.py
        address_index.py
        retrieval.py
        read_router.py
        ranking.py
        persistence.py

    tests/
        test_write_routing.py
        test_descriptor.py
        test_retrieval.py
        test_multi_cell.py
        test_cross_session.py
        test_persistence.py
        test_failure_atomicity.py

    scripts/
        run_experiment.py
        analyze_results.py

    artifacts/
        address_records.jsonl
        query_results.jsonl
        metrics.json
        RESULTS.md
```

以上是建议结构，不要求为了遵循目录而重复实现仓库中已经存在的模块。

# 14. 执行要求

提供统一入口：

```bash
python scripts/run_experiment.py --mode all
```

支持：

```bash
--mode write
--mode retrieval
--mode cross-session
--mode persistence
--mode all
```

运行要求：

- CPU 可以完成。
- 固定随机种子 42。
- 固定 Embedding Model Revision。
- 记录 Git Commit SHA。
- 记录依赖版本。
- 记录 Dataset SHA-256。
- 不得静默下载或重复加载大型 Backbone。
- Embedding 应当批量生成并缓存。
- Holdout 结果只能在配置冻结后运行。

如果 Embedding Model 不可用，不得悄悄将 Dense Retrieval 替换成关键词检索并声称实验通过。应明确标记 BLOCKED。

# 15. 最终交付

Codex 必须交付：

1. 可运行的完整实验代码。
2. 固定的 Cells、Events 和 Queries 数据集。
3. 单元测试及集成测试结果。
4. E0–E3 对比数据。
5. 跨 Session 寻址结果。
6. Address Index 持久化和 Failure Injection 结果。
7. 完整 `RESULTS.md`。

RESULTS.md 必须明确回答：

- 自然语言 Query 能否自动找到正确 Cell？
- 是否能够同时找到多个不同 Category 的 Cell？
- 是否能够找到不同 Session 创建的 Cell？
- Dense、BM25、Entity 和 Hybrid 哪种效果最好？
- Descriptor 是否带来了实际收益？
- 无关问题能否正确拒绝路由？
- Address Index 是否能够跨进程持久化？
- 当前机制还有哪些明确的失败模式？

# 16. 最终决策标准

根据实验结果进行以下决策：

**PASS**

满足主要指标。

下一步研究自动 Descriptor Builder，以及真实 Titans Cell 的接入。

**PARTIAL PASS**

写入和基础语义寻址成功，但跨类别、多 Cell 或拒绝路由存在明显问题。

只修复已定位的问题，不扩大实验范围。

**FAIL**

Gold Descriptor 条件下仍然无法稳定找到相关 Cell。

说明当前 Address Index + Hybrid Retrieval 设计本身需要重新考虑，不应继续接入 Titans 或 MoM。

**BLOCKED**

由于依赖、模型下载或环境问题无法完成实验。不得将其当作科学失败或成功。

---

# 17. 明确禁止的事项

本实验最重要的是保持研究边界。

1. 不训练 Qwen。
2. 不训练 Titans。
3. 不加载 MoM-GatedDeltaNet。
4. 不改变现有 Memory 架构。
5. 不重写已有 Cell Runtime。
6. 不使用 Cell Category 作为读取硬过滤条件。
7. 不允许用户在 Query 中必须指定 Cell。
8. 不使用 Address Descriptor 直接生成最终答案。
9. 不在 Holdout 结果出来后临时修改指标。
10. 不为了追求结果而无限增加实验组。

**本实验只回答一个问题：我们能否建立一个可靠的、跨 Session 的、以 Cell ID 为寻址目标的外部神经记忆索引？**

如果这个问题得到肯定答案，我们才进入下一阶段的 Titans + Address Index 集成。
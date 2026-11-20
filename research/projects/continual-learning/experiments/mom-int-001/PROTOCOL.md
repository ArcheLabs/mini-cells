# MoM Integration：多神经记忆与线性记忆集成可行性实验

**实验编号：** MOM-INT-001
**项目：** MiniCells
**实验类型：** 架构可行性验证
**实验基座：** MoM-Gated-DeltaNet-340M
**后续正式基座候选：** MoM-Gated-DeltaNet-1.3B
**前置实验：** MA-001、MA-001.4
**优先级：** P0
**状态：** READY FOR IMPLEMENTATION

---

# 1. 实验目的

本实验验证 MiniCells 的混合神经记忆架构能否成立。

我们此前已经通过 MA-001.4 证明：

- 多个真实 Titans 可以独立存在。
- 多个 Titans 可以共享静态模型参数。
- 一次操作可以更新多个 Titans。
- 一次推理可以读取多个 Titans。
- Titans 可以独立 Checkpoint、Restore 和 Migration。
- Address Index 可以找到多个真实 Titans Cell。

但是，目前的架构还缺少一个重要部分：线性记忆。

我们希望在多个独立 Titans 之外，加入一个线性记忆模块，作为全局概要记忆。

两者承担不同职责：

**Titans Neural Memory**
- 保存不同项目、专业和主题的独立记忆。
- 支持显性类别写入。
- 支持跨 Session 读取。
- 可以独立 Checkpoint 和迁移。
- 可以在一次推理中组合多个 Cell。

**Linear Memory（MoM / DeltaNet）**
- 维护共享的、紧凑的连续记忆状态。
- 提供跨 Cell 的全局记忆视图。
- 保存有助于后续推理的概要信息。
- 不要求用户显式指定类别。
- 与多个 Titans 同时参与一次推理。

本次实验需要回答：

1. MoM 线性记忆能否与多个独立 Titans 同时运行？
2. 一次写入能否同时更新线性记忆和多个 Titans？
3. 一次推理能否同时读取线性记忆和多个 Titans？
4. 线性记忆与 Titans 的状态是否能够保持独立？
5. 所有记忆状态能否一起保存、恢复和迁移？
6. 是否能够通过现有 Address Index 找到 Titans，并将其与线性记忆组合？

**本实验只验证架构可行性，不验证记忆质量和回答质量。**

# 2. 实验基座与范围

## 2.1 模型选择

本次使用：

**MoM-Gated-DeltaNet-340M**

选择原因：

- 作为最低成本的 MoM 集成实验基座。
- 优先验证架构，而不是模型规模。
- 不需要从头训练。
- 不需要重新训练 Titans。
- 不需要修改 Backbone 权重。

MoM-Gated-DeltaNet-1.3B 保留为后续正式基座候选。

本轮不启动 1.3B。

只有当 340M 的实际架构或接口无法验证目标能力时，才重新评估是否需要更大模型。

## 2.2 必须先核实实际架构

开始实施前，检查 340M 模型的真实实现，明确：

- MoM 的组成结构。
- Linear Memory 的具体实现。
- 是否存在多个线性记忆专家。
- 内部 Router 的输入和输出。
- Memory State 的数据结构。
- 状态更新机制。
- Read / Forward 的实际行为。
- Checkpoint 所需的全部状态。
- 是否能够在一次 Forward 中保持和恢复记忆状态。

不得根据模型名称推断具体实现。

特别注意：

MoM 内部 Router 与 MiniCells 外部 Address Router 是两个不同层级的组件。

- MoM Router：负责模型内部的记忆模块选择。
- Address Router：负责选择外部 Titans Cell。

本次不修改 MoM 内部 Router，也不要求它负责显性的项目分类。

如果 MoM 使用多个线性记忆专家，应当保留其原生机制，不得擅自将它们等同于 Titans Cell。

## 2.3 关于概要记忆的定义

本实验将 MoM 的原生线性记忆状态作为共享记忆通道，目标是让它承担跨 Cell 的全局记忆作用。

但必须区分：

- 线性记忆的紧凑递归状态；
- 人类可读的自然语言摘要。

DeltaNet 的矩阵状态并不天然等同于一段文本摘要。

本次不要求实现高质量的自动摘要生成。

如果需要摘要输入，可以使用固定的、预先构造的测试内容，以隔离架构集成问题。

不得将外部文本缓存或普通 KV Cache 冒充 MoM Linear Memory。

# 3. 目标架构

目标结构：

```text
                    User Query
                         |
                         v
                 Memory Orchestrator
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Linear Memory             Address Index
        (MoM State)                   |
             |                       v
             |                  Cell Registry
             |                       |
             |             +---------+---------+
             |             |         |         |
             |             v         v         v
             |          Titans A Titans B Titans C
             |             |         |         |
             |             +---------+---------+
             |                       |
             +-----------+-----------+
                         |
                         v
                  Memory Integration
                         |
                         v
                 MoM Backbone 340M
                         |
                         v
                   Final Output
```

其中：

**Linear Memory**

一个共享的线性记忆状态，服务于整体系统。

**Titans Cells**

多个独立的神经记忆状态，分别保存不同领域或项目的记忆。

**Address Index**

继续使用 MA-001 的现有实现，不在本轮重新设计。

**Memory Orchestrator**

负责协调线性记忆和 Titans 的写入、读取、状态管理。

**Memory Integration**

负责将不同记忆来源提供的结果组合到同一次推理中。

# 4. 状态管理设计

本次必须严格区分以下状态。

| 状态 | 数量 | 说明 |
|---|---:|---|
| MoM Backbone Parameters | 1 | 共享静态模型参数 |
| MoM Linear Memory State | 1 | 共享线性记忆状态 |
| Titans Cell A State | 1 | 独立神经记忆 |
| Titans Cell B State | 1 | 独立神经记忆 |
| Titans Cell C State | 1 | 独立神经记忆 |
| Address Index | 1 | 外部地址索引 |
| Cell Registry | 1 | Cell 生命周期和元数据 |

具体 MoM State 数量应根据 340M 的真实实现确定。如果模型原生具有多个线性记忆状态或 Router 状态，必须完整保留。

核心要求：

1. MoM 线性记忆不能与 Titans State 共享可变状态。
2. Titans A、B、C 继续保持 MA-001.4 的独立性。
3. 更新线性记忆不能意外修改 Titans。
4. 更新 Titans 不能意外修改线性记忆。
5. 不允许每个 Cell 重复加载一份完整 Backbone。
6. 所有记忆状态必须具有明确的版本和生命周期。

# 5. 实验一：Linear Memory 独立运行

**目的：确认 MoM 的线性记忆可以作为独立的可变状态使用。**

执行：

1. 加载 MoM-Gated-DeltaNet-340M。
2. 初始化 Linear Memory State。
3. 使用固定输入序列更新线性记忆。
4. 执行实际的 MoM Memory Read / Forward。
5. 记录状态指纹。
6. 再次写入不同输入。
7. 检查状态变化。
8. 验证读取不会意外改变持久化状态。

验收：

- 使用真实 MoM Linear Memory。
- 状态可以独立初始化。
- 状态可以独立更新。
- 状态可以独立读取。
- 能够获取状态指纹。
- 不依赖外部文本缓存。

不要求证明其语义记忆效果。

# 6. 实验二：Linear Memory 与 Titans 共存

**目的：证明两种神经记忆可以同时存在。**

创建：

| 记忆 | 内容 |
|---|---|
| Linear Memory | 全局概要信息 |
| Titans A | MiniJAM 项目记忆 |
| Titans B | MiniCells 研究记忆 |
| Titans C | JamScript 项目记忆 |

要求：

- MoM 使用一份共享模型参数。
- Linear Memory 独立维护自己的状态。
- Titans A、B、C 分别维护独立状态。
- 四种记忆可以同时加载。
- 各自可以单独更新。
- 更新其中一个，不得意外修改其他记忆。

执行交叉隔离测试：

1. 仅更新 Linear Memory。
2. 检查三个 Titans State。
3. 仅更新 Titans A。
4. 检查 Linear Memory、B、C。
5. 更新 Titans B、C。
6. 再检查全部状态。

验收：所有状态隔离测试通过。

# 7. 实验三：一次操作写入多种记忆

**目的：验证一次逻辑操作可以同时更新 Linear Memory 和多个 Titans。**

这是本实验的重要部分。

输入示例：

```text
New information:

MiniJAM has completed its multi-cell runtime validation.

The runtime supports independent Titans states,
multi-cell reads, atomic writes and checkpoint recovery.

The next research direction is MoM integration.
```

预先指定写入目标：

```text
Linear Memory:
    更新全局概要

Titans A:
    写入 MiniJAM 相关事件

Titans B:
    写入 MiniCells 架构研究事件
```

注意：

写入目标可以在测试中显式指定。

本轮不验证自动类别分类能力。

接口建议：

```python
write_memory(
    event=event,
    linear_memory=True,
    cell_updates=[
        {
            "cell_id": "cell_a",
            "payload": payload_a
        },
        {
            "cell_id": "cell_b",
            "payload": payload_b
        }
    ]
)
```

必须证明：

- 一次逻辑操作可以更新三种状态。
- Linear Memory 实际发生更新。
- Titans A、B 实际发生更新。
- Titans C 保持不变。
- 每种记忆的状态指纹可以独立检查。

## 7.1 原子性与失败回滚

增加故障注入：

场景：

1. Linear Memory 更新成功。
2. Titans A 更新成功。
3. Titans B 更新失败。

预期：

- Linear Memory 不应留下部分提交的新状态。
- Titans A 不应留下部分提交的新状态。
- Titans B 不应留下部分提交的新状态。
- 所有版本保持一致。

建议采用 Copy-on-Write：

```text
Original States
       |
       v
Create Staged States
       |
       v
Update Linear Memory
       |
       v
Update Titans A
       |
       v
Update Titans B
       |
       +------ Failure
       |          |
       |          v
       |     Discard All
       |
       v
Atomic Commit
```

本次只要求单进程范围内的原子提交，不要求分布式事务。

# 8. 实验四：一次推理读取 Linear Memory + 多个 Titans

**目的：证明三种记忆可以在同一次推理中共同发挥作用。**

这是本次实验最关键的验收项目。

输入：

```text
Query:
What is the current architecture and research direction
of MiniCells?
```

执行：

1. 读取 Linear Memory。
2. 通过现有 Address Index 查找相关 Titans Cell。
3. 找到 Titans A、B。
4. 分别执行真实 Titans Read。
5. 获取各自的实际记忆读取结果。
6. 将 Linear Memory 和 Titans A、B 的结果送入同一次 Backbone 推理。
7. 保留各来源的身份和版本。

接口建议：

```python
read_memory(
    query=query,
    include_linear_memory=True,
    cell_ids=[
        "cell_a",
        "cell_b"
    ]
)
```

输出结构：

```json
{
  "query_id": "query_001",
  "linear_memory": {
    "source": "mom_linear_memory",
    "state_version": 2,
    "result": "..."
  },
  "titans_cells": [
    {
      "cell_id": "cell_a",
      "state_version": 3,
      "result": "..."
    },
    {
      "cell_id": "cell_b",
      "state_version": 2,
      "result": "..."
    }
  ],
  "inference": {
    "used_linear_memory": true,
    "used_titans_cells": [
      "cell_a",
      "cell_b"
    ]
  }
}
```

注意：

必须是真正的神经记忆读取结果。

不能仅仅将 Address Descriptor 拼接起来，然后宣称完成了多神经记忆读取。

## 8.1 Memory Integration

首先检查 MoM 340M 是否存在可供外部记忆接入的原生接口。

如果没有，则实现最小的 Memory Integration Adapter。

优先顺序：

1. 原生 Memory Interface。
2. 原生隐藏状态或 Memory Token 接口。
3. 最小的外部记忆表示注入接口。

如果 Titans 与 MoM 的表示维度不同：

- 先检查是否存在现有兼容映射。
- 如果没有，可以实现固定、确定性的维度适配。
- 不进行训练。
- 不得将固定映射的成功解释为已经获得有效语义融合能力。

必须保留每种记忆的来源标记。

## 8.2 因果数据流验证

仅仅调用多个 Read 函数还不够。

必须证明它们的结果实际参与了 Backbone 推理。

对同一 Query 执行：

A. 只使用 Linear Memory。

B. 只使用 Titans A、B。

C. 同时使用 Linear Memory 和 Titans A、B。

比较：

- 输入 Backbone 的记忆表示。
- 关键隐藏状态。
- 最终输出 logits 或其他可观测输出。

通过关闭某个记忆来源，确认对应输入确实从推理路径中移除。

这里不比较回答准确率。

只证明不同记忆来源真实进入同一次推理。

# 9. 实验五：Checkpoint、Restore 与 Migration

**目的：验证混合记忆架构能够完整保存和恢复。**

必须同时保存：

1. MoM Linear Memory State。
2. MoM 必要的可变运行状态。
3. Titans A State。
4. Titans B State。
5. Titans C State。
6. Cell Registry。
7. Address Index。
8. 模型版本和参数指纹。
9. 各记忆状态版本。

Manifest 示例：

```json
{
  "checkpoint_version": 1,
  "backbone": {
    "model_id": "mom-gated-deltanet-340m",
    "model_fingerprint": "..."
  },
  "linear_memory": {
    "state_version": 2,
    "state_hash": "..."
  },
  "titans_cells": [
    {
      "cell_id": "cell_a",
      "state_version": 3,
      "state_hash": "..."
    },
    {
      "cell_id": "cell_b",
      "state_version": 2,
      "state_hash": "..."
    }
  ]
}
```

实际 Manifest 必须包含继续运行所需的全部状态，不能只保存上述示例字段。

## 9.1 跨进程恢复

Process A：

1. 加载 340M。
2. 初始化 Linear Memory。
3. 初始化三个 Titans。
4. 执行多路写入。
5. 执行一次联合读取。
6. 保存全部状态和 Hash。
7. 退出进程。

Process B：

1. 启动全新 Python 进程。
2. 加载相同版本的 340M。
3. 恢复 Linear Memory。
4. 恢复三个 Titans。
5. 恢复 Address Index 和 Registry。
6. 校验所有状态 Hash。
7. 执行同样的联合读取。
8. 再次执行多路写入。
9. 检查更新后的状态和后续读取。

验收：

- 不依赖原进程对象。
- 不重新构造记忆。
- 不重新执行原始写入。
- 所有记忆状态完整恢复。
- 恢复后能够继续运行。

## 9.2 Migration

将整个 Checkpoint Bundle 复制到新的目录。

在新的存储目录中：

1. 校验文件 SHA-256。
2. 恢复 MoM Linear Memory。
3. 恢复 Titans A、B、C。
4. 恢复 Registry 和 Address Index。
5. 执行联合读取。
6. 再执行一次写入。

验证迁移后的运行状态与原环境一致。

本轮不要求跨机器部署。

# 10. 实验六：Address Index 完整集成

**目的：验证原有寻址架构可以与混合神经记忆共同工作。**

继续使用 MA-001 的冻结 Raw Hybrid，不重新调整权重和阈值。

Session A：

- 创建 Titans A。
- 创建 Titans B。
- 写入不同记忆。

Session B：

- 创建 Titans C。
- 写入不同记忆。

Linear Memory：

- 维护全局概要状态。
- 不依赖某一个 Titans Cell 的生命周期。

Session C：

1. 输入自然语言 Query。
2. 不指定 Cell ID。
3. 不指定项目类别。
4. Address Index 找到相关 Cell。
5. 加载对应 Titans。
6. 读取共享 Linear Memory。
7. 同一次推理组合 Linear Memory 和多个 Titans。

注意：

只使用少量预先选定的有效 Query。

本次不重新测试 MA-001 的语义泛化能力，也不把单条成功 Query 解释为寻址质量已经得到证明。

# 11. 验收标准

| 验收项目 | 通过条件 |
|---|---|
| MoM 340M | 成功加载真实模型 |
| 原生 Linear Memory | 成功初始化、更新和读取 |
| 多 Titans 共存 | 至少三个独立 Titans |
| 状态隔离 | Linear Memory 与 Titans 相互独立 |
| 多路写入 | 一次操作更新 Linear Memory 和多个 Titans |
| 原子回滚 | 任一更新失败时全部不提交 |
| 联合读取 | 一次推理读取 Linear Memory 和多个 Titans |
| 因果数据流 | 不同记忆来源实际进入 Backbone |
| 来源追踪 | 保留 Cell ID 和版本 |
| Checkpoint | 完整保存所有必要状态 |
| Restore | 全新进程恢复并继续运行 |
| Migration | 新目录中恢复并继续运行 |
| Address Integration | 自然语言查询能够找到多个 Titans，并与 Linear Memory 联合使用 |

以上核心项目全部通过，才能判定本次集成可行性 PASS。

# 12. 资源与工程约束

本次实验坚持最低成本原则。

- 只使用 340M。
- 不启动 1.3B。
- 不训练 Backbone。
- 不训练 Titans。
- 不训练 Router。
- 不训练融合 Adapter。
- 不进行大规模 Benchmark。
- 不优化 NLI。
- 不重新调整 Address Index。
- 不进行模型蒸馏。
- 不增加额外的大型依赖，除非实际集成需要。

记录：

- 模型加载时间。
- Linear Memory 初始化时间。
- Titans 初始化时间。
- 多路写入时间。
- 联合读取时间。
- Checkpoint 大小。
- Restore 时间。
- 峰值内存。

这些数据用于资源预算，不作为本轮效果指标。

# 13. 实验产物

建议目录：

```text
research/projects/continual-learning/experiments/mom-integration/

    README.md
    EXPERIMENT.md
    config.yaml

    src/
        mom_backend.py
        linear_memory.py
        titans_adapter.py
        memory_orchestrator.py
        memory_integration.py
        checkpoint_manager.py

    tests/
        test_mom_linear_memory.py
        test_multi_titans_coexistence.py
        test_state_isolation.py
        test_multi_memory_write.py
        test_atomic_rollback.py
        test_joint_memory_read.py
        test_causal_integration.py
        test_checkpoint_restore.py
        test_migration.py
        test_address_integration.py

    fixtures/
        fixed_memory_inputs.json
        fixed_queries.json

    scripts/
        run_experiment.py

    artifacts/
        environment.json
        model_manifest.json
        memory_manifest.json
        checkpoint_manifest.json
        state_hashes.json
        runtime_metrics.json
        test_results.json
        RESULTS.md
```

实际目录应优先遵循仓库已有结构，不要重复实现 MA-001.4 已有的通用组件。

# 14. GO / NO-GO 决策

## PASS

证明：

- MoM 线性记忆可以独立运行。
- 多个 Titans 可以与其同时存在。
- 一次操作能够更新多种记忆。
- 一次推理能够读取多种记忆。
- 记忆来源实际参与 Backbone 推理。
- 全部记忆能够保存、恢复和迁移。
- Address Index 能够与混合记忆运行时协作。

这意味着 MiniCells 的混合神经记忆架构在工程运行层面成立。

下一步可以考虑使用 1.3B 进行确认，并开始研究更完整的记忆效果。

## PARTIAL PASS

部分记忆路径成立，但存在明确的接口或状态管理限制。

必须说明：

- 哪个模块失败。
- 是否影响核心架构。
- 是否可以通过轻量 Adapter 解决。
- 是否需要改变原有 Titans 或 MoM 实现。

## FAIL

MoM Linear Memory 与 Titans 无法在同一次推理中共存，或无法保持独立状态和一致性。

需要重新审视架构，不得通过简单拼接文本或外部缓存掩盖问题。

## BLOCKED

340M 模型或必要的真实 Titans 实现无法加载，导致无法完成实验。

不得使用 Mock Linear Memory 或 Mock Titans 宣布成功。

# 15. 明确不在本次范围内的研究

本轮禁止扩大为以下工作：

1. NLI 事实判断优化。
2. Semantic Reranker 训练。
3. Address Index Recall 优化。
4. 自动类别识别。
5. 自动摘要质量评估。
6. 长期记忆效果 Benchmark。
7. 多 Cell 答案准确率 Benchmark。
8. MoM 内部 Router 优化。
9. 1.3B 训练或大规模测试。
10. 新的 Titans 架构搜索。

特别注意：

本次验证线性记忆与 Titans 的架构组合，不验证线性记忆能否自动形成高质量的人类可读摘要。

如果集成成功，这将为后续研究提供运行基础。

# 16. 最终研究目标

MiniCells 最终希望形成以下结构：

```text
                  User Query
                       |
                       v
               Memory Orchestrator
                       |
           +-----------+-----------+
           |                       |
           v                       v
     Global Linear Memory      Address Index
       (MoM / DeltaNet)             |
           |                       v
           |                  Titans Cells
           |                 +-----+-----+
           |                 |     |     |
           |                 A     B     C
           |                 |     |     |
           +-----------------+-----+-----+
                       |
                       v
               Memory Integration
                       |
                       v
                 MoM Backbone
                       |
                       v
                  Final Output
```

其中：

- Linear Memory 提供共享的全局记忆通道。
- Titans 提供多个独立、可迁移的神经记忆。
- Address Index 负责发现需要访问的 Titans。
- Memory Orchestrator 负责协调读写。
- MoM Backbone 负责综合不同来源的信息。

用户不需要知道某个记忆存储在哪个 Cell，也不需要显式指定项目或专业。

用户只需要提出问题，系统即可组合全局线性记忆和多个独立 Titans。

**本次实验最终只需要证明：线性记忆与多个 Titans 可以组成一个统一的、可写入、可读取、可持久化、可迁移的混合神经记忆系统。**

如果 MOM-INT-001 通过，我们就完成了 MiniCells 核心混合记忆架构的工程可行性验证。

之后再研究它是否能够记得更好、找到得更准、回答得更可靠。
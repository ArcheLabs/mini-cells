这次结果需要认真处理，但我认为它并不意味着 Titans Cell 的跨 Session 记忆路线已经失败。

目前可以确定的是：EPCLM-006 没有证明 Cell 学会了跨 Session 知识迁移。与此同时，实验设计还存在一个足以使 Exact Match 失去解释力的问题：训练答案 Token 池与测试答案 Token 池完全分离。

我认为不能简单地把它归结为训练步数不足，然后把 32 次更新增加到几百次。我们需要先区分三个问题：

1. 模型有没有学会输出答案？ 即使知道正确答案，模型是否具备生成对应 Token 的能力？
2. Titans 有没有学会写入和读取关联？ Cell 是否真正保存了 Support Facts 与答案之间的关系？
3. 跨 Session 是否成功？ 新 Session 是否能利用导入的 Cell 恢复这种关联？

这三个问题必须分别测量。否则，即使继续增加训练量，也可能只是在一个无法正确评价的目标上消耗 GPU。

## 一、我检查代码后的结论

我检查了发布提交 `65c19b4` 中的实验结果，以及它实际使用的源代码提交 `d6f25b1`。

发现的问题不止是 Token 池划分。

其中有一个非常重要的训练逻辑错误：Cell Dropout 关闭了记忆分支，却没有同时取消对应样本的答案监督。

这意味着模型被要求完成一部分在信息论上无法完成的任务。

### 1. P0：Cell Dropout 制造了不可学习的训练目标

位置：[portable_model.py](https://github.com/ArcheLabs/mini-cells/blob/d6f25b1cb1b7e62351dab8d7c83b88356ef374ff/research/projects/cellular-language-models/experiments/epclm-006-cross-session-cell-portability/portable_model.py?utm_source=chatgpt.com)

当前代码：

```
keep = torch.rand(len(ids), device=ids.device) >= (    P['cell_dropout'] * len(ids) / max(1, len(ids)-1))keep[0] = True
```

随后：

```
delta = delta * keep[:, None, None]
```

问题在于，`keep=False` 时，模型虽然不再使用 Cell 输出，但是原来的 `labels` 完全没有改变。

当前配置：

- Batch = 4
- Cell Dropout = 0.15
- 每个 Episode 有 4 个查询。

因此，每个 Batch 平均有 0.6 个 Episode 被关闭 Cell 分支，却仍然要求模型回答正确。

这些答案是每个 Episode 随机生成的 Key-Value 映射。没有 Cell，也没有其他历史信息，模型根本不知道答案是什么。

这实际上是在训练模型：

> 即使没有获得某个随机映射，也必须猜出这个映射对应的答案。

预计 512 个训练目标中，约有 77 个目标受到这种不可解监督的影响。

必须修复

我的修正建议：

- 第一轮记忆能力验证，训练期间完全关闭 Cell Dropout。
- 所有训练查询必须经过正确 Cell。
- C0 无 Cell 仅作为评估对照。
- 如果以后需要训练无 Cell 的可选模式，必须增加明确的 `UNKNOWN / NO_MEMORY` 目标，不能继续监督随机答案。

这是我认为当前训练实现中最需要优先修复的问题。之前实验规格中提出的 Optional Cell Dropout 没有充分区分这两种训练目标，这一点需要纠正。

### 2. P0：实际上进行了 unseen-symbols，而不是 unseen-pairs

位置：[portability_data.py](https://github.com/ArcheLabs/mini-cells/blob/d6f25b1cb1b7e62351dab8d7c83b88356ef374ff/research/projects/cellular-language-models/experiments/epclm-006-cross-session-cell-portability/portability_data.py?utm_source=chatgpt.com)

当前：

```
POOLS = {    'train': (0, 128),    'validation': (128, 192),    'test': (192, 256),}
```

这里不仅答案 Token 不同，Key Token 也不同。

| 数据集        | Key ID  | Value ID |
| ---------- | ------- | -------- |
| Train      | 8–135   | 264–391  |
| Validation | 136–199 | 392–455  |
| Test       | 200–263 | 456–519  |

这导致：

- 验证阶段的 Key Embedding 没有经过训练。
- 验证阶段的 Value Embedding 没有作为正确答案经过训练。
- LM Head 与 Token Embedding 又是绑定的。
- 模型必须同时泛化到从未学习过的 Key 和 Value 符号。

这不是我们想验证的能力。

我们想验证的是：

模型已经认识 Key 和 Value 的基本符号，但不知道当前 Episode 中它们之间的映射关系。它必须通过 Cell 找到映射。

这与已有的 EPCLM-004 数据设计恰好有对应关系：

`MQARConfig` 已经区分：

- `unseen_pairs`
- `unseen_symbols`

EPCLM-006 实际采用了第二种，而下一轮应该采用第一种。

正确设计：

- Train、Validation、Test 使用相同的 Key 符号集合。
- 使用相同的 Value 符号集合。
- 三个 Split 的 Key-Value Pair 不得重叠。
- 各种 Key 和 Value 在训练中都必须获得充分覆盖。
- 每个 Episode 内仍然生成独立的随机映射。

这样才能验证模型是否学会了关联检索，而不是识别一个陌生 Token。

### 3. P0：E3 实际上是随机初始化的，不是训练完成的 Backbone

位置：

- [portable_model.py](https://github.com/ArcheLabs/mini-cells/blob/d6f25b1cb1b7e62351dab8d7c83b88356ef374ff/research/projects/cellular-language-models/experiments/epclm-006-cross-session-cell-portability/portable_model.py?utm_source=chatgpt.com)
- [baselines.py](https://github.com/ArcheLabs/mini-cells/blob/d6f25b1cb1b7e62351dab8d7c83b88356ef374ff/research/projects/cellular-language-models/experiments/epclm-004-unified-memory-10m/m2/baselines.py?utm_source=chatgpt.com)

当前执行路径：

```
self.backbone = build(    'E3',    config,    P['model_seed'],    state_size=state,)
```

`build()` 内部重新创建 E3 和 E0 Reference，然后复制初始化参数。

这里没有加载一个已经完成语言模型训练的 E3 Checkpoint。

所以当前模型实际是：

- 约 10.17M 参数。
- 随机初始化。
- 只有 32 次优化更新。
- 只有 512 个监督答案 Token。

而且，这 32 次更新来自 M2 工程测试训练配置的扩展版本。

M2 的原始协议明确写着：

`scope: engineering correctness and cost only`

训练配置是：

- Learning Rate：0.0003
- Weight Decay：0.1
- Updates：2
- Scheduler：`constant-engineering-smoke`

EPCLM-006 只是把 Updates 改成了 32，其他训练策略仍然来自工程 Smoke 配置。

因此，不能把它当作一个合理的 10M 模型训练方案。

### 4. P1：8192 类输出空间不适合当前检索任务

当前：

```
self.lm_head = self.backbone.lm_head
```

LM Head 与 Token Embedding 共享权重。

训练使用全词表 Cross Entropy，但每个查询实际上只有 8 个候选 Value。

也就是说，模型需要从 8192 个 Token 中生成答案，而任务本身只需要从 8 个候选值中找到正确的一个。

当前输出任务

# 8,192

全词表分类空间

实际关联检索任务

# 8

每个查询的有效候选值数量

这使得训练同时承担了：

1. 学习 Key-Value 关联。
2. 学习将记忆表示解码为 Token。
3. 学习从巨大词表中排除无关 Token。

我们现在最需要验证的是第一项。

因此，应当先采用 Candidate Ranking，而不是继续使用完整的 8192 类 LM Loss。

### 5. P1：K、V、Q 的表示存在不必要的耦合

这是一个值得修正的架构问题。

目前：

```
hidden = self.encode_packets(segments)k, v, _ = self.memory.project(hidden)
```

K 和 V 都来自完整的 Fact Segment：

```
BOS → FACT → KEY → VALUE
```

但是 Q 来自：

```
BOS → QUERY → KEY → ANSWER
```

也就是说，写入时的 Key 表示已经包含 Value 信息，而读取时的 Query 表示没有 Value 信息。

虽然模型理论上可以学习通过 `Wk` 忽略 Value 部分，但这增加了不必要的表示学习难度。

我建议在下一版本中明确分离：

- K：只从 Key 编码。
- V：只从 Value 编码。
- Q：只从查询 Key 编码。

即：

\\[ K=W_k(E\_{\text{key}}(k)) \\]

\\[ V=W_v(E\_{\text{value}}(v)) \\]

\\[ Q=W_q(E\_{\text{query}}(k)) \\]

这样，Titans Memory 的任务就明确成为：

\\[ (k,v)\rightarrow Cell,\qquad k\rightarrow v \\]

而不是让它自行从两个不同上下文中学习如何分离 Key 和 Value。

这需要升级 Cell Interface 版本，不能直接覆盖旧版 `canonical-reset-segment-v1`。

## 二、现有实验到底证明了什么？

我们需要把工程正确性与模型能力分开。

| 项目                    | 结果      | 判断   |
| --------------------- | ------- | ---- |
| Cell Export / Import  | PASS    | 工程通过 |
| Fresh Process Restore | PASS    | 工程通过 |
| Session State 隔离      | PASS    | 工程通过 |
| Read-only Hash        | PASS    | 工程通过 |
| Cell 大小               | 39.2 KB | 工程指标 |
| 跨 Session Recall      | 0%      | 未证明  |
| Correct vs Unrelated  | 无显著差异   | 未证明  |
| 模型训练泛化                | 预测坍缩    | 失败   |

还有一个非常重要的细节。

训练记录显示：

| 指标              | 初始       | 最终          |
| --------------- | -------- | ----------- |
| Validation NLL  | 9.0876   | 9.7250      |
| Validation EM   | 0%       | 0%          |
| Validation 预测分布 | 多种 Token | 128 次均为 328 |

这意味着问题不仅仅是测试答案 Token 不在训练词表。

模型本身已经出现了严重的训练泛化退化。

另外，C1 相比 C2 的 NLL 差异只有 0.000272，95% CI 包含零；C1 相比 C0 的 NLL 反而恶化约 0.437。

这说明目前没有证据证明模型能够区分正确 Cell 和无关 Cell。

还有一点：192 个评估 Episode 实际来自 Validation Split 的三个评估 Bank，并不是独立 Test Split。真正的 Test 仍然封存。这一点在下一轮需要继续保持。

## 三、下一步不应该继续扩大模型，而应该拆开验证学习机制

我建议新建一个独立实验：

建议的新实验

# EPCLM-007

Cell Retrieval Learning Validation

核心目标：

验证 Titans Memory 是否真正学会了在独立 Episode 中建立 Key-Value 关联，并在新 Session 中恢复这种关联。

不覆盖 EPCLM-006，不修改其历史结论。

我建议分成三个阶段。

### Phase A：先证明 Titans 本身能够学习关联

暂时不要引入完整 E3。

先构造一个非常小的实验：

```
Key Encoder ──→ K ──┐
                    │
Value Encoder ─→ V ─┼──→ Titans Cell
                    │
                    └──→ Online Update

Query Encoder ─→ Q ───→ Cell Read
                           │
                           ▼
                     Retrieved Value
```

输入 8 组随机 Key-Value：

```
K1 → V7
K2 → V3
K3 → V9
K4 → V1
...
```

随后使用独立 Query：

```
Q1 → ?
Q3 → ?
Q2 → ?
```

要求模型恢复正确的 Value。

这一阶段需要验证：

1. Titans 是否能够通过在线更新写入 8 组关联。
2. 是否能够在查询时恢复正确 Value。
3. 如果改变 Value 的映射，是否能够识别出不同的结果。
4. Cell 是否在读取期间保持不变。
5. 是否能够通过 Export / Import 保持相同的检索能力。

这里最重要的是，先使用确定的 K/V/Q 表示，避免同时调试 Backbone、Tokenizer 和 Memory。

如果这一阶段失败，就应该直接研究 Titans 更新规则，而不是继续调整 E3。

### Phase B：引入 E3，但冻结 Backbone

Phase A 通过后，再引入 E3。

采用：

- E3 / Samba-lite：冻结。
- Mamba State：独立。
- SWA KV Cache：独立。
- Titans Cell：独立。
- K/V/Q Projection：训练。
- Titans Update Parameters：训练。
- Memory Readout：训练。

这样能够减少需要优化的参数数量。

同时，必须采用新的 Cell Interface：

```
K = Wk(E3(Key Segment))
V = Wv(E3(Value Segment))
Q = Wq(E3(Query Segment))
```

三个 Segment 都通过 Canonical Reset Interface 编码。

这次我们不再让 K 和 V 从同一个完整 Fact Hidden 中提取。

如果仓库没有真正训练过的 E3 Checkpoint，就必须明确将它标记为随机初始化的冻结特征提取器，不能称为预训练语言模型。

### Phase C：重新验证跨 Session Portability

只有前两阶段通过，才进入真正的跨 Session 实验。

实验条件：

| 条件       | 描述                       |
| -------- | ------------------------ |
| C0       | 没有 Cell                  |
| C1       | 正确 Cell                  |
| C2       | 错误映射 Cell                |
| C3       | Source Session 参考        |
| C3-reset | 新 Session + 直接加载 Cell    |
| C4       | 新进程 + Export/Import Cell |

这里我建议改变 C2 的设计。

不要再单纯使用另一组完全不同的 Key-Value。

应该使用相同的 Key 集合和 Value 候选集合，但将 Value 随机置换，且对被查询的 Key 保证没有固定点。

这样 C1 和 C2 的数据分布完全一致，区别只有映射关系。

这比当前 C2 更适合验证记忆关联是否真正有效。

## 四、必须修改的训练目标

这是新实验最重要的变化。

### 1. 从语言生成改为候选检索

每个 Episode 提供 8 个候选 Value：

Source Session

写入 Cell 的事实

K1 → V7

K2 → V3

K3 → V9

K4 → V1

K5 → V6

K6 → V2

K7 → V8

K8 → V4

Target Session

查询：`K3 → ?`

候选：

V1

V3

V4

V6

V7

V8

V9

V2

正确答案：V9

Top-1

模型不再需要在 8192 个 Token 中生成答案，而是比较 Memory Read 与 8 个候选 Value 的表示。

例如：

\\[ s_j=\operatorname{cosine}(R(Q),V_j)/\tau \\]

\\[ L\_{\text{retrieval}}= \operatorname{CrossEntropy}(s,\text{correct index}) \\]

这里：

- \\(R(Q)\\)：Titans 读取结果。
- \\(V_j\\)：候选 Value 表示。
- \\(\tau\\)：温度参数。

训练模型选择正确候选。

以后我们再增加自然语言解码，将检索出的内容转化为文本。这是另外一个问题，不应该与当前记忆机制验证混在一起。

### 2. 修复数据集

建议：

| 参数                  | 新实验建议 |
| ------------------- | ----- |
| Key 符号              | 256   |
| Value 符号            | 256   |
| 每个 Cell 的事实         | 8     |
| 每个 Episode 查询       | 4     |
| Train Episodes      | 4096  |
| Validation Episodes | 512   |
| Test Episodes       | 512   |
| 候选 Value 数量         | 8     |

所有 Split 共享符号集合，但 Key-Value Pair 不得重复。

另外必须加入三个数据测试：

- Oracle Dictionary Retrieval：100% 正确。
- Random Candidate Selection：约 12.5%。
- Pair Overlap：0。

这样能够确保数据集本身没有问题。

### 3. 重新设计训练过程

建议采用：

- 第一阶段：Memory-only Meta Training。
- 第二阶段：冻结 E3 的 Memory Integration。
- 初始 Screening：1024 次优化更新。
- Batch：先保持 4。
- 每 100 次更新执行一次 Validation。
- 根据 Validation 选择最佳 Checkpoint。
- 不再直接采用 M2 的 Engineering Smoke 配置。
- 不使用全模型统一的 0.1 Weight Decay。
- 对 Fast Weight Initialization、Norm 和 Bias 设置独立优化器参数组。

1024 次更新只是下一轮的初始 Screening 预算，不代表训练一定会收敛。应先进行预算测试和小规模过拟合验证。

特别注意：初始阶段不要启用 Cell Dropout。

等 Cell 能够稳定完成关联检索后，再单独研究无记忆模式和拒答机制。

## 五、必须增加的诊断指标

当前实验最大的问题之一，是我们只知道 Cell Hash 发生了变化，却不知道它究竟学到了什么。

现有 `NeuralMemory.learn()` 已经返回：

```
{    "inner_loss": ...,    "eta": ...,    "theta": ...,    "alpha": ...}
```

但 `PortableModel.acquire()` 直接丢弃了这些诊断信息。

下一轮必须记录：

| 指标                       | 用途                           |
| ------------------------ | ---------------------------- |
| Inner Loss               | 判断 Fast Weight 更新是否有效        |
| Fast Weight Delta        | 判断 Cell 是否实际学习               |
| Momentum Norm            | 观察更新动力学                      |
| Eta / Theta / Alpha      | 观察 Titans 更新控制参数             |
| Memory Read Cosine       | 检查读取结果是否接近正确 Value           |
| Candidate Top-1          | 检查关联检索                       |
| Candidate Margin         | 检查正确答案与错误答案的区分度              |
| Gate Mean / Std          | 判断模型是否使用 Memory              |
| Delta Norm / Hidden Norm | 判断 Memory 输出影响程度             |
| C1 / C2 Logit Difference | 判断正确 Cell 与错误 Cell 是否产生可区分结果 |
| Per-module Gradient Norm | 检查各个学习路径                     |

特别关注：

Memory Read 是否真的包含正确 Value 的信息，而不只是输出一个经过训练的固定向量。

## 六、下一轮验收门禁

我建议设置以下门禁，避免再次浪费 GPU 时间。

Gate 0 — 数据与目标

- Train / Validation / Test 共享符号集合
- Pair Split 无重叠
- Oracle Retrieval 100%
- Random Baseline 约 12.5%
- 不存在无 Cell 却要求回答随机映射的训练样本

Gate 1 — Titans Memory

- 单 Episode 可以过拟合
- 固定 K/V/Q 测试通过
- Fast Weights 有效更新
- 正确 Cell 优于置换 Cell
- 读取结果与正确 Value 有明显关联

Gate 2 — E3 Integration

- K/V/Q 分离
- E3 状态隔离
- Candidate Retrieval 显著高于随机水平
- 不出现预测坍缩
- Validation Loss 持续改善

Gate 3 — Portability

- C1 优于 C0
- C1 优于 C2
- C4 新进程恢复后检索能力不下降
- Cell Read-only Hash 不变
- 不传递 Source Runtime

Gate 4 — Confirmation

- 三个独立 Training Seeds
- 独立 Test Split
- 配对 Bootstrap CI
- 完整资源与复现报告

对于 Gate 2，我建议将 8 选 1 的验证集 Top-1 准确率达到 50% 作为初始 Screening 目标，同时要求 C0 和 C2 接近随机水平，且 C1 相对两者的配对置信区间下界大于零。这个门槛需要在正式执行前冻结。

## 七、交给 Codex 的具体修改范围

我建议新建目录，不直接修改 EPCLM-006 的历史实现：

```
epclm-007-cell-retrieval-learning/
    README.md
    SPEC.zh-CN.md
    config.json

    data/
        generator.py
        pair_split.py
        manifest.py

    model/
        key_value_interface.py
        retrieval_model.py
        candidate_scorer.py

    memory/
        titans_adapter.py
        diagnostics.py

    training/
        trainer.py
        losses.py
        checkpoint.py

    evaluation/
        retrieval.py
        portability.py
        metrics.py
        bootstrap.py

    tests/
        test_data_split.py
        test_candidate_ranking.py
        test_no_cell_supervision.py
        test_memory_overfit.py
        test_kvq_separation.py
        test_cell_sensitivity.py
        test_cross_session.py
        test_fresh_process.py
```

Codex 的实施顺序：

1. 审计 EPCLM-006，生成问题核查报告。
2. 修复数据生成器和训练目标。
3. 增加直接 K/V/Q 的 Titans Memory 测试。
4. 实现新的 Cell Interface v2。
5. 增加 Candidate Ranking Loss。
6. 完成单 Episode Overfit 测试。
7. 冻结 E3，执行 Memory Integration。
8. 通过 CPU 和 T4 小规模预算测试后，才启动正式 Screening。
9. 通过后再验证 Portable Cell。

不要修改 EPCLM-004 的历史训练协议，也不要覆盖 EPCLM-006 的 Checkpoint 和实验报告。

## 八、我对整个研究路线的判断

这里有一个需要重新明确的研究层次。

我们实际上在研究三种不同能力：

第一层

Associative Learning

Titans 是否能够通过有限次在线更新，学习 Key-Value 关联？

第二层

Memory Retrieval

模型是否能够利用 Query，从 Cell 中准确恢复相关信息？

第三层

Cross-Session Portability

这种能力是否能够随 Cell 一起迁移到完全独立的 Session？

EPCLM-006 试图一次性验证这三件事，但训练目标、符号泛化和 Cell Dropout 又引入了额外困难。

我认为下一轮应该严格按这个顺序来。

先证明它会学习，再证明它会读取，最后证明它可以迁移。

这也意味着我们暂时不应该继续扩展 Titans 架构，不应该增加更大的 Memory，不应该增加更多 Gate。

当前最重要的工作是把学习任务设计正确。

如果 EPCLM-007 的直接 K/V/Q 测试都无法通过，我们就应该集中研究 Titans 的 Fast Weight 更新、Momentum 和 Meta-gradient，而不是继续投入 E3 的训练。

如果直接测试通过，但 E3 Integration 失败，问题就可以收敛到表示学习和接口设计。

如果 E3 Integration 通过，但跨 Session 失败，才需要重新研究 Portable Cell 的状态边界。

这会比再次运行一个 10M 参数模型，然后得到一组无法解释的 0% EM，更有效率。

我的建议是：EPCLM-006 正式冻结为 INCONCLUSIVE，下一步实施 EPCLM-007，不再对 EPCLM-006 进行补丁式重跑。
# EPCLM Generative Core-001
## 自回归生成引擎与基线评估设计

**Version:** v0.1\
**Project:** MiniCells / EPCLM\
**Experiment ID:** EPCLM-GC-001\
**Status:** PROPOSED\
**Base Architecture:** EPCLM Core-001 / C1\
**Execution:** Kaggle GPU\
**Training:** Reuse existing C1 checkpoint; train additional baselines under controlled budgets

---

# 1. 项目目标

## 1.1 背景

EPCLM Core-001 已经实现了第一版 Cellular Language Model。

当前 C1 具备：

- 16 个 Cell；
- 每个 Cell 128 维持久状态；
- 总 State Dimension 为 2,048；
- 固定 1D Ring，Radius 1；
- Cell 间局部通信；
- 共享 Cell Update；
- 残差状态更新；
- K=4 内部演化；
- Token Event 注入；
- State Readout；
- Next-Token Prediction；
- TBPTT；
- Snapshot / Restore；
- Checkpoint 和训练恢复；
- GPU 训练；
- Inspector 和机制诊断。

C1 已经完成 250K Token 开发训练。

后续机制实验已经初步证明 State 和 Cell Communication 对语言预测存在因果作用。

但是，当前项目还缺少一个经过完整验证的自回归生成产品，以及一套能够作为未来架构研究基准的标准化评估体系。

## 1.2 本项目的目标

本项目分为两个部分。

**A. Generative Core**

将当前 C1 变成完整的自回归语言生成引擎。

要求：

- 接收任意符合 Tokenizer 约定的 Prompt；
- 逐 Token 建立 Cellular State；
- 从 State 生成下一个 Token；
- 将生成 Token 作为新的 Event；
- 持续更新 State；
- 不依赖完整历史 Token 的 Attention 或 KV Cache；
- 支持连续生成、采样、停止、保存和恢复。

**B. Baseline Evaluation**

建立 EPCLM 的第一套可复现语言模型基准。

至少比较：

1. Cellular C1；
2. Flat Recurrent；
3. No-Communication Cellular；
4. 小型 Causal Transformer Reference。

如果仓库已经有成熟的 GRU、LSTM、SSM 或其他模型实现，优先评估复用，不重复开发相同基线。

## 1.3 本项目不解决的问题

本次不尝试：

- 证明 State Sufficiency；
- 实现完整语义记忆；
- 实现复杂指令跟随；
- 实现通用聊天能力；
- 证明额外内部迭代能够改善推理；
- 引入 Adaptive NCA；
- 引入 Path-Conditioned Execution；
- 引入 Hierarchical Cellular；
- 引入动态 Cell Growth；
- 引入在线权重学习。

这些属于后续独立研究。

本次的目标是建立一个可靠、可复现的生成模型和比较基准。

---

# 2. 架构冻结

## 2.1 C1 必须保持不变

本次使用已经训练的 C1，不重新设计基础架构。

| 项目 | 固定配置 |
|---|---|
| Cell Count | 16 |
| Cell Dimension | 128 |
| Total State | 2,048 |
| Topology | 1D Ring |
| Neighborhood | Radius 1 |
| Communication | Fixed Local |
| Update | Shared Parameters |
| Residual Update | 保持现有实现 |
| Event | Single Token |
| Default Evolution | K=4 |
| Precision | FP32 |
| Vocabulary | 现有 2,048 词表 |
| Objective | Next-Token Prediction |

必须通过当前仓库源码确认以上参数与真实实现一致。

如果发现历史文档与代码不一致，以实际 Checkpoint 和代码为准，并在审计报告中说明。

## 2.2 架构原则

本次不能引入：

- 完整历史 Token Attention；
- Transformer KV Cache；
- 外部历史 Token Memory；
- 隐式的历史序列重放；
- 依赖完整 Prompt 的 Readout。

Prompt 可以在生成过程中保留用于输出显示和调试，但已经消费的 Token 不得重新进入模型计算。

模型的历史信息只能通过当前 Cellular State 传递。

---

# 3. 自回归生成机制

这是本次最重要的实现部分。

## 3.1 基本计算

当前模型：

\[
S_{t+1}=F_\theta(S_t,x_t)
\]

\[
P(x_{t+1})=G_\theta(S_{t+1})
\]

其中：

- \(x_t\) 是当前输入 Token；
- \(S_t\) 是之前积累的 Cellular State；
- \(F_\theta\) 包含 Inject 和 Cellular Evolution；
- \(G_\theta\) 是 Readout。

生成过程：

\[
x_t
\rightarrow
S_{t+1}
\rightarrow
P(x_{t+1})
\rightarrow
\hat{x}_{t+1}
\]

生成的 Token 再作为新的 Event。

## 3.2 Prompt Prefill

输入：

```text
Once upon a time, Alice lived in a small village.
```

Tokenizer 将文本转换为：

```python
prompt_tokens = tokenizer.encode(prompt)
```

模型必须严格按照顺序处理：

```python
for token in prompt_tokens:
    core.inject(token)
    core.evolve(steps=4)
```

完成 Prompt Prefill 后：

```python
logits = core.read()
```

这个 Logits 用于预测 Prompt 之后的第一个 Token。

注意：

- 不需要为 Prompt 中每个 Token 保存完整历史表示；
- 不需要创建 KV Cache；
- 不需要将整个 Token Sequence 输入 Readout；
- State 是唯一的历史计算载体。

如果现有 Runtime 已经有等价的 Prefill 实现，应直接复用。

## 3.3 Autoregressive Generation

生成第一个 Token：

```python
next_token = sample(logits)
```

然后：

```python
core.inject(next_token)
core.evolve(steps=4)

logits = core.read()
```

重复这一过程：

```python
for _ in range(max_new_tokens):
    token = sample(logits)

    if token == eos_token_id:
        break

    output_tokens.append(token)

    core.inject(token)
    core.evolve(steps=4)

    logits = core.read()
```

以上为概念流程，实际实现需要根据 EOS 处理和现有 Tokenizer 语义调整。

**关键约束：每个生成 Token 只能注入一次。**

不得出现：

- Token 已经注入，又因为生成循环重复注入；
- Prompt 最后一个 Token 被重复消费；
- 每轮生成都重新处理整个 Prompt；
- 使用已生成的完整历史序列重新计算 State。

## 3.4 生成闭环

最终必须实现：

```text
Prompt
  |
  v
Tokenizer
  |
  v
Prompt Prefill
  |
  v
Cellular State
  |
  v
Readout
  |
  v
Next Token
  |
  +------> New Event
               |
               v
         Cellular Update
               |
               v
         Persistent State
               |
               v
            Readout
               |
               v
          Next Token
```

这个循环不依赖完整历史 Token。

---

# 4. Generation Runtime API

本次在现有 Runtime 上增加生成能力。

不要重新创建一套独立模型执行引擎。

建议提供以下接口。

## 4.1 Prefill

```python
runtime.prefill(prompt)
```

行为：

1. Reset 或创建新的 Runtime Session；
2. Tokenize Prompt；
3. 按顺序消费所有 Prompt Token；
4. 更新 Persistent State；
5. 返回第一个 Next-Token Logits。

要求：

- 不保存供模型重复访问的历史 Token；
- 允许返回消费 Token 数量；
- 允许检查当前 State；
- 支持空 Prompt 时按照现有 BOS 规则处理。

## 4.2 Generate

```python
runtime.generate(
    max_new_tokens=128,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
)
```

要求：

- 使用当前 State；
- 从 Readout 获取 Logits；
- 根据 Sampling 配置选择 Token；
- 将生成 Token 作为新 Event；
- 持续更新 State；
- 支持 EOS；
- 支持最大生成长度；
- 返回生成 Token 和文本。

## 4.3 Read

```python
logits = runtime.read()
```

Read 必须是只读操作。

重复调用 Read 不应改变：

- Cellular State；
- Token Position；
- RNG State；
- Runtime 状态。

## 4.4 Evolve

```python
runtime.evolve(steps=12)
```

必须保留原有的独立内部演化能力。

但本次默认生成过程固定为 K=4。

不要默认在每个 Token 后追加额外的 12 轮计算。

额外演化作为未来独立实验，不应混入本次标准生成基准。

## 4.5 Snapshot / Restore

```python
snapshot = runtime.snapshot()

runtime.restore(snapshot)
```

Snapshot 必须保存：

- 完整 Cellular State；
- 当前 Runtime Position；
- 模型 Checkpoint Identity；
- Model Configuration Hash；
- Tokenizer Identity；
- 当前 Generation Configuration；
- RNG State；
- 必要的运行状态元数据。

必须支持：

```text
Prompt Prefill
      |
      v
Generate 20 Tokens
      |
      v
Snapshot
      |
      v
Restore
      |
      v
Continue Generate
```

在确定性条件下，恢复后的结果应与未中断的连续生成一致。

随机采样时必须保存 RNG State。

模型权重和 Runtime Snapshot 应分别管理。

不允许用不匹配的模型 Checkpoint 恢复旧 State。

---

# 5. Tokenizer、EOS 与停止条件

首先审计现有 Tokenizer。

不得自行假设：

- BOS Token ID；
- EOS Token ID；
- PAD Token ID；
- 文档结束标记；
- 特殊 Token 的编码规则。

## 5.1 EOS

如果现有训练数据确实使用 EOS，并且模型接受过相应的 EOS 预测训练：

- 使用现有 EOS；
- 生成 EOS 后停止；
- EOS 不应作为普通输出文本显示。

如果当前 Tokenizer 或训练数据没有有效 EOS 训练机制：

- 不要创建一个没有训练依据的新 EOS；
- 使用 `max_new_tokens`；
- 可支持用户指定的 Stop Sequence；
- 明确报告当前模型没有学习到可靠的自然停止能力。

如果未来需要 EOS 学习，应作为独立训练改进处理。

## 5.2 Sampling

第一版支持：

| 模式 | 优先级 |
|---|---|
| Greedy | 必须 |
| Temperature | 必须 |
| Top-k | 必须 |
| Top-p | 必须 |
| Repetition Penalty | 可选 |
| Beam Search | 暂不实现 |

Sampling 必须直接作用于当前 Readout Logits。

不得引入新的上下文处理机制。

## 5.3 Determinism

支持：

- 固定 Seed；
- Greedy Deterministic Generation；
- Sampling RNG State 保存；
- 恢复后继续生成。

GPU 浮点计算可能存在确定性限制。

报告必须区分：

- 数值确定性；
- Token 序列确定性；
- 随机采样可恢复性。

---

# 6. Closed-loop Generation 验收

本次必须证明：

**生成的 Token 确实成为新的 Event，而不是通过隐藏的历史重放继续预测。**

## 6.1 Prefill 一致性

比较：

A. 一次性 Prefill 整个 Prompt。

B. 手动逐 Token Inject / Evolve。

最终 State 和 Logits 应保持一致。

## 6.2 Streaming 一致性

比较：

A. 一次性生成 64 Token。

B. 先生成 20 Token，Snapshot，再 Restore，继续生成 44 Token。

Greedy 模式下应得到相同结果。

## 6.3 No-History-Replay

加入测试或执行监控，确认：

- 模型每次只接收当前 Event；
- 已消费 Token 不会再次输入；
- 不存在完整历史 Attention；
- 不存在隐藏的 KV Cache；
- Runtime 只保留必要的 State 和输出 Token。

## 6.4 State Continuity

验证：

- 每个生成 Token 都更新 State；
- Read 不改变 State；
- Reset 只影响当前 Session；
- 新 Session 不继承上一 Session 的状态。

---

# 7. 生成质量评估

本项目的训练数据仍然是现有 TinyStories 体系。

因此第一版生成模型首先定位为：

**Small Autoregressive Story Language Model**

不宣称是通用聊天模型。

## 7.1 数据集

优先复用仓库现有数据：

- 原有 Tokenizer；
- 原有训练数据；
- 原有 Validation 数据；
- 原有测试数据划分；
- 原有文档边界处理方式。

必须检查：

1. Train / Validation / Test 是否严格隔离；
2. 文档是否存在跨集合重复；
3. 文档之间是否正确 Reset State；
4. EOS 和特殊 Token 处理是否一致；
5. 测试时是否存在训练数据泄漏。

如果现有数据划分不符合要求，应先修正评估协议，不要直接开始基线比较。

## 7.2 Teacher-forced Evaluation

使用真实历史 Token 更新 State：

```python
for token in sequence:
    core.inject(token)
    core.evolve(steps=4)
    logits = core.read()
    loss = criterion(logits, next_token)
```

实际代码应遵循当前训练器的预测对齐约定，避免发生一位 Token 的偏移。

报告：

- Validation NLL；
- Validation Perplexity；
- Test NLL；
- Test Perplexity。

其中：

\[
PPL=\exp(NLL)
\]

所有模型使用相同的 Tokenizer 和评估 Token 集合。

## 7.3 Free-running Generation

不能只报告 Teacher-forced PPL。

必须实际生成文本。

建议：

- 16 个固定 Prompt；
- 每个 Prompt 最大生成 128 Token；
- Greedy；
- Temperature 0.8；
- 固定 Seed；
- 保存全部生成结果。

如果资源允许，再增加不同 Sampling Seed 的结果。

## 7.4 Generation Metrics

至少记录：

| 指标 | 用途 |
|---|---|
| Generated Tokens | 实际生成数量 |
| Generation Latency | 生成延迟 |
| Tokens / Second | 生成吞吐 |
| EOS Rate | 自然停止情况 |
| Repeated 3-gram Rate | 重复退化 |
| Distinct-1 | Unigram 多样性 |
| Distinct-2 | Bigram 多样性 |
| Average Token Entropy | 输出分布变化 |
| Average Generated Length | 生成长度 |

这些指标不能替代人工阅读。

必须保留完整生成文本，检查：

- 是否存在连续重复；
- 是否出现 Token 退化；
- 是否能够维持基本句子结构；
- 是否存在异常终止；
- 是否出现持续无意义输出。

不得只展示最好的生成样例。

---

# 8. Baseline Architecture

本次建立两类比较：

1. 架构对照；
2. 参考模型。

不能把它们混为一谈。

## 8.1 Baseline A — Cellular C1

<badge color="success">Primary Model</badge>

使用现有 C1 Checkpoint。

固定：

- 16 Cells；
- 128 Dimensions；
- Ring Radius 1；
- K=4；
- 原有训练权重。

这是所有基线比较的中心模型。

优先复用现有 250K Token Checkpoint，不重新训练。

## 8.2 Baseline B — No-Communication Cellular

<badge color="info">Required Ablation</badge>

目的：

研究 Cellular Communication 对语言模型能力的贡献。

保持：

- 相同 Cell 数量；
- 相同 State Dimension；
- 相同 Event；
- 相同 Readout；
- 相同 K；
- 尽可能相同的参数预算。

禁用跨 Cell 信息传递。

需要区分两种实验：

**B1 — Frozen Inference Ablation**

直接在冻结 C1 上关闭通信。

这是机制诊断，不是独立训练模型。

已有机制诊断结果可以复用。

**B2 — Trained No-Communication Baseline**

使用相同的语言训练目标，训练一个没有跨 Cell 通信的模型。

这是架构对照。

B2 需要独立训练，不能用 B1 代替。

优先保留 Cell 的独立状态和共享更新机制，只移除跨 Cell 消息。

不得无意间同时改变多个架构因素。

## 8.3 Baseline C — Flat Recurrent

<badge color="info">Required Architecture Baseline</badge>

这是非常重要的对照。

目的：

回答一个核心问题：

> Cellular State 的组织方式是否优于一个相同规模的普通递归状态？

建议：

\[
S_t\in\mathbb{R}^{2048}
\]

\[
S_{t+1}=F_\theta(S_t,E_t)
\]

\[
Y_{t+1}=G_\theta(S_{t+1})
\]

不再拆分成 16 个 Cell。

保持：

- Persistent State；
- Event；
- Residual Recurrent Update；
- Next-Token Prediction；
- 相同 Tokenizer；
- 相同训练数据；
- 相同训练 Token 预算。

Flat Recurrent 应当是合理、可训练的递归模型，不能故意使用过弱的简单线性层。

实施前计算 C1 的实际参数量和 FLOPs，再确定 Flat Recurrent 的内部宽度或低秩结构。

建议建立两个比较口径：

1. State Dimension Matching：保持 2,048 维状态。
2. Parameter Matching：尽量匹配 C1 参数量。

如果无法同时满足，分别报告，不要人为隐藏差异。

## 8.4 Baseline D — Causal Transformer Reference

<badge color="secondary">Reference Model</badge>

建议增加一个小型 Causal Transformer，作为语言质量和计算成本的参考。

保持：

- 相同 Tokenizer；
- 相同训练数据；
- 相同训练 Token 数量；
- 尽可能相近的参数规模；
- 合理的 Context Window。

Transformer 可以使用正常的 Causal Self-Attention 和 KV Cache。

但必须明确：

**Transformer 是参考模型，不是与 State-first CLM 完全等价的架构对照。**

原因：

- Transformer 可以保留历史 Token 的分层表示；
- Transformer 可以直接访问 Context Window 内的历史；
- C1 依赖固定维度的 Persistent State；
- 两者的历史信息容量和访问方式不同。

因此需要同时报告：

- 参数量；
- Context Window；
- KV Cache 内存；
- Prefill 成本；
- Decode 成本；
- 训练 Token；
- 推理吞吐。

不能仅凭相同 Token 数量，就声称两者具有完全相同的计算预算。

如果 Kaggle 资源不足，Transformer 可以作为第二批参考实验，不得影响核心 C1 生成闭环的完成。

## 8.5 现有基线复用

实施前必须检查仓库中已有的：

- Flat RNN；
- GRU / LSTM；
- SSM；
- T1；
- X4；
- 其他已完成实验模型。

如果已经存在可复用实现，优先复用。

不得重复开发已经存在的基线。

---

# 9. Baseline Fairness Protocol

这是本次最重要的实验规范之一。

不同架构不能只比较最终 PPL，而不说明训练条件。

## 9.1 固定训练条件

尽可能保持：

| 项目 | 要求 |
|---|---|
| Tokenizer | 相同 |
| Vocabulary | 相同 |
| Training Data | 相同 |
| Validation Data | 相同 |
| Test Data | 相同 |
| Training Token Budget | 相同 |
| Data Order | 固定 |
| Reset Boundary | 相同 |
| Training Objective | Next-Token Prediction |
| Precision | FP32 优先 |
| Optimizer | 优先沿用现有训练协议 |
| Scheduler | 记录并保持一致 |
| TBPTT | 相同或明确记录差异 |
| Seed | 记录 |
| Checkpoint Selection | 仅使用 Validation |

C1 当前已有 250K Token Checkpoint。

优先复用它。

其他训练基线先使用同样的 250K Token 预算。

如果 C1 的历史训练数据、顺序或参数无法确认，则必须将比较标记为开发阶段比较，而不能声称严格公平。

## 9.2 三种预算必须分开

### A. Data Budget

训练 Token 数量。

### B. Parameter Budget

模型参数量。

### C. Compute Budget

训练 FLOPs、GPU 时间和推理 FLOPs。

这些预算无法保证同时完全一致。

因此至少建立：

- Token-matched comparison；
- Parameter-matched comparison；
- Compute-reported comparison。

如果资源允许，再进行 Compute-matched comparison。

不能仅仅因为两个模型训练了相同数量的 Token，就认定它们具有相同计算成本。

## 9.3 Hyperparameter Fairness

不同架构可能需要不同的学习率和优化参数。

本次不进行大规模超参数搜索。

建议：

1. 先沿用 C1 的训练协议；
2. 对明显不稳定的基线进行有限调整；
3. 所有模型使用相同的调整预算；
4. 记录每个模型实际使用的配置；
5. 不允许针对测试集调参。

第一轮结果属于 Development Benchmark。

只有在后续增加 Seed 和复现实验后，才能形成更强的比较结论。

---

# 10. 训练与推理的一致性

## 10.1 Teacher Forcing

现有训练使用 Next-Token Prediction。

训练阶段：

```text
真实 Token
    |
    v
Inject
    |
    v
Evolve
    |
    v
Read
    |
    v
预测下一个真实 Token
```

## 10.2 Free-running Inference

生成阶段：

```text
真实 Prompt Token
    |
    v
Inject
    |
    v
Evolve
    |
    v
Read
    |
    v
采样 Token
    |
    v
将采样 Token 作为新 Event
```

两者的区别在于，训练使用真实历史 Token，生成使用模型自己的预测结果。

这会产生 Exposure Bias。

本次不通过 Scheduled Sampling 等机制提前解决。

但必须分别报告 Teacher-forced 与 Free-running 结果。

## 10.3 TBPTT

必须检查：

- State 是否跨 TBPTT 截断点继续传递；
- 是否正确 detach；
- 文档边界是否 Reset；
- 不同模型是否采用相同截断策略；
- 训练和推理的 State 生命周期是否一致。

特别注意：

State 可以跨截断点传递，不代表梯度也跨截断点传播。

报告必须说明两者的区别。

---

# 11. 性能评估

本次除了语言质量，还要建立计算成本基准。

## 11.1 参数与状态

记录：

- Total Parameters；
- Trainable Parameters；
- Persistent State Dimension；
- State Memory Bytes；
- Model Checkpoint Size；
- Runtime Snapshot Size。

## 11.2 推理性能

分别测试：

- Prompt Prefill Latency；
- Single Token Decode Latency；
- 128 Token Generation Latency；
- Tokens / Second；
- Peak GPU Memory；
- CPU Memory；
- GPU Utilization（如果可靠可用）。

特别区分：

Prefill 与 Decode。

对于 C1：

Prompt Prefill 需要逐 Token 执行状态更新。

生成阶段只处理新生成的 Token。

历史 Token 不应反复进入模型。

## 11.3 历史长度测试

选择：

- 32 Tokens；
- 128 Tokens；
- 512 Tokens；
- 1,024 Tokens。

在不改变模型架构的前提下，测量：

1. Prefill 总耗时；
2. Prefill 平均每 Token 耗时；
3. 单 Token Decode 延迟；
4. State Memory；
5. Snapshot Size。

需要特别说明：

C1 的 Persistent State 大小理论上不随 Prompt 长度增长，但处理 Prompt 的总计算仍然随输入 Token 数量增长。

不能将固定 State Memory 误解为 Prompt Prefill 的常数时间。

Transformer 参考模型则应报告 Context Window 和 KV Cache 增长。

---

# 12. 运行预算与执行策略

考虑已有 Kaggle 实验的成本控制经验，本项目不进行大规模本地训练测试。

## 12.1 本地测试

仅保留必要的轻量测试：

1. Prefill / Manual Event Loop Equivalence；
2. Generated Token Exactly-once Consumption；
3. Read Is Immutable；
4. Snapshot / Restore Continuity；
5. Session State Isolation；
6. EOS / Stop Semantics；
7. No-History-Replay Contract。

CPU 测试只验证接口、状态转换和正确性。

不进行完整语言训练。

## 12.2 GPU Pilot

首先使用已有 C1 Checkpoint。

执行：

- 5 个固定 Prompt；
- 每个 Prompt 32 个生成 Token；
- Greedy；
- Snapshot / Restore；
- 显存和延迟记录。

确认闭环正确后，进入完整生成评估。

不需要为生成 Runtime 再做多轮重复 GPU Pilot。

## 12.3 Baseline Training

推荐第一轮顺序：

| 顺序 | 模型 | 训练预算 |
|---|---|---|
| 1 | Existing C1 | 复用 250K Checkpoint |
| 2 | No-Communication Cellular | 250K Tokens |
| 3 | Flat Recurrent | 250K Tokens |
| 4 | Causal Transformer | 可选第二批 |

每个新增基线第一轮只使用一个 Seed。

这是开发筛选，不是正式统计确认。

如果前两个基线已经出现明显实现错误或训练不收敛，停止后续训练，先修复。

不要为了完成表格而继续消耗 GPU 资源。

## 12.4 恢复机制

复用现有 Kaggle 训练恢复和结果发布器。

每个模型必须独立记录：

- Model ID；
- Configuration Hash；
- Training Data Identity；
- Training Token Count；
- Checkpoint SHA；
- Seed；
- Training Status。

未完成训练必须标记 INCOMPLETE。

不允许将不完整模型纳入正式对比。

---

# 13. 推荐执行阶段

## Phase A — Repository Audit

先只读检查：

- 当前 C1 Model；
- Runtime；
- Trainer；
- Tokenizer；
- Dataset；
- Checkpoint；
- Snapshot；
- Inspector；
- 现有 Baselines；
- Kaggle Notebook；
- Result Publisher。

输出：

`generative-core-audit.md`

明确：

1. 哪些生成能力已经存在；
2. 哪些只需要包装；
3. 哪些必须新增；
4. 哪些接口不兼容；
5. 是否存在现有基线可以直接复用。

本阶段不修改模型架构。

## Phase B — Generative Runtime

实现：

- Prefill；
- Generate；
- Sampling；
- EOS / Stop；
- Snapshot / Restore Continuation；
- Streaming Output（如现有 Runtime 适合）。

不进行训练。

验收完整自回归闭环。

## Phase C — Frozen C1 Generation Evaluation

使用已有 Checkpoint。

完成：

- 固定 Prompt；
- Greedy Generation；
- Sampling Generation；
- 生成文本；
- 重复率；
- 生成速度；
- Snapshot / Restore；
- 长度扩展测试。

输出第一份真实生成报告。

## Phase D — Baseline Implementation

按优先级：

1. No-Communication Cellular；
2. Flat Recurrent；
3. Causal Transformer Reference。

先检查已有实现，避免重复开发。

## Phase E — Controlled Training

使用相同的 250K Token 开发预算。

每个模型独立 Checkpoint。

禁止在测试集上调整参数。

## Phase F — Unified Evaluation

所有模型使用同一个评估入口。

统一输出：

- NLL / PPL；
- Generation Samples；
- Generation Metrics；
- Parameter Count；
- Compute；
- Memory；
- Latency；
- Training Cost。

最终生成比较报告。

---

# 14. 项目目录

建议在现有项目结构下增加：

```text
research/projects/cellular-language-models/
  experiments/
    epclm-core-001/
      implementation/
        ...
      generative-core/
        README.md
        generation.py
        sampling.py
        evaluation.py
        generation_tests.py

      baselines/
        flat-recurrent/
        no-communication/
        causal-transformer/

      reports/
        generative-core-001/
          decision.json
          summary.md
          generation-samples.md
          generation-metrics.json
          baseline-comparison.json
          performance.json
          metadata.json

      notebooks/
        generative-core-001.ipynb
```

以上为建议结构。

如果仓库已有统一的实验目录、代码组织和发布规范，应优先沿用现有约定，不要为了本项目重复创建基础设施。

---

# 15. 结果格式

## 15.1 Unified Baseline Report

建议：

| Model | Parameters | State | Train Tokens | Val PPL | Test PPL | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| C1 | | | 250K | | | |
| No-Comm | | | 250K | | | |
| Flat RNN | | | 250K | | | |
| Transformer | | | 250K | | | |

所有空值必须使用明确的 `NOT_RUN` 或 `NOT_AVAILABLE`，不得用估算值冒充实际测量。

Transformer 还需要独立报告 Context Window 和 KV Cache。

## 15.2 Generation Samples

每个模型保存相同 Prompt 的生成结果。

例如：

```text
Prompt:
Once upon a time, there was a little girl...

C1:
...

Flat RNN:
...

No-Communication:
...

Transformer:
...
```

所有模型使用相同的 Sampling 配置。

不得只展示表现最好的样例。

## 15.3 Decision Report

最终报告分别回答：

1. C1 是否能够完成完整自回归生成？
2. 生成是否真正依赖 Persistent State？
3. 生成质量是否具有基本语言连贯性？
4. Cellular Communication 是否具有可重复的语言质量贡献？
5. Cellular Organization 是否优于 Flat Recurrent？
6. C1 与 Transformer 的语言质量和计算成本有什么区别？
7. 哪些差异可能由模型大小、训练预算或超参数造成？

---

# 16. 成功标准

必须区分工程成功与科学成功。

## 16.1 Engineering PASS

必须满足：

- Prompt Prefill 正确；
- Autoregressive Generation 正确；
- Generated Token 只注入一次；
- 不使用完整历史 Attention；
- 不使用 KV Cache；
- Read 不修改 State；
- Snapshot / Restore 可以继续生成；
- Session 不发生状态泄漏；
- EOS 和停止逻辑正确；
- 生成结果可复现；
- 结果成功发布。

## 16.2 Language Generation Evaluation

必须获得：

- 完整 Validation PPL；
- Test PPL；
- 真实生成样例；
- 重复率；
- 生成速度；
- 内存使用。

不能仅凭工程 PASS 宣称模型已经达到 LLM 水平。

## 16.3 Architecture Evaluation

需要：

- C1；
- Trained No-Communication；
- Flat Recurrent；
- 统一评估；
- 参数量和训练预算记录。

只有这些比较完成后，才能讨论 Cellular Organization 的实际优势。

## 16.4 当前不能宣称的结论

即使本项目全部完成，也不能直接宣称：

- State 已经达到理论充分性；
- 模型具有可靠的长期语义记忆；
- 模型具有通用推理能力；
- Cellular 架构优于 Transformer；
- Cellular 架构在所有条件下优于 RNN；
- 额外内部演化可以提高推理能力。

这些需要独立实验。

---

# 17. 后续研究接口

本次生成引擎必须保留以下扩展空间：

| 后续研究 | 本次要求 |
|---|---|
| State Sufficiency | 保留 State Inspection 和 Reset |
| Selective Memory | 保留 Event 和 State Update 接口 |
| State Retrieval | 保留可替换 Readout |
| Adaptive NCA | 保留可替换 Communication |
| Path-Conditioned Execution | 保留可替换 Evolution Scheduler |
| Hierarchical Cellular | 不将 Cell 数量写死在公共接口 |
| Reasoning NCA | 保留无新 Event 的 Evolve |
| Variable Compute | 保留 `evolve(steps)` |
| Long Context | 保留 State-only Generation |
| JAM/PVM | 不引入平台依赖 |

这些扩展不能影响当前 C1 的冻结配置和历史 Checkpoint 兼容性。

---

# 18. Codex 实施要求

请严格按以下顺序执行。

### 第一阶段：审计

只读检查当前仓库。

不要立即修改代码。

首先报告：

- 当前生成能力；
- Runtime 结构；
- Checkpoint 和 Snapshot；
- Tokenizer 特殊 Token；
- 数据集划分；
- 训练预算；
- 现有 Baseline；
- 需要新增的代码。

### 第二阶段：实现

在确认现有结构后：

- 优先复用 Runtime；
- 只补充必要的 Generation API；
- 不修改 C1 核心计算机制；
- 不修改已有 Checkpoint；
- 不改变原始实验结果；
- 不进行不必要的代码重构。

### 第三阶段：验证

先运行轻量 CPU Contract Tests。

通过后执行一次小型 GPU Pilot。

确认正确后再进行完整生成评估。

### 第四阶段：基线

复用已有模型实现。

仅为缺少的核心对照开发新模型。

不要一次性开发大量架构。

### 第五阶段：发布

使用独立功能分支和独立实验结果分支。

不得直接修改 main。

不得覆盖既有 C1、C1 Diagnostic 和 CP-001 结果。

---

# 19. 最终交付

本项目完成后，应当得到：

1. 一个真正可用的 EPCLM 自回归生成 Runtime；
2. 一个能够持续运行、保存和恢复的 Cellular Language Model；
3. 完整的 Greedy 和 Sampling Generation；
4. C1 的真实文本生成结果；
5. No-Communication Cellular 对照；
6. Flat Recurrent 对照；
7. Causal Transformer 参考结果（资源允许时）；
8. 统一的 PPL、Generation、Latency 和 Memory 报告；
9. 可复用的 Baseline Evaluation Pipeline；
10. 为未来 EPCLM 架构研究冻结的第一版生成基准。

---

# 20. 最终研究定位

本项目不试图证明 EPCLM 已经解决了所有语言模型问题。

它首先要建立一个完整的、可运行的 State-first Cellular Language Model。

最终系统应当具备：

```text
External Event
      |
      v
Cellular State Injection
      |
      v
Local Communication
      |
      v
Persistent State Evolution
      |
      v
Language Readout
      |
      v
Autoregressive Generation
      |
      v
Generated Token as New Event
      |
      +-----------> Continue
```

这个系统不依赖完整历史 Token Attention。

它通过持续演化的 Cellular State 处理历史输入，并从 State 中生成新的语言。

**EPCLM-GC-001 的最终目标，是建立一个可靠的生成模型和可复现的架构比较基准，使以后所有关于 State、Memory、Communication、Adaptive NCA、Path-First 和 Reasoning 的研究，都能够在同一套生成引擎和评估协议上进行。**

文档状态：PROPOSED v0.1。

实施前必须完成 Repository Audit，并以当前实际代码为准。

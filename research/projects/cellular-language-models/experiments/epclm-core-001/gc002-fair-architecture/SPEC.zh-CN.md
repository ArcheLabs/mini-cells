# MiniCells 研究协议：CLM 与 Transformer 公平架构比较

**项目：** MiniCells / EPCLM  
**研究方向：** Direction C — Fair Architecture Comparison  
**协议版本：** v1.0  
**状态：** PROTOCOL DRAFT — NOT AUTHORIZED TO TRAIN  
**研究对象：** Cellular Language Model (CLM)  
**参照架构：** Transformer

---

## 1. 研究目的

### 1.1 核心定位

本研究的主要目标是评估 CLM 的语言建模能力，以及 Cellular State 作为一种持续计算与记忆机制的实际价值。

Transformer 仅作为参照架构，用来回答：

1. 在相同参数规模下，CLM 的语言建模能力如何？
2. 在相同训练计算预算下，CLM 能够达到怎样的语言建模效果？
3. 在相同任务下，两种架构的训练和推理成本分别是多少？
4. CLM 的持续状态是否能够以较低的存储成本保留并利用历史信息？
5. CLM 是否存在区别于 Transformer Context 的架构优势？

本研究不以证明 CLM 全面优于 Transformer 为目标。

我们需要建立的是一个可靠的评价体系，使 CLM 的进展能够得到独立、可重复的衡量。

### 1.2 当前问题

GC-001 已经完成了以下工作：

- Batch A：Cellular Communication 对照。
- Batch B：Native T1 Transformer 参考模型。
- Batch C：训练、生成、状态及统计诊断。

但当前 T1 不能作为严格公平的架构参照。

主要问题：

| 指标 | Cellular | T1 |
|---|---:|---:|
| Active Parameters | 约 965K | 1,894,080 |
| 训练 Token | 需要统一核验 | 250K |
| Transformer Context | 不适用 | 128 Tokens |
| Persistent State | 8,192 Bytes | 516 Bytes 的 Token Context 包装器 |
| KV Cache | 不适用 | Disabled |
| FLOPs | 尚未完整测量 | 尚未完整测量 |
| Training Seed | 1 | 1 |

T1 具有接近 Cellular 两倍的参数量。

同时，T1 没有启用 KV Cache，导致推理时重复处理历史上下文。

这意味着：

- 参数规模不匹配。
- 训练预算需要统一核验。
- 推理实现没有经过优化。
- 计算成本没有统一计量。
- 状态存储机制无法直接比较。

因此，当前 PPL 和训练速度结果只能作为历史参考，不能作为正式的架构优劣判断。

### 1.3 研究原则

本项目遵循以下原则：

1. CLM 是研究主体。
2. Transformer 是参照，不是研究目标。
3. 参数量、训练 Token 和计算量必须分别控制。
4. 推理实现必须达到合理的工程优化水平。
5. 持久状态和 KV Cache 必须采用统一的存储评价口径。
6. 语言建模能力与长期记忆能力必须分开评价。
7. 不允许通过改变数据、Tokenizer 或评估条件制造性能优势。
8. 不允许根据单次实验结果宣称架构优势。

---

## 2. 当前实验基线冻结

本节记录 GC-001 的已有事实，不修改历史实验结果。

### 2.1 CLM

主要实验模型：

- C1-BROADCAST
- IC-ALL-SELF
- F0-PM-BROADCAST

参数量约为：

- C1-BROADCAST：966,784
- IC-ALL-SELF：966,784
- F0-PM-BROADCAST：964,736

Cellular Persistent State：

- 2,048 个 FP32 数值。
- 总计 8,192 Bytes。

### 2.2 Transformer T1

当前配置：

- 4 Transformer Layers
- Hidden Dimension: 192
- Attention Heads: 6
- SwiGLU
- RoPE
- RMSNorm
- Context Length: 128
- Parameters: 1,894,080
- FP32

训练记录：

- Training Tokens: 250,000
- Updates: 977
- GPU: Tesla T4
- Training Time: 4,013 秒
- Throughput: 62.29 Tokens/s
- KV Cache: Disabled

### 2.3 必须复核的历史元数据

T1 的历史记录同时出现：

- resumed_from_tokens: 192000
- new_tokens: 58000
- training.tokens: 250000
- execution_policy: FRESH_NO_PRIOR_CHECKPOINT

在新实验开始之前，必须明确这些字段的统计含义。

另外，历史 Data Audit 记录的 Training Prediction Tokens 为 416,027，而 T1 的训练预算记录为 250,000。

必须区分：

- 数据集可用 Prediction Tokens。
- 实际训练消费的 Tokens。
- 每个模型重复处理的 Tokens。
- 训练更新次数。

不得将这些不同的统计口径混为一谈。

### 2.4 历史结果的使用边界

GC-001 的原始结果保持不变：

- Batch A、B、C 不重新计算或覆盖。
- H1–H5 保持 NOT_EVALUATED。
- Human Review 保持 PENDING。
- AI Review 仍然是独立的辅助评审。
- 不将本协议视为新训练授权。

---

## 3. 实验总体设计

本研究分为四个独立比较维度。

| 实验轨道 | 控制条件 | 主要研究问题 |
|---|---|---|
| A. Parameter-Matched | 参数量相同 | 相同模型容量下的能力 |
| B. Token-Matched | 训练 Token 相同 | 相同数据暴露下的能力 |
| C. Compute-Matched | 实际训练计算量相同 | 计算效率 |
| D. Inference & Memory | 相同任务和明确存储预算 | 推理成本与持久状态能力 |

A、B、C 是不同的控制条件，不应混成一个指标。

D 是独立的推理和记忆评价轨道。

### 3.1 轨道 A：Parameter-Matched

这是正式架构比较的主要基线。

目标：

使 Transformer 与 CLM 的 Active Trainable Parameters 尽可能接近。

目标参数规模：

约 965K。

要求：

- Transformer 参数量与 CLM 相差不超过 1%。
- 统计所有参与前向计算的可训练参数。
- Embedding、Output Head、Normalization、Bias 等均纳入统计。
- 不得通过遗漏参数制造虚假的匹配结果。

需要根据参数预算重新设计 Transformer：

- 调整 Layer 数量。
- 调整 Hidden Dimension。
- 调整 Feed Forward Dimension。
- 保持合理的 Attention Head 结构。

最终配置由参数统计脚本自动验证。

必须生成：

`parameter-match-report.json`

内容至少包括：

- 每个模块的参数量。
- 总参数量。
- Active Trainable Parameters。
- Non-trainable Parameters。
- 参数差异百分比。
- 最终模型配置。

如果无法在合理结构下实现 1% 匹配，可以使用 2% 作为有明确说明的次级容差，但不得直接将原始 T1 当作匹配模型。

### 3.2 轨道 B：Token-Matched

目标：

让 CLM 与 Transformer 接触完全相同的训练数据量。

控制条件：

- 相同 Tokenizer。
- 相同训练文档。
- 相同文档顺序。
- 相同 EOS 规则。
- 相同训练 Token 总量。
- 相同的有效 Batch Token 数。
- 相同的数据重复规则。

每次更新的 Token 数应尽量一致，使训练更新次数也保持一致。

必须记录：

- Total Training Tokens。
- Unique Training Tokens。
- Repeated Training Tokens。
- Optimizer Updates。
- Tokens Per Update。
- Training Epochs。
- Data Order Hash。

这里的重点是数据暴露量一致，而不是训练时间一致。

参数量相同也不代表计算量相同。

因此，轨道 B 必须与轨道 C 分开。

### 3.3 轨道 C：Compute-Matched

目标：

比较相同计算预算下，两种架构能够完成多少有效学习。

这是判断 CLM 计算效率的关键实验。

核心控制条件：

`Total Training FLOPs`

CLM 和 Transformer 应使用相同的训练 FLOPs 预算。

建议目标误差不超过 5%。

由于两种架构每个 Token 的计算量可能不同，允许：

- 实际训练 Token 数不同。
- Optimizer Updates 不同。
- 实际训练时间不同。

但必须保持：

- 相同数据来源。
- 相同 Tokenizer。
- 相同数据抽样规则。
- 相同评估集。
- 相同 FLOPs 统计标准。

主要输出：

- PPL vs Training FLOPs。
- NLL vs Training FLOPs。
- Accuracy vs Training FLOPs。
- Training Tokens vs Training FLOPs。

不能只报告最终一个 FLOPs 数值。

应当在多个预先指定的计算预算检查点保存模型表现，形成学习效率曲线。

### 3.4 轨道 D：Inference & Memory

目标：

比较两种架构在实际推理和状态保存方面的成本。

需要比较：

1. Prefill Cost。
2. Decode Cost。
3. FLOPs Per Generated Token。
4. Inference Latency。
5. Throughput。
6. Peak VRAM。
7. Persistent Per-Sequence State。
8. Total Runtime Memory。

Transformer 必须使用优化后的 KV Cache 实现。

CLM 必须使用正常的持续 State 更新机制。

不允许用没有 KV Cache 的 Transformer 作为正式性能参照。

---

## 4. Transformer 参考实现

### 4.1 模型配置

保留两种 Transformer 参考实现。

**T1-REFERENCE**

原始 T1：

- 1,894,080 Parameters。
- 4 Layers。
- Hidden 192。
- 6 Heads。
- Context 128。

用途：

保留历史 GC-001 的可重复参考。

不参与 Parameter-Matched 的正式结果。

**T1-KV**

在原始 T1 基础上：

- 保持模型参数不变。
- 实现 KV Cache。
- 实现增量式 Token Decode。
- 实现 Context Window 管理。
- 验证缓存前后的数值一致性。

T1-KV 是正式推理性能参照。

**T1-PM-KV**

重新设计约 965K 参数的 Transformer：

- 参数量与 CLM 匹配。
- 使用优化后的 KV Cache。
- 用于轨道 A、B、D。

**T1-CM-KV**

根据轨道 C 的计算预算，使用预先冻结的模型配置和训练预算。

不允许看到结果后临时改变配置。

### 4.2 KV Cache 实现要求

KV Cache 必须真正实现增量计算。

正常流程：

1. 对 Prompt 执行 Prefill。
2. 保存各层 Key 和 Value。
3. 每次只处理新生成的 Token。
4. 将新 Token 的 Key、Value 添加到缓存。
5. 不重复计算已有历史 Token。
6. Context 达到限制时执行明确的 Window 管理。

特别注意：

原始 T1 的 Context 截断语义必须先核验。

如果原始实现每次都重新截取最近 128 个 Token 并从头计算，那么简单保留旧 KV 并删除最早条目，不一定与原始实现等价。

新实现必须明确采用哪一种语义：

- 完整因果上下文。
- 从序列开始就启用的 128 Token Sliding-Window Attention。

推荐采用明确的 Sliding-Window Attention，并让 KV Cache 与相同的 Attention Mask 保持一致。

RoPE 的 Position ID 必须保持一致，不得在窗口移动时无意中重置。

### 4.3 KV Cache 正确性测试

必须实现：

`test_kv_cache_equivalence`

在相同输入、相同模型权重和相同 Attention Mask 下：

- 比较 Cached Forward。
- 比较 Uncached Forward。
- 比较每个 Token 的 Logits。
- 比较最终预测结果。

FP32 下应设定严格的数值误差阈值，例如最大绝对误差不超过 1e-5；如果运行环境存在非确定性算子，必须明确记录并制定经过验证的容差。

还需要测试：

- Context Length 32。
- Context Length 128。
- Window 滑动。
- EOS。
- 长序列连续生成。
- Batch Size 1。

如果缓存正确性测试失败，不允许将该实现用于正式性能比较。

### 4.4 KV Cache 存储计算

按照当前 T1 配置，如果：

- Layers = 4
- Hidden = 192
- Sequence = 128
- K/V 均使用 FP32

理论 KV Cache 大小：

`2 × 4 × 128 × 192 × 4 = 786,432 Bytes`

约为 768 KiB。

这只是理论计算值，实际运行还要记录：

- Cache Metadata。
- Token Buffer。
- Temporary Activations。
- Framework Overhead。

必须测量实际内存，不得只使用理论估算。

需要特别说明：KV Cache 主要优化自回归推理，不直接优化正常 Teacher-Forced 训练。因此，不能把 KV Cache 的缺失解释为 T1 训练吞吐率低的原因。

---

## 5. CLM 参考实现

### 5.1 保留现有模型

本轮首先保留：

- C1-BROADCAST。
- IC-ALL-SELF。
- F0-PM-BROADCAST。

不在本轮同时改变 Cellular 架构。

这样可以避免将架构优化和实验条件变化混在一起。

### 5.2 State 管理

必须明确区分：

- Model Parameters。
- Persistent State。
- Temporary Runtime Memory。
- Training Activations。
- Optimizer State。

当前 8,192 Bytes 是 Cellular Persistent State，不代表模型全部内存成本。

正式报告必须记录：

`persistent_state_bytes_per_stream`

以及：

`total_inference_memory_bytes`

### 5.3 State 生命周期

语言建模任务中：

- 每个独立文档按照既定协议初始化 State。
- 不允许无意间跨文档泄漏状态。
- 必须记录初始化和重置策略。

持续记忆任务中：

- State 必须跨输入块保留。
- 不允许在每个 Chunk 之间重新初始化。
- 必须明确记录 State 的持久化与恢复过程。

这两种任务不能使用同一种 State Reset 规则。

---

## 6. 数据与 Tokenizer

### 6.1 数据冻结

CLM 和 Transformer 必须使用同一份数据身份。

冻结：

- Dataset Manifest。
- Document IDs。
- Tokenizer。
- Tokenizer Vocabulary。
- Special Tokens。
- EOS ID。
- Train/Validation/Development Holdout 划分。

每个文件必须记录 SHA-256。

### 6.2 数据泄漏问题

GC-001 曾存在 Tokenizer 训练可能接触 Development 数据的问题。

新实验需要区分两种用途：

**Reproduction Track**

可以继续使用历史 Tokenizer 和历史数据划分，用于与 GC-001 比较。

但必须明确标记：

`DEVELOPMENT_ONLY`

**Confirmatory Track**

新建严格的数据协议：

- Tokenizer 仅使用 Training 数据训练。
- Validation 用于调参。
- Development Holdout 用于开发阶段分析。
- 独立 Test 数据保持封存。
- 最终配置冻结后才允许访问 Test。

不得把 Development Holdout 重新命名为 Test。

### 6.3 Token 统计

必须统一以下字段：

- dataset_prediction_tokens。
- train_budget_tokens。
- consumed_training_tokens。
- unique_training_tokens。
- repeated_training_tokens。
- validation_tokens。
- development_holdout_tokens。
- independent_test_tokens。

所有统计均以实际 Token ID 序列为依据。

---

## 7. 训练协议

### 7.1 统一条件

两种架构使用：

- 相同训练数据。
- 相同 Tokenizer。
- 相同数据顺序。
- 相同有效 Batch Token 数。
- 相同训练预算口径。
- 相同评估时间点。

Optimizer 可以根据架构进行合理配置，但必须遵守相同的超参数搜索预算。

不要求两个模型使用完全相同的 Learning Rate。

要求：

- 相同的调参次数上限。
- 相同的 Validation 数据。
- 相同的模型选择规则。
- 不允许使用 Holdout 或 Test 调参。

### 7.2 Seed

正式比较至少使用 3 个独立 Seed。

建议在确认性研究中使用 5 个 Seed。

每个 Seed：

- CLM 与 Transformer 使用相同的数据抽样顺序。
- 独立初始化模型参数。
- 独立记录训练曲线。
- 独立保存 Checkpoint。

不得将同一个 Seed 的不同 Prompt 误认为多个独立训练实验。

### 7.3 训练记录

必须记录：

- Training Loss。
- Validation NLL。
- Development Holdout NLL。
- Training Tokens。
- Updates。
- Training FLOPs。
- Training Time。
- Tokens/s。
- FLOPs/s。
- Peak VRAM。
- Optimizer State Memory。
- Gradient Norm。
- Numerical Overflow。
- Checkpoint Identity。

训练中断、恢复和重新开始必须使用不同的事件记录。

---

## 8. 持久状态与记忆能力评价

这一部分是 MiniCells 的核心研究方向，不能被 PPL 比较取代。

### 8.1 独立 Memory Benchmark

建立独立的、可控的记忆测试集。

不依赖 TinyStories 的自然语言生成结果。

建议包含以下任务。

**Task M1：Delayed Fact Recall**

输入一个事实：

Alice has a dog named Max.

随后输入大量无关内容。

最后询问：

What is Alice's dog's name?

**Task M2：Long-Range Recall**

逐步增加干扰 Token：

- 128
- 512
- 1,024
- 4,096
- 8,192

测量记忆准确率随距离变化的曲线。

**Task M3：Memory Update**

先输入：

Alice lives in London.

随后更新：

Alice moved to Paris.

最后询问 Alice 当前居住在哪里。

测量是否能够更新旧信息。

**Task M4：Interference**

输入多个相似实体和事实。

例如：

Alice owns Max.

Bob owns Max.

Alice owns a cat named Lily.

测量是否发生信息混淆。

**Task M5：State Persistence**

将输入拆分为多个 Chunk。

在 Chunk 之间：

- 保存 State。
- 恢复 State。
- 验证恢复后的记忆能力。

### 8.2 Transformer 对照

Transformer 使用 KV Cache 保存上下文。

CLM 使用 Persistent Cellular State。

对照条件：

- 相同输入内容。
- 相同信息位置。
- 相同问题。
- 相同输出规则。
- 相同评估标准。

至少进行两类比较：

1. 自然存储预算下的能力比较。
2. 明确限制每个 Session 的 Persistent Memory 后的能力比较。

第二种比较应设置多个存储预算点，形成 Memory Accuracy vs Memory Bytes 曲线。

不能简单地将 8,192 Bytes 的 Cellular State 与 516 Bytes 的 T1 Token Context 包装器比较。

必须把实际 KV Cache 纳入 Transformer 存储成本。

### 8.3 记忆评价指标

- Exact Recall Accuracy。
- Update Accuracy。
- Interference Error Rate。
- False Recall Rate。
- Accuracy vs Distractor Length。
- Accuracy vs Persistent Memory Bytes。
- State Serialization Size。
- State Recovery Correctness。
- FLOPs Per Recall Query。

最终需要形成：

`memory-retention-curves.json`

---

## 9. 性能评价

### 9.1 训练性能

必须分别报告：

- Tokens/s。
- FLOPs/s。
- Total Training FLOPs。
- Training Wall Time。
- Peak VRAM。
- Energy Consumption（硬件支持时）。

训练速度不能仅使用 Tokens/s 评价。

因为不同架构每个 Token 的计算量不同。

### 9.2 推理性能

Transformer 必须报告：

- Prefill Latency。
- Decode Latency。
- Cached Decode Throughput。
- KV Cache Memory。
- Peak VRAM。

CLM 必须报告：

- State Update Latency。
- Decode Throughput。
- Persistent State Memory。
- Peak VRAM。

两种架构均报告：

- Median Latency。
- P95 Latency。
- FLOPs Per Token。
- Total Runtime Memory。

### 9.3 硬件条件

优先在同一 GPU、同一软件环境下完成比较。

记录：

- GPU Model。
- Driver Version。
- CUDA Version。
- Framework Version。
- Precision。
- Batch Size。
- Kernel Configuration。
- Power Limit。

如果需要使用不同 GPU，必须单独标记硬件差异。

不得直接将不同硬件上的 Wall Time 当成架构效率差异。

---

## 10. 生成质量评价

PPL 不能代替生成质量。

所有模型使用相同的 Prompt 集合。

固定两种解码方式：

**Greedy**

用于诊断确定性退化和重复生成。

**Sampling**

固定 Temperature、Top-K、Top-P 和随机 Seed。

必须记录：

- EOS Rate。
- Repeated 3-gram Rate。
- Distinct-1。
- Distinct-2。
- Grammar。
- Local Coherence。
- Semantic Consistency。
- Prompt Relevance。
- Story Progression。

AI Review 仅作为辅助诊断。

正式质量评价必须由至少两名独立人工评审者完成，并记录评审者间一致性。

评审者不得提前看到模型身份和模型汇总结果。

---

## 11. 统计分析

### 11.1 配对比较

使用相同 Document ID 和 Token Count 进行配对分析。

报告：

- Mean NLL Difference。
- Relative PPL Change。
- Document Bootstrap 95% CI。
- Seed-Level Mean。
- Between-Seed Variance。

### 11.2 统计单位

必须区分：

- Document-level Variance。
- Training Seed Variance。
- Prompt-level Variance。
- Human Reviewer Variance。

Document Bootstrap 不能代替 Training Seed Variance。

多个 Prompt 也不能代替独立训练 Seed。

### 11.3 禁止的结论

不允许仅凭：

- 单 Seed。
- 单一 PPL。
- 单一训练时间。
- 单次生成样本。
- 单一 Memory Benchmark。

宣称某种架构全面优于另一种架构。

---

## 12. 实验矩阵

正式实施前必须冻结以下矩阵。

| 实验 | 模型 | 参数匹配 | Token 匹配 | FLOPs 匹配 | KV Cache |
|---|---|---|---|---|---|
| A1 | C1-BROADCAST | CLM 基准 | 是 | 否 | N/A |
| A2 | IC-ALL-SELF | CLM 基准 | 是 | 否 | N/A |
| A3 | F0-PM-BROADCAST | CLM 基准 | 是 | 否 | N/A |
| A4 | T1-PM-KV | 是 | 是 | 否 | Yes |
| B1 | C1-BROADCAST | CLM 基准 | 是 | 否 | N/A |
| B2 | T1-PM-KV | 是 | 是 | 否 | Yes |
| C1 | CLM Primary | CLM 基准 | 否 | 是 | N/A |
| C2 | T1-CM-KV | 按协议冻结 | 否 | 是 | Yes |
| D1 | CLM Primary | 原始参数 | 任务一致 | 单独测量 | N/A |
| D2 | T1-KV | 原始参数 | 任务一致 | 单独测量 | Yes |

说明：

- A、B、C 是不同的比较轨道。
- D 是推理与记忆测试。
- T1-REFERENCE 只作为历史参考，不参与正式公平比较。
- 所有模型都必须执行同一套评估程序。
- Seed 数量在正式运行前统一冻结。

---

## 13. 产物与目录结构

建议使用独立实验目录，不覆盖 GC-001。

```text
artifacts/experiments/epclm-core-001/
  gc002-fair-architecture/
    protocol/
      protocol.md
      protocol.json
      protocol.sha256

    audit/
      baseline-audit.json
      data-audit.json
      tokenizer-audit.json
      parameter-match-report.json

    models/
      clm/
        model-manifest.json
      transformer/
        t1-reference.json
        t1-pm-kv.json
        t1-cm-kv.json

    training/
      seed-001/
      seed-002/
      seed-003/

    profiling/
      training-flops.json
      inference-profile.json
      memory-profile.json
      kv-cache-equivalence.json

    evaluation/
      language-modeling.json
      generation-metrics.json
      memory-retention-curves.json
      human-evaluation.json

    statistics/
      paired-document-bootstrap.json
      seed-statistics.json
      effect-sizes.json

    reports/
      summary.md
      technical-report.md
      research-decision.json
```

所有产物必须记录：

- Git Commit SHA。
- Model SHA-256。
- Dataset SHA-256。
- Tokenizer SHA-256。
- Protocol SHA-256。
- Seed。
- Hardware。
- Software Environment。

---

## 14. 实施阶段与门禁

### Phase 0：Baseline Audit

不进行训练。

完成：

1. 核验 GC-001 三个 CLM 模型。
2. 核验 T1 参数统计。
3. 统一 Token 统计口径。
4. 核验历史训练时间。
5. 核验数据和 Tokenizer 身份。
6. 确认当前 KV Cache 缺失的实现原因。
7. 计算 Parameter-Matched Transformer 的候选配置。

输出：

`baseline-audit.json`

门禁：

`BASELINE_AUDIT_PASS`

### Phase 1：Transformer Reference Engineering

不进行新的模型训练。

完成：

1. 实现 KV Cache。
2. 实现增量 Decode。
3. 实现 Context Window。
4. 通过 KV Cache Equivalence。
5. 完成实际 KV Memory 测量。
6. 完成参数匹配配置搜索。

输出：

- `kv-cache-equivalence.json`
- `parameter-match-report.json`
- `inference-profile.json`

门禁：

`TRANSFORMER_REFERENCE_READY`

### Phase 2：Protocol Freeze

冻结：

- 模型结构。
- Tokenizer。
- Dataset。
- Training Budget。
- FLOPs Budget。
- Seed。
- Evaluation。
- Memory Benchmark。
- Statistical Analysis。

输出：

`protocol.sha256`

门禁：

`PROTOCOL_FROZEN`

### Phase 3：Pilot

只有在单独获得训练预算批准后才能启动。

目标：

验证训练和评估管线正确性。

不得根据 Pilot 结果临时改变正式协议。

如需调整，必须生成新的协议版本。

### Phase 4：Formal Comparison

按照 A、B、C、D 四条轨道执行。

每条轨道分别保存结果。

不得将不同控制条件的结果混在同一排行榜中。

### Phase 5：Research Decision

形成最终研究报告。

回答：

1. CLM 在相同参数量下的能力如何？
2. CLM 在相同 Token 预算下的能力如何？
3. CLM 在相同计算预算下的学习效率如何？
4. CLM 的持续状态是否具有实际记忆价值？
5. CLM 的推理和存储成本是否存在优势？
6. 当前 CLM 的主要限制来自哪里？

---

## 15. 验收标准

### 工程验收

- [ ] 所有模型参数统计通过。
- [ ] Tokenizer 和数据身份一致。
- [ ] Token 统计口径一致。
- [ ] Transformer KV Cache 正确性通过。
- [ ] FLOPs 统计能够复现。
- [ ] Persistent Memory 统计完整。
- [ ] 每个模型具有完整 Seed 和 Checkpoint 身份。
- [ ] 所有结果可以追溯到 Git Commit。

### 实验公平性验收

- [ ] Parameter-Matched 误差符合规定。
- [ ] Token-Matched 精确匹配。
- [ ] Compute-Matched 误差不超过 5%。
- [ ] 所有模型使用相同的评估数据。
- [ ] 没有 Test 数据泄漏。
- [ ] Transformer 使用优化后的推理实现。
- [ ] CLM State 与 Transformer KV Cache 均纳入存储统计。

### 研究验收

- [ ] 完成多 Seed 统计。
- [ ] 完成 Language Modeling 对照。
- [ ] 完成 Generation Quality 评价。
- [ ] 完成 Memory Benchmark。
- [ ] 完成 Training/Inference Efficiency 分析。
- [ ] 完成独立人工评审。
- [ ] 研究结论明确区分已验证、未验证和不确定事项。

---

## 16. 研究决策原则

本研究不预设 Transformer 或 CLM 获胜。

可能出现以下结果：

**情况 A：CLM 的 PPL 较低或相近，且计算成本更低。**

说明 CLM 在当前任务和预算下具有较好的语言建模效率。

**情况 B：CLM 的 PPL 较高，但记忆任务表现更好。**

这仍然可能是有价值的架构结果，尤其是当其使用更小的持久状态并取得更低的长期存储成本时。

**情况 C：Transformer 的 PPL 更低，CLM 的记忆能力也没有优势。**

需要重新评估 Cellular State 的设计和计算机制。

**情况 D：CLM 在某些记忆任务上更好，但训练成本较高。**

需要进一步研究局部计算、状态更新和模型规模之间的关系。

任何结果都必须基于相应轨道的证据，不得从单一指标推导全面结论。

---

## 17. 当前执行边界

本协议目前只授权研究设计和工程准备。

明确禁止：

- 自动启动训练。
- 自动扩大训练 Token 预算。
- 覆盖 GC-001 原始结果。
- 将 AI Review 标记为 Human Review。
- 将 Development Holdout 当作 Independent Test。
- 使用未验证的 KV Cache 结果进行性能结论。
- 未经批准改变正式训练预算。

Phase 0 和 Phase 1 可以先行开展。

Phase 2 完成后，提交正式运行计划、预计 GPU 小时、Token 预算和实验矩阵，等待单独批准。

---

## 18. 最终研究定位

MiniCells 研究的重点不是证明 Cellular 可以复制 Transformer。

我们需要验证的是：

**能否通过 Cellular State、局部计算和持续状态演化，建立一种不同于 Transformer Context 的语言建模与记忆机制，并在有限的计算和存储资源下取得有意义的能力。**

Transformer 为我们提供一个成熟的参照，使 CLM 的性能差距、计算成本和记忆优势能够被量化。

GC-001 主要建立了初步语言建模基线。

本次公平架构比较则需要建立可信的比较条件。

只有完成这一步，我们才能判断 CLM 当前的差距究竟来自参数规模、训练预算、计算效率，还是 Cellular 架构本身。

**研究主体始终是 CLM。Transformer 是测量 CLM 的参照工具。**
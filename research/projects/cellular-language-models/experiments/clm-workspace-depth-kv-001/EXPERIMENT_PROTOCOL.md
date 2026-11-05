# CLM-Workspace-DepthKV-001
## 计算深度索引的历史 KV：实验设计与实施协议

**Version:** 0.1

**Status:** Implementation Specification

**Repository:** `ArcheLabs/mini-cells`

**Previous Experiment:** `clm-workspace-controller-001`

**New Experiment ID:** `clm-workspace-depth-kv-001`

---

# 1. 研究目标

本实验研究一个明确的问题：

**在无固定 Cell 层级的 CLM 中，历史 Token 的 KV 是否应该按照计算深度组织，而不是让所有计算步骤共享同一份历史 KV？**

现有 CLM v0.4 使用 Shared Workspace：

- 每个历史 Token 只生成一份 KV。
- 所有计算步骤读取同一份历史 KV。
- 当前 Token 的表示不断变化，因此每次生成不同的 Query。
- Cell 完成计算后，不会更新历史 Workspace。

本实验提出一种替代方案：

**Depth-indexed Workspace：每个历史 Token 在不同计算深度形成独立的表示，并保存对应 KV。当前 Token 在计算深度 d 时，读取历史 Token 在计算深度 d 时形成的 KV。**

我们不改变 Transformer 的自回归语言建模目标，也不改变 CLM 的 Cell 计算机制。

本实验只验证历史信息交互方式是否应该与计算深度对应。

## 1.1 核心假设

H1：

在不同计算阶段，历史 Token 已经形成了不同的计算表示。使用与当前计算深度对应的历史表示，可能比反复读取同一份初始历史表示更有效。

H0：

单一 Shared Workspace 已经能够提供足够的历史信息。按照计算深度保存多份 KV 不会带来显著预测收益，或者其收益不足以抵消额外成本。

本实验不预设 H1 成立。

# 2. 实验范围与非目标

本实验只改变 KV 的组织方式。

以下内容全部保持不变：

- Transformer 自回归语言建模目标。
- W0 Causal Workspace Encoder。
- 8 个独立 SwiGLU Cells。
- 每个计算步骤 Top-2 Routing。
- 两个固定计算步骤。
- GRU Controller State。
- Cell 的输入、输出及 Residual 更新机制。
- Router 参数和训练方式。
- Tokenizer、数据集、训练预算和优化器。
- LM Head 和预测目标。

本实验不研究：

1. READ / COMPUTE / STOP 动态决策。
2. 动态计算深度。
3. Cell-specific KV。
4. 多视图 Workspace。
5. Cell 之间的连接拓扑。
6. Persistent Cell Memory。
7. 新的 Router 或 Controller。
8. Attention 的稀疏化、压缩或其他性能优化。

所有 Token 仍然执行固定的两个计算步骤，不引入提前停止。

# 3. 实验分组

只保留两个主要 CLM 实验组，以及现有 Transformer 对照组。

| 实验组 | 架构 | 历史 KV 机制 |
|---|---|---|
| T0-P | Dense Transformer | 标准逐层 KV |
| T0-F | Dense Transformer | 原有 FLOPs-matched 对照 |
| C2 | CLM v0.4 | 两个步骤共享同一份 KV |
| C-DKV | CLM v0.4 | 每个计算深度使用对应的 KV |

其中：

- C2 是现有完整 v0.4 基线。
- C-DKV 是本次唯一新增架构。
- T0-P 和 T0-F 沿用已有 Transformer 配置，不重新设计。

注意：T0-F 原本按照 C2 的计算量进行匹配。由于 C-DKV 会增加 KV 投影和存储成本，不能继续将 T0-F 称为 C-DKV 的 FLOPs-matched 对照。

本实验的主要因果比较是：

**C2 vs C-DKV。**

Transformer 组用于提供外部性能参照，不是本实验主要变量。

# 4. 现有 C2 架构基线

Codex 必须先从现有实现中核对并冻结以下结构。

| 组件 | 配置 |
|---|---|
| Hidden Dimension | 256 |
| Attention Heads | 4 |
| Workspace | One Shared Causal Workspace |
| Cells | 8 independent SwiGLU Cells |
| Routing | Dynamic Top-2 |
| Computation Steps | 2 |
| Controller | GRU |
| Controller Hidden Dimension | 256 |
| Workspace Read | 每个步骤读取一次 |
| STOP | Disabled |

以上为已确认的 v0.4 架构摘要。其他实现细节必须从当前代码和实验配置中读取，不得从其他历史 CLM 实验推测。

## 4.1 C2 的计算流程

对于当前 Token t：

初始表示：

\[
z_t^0=E(x_t)
\]

第一次信息交互：

\[
a_t^0=Attention(q_t^0,KV_{<t}^0)
\]

根据当前表示和 Controller State 执行 Top-2 Cell 计算：

\[
z_t^1=CellUpdate(z_t^0,a_t^0,s_t^0)
\]

第二次信息交互：

\[
a_t^1=Attention(q_t^1,KV_{<t}^0)
\]

再次执行 Cell 计算：

\[
z_t^2=CellUpdate(z_t^1,a_t^1,s_t^1)
\]

最终：

\[
P(x_{t+1})=LMHead(z_t^2)
\]

关键特征：

\[
\boxed{KV^0=KV^0}
\]

两个计算步骤读取同一份 Workspace。

历史 KV 不会因为 Cell 计算而更新。

# 5. 新架构：C-DKV

## 5.1 核心设计

C-DKV 保留现有 C2 的全部计算结构。

唯一变化：

**第二次信息交互读取历史 Token 在第一次 Cell 计算完成后形成的表示。**

我们引入两个计算深度：

- Depth 0：初始 Workspace 表示。
- Depth 1：完成第一次 Cell 计算后的表示。

对应：

\[
KV_j^0=ProjectKV(W0_j)
\]

\[
KV_j^1=ProjectKV(z_j^1)
\]

其中 \(z_j^1\) 是历史 Token j 完成第一次 Cell 计算后的统一隐藏表示。

当前 Token 的计算：

第一步：

\[
a_t^0=Attention(q_t^0,KV_{\leq t}^0)
\]

\[
z_t^1=CellUpdate(z_t^0,a_t^0,s_t^0)
\]

第二步：

\[
a_t^1=Attention(q_t^1,KV_{\leq t}^1)
\]

\[
z_t^2=CellUpdate(z_t^1,a_t^1,s_t^1)
\]

最终：

\[
P(x_{t+1})=LMHead(z_t^2)
\]

因此，两种架构的唯一区别是：

| 计算步骤 | C2 | C-DKV |
|---|---|---|
| Step 0 | Read KV⁰ | Read KV⁰ |
| Cell Update 0 | 相同 | 相同 |
| Step 1 | Read KV⁰ | Read KV¹ |
| Cell Update 1 | 相同 | 相同 |

C-DKV 不需要保存 KV²，因为模型只有两个信息交互步骤，没有第三次 Attention。

## 5.2 KV 投影规则

为了避免引入额外参数，本实验要求：

**Depth 0 和 Depth 1 使用相同的 K/V 投影参数。**

即：

\[
K_j^0=W_K W0_j
\]

\[
V_j^0=W_V W0_j
\]

\[
K_j^1=W_K z_j^1
\]

\[
V_j^1=W_V z_j^1
\]

其中 \(W_K,W_V\) 在两个深度之间共享。

Query 投影、Attention Head、RoPE 及其他已有机制，全部沿用 C2 的原始实现。

不得为不同深度增加独立 K/V Projection。

不得增加 Depth Embedding、额外 Adapter 或新的可训练模块。

这样，C2 和 C-DKV 的可训练参数数量应当完全一致。

如果当前 W0 输出和 Cell Hidden 的维度无法直接兼容现有 K/V 投影，必须先报告具体 Tensor Shape 和原因，不得自行添加 Adapter 解决。

## 5.3 Cell 表示空间

这是本实验必须明确验证的一个问题。

假设：

\[
z_A^2=C_2(C_1(z_A^0))
\]

\[
z_B^2=C_5(C_2(z_B^0))
\]

两个 Token 虽然都处于 Depth 2，但其 Cell 路径可能不同。

我们希望不同 Cell 的输出仍然能够映射到统一的 Hidden Representation Space。

本实验必须沿用现有 Cell 的输出维度、Residual 和 Normalization 机制。

不增加 Cell-specific Projection。

如果现有实现不能保证不同 Cell 的输出进入同一个 Hidden Stream，停止实施并报告，不得自行创造新的对齐架构。

不同 Cell 路径产生的表示是否具有足够的 Attention 兼容性，正是 C-DKV 需要验证的内容。

# 6. KV 写入与因果顺序

这是实现中最重要的部分。

## 6.1 Depth 0

历史 Token 经过现有 W0 Causal Workspace Encoder。

生成：

\[
KV_j^0
\]

其计算和写入方式完全沿用 C2。

## 6.2 Depth 1

历史 Token 完成第一次 Cell 计算后：

\[
z_j^1=CellUpdate(z_j^0,a_j^0,s_j^0)
\]

随后生成：

\[
KV_j^1=ProjectKV(z_j^1)
\]

这里的 \(z_j^1\) 必须是当前实现中传递给第二个计算步骤的 Token Hidden Representation。

GRU Controller State 仍然独立保存并按照 C2 的原有方式更新。除非现有 Workspace 明确将其包含在 KV 输入中，否则不得将 Controller State 拼接进 KV。

注意：

Cell 不直接写入或修改 Workspace。

KV¹ 是模型执行第一次计算后，由运行时根据统一 Hidden Representation 生成的历史表示。

## 6.3 Causal Mask

Depth 0 和 Depth 1 均必须遵循现有 C2 的因果 Mask 约定。

不允许读取未来 Token：

\[
j>t \Rightarrow Attention(t,j)=0
\]

对于当前位置是否允许读取自身，严格沿用 C2 现有的对角线 Mask 语义，不得在两个架构之间改变。

Depth 1 KV 必须在对应 Token 完成第一次计算后才能生成。

严禁使用未来 Token 的表示生成当前 Token 的 KV。

# 7. 训练阶段的计算实现

训练仍然采用现有的 Teacher Forcing 训练方式。

C-DKV 应实现两个因果计算阶段。

### Stage 0

1. 计算所有 Token 的 W0 表示。
2. 生成 KV⁰。
3. 使用因果 Attention 读取 KV⁰。
4. 执行第一次 Top-2 Cell Routing。
5. 更新 GRU Controller State。
6. 计算所有 Token 的 Depth 1 表示 \(z^1\)。

### Stage 1

1. 根据所有 Token 的 \(z^1\) 生成 KV¹。
2. 当前 Token 根据 \(z_t^1\) 生成第二次 Query。
3. 使用因果 Mask 读取对应历史 KV¹。
4. 执行第二次 Top-2 Cell Routing。
5. 更新 Controller State。
6. 得到最终表示 \(z^2\)。
7. 计算 Next-Token Prediction Loss。

必须保证：

\[
KV_j^1
\]

只依赖于 Token j 及其之前的历史。

两个阶段可以使用批量矩阵计算，但不得为了并行化而破坏因果关系。

# 8. 推理阶段的计算实现

必须实现真正的增量 Decode，而不只是训练前向。

处理 Token t 时：

1. 生成当前 Token 的 W0 表示。
2. 生成并缓存当前 Token 的 KV⁰。
3. 使用历史 KV⁰ 执行第一次 Attention。
4. 执行第一次 Cell 计算。
5. 生成当前 Token 的 KV¹。
6. 使用历史 KV¹ 执行第二次 Attention。
7. 执行第二次 Cell 计算。
8. 输出下一个 Token 的概率分布。

每个序列分别维护：

- KV Cache 0。
- KV Cache 1。
- 必要的 Controller State。

缓存必须支持：

- 新序列初始化。
- Batch 内多个独立序列。
- Batch 重排。
- 序列结束和缓存释放。
- 连续自回归生成。

必须保证 Teacher Forcing 和 Incremental Decode 的计算结果一致，允许合理的浮点误差。

# 9. 公平性与变量控制

C2 与 C-DKV 必须满足：

| 项目 | 要求 |
|---|---|
| Trainable Parameters | 完全一致 |
| Cell 数量 | 8 |
| Cell 类型 | SwiGLU |
| Routing | Top-2 |
| Computation Steps | 2 |
| Controller | 相同 GRU |
| Hidden Dimension | 256 |
| Attention Heads | 4 |
| W0 | 完全相同 |
| K/V Projection | 跨深度共享参数 |
| Query Projection | 沿用原实现 |
| Optimizer | 完全相同 |
| Training Tokens | 完全相同 |
| Data Order | 配对一致 |
| Loss | 完全相同 |

唯一允许的架构差异：

**第二个 Attention 步骤所读取的历史 KV 来源。**

C2 读取 KV⁰。

C-DKV 读取 KV¹。

不得通过修改 Cell 宽度、Router、训练步数、Loss 权重等方式补偿实验结果。

# 10. 数据集与训练配置

沿用 `clm-workspace-controller-001` 的原始实验配置。

已确认的 Pilot 设置：

| 项目 | 配置 |
|---|---|
| Dataset | TinyStories |
| Vocabulary | 2,048 |
| Context Length | 256 |
| Training Budget | 2,000,000 targets/model |
| Pilot Updates | 1,068 |
| Validation | 32 documents / 5,997 targets |
| Precision | FP32 |
| Hardware | 2 × NVIDIA T4 |
| Pilot Seed | 17 |

上述配置用于复现既有实验。

正式实验使用三个全新的 Model Seeds，建议：

```text
18001
18002
18003
```

Seed 17 仅属于此前的探索性 Pilot，不得作为新的 Confirmatory Seed。

同一 Seed 下，C2 与 C-DKV 应使用：

- 相同的数据顺序。
- 相同的公共参数初始化。
- 相同的训练预算。
- 相同的验证样本。

其他训练超参数，包括 Batch Size、Learning Rate、Scheduler、AdamW 配置、Weight Decay、Gradient Clipping、初始化方式和 Dropout，必须从原始实验配置及代码中提取并冻结。

不得根据其他 CLM 实验的配置自行补齐。

如果无法恢复原始配置，先报告缺失项，不得开始正式训练。

# 11. 现有 Pilot 结果

此前 Seed 17 Pilot 的结果如下：

| 模型 | Validation NLL | PPL |
|---|---:|---:|
| T0-P | 3.1401 | 23.11 |
| T0-F | 3.1727 | 23.87 |
| C2 | 3.2031 | 24.61 |

T0-P 为 Parameter-matched Transformer，T0-F 为 FLOPs-matched Transformer。

该 Pilot 每个模型训练 2M targets，六个模型均完成，使用 FP32 和两张 T4，耗时约 37.2 分钟。

这些结果仅作为历史参考，属于探索性结果，不构成正式的架构优劣结论。

本实验应重点报告 C2 与 C-DKV 的配对差异。

# 12. 必须实现的测试

## 12.1 C2 Baseline Regression

确保现有 C2 代码路径不发生改变。

使用原有配置时，输出应与修改前保持一致。

不得因为新增 Depth-indexed 机制而改变 C2 的计算图。

## 12.2 Causal Correctness

构造两组输入：

- 输入 A：正常序列。
- 输入 B：只修改未来 Token。

比较两组输出。

对于未来 Token 修改位置之前的所有预测：

\[
Logits_A[:,0:t]=Logits_B[:,0:t]
\]

允许数值计算误差，但不得出现未来信息泄漏。

## 12.3 Depth 1 KV Correctness

验证：

\[
KV_j^1=ProjectKV(z_j^1)
\]

必须确认：

- KV¹ 来自第一次 Cell 计算之后。
- 不来自初始 Embedding。
- 不来自最终 LM Head 表示。
- 不包含未来 Token 信息。
- 不会在第一次 Cell 计算完成前写入。

## 12.4 Routing Correctness

确保：

- 两个步骤均执行 Top-2。
- Router 输入沿用原有实现。
- GRU Controller State 更新顺序不变。
- 不增加额外 Routing。
- 不增加 STOP。

## 12.5 Cell Path Compatibility

测试不同 Cell 路径下的输出：

- Hidden Dimension 一致。
- KV Tensor Shape 一致。
- 无 NaN / Inf。
- Attention 输出有限。
- 不同路径下的表示范数保持可控。

记录不同 Cell 路径的表示统计，但不得通过额外 Adapter 修复结果。

## 12.6 Gradient Correctness

验证第二步的 Loss 可以正常反向传播到：

- 第二步 Cell。
- 第一步 Cell。
- 第一步 Router。
- GRU Controller。
- KV¹ 所依赖的 Hidden Representation。

不得意外 detach Depth 1 表示。

## 12.7 Training / Inference Parity

对相同 Token 序列：

- 使用 Teacher Forcing 计算。
- 使用逐 Token Incremental Decode 计算。

比较对应位置的 Logits。

两种模式必须保持一致。

## 12.8 Cache Isolation

测试不同序列之间：

- KV⁰ 不混合。
- KV¹ 不混合。
- Controller State 不混合。
- Reset 后不存在上一序列的残留状态。

# 13. 性能评估

本实验不能只报告 PPL。

Depth-indexed KV 会增加历史表示的存储成本，因此必须同时评价质量和推理成本。

## 13.1 模型质量

- Validation NLL。
- Validation PPL。
- 每个 Seed 的独立结果。
- C-DKV 相对 C2 的 NLL 差值。
- 三个 Seed 的平均值和离散程度。

定义：

\[
\Delta NLL=NLL_{C-DKV}-NLL_{C2}
\]

负值表示 C-DKV 改善预测质量。

## 13.2 计算成本

分别报告：

- Forward FLOPs/token。
- KV Projection FLOPs。
- Attention FLOPs。
- Cell FLOPs。
- Controller FLOPs。
- Training Tokens/s。

不能只用理论 FLOPs 代替实际 GPU 性能。

## 13.3 推理成本

必须分别测试：

| 指标 | 说明 |
|---|---|
| TTFT | 首 Token 延迟 |
| Prefill Latency | 输入上下文处理时间 |
| TPOT | 每个生成 Token 的时间 |
| Decode tok/s | 实际生成吞吐 |
| Peak VRAM | 峰值显存 |
| KV Cache Bytes | KV 实际存储量 |

测试 Context：

- 128 Tokens
- 256 Tokens

测试 Batch：

- Batch 1
- Batch 8

Decode 测试使用固定 Prompt 和相同的生成长度，确保总序列长度不超过 256。

记录 P50、P95 延迟。

所有计时必须进行 CUDA Synchronization，并包含 Warmup。

## 13.4 预期成本变化

理论上：

C2：

\[
Memory_{KV}=O(Td)
\]

C-DKV：

\[
Memory_{KV}=O(2Td)
\]

因此 C-DKV 的 KV 存储量预计接近两份历史 KV。

同时，C-DKV 增加了 Depth 1 KV 的生成成本。

必须如实报告这些额外成本，不能将 KV 的增加解释为推理效率提升。

# 14. 实施步骤

## Phase A：Repository Audit

Codex 首先检查：

- 当前 Git Branch 和 HEAD。
- `clm-workspace-controller-001` 的代码。
- C2 模型定义。
- W0 Workspace Encoder。
- KV Cache 实现。
- Cell Router。
- GRU Controller。
- 训练配置。
- Transformer T0-P/T0-F 实现。
- 原始 Pilot Manifest 和 Metrics。

输出：

`BASELINE_AUDIT.md`

必须明确列出：

1. C2 实际计算流程。
2. KV⁰ 的来源。
3. 第二次 Query 的输入。
4. Cell 输出 Tensor Shape。
5. KV 投影权重。
6. 完整训练超参数。
7. 原始数据和 Tokenizer 的身份及 Hash。

如果发现当前实现与本协议中的基线摘要不一致，先报告，不得直接修改架构。

## Phase B：Implementation

新增实验配置：

```yaml
experiment_id: clm-workspace-depth-kv-001

architecture:
  baseline: C2
  workspace_kv_mode: depth_indexed
  computation_steps: 2
  cell_count: 8
  top_k: 2
  controller: gru

workspace:
  depth_0: initial_workspace_representation
  depth_1: post_first_cell_representation
  kv_projection: shared
```

实际字段名称可以按照现有代码风格调整。

要求：

- C2 继续保留原有 Shared KV 模式。
- 新增 C-DKV 模式。
- 两种模式尽可能复用同一执行代码。
- 不复制或重写整个模型。
- 不改变现有默认配置。
- 不破坏既有实验入口。

## Phase C：Engineering Smoke

先使用原有 Smoke 流程。

检查：

- 前向通过。
- 反向通过。
- Causal Mask 正确。
- Incremental Decode 正确。
- Cache Reset 正确。
- 无 NaN / Inf。
- 参数量一致。
- C2 Regression 通过。

Smoke 结果不得作为科学结论。

## Phase D：Formal Training

运行：

- C2。
- C-DKV。
- T0-P。
- T0-F。

使用三个预先冻结的新 Seed：

```text
18001
18002
18003
```

每个模型使用相同的原始训练预算。

不允许根据中间验证结果修改模型或提前终止某个实验组。

如果正式运行中发现实现错误，停止本轮，修复后重新开始受影响的配对实验。

## Phase E：Inference Benchmark

所有训练完成后：

- 加载相应 Checkpoint。
- 使用相同 Tokenizer。
- 使用相同 Prompt。
- 使用相同 Batch。
- 使用相同 Context。
- 使用相同生成长度。

分别测试 Prefill 和 Decode。

不能用训练吞吐代替推理性能。

## Phase F：Analysis

输出：

1. C2 vs C-DKV 的配对 NLL 比较。
2. 三个 Seed 的独立结果。
3. PPL 和 NLL 汇总。
4. 参数量和 FLOPs。
5. KV Cache 内存对比。
6. Prefill 和 Decode 延迟。
7. Cell Routing 分布。
8. Attention 统计。
9. 训练和推理一致性报告。

# 15. 结果与交付物

遵循仓库现有实验产物组织方式，建立独立实验目录，不覆盖原有结果。

至少交付：

```text
clm-workspace-depth-kv-001/
    README.md
    BASELINE_AUDIT.md
    EXPERIMENT_PROTOCOL.md
    CONFIG.yaml

    metrics/
        seed-18001.json
        seed-18002.json
        seed-18003.json
        aggregate.json

    benchmarks/
        prefill.json
        decode.json
        memory.json
        compute.json

    tests/
        causal_correctness.json
        cache_correctness.json
        training_inference_parity.json
        regression.json

    report/
        RESULTS.md
        comparison.csv
```

具体目录可以遵循仓库既有约定，但上述信息必须完整保留。

不得提交模型 Checkpoint、大型原始数据分片或其他不必要的大型产物。

# 16. 最终判断标准

本实验最终只回答一个问题：

**对于无层级的 Cellular Computation，历史信息按照计算深度组织，是否比所有计算步骤共享同一份 Workspace 更有效？**

结果分为三类：

### A. Depth-indexed 优于 Shared Workspace

如果 C-DKV 在多个 Seed 上稳定改善 NLL/PPL，且额外存储和计算成本具有合理性，则保留 Depth-indexed Workspace。

### B. 两者预测质量没有明确差异

如果差异不显著，而 C-DKV 需要额外 KV 存储，则优先保留更简单的 C2 Shared Workspace。

### C. Depth-indexed 明显更差

如果 C-DKV 的预测质量下降，重点检查：

- 不同 Cell 路径产生的表示兼容性。
- Depth 1 KV 的信息质量。
- Attention 分布。
- Router 与 Controller 的训练稳定性。

不得通过增加 Adapter、独立 Cell KV 或额外表示层来挽救本轮结果。这些属于新的架构假设，需要另行论证。

---

# 17. Codex 实施约束

1. 先完成 Repository Audit，再开始修改。
2. 不得将现有 C1 当作新架构；新架构名称必须是 C-DKV。
3. C2 必须保持原样，作为严格基线。
4. 不允许引入任何本协议未定义的架构模块。
5. 不允许修改 Cell、Router、GRU Controller 或训练目标。
6. 不允许使用未来 Token 信息。
7. 不允许为了修复不同 Cell 路径而偷偷增加 Adapter。
8. 不允许只报告 FLOPs 而忽略真实推理延迟。
9. 不允许将单 Seed 或 Smoke 结果描述为确认性结论。
10. 所有配置、Seed、Checkpoint、指标和代码版本必须可追溯。

**最终交付目标：完成 C-DKV 的实现、正确性测试、与 C2 的配对训练、Transformer 对照以及完整推理性能报告。**

本实验完成后，不自动进入新的架构设计。先根据结果判断 Depth-indexed Workspace 是否值得保留。

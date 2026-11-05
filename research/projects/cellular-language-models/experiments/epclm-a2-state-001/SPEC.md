# EPCLM-A2-State-001
# Persistent State Interaction Experiments

**项目：** MiniCells / Cellular Language Models
**仓库：** `ArcheLabs/mini-cells`
**实验阶段：** EPCLM Architecture Research — Stage 1
**状态：** Proposed
**实验基准：** A2 — Minimal CLM
**训练预算：** 每个候选 1M Tokens
**实验组：** B1 / B2 / B3

---

# 1. 研究目标

## 1.1 背景

在 EPCLM-X4C-001 中，我们完成了 X4 的计算深度消融。

实验发现：

- X4 原始结构为三个 Stage，每个 Stage 执行四轮 Attention 和 FFN。
- A2 将每个 Stage 减少为一轮 Attention 和一轮 FFN。
- A2 的 Validation PPL 为 51.397。
- A2 的解析前向 FLOPs 约为 3.855M / Token。

RCLM 的重复 FFN 实验虽然获得了更好的语言建模结果，但其改善依赖额外的重复计算。

因此，本研究不再继续优化重复 FFN，也不把 RCLM 作为后续架构基础。

我们将 A2 确立为新的基础 CLM。

本次实验的核心问题是：

**Persistent State 应该如何与 Attention 和 FFN 交互，才能改善语言计算，而不是简单地增加模型参数或计算量？**

## 1.2 核心研究假设

我们将三个计算要素明确区分：

| 机制 | 主要职责 |
|---|---|
| Attention | 内容寻址、Token 信息检索与交互 |
| FFN | 非线性特征变换与计算 |
| State | 保存、更新和提供持续演化的内部信息 |

我们希望验证：

1. State 是否能够成为 Attention 的持久化信息来源？
2. State 是否能够为 FFN 提供持续的计算上下文？
3. 同时与 Attention 和 FFN 交互，是否比单独交互产生更好的结果？

本次实验不预先认定哪一种机制更好。

## 1.3 本次实验范围

只实施：

- B1：State ↔ Attention。
- B2：State ↔ FFN。
- B3：State ↔ Attention + FFN。

不实施：

- Cell-specific 独立计算参数。
- Top-k Routing。
- Cell Growth / Mitosis。
- 动态拓扑。
- 增加 Attention 或 FFN 迭代次数。
- 在线权重学习。
- Flat State 与 Cellular State 的完整比较。

本次使用 16 个 Cell，每个 Cell 128 维状态，共享 State Read / Write 和 Update 参数。

这次实验验证的是 Cellular State 的交互机制，不是证明独立 Cell 计算能力已经成立。

---

# 2. A2 基准冻结

## 2.1 历史实验记录

本次使用 EPCLM-X4C-001 的 A2 作为历史对照。

| 项目 | 历史记录 |
|---|---|
| Run ID | `epclm-x4c-001-bc1288d34748-b8cc5876` |
| Source Commit | `bc1288d34748038455b2e9b441095821da6d5bc3` |
| Protocol SHA256 | `0958e1282974b06d95d09d732799cc0c7c0932d56a9177d9b51fc9e11b580496` |
| Seed | 91001 |
| Profile | dev |
| Match | tokens |
| Training Tokens | 1,000,000 |
| Optimizer Updates | 1,000 |
| GPU | Tesla T4 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Precision | float16 autocast + FP32 parameters + GradScaler |

原始实验分支：

`kaggle/epclm-x4c-001-bc1288d34748-b8cc5876`

## 2.2 A2 模型配置

| 参数 | 配置 |
|---|---|
| Stages | 3 |
| Attention / Stage | 1 |
| FFN / Stage | 1 |
| Attention Windows | 8 / 32 / 128 |
| Hidden Dimension | 188 |
| Attention Heads | 4 |
| FFN | SwiGLU |
| FFN Hidden Dimension | 639 |
| Position | RoPE |
| Normalization | RMSNorm |
| Persistent State | None |

A2 历史结果：

| 指标 | A2 |
|---|---:|
| Validation PPL | 51.397 |
| Forward FLOPs / Token | 3.855M |

A2 是本次唯一的历史主基线。

Codex 必须从原始实验产物中定位 A2 的模型配置、初始权重、最终 Checkpoint、评估数据和训练记录。

不能根据上述摘要重新猜测模型参数或训练超参数。

## 2.3 数据身份冻结

新实验必须复用原实验的数据资产。

| 数据资产 | SHA256 |
|---|---|
| Dataset Manifest | `dfc7eeee33aaab5693a43bf72c88a991ee572b84cc93a5793e6901bf2c9cb6dc` |
| Tokenizer | `becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2` |
| Training Data | `3c8edf7a559e2e677be15ecc25eb5da90c19096bad82ab4e7a0c28b44b426c93` |
| Validation Data | `4a8c15e6a53657488c78209e37fdd98e67e4e159875946cce1fc8663221f8d8f` |

必须重新计算 Hash 并验证。

任何 Hash 不一致，都必须停止正式实验并记录原因。

Independent Test 不得访问。

## 2.4 初始化与公平性

B1、B2、B3 均从头开始训练，不得直接从已经训练完成的 A2 Checkpoint 继续训练。

基础 A2 参数必须使用与历史 A2 相同的初始化。

新增 State 模块使用独立、确定性的随机初始化，不得改变基础 A2 参数的初始化结果。

三个实验组使用相同的基础参数初始化、训练数据顺序、优化器配置和随机种子。

如果无法恢复历史 A2 的初始参数，必须先使用原始初始化逻辑重建，并验证基础模型参数 Hash。

无法验证时，应标记为初始化未完全匹配，不得宣称严格复现。

---

# 3. 统一的 State 架构

## 3.1 Cellular State Bank

所有实验组采用同一种 State Bank。

形状：

`[Batch, 16, 128]`

其中：

- Cell 数量：16。
- 每个 Cell 状态维度：128。
- State 总维度：2048。
- 每个独立序列拥有自己的 State。
- 不同 Batch 样本之间完全隔离。

每个 Cell 只拥有独立的动态 State。

所有 Cell 共享：

- State Read 参数。
- State Write 参数。
- State Update 参数。

允许使用 16 个独立的可学习初始 Slot Embedding，使不同 Cell 能够形成不同的状态表示。

但不允许为每个 Cell 增加完整独立 FFN。

## 3.2 State 初始化

每个独立序列开始时：

\[
S_0=E_{\text{slot}}
\]

其中：

\[
E_{\text{slot}}\in\mathbb R^{16\times128}
\]

训练和推理必须使用相同的初始化规则。

同一个序列内，State 跨 Chunk 保留。

独立序列之间，State 必须重置。

## 3.3 State Read

采用内容寻址机制：

\[
r_t=\operatorname{Softmax}
\left(
\frac{q(h_t)K(S)^T}{\sqrt{d_k}}
\right)V(S)
\]

其中：

- Query 来自当前隐藏表示。
- Key 来自 16 个 Cell State。
- Value 来自 16 个 Cell State。
- 输出维度为 188，与 A2 Hidden Dimension 一致。

第一版使用单头 State Read，不增加复杂多头结构。

对于 B1，优先复用原始 Attention 的 Q/K/V 投影，将 State 作为额外的 Key/Value Memory Slots。

对于 B2，使用相同的 State Read 投影机制，在 FFN 输入路径生成 State Context。

尽可能共享 State Projection 和 Read 参数，避免 B1、B2、B3 因不同参数规模而失去可比性。

## 3.4 State Write

State Write 必须有明确的信息来源。

对于每个 Chunk：

- 从允许参与 State Write 的模块输出中提取表示。
- 对 Token 和 Stage 进行 Masked Mean Pooling。
- 形成 188 维 Chunk Summary。
- 使用共享的 Cell Update 网络更新 16 个 Cell。

建议采用共享 GRUCell：

\[
\tilde s_i=GRUCell(p_b,s_i)
\]

其中 \(p_b\) 是 Chunk Summary。

更新后的 State：

\[
S_{b+1}=\{\tilde s_1,\ldots,\tilde s_{16}\}
\]

State Update 使用 FP32 计算，必要时在 Read 时转换为模型计算精度。

不得使用未来 Chunk 的信息更新当前 State。

---

# 4. Chunk 与因果语义

## 4.1 Chunk Size

固定：

`Chunk Size = 32 Tokens`

如果历史 A2 的训练序列长度为 125，则每个序列分成：

- Chunk 1：32 Tokens。
- Chunk 2：32 Tokens。
- Chunk 3：32 Tokens。
- Chunk 4：29 Tokens。

必须正确处理最后一个不完整 Chunk。

## 4.2 Attention Cache

Chunk 划分不得改变 A2 原有 Attention 语义。

尤其是 Stage 3 的 128 Token Attention Window。

必须跨 Chunk 保存必要的 KV Cache，使其在相同输入下与原始完整序列 Causal Attention 等价。

不同 Stage 的 KV Cache 独立维护。

不能因为引入 State，就将 Attention Window 限制为 32。

State 与 KV Cache 是两种不同的机制：

- KV Cache 保存 Attention 所需的 Token 表示。
- Persistent State 保存持续演化的 Cell 状态。

## 4.3 State 更新时序

每个 Chunk 开始时读取：

\[
S_b
\]

Chunk 内所有 Token 使用同一版本的 State。

Chunk 完成后：

\[
S_{b+1}=Update(S_b,H_b)
\]

下一 Chunk 才能读取 \(S_{b+1}\)。

同一个 Chunk 内不允许先处理部分 Token、提前更新 State，再让后续 Token 读取该更新。

这次采用 Block-level State Evolution，不实现逐 Token State Update。

对于 Chunk 边界上的 Next-token Prediction，严格沿用原始自回归目标位移约定。State 在 Chunk 完成后更新，供后续 Chunk 的计算使用，不增加特殊的边界重算路径。

## 4.4 训练数据边界

必须检查原始数据是否包含文档边界和独立样本信息。

如果历史训练使用独立的 125 Token 样本：

- State 在样本开始时重置。
- State 仅在该样本的四个 Chunk 之间保留。
- 不允许在不同样本之间传播 State。

本次 1M 实验不改变原始数据采样协议。

特别注意：

如果 State 只在 125 Token 样本内部持续，它并没有证明模型获得了超过 128 Token 的长期记忆能力。

本次主要验证 State 的计算交互机制。

真正跨越多个 128 Token 上下文窗口的长期状态能力，需要后续独立的连续文档流实验。

---

# 5. B1 — State ↔ Attention

## 5.1 研究假设

**H1：Persistent State 可以成为 Attention 的额外信息来源，使 Attention 能够检索此前计算形成的内部状态。**

B1 不允许 FFN 直接读取 State。

## 5.2 信息流

原始 A2：

\[
Attention(Q,K_{\text{token}},V_{\text{token}})
\]

B1：

\[
Attention(
Q,
[K_{\text{token}},K_{\text{state}}],
[V_{\text{token}},V_{\text{state}}]
)
\]

其中 State 作为额外的 Memory Slots。

Token Attention 的原始因果 Mask 保持不变。

State Slots 对当前 Chunk 的所有 Token 可见，因为它们只包含之前 Chunk 已经形成的状态。

## 5.3 State Write

B1 只能从 Attention 路径获得 State Write 信息。

定义：

\[
p_A=\operatorname{MeanPool}(H_A)
\]

其中 \(H_A\) 是所有 Stage 在当前 Chunk 的 Attention 输出。

随后：

\[
S_{b+1}=Update(S_b,p_A)
\]

FFN 输出不直接参与 State Write。

需要注意：State 修改 Attention 输出后，后续 FFN 会自然接收到改变后的隐藏表示。这属于正常计算依赖，不代表 FFN 获得了独立 State Read 能力。

## 5.4 必须验证

- State 是否被 Attention 有效读取？
- State 是否改变 Attention Distribution？
- 不同 Cell 是否具有不同的读取权重？
- State 更新是否产生有效的跨 Chunk 信息？
- 关闭 State Read 后，语言预测是否明显变化？

---

# 6. B2 — State ↔ FFN

## 6.1 研究假设

**H2：Persistent State 可以为 FFN 提供持续计算上下文，使 FFN 在执行非线性特征变换时能够利用已有状态。**

B2 的原始 Token Attention 必须保持不变。

## 6.2 信息流

原始 A2：

\[
h_A\rightarrow FFN(h_A)
\]

B2：

\[
r_F=StateRead(h_A,S_b)
\]

\[
\tilde h_A=h_A+g\cdot r_F
\]

\[
h_F=FFN(\tilde h_A)
\]

其中：

- State Read 只在 FFN 输入路径执行。
- \(g\) 为可学习门控。
- 原始 FFN 内部结构、参数和残差语义保持不变。
- 不得让 State 进入原始 Token Attention 的 K/V。

State Read 必须在原始 FFN 的正确输入位置集成，不能破坏 A2 原有 Pre-Norm / Post-Norm 结构。

## 6.3 State Write

B2 只能从 FFN 路径获得 State Write 信息：

\[
p_F=\operatorname{MeanPool}(H_F)
\]

\[
S_{b+1}=Update(S_b,p_F)
\]

Attention 输出不直接参与 State Write。

Attention 仍然是 FFN 的上游输入来源，但不存在独立的 State-Attention 交互。

## 6.4 必须验证

- State 是否改变 FFN 的计算结果？
- FFN 是否有效利用了 State？
- State 是否保留了此前 FFN 形成的计算结果？
- 关闭 State Read 后，预测是否退化？
- 固定 State 与持续更新 State 是否存在明显差异？

---

# 7. B3 — State ↔ Attention + FFN

## 7.1 研究假设

**H3：State 同时参与 Attention 和 FFN，能够形成比单独交互更完整的持续计算机制。**

B3 同时采用 B1 和 B2 的 State Read。

## 7.2 信息流

Attention：

\[
h_A=Attention(Q,K_{\text{token}}\cup K_{\text{state}},V_{\text{token}}\cup V_{\text{state}})
\]

FFN：

\[
r_F=StateRead(h_A,S_b)
\]

\[
h_F=FFN(h_A+g\cdot r_F)
\]

其中两个模块读取同一个冻结版本：

\[
S_b
\]

Attention 完成后不得提前更新 State，再让 FFN 读取更新后的版本。

这样可以确保一个 Chunk 内的 State Read 都基于相同的历史状态。

## 7.3 State Write

B3 同时从两个模块获得更新信息：

\[
p_A=\operatorname{MeanPool}(H_A)
\]

\[
p_F=\operatorname{MeanPool}(H_F)
\]

\[
p_B=\frac{p_A+p_F}{2}
\]

\[
S_{b+1}=Update(S_b,p_B)
\]

使用与 B1、B2 相同的 Writer 和 Update 参数。

## 7.4 必须验证

- 两条交互路径是否都具有独立贡献？
- Attention 和 FFN 是否形成互补的信息利用？
- B3 是否优于 B1 和 B2？
- B3 的改善是否值得其额外计算成本？
- 是否存在其中一条路径被模型忽略的情况？

---

# 8. 三组实验的严格比较

| 维度 | B1 | B2 | B3 |
|---|---|---|---|
| State Read → Attention | 是 | 否 | 是 |
| State Read → FFN | 否 | 是 | 是 |
| Attention → State Write | 是 | 否 | 是 |
| FFN → State Write | 否 | 是 | 是 |
| State Bank | 16×128 | 16×128 | 16×128 |
| Shared Update | 是 | 是 | 是 |
| Chunk Size | 32 | 32 | 32 |
| Attention Iterations | 3 | 3 | 3 |
| FFN Iterations | 3 | 3 | 3 |

注意：这里的 3 次 Attention 和 3 次 FFN 指三个 Stage 各执行一次，不是三轮重复迭代。

B1、B2、B3 必须尽可能共享相同的可训练 State 模块参数规模。

B3 由于存在两次 State Read，可能具有更高计算成本。必须如实统计，不允许为了获得相同 FLOPs 而关闭其某条交互路径。

---

# 9. 训练协议

## 9.1 训练预算

每个模型：

- 1,000,000 Training Tokens。
- 1,000 Optimizer Updates。
- Seed 91001。
- Profile dev。
- Match tokens。

按原始 A2 的配置执行。

如果原始记录确认 Batch Size 8、Sequence Length 125，则保持不变。

所有优化器、学习率、Warmup、Cosine Scheduler、Gradient Clipping、Weight Decay 等配置，必须直接从 A2 的真实配置文件读取。

不得重新猜测。

## 9.2 训练顺序

B1、B2、B3 分别独立训练。

三个实验组必须：

1. 使用相同基础 A2 初始化。
2. 使用相同训练数据顺序。
3. 使用相同有效训练 Token 数。
4. 使用相同优化器配置。
5. 使用相同验证集。
6. 使用相同评估代码。

State 模块使用独立、确定性的初始化 RNG。

不得因创建 State 模块而改变基础 A2 权重初始化或数据采样 RNG。

## 9.3 梯度传播

同一独立序列内的 State Update 必须保留必要的梯度关系。

本次每个 125 Token 样本最多包含四个 Chunk，可以在样本内部保留跨 Chunk 的计算图。

样本结束后重置 State，不跨独立样本传播梯度。

必须记录：

- State Update 的梯度范数。
- State Read 投影的梯度范数。
- 每个交互路径的梯度。
- 是否出现梯度消失或数值异常。

不得在每次 Chunk 更新 State 后立即无条件 Detach。

## 9.4 精度

优先沿用历史环境：

- FP32 Model Parameters。
- FP16 Autocast。
- GradScaler。
- FP32 Persistent State Update。

如果硬件或框架版本不同，必须单独记录，不得隐瞒环境变化。

---

# 10. State 消融与机制诊断

除完整 B1/B2/B3 训练结果外，必须对每个训练完成的 Checkpoint 执行以下推理消融。

这些是冻结模型诊断，不增加独立训练组。

## 10.1 Persistent State

正常运行：

- State Read 开启。
- State Write 开启。
- State 跨 Chunk 保留。

## 10.2 Read Disabled

禁用 State Read。

保留 Writer 的执行，但不允许 State 影响语言计算。

目的：确认模型是否真正利用 State。

## 10.3 Write Disabled

禁用 State Update。

State 始终保持初始状态。

目的：判断收益来自动态状态演化，还是仅来自可学习的静态 Memory Slots。

## 10.4 Reset Every Chunk

每个 Chunk 结束后将 State 重置为初始 Slot Embedding。

Read 仍然开启。

目的：判断跨 Chunk 持久化是否具有独立贡献。

## 10.5 State Perturbation

可选：

- 将某个 Cell State 替换为零。
- 将两个 Cell State 交换。
- 对 State 添加有限幅度扰动。

记录预测分布的 JS Divergence 和 NLL 变化。

不得使用任意大幅度扰动制造虚假的 State 重要性。

---

# 11. 评价指标

## 11.1 语言质量

必须报告：

- Training NLL。
- Validation NLL。
- Validation PPL。
- Learning Curve。
- Greedy Generation。
- Sampling Generation。

A2 与 B1/B2/B3 必须使用完全一致的 Tokenizer、验证样本和预测位置。

## 11.2 计算成本

必须报告：

- Total Trainable Parameters。
- FLOPs / Token。
- Attention FLOPs。
- FFN FLOPs。
- State Read FLOPs。
- State Write FLOPs。
- Training Tokens/s。
- Total Training Time。
- Peak VRAM。

## 11.3 推理成本

至少报告：

- Prefill Latency。
- Batch-1 Decode Median。
- Batch-1 Decode P95。
- Decode Tokens/s。
- State Memory。
- KV Cache Memory。
- Snapshot / Restore Cost。

B1/B2/B3 必须在同一硬件和精度下比较。

## 11.4 State 专项指标

记录：

- State Norm。
- State Update Magnitude。
- Cell Pairwise Cosine Similarity。
- State Read Attention Entropy。
- Per-Cell Read Mass。
- State Read Gate。
- State Update Gate。
- Read/Write Gradient Norm。
- State Reset 与 Persistent State 的预测差异。

由于本次没有 Routing，不得使用 Cell Activation Frequency 或 Active Cell Count 作为已经实现的指标。

---

# 12. 实验判定标准

本次属于 1M Token Development Pilot，不是最终科学确认。

## 12.1 第一目标

确认 B1、B2、B3 是否能够稳定训练，State 是否能够正常读写，以及是否出现明确的预测改善。

## 12.2 初步筛选

相对于 A2：

- PPL 改善是积极信号。
- PPL 恶化不超过 2% 可以作为初步非劣筛选参考。
- 额外 FLOPs 必须单独计算。
- 实测推理成本不能明显恶化。
- Reset Every Chunk 应当帮助判断持久化是否真正发挥作用。

如果 PPL 改善，但 Reset Every Chunk 后完全没有退化，不能据此认定持续 State 有效。

如果 B3 的改善仅来自额外计算，不能直接宣称 Cellular 架构具有效率优势。

## 12.3 不设置强制胜者

可能出现：

- B1 最好：State 更适合参与内容寻址。
- B2 最好：State 更适合辅助特征计算。
- B3 最好：两者具有互补作用。
- 三者相近：State 交互尚未产生明显收益。
- 三者都退化：当前 State 机制设计不合理。

所有结果都应保留。

不能为了让 B3 成为最终方案而调整评价标准。

## 12.4 研究限制

三个实验均只有一个 Seed，不能据此形成科学确认。

如果出现积极结果，下一步至少使用三个独立 Seed 进行复核，再考虑扩大到 10M Tokens。

Independent Test 保持未访问状态。

---

# 13. 正确性测试

必须新增：

| 测试 | 要求 |
|---|---|
| `test_a2_reference_equivalence` | A2 Chunked Forward 与原始完整 Forward 等价 |
| `test_attention_state_memory` | B1 State Memory 因果正确 |
| `test_ffn_state_conditioning` | B2 State 只进入 FFN 路径 |
| `test_combined_state_interaction` | B3 两条路径均有效 |
| `test_state_reset` | 独立样本正确重置 |
| `test_state_persistence` | Chunk 之间正确保留 |
| `test_batch_isolation` | 不同 Batch 样本不共享 State |
| `test_no_future_state_leak` | 不使用未来 Chunk 更新当前 State |
| `test_kv_cache_equivalence` | KV Cache 与原始 Attention 等价 |
| `test_state_read_ablation` | Read Disabled 生效 |
| `test_state_write_ablation` | Write Disabled 生效 |
| `test_reset_every_chunk` | State 重置诊断生效 |
| `test_gradient_across_chunks` | 必要梯度能够跨 Chunk 传播 |
| `test_snapshot_restore` | State 保存与恢复一致 |
| `test_parameter_accounting` | 参数量统计正确 |
| `test_flops_accounting` | 三组计算量分别统计 |

特别要求：

B1 必须证明 FFN 没有独立 State Read。

B2 必须证明原始 Token Attention 没有接入 State。

B3 必须证明两个路径都读取同一个 Chunk 开始时的 State，而不是读取中途更新的 State。

---

# 14. 工程目录与交付

建立独立实验目录：

`research/projects/cellular-language-models/experiments/epclm-a2-state-001/`

建议包含：

- `README.md`
- `SPEC.md`
- `protocol.json`
- `architecture.py`
- `state.py`
- `state_read.py`
- `state_write.py`
- `train.py`
- `evaluate.py`
- `flops.py`
- `runtime.py`
- `tests/`
- `reports/`

优先复用原始 A2、X4C 和 EPCLM Core-001 代码。

不要复制整个模型实现形成多个难以维护的分支。

训练产物放在：

`artifacts/experiments/epclm-a2-state-001/`

每组独立保存：

- `config.json`
- `training-metrics.json`
- `validation-results.json`
- `inference-profile.json`
- `flops-report.json`
- `state-diagnostics.json`
- `decision.json`

最终报告：

`RESEARCH-REPORT.zh-CN.md`

---

# 15. 最终报告

必须包含：

## 15.1 Baseline

A2 的历史配置、结果和数据身份。

## 15.2 B1

State 与 Attention 的交互结果。

## 15.3 B2

State 与 FFN 的交互结果。

## 15.4 B3

State 同时与 Attention 和 FFN 交互的结果。

## 15.5 对比

统一比较：

| 指标 | A2 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| Validation PPL | 51.397* | | | |
| Parameters | | | | |
| FLOPs / Token | 3.855M* | | | |
| Training Tokens/s | | | | |
| Decode Median | | | | |
| Peak VRAM | | | | |
| State Memory | 0 | | | |

* A2 数值来自历史 G1 记录；必须区分历史值与重新评估值。

## 15.6 机制结论

明确回答：

1. State 是否能有效参与 Attention？
2. State 是否能有效参与 FFN？
3. 哪条路径的贡献更明显？
4. 两条路径是否存在互补关系？
5. State 是否真正需要跨 Chunk 持续保留？
6. 性能改善是否值得额外计算成本？
7. 是否值得进入下一轮架构研究？

## 15.7 科学结论

每个假设必须标记：

- `SUPPORTED`
- `NOT_SUPPORTED`
- `INCONCLUSIVE`
- `NOT_RUN`

不得把尚未完成的实验写为 PASS。

---

# 16. Codex 执行顺序

**Step 1 — Audit**

检查原始 X4C 远端分支，定位 A2 的真实配置、初始参数、Checkpoint、数据和评估协议。

**Step 2 — Freeze A2**

完成 A2 Reference Forward、Chunked Forward 与 KV Cache 等价测试。

**Step 3 — Implement Shared State**

实现 16×128 State Bank、State Read、State Write 和共享 Update。

**Step 4 — Implement B1/B2/B3**

确保三个模型仅在 State 交互位置上不同。

**Step 5 — Correctness Tests**

先完成 CPU Smoke、Causal Test、Gradient Test、Snapshot Test。

**Step 6 — GPU Training**

使用 Kaggle Tesla T4，分别执行 B1/B2/B3，每个模型 1M Tokens。

不改变 A2 的训练预算和训练协议。

**Step 7 — Evaluation**

统一评估 A2、B1、B2、B3。

完成 State Read/Write 消融和计算成本分析。

**Step 8 — Publication**

将源码、Protocol、训练结果、模型 Checkpoint 和报告发布到独立结果分支。

验证远端 Commit SHA。

使用独立 Publication Receipt 记录最终验证状态，不要事后修改已经纳入 SHA 清单的 decision.json。

---

# 17. 最终研究命题

**Can persistent cellular state improve language computation through content-based interaction with Attention, feature computation through FFN, or both?**

持久化 Cellular State 能否通过与 Attention 的内容寻址交互、与 FFN 的特征计算交互，或者同时与二者交互，改善语言模型的计算能力？

本次实验不要求证明 EPCLM 一定优于 A2。

我们需要获得三个清晰的机制结论：

- State 与 Attention 的关系。
- State 与 FFN 的关系。
- State 同时参与两种计算时的关系。

**最终目标是找到 State 在语言模型计算架构中的合理位置，而不是简单地把 State 附加到现有 Transformer 上。**

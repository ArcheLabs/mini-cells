# EPCLM Experiment 001 — Formal Execution Protocol v0.4

**实验名称：** Incremental and Event-driven Execution  
**项目：** MiniCells / EPCLM  
**版本：** v0.4 — Formal Execution  
**日期：** 2026-10-30  
**目标仓库：** `ArcheLabs/mini-cells`  
**工作分支：** `codex/epclm001-incremental`  
**实验规模：** 约 1.9M Parameters  
**实验范围：** E1-A / E1-B / E1-C

## 0. 执行决定

本次决定正式完成 EPCLM-001 的三个核心实验。

已有基础：

- E1 的增量推理适配器和事件执行器已经实现。
- T1、X3、X4 已完成 10M Tokens 的受控重新训练。
- 三个模型的训练 Checkpoint 和独立推理 Checkpoint 均已保存。
- 真实模型在原验证集上的 PPL 与历史结果一致。
- 训练资产的源代码、模型配置、Tokenizer 和数据身份已经登记。
- 工程测试和真实权重的基础验证已经完成。

本次不再开展模型训练或架构探索。

**唯一研究任务是：验证现有 CLM 的精确增量计算、事件驱动执行的硬件效率，以及与现代 Transformer 基线的实际推理效率差异。**

原始 `PROTOCOL_DRAFT.md`、`PROTOCOL.v0.2.md`、`PROTOCOL.v0.3.md` 必须保留。

新增 `PROTOCOL.v0.4.md`，仅替代正式 E1-A/B/C 实验的执行授权和测量规则，不改变已有模型或历史训练结论。

本次明确授权有界的 1.9M 正式 GPU 推理实验。该授权不包含训练、30M 扩展或后续 EPCLM 新架构实验。

---

# 1. 实验资产冻结

## 1.1 模型

只允许使用以下三个模型。

| Model | Parameters | Architecture |
|---|---:|---|
| T1 | 1,894,080 | Modern Transformer |
| X3 | 1,893,536 | Native CLM — LayerNorm / SwiGLU / RoPE |
| X4 | 1,893,912 | Native CLM — RMSNorm / SwiGLU / RoPE |

三个模型均为：

- Training Seed：91002
- Training Tokens：10,000,000
- Training Loop Steps：10,000
- Vocabulary：2048
- Context Length：128
- Dataset：TinyStories
- Asset Generation：`CONTROLLED_RETRAIN_001`

保留原始训练身份。T1 的实际初始化 Seed 为 91003，X3/X4 为 91002。

不要因三个模型的训练源码提交不同而拒绝已经验证通过的真实资产，也不得将当前适配器 HEAD 替代原始训练源码身份。

## 1.2 真实推理权重

必须从本次真实训练资产中读取 `inference.pt`，不得使用随机初始化模型进行正式测量。

Hugging Face 仓库：

`archelabsxyz/native-clm-v0`

Run ID：

`fbf1959b2a364022959ded11c1e4fea6`

按以下固定 Revision 读取：

| Model | HF Revision | Inference SHA-256 |
|---|---|---|
| T1 | `732dfa2fb500656d34b2b097187f0d87d9d0ab13` | `89f78262eac99adb2f56ec281324521c8e3e829e0010e5d73e7288bfc6904bdc` |
| X3 | `42e2b82948292e047e21adafb7a2dc6805ccefa8` | `38c5755735486e91de8d0c853be644033e89f5eddd050c7217d36fbebb702ad0` |
| X4 | `13c6c975390fdda4b18bfabfe8e6bf2be3a67d54` | `b2c6e7cb7c523e2112fee7e5f3aef10f34cb720a93bd666747af77137303a2e8` |

文件位于对应 Revision 的：

`epclm001/controlled-retrain-001/<RUN_ID>/<MODEL>/inference.pt`

每次实验必须记录：

- Checkpoint SHA-256
- 训练源码 SHA
- 实验源码 SHA
- 模型参数量
- Tokenizer SHA
- 数据 Manifest SHA
- 实际计算精度

Tokenizer SHA：

`becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2`

如果 Hash 不匹配，停止对应模型实验。

## 1.3 历史模型质量

冻结以下真实验证结果：

| Model | Validation PPL |
|---|---:|
| T1 | 12.336077958 |
| X3 | 11.836404714 |
| X4 | 11.912722684 |

三者使用同一训练与验证协议。

这证明它们具有可比较的开发实验条件，但不代表相同的模型质量。

此次实验不得重新训练、调整模型参数或选择其他 Checkpoint 来优化结果。

---

# 2. 研究假设

## H1 — Exact Incremental Execution

**H1a：** 现有 Native CLM 可以在模型数学结构不变的情况下，通过缓存历史中间结果，精确执行增量推理。

**H1b：** 与同一模型的重复 Full Forward 相比，增量推理能够在某些合法上下文长度和硬件条件下降低实际执行时间。

分开判定 H1a 和 H1b。

正确性通过不意味着性能一定更好。

## H2 — Event-driven Physical Execution

**H2a：** 在给定合法事件轨迹和固定 Cell 更新规则下，Dense 与 Sparse 实现可以保持相同的状态演化结果。

**H2b：** 对足够稀疏的事件分布，实际跳过非活跃 Cell 的运算，可以降低 GPU 执行成本。

必须测量实际加速，而不是仅测量理论 FLOPs。

## H3 — Competitive Inference Efficiency

使用相同训练规模、相同 Tokenizer、相同推理设备、相同上下文限制，比较：

- T1 Cached Transformer
- X3 Incremental CLM
- X4 Incremental CLM

验证 CLM 是否能够缩小与 Transformer 的性能差距，或在特定工作负载下形成优势。

必须同时报告模型质量差异。

---

# 3. 全局实验规则

## 3.1 禁止范围

不得：

1. 重新训练 T1、X3、X4。
2. 执行 30M 模型。
3. 使用 Confirmation Seeds 进行新的架构选择。
4. 修改已训练模型的结构或权重。
5. 把随机模型测试替代为真实模型实验结果。
6. 自动修改测量工作负载以获得有利结果。
7. 因为某个模型更慢而不报告该模型。
8. 将合成稀疏执行结果称为完整 CLM 推理加速。
9. 将 KV Cache 称为持久 EPCLM Cell Memory。
10. 自动合并 `main`。

## 3.2 GPU 环境

优先复用 Kaggle 上已成功完成训练的 GPU 环境。

正式延迟测量应使用：

- 一个 GPU Worker。
- 同一块 GPU。
- 模型顺序加载和测量。
- Batch Size = 1，作为主要结论条件。
- 固定 CUDA/PyTorch 环境。
- 无其他训练任务并行运行。

如果设备有两块 T4，不允许将 T1 和 X3 分别放在不同 GPU 上测量后直接比较。

可以在第二阶段开展 Batch 8 辅助测量，但不得混入 Batch 1 结果。

## 3.3 精度与数值正确性

首先验证 FP32 的真实模型推理。

随后可以增加 FP16 模式，但必须单独经过数值正确性门禁。

FP16/BF16 的数值容差应通过独立诊断输入和数值尺度分析预先冻结，再使用未参与容差调整的正式输入测试。

任何精度下均不得因测试失败而自动放宽容差。

报告：

- Maximum Absolute Logit Error
- Mean Absolute Logit Error
- Relative Error
- Top-1 Disagreement
- Top-5 Overlap
- 逐前缀 Allclose 结果

对于非常接近零的 Logit，Relative Error 可能不稳定，因此正确性判定应以冻结的绝对加相对误差准则为主。

## 3.4 测量顺序

严格按照以下步骤：

1. 资产验证。
2. 正确性验证。
3. 有界 Pilot Benchmark。
4. 冻结正式工作负载与测量配置。
5. 正式性能测量。
6. 数据分析。
7. 自动生成报告。

不得先测量性能，再根据观察结果重新选择研究条件。

---

# 4. E1-A — Exact Incremental CLM

## 4.1 实验目标

验证真实 X3/X4 在不改变原有模型语义的条件下，通过缓存历史 KV 和中间表示，能否避免重复计算。

核心对比：

`Full Forward → A0 Incremental → A1 Preallocated Incremental`

其中：

- Full Forward：原始模型，对每个前缀完整重新执行。
- A0：现有标准增量缓存。
- A1：预分配缓存，减少部分缓存分配和拼接成本。

## 4.2 正确性

使用真实 X3/X4 权重。

验证：

- 单 Token Decode。
- Full Forward Prefix 对照。
- Prefix Length = 1、7、8、9、31、32、33、63、64、95、96、127、128。
- 其余所有合法前缀位置。
- Chunk Prefill。
- Reset / Restore。
- Batch 1 和 Batch 2。
- 不同缓存实例之间的隔离。
- 错误 Checkpoint 身份拒绝。
- RoPE Absolute Position。
- Stage/Iteration KV 一致性。

每个模型的三阶段、四次迭代必须保持原来的计算顺序。

不得利用未来 Token 信息。

`decode_one()` 只消费当前已知 Token，并产生下一个位置的预测。

必须首先证明：

\[
\operatorname{Incremental}(x_{\leq t})
\approx
\operatorname{FullForward}(x_{\leq t})_t
\]

所有正式性能路径必须通过对应精度的正确性检查。

## 4.3 性能工作负载

建议固定以下有代表性的输入长度。

| Prefill | Decode 消费步数 | 总位置 |
|---:|---:|---:|
| 16 | 16 | 32 |
| 32 | 32 | 64 |
| 64 | 32 | 96 |
| 96 | 32 | 128 |
| 112 | 16 | 128 |

全部处于现有模型的 128 Token 上下文限制内。

使用固定、真实 Tokenizer 编码的 Token 流进行 Teacher-forced 推理。

选择训练数据之外的冻结输入片段；明确它们的来源和哈希。

正式测试前将所有输入和输出位置冻结。

## 4.4 性能指标

每个工作负载记录：

- Full Forward 总耗时。
- A0 Prefill 耗时。
- A1 Prefill 耗时。
- A0/A1 单 Token Decode 耗时。
- TPOT p50/p95/p99。
- 总 Decode 时间。
- 缓存分配量。
- KV Cache Bytes。
- 峰值 GPU 显存。
- Host Wall Time。
- Device Elapsed Time。
- 可用的 Kernel Launch 与内存流量指标。
- FLOPs 估算与其覆盖范围。

计算：

\[
S_{A0}=
\frac{T_{\mathrm{Full}}}{T_{\mathrm{A0}}}
\]

\[
S_{A1}=
\frac{T_{\mathrm{Full}}}{T_{\mathrm{A1}}}
\]

以上加速比只在相同模型、相同输入、相同计时边界下计算。

必须区分：

- Prefill 加速比。
- Decode 加速比。
- Prefill + Decode 总加速比。

如果 A1 在小模型上慢于 A0，应报告原因，例如预分配管理成本或 Kernel Launch，而不是删除结果。

## 4.5 E1-A 验收

E1-A 完成条件：

1. X3/X4 真实模型数值正确性通过。
2. A0/A1 的缓存符合原始模型语义。
3. 完成冻结长度的 GPU 测量。
4. 生成逐前缀误差和性能曲线。
5. 报告是否存在真实加速区间。
6. 说明没有观察到加速的区间。

H1a 与 H1b 分开给出结论。

---

# 5. E1-B — Event-driven Execution

## 5.1 实验目标

这一实验不直接改变已经训练完成的 X3/X4。

它独立验证：

**当我们已经知道哪些 Cell 需要更新时，是否值得只计算活跃 Cell？**

这是执行机制实验，不是语言模型能力实验。

## 5.2 四种执行器

保留当前实现：

| Executor | 执行方式 |
|---|---|
| D0 | Dense Reference |
| D1 | Masked Dense |
| S0 | Sparse Cell |
| S1 | Sparse Region |

四种执行器必须使用相同的状态、输入事件和更新函数。

在每个 Microstep：

- 同一状态快照用于本轮读取。
- 只允许合法活跃 Cell 的状态变化。
- 新事件最早在下一 Microstep 可见。
- Region 内非活跃 Cell 即使参与候选计算，也不得写回状态。
- 空事件不得执行无意义的全量计算。

## 5.3 两级更新核

### Kernel B0：原合成更新核

保留现有的 `tanh` 共享线性更新核。

用途：

- 验证事件语义。
- 确定原始的稀疏执行成本。
- 作为简单且可复现的对照。

不得将其结果解释为完整 CLM 加速。

### Kernel B1：Native-like Cell Update

新增独立的 Native-like 工作负载。

使用：

- Cell State Width = 188。
- FFN Hidden Width = 638（X3-like）。
- FFN Hidden Width = 639（X4-like）。

计算可以包含与 Native CLM 相近的归一化、SwiGLU 和残差操作。

但它仍然是**固定外部事件驱动的更新核**，不是经过训练的持久 Cell 模型。

如果复用 X3/X4 已训练的某些子模块权重，需要明确声明其来源与作用，不能宣称 B1 等价于完整 X3/X4。

D0/D1/S0/S1 使用严格相同的 B1 数学函数。

## 5.4 正式工作负载

避免庞大的笛卡儿积扫描。

第一轮采用：

| 参数 | 取值 |
|---|---|
| Cell Count | 256、1024 |
| State Width | 188 |
| Activity Ratio | 0%、5%、50%、100% |
| Distribution | Uniform Random、Clustered |
| Primary Tile | 16 |
| Microsteps | 4 |
| Batch | 1 |

零活动和全活动作为负控制，不需要重复所有分布组合。

另外加入：

- Wavefront：验证局部传播。
- Burst：验证突发事件与调度成本。

这些测试点必须在正式运行前具体枚举并冻结，不允许根据初次测量的快慢进行筛选。

必要时再进行 Tile 32 的单独敏感性测试，不自动扩展为全矩阵。

## 5.5 正确性要求

逐 Microstep 比较：

- Cell State。
- Event Consumption。
- Event Ordering。
- Message Emission。
- Version。
- READ_QUERY。
- RESET。
- Empty Event。
- All-active Negative Control。

要求物理执行器不改变逻辑结果。

对浮点计算使用冻结容差；事件顺序、事件 ID、版本与消费集合要求严格一致。

## 5.6 性能指标

报告：

\[
\rho_{\mathrm{logical}}=
\frac{U_{\mathrm{logical}}}{N K}
\]

\[
\rho_{\mathrm{physical}}=
\frac{U_{\mathrm{physical}}}{N K}
\]

其中 Region 的无效候选计算必须计入 Physical Updates。

同时记录：

- Total Wall Time。
- GPU Stream Elapsed Time。
- Dense Kernel Time。
- Sparse Kernel Time。
- Queue / Compaction / Gather / Scatter / Message Cost。
- 逻辑和物理更新次数。
- Region Overcompute。
- 有效消息与丢弃消息。
- 实测显存占用。
- 可获取的 DRAM 流量。
- Kernel Launch 数量。

不可获取的测量项目标记为 `UNAVAILABLE`，不得使用估算数字冒充实测数据。

计算：

\[
S_{\mathrm{sparse}}=
\frac{T_{\mathrm{dense}}}{T_{\mathrm{sparse}}}
\]

必须分别比较 S0/D0、S1/D0，以及与 D1 的关系。

## 5.7 盈亏交叉点

找出：

- 什么活跃率下 S0 比 D0 更快。
- 什么活跃率下 S1 更有优势。
- Region 是否在聚集分布中更有效。
- Random 分布是否导致 Gather/Scatter 成本增加。
- 在 All-active 条件下稀疏执行是否失去优势。
- 在 Zero-active 条件下是否存在不必要的计算。

不得预设某个倍率为成功条件。

如果所有条件下 Sparse 都更慢，也应将其记录为有效的负结果。

## 5.8 E1-B 验收

1. 四种执行器语义一致。
2. B0、B1 两类工作负载清楚分开。
3. 完成冻结 GPU 工作负载。
4. 输出稀疏与 Dense 的性能曲线。
5. 定位物理稀疏更新的有效区间或无效原因。
6. 对硬件和工作负载的外推范围作出限制。

---

# 6. E1-C — Competitive Inference Benchmark

## 6.1 实验目标

比较现代 Transformer 与 Native CLM 在真实 GPU 推理中的效率。

这是 E1 的最终竞争性比较。

必须在 E1-A 的真实模型增量正确性通过后进行。

## 6.2 比较对象

| Model | Execution |
|---|---|
| T1 | Full Forward（诊断） |
| T1 | Parallel Prefill + Correct KV Cache |
| X3 | Full Forward |
| X3 | A0 Incremental |
| X3 | A1 Preallocated Incremental |
| X4 | Full Forward |
| X4 | A0 Incremental |
| X4 | A1 Preallocated Incremental |

正式跨模型比较的主基线为：

**T1 Cached vs X3 A1 vs X4 A1。**

另外报告 X3/X4 A0，避免将预分配缓存带来的收益混入架构本身。

T1 Full Forward 仅用于同模型诊断，不得作为未经缓存优化的弱对手来宣称 CLM 更快。

## 6.3 公平性要求

所有模型应采用：

- 相同 GPU。
- 相同输入 Token。
- 相同 Batch。
- 相同推理精度。
- 相同上下文限制。
- 相同 Teacher-forced Decode 规则。
- 相同 Warmup 和测量流程。
- 相同的模型加载、计时和结果记录规范。

T1 必须使用正确的 GQA KV Cache 和已经实现的 Parallel Prefill。

不得使用逐 Token Prefill 作为 T1 的唯一参考路径。

X3/X4 当前 Prefill 的顺序计算成本必须如实报告；不允许隐瞒或假设未来优化已经实现。

同时，这只是当前研究实现之间的比较，不代表 T1 已达到 vLLM、SGLang 等生产级推理系统的全部优化水平。

## 6.4 工作负载

沿用 E1-A 的五组输入长度。

Batch 1 为正式主结果。

建议首先使用：

- FP32：正确性和可复现性基线。
- FP16：通过独立正确性门禁后作为硬件效率主分析候选。

不得将 FP32 与 FP16 的结果混合计算加速比。

每个测试点进行：

- 预热。
- 顺序交错的成对测量。
- 建议至少 30 次正式重复。
- 必要的同步。
- 独立记录所有异常、OOM 和失败。

测量重复次数应在试运行后冻结，不得根据某一架构的表现进行调整。

## 6.5 必须拆分的时间

### Prefill

测量处理完整 Prompt 的时间。

### Decode

使用 Teacher-forced Token Stream，测量每次消费已知 Token 并输出下一位置 Logits 的延迟。

### TTFT

如果实现了完整的首 Token 采样路径，测量从 Prompt 可用至首个 Token 输出的时间。

未实现时标记为 `NOT_MEASURED`。

### End-to-End

完整 Tokenize → Prefill → Sample → Decode → Return 的用户路径，如果未实际实现，不得用 Kernel Time 冒充。

此外，所有计时需要明确 CUDA Synchronization 边界，区分 Host 时间和设备时间。

## 6.6 显存与缓存

分别测量：

- Model Weight Bytes。
- Resident GPU Memory。
- KV Cache Bytes。
- Cache Allocation Overhead。
- Peak GPU Memory。
- Prefill Working Memory。
- Decode Working Memory。
- Context Length 增长时的缓存曲线。

特别需要分别统计 X3/X4 在各 Stage 和 Iteration 保存的 KV。

比较 T1 的 4 个 Block 与 X3/X4 的 12 组迭代缓存带来的容量差异。

不得仅凭参数量接近就认为推理内存成本相同。

## 6.7 模型质量

使用相同冻结验证语料，对三个模型进行质量回归检查。

报告：

| Model | PPL | TPOT | Prefill | Cache Bytes |
|---|---:|---:|---:|---:|
| T1 | 实测 | 实测 | 实测 | 实测 |
| X3 | 实测 | 实测 | 实测 | 实测 |
| X4 | 实测 | 实测 | 实测 | 实测 |

禁止用已有 PPL 直接代替这次质量回归记录。

如果验证集和评价脚本与已验证的受控训练结果完全一致，可以复用经过哈希绑定的结果，并明确没有重新评价；如有变更则需要重新测量。

注意 X3/X4 的 PPL 比 T1 低，不能直接宣称它们在相同质量下更快。

应报告真实的质量—延迟对照点，而不是凭空插值得出质量前沿。

## 6.8 E1-C 验收

必须完成：

1. T1 KV 正确性。
2. X3/X4 A0/A1 正确性。
3. 同条件 GPU 性能测量。
4. Prefill 与 Decode 独立统计。
5. 相同验证条件下的模型质量记录。
6. 显存和缓存曲线。
7. 竞争性结果表。
8. 负面结果与限制说明。

---

# 7. 测量与统计协议

三个实验使用一致的科学证据规则。

## 7.1 正式测量之前

冻结：

- 模型权重和哈希。
- 测量代码 SHA。
- GPU 型号。
- PyTorch/CUDA 版本。
- Dtype。
- 工作负载矩阵。
- 随机输入和事件轨迹种子。
- Holdout 输入。
- Warmup / Repeat Count。
- 数值容差。
- 主要对照指标。

Pilot 结果与 Formal 结果必须分目录保存。

## 7.2 统计结果

报告：

- Raw Samples。
- Mean。
- Median。
- Standard Deviation。
- p50/p95/p99。
- 成对加速比。
- 按预先指定方法计算的不确定性区间。
- 成对测量时间顺序。

不要只报告最好的那一次执行。

有异常值时保留原始数据，明确是否以及为什么进行统计标记。

## 7.3 设备可重复性

记录：

- GPU Name。
- GPU Memory。
- Driver / CUDA / PyTorch。
- Python 版本。
- 功耗与频率（如果可获取）。
- 每个测试的工作集。
- 测量时设备是否存在其他任务。

尽量使用同一 GPU 进行全部正式测量。

必要时使用 Profiler 定位瓶颈，但 Profiler 数据应单独采集，不与常规延迟数据混合。

---

# 8. 资源预算与分阶段执行

继续以低 GPU 成本为优先原则。

建议设置一个独立于 v0.3 训练预算的 **7,200 GPU Worker 秒初始上限**，覆盖 E1-A/B/C 正式测试及其必要 Pilot；这是资源保护上限，不是预计运行时间。

该上限不自动转移或重置训练阶段的历史账目。

执行顺序：

**Phase P0 — Asset and Harness**

只验证 Checkpoint、源码、设备与实验入口。

**Phase P1 — E1-A Correctness**

运行真实权重数值正确性测试，不进行正式性能决策。

**Phase P2 — E1-A Pilot and Formal**

先使用少量长度判断时间成本，再冻结并完成正式测量。

**Phase P3 — E1-B Semantics and Pilot**

完成 D0/D1/S0/S1 语义验证及 B0/B1 工作负载检查。

**Phase P4 — E1-B Formal**

完成冻结事件轨迹的 GPU 测量。

**Phase P5 — E1-C Formal**

在缓存正确性和公平性门禁通过后，对三个真实模型进行竞争性测试。

**Phase P6 — Analysis & Publication**

生成结论、机器数据、图表和研究报告。

如果资源上限不足：

- 停止尚未开始的阶段。
- 保留已经完成且有效的结果。
- 报告剩余资源需求。
- 不自动增加预算。
- 不把部分结果标记为实验全部完成。

---

# 9. 实现要求

## 9.1 原有代码复用

实验主目录：

`research/projects/cellular-language-models/experiments/epclm-001-incremental-execution/`

必须优先复用：

- `implementation/epclm001/incremental.py`
- `implementation/epclm001/events.py`
- `implementation/epclm001/benchmark.py`
- `implementation/epclm001/validation.py`
- `implementation/epclm001/assets.py`
- `implementation/epclm001/analysis.py`

同时复用 v0.3 已经完成的训练资产读取与身份校验机制。

不得重复实现一套不同语义的 CLM。

## 9.2 独立正式实验入口

新增正式入口，不要使现有 CPU 工程 Smoke 自动变成科学 Benchmark。

建议：

```text
e1-preflight
e1a-correctness
e1a-benchmark
e1b-correctness
e1b-benchmark
e1c-benchmark
e1-analyze
e1-publish
```

以上为需要新增或映射的命令契约，不假定当前代码已经实现。

所有正式 Benchmark 命令必须检查对应门禁与冻结配置。

## 9.3 Notebook

提供一个适用于 Kaggle 的正式实验 Notebook。

要求：

- 从固定源码 Commit Checkout。
- 只启用一个 GPU Worker。
- 下载并验证三个真实 Checkpoint。
- 优先完成正确性。
- 分阶段运行 E1-A/B/C。
- 保存所有原始结果。
- 将结果安全写入持久化存储。
- 自动返回 GitHub 报告。
- 不包含任何训练步骤。

Notebook 必须支持独立执行某个阶段。

不得因为重新执行 Notebook 而重复运行已经完成且有有效 Manifest 的正式实验，也不得静默覆盖已有结果。

---

# 10. 输出目录与交付

建议目录：

```text
epclm-001-incremental-execution/
├── PROTOCOL.v0.4.md
├── FREEZE.md
├── protocol.json
├── e1_formal.ipynb
├── implementation/
│   └── epclm001/
│       ├── incremental.py
│       ├── events.py
│       ├── benchmark.py
│       ├── formal.py
│       └── analysis.py
└── reports/
    └── e1-formal-001/
        ├── RUN_MANIFEST.json
        ├── ASSET_MANIFEST.json
        ├── RESOURCE_MANIFEST.json
        ├── E1A_PARITY.json
        ├── E1A_TIMING.json
        ├── E1B_PARITY.json
        ├── E1B_TIMING.json
        ├── E1C_TIMING.json
        ├── E1C_QUALITY.json
        ├── DECISIONS.json
        ├── SUMMARY.md
        ├── REVIEW.md
        ├── WHAT_NOT_PROVED.md
        └── figures/
```

可以依据现有代码结构调整文件名，但输出内容和证据不可缺失。

生成至少以下图表：

1. X3/X4 的逐前缀 Logit Error。
2. Full Forward / A0 / A1 的 Decode 延迟曲线。
3. E1-B 活跃率与实际加速比。
4. Event Logical Updates 与 Physical Updates。
5. Dense / Sparse 的工作量和执行时间比较。
6. T1 / X3 / X4 的 Prefill 对照。
7. T1 / X3 / X4 的 Decode TPOT 对照。
8. 模型质量 PPL 与延迟散点图。
9. Cache Bytes 与 Context Length。

所有图表必须从原始机器数据自动生成。

---

# 11. 实验决策规则

三个研究结论独立判定。

允许以下状态：

- `SUPPORTED`
- `NOT_SUPPORTED_IN_TESTED_REGIME`
- `INCOMPLETE`
- `INVALID`

## H1_INCREMENTAL

**SUPPORTED：** 真实权重的精确增量结果符合冻结数值条件，并在某些预注册的工作负载上观察到可复现的实际加速。

**NOT_SUPPORTED_IN_TESTED_REGIME：** 正确性通过，但正式测试范围内没有观察到有意义的性能收益；或精确增量计算存在不可消除的语义障碍。

**INCOMPLETE：** 权重、正确性覆盖或有效测量不足。

## H2_EVENT_RUNTIME

**SUPPORTED：** 语义等价，物理稀疏执行在预注册的某些有代表性的活动条件下产生可重复的实际效率收益。

**NOT_SUPPORTED_IN_TESTED_REGIME：** 语义正确，但在预注册设备与工作负载中没有观察到实用优势。

**INCOMPLETE：** 缺少语义或 GPU 数据。

## H3_COMPETITIVE

**SUPPORTED：** 真实 CLM 与合法 Cached T1 形成可复现的特定性能优势，或在预注册的比较维度中明确缩小性能差距。

必须进一步区分：

- Gap Reduced
- CLM Faster
- Lower Memory
- Better Quality but Slower
- Higher Cost

不得把这些不同的结论合并成一句「CLM 更高效」。

**NOT_SUPPORTED_IN_TESTED_REGIME：** CLM 没有观察到竞争优势，或额外迭代与 Attention 成本抵消了增量收益。

**INCOMPLETE：** 缺少正确性、公平性、质量或性能证据。

**INVALID：** 实验发生协议漂移、基线不等价或数据污染，导致结果不具可解释性。

---

# 12. 本次实验不证明什么

即使 E1-A/B/C 全部得到积极结果，也不能证明：

1. 当前 CLM 已摆脱 Attention。
2. 当前 CLM 已具备持久 Cell Memory。
3. Cell State 已能取代历史 Token KV。
4. 模型已实现自适应事件路由。
5. 稀疏更新能够在维持语言质量的同时应用于真实模型。
6. CLM 在 30M、1B 或更大规模上同样有效。
7. 小型 T1 能代表所有经过优化的 Transformer。
8. 较低 PPL 和较低延迟同时出现就意味着严格的等质量优势。

E1 的目标是建立可验证的工程与性能基础。

真正的持久 Cell State 和去 Token Attention 架构，需要后续独立的模型实验。

---

# 13. Codex 执行任务与完成定义

请按照本协议完成 EPCLM-001 v0.4。

实施步骤：

1. 核对现有分支、最新训练报告与三个模型资产。
2. 新增正式实验协议、配置与运行入口。
3. 修复阻碍真实权重实验的工程问题。
4. 建立并冻结独立的正确性/性能工作负载。
5. 实施并完成 E1-A。
6. 实施并完成 E1-B。
7. 在 E1-A 正确性门禁通过后实施 E1-C。
8. 保存全部机器可读数据。
9. 自动生成图表与中文研究总结。
10. 分别给出 H1/H2/H3 的最终结论。
11. 提交并推送至 `codex/epclm001-incremental`。
12. 报告远端 Commit SHA、实验资产地址、GPU 实际开销以及所有剩余限制。

如果当前执行环境没有 GPU 或所需持久化凭证，先完成可验证的工程与冻结准备，并交付可在 Kaggle 直接运行的 Notebook。

不得将没有实际执行的 GPU 结果标记为完成。

如果云端资源可用，并且对应阶段满足本协议门禁，可在已授权的 1.9M 和资源预算内执行正式实验，无需重新训练或再次设计实验范围。

**最终交付标准：三个实验各自具有真实结果、可复现的机器证据和独立的研究结论。**

本轮结束后，才能决定是否继续研究持久 Cell State、事件驱动语言模型以及替代 Token Attention 的新架构。
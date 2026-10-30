# EPCLM Experiment 001 — Incremental and Event-driven Execution

## 增量推理与事件驱动执行：正式实验设计与实施协议草案

| 字段 | 内容 |
|---|---|
| 项目 | MiniCells / EPCLM |
| 实验编号 | `EPCLM-001`（新研究线；不占用或覆盖现有 MiniCells `001–007`、`CLM-001` 等历史编号） |
| 版本 | Research Protocol Draft v0.1 |
| 日期 | 2026-10-30 |
| 状态 | **PROPOSED — NOT FROZEN — NOT EXECUTED** |
| 目标 | 复用现有两套 E0 模型与结果，检验精确增量执行、真正跳过更新的事件驱动运行时，以及现代 CLM 对照现代 Transformer 的推理效率 |
| 训练 | **禁止新模型训练**；旧模型权重冻结。仅允许不影响输出语义的运行时工程优化及独立合成执行器验证 |
| 后续关系 | 为 EPCLM 的持久 Cell State、学习型激活与通信提供执行证据；**不直接证明完整 EPCLM 成立** |
| 参考架构 | `EPCLM_架构研究文档.md` v0.1；本协议细化其事件调度与硬件效率相关研究 |

> **核心要求：** 在 E1 内严格区分三件事：（A）由因果性带来的**精确历史计算复用**，（B）具有外部给定事件／活跃集的**真正稀疏物理执行**，（C）真实运行时下的**跨架构效率对比**。A 不等于 B；A/B 的成功也不自动代表可学习、持久状态的 EPCLM 已经成功。

---

# 0. 执行摘要

## 0.1 科学问题

已经有两套不同目的的 E0 证据：

1. **E0-A：原始 TextNCA 改进架构 vs 传统 Transformer。** 目的是确认 NCA 语言建模能力及原始计算成本。历史讨论曾报告约 29.60M / 29.46M 参数、PPL 5.3532 / 5.3290 及不同吞吐与显存；这些数字在本协议中只作为**待定位原始报告和 Checkpoint 的历史线索**，不能自动当成增量 Decode 结果。
2. **E0-B：优化 Native CLM vs 现代 Transformer。** 仓库 `native-clm-v0/implementation/README.md` 描述约 1.9M 参数、TinyStories 10M tokens、T1 现代 Transformer，以及 C0、M4、N1、X3、X4 等候选。T1 有 RMSNorm、RoPE、SwiGLU、GQA；X3 是 N1 残差迭代 + SwiGLU + RoPE；X4 在 X3 上改用 RMSNorm。仓库文档报告初始开发与阶段搜索结果，但必须在启动前核验每个拟用 Checkpoint 的原件、哈希和正式证据状态。

**E0 不再重复训练或重新开一个探索项目。** EPCLM-001 的资产登记（Asset Lock）只核查哪些已训练权重、数据和可复现源码确实可用，以免对不存在或不一致的 Checkpoint 进行性能比较。

本实验由三个逻辑工作包构成：

- **E1-A｜Exact Incremental CLM：** 为既有 CLM 实现逐 Token 因果缓存、精确增量推理；必须证明 `prefill + decode_one` 与原始 `full_forward(prefix)` 输出在约定数值容差内一致。
- **E1-B｜Event-driven Execution Kernel：** 在无新训练的前提下，实现 Dense、Masked Dense、Sparse Cell、Sparse Region 执行器，以相同事件轨迹、相同 Cell 更新规则测量真正跳过运算的盈亏区间。
- **E1-C｜Competitive Inference Benchmark：** 用经过缓存优化的现代 Transformer 对比优化 CLM，测量 TTFT、TPOT、p50/p95、内存流量、内存容量与运行时开销；声明范围不得超过模型质量和实验条件所能支持的范围。

## 0.2 实验的科学输出

最终需要给出三项**独立决定**：

| 决定 | 核心命题 | 允许的结论 |
|---|---|---|
| `H1_INCREMENTAL` | 对现有 CLM，历史因果状态能否被精确缓存并转成真实增量执行收益？ | 支持 / 不支持 / 不完整 |
| `H2_EVENT_RUNTIME` | 固定逻辑更新语义下，物理稀疏执行是否存在有效的加速区间？ | 支持（注明区间和设备）/ 不支持 / 不完整 |
| `H3_COMPETITIVE` | 在合法公平的现实推理对比中，优化 CLM 是否缩小与现代 Transformer 的成本差距或形成优势区间？ | 支持某区间 / 未观察优势 / 不完整 |

**不得**由 H1 支持直接推断 H2，或由 H2 的合成事件执行结果直接推断真实语言模型端到端效率。即使 H3 没有优势，H1 或 H2 仍可能具有独立工程、科学价值。

## 0.3 不属于本实验

不实现新型固定容量持久 Cell State；不引入跨会话记忆、在线可塑性、动态 Cell 增殖；不学习事件触发策略、图拓扑和自适应停止；不改变 tokenizer、模型层数、Cell 更新公式或生成概率分布；不重新训练大模型；不修改和覆盖历史 E0/旧 CLM 实验状态。本实验中的“事件”是**执行调度对象**，而不是已学会的神经计算选择。

---

# 1. E0 资产继承：两套基线分别管理

## 1.1 E0-A：原始 TextNCA 改进型

该基线提供原始 TextNCA 改进模型与传统 Transformer 的质量和效率背景。它与 E0-B 不是同一模型规模、训练协议或架构，因此严禁把它们直接拼成一张未经归一化的质量／速度排行榜。

正式运行前登记：

- CLM 与 Transformer 的准确模型类、配置、精度、分词器及参数数目；
- 数据集修订版本、训练 Tokens、验证集、训练随机种子、评价代码；
- 原始结果文件、代码 commit、权重 Checkpoint、SHA-256；
- 历史吞吐数字究竟来自训练、全序列前向、预填充还是自回归解码；
- 是否存在可加载权重、完整模型推理入口和可复现的原始 `full_forward`。

如果 E0-A 缺少可读取的权重，**登记为 `ASSET_UNAVAILABLE`**。可以继续 E0-B 和 E1-B，不能从旧 PPL/吞吐虚构 E0-A 的增量运行结果；也不能随意重训一个“近似复刻品”称其为旧 E0-A。重训属于单独授权的复现研究。

## 1.2 E0-B：现代化 Native CLM

已在 `ArcheLabs/mini-cells` 的 `native-clm-v0` 研究实现中确认的**仓库设计定义**：

| 模型 | 角色 | 设计属性 | 在 E1 中的用途 |
|---|---|---|---|
| `C0` | 历史 TextNCA 缩小重建 | 局部因果注意力；窗口 8/32/128；每阶段 4 次迭代；GRU/LayerNorm/GELU | 可选：验证旧型更新规则与缓存等价 |
| `T1` | 现代 Transformer anchor | RMSNorm、RoPE、SwiGLU、GQA | **必选的现代对照**（若有可核验权重） |
| `M4` | 现代化 CLM 研究候选 | C1 路线基础上现代化 | 具备可验证权重时作为附加对照 |
| `X3` | 优化 CLM 候选 | 固定残差迭代 + SwiGLU + RoPE | **现代 CLM 的首选候选**（以资产核验为条件） |
| `X4` | 优化 CLM 候选 | X3 + RMSNorm | 具备可验证权重时与 X3 并列 |

仓库 README 报告 T1（PPL 12.1907）等开发种子结果、并说明 Phase-C 实验需进一步跨种子核验。**这是仓库文档中的历史报告，不是本次 E1 已完成确认；正式实验以所选原始 JSON / Checkpoint 为准。**

实现基准位于：

- `research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_runtime.py`
- `.../native_clm_phase_b.py`
- `.../native_clm_phase_c.py`
- `.../README.md`
- `.../BASELINE_LOCK.md`
- `.../REPLICATION.md`

本协议起草时读取的主分支源码 SHA：`36b0770f2858b89314911d3cdbaf3ac5505e4785`（2026-10-30 UTC）；它是**设计核对锚点**，不是对最终 Checkpoint 或后续工作树的自动锁定。

## 1.3 Asset Lock：不重做 E0，但必须审计输入

登记 `baseline_assets.json`，对每个模型保存：

```json
{
  "baseline_id": "E0B_X3",
  "model_name": "X3-n1-swiglu-rope",
  "architecture_source_sha": "<full_sha>",
  "checkpoint_uri": "<artifact_or_path>",
  "checkpoint_sha256": "<sha256>",
  "tokenizer_sha256": "<sha256>",
  "dataset_manifest_sha256": "<sha256_or_null>",
  "evaluation_source_sha": "<full_sha>",
  "training_seed": "<original_seed>",
  "evidence_status": "DEVELOPMENT_OR_CONFIRMATION_OR_UNKNOWN",
  "parameter_count": null,
  "precision": "<dtype>",
  "loadable": false
}
```

尖括号为实施时填写的占位符，不得把它们保留在正式结果清单中。允许 `dataset_manifest_sha256=null`，但缺失原因须明确记载；缺少必要权重的模型不得进入性能结果汇总。

**原则：** 历史 E0 质量数字可以作为背景，但 E1 同一张质量／效率图中使用的每个点，必须具有对应的实际加载权重、可复现评价与真实性能测量。

---

# 2. 研究假设与预注册边界

## H1 — 精确增量计算

**H1a（正确性）：** 对严格因果的 TextNCA/CLM，保存正确的 Stage/Iteration 内部历史 K/V 与状态后，只计算新位置能够在既定有效上下文长度内重建 `full_forward(prefix)` 最后位置的 logits（允许明确的浮点精度误差）。

**H1b（系统收益）：** 该实现减少重复的历史位置计算，且在真实 Batch=1 Decode 中达到可测的端到端收益（不保证任何长度都更快）。

## H2 — 事件驱动物理执行

**H2a（语义）：** 在外部固定的逻辑事件轨迹下，Sparse Cell / Sparse Region 和 Dense Masked Reference 对状态、消费事件及发出事件保持一致。

**H2b（硬件）：** 在预注册的活动比例、活动空间分布、状态维度和计算核大小的某些区间，实际物理稀疏执行减少工作量／字节流量并降低端到端执行时间。

## H3 — 合法推理比较

相对于拥有正确 KV Cache 的 T1 现代 Transformer，优化 CLM 在明确的上下文、Batch、精度、模型质量条件下，可以缩小推理开销差距；若出现优势，只能限定在测量到的范围。跨 E0-A/E0-B 的绝对 PPL、速度不能混算。

## 预注册纪律

工程预检阶段可以调整实现、容差的依据与测试配置；一旦完成 `FREEZE.md`，最终确认测试不得因为结果不理想而变更工作负载、选择更有利的模型/设备、修改判定条件或删除负结果。每次重大修改生成新的源码 SHA 与协议修订记录。**E1 不设凭空规定的“必须 2×、5×”速度门槛；由基准结果报告速度曲线与交叉区间。**

---

# 3. 关键概念：三种“跳过计算”不得混淆

| 类型 | 机制 | 是否改变模型语义 | 本实验位置 |
|---|---|---|---|
| **历史缓存 / 精确增量** | 不重算因果历史位置已确定的张量 | 原则上不改变（验证 FP 误差） | E1-A |
| **已知事件集上的稀疏执行** | 不执行本轮逻辑上未被激活的 Cell 更新 | 不改变给定事件更新规则 | E1-B |
| **模型自适应跳过更新** | 由可学习策略决定哪些 Cell/迭代不执行 | 可能改变输出，需要训练与质量检验 | **后续 E2/E3；不属于 E1** |

尤其不能把 E1-A 的 Cache 叫作固定容量 EPCLM 记忆：历史 K/V 缓存依赖局部上下文窗口、模型层/迭代次数，可能随可用窗口增长，且没有独立的跨 Token 信息压缩策略。

---

# 4. E1-A：Exact Incremental CLM

## 4.1 正确性的数学依据

设一个具有严格因果依赖的模型，在位置 `j`、Stage `s`、迭代 `k` 的隐藏状态为 `h[s,k,j]`。若其所有依赖位置 `p ≤ j`，新增 Token 位于 `t > j`，则在模型参数、随机数状态和相同输入前缀不变时：

\[
 h^{\mathrm{full}(t)}_{s,k,j}=h^{\mathrm{full}(j)}_{s,k,j},\quad j<t.
\]

于是，对于新位置 `t`，可以重用历史位置在**对应阶段与迭代层级**上已经完成的 K/V，而不需要让旧位置重新进入更新核。

**重要限定：** 该等式是对严格因果且无跨位置非因果操作的数学计算图的性质；不能仅凭模块名字是 `causal_attention` 就假定模型全路径满足因果性。必须静态检查归一化、RoPE、阶段交互、位置编码、Dropout、BatchNorm、路由聚合以及输出读出是否引入未来位置依赖。

## 4.2 仓库模型的专门要求

已核对的 E0-B 实现使用：三个阶段、每阶段四次更新、局部因果注意力窗口分别为 `8/32/128`。因此，**缓存粒度至少是 `(stage, iteration, historical_position)`**，不能只缓存 Stage 最终 hidden state，更不能只建一个全局 K/V 缓存复用到所有迭代。

每一迭代实际注意力输入来自该步的 `conditioned = state + step_embedding`（或现代化候选的等效定义），必须用这一迭代对应的归一化、权重与 RoPE 位置生成 K/V。不同 Stage、不同迭代、不同 Attention 权重不得串用缓存。

可以将每步缓存限制在该步窗口所需的最近 `w` 个位置，但这仅是**所选因果局部 Attention 的精确上下文缓存**，不代表模型的可用位置编码或训练长度可以自动外推；若 Checkpoint 的最大上下文 128，正式质量声明限定在已验证范围内。

## 4.3 增量接口规范

应提供纯粹的推理适配层，保留原始模型与权重：

```python
class IncrementalLM:
    def init_cache(self, batch_size: int, device, dtype, *, max_context: int): ...
    def prefill(self, input_ids, cache, *, start_position: int = 0):
        """按合法语义吸收完整前缀，返回最后位置 logits 和更新后的 cache。"""
    def decode_one(self, token_id, cache, *, absolute_position: int):
        """只吸收一个已确认 token，输出用于预测下一 token 的 logits。"""
    def reset(self, cache): ...
    def snapshot(self, cache): ...
    def restore(self, snapshot): ...
```

接口必须明确定义：`prefill([x0,..,xn])` 返回预测 `x(n+1)` 的 logits；`decode_one(x(n+1))` 返回预测 `x(n+2)` 的 logits。不能将当前目标 Token 提前注入预测状态。

缓存记录：模型版本、Checkpoint SHA、Session ID、位置前沿、每 Stage/Iteration 的 K/V 与必要递归状态、序列长度、dtype/device、有效窗口和重置标志。对于共享权重、位置索引和可能的 tied head，按原实现精确复现。

## 4.4 两种实现路径

**A0：Reference Incremental（优先保证正确）**

- 在 Python/PyTorch 中逐 Stage、逐迭代为新 Token 执行对应核；
- 以实际历史的 Step-local K/V 进行局部 Attention；
- 不提前做 Kernel 融合，不改低精度和 FFN；
- 保留原 `full_forward` 原样作为严格参考。

**A1：Optimized Incremental（只在 A0 合格后）**

- 预分配有界缓存和固定形状缓冲区；
- 减少逐 Token 张量拼接/拷贝、Python 小对象和 allocator 压力；
- 在不改变数学结果的前提下考虑算子融合或图捕获；
- 分开记录 A0→A1 的工程收益；不得将 A1 的数值／性能回归隐藏在平均值内。

## 4.5 因果性与缓存一致性测试

至少包含：

1. **Prefix invariance：** 用相同前缀加不同后缀，比较前缀各位置、各 Stage/Iteration 可观察状态，验证未来 Token 不改变过去。
2. **One-step parity：** 对每个有效 `t`，`decode_one` 和 `full_forward(x[:t+1])` 的最后位置 logits 对比；同时对齐所存的层间状态。
3. **Chunk partition invariance：** `[a,b,c]` 一次 prefill 与 `prefill([a])+decode_one(b)+decode_one(c)` 等价；再测试不等长分块。
4. **Boundary：** 序列长度 1、窗口边界前后（8、32、128 的相邻位置）、Padding/无 Padding、BOS/EOS、位置上限、Batch=1 与支持的多 Batch 情况。
5. **Reset/replay：** 连续运行、清空后重跑、Snapshot→Restore 后继续；检查缓存版本和长度一致。
6. **Data types：** FP32 作为诊断参考，FP16/BF16 单列容差与误差分布。混合精度应以真实执行的稳定输出为准。
7. **Token selection：** 在固定输入 Token 序列上做 Logit parity；自由采样生成须使用相同 RNG 和采样策略。若极小 logit 差导致分支采样不同，不以整个自由生成字符串作为唯一正确性裁判。
8. **Cache corruption：** 故意串用不同 checkpoint、位置或 stage 缓存时必须拒绝，不能静默输出。

**容差政策：** 在开发预检中确定并解释 `max_abs`, `mean_abs`, `max_rel`、logit 排名变化、top-k 重合率；为所有设备/精度登记容差并在确认前冻结。不得在最终报告临时放宽阈值。“平均误差较小”不能覆盖少数严重异常。

## 4.6 E1-A 账目

对每个长度与 Batch，报告：

- `prefill_latency_ms`；
- `decode_one_host_latency_ms`、`decode_one_device_latency_ms`；
- 端到端 `TTFT`、`TPOT(p50,p95,p99)`、稳态 tokens/s；
- `number_of_stage_iteration_updates`（逻辑／物理）与 attention key-value 历史访问量；
- 常驻模型权重、KV/cache bytes、峰值设备内存、CPU 端缓存管理 bytes；
- CUDA Kernel 个数、launch 总耗时与 profiler 支持的权重／状态／缓存流量；
- Prefill 与 Decode 分开报告；不得将 Full Forward 的训练吞吐率与 Decode TPOT 混合比较。

---

# 5. E1-B：Event-driven Execution Kernel

## 5.1 科学定位

E1-B 是一个**独立的事件驱动执行内核机制实验**，不在原 CLM 中任意跳过其既定状态更新。本部分验证“指定逻辑活跃集后，能否真正降低物理执行成本”。它不评价事件活跃集由神经模型学习是否合理；后者留给后续实验。

E1-B 必须按总架构文档的“区域微步”参考语义实现。默认采用同步微步：第 `k` 步读取冻结的就绪事件与状态快照，完成活跃 Cell/Region 计算，在步末提交，出站事件最早在第 `k+1` 步可见。**禁止**让实现的 GPU 调度顺序隐式改变事件可见性。

## 5.2 参考执行模型

固定 Cell 集 `V={1,..,N}`；每个 Cell 拥有状态 `s_i∈R^d` 和地址/版本，固定的共享更新核 `Fθ`，以及消息队列 `Q`。对输入事件轨迹 `T=(e_1,..,e_m)`，逻辑活跃集：

\[
A_k=\{i:\exists\ \text{合法、已就绪的目标事件 }e\in Q_{k,i}\}\cup \{\text{到期自唤醒的 Cell}\}.
\]

\[
s_i^{k+1}=\begin{cases}
F_\theta(s_i^k,\operatorname{Agg}(Q_{k,i})), & i\in A_k,\\
s_i^k, & i\notin A_k.
\end{cases}
\]

出站事件的目标、内容、到达时间依据相同、确定的 `Sendθ` 规则生成。在合成测试中，可以让发送规则由种子确定，不需要任何训练。

## 5.3 四个独立物理执行器

| 实现 | 逻辑更新 | 物理执行 | 基准目的 |
|---|---|---|---|
| `D0_dense_reference` | 用活跃掩码决定写回 | 可执行全体 Cell 的候选更新后选择性提交 | 标准正确性基线，成本高但简单 |
| `D1_masked_dense` | 与 D0 相同 | Dense Kernel+mask，尽可能融合，仍可能对休眠 Cell 做数学运算 | 区分掩码开销和真实跳算 |
| `S0_sparse_cell` | 仅活跃 Cell 写回 | 压实活动 ID、Gather/Compute/Scatter | 稀疏 Cell 核的硬件盈亏 |
| `S1_sparse_region` | 仅活跃 Cell 的状态发生改变 | 按连续 Region/Tile 批量执行；可对 Tile 中无关 Cell 计算候选值，但**不得提交其更新或发出事件** | 低访存与大批量执行的折衷 |

所有运行时必须执行**同一输入事件轨迹、相同微步语义和同一共享更新核**，在匹配浮点误差下保持更新后状态和出站事件等价。对 `S1` 的额外物理候选计算单独计入，不能把 Region 中本应休眠但实际参与计算的 Cell 称为“物理未更新”。

## 5.4 事件种类（E1 最小集）

| 事件 | 处理规则 | 注意事项 |
|---|---|---|
| `INPUT` | 激活目标 Cell/Region | 顺序与来源固定 |
| `STATE_MESSAGE` | 在下一微步向目标发送绝对摘要 | 同源顺序 / 版本一致 |
| `SELF_TICK` | 在有界预算下请求再次更新 | 防止无限循环 |
| `READ_QUERY` | 请求特定可观测状态或聚合读出 | 不能偷偷扫描所有 Cell 而不计费 |
| `RESET` | 清理状态、队列、版本、计数器 | 重跑必须确定 |

本实验不要求持久化跨会话磁盘日志、在线拓扑变更和生产级分布式恢复；这些可以作为正确性接口的最小占位，不能成为 E1-B 测速的主要负担。

## 5.5 事件轨迹与工作负载矩阵

需要覆盖不同类型，避免只挑稀疏有利的工作负载：

| 轨迹类型 | 特征 | 用途 |
|---|---|---|
| `uniform-random` | 活跃 Cell 随机分散 | 最困难的 Gather/Scatter 情况之一 |
| `clustered` | 活跃 Cell 集中于少数区域 | 测量 Tile/Region 优势 |
| `wavefront` | 活跃从局部逐微步向邻居传播 | 模拟局部通信与因果链 |
| `hotspot` | 频繁访问少数热点 | 测量负载不均衡与缓存 |
| `burst` | 低活跃区间与高峰交替 | 测量队列、背压、尾延迟 |
| `all-active` | 所有 Cell 激活 | 负控制：稀疏引擎不应凭空节省运算 |
| `zero-active` | 无合法事件 | 验证真正空闲不触发全量扫描或无意义更新 |

**建议的开发扫描点（冻结时可调整，但需全范围覆盖）：**

- 逻辑活动率 `ρ`：0、1、5、10、25、50、75、100%；
- Cell 数 `N`：256、1024、4096、16384（可依设备内存缩减并记录）；
- 状态维度 `d`：64、256、512（对应可复现合成核，需注明 FFN 宽度）；
- 每次外部输入可引发的微步数：1、4、12、32；
- Region/Tile：16、32、64、128；
- Batch/会话数：1、8、32（其中首要关注 Batch=1）。

此矩阵为多维条件空间，**不是要求做所有笛卡儿积**。预注册时使用分层抽样与最坏情况集合，明确必须覆盖的组合；不能根据初步速度挑选只对 Sparse 有利的点。若没有可靠设备资源，先固定设备并完成其中一组精简扫描，额外测试作为扩展。

## 5.6 事件轨迹冻结与可复现

- 用固定随机种子生成输入流、初态、活跃目标、消息时序；生成器及输入摘要写入 manifest；
- 正式确认前将至少一组未用来调参的轨迹冻结为 Holdout；
- 任何执行器不得利用“未来会被激活的 Cell”提前排布非法状态变更；
- 使用严格的 per-microstep state/event hash；
- 结果标记 warm/cold，内存分配和事件生成是否纳入计时要分开列示。

## 5.7 物理稀疏性定义

逻辑活动率：

\[
\rho_{logical}=\frac{\sum_k |A_k|}{NK}.
\]

物理更新负担：

\[
U_{physical}=\sum_k\big(\text{执行了更新核的 Cell 槽位数（含 Tile 过量计算）}\big).
\]

\[
\rho_{physical}=\frac{U_{physical}}{NK}.
\]

还必须分别给出 `unique_cells_touched`、重复激活次数、实际发送字节、执行 FLOPs、Region 额外过量计算、事件队列及路由 CPU/GPU 时间。对 `K=0` 的零事件场景，活动率标为 N/A，不用零除计算。

## 5.8 运行时性能模型

\[
T_{sparse}\approx T_{queue}+T_{compact}+T_{gather}+T_{compute}+T_{scatter}+T_{message}+T_{sync}+T_{launch},
\]

\[
T_{dense}\approx T_{scan}+T_{dense\_compute}+T_{mask}+T_{launch}.
\]

关键不是理论上 `ρ` 很低，而是寻找实测交叉点：

\[
\exists (N,d,\rho,\mathrm{layout},\mathrm{device}):\quad T_{sparse}<T_{dense}.
\]

此结论只对所测设备、更新核、活动轨迹和内存布局有效，不应被表述为通用倍率。

---

# 6. E1-C：现代 CLM 与现代 Transformer 的合法比较

## 6.1 竞争性对照的公平性

必须比较至少三类执行路径：

1. `T1 + correct incremental KV cache`（现代 Transformer）;
2. `X3/X4 + full prefix forward`（用于描述旧实现的成本，不作为公平竞争基线）;
3. `X3/X4 + exact incremental cache`（E1-A 实现）。

可再增加 `C0 + full/incremental` 和 E0-A 的原模型，以及 A1 融合 CLM，但应**在各自规模和质量条件内分析**。

`T1 + full prefix forward` 可作为诊断参考，但不得用于构造 CLM 优势结论。T1 的 KV Cache 需与 Full Forward parity；不能假定已有代码一定实现了高效合法 KV Cache。

## 6.2 质量条件

E1 不改变权重，模型质量沿用所选 Checkpoint，但仍应使用完全相同评估集进行最小质量回归检查。即使增量路径 logits 等价，两个**不同架构**的原始 PPL 也可能有差异；性能结论必须注明这一差异。

指标分三类报告：

- **同模型收益：** `CLM full → CLM incremental → CLM optimized incremental`，可在保持模型质量的前提下比较速度；
- **同协议跨模型：** `T1 vs X3/X4`，报告 PPL、参数、运算预算和延迟；当 PPL 不同，必须注明不是严格等质量比较；
- **质量—成本前沿：** 当有足够多的已训练 Checkpoint 支持时画 Pareto，不允许凭一两个点或人为插值宣布“等质量优势”。

## 6.3 必须拆分的推理时间

- 模型加载和初始化时间（不混入 warm steady-state decode）；
- `prefill` 处理整个提示词的时间；
- TTFT：定义从完整提示词可用到首个生成 Token 被选出（明确是否含 tokenize、采样及首轮运算）；
- Decode TPOT：仅逐 Token 处理，给 p50/p95/p99；
- E2E 用户延迟：含 tokenizer、调度、必要拷贝、采样和回传；
- 生成总耗时、输出 Tokens/s、每请求峰值内存；
- 精确缓存是否发生频繁复制／扩容；
- 多请求场景仅在单请求明确后追加。

不要把不同显卡上的 tokens/s 直接列成同一“架构效率”结论；必须注明设备、驱动、CUDA、PyTorch、功耗状态、频率和软件实现版本。

## 6.4 测试工作负载

- **基线长度：** 在 Checkpoint 官方上下文上限内，例如 1、8、16、32、64、96、128；如果不支持，按各模型的实际协议收缩；
- **Token 生成：** 固定教师强制 Token 流用于稳定可比的 logit 和 Decode 测速；自由生成作为后续辅助测试；
- **固定输出长度：** 在合法上下文预算内测短输出和多步输出；
- **Warmup：** CUDA/CPU 各自预热；避免 CUDA 异步计时错误，设备时间须用对应事件同步；
- **统计：** 至少 30 次独立重复（建议；冻结前确认），同时报告均值、p50/p95、标准差/置信区间；测试顺序交错或随机化，避免热和频率漂移偏向一个模型。

若模型上限为 128 Token，输入与输出总位置不能超出经验证的上限。任何 extrapolation 单列为探索，不能与正式结果混合。

## 6.5 资源记账

最小报告：

| 资源 | 说明 |
|---|---|
| 参数与模型权重 | 真实常驻 bytes、是否量化、是否权重共享 |
| KV / 中间缓存 | 每模型每会话 bytes，按 Stage/Iteration 拆分 |
| 激活 / 工作区 | 峰值设备内存，区分权重和缓存 |
| 逻辑计算 | Attention、FFN、Gate、归一化、输出 Head、索引与调度 |
| 物理运算 | Profiler 观测到的 Kernel、GEMM/SDPA 等实际算子 |
| 内存流量 | HBM/DRAM、必要 CPU-GPU 传输，确实测不到就记 N/A |
| Host 调度 | Python、队列、张量拼接、分配和 launch 开销 |
| 延迟和能耗 | TTFT、TPOT、整体时间；能耗仅在可测设备上报告 |

**禁止：** 把 `unique touched weights`、估算 FLOPs、GPU 峰值利用率或逻辑活动比例直接当成真实内存带宽/功耗。

---

# 7. 分离影响因素的消融矩阵

| 比较 | 改变了什么 | 不改变什么 | 能证明什么 |
|---|---|---|---|
| Full CLM vs A0 incremental | 有无历史缓存 | 相同权重、输入、输出公式 | 重复历史计算的可消除部分 |
| A0 vs A1 | 缓存布局和优化执行 | 相同逐 Token 数学语义 | 工程实现收益 |
| D0 vs D1 | 扫描/掩码实现 | 相同逻辑事件更新 | Dense Masking 优化收益 |
| D1 vs S0 | 物理是否跳过 Cell | 相同活跃集合与更新核 | 真正 Cell 稀疏执行收益 |
| S0 vs S1 | 是否区域批量融合 | 相同事件与提交语义 | 内存局部性和 Region 折衷 |
| T1 KV vs X3/X4 incremental | 不同已训练架构 | 同样合法增量实现和硬件条件 | 竞争性系统效率（需质量注明） |

**不可做的因果推断：** 将 E1-A 的历史缓存收益解释成“自适应事件路由学会减少计算”；将 E1-B 合成轨迹里 5% 活动率直接套到语言模型；将 E0-A 的 30M PPL 与 E0-B 的 1.9M 延迟组成一个跨规模的 Pareto 点。

---

# 8. 实施架构：只新增 E1 研究路径，禁止改写 E0

建议在原仓库研究树中新建独立路径（实际位置以仓库结构审查为准）：

```text
research/projects/cellular-language-models/experiments/epclm-001-incremental-event/
  README.md
  PROTOCOL_DRAFT.md
  protocol.json
  FREEZE.md
  ASSET_LOCK.md
  RESULTS.md
  manifests/
    baseline_assets.json
    checkpoint_inventory.json
    workload_matrix.json
  implementation/
    adapters/
      historical_clm_adapter.py
      native_clm_adapter.py
      transformer_kv_adapter.py
    incremental/
      cache.py
      reference_decode.py
      optimized_decode.py
      parity.py
    event_runtime/
      events.py
      reference_dense.py
      masked_dense.py
      sparse_cell.py
      sparse_region.py
      traces.py
      accounting.py
    benchmarks/
      bench_incremental.py
      bench_event_runtime.py
      bench_competitive.py
      profile_tools.py
    reporting/
      analyze.py
      decision.py
      plots.py
  tests/
    test_checkpoint_identity.py
    test_causality.py
    test_incremental_parity.py
    test_cache_reset_and_restore.py
    test_event_semantics.py
    test_sparse_dense_parity.py
    test_event_determinism.py
    test_accounting.py
    test_perf_harness.py
  artifacts/
    <experiment-source-sha>/<session-id>/...
```

新路径仅作**建议结构**，不是声称这些文件已存在。既有 `native-clm-v0` 训练/实验源码和冻结结果原则上只读；为 E1 创建 adapter，不把缓存代码硬改进旧研究模型造成基线漂移。

## 8.1 应实现的 CLI 契约

以下命令为**预期的实施接口，不是仓库当前已有命令**：

```bash
python -m epclm001.cli inventory \
  --repo-root . --output <experiment_dir>/manifests/baseline_assets.json

python -m epclm001.cli parity \
  --model E0B_X3 --checkpoint <file> --mode incremental

python -m epclm001.cli bench-incremental \
  --model E0B_X3 --checkpoint <file> --matrix <workloads.json>

python -m epclm001.cli bench-event \
  --matrix <workloads.json> --impl all --device cuda

python -m epclm001.cli bench-competitive \
  --baseline E0B_T1 --candidate E0B_X3 --matrix <workloads.json>

python -m epclm001.cli analyze \
  --artifact-root <experiment_dir>/artifacts --protocol <protocol.json>
```

实现者应先提供 `--help`、参数验证和 `dry-run`，再运行 GPU Benchmark。命令中 `<...>` 为占位符。

## 8.2 事件调度核心伪代码

```python
def execute_microstep(state, ready_events, runtime):
    # 1. 基于已经到达的合法事件确定更新目标；不扫描全状态判活。
    active_ids = runtime.extract_active_ids(ready_events)
    snapshot = runtime.read_consistent_snapshot(state, active_ids, ready_events)

    # 2. 物理执行器可 Dense / Sparse Cell / Sparse Region，但逻辑语义一致。
    proposed = runtime.compute_updates(snapshot, ready_events, active_ids)
    outgoing = runtime.prepare_outgoing(proposed, ready_events)

    # 3. 步末原子/逻辑一致地提交；出站消息不在本微步立即回读。
    next_state = runtime.commit(state, proposed, consumed=ready_events)
    runtime.enqueue_for_next_microstep(outgoing)
    return next_state
```

对于 CPU/GPU 单机研究原型，`commit` 可以是受控的版本化内存写回；本实验无需实现生产级分布式共识。但不得发生重复消费、非确定性越步可见或未活跃 Cell 被错误写回。

---

# 9. 测量程序、统计与结果形式

## 9.1 计时防偏差要求

- 冷启动和预热稳定路径分开测量，不使用首次 CUDA 初始化时间评价稳态；
- 将 Host Wall Time 与 GPU Kernel Time 分开，后者不能代表完整用户延迟；
- 对每个测试点进行预热、同步、重复测量，交错执行不同版本；
- CPU Benchmark 固定线程数、亲和性/NUMA（可控时）；GPU 记录设备、驱动、CUDA、显存、功耗与温度状态；
- 避免双 T4 共享 CPU/GPU/显存下同时跑两个延迟 Benchmark：训练/构建可并行，但最终延迟测试应单 worker 隔离执行；
- 如果无法控制 DVFS 或温度，记录时间序列，避免选择最有利瞬时样本；
- 失败、OOM、异常、Fallback 必须以独立状态记录，不能丢弃样本后只报告成功速度；
- 优先比较 paired workload 的时间比，报告分布而不是仅给“最快样本”。

## 9.2 每个测试点的标准证据

一个 `result.json` 至少需要：

```json
{
  "experiment": "EPCLM-001",
  "protocol_version": "draft-or-frozen-version",
  "source_sha": "<full_sha>",
  "checkpoint_sha256": "<hash-or-null>",
  "baseline_id": "E0B_X3",
  "implementation": "incremental_reference",
  "device": {"name": "<device>", "driver": "<version>", "torch": "<version>"},
  "workload": {"batch": 1, "prefill_tokens": 32, "decode_tokens": 16, "dtype": "float16"},
  "correctness": {"passed": null, "max_abs_logit_error": null},
  "timing": {"ttft_ms": null, "tpot_p50_ms": null, "tpot_p95_ms": null},
  "resources": {"peak_vram_bytes": null, "cache_bytes": null, "kernel_count": null},
  "costs": {"flops_estimate": null, "measured_dram_bytes": null},
  "status": "INCOMPLETE",
  "limitations": []
}
```

占位值不可直接用作最终结果。若设备无法可靠读取 DRAM bytes，记录 `null` 与 `measurement_status=UNAVAILABLE`，不能填 0。

事件运行时额外记录 `trace_hash`、逻辑/物理更新次数、消息数与字节、事件合并／丢弃与队列等待、Region 过量计算。模型缓存实验记录 `(stage,iteration,position)` 的缓存结构与 size。

## 9.3 不可缺少的图表

- `CLM length → full vs incremental TPOT`（两套 E0 分开）；
- `CLM incremental parity error vs prefix length`；
- `speedup vs activity ratio`（每类事件空间分布单独曲线）；
- `physical_updates vs logical_updates`；
- `queue/compact/gather/kernel/scatter` 时间占比；
- `quality (PPL) vs decode latency`（同一 E0 条件内比较）；
- `cache/resident memory vs context length`；
- `T1 / X3 / X4` 现代竞争性对比表（注明开发或确认的证据等级）。

图表必须由原始机器记录自动生成，不得手工编辑使负结果消失。

---

# 10. 硬门禁、判定与停止规则

## Gate 0 — Assets & Harness（无新模型训练）

必须确认：

- 至少一套有真实已训练 CLM 权重与原始 Full Forward 可运行；
- E0-B 的 T1 和至少一个 X3/X4（如果要做 H3）具备可审计的 Checkpoint 和 Tokenizer；
- 数据、源码、权重哈希可生成或注明缺失；
- 计时、Profiler、工作负载、设备 Manifest 端到端可运行；
- 源码修改不污染历史 E0 结果路径。

若无法取得 E0-A 权重，标记对应覆盖范围缺失，不自动阻塞可用的 E0-B；若缺少所有 CLM 权重，H1/H3 无法执行，但 H2 合成运行时仍可独立进行。不得假装所有实验已完成。

## Gate 1 — Exact Incremental Parity

- 正确性用固定 Teacher-forced Token 前缀验证；
- Stage/Iteration 缓存边界、窗口、RoPE 绝对位置正确；
- Full Forward 与 Incremental logits 在冻结容差内；
- Cache Reset/Restore、Chunk Partition、BOS/EOS、上下文边界与版本冲突测试全部通过。

**若 Gate 1 不通过：禁止为相应模型发布正式增量速度对比或 H3 优势结论。** 允许保留错误诊断与原始数据，但不能以“不完全一样却更快”的输出替代原模型。

## Gate 2 — Event Execution Semantics

- D0/D1/S0/S1 对同一事件轨迹的逻辑状态、消费事件与出站事件一致；
- 非活跃 Cell 不提交更新；版本／顺序／消息在微步后可见；
- 零事件不触发全量更新，All-active 负控制正确；
- 冻结事件轨迹与成本计数能复现。

**若 Gate 2 不通过：禁止声明事件驱动加速。** 不得拿行为不同的输出与 Dense 作硬件速度比较。

## Gate 3 — Performance Validity

- 隔离设备上完成预热、同步、重复和完整资源记账；
- 同时报告 Host 与 Kernel 时间、Cache 容量、Physical Updates、必要搬运和 Kernel；
- 测试覆盖随机与聚集活动、低与高活跃率以及负控制；
- 没有无法解释的协议漂移、Tensor dtype 差异和非法超上下文生成；
- 结果全量保留；正向与负向结果同时提交。

**若 Gate 3 不通过：性能决定 `INCOMPLETE`，不能用单次微基准和理论 FLOPs 替代。**

## 研究决定规则

| 决定 | 支持 | 不支持 | 不完整 |
|---|---|---|---|
| `H1_INCREMENTAL` | Gate1 通过，真实端到端 Benchmark 有复现的正收益区间 | Gate1 通过但在预设区间内没有收益，或因果缓存不可合法实现 | 权重／正确性／测量缺失 |
| `H2_EVENT_RUNTIME` | Gate2 通过，并在有代表性的合法负载下存在经确认的物理稀疏加速区间 | 语义通过但预设设备/负载中未观察到可用收益 | 语义或设备证据不足 |
| `H3_COMPETITIVE` | 合法 T1 KV 和 CLM Incremental 均通过，记录到可复现的差距改善或特定条件优势 | 未观察到差距改善或优势，或较高固定开销抵消复用收益 | 缺少可比权重、质量/性能或 Transformer KV 校验 |

必须将**“缩小差距”**与**“CLM 比 T1 更快”**分别报告；若质量不等，不能写“等质量更快”。完成独立判定不等于证明 EPCLM 作为全新语言模型成立。

## 停止与改变方向

- 若存在因果性违例，先修正/限定模型语义；**不得**跳过 parity 强行优化；
- 若 Incremental 正确但 Host Overhead 主导，优先预分配/融合 Kernel，不改数学公式；
- 若 sparse cell 随机负载慢而 clustered 加速，考虑 Region-aware 路由的后续训练假设；
- 若所有负载下 Sparse 都慢，保留失败结论，评估 CPU 小计算核或事件调度粒度，而不是悄悄换到有利设备；
- 若 E0-B 现代 CLM 的优化 Decode 仍慢于现代 T1，按 Kernel/权重/缓存/迭代成本拆解瓶颈，为 E2/E3 提出可证伪问题，不自动升级为“新训练架构必然有优势”。

---

# 11. Codex 实施顺序与交付定义

**本节描述同一实验内部的工程依赖，不是新的 EPCLM 科学阶段规划。**

| 工程检查点 | 工作 | 允许启动下项的条件 |
|---|---|---|
| `P0` | 仓库审计、Checkpoint inventory、E0 资产清单、规范的实验目录与 CLI dry-run | 选用 Checkpoint 有哈希、可加载；缺失如实记录 |
| `P1` | 原始 Full Forward 适配、因果性检查、增量 Reference Cache 与逐 Token parity | Gate 1 正确性达标 |
| `P2` | 增量缓存执行优化及可测性能；现代 T1 KV Cache parity | 原模型数学语义不变，测量框架可靠 |
| `P3` | D0/D1/S0/S1 事件执行器、确定性轨迹和密集/稀疏 parity | Gate 2 正确性达标 |
| `P4` | 预注册硬件矩阵、隔离性能采样、统计与成本分解 | Gate 3 有效性达标 |
| `P5` | 自动分析图表、独立 H1/H2/H3 决定、完整 What-Not-Proved | 所有已执行路径资料可复现，不隐去缺项 |

### 最小工程测试条件

- 新增研究路径下所有测试可用仓库惯例执行；
- `git diff --check`、静态检查、单元测试、模型 parity、事件 parity、Benchmark smoke 均输出机读结果；
- 不能为了修复新模块直接更改 E0 模型算子定义、旧训练协议和冻结结果；
- 未运行的 GPU Benchmark 只能标记 `NOT_RUN`；环境没有真实 CUDA 设备不能写 GPU `PASS`；
- 可提交开发/预检产物，但正式科学决定必须依赖冻结后完整、可信的结果集；
- 如果跨工作会话，需要持久化工作进度、运行参数和 Checkpoint 哈希，拒绝无法对应代码的孤立数字。

### 必须交付

1. `PROTOCOL_DRAFT.md`、冻结协议 `protocol.json`、`FREEZE.md` 和基线资产清单；
2. 两套 E0 的可用资产状态及旧结果引用（缺项明确）；
3. 至少一条 CLM Exact Incremental Adapter 与正确性测试；
4. 合法的现代 T1 增量 KV Cache 对照与 parity 记录；
5. D0/D1/S0/S1 四种事件执行器及统一语义/性能测试；
6. 原始 Performance Traces、Profiler 报告、资源 Manifest、CSV/JSON 统计；
7. 自动生成图表、H1/H2/H3 独立机器决定与中文研究总结；
8. `WHAT_THIS_DOES_NOT_PROVE.md`：明确本实验没有证明学习型事件路由、固定容量持久记忆或整个 EPCLM 比 Transformer 更高效。

---

# 12. 与总 EPCLM 架构文档的契约一致性

本实验与总文档 v0.1 的边界对应：

| EPCLM 总文档章节 | E1 的覆盖 | 不覆盖 |
|---|---|---|
| §3 组件契约 | 活跃区域执行核、调度、状态读写、计量 | 跨会话记忆提交、完整生产恢复 |
| §4 数学状态语义 | 用固定 Cell 更新核模拟版本化状态；模型 Cache 一致性 | 新的快慢状态学习模型 |
| §5 活动/路由 | 由事件驱动的活跃集、消息传播 | 自主学习的动态拓扑 |
| §7 事件协议 | 最小事件类型、顺序和版本 | 分布式多机事务协议 |
| §8 调度 | 区域微步、活跃率和停止预算基本约束 | 自适应学习停止控制器 |
| §10 训练 | 保留旧训练／权重定义作为冻结条件 | Stateful Training、BPTT 改造、路由梯度 |
| §11 推理 | Prefill、Decode、重置、读出一致性 | 固定容量 EPCLM 完整语言模型 |
| §12–13 硬件/复杂度 | 真实硬件计时、内存、稀疏盈亏 | 新 ASIC/NPU 系统设计 |
| §15–17 正确性/风险 | 无未来泄漏、无重复消费、负结果保存 | 长期记忆遗忘/隐私与持续学习 |

---

# 13. 最重要的失败模式与解释边界

1. **Cache 粒度错误。** 仅缓存 Stage 输出而忽视迭代内部 Attention K/V，得到不等价却看似更快的推理。
2. **错误的 RoPE 位置。** 单 Token 输入时相对位置重置为 0，导致与完整前向的 Q/K 不一致。
3. **错误的因果假设。** 任一全局归一化/聚合让旧位置依赖未来位置，破坏精确复用条件。
4. **只做 Masked Dense。** CUDA 仍执行全部 Cell 更新，却将低逻辑活动率标成低物理 FLOPs。
5. **事件轨迹不一致。** Sparse 因漏掉消息而更快，实际上是不同模型语义。
6. **小 Kernel 启动成本。** Sparse 减少运算但 Host/kernel launch、Gather/Scatter 超过节省量。
7. **不公平 Transformer。** 对 T1 不做 KV Cache，或让 CLM 使用 BF16 而 T1 使用 FP32，导致伪竞争优势。
8. **混淆不同 E0。** 30M TextNCA 的 PPL 和 1.9M 现代 CLM 的 latency 不能混成一个模型点。
9. **未核实原件。** 历史 E0 数字可出现在背景中，不能代替可加载权重和测量工件。
10. **短上下文过度外推。** 在训练/验证上限 128 内取得增量效率，不意味着长上下文精确记忆或稳定无限流。
11. **仅报告 FLOPs。** 实际内存搬运、缓存容量和调度可能吞掉 FLOPs 收益。
12. **重复运行污染证据。** 修改原 Frozen Baseline 或重写旧结果目录破坏科学溯源。
13. **将 E1-B 结果直接套到 EPCLM。** 手动外部活跃集在合成核上快，不代表语言模型会学到同样分布。
14. **把缓存当成永久知识。** K/V 缓存主要是精确历史计算复用，不是可以持续学习的压缩 Cell State。

---

# 14. 下一步由 E1 回答、而不是提前宣称的研究问题

E1 完成后，E2/E3 需要根据**观察到的**而不是臆测的瓶颈来设计：

- 如果局部区域活跃才有加速：模型训练是否应该鼓励空间/图区域上的活动聚集？
- 如果 Kernel/Host 固定开销占主导：能否减少每 Token 的迭代次数或融合更新核，而不是追求极端 Cell 级稀疏？
- 如果 Cache bytes 迅速增大：持久 Cell State 能否压缩历史依赖并在可比语言任务上保留信息？
- 如果 E1-A 已消除大量重复计算但仍慢于 T1：主要差距来自 12 次迭代、FFN、局部注意力、输出头还是内存布局？
- 如果事件调度只有 CPU 上有效：下一代 Cell 更新核是否需要按硬件类型重构？
- 如果长期读取非常昂贵：是否有必要重新定义 Token 与 Cell 的映射、引入独立 Cell 地址或外部记忆？

这些是**下一阶段的研究输入**，不是本实验要实现的新功能。

---

# 附录 A：参考资料与出处

1. 用户已提供：《EPCLM_架构研究文档.md》，v0.1，2026-10-30，特别参考 §3–§5、§7–§8、§11–§13、§15–§19。
2. MiniCells Native CLM 架构实验：[README](https://github.com/ArcheLabs/mini-cells/blob/36b0770f2858b89314911d3cdbaf3ac5505e4785/research/projects/cellular-language-models/experiments/native-clm-v0/implementation/README.md)。
3. E0-B 基线判定与未锁定条件：[BASELINE_LOCK](https://github.com/ArcheLabs/mini-cells/blob/36b0770f2858b89314911d3cdbaf3ac5505e4785/research/projects/cellular-language-models/experiments/native-clm-v0/implementation/BASELINE_LOCK.md)。
4. E0-B 复制协议：[REPLICATION](https://github.com/ArcheLabs/mini-cells/blob/36b0770f2858b89314911d3cdbaf3ac5505e4785/research/projects/cellular-language-models/experiments/native-clm-v0/implementation/REPLICATION.md)。
5. E0-B 原始架构：[native_clm_runtime.py](https://github.com/ArcheLabs/mini-cells/blob/36b0770f2858b89314911d3cdbaf3ac5505e4785/research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_runtime.py)。
6. E0-B 现代化候选：[native_clm_phase_c.py](https://github.com/ArcheLabs/mini-cells/blob/36b0770f2858b89314911d3cdbaf3ac5505e4785/research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_phase_c.py)。
7. MiniCells Learned Cell Connections 的独立研究草案：《MiniCells Experiment 005 — Learned Cell Connections and Compositional Efficiency》，此实验研究 FFN Cell 组合而非 EPCLM Event State；两者不得相互替换科学结论。

# 附录 B：术语与结果用语

- `Exact Incremental`：严格因果计算图上对历史计算的合法缓存复用。
- `Event-driven`：由合法到达事件直接驱动待更新集；不是 Dense 后乘 Mask。
- `Logical Update`：模型语义上需要改变某 Cell 状态。
- `Physical Update`：物理计算核实际覆盖的 Cell 槽位，包含 Region 过量计算。
- `Dense Reference`：对同一更新语义的完整/掩码参考实现。
- `Region Batch`：以连续 Cell 区块为物理调度单元；逻辑上仍只提交活跃 Cell 更新。
- `Checkpoint identity`：权重原件与源代码、配置、Tokenizer 的绑定关系。
- `PASS / NOT_SUPPORTED / INCOMPLETE`：分别表示特定科学命题得到支持、在注册条件下未获支持、证据不足；工程单元测试通过不等于科学 PASS。
- `Observed speedup`：必须带工作负载、设备、精度、实现版本、上下文长度和统计区间。

---

**最终验收原则：** EPCLM-001 的责任不是证明 EPCLM 已经优于 Transformer，而是可靠地测出：（1）现有 CLM 多少重复工作可用精确因果缓存消除；（2）有外部给定事件时，物理稀疏 Cell 执行何时有利；（3）经过合法推理优化后，优化 CLM 与现代 Transformer 的真实效率差距。三项结果必须独立、可复现、包含负结果，并为后续重新设计训练与持久 Cell State 提供有证据的约束。

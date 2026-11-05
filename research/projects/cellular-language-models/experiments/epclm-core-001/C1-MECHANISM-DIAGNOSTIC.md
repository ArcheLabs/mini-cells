# EPCLM Core-001 — C1 Mechanism Diagnostic

**持久 Cellular State 机制诊断实验规范 v0.1**

- 项目：MiniCells / EPCLM
- 实验：Core-001 / C1 Mechanism Diagnostic
- 类型：开发阶段、冻结模型、只读机制诊断
- 状态：PROPOSED
- 模型：C1 — Fixed Local Cellular Network
- 执行环境：Kaggle GPU / PyTorch FP32
- 前置条件：Core-001 C1/F0 Development Pair 已完成
- 主要目标：验证 State、Connection、Internal Evolution 对模型输出的实际作用

---

## 1. 实验背景

Core-001 已完成第一轮真实语言模型开发训练。

训练源提交：

`398a191dbb32e383c86c89b9c981777c7cd6c77f`

结果分支：

`kaggle/epclm-core-001-398a191dbb32`

[开发实验结果](https://github.com/ArcheLabs/mini-cells/blob/kaggle/epclm-core-001-398a191dbb32/artifacts/experiments/epclm-core-001/development/decision.json)

### 1.1 已有结果

| 指标 | C1 Cellular | F0 Flat |
|---|---:|---:|
| 训练 Token | 250,000 | 250,000 |
| 参数量 | 966,784 | 1,493,120 |
| Persistent State | 2,048 floats | 2,048 floats |
| 验证 NLL | 5.814626 | 5.830024 |
| 验证 PPL | 335.166 | 340.367 |
| 训练速度 | 437.6 tokens/s | 717.2 tokens/s |
| 训练精度 | FP32 | FP32 |
| 训练状态 | COMPLETE | COMPLETE |

这组结果说明 C1 已能够完成真实语言模型训练，并在当前开发集上略优于 F0。

但尚未证明：

1. C1 是否真正利用持久 State 表示有用的历史信息。
2. Cell 间通信是否对预测质量具有因果贡献。
3. 内部重复演化是否能够改善计算结果。
4. C1 的性能差异是否来源于 Cellular 组织。

此外，本次训练仅为一个开发 Seed，不能作为正式科学确认。

### 1.2 本次实验的任务

本次不再研究“模型是否可以训练”。

本次要回答：

> 已经训练完成的 C1，在实际进行语言计算时，State、Connection 和内部演化究竟发挥了什么作用？

通过直接干预模型运行过程，观察语言预测和内部状态如何变化。

---

## 2. 实验范围

### 2.1 必须完成

本次实验包含四个部分：

| 诊断 | 核心问题 | 类型 |
|---|---|---|
| D0 — Baseline Integrity | 能否准确复现已训练模型的推理行为？ | 工程门禁 |
| D1 — State Causality | 模型是否依赖此前事件形成的状态？ | 机制诊断 |
| D2 — Connection Causality | 跨 Cell 通信是否影响有用计算？ | 机制诊断 |
| D3 — Internal Evolution | 更多或更少的内部计算是否影响预测质量？ | 探索性诊断 |

D0 是必需门禁。

D1–D3 属于模型机制研究，不是新的训练实验。

### 2.2 严格禁止

本次不得执行：

- 新模型训练；
- Optimizer Step；
- 反向传播；
- 改写模型权重；
- 重新训练 C0/F0/L0；
- 重新训练不同 K 的 C1；
- 新的架构搜索；
- 扩大 Cell 数量；
- Adaptive Attention；
- Dynamic Routing；
- 正式 Confirmation；
- 多 Seed 训练；
- 自动进入后续 1M/10M Token 训练。

只允许使用冻结权重进行前向推理与因果干预。

---

## 3. 训练模型与数据身份

### 3.1 C1 模型身份

必须使用已训练完成的 C1 Checkpoint。

训练 Source SHA：

`398a191dbb32e383c86c89b9c981777c7cd6c77f`

训练 Protocol SHA-256：

`2835ced239c30dc71ad1f8066b7e61aa329203ca448c5dcf5adb98fff2170aa2`

数据 Manifest SHA-256：

`1eac9053e15ab4666a4f77635dd785728801a2cbea44a22472c42e8a920092cb`

C1 Checkpoint：

`epclm-core-001-development-pair/C1.pt`

Checkpoint SHA-256：

`36a58f642f93b859ce33f5fd55bd92c9f4196669dc8c62293088ea43ce8e7fd1`

模型权重 SHA-256：

`3787241e0ea18a206fb9625828d512845dce8b7156908255792fe420f12eacbf`

### 3.2 Checkpoint 获取规则

Checkpoint 位于私人 Kaggle Output，不在公开 GitHub 结果中。

公开结果里的 `saved_version` 仍是 `PENDING_SAVE`，因此不能仅凭 GitHub 记录假定文件已经可以恢复。

诊断程序必须先：

1. 找到源自上述训练提交的 Kaggle Output。
2. 检查原始恢复 Manifest。
3. 下载并校验 C1 Checkpoint。
4. 使用仓库现有的 `load_checkpoint()` 解析受信任的训练资产。
5. 提取模型 `state_dict`。
6. 严格加载 C1 权重。
7. 重新计算 `weight_hash()` 并验证。
8. 验证 Tokenizer、数据 Manifest 和模型配置。

只有全部通过才允许开始诊断。

**特别注意：**

现有 `storage.recover()` 会根据当前 Source/Protocol 判断是否复用完整训练资产。诊断源码的 SHA 必然不同于训练源码，因此不能直接使用现有恢复逻辑来判断“模型不兼容”，更不能因此重新训练 C1。

需要增加一个只读的已训练模型加载路径：

`load_frozen_model(training_identity, checkpoint_identity)`

模型训练来源和诊断代码来源必须分别记录。

如果旧 Checkpoint 找不到，应停止并报告 `BLOCKED_MODEL_ASSET_UNAVAILABLE`，不得静默训练新模型替代。

### 3.3 冻结模型

加载成功后：

```python
model.eval()

for parameter in model.parameters():
    parameter.requires_grad_(False)
```

所有诊断在 `torch.inference_mode()` 下运行。

不创建 Optimizer。

不使用 GradScaler。

保持 FP32，不使用 FP16 Autocast。

诊断前后计算模型权重哈希，必须完全一致。

---

## 4. D0 — Baseline Integrity

### 4.1 目的

确认重新加载的 C1 与之前训练完成的模型一致。

否则，后续任何消融结果都没有可靠依据。

### 4.2 验证集

首先精确复用原训练报告中的：

`development[:16]`

保留相同文档顺序、Tokenizer、事件顺序、K=4 和文档重置规则。

不使用新的确认数据。

### 4.3 基础执行

每个独立文档：

1. Reset State。
2. 按顺序输入 Token。
3. 每个 Token 注入一次 Event。
4. 执行 K=4 次 Cellular Evolution。
5. Readout 预测下一个 Token。
6. 累计 Token NLL。
7. 文档结束后 Reset。

不得读取任何未来 Token。

### 4.4 验收条件

预期：

`Validation NLL ≈ 5.8146263923`

`Validation PPL ≈ 335.16615445`

建议 NLL 绝对误差容差为 `1e-4`。若存在合理的浮点归约差异，可先调查，但不能在未解释差异的情况下放宽并宣布通过。

同时检查：

- `model.eval()` 确实生效；
- Model Weight Hash 不变；
- Readout 不改变 State；
- 同一输入重复执行结果一致；
- 完整流式执行与分段执行数值一致；
- Snapshot/Restore 保持后续预测一致。

D0 失败时终止后续诊断。

---

## 5. D1 — State Causality

### 5.1 研究问题

我们需要判断：

> 模型当前的预测是否依赖此前事件所形成的 Cellular State？

这里需要区分两个层次：

**状态依赖**：改变 State 会改变模型输出。

**有用的历史状态**：此前状态能够帮助模型对正确的后续 Token 给出更高概率。

前者只能说明模型依赖 State。

后者才能说明 State 具有实际信息价值。

### 5.2 实验 D1-A：自然语言 State Reset

选取现有开发集文档。

对同一输入分别运行：

| 条件 | 行为 |
|---|---|
| Normal | 正常跨 Token 保留 State |
| Reset Every Token | 每个 Token 前清零 State |
| Reset at Probe | 只在预先选择的位置清零一次 |
| Entry Only | 保留入口 Cell，清零其他 Cell |

主要指标：

`ΔNLL = NLL_intervention - NLL_normal`

当 ΔNLL 为正，表示干预后的预测质量下降。

应重点报告 `Reset at Probe` 与 `Entry Only`。

`Reset Every Token` 属于极端干预，可以验证历史状态路径，但因为破坏了原训练分布，不应独立作为长期记忆成功的证据。

### 5.3 实验 D1-B：相同后缀、不同历史

这是更重要的 State 实验。

构造配对输入：

**Sequence A**

`Alice put a red ball in the box. Then everyone went outside.`

**Sequence B**

`Alice put a blue ball in the box. Then everyone went outside.`

之后给两个状态完全相同的后续输入：

`Later, Alice opened the box. The ball was`

预测目标分别为 `red` 和 `blue`。

执行方式：

1. 分别处理两个历史 Prefix。
2. 保留处理完 Prefix 的 Cellular State。
3. 对两条序列注入完全相同的 Suffix Event。
4. 在 Suffix 完成后，从状态读取预测。
5. 比较正确目标与错误目标的条件概率。
6. 将 Prefix State 交换后重复。
7. 将 Prefix State 清零后重复。

可以使用现有的 `history_test()` 作为起点，但需要完善实验控制。

### 5.4 配对任务规则

必须保证：

- 同一对测试具有完全相同的 Suffix Token ID；
- Prefix 的目标事实之外不引入额外语义差异；
- 交换 State 不改变 Suffix；
- Tokenizer 的分段编码不会产生不一致的边界；
- 比较的目标具有可比 Token 长度；
- 颜色、人物和顺序进行平衡；
- 不通过词频偏差判断成功。

建议先使用 8～16 个配对样本。

对于每对样本，计算正确事实与替代事实的对数概率差。

例如：

`Margin = log P(red | red-history, suffix) - log P(blue | red-history, suffix)`

在交换历史时，对应目标应该发生相应变化。

同时记录：

- Normal Margin；
- Reset Margin；
- Swapped-State Margin；
- 正确目标的 NLL；
- 配对方向是否正确。

如果 Prefix 的 Token 长度不同，应明确排除、调整或单独报告，不得将不同事件数量造成的差异直接解释为事实记忆。

### 5.5 结果解释

**情况 A：Normal 优于 Reset，Swap 导致正确的方向变化**

说明模型在这类受控输入中表现出有用的历史状态依赖。

**情况 B：Reset 降低总体语言质量，但事实配对没有正确变化**

说明 State 对语言预测有作用，但尚无证据表明它能够稳定保存特定语义事实。

**情况 C：State 交换改变输出，但未改善事实预测**

说明 State 具有因果作用，但无法证明这种作用是有用的语义记忆。

**情况 D：几乎无差异**

当前训练阶段没有观察到明显的 State 作用。由于模型只训练了 250K Token，不能直接推断整个 State-first 架构不可行。

本实验属于开发诊断，不能单凭结果将 H1 判为正式 SUPPORTED。

---

## 6. D2 — Connection Causality

### 6.1 研究问题

判断 C1 是否真正利用 Cell 间的局部信息传递。

C1 采用：

`Neighbors(i) = {i-1, i, i+1}`

每个 Cell 通过共享更新规则处理邻域消息。

如果模型主要依赖入口 Cell，那么局部通信可能只起到很有限的作用。

### 6.2 实验 D2-A：完整通信消融

在相同文档上比较：

| 条件 | 含义 |
|---|---|
| Normal | 完整局部通信 |
| Replace Neighbors with Self | 将左右邻居状态替换为当前 Cell 自身状态 |
| Remove Cross-Cell Terms | 移除左右跨 Cell 消息，仅保留 Self 分量 |

注意现有 `communication=False` 的具体语义：

它并不是简单移除 Connection，而是让 `left`、`right`、`center` 三个投影都读取同一个 Cell 的状态。

因此，该模式应准确命名为：

`NEIGHBORS_REPLACED_BY_SELF`

不要直接命名为 `NO_CONNECTION`。

如果需要真正的跨 Cell 消息移除，应增加独立的推理干预模式。

该新增模式只能改变消息输入或连接边，不得修改已训练参数。

### 6.3 实验 D2-B：单事件通信干预

为区分“长期改变状态轨迹”和“当前事件需要通信”，增加单事件干预。

步骤：

1. 使用正常通信处理此前全部 Event。
2. 保存当前位置的 State。
3. 从相同 State 出发，分别处理当前 Event。
4. 第一条使用正常通信。
5. 第二条仅在当前 Event 关闭跨 Cell 消息。
6. 比较后续 Token 的预测分布和正确目标 NLL。

这样可以避免在整个历史序列上关闭通信所造成的累积状态分布偏移，因果解释更明确。

主要指标：

- `ΔNLL_current_event`
- 预测分布变化；
- Cell State Difference；
- 信息向不同 Cell 的传播变化。

### 6.4 实验 D2-C：Cell 状态敏感性

选择少量具有代表性的事件位置。

从相同基础 State 出发，分别清零某个非入口 Cell，观察后续预测变化。

记录：

- 被干预的 Cell ID；
- 与入口 Cell 的拓扑距离；
- State Norm；
- 干预前后 NLL；
- 输出分布差异。

不必针对所有 Token 执行全部 16 个 Cell 的完整消融。

只需选择少量固定位置，优先比较入口、近邻和较远的 Cell。

### 6.5 结果解释

如果关闭跨 Cell 通信后，语言质量明显下降：

可以说明当前训练模型对跨 Cell 消息具有因果依赖。

但**不能直接说明 Cellular 结构优于 Flat RNN**，也不能仅根据这种干预证明有连接模型优于一个重新训练的无连接模型。

如果通信消融影响很小：

说明当前模型可能尚未充分利用跨 Cell 计算，也可能存在 Readout 或训练任务允许的其他计算路径。

这将成为后续是否研究 Adaptive NCA 的重要依据。

本次不训练新的 C0，也不重做 H2 正式确认。

---

## 7. D3 — Internal Evolution

### 7.1 研究问题

这是与 Reasoning NCA 关联最直接的部分：

> 没有新 Event 时，Cellular State 继续演化，是否能够改善模型的计算结果？

必须区分“状态发生变化”和“计算结果更好”。

### 7.2 实验 D3-A：不同内部计算步数

对同一开发数据，比较：

`K ∈ {2, 4, 8}`

其中 K=4 是模型训练配置。

K=2 和 K=8 都属于未针对该计算深度独立训练的执行方式。

报告：

- NLL；
- PPL；
- State Norm；
- State Delta；
- 推理延迟；
- 数值稳定性。

### 7.3 实验 D3-B：仅对最后一个事件继续演化

为了分离历史状态变化与当前内部计算深度，增加一个更严格的实验。

1. 过去所有 Event 均使用 K=4。
2. 保存目标事件处理前的 State。
3. 使用相同事件和相同状态分别执行不同步数。
4. 比较输出。

特别需要比较：

- K=4：正常预测。
- K=4 后追加 4 次无 Event 演化：额外内部计算。
- K=4 后追加 8 次无 Event 演化：更长内部计算。

所有候选必须从同一个正常状态出发。

不要重新注入最后一个 Token。

### 7.4 核心指标

定义：

`Gain(K) = NLL(K=4) - NLL(K)`

如果 Gain 为正，代表该位置在更多内部计算后 NLL 改善。

同时记录额外计算引起的 State 变化：

`ΔS(K) = ||S_K - S_4||`

观察：

- 状态是否继续变化；
- 预测是否继续变化；
- NLL 是否改善；
- 改善是否具有一致性；
- 是否发生过度演化导致预测恶化。

### 7.5 重要解释边界

额外计算改善 Next-Token NLL，不等于证明实现了 Reasoning。

真正的推理能力需要可验证任务或明确监督。

本实验最多说明：

- 内部状态可以自主演化；
- 额外演化对预测具有何种影响；
- 当前固定 K=4 是否可能存在计算不足或过度演化。

若 K=8 更差，也不说明 NCA 不具备推理潜力，因为模型并没有针对 K=8 进行训练。

---

## 8. 评估数据与成本控制

### 8.1 最小评估方案

为了减少 GPU 消耗，建议使用两级流程。

**第一层：基础诊断**

使用原有 16 篇开发文档，完成 D0、D1-A、D2-A 和 D3-A 的主要检查。

D1-B 使用 8～16 个受控配对输入。

D2-B、D2-C 和 D3-B 只选择固定的少量事件位置进行诊断。

**第二层：可选扩展**

仅在第一层结果有意义但不够稳定时，扩展到最多 64 篇已有开发文档。

不重新下载或生成新数据。

不要自动运行完整开发数据集。

### 8.2 计算复用

同一文档的正常前向状态轨迹可以缓存，用于多个诊断。

例如：

- Normal Baseline；
- Reset at Probe；
- Single-Event Communication Intervention；
- Extra Evolve。

这些实验可以从相同的已保存中间状态分叉执行。

不应为每一个干预重新计算完整文档历史。

但是涉及完整轨迹干预的实验，例如 `Reset Every Token` 或整个序列禁用通信，必须独立执行对应前向过程，不能假装使用原有正常状态轨迹即可得到结果。

### 8.3 GPU 约束

- 单模型 C1；
- FP32；
- 推理模式；
- 无梯度；
- 无 Optimizer；
- 不保存大量完整 State Tensor；
- 不执行大规模 Kernel Profiling；
- 不重复训练；
- 不重复 CPU Benchmark；
- 诊断计算时间设置独立上限，并记录各子任务耗时。

模型资产恢复和网络下载耗时需要单独记录，不能混入模型推理吞吐量。

若 Kaggle Output 恢复本身出现超时、429 或暂时无法定位文件，应保留诊断进度并停止，不得重新训练代替。

---

## 9. 统计与报告

### 9.1 主要指标

**语言质量：**

- Baseline NLL / PPL；
- Reset ΔNLL；
- Connection ΔNLL；
- K Sensitivity ΔNLL。

**State 诊断：**

- Prefix-State Swap 的方向正确率；
- 配对目标 Margin；
- State Norm；
- State Difference；
- 单 Cell 干预敏感性。

**通信诊断：**

- Full-trajectory Communication Ablation；
- Single-Event Communication Ablation；
- Cross-Cell Message Contribution。

**内部计算诊断：**

- 额外演化步数；
- NLL 改善或退化；
- 演化状态变化；
- 每 Token 推理延迟。

### 9.2 统计方式

对于自然语言开发文档，优先比较同一文档上的配对 NLL 差异。

报告：

- 平均差异；
- 文档级配对 Bootstrap 95% 区间；
- 实际文档数；
- 实际 Token 数；
- 方向一致的文档比例。

对于少量合成配对任务，报告每对结果与描述性统计，不使用它们作正式显著性验证。

16 篇开发文档的数据量较小，因此任何区间估计都只能视为探索性证据，不能作为正式 H1/H2/H5 的确认结果。

### 9.3 状态分类

报告状态应分为：

**Execution Status**

- COMPLETE
- INCOMPLETE
- BLOCKED_MODEL_ASSET_UNAVAILABLE
- FAILED_BASELINE_INTEGRITY
- FAILED_NUMERICAL_CHECK

**Diagnostic Findings**

- STATE_DEPENDENCE_OBSERVED
- SEMANTIC_HISTORY_SIGNAL_OBSERVED
- COMMUNICATION_DEPENDENCE_OBSERVED
- INTERNAL_EVOLUTION_GAIN_OBSERVED
- NO_CLEAR_SIGNAL
- INCONCLUSIVE

这些是开发诊断标签，不等同于正式科学结论。

Core-001 原有 H1–H5 状态保持 `NOT_EVALUATED`，直到新的正式确认协议完成。

---

## 10. 代码实施

### 10.1 新增诊断模块

建议在现有项目内新增：

```text
epclm-core-001/
  C1-MECHANISM-DIAGNOSTIC.md
  implementation/
    frozen_model.py
    mechanism_diagnostic.py
  reports/
    mechanism-diagnostic/
```

可以复用现有 `diagnostics.py`、`runtime.py`、`model.py` 和 `trainer.evaluate()` 的通用逻辑。

不需要重构训练架构。

### 10.2 新执行入口

推荐：

```bash
python research/projects/cellular-language-models/experiments/epclm-core-001/run.py \
  --diagnose-c1 \
  --output /kaggle/working/epclm-core-001-c1-diagnostic
```

该入口必须：

1. 验证模型来源。
2. 恢复冻结 C1 权重。
3. 完成 D0。
4. 完成 D1–D3。
5. 保存诊断结果。
6. 发布汇总报告。
7. 正常退出。

不得自动转入训练模式。

### 10.3 运行身份

模型训练身份与本次诊断身份必须独立保存。

例如：

```json
{
  "experiment": "epclm-core-001-c1-diagnostic",
  "mode": "FROZEN_MODEL_DIAGNOSTIC",
  "training_source": "398a191dbb32e383c86c89b9c981777c7cd6c77f",
  "training_weight_sha256": "3787241e0ea18a206fb9625828d512845dce8b7156908255792fe420f12eacbf",
  "diagnostic_source": "CURRENT_COMMIT_SHA",
  "training_tokens": 250000,
  "optimizer_updates": 0,
  "formal_confirmation": false
}
```

必须同时记录本次诊断协议哈希、数据 Manifest 哈希以及各干预配置。

### 10.4 测试要求

遵循最小测试成本原则。

只增加必要的 CPU 快速断言：

- 冻结模型权重未发生变化；
- D0 可以复现基础结果；
- 状态分叉不会互相污染；
- 通信干预只改变指定连接；
- 额外 Evolve 不重复注入 Event；
- 禁止进入训练入口。

不增加大型本地测试矩阵。

---

## 11. 结果文件

建议发布结构：

```text
reports/mechanism-diagnostic/
  decision.json
  summary.md
  baseline.json
  state-causality.json
  connection-causality.json
  internal-evolution.json
  mechanism-diagnostic.png
```

私有 Kaggle Output 可以额外保存：

- 单文档 NLL；
- 受控历史配对明细；
- 指定事件的 Cell 状态轨迹；
- 调试记录。

公开 GitHub 结果只发布必要的汇总数据、指标、图表和实验身份。

不发布模型 Checkpoint、Optimizer State、完整数据集或不必要的原始 Token 记录。

### 11.1 图表

建议生成一张包含三个独立分区的诊断图：

**State Causality**

Normal / Reset / Swap 的 NLL 差异。

**Connection Causality**

Normal / Neighbor-Replaced / Cross-Cell-Removed 的 NLL 差异。

**Internal Evolution**

K=2 / K=4 / K=8 的 NLL 和计算成本。

另可附一张有限状态热力图，仅用于辅助观察，不作为语义解释依据。

---

## 12. 停止条件

以下情况必须停止：

1. C1 Checkpoint 不可获得或哈希不匹配。
2. 加载权重后的 Model Weight Hash 不匹配。
3. D0 无法复现原有验证结果。
4. 检测到模型参数发生变化。
5. 输入数据与原 Data Manifest 不一致。
6. 发生非有限 State 或 Logits。
7. 诊断超出执行预算。
8. Kaggle Output 恢复无法完成。

失败时保留已完成诊断，不自动重试整个实验，不重新训练模型。

---

## 13. 实验结果如何影响下一步决策

| 观察结果 | 建议下一步 |
|---|---|
| State 与通信均有明显作用 | 继续扩大语言训练，开展匹配基线实验 |
| State 有效，通信作用较弱 | 优先研究 Connection 设计 |
| State 效果不明显 | 优先检查 Event 注入与状态保持机制 |
| K=8 有持续质量收益 | 单独设计计算深度训练实验 |
| K=8 退化明显 | 研究长迭代稳定性，暂不直接增加推理轮数 |
| 机制表现不明确 | 保留 Core，进行针对性诊断，不盲目扩大模型 |

这里应特别注意：

**D1、D2 或 D3 中任意一项没有观察到收益，并不自动意味着 EPCLM 研究方向失败。**

本次使用的是仅训练 250K Token 的初始模型。模型语言质量仍然较低，且部分干预超出了训练时的状态分布。

因此，本次的目的主要是寻找机制证据和架构瓶颈，而不是给整个研究方向作最终判断。

---

## 14. Codex 实施验收

本次开发完成时必须满足：

- [ ] 可以从现有 Kaggle Output 找到并验证原始 C1 Checkpoint。
- [ ] 不进行任何新训练。
- [ ] 不改变 C1 的模型架构和参数。
- [ ] D0 准确复现原语言质量。
- [ ] D1 包含自然语料 State 干预和受控历史配对。
- [ ] D2 区分邻居替换与真实跨 Cell 消息移除。
- [ ] D3 区分全序列改变 K 与最后事件额外演化。
- [ ] 结果包含配对差异和明确的解释边界。
- [ ] 报告记录真实推理成本和模型身份。
- [ ] H1–H5 不提前标记为 SUPPORTED。
- [ ] 运行成功后将结果发布到单独的 GitHub 结果分支。
- [ ] 不改写此前训练结果或历史实验报告。

---

## 15. 最终研究目标

Core-001 的主要价值不在于把语言历史简单存进一个固定数组。

我们真正想验证的是：

**一个持续存在的 Cellular State，是否能够通过 Cell 间通信和多轮内部演化形成有效计算，并且让这些计算结果跨 Event 延续。**

本次 C1 Mechanism Diagnostic 不需要证明 Cellular Network 已经优于其他语言模型。

它需要回答三个更加基础的问题：

**第一：State 是否真的承担了有用的跨事件计算？**

**第二：Cell 之间是否真的形成了有作用的计算协作？**

**第三：内部计算继续演化时，是否能产生比固定计算深度更好的结果？**

这三个答案将决定下一阶段应该优先研究 State、Connection 还是 Internal Computation。

本诊断完成后，再决定是否进入 Core-001 的 1M/10M Token 扩展训练。

**最终原则：先理解已经训练出来的模型，再决定如何改进它。**
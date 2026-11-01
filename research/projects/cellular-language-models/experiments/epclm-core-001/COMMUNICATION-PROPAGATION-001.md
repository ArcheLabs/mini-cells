# MiniCells — EPCLM Core-001 / Communication Propagation-001

**跨事件 Cellular Communication 传播实验设计 v0.1**

- **实验编号**：EPCLM-CORE-001-CP-001
- **阶段**：Development / Frozen Model Mechanism Research
- **状态**：PROPOSED
- **实验模型**：C1 Fixed Local Cellular Network
- **执行环境**：Kaggle Tesla T4 / PyTorch FP32
- **模型训练**：禁止
- **实验性质**：冻结模型、只读前向、因果扰动与传播分析
- **前置研究**：Core-001 C1 Mechanism Diagnostic

---

## 1. 研究背景

### 1.1 已有结果

Core-001 已经完成 250K Token 语言训练，以及一次冻结模型机制诊断。

机制诊断源码：

`7ee5cd46e10c29bf70854c3cd52a232fa9da1adc`

[机制诊断报告](https://github.com/ArcheLabs/mini-cells/blob/kaggle/epclm-core-001-c1-diagnostic-7ee5cd46e10c/artifacts/experiments/epclm-core-001/mechanism-diagnostic/decision.json)

主要结果：

| 诊断 | ΔNLL |
|---|---:|
| 每个 Token 重置 State | +1.27364 |
| 单次 State Reset | +0.05913 |
| 只保留入口 Cell | +0.11230 |
| 整个序列移除跨 Cell 通信 | +0.11649 |
| 整个序列将邻居替换为自身 | +0.10754 |
| 仅在单个 Event 移除跨 Cell 通信 | −0.00021 |

其中 ΔNLL 定义为：

`NLL_intervention - NLL_normal`

正数代表干预使预测质量下降。

已有结果说明：

1. 模型依赖跨事件保留的 State。
2. 跨 Cell 通信对长期语言预测具有实际作用。
3. 对单个 Event 施加通信干预时，当下的平均预测质量变化很小。

第三个现象尤其值得进一步研究。

但目前还不能据此认定通信的收益一定延迟到未来 Event 才出现。我们只测量了当下的预测，没有继续观察干预后的状态如何演化。

### 1.2 本实验的问题

本实验希望回答：

> 当某个 Event 中的 Cell 通信受到一次局部干预后，这种变化如何在 Cellular Network 中传播、衰减或保留，并如何影响之后多个 Event 的计算结果？

需要将三个问题分开：

**Q1 — Spatial Propagation**

某条 Cell 连接造成的状态扰动，是否会沿环形邻域逐渐传播到其他 Cell？

**Q2 — Temporal Persistence**

扰动是否在后续 Event 中持续存在？其强度如何随时间变化？

**Q3 — Predictive Consequence**

扰动是否会影响后续 Token 的预测？影响是改善、退化，还是只有数值变化而没有可测量的质量变化？

---

## 2. 科学假设

### H-CP1：局部状态扰动具有空间传播特征

对单条有向 Cell 连接进行一次干预后：

- 初始差异只发生在对应的接收 Cell；
- 后续 Cellular Update 可以使差异到达邻近 Cell；
- 传播范围应符合实际的局部连接拓扑。

这首先是计算机制问题，不是语言能力问题。

### H-CP2：通信扰动具有跨事件延续性

在 Event `t` 发生干预后，恢复全部正常通信。

如果后续 Event 的状态仍与原始执行路径不同，就说明该次通信扰动产生了跨事件持续影响。

需要观察这种影响如何增长、衰减或消失。

### H-CP3：跨事件影响能够改变未来语言预测

对于相同的未来 Token 序列，干预路径和正常路径可能给出不同的预测分布。

这种影响必须通过未来事件上的预测差异直接测量。

### H-CP4：传播影响具有实际语言质量价值

只有当干预导致后续目标 Token 的 NLL 在配对比较中持续恶化，才可以进一步提出“该通信对未来预测具有有用贡献”的开发阶段证据。

单纯的 State Difference 或 Logit Difference 不足以支持 H-CP4。

本次 H-CP1 至 H-CP4 均属于探索性假设，不属于 Core-001 正式 H1–H5 确认。

---

## 3. 实验边界

### 3.1 必须遵守

- 复用现有 C1 冻结模型。
- 不进行训练。
- 不执行反向传播。
- 不更新模型权重。
- 不修改 Cell 数量、状态维度或默认 K。
- 不修改原始训练 Checkpoint。
- 不使用 Transformer、F0、C0 或 L0 作为本次额外训练对象。
- 不使用正式 Confirmation 数据集。
- 不进行架构搜索或调参。
- 所有比较使用完全相同的后续 Token。
- 每次干预都必须从与 Baseline 完全相同的状态开始。

### 3.2 模型身份

沿用 Core-001 已有的 C1 模型：

| 项目 | 固定值 |
|---|---|
| Training Source | `398a191dbb32e383c86c89b9c981777c7cd6c77f` |
| Training Tokens | 250,000 |
| Cell Number | 16 |
| State Dimension | 128 |
| Persistent State | 2,048 FP32 数值 |
| Topology | 1D Ring, Radius 1 |
| Event | Token |
| Internal Evolution | K=4 |
| Precision | FP32 |

原始模型权重 SHA-256：

`3787241e0ea18a206fb9625828d512845dce8b7156908255792fe420f12eacbf`

Checkpoint SHA-256：

`36a58f642f93b859ce33f5fd55bd92c9f4196669dc8c62293088ea43ce8e7fd1`

允许复用上次机制诊断已经实现的 `frozen_model.py` 恢复路径。

此前已验证原始模型可从 Kaggle Saved Output version 142 恢复。

优先访问这个已知版本并校验哈希，不必每次重新扫描大量历史 Kaggle Version。

如该版本无法获取，按现有安全恢复规则处理，不允许重新训练替代。

---

## 4. 实验的基本计算定义

### 4.1 双时间尺度

定义：

- `t`：外部 Event 序号。
- `k`：某个 Event 内部的计算轮次。
- `i`：Cell 序号。
- `S(t,k)`：第 t 个 Event、第 k 次内部更新后的完整 Cellular State。

正常计算：

`S(t,0) = Inject(S(t-1,K), event_t)`

`S(t,k+1) = CellularUpdate(S(t,k))`

`S(t+1) = S(t,K)`

其中 K=4。

### 4.2 正常路径

设正常执行状态为：

`S_normal(t,k)`

模型使用完整通信规则处理所有 Event。

### 4.3 干预路径

设干预执行状态为：

`S_intervention(t,k)`

两条路径共享：

- 完全相同的模型权重；
- 完全相同的初始 State；
- 完全相同的 Event 序列；
- 完全相同的 Tokenizer；
- 完全相同的计算精度；
- 完全相同的默认更新次数。

唯一差别是：

**指定 Event 的指定通信操作受到一次干预。**

干预后立即恢复正常运行。

### 4.4 核心测量

每个未来时间点记录：

`DeltaState(h) = S_intervention(t+h) - S_normal(t+h)`

`DeltaNLL(h) = NLL_intervention(t+h) - NLL_normal(t+h)`

其中 `h=0` 代表干预 Event 本身，`h>0` 代表之后的 Event。

计算预测差异时：

- 两条路径都必须实际消费相同的 `event_(t+h)`；
- 然后读取下一个 Token 的预测；
- 不允许提前使用未来 Token；
- 不允许让模型各自采样后再比较不同生成序列。

---

## 5. 实验 A — Global Single-Event Communication Pulse

### 5.1 目的

直接检验已有现象：

**只关闭一次通信，是否会影响未来多个 Event？**

此实验主要观察时间效应。

### 5.2 干预操作

在选定的 Event `t`：

`communication = CROSS_CELL_REMOVED`

即：

- 在该 Event 的四轮更新中，全部移除左右邻居贡献；
- 保留当前 Cell 的 Self Contribution；
- 不修改模型参数；
- 不重置 State。

只有这一个 Event 使用干预。

从 Event `t+1` 开始，恢复完整通信。

### 5.3 对照执行

```text
Normal:
S(t-1) -> Event t [Normal]
       -> Event t+1 [Normal]
       -> Event t+2 [Normal]
       -> ...

Intervention:
S(t-1) -> Event t [Cross-Cell Removed]
       -> Event t+1 [Normal]
       -> Event t+2 [Normal]
       -> ...
```

两条路径必须从同一个 `S(t-1)` 出发。

### 5.4 未来观察窗口

固定观察：

`h = {0, 1, 2, 4, 8, 16}`

如果目标位置剩余 Token 不足 16 个，应缩短实际观察窗口并明确记录可用 Horizon。

不得跨越独立文档的 Reset 边界。

### 5.5 记录内容

每个 Horizon 记录：

- `delta_nll`
- `prediction_js_divergence`
- `prediction_total_variation`
- `state_difference_norm`
- `relative_state_difference`
- `cell_state_difference[16]`
- `state_norm_normal`
- `state_norm_intervention`

其中 JS Divergence 是预测分布之间的 Jensen–Shannon Divergence，用于观察预测是否发生变化。

即使 NLL 差异接近零，也可能存在明显的预测分布变化。

### 5.6 预期解释

**情况 A：h=0 影响很小，h>0 影响增加**

说明一次通信干预可能通过后续状态演化，对未来预测产生更明显的影响。

**情况 B：h=0 影响明显，之后逐渐衰减**

说明扰动具有即时影响，但模型状态可能具有一定恢复能力。

**情况 C：状态差异持续存在，预测差异始终很小**

说明模型存在状态传播或保持，但这些差异未必对应有用的语言计算。

**情况 D：状态与预测差异都迅速消失**

说明该类干预在当前模型中没有产生明显的持续影响。

不能仅凭其中任何一种情况，直接宣称模型形成了语义记忆或推理能力。

---

## 6. 实验 B — Local Edge Propagation

### 6.1 目的

实验 A 的不足在于：

它同时删除整个网络的跨 Cell 通信，因此所有 Cell 都可能在同一 Event 内发生变化。

这样的干预适合研究长期影响，但不能清楚揭示信息沿单条连接逐步传播的空间过程。

实验 B 应只干预一条有向连接。

### 6.2 单连接干预

在原始 C1 中，每个 Cell 的消息由三个部分构成：

`message_i = center(h_i) + left(h_(i-1)) + right(h_(i+1))`

对于指定接收 Cell `i`，本实验只在 Event `t` 的**第一轮内部更新**中移除一项：

`left(h_(i-1))`

因此：

`message_i_intervention = center(h_i) + right(h_(i+1))`

其他 Cell 的消息完全不变。

Event `t` 的第二轮至第四轮恢复全部连接。

后续 Event 也恢复正常。

### 6.3 为什么只干预第一轮？

如果对整个 Event 的四轮都移除一条连接，就会在四个不同的内部时间点不断注入扰动。

这不利于观察单个通信事件的传播。

因此，实验 B 必须采用一次性的：

`EDGE_DROP_ONCE(event=t, internal_step=0, receiver=i, source=i-1)`

其中 `internal_step=0` 表示 Event 注入后的第一轮更新。

### 6.4 固定测试连接

为了控制成本，选择四个接收 Cell：

`receiver ∈ {0, 1, 4, 8}`

干预连接均为：

`source = (receiver - 1) mod 16`

即：

| Source | Receiver |
|---|---|
| Cell 15 | Cell 0 |
| Cell 0 | Cell 1 |
| Cell 3 | Cell 4 |
| Cell 7 | Cell 8 |

这样可以观察接近入口、靠近入口和远离入口的不同连接。

不能根据试验结果动态选择表现最明显的连接，再将其作为预先指定的成功证据。

---

## 7. 空间传播分析

### 7.1 Cell Difference

对每个 Cell：

`D_i(t,k) = ||h_i_intervention(t,k) - h_i_normal(t,k)||₂`

需要分别记录每一轮内部更新后的结果。

例如：

| 时间点 | 理论上可能受到影响的区域 |
|---|---|
| Event t，Update 1 | 仅 Receiver |
| Event t，Update 2 | Receiver 及距离 1 的 Cell |
| Event t，Update 3 | Receiver 及距离不超过 2 的 Cell |
| Event t，Update 4 | Receiver 及距离不超过 3 的 Cell |
| Event t+1，Update 4 | 距离不超过 7 的 Cell |
| Event t+2 | 可能到达整个 16-Cell 环 |

这里的范围是当前局部更新计算图的理论可达范围，不代表所有可达 Cell 都必然产生非零差异。

环形网络最远距离为 8。因此，在 K=4 的配置下，空间传播可能在很少的 Event 内覆盖整个网络。

这也是本实验必须保留**内部更新轮次轨迹**的原因。

### 7.2 Causal Cone Validation

对于单连接干预，第一轮更新结束时，只有接收 Cell 允许发生状态差异。

后续差异只能通过现有连接向外传播。

实验必须检查：

- 不存在超出理论传播范围的状态变化；
- 第一轮干预仅改变指定 Receiver；
- 所有未干预连接保持正常；
- 传播边界满足环形拓扑距离约束。

数值比较采用预先固定的 FP32 容差，并以未干预的 SHAM 路径校验数值噪声水平。

若发现超出因果传播范围的差异，应优先判为实现问题，而不是发现新的传播机制。

### 7.3 传播强度

记录：

- 每个 Cell 的绝对状态差异；
- 相对于正常 State Norm 的差异；
- 每轮出现可测量差异的 Cell 数量；
- 影响到达不同拓扑距离的最早轮次；
- 差异幅度随时间的变化；
- 事件结束时的扰动总强度。

建议分别计算：

`AbsoluteDifference_i = ||Δh_i||₂`

`RelativeDifference_i = ||Δh_i||₂ / (||h_i_normal||₂ + epsilon)`

用于判断扰动是否真正扩散，而非仅因为状态范数较大导致视觉上显著。

### 7.4 非常重要的解释限制

当前 Readout 会聚合所有 Cell 的状态。

所以，即使一个扰动只影响单个 Cell，也可能立即改变最终预测。

因此：

**预测变化发生得早，不代表扰动已经传播到其他 Cell。**

空间传播的主要证据必须来自各 Cell 状态差异的演化，而不能仅依靠输出 Logit 或 NLL。

---

## 8. 实验 C — Future Prediction Impact

### 8.1 目的

实验 B 可以显示局部计算传播。

但真正重要的问题是：这些变化是否影响未来语言预测？

实验 C 使用与实验 B 相同的单连接干预，同时观察未来 Event。

### 8.2 执行方式

从干预 Event `t` 开始：

1. 正常路径处理 Event。
2. 干预路径仅在第一轮移除指定连接。
3. 两条路径分别产生完整 Event 后状态。
4. 随后的 Event 全部恢复正常通信。
5. 两条路径消费完全相同的未来 Token。
6. 在固定 Horizon 比较语言预测。

### 8.3 固定观察窗口

`h = {0, 1, 2, 4, 8, 16}`

每个 Horizon 同时记录：

- 状态差异；
- 输出分布差异；
- 正确目标 Token 的 NLL 差异；
- JS Divergence；
- 当前扰动可达的 Cell 数。

### 8.4 两种不同的效应

**Prediction Sensitivity**

预测分布发生变化：

`JS(P_normal, P_intervention) > numerical_noise`

这说明该连接的干预影响了当前或未来的输出。

**Predictive Utility**

干预使正确目标的 NLL 上升：

`ΔNLL(h) > 0`

这说明该事件位置上的连接对正确预测有帮助。

两者必须分开报告。

不能将所有预测分布变化都解释为模型质量退化。

### 8.5 建议增加一个累计指标

除了每个 Horizon 的单点 NLL，还应计算未来窗口内的累计差异：

`CumulativeDeltaNLL(H) = mean(ΔNLL(h), h=0..H)`

例如 H 为 4、8、16。

单点结果可能正负波动，而累计结果能够更容易观察该次通信干预是否对后续一段语言预测产生系统性影响。

此外报告平均绝对 NLL 差异，避免正负抵消掩盖真实敏感性。

---

## 9. 对照与数值正确性

### 9.1 SHAM 干预

加入一个无实际变化的分支：

`SHAM = clone_state + normal_execution`

它执行与干预分支相同的状态复制和推理流程，但不改变 Connection。

预期：

- State Difference ≈ 0；
- Logit Difference ≈ 0；
- ΔNLL ≈ 0。

如果 SHAM 产生不可忽略的差异，先解决确定性问题，不进入传播解释。

### 9.2 前向状态一致性

正常路径必须能够复用原始 C1：

`model.step(state, token, K=4)`

诊断代码不得悄悄改变正常更新路径。

### 9.3 冻结权重

干预只能修改运行时传递的信息。

不得修改：

- `left.weight`
- `right.weight`
- `center.weight`
- `update` 参数
- Embedding
- Readout

实验前后计算完整模型的 `weight_hash()`。

必须完全一致。

### 9.4 相同未来输入

所有分支使用 Teacher Forcing。

不能让两个分支分别生成 Token 后比较，因为不同生成 Token 会引入额外因果因素。

### 9.5 不重复注入 Event

执行额外状态更新时：

`Evolve(State, steps)`

不得重新执行：

`Inject(Event)`

这项约束同样适用于对内部轮次的传播观察。

---

## 10. 数据选择

### 10.1 数据来源

复用上次机制诊断的原始开发文档：

- 16 篇 TinyStories；
- 共 3,322 个有效预测 Token；
- 原有 Tokenizer；
- 原有文档顺序；
- 原有 Story ID。

不创建新训练集。

不访问正式确认集。

### 10.2 干预位置

每个文档选择三个位置：

- 25%；
- 50%；
- 75%。

最多得到：

`16 × 3 = 48` 个基础 Event Probe。

与上一轮机制诊断保持一致，方便对比。

如果某个位置距离文档末尾不足 16 个 Event，记录可用 Horizon，并在各 Horizon 的统计中注明实际样本数量。

### 10.3 实验规模

实验 A：

`48 probes × 1 global intervention`

实验 B/C：

`48 probes × 4 local edges`

合计最多：

`240 intervention branches`

所有分支只执行少量前向计算，不训练。

单连接实验 B 和 C 必须使用同一次干预轨迹，不能分别重新运行相同模型计算。

---

## 11. 运行性能优化

### 11.1 缓存正常轨迹

先对 16 篇文档执行一次正常推理。

只在指定的 48 个 Probe 位置缓存：

- Event 之前的完整 State；
- Event Token；
- 后续最多 17 个输入 Token；
- 对应目标 Token；
- 正常预测；
- 必要的正常 State 轨迹。

每个 State 仅约 8 KiB。

缓存 48 个起始状态的成本很低。

### 11.2 从相同起点分叉

每个 Probe：

```python
initial = saved_state.clone()

normal = replay_normal(
    initial.clone(),
    future_tokens,
)

global_pulse = replay_global_pulse(
    initial.clone(),
    future_tokens,
)

local_pulse = replay_local_edge_pulse(
    initial.clone(),
    future_tokens,
    receiver=0,
)
```

其余接收 Cell 使用相同起点重复局部干预。

不需要对每个分支重新处理完整历史文档。

### 11.3 GPU 使用原则

- FP32。
- `torch.inference_mode()`。
- 不执行 Autocast。
- 不创建 Optimizer。
- 不计算梯度。
- 不执行 Kernel Profiling。
- 不创建大量中间日志。
- 只保留选中 Probe 的 Cellular 轨迹。
- 优先复用 Kaggle Version 142 的已验证模型资产。

不得因为缺少模型资产而自动启动训练。

---

## 12. 结果指标

### 12.1 时间传播

每个 Horizon：

| 指标 | 含义 |
|---|---|
| `mean_delta_nll` | 平均正确目标 NLL 变化 |
| `mean_abs_delta_nll` | 平均绝对 NLL 变化 |
| `mean_js_divergence` | 预测分布差异 |
| `mean_total_variation` | 预测概率变化 |
| `mean_state_difference` | 完整 State 差异 |
| `mean_relative_state_difference` | 相对状态差异 |
| `positive_delta_fraction` | NLL 退化的 Probe 比例 |

### 12.2 空间传播

每个 Cell、每个内部轮次：

- `receiver_id`
- `source_id`
- `cell_id`
- `ring_distance`
- `internal_step`
- `event_offset`
- `absolute_state_delta`
- `relative_state_delta`
- `within_theoretical_causal_cone`

### 12.3 累计预测影响

分别报告：

- H=4；
- H=8；
- H=16。

记录平均累计 ΔNLL、平均绝对 ΔNLL 和对应文档数。

### 12.4 统计原则

- 使用相同文档、相同位置的配对比较。
- 按 Story 聚类，不把 48 个 Probe 当作 48 个独立训练样本。
- 文档级 Bootstrap 可以报告探索性 95% 区间。
- 多个 Horizon 和 Cell 位置不作为未经调整的多重独立确认。
- 所有原始 Horizon 均必须报告，不能仅展示效果最佳的位置。
- 单 Seed 结果不支持架构普遍优越性的正式主张。

---

## 13. 建议的可视化

本实验至少生成两张研究图。

### Figure A — Temporal Pulse Response

横轴：

`Future Event Offset`

纵轴分别表示：

- State Difference；
- JS Divergence；
- ΔNLL。

区分：

- Global Single-Event Pulse；
- Local Edge Pulse。

应保留带正负号的 ΔNLL，并分别展示变化幅度，不能只绘制绝对值。

### Figure B — Cellular Propagation Heatmap

横轴：

`Cell ID: 0..15`

纵轴：

`Internal Update / Event Time`

颜色：

`Relative State Difference`

每张图显示一次预先选定的局部连接干预。

重点观察：

- 第一轮是否只影响接收 Cell；
- 后续是否向邻域扩散；
- 差异是否符合环形拓扑因果锥；
- 后续 Event 是否出现扩散、衰减或消失。

可附一张正常状态和干预状态的对照热力图，但不得将单个 Cell 的数值变化直接解释为明确的语义记忆。

---

## 14. 代码组织

建议在现有 Core-001 目录内继续开发：

```text
epclm-core-001/
  COMMUNICATION-PROPAGATION-001.md
  communication-propagation-protocol.json
  implementation/
    frozen_model.py
    mechanism_diagnostic.py
    communication_propagation.py
  reports/
    communication-propagation/
```

### 14.1 模型兼容性

复用现有：

- `Core`
- `DiagnosticCore`
- `frozen_model.py`
- `Runtime`
- `weight_hash`
- `load_checkpoint`
- Kaggle Frozen Asset Recovery

只增加单连接干预接口。

建议接口：

```python
state = diagnostic_core.evolve(
    state,
    steps=4,
    intervention={
        "type": "EDGE_DROP_ONCE",
        "receiver": 4,
        "source": 3,
        "internal_step": 0,
    },
)
```

该接口只用于推理诊断，不引入训练模式。

必须保证 `intervention=None` 时与原始 C1 前向路径等价。

### 14.2 独立执行入口

建议：

```bash
python research/projects/cellular-language-models/experiments/epclm-core-001/run.py \
  --communication-propagation \
  --output /kaggle/working/epclm-core-001-cp001
```

执行流程：

1. 校验 Diagnostic Source 与 Protocol。
2. 恢复已训练 C1。
3. 校验模型权重和数据哈希。
4. 运行少量 Baseline/SHAM 正确性检查。
5. 缓存正常 Probe State。
6. 执行 Global Pulse。
7. 执行 Local Edge Pulse。
8. 收集空间与时间传播数据。
9. 生成统计报告和图表。
10. 校验权重未改变。
11. 发布结果。
12. 正常退出。

不得自动执行后续训练。

---

## 15. 最小测试与执行预算

考虑当前研究优先级，继续减少不必要的本地测试。

### 15.1 本地测试

仅保留三个关键测试：

**Test 1 — No-op Equivalence**

不进行实际干预时，原始 Core 和 DiagnosticCore 产生相同结果。

**Test 2 — Single-Edge Isolation**

第一轮局部干预仅改变指定接收 Cell。

**Test 3 — Causal Cone**

后续传播不超出固定环形连接允许的理论范围。

这些测试使用极小状态规模即可，不需要完整语言训练。

不增加复杂的 CPU 测试矩阵。

### 15.2 GPU 预算

使用一次 Kaggle 执行完成所有诊断。

建议配置：

| 项目 | 约束 |
|---|---|
| 模型 | 已训练 C1 |
| 新训练 Token | 0 |
| Optimizer Updates | 0 |
| 文档数 | 16 |
| Probe 数 | 最多 48 |
| Local Edges / Probe | 4 |
| 最大 Future Horizon | 16 |
| 恢复预算 | 最多 600 秒 |
| 诊断预算 | 最多 900 秒 |
| 发布与安全余量 | 单独保留 |

以上是执行上限，不是预期运行时间。

如果某一部分已经完成，应保留可验证结果，恢复时跳过已完成部分。

不需要多次 GPU Pilot。

---

## 16. 结果判定

实验完成后，分别判断四项。

### 16.1 SPATIAL_TRANSMISSION_OBSERVED

支持条件：

- 单连接干预后，状态差异首先出现在指定 Receiver；
- 后续正常更新将差异传播至其他 Cell；
- 传播符合理论拓扑因果锥；
- 差异超过预先确定的数值噪声阈值。

该结果说明局部计算影响能够传播，不等于语义通信得到验证。

### 16.2 TEMPORAL_PERSISTENCE_OBSERVED

支持条件：

- 干预 Event 结束后；
- 所有通信恢复正常；
- 后续 Event 的状态仍保留可测量差异；
- SHAM 不出现相同规模的差异。

该结果说明通信扰动能够跨事件延续。

### 16.3 FUTURE_PREDICTION_EFFECT_OBSERVED

支持条件：

- 在 `h>0` 的一个或多个预定 Horizon；
- 预测分布出现超过数值噪声水平的变化；
- 相同未来 Event 输入下可重复观察到；
- 变化不是由不同生成 Token 引入。

该结果说明通信扰动对未来语言计算具有因果影响。

### 16.4 USEFUL_DELAYED_COMMUNICATION_SIGNAL

这是更严格的探索性标签。

只有当干预导致未来目标 Token 的 NLL 在跨文档配对比较中呈现一致的恶化方向，并且效应具有实际意义时，才允许使用该标签。

如果只是 JS Divergence 大于零，而平均 ΔNLL 接近零或方向不一致，只能报告 Future Prediction Effect，不能报告有用的延迟通信收益。

以上均为 Development Findings。

原有 Core-001 H1–H5 保持 `NOT_EVALUATED`。

---

## 17. 失败与停止规则

以下情况终止：

1. 冻结 C1 Checkpoint 不可获得。
2. Checkpoint / Weight Hash 不匹配。
3. SHAM 正常执行无法保持一致。
4. 单连接干预同时改变了其他不相关 Cell 的第一轮状态。
5. 传播超出理论因果锥。
6. 发生 NaN / Inf。
7. 模型权重发生修改。
8. 诊断预算耗尽。

失败时保留已完成的有效数据，但实验总状态不得错误标记为 COMPLETE。

禁止通过重新训练模型解决诊断失败。

---

## 18. 结果发布

建议输出：

```text
artifacts/experiments/epclm-core-001/
  communication-propagation/
    decision.json
    summary.md
    global-pulse.json
    local-edge-pulse.json
    temporal-impact.json
    spatial-propagation.json
    temporal-response.png
    propagation-heatmap.png
    metadata.json
```

详细逐 Probe 数据可以留在 Kaggle 私有 Output。

公开 GitHub 仅发布：

- 固定实验配置；
- 模型身份；
- 统计汇总；
- 必要的可复现数据；
- 图表；
- 明确的限制；
- 开发阶段结论。

不得发布模型权重或原始数据集。

建议独立结果分支：

`kaggle/epclm-core-001-cp001-<source-sha>`

不得覆盖既有 Mechanism Diagnostic 的结果。

---

## 19. 本实验与下一阶段研究的关系

本实验不是为了寻找新的更强语言模型，而是理解当前 C1 的计算动态。

根据结果，下一阶段应有所区别。

| 观察结果 | 后续研究方向 |
|---|---|
| 扰动能空间传播，但很快消失 | State Update 稳定性与信息保持 |
| 扰动长期存在，但预测影响很小 | State 可用信息与 Readout |
| 未来预测受影响，但质量方向不一致 | 通信选择性与训练目标 |
| 局部扰动对未来预测持续有益 | 扩大训练并进行匹配通信对照 |
| 扰动快速影响全部 Cell | 研究局部通信和稀疏执行的必要性 |
| 传播主要集中于少数 Cell | 研究动态路径与 Cell 分工 |
| 通信具有明显的延迟收益 | 进一步研究 Event-aware Communication |

这里不预先确定一定进入 Adaptive NCA、Path-First 或 Hierarchical Cellular。

新的机制应由实测瓶颈决定。

---

## 20. 最终验收清单

- [ ] 冻结 C1 模型恢复成功。
- [ ] 原模型权重哈希一致。
- [ ] 未执行任何训练与参数更新。
- [ ] Global Single-Event Pulse 完成。
- [ ] Local Edge Pulse 完成。
- [ ] SHAM 一致性通过。
- [ ] 第一轮单连接干预仅影响指定 Receiver。
- [ ] 空间传播符合因果锥。
- [ ] 未来 Horizon 0/1/2/4/8/16 均按可用范围测量。
- [ ] State Difference 与 Prediction Difference 分别报告。
- [ ] Signed ΔNLL 与 Absolute ΔNLL 分别报告。
- [ ] 配对统计按文档聚类。
- [ ] 可视化明确区分 Spatial 与 Temporal Propagation。
- [ ] 模型权重前后哈希完全一致。
- [ ] 所有结果标记为开发阶段探索性证据。
- [ ] H1–H5 不提前标记为正式通过。
- [ ] 实验结果成功发布独立 GitHub 分支。

---

## 21. 核心研究结论目标

CP-001 必须将以下三个层次严格区分：

**层次一：计算传播**

某个 Cell 的计算变化是否会传递到其他 Cell？

**层次二：跨事件保留**

这种变化是否在后续 Event 中继续存在，并改变网络状态？

**层次三：有效信息传播**

这种变化是否对未来正确语言预测具有可重复的有用贡献？

只有第三个层次获得积极证据，我们才能进一步研究是否已经形成一种对语言计算有意义的、跨事件协作的 Cellular Computing 机制。

本实验的核心问题最终可以表述为：

> **Cellular Communication 是否能够将某个 Event 产生的局部计算结果，通过 Cell 网络传播并延续至后续 Event，从而对未来语言计算形成有用影响？**

这是从“网络确实使用 Connection”迈向“理解 Connection 如何产生计算价值”的下一步。

**文档状态：PROPOSED v0.1。正式执行前冻结诊断协议与源提交，保持所有模型权重及历史实验结果不变。**
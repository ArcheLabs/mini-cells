# MiniCells R3g — Cell–Connection Co-Adaptation Attribution

**Cell–Connection 协同适应归因与训练—推理路由一致性审计**

**版本：** v1.0  
**实验 ID：** `CLM-001-R3g-CoAdaptation-Attribution-001`  
**项目：** MiniCells / Cellular Language Models  
**仓库：** `ArcheLabs/mini-cells`  
**性质：** 只读、探索性机制审计  
**环境：** Kaggle T4（单 GPU 即可）  
**执行：** 一次性执行、自动完成、无需中途人工授权  
**Optimizer Updates：** 0  
**Confirmation：** `SEALED`

---

## 1. 研究背景

### 1.1 R3f-v2 的已知结果

R3f-v2 实验身份：

```text
Source:
df096d4567326257f19f7bd58359a401fde3c3e5

Config SHA-256:
2648a588dacea2119552ff487929942fe2af8c9b42d9fdd6482901639e84a1e5

Run ID:
6b79bb4c6c29694e10e5ad77a20285613e3b98ff247b58845f218b48ded8c39b

Execution ID:
31633d7661954111aa3727535f044d00
```

已完成：

- 一个 Seed：`151001301`。
- 两个 Arm：`A-C`、`B-C`。
- 每个 Arm 生成一个真实 Connection 提案。
- 均通过 Trust Region。
- 均执行 10 步旧、新 Cell 配对适应。
- GitHub 结果发布成功。

结果如下：

| 指标 | A-C | B-C |
|---|---:|---:|
| 更新尺度 | 1/4096 | 1/4096 |
| 硬动作变化率 | 2.64% | 2.86% |
| 完整路径变化率 | 9.38% | 18.75% |
| 旧组织 Calibration NLL | 4.415665 | 4.446571 |
| 新组织 Calibration NLL | 4.414216 | 4.445827 |
| 改善 | 0.001449 | 0.000744 |
| 最终接受新组织 | 是 | 否 |

这些结果说明 R3f-v2 已经具备局部 Connection 更新准入、实际路径改变和配对训练能力。

但它们不足以证明 Cell 在新的 Connection 下学到了更有价值的计算能力。

### 1.2 尚未解决的关键问题

R3f-v2 的 `TRAINING_ROUTE_DIFFERENCE.json` 表明：

- A-C 的 10 次适应更新中，仅 Step 108 发生一次实际训练路径差异。
- B-C 的 10 次适应更新中，实际训练路径完全相同。
- 新旧随机训练策略的 KL 约为 \(10^{-10}\)。
- 但确定性推理路径已经发生明显变化。

这揭示出一个潜在的架构性问题：

**Connection 形成的推理组织，与 Cell 实际接受梯度训练的组织，可能不是同一种分布。**

### 1.3 本轮核心问题

本轮需要回答：

**Q1：R3f-v2 的 Calibration NLL 改善，有多少来自直接替换 Connection？**

**Q2：新 Connection 是否使 Cell 训练结果发生有益变化？**

**Q3：新旧 Cell 权重之间是否存在可观测、可重复验证的差异？**

**Q4：随机训练路由与确定性推理路由之间究竟存在多大的分布差距？**

**Q5：是否已有充分依据进入下一阶段训练机制研究？**

---

## 2. 研究范围

本轮是只读归因实验。

不启动任何新的 Cell 或 Connection 训练。

实验范围：

| 项目 | 设置 |
|---|---|
| Seed | 151001301 |
| Arm | A-C、B-C |
| 原始状态 | Step 100 |
| 适应后状态 | Step 110 |
| Cell 分支 | Old、New |
| Connection | Old、New |
| 主要实验 | 2×2 交叉评价 |
| 辅助分析 | 权重差异、路径差异、训练采样分布 |
| 数据 | Calibration；Admission 仅用于路由分析 |
| Development | 不读取 |
| Confirmation | 不读取 |
| 优化器更新 | 0 |

明确禁止：

- 重新生成 Connection 候选。
- 重新执行 K=10 Cell 适应。
- 重新训练或修正旧 Cell 权重。
- 修改 R3f-v2 的历史 Checkpoint。
- 通过 Development 选择后续训练参数。
- 依据审计结果自动启动正式多 Seed 实验。

本轮仅复用已经保存的历史模型状态。

---

# 3. 数据与模型恢复

## 3.1 输入来源

优先从 R3f-v2 已保存的私有 HF Checkpoint 获取输入。

使用历史实验的：

```text
run_id:
6b79bb4c6c29694e10e5ad77a20285613e3b98ff247b58845f218b48ded8c39b

execution_id:
31633d7661954111aa3727535f044d00
```

从对应的私有 HF Manifest 或持久化 Kaggle Output 恢复。

HF 指针路径按照已有实现的 `r3fv2/<run_id>/latest.json` 解析。

必须验证实际返回的 Manifest、Revision、文件哈希、Store Receipt 和模型身份。

不得只通过目录名称猜测 Checkpoint 身份，也不得将一个新修订版默认等同于历史执行时的修订版。

## 3.2 需要恢复的模型状态

每个 Arm 必须恢复以下对象：

### A. 初始官方状态

Step 100 正式模型：

\[
M_0=(\theta_0,\phi_0)
\]

其中：

- \(\theta_0\) 是适应前 Cell 权重。
- \(\phi_0\) 是旧 Connection。

### B. 旧 Connection 适应分支

Step 110：

\[
M_O=(\theta_O,\phi_O)
\]

其中：

\[
\phi_O=\phi_0
\]

### C. 新 Connection 适应分支

Step 110：

\[
M_N=(\theta_N,\phi_N)
\]

其中：

\[
\phi_N=
\phi_0+\alpha\Delta\phi
\]

本轮选择的：

\[
\alpha=\frac1{4096}
\]

需要分别恢复 \(\theta_O,\theta_N,\phi_O,\phi_N\)。

## 3.3 必须保留被拒绝分支

这是本轮最重要的工程前提。

R3f-v2 的 A-C 最终选择新分支，而 B-C 最终选择旧分支。

如果只读取最终 Selected Checkpoint：

- A-C 会丢失旧适应分支。
- B-C 会丢失新适应分支。

因此不能只使用 Selected Checkpoint。

**必须读取 Step 110 配对适应的事务存档。**

根据 R3f-v2 实际实现，应优先读取：

```text
private-checkpoints/jobs/<arm>/
  transactions/proposal-0100/
```

在对应 Store 中使用第 10 次适应更新的事务 Receipt，取得包含：

```python
package["old"]
package["new"]
package["adapt_done"]
package["phase"]
```

的完整事务状态。

历史实现会在每次适应更新之后保存配对状态，因此第 10 步事务应当包含两条分支的模型。

不得只读取最终 `COMMITTED` 事务，因为最终提交记录可能只保留 Selected 状态。

如果对应事务文件未被 HF 保存，应检查保留的 Kaggle Output。

如果两个来源均缺少完整分支，停止该 Arm 的完整交叉归因，并报告：

`PAIRED_BRANCH_STATE_NOT_RETAINED`

不允许重新训练一个替代分支并当作原始实验结果。

---

# 4. 恢复完整性与身份校验

开始评价之前，必须验证：

1. 两条分支来自相同 Step 100。
2. Old/New 均完成 K=10 更新。
3. 两条分支具有相同初始 Cell 权重。
4. 两条分支具有相同初始 Cell Optimizer。
5. 两条分支的数据顺序一致。
6. Route RNG 配对状态符合历史记录。
7. Old Connection 与原始 \(\phi_0\) 一致。
8. New Connection 与实际选择的 \(\phi_N\) 一致。
9. Route Margin 和 Numeric Tolerance 与原始实验一致。
10. 所有 Checkpoint 的内容哈希与 Receipt 一致。

还必须验证所有非 Cell、非 Connection 的计算参数在 Old/New 分支中相同。

例如：

- Embedding。
- ContextRead。
- Gate。
- LayerNorm。
- LM Head。
- 其他冻结计算权重。

如果上述参数不一致，必须记录差异并停止将结果解释为纯粹的 Cell–Connection 归因。

不允许为了消除差异，静默地覆盖某一分支的参数。

## 4.1 Selected State 验证

A-C：

\[
M_{\mathrm{selected}}=M_N
\]

B-C：

\[
M_{\mathrm{selected}}=M_O
\]

需要比较模型参数哈希与必要的优化器状态。

若 Selected State 与对应事务分支不一致，报告：

`SELECTED_BRANCH_STATE_MISMATCH`

---

# 5. 核心实验：2×2 Cell–Connection 交叉评价

## 5.1 四种组合

每个 Arm 构造四个模型：

| 实验 | Cell 权重 | Connection | NLL |
|---|---|---|---|
| OO | Old | Old | \(L_{OO}\) |
| ON | Old | New | \(L_{ON}\) |
| NO | New | Old | \(L_{NO}\) |
| NN | New | New | \(L_{NN}\) |

其中：

- O = Old。
- N = New。
- 行代表 Cell 参数来源。
- 列代表 Connection 参数来源。

除了 Cell 和 Connection，所有其他计算参数必须完全相同。

所有四个组合必须：

- 使用相同 Calibration 输入。
- 使用相同模型精度。
- 使用相同确定性推理规则。
- 使用相同 Evaluation Batch。
- 不使用训练期随机路由。
- 不执行梯度更新。

具体使用：

```python
model.eval()

with torch.no_grad():
    ...
```

设定：

```text
epsilon = 0
deterministic_route = true
```

## 5.2 第一项：Connection 的直接效果

固定旧 Cell 参数，只替换 Connection：

\[
\boxed{
\Delta_{\mathrm{route|old}}
=
L_{ON}-L_{OO}
}
\]

解释：

- 负值：新 Connection 在旧 Cell 下更好。
- 零附近：单独更换 Connection 影响很小。
- 正值：新 Connection 在旧 Cell 下更差。

这一项可以判断：是否仅凭改变计算路径，就能够改善 NLL。

## 5.3 第二项：新 Cell 下的 Connection 效果

固定新 Cell：

\[
\boxed{
\Delta_{\mathrm{route|new}}
=
L_{NN}-L_{NO}
}
\]

如果两种 Cell 权重下，新 Connection 都有类似收益，说明性能变化可能主要来自路由选择本身。

## 5.4 第三项：Cell 训练适应的直接效果

固定旧 Connection：

\[
\boxed{
\Delta_{\mathrm{cell|old}}
=
L_{NO}-L_{OO}
}
\]

固定新 Connection：

\[
\boxed{
\Delta_{\mathrm{cell|new}}
=
L_{NN}-L_{ON}
}
\]

这些指标比较的是在相同 Connection 下，两条配对训练分支产生的 Cell 参数差异及其对 NLL 的影响。

需要注意：

这并不是传统意义上的「Cell 训练总收益」。

因为 Old/New 两组都已经完成 K=10 Cell 更新。

它们衡量的是：

**在不同 Connection 下进行适应训练，是否导致 Cell 最终形成不同的、具有质量差异的参数。**

## 5.5 第四项：Cell–Connection 交互作用

定义：

\[
\boxed{
I=
L_{NN}-L_{NO}-L_{ON}+L_{OO}
}
\]

也可以写成：

\[
I=
\Delta_{\mathrm{route|new}}
-
\Delta_{\mathrm{route|old}}
\]

解释：

- \(I\approx0\)：当前四个组合没有明显非加性相互作用。
- \(I<0\)：新 Cell 与新 Connection 的组合表现出有利的非加性交互。
- \(I>0\)：交互方向不利于新 Cell–Connection 组合。

**交互项不为零，不自动等于已经形成有意义的 Cell 功能专业化。**

它仅说明，在本次模型状态与评价数据下，Cell 参数来源会影响 Connection 替换的收益。

## 5.6 与 R3f-v2 原始结果对齐

必须要求：

A-C：

\[
L_{OO}\approx4.4156653434
\]

\[
L_{NN}\approx4.4142163992
\]

B-C：

\[
L_{OO}\approx4.4465711489
\]

\[
L_{NN}\approx4.4458271340
\]

使用相同评估代码与数据身份时，建议绝对误差不超过：

\[
10^{-6}
\]

若因设备、计算内核或数值精度造成误差超过该范围，应首先定位差异。

不得为匹配历史数字而手动修正 NLL。

输出状态：

`HISTORICAL_CALIBRATION_REPRODUCED`

或者：

`HISTORICAL_CALIBRATION_MISMATCH`

若无法复现，停止后续质量归因。

---

# 6. 初始 Cell 的对照评价

除了四个适应后的组合，还需要恢复 Step 100 初始 Cell：

\[
\theta_0
\]

评价：

\[
L_{0O}
=
L(\theta_0,\phi_O)
\]

\[
L_{0N}
=
L(\theta_0,\phi_N)
\]

二者的差：

\[
\boxed{
\Delta_{\mathrm{immediate}}
=
L_{0N}-L_{0O}
}
\]

表示在任何新 Cell 适应发生前，Connection 替换的即时质量影响。

通过与适应后的结果比较，可以区分：

- Connection 在初始 Cell 下是否已经带来收益。
- 完成 K=10 更新之后，这种收益是否变大。
- 新 Cell 是否比旧 Cell 更适合新的 Connection。

同时记录：

\[
L_{OO}-L_{0O}
\]

与：

\[
L_{NN}-L_{0N}
\]

分别表示两条训练分支在各自 Connection 下的总变化。

但这些总变化仍不能单独解释为某种新的 Cell 专业化。

本轮共需完成每个 Arm 六种确定性配置的评价。

---

# 7. Cell 参数差异审计

## 7.1 精确权重比较

对 Old/New Cell 参数逐张量比较。

必须输出：

- 完全相同的参数张量数量。
- 存在差异的参数张量数量。
- 非零参数差异数量。
- 最大绝对差异。
- RMS 差异。
- L2 参数差异。
- 相对参数变化比例。
- 各 Cell 的独立结果。

对每个 Cell：

\[
\Delta\theta_i
=
\theta_{N,i}-\theta_{O,i}
\]

统计：

\[
\|\Delta\theta_i\|_2
\]

以及：

\[
\|\Delta\theta_i\|_\infty
\]

输出六个 Cell 的独立参数哈希。

## 7.2 特别验证 B-C

R3f-v2 记录表明 B-C 的 10 步适应训练中，实际训练路由完全相同，而且梯度范数记录完全一致。

因此存在一个重要假设：

\[
\theta_N=\theta_O
\]

如果确实如此，那么：

\[
L_{NO}=L_{OO}
\]

\[
L_{NN}=L_{ON}
\]

并且：

\[
I=0
\]

在数值容差范围内应成立。

此时 B-C 的 Calibration 改善就应当主要归因于 Connection 替换，而不是不同的 Cell 训练适应。

但不能只根据梯度范数相同就得出这个结论。

必须比较完整参数张量与状态哈希。

## 7.3 特别验证 A-C

R3f-v2 在 Step 108 出现一次实际训练路径差异：

- 一个 Token 的完整路径不同。
- 一个路由决策不同。
- 两条分支的 Cell 调用计数和梯度有轻微差异。

因此，需要检查：

- 从哪些 Cell 开始出现参数差异。
- 差异是否仅涉及 Step 108 所改变的 Cell。
- 之后的更新是否扩大了差异。
- 最终 Cell 权重差异是否对 Calibration NLL 产生可观测影响。

现有记录只支持这一次训练路径分叉，不能预先假定产生了大规模专业化。

---

# 8. 训练路由与推理路由的一致性分析

这是第二项核心研究。

## 8.1 当前两种不同的路由机制

在 R3f-v2 中，确定性推理采用：

\[
a_{\mathrm{eval}}
=
\operatorname{HardRoute}_{\tau}(q)
\]

其中 \(\tau\) 是基于 Gate Margin 的稳定切换阈值。

训练使用：

\[
\boxed{
\pi_{\mathrm{train}}(a|s)
=
(1-\epsilon)\operatorname{Softmax}(q)_a
+
\epsilon U(a)
}
\]

其中：

- \(q\) 是修正后的原始 Connection Logit。
- \(U(a)\) 是均匀探索分布。
- Softmax 温度沿用原始配置。

A-C 的动作空间大小：

\[
|\mathcal A|=15
\]

B-C：

\[
|\mathcal A|=6
\]

两种策略存在潜在差异。

即使确定性路由会因为一个很小的 Margin 翻转，训练采样概率可能依然接近均匀分布。

## 8.2 计算训练策略熵

对固定的 Calibration 路由参考状态：

\[
H(\pi)
=
-\sum_a\pi(a|s)\log\pi(a|s)
\]

分别计算：

- 原始 Softmax 熵。
- Epsilon 混合后熵。
- 熵的平均值。
- 熵的分位数。
- 最大动作概率。
- 第二大动作概率。
- 最优动作与次优动作概率差。

使用归一化熵：

\[
\boxed{
H_{\mathrm{norm}}
=
\frac{H(\pi)}{\log|\mathcal A|}
}
\]

如果归一化熵接近 1，表示训练动作分布接近均匀。

这将支持一个重要解释：

**Cell 在训练中可能主要经历高熵采样路径，而不是推理时使用的确定性计算路径。**

## 8.3 硬路由动作在训练分布中的概率

对每个固定状态，计算：

\[
a^*
=
\operatorname{HardRoute}_{\tau}(q)
\]

再计算：

\[
\boxed{
P_{\mathrm{match}}(s)
=
\pi_{\mathrm{train}}(a^*|s)
}
\]

该值表示训练采样恰好选择确定性推理动作的概率。

分别统计：

- Old Connection。
- New Connection。
- A-C。
- B-C。
- 不同 Stage。
- 不同前驱 Cell。

如果 A-C 的该概率接近 \(1/15\)，或者 B-C 接近 \(1/6\)，则表明训练动作与推理动作之间可能存在显著分布差异。

这将为下一阶段是否调整 Gumbel-Max 采样机制提供直接证据。

注意：该指标是在相同固定状态上的动作匹配概率，不等于完整闭环计算路径的匹配概率。

## 8.4 重新审计实际训练路径

直接使用原始：

`TRAINING_ROUTE_DIFFERENCE.json`

不得重跑训练来生成替代观测。

必须输出：

- 10 次适应更新中的动作差异率。
- 完整路径差异率。
- 存在差异的 Step。
- Cell 调用数量的差异。
- Cell 梯度范数的差异。
- 新旧训练策略 KL。
- 每次更新实际使用的 Token 数。

并验证其与原始发布报告一致。

这能够区分：

1. 确定性推理动作不同。
2. 训练采样分布略有变化。
3. 实际训练样本恰好执行了相同动作。
4. 最终 Cell 参数是否发生变化。

这四个层级必须分别报告。

---

# 9. Cell–Connection 适应收益的进一步判定

本轮应避免使用模糊的“联合学习成功”作为单一状态。

建议将归因结果划分为四个独立类别。

## 9.1 Route-Only Benefit

条件：

\[
L_{ON}<L_{OO}
\]

且新旧 Cell 参数差异对质量贡献很小。

解释：

新 Connection 主要通过改变推理计算路径获得收益。

不表明 Cell 已适应新的组织。

## 9.2 Cell Adaptation Benefit

条件：

\[
L_{NN}<L_{ON}
\]

解释：

在相同新 Connection 下，新分支训练出的 Cell 比旧分支训练出的 Cell 更好。

这是新组织训练可能具有正向适应收益的证据。

但仍需要量化效果大小，并考虑数值精度、Calibration 选择偏差和单 Seed 限制。

## 9.3 Co-Adaptive Interaction

条件：

\[
I<0
\]

且其绝对值大于重复评价的数值误差。

解释：

新 Cell 和新 Connection 之间存在有利的非加性交互。

这比单纯观察：

\[
L_{NN}<L_{OO}
\]

更接近我们真正关心的共同适应问题。

但本轮只提供探索性、有限状态下的证据，不得将其提升为一般收敛或专业化结论。

## 9.4 No Detectable Adaptation

如果：

\[
\theta_N=\theta_O
\]

或者：

\[
L_{NN}\approx L_{ON}
\]

并且：

\[
L_{NO}\approx L_{OO}
\]

则没有检测到独立的 Cell 适应收益。

这并不等于 Cell 不具备适应能力。

可能只是本轮 Connection 更新非常小，训练采样路径几乎没有改变。

---

# 10. 测量精度与数据边界

## 10.1 禁止为了得出正向结果而改变数据

本次固定使用原来的 Calibration 数据和原来的 Tokenizer。

不得：

- 调整故事子集。
- 根据结果挑选 Window。
- 删除不利于某个 Arm 的样本。
- 调整模型精度以放大差异。
- 根据交叉评价结果重新选择更新尺度。

## 10.2 数值不确定性

本轮必须记录：

- NLL 的原始浮点值。
- 各交叉组合的最大重复计算误差。
- Cell 参数哈希。
- 路由路径哈希。
- Eval 软件环境与实际设备。

若两个 NLL 差值与重复评价误差接近，结论应为：

`BELOW_NUMERICAL_RESOLUTION`

不得直接判定为正向或负向的 Cell 适应效果。

## 10.3 统计边界

只有一个训练 Seed。

Calibration 已经用于选择新 Connection 是否被最终接受，因此不属于独立验证数据。

可以提供 Story 级配对差异与描述性 Bootstrap 区间，但必须明确它们不能消除 Selection Bias。

不得进行强统计确认声明。

Development 保持不读取。

Confirmation 保持封存。

---

# 11. 计算预算

本次为只读审计，不涉及训练。

主要计算：

| 操作 | A-C | B-C |
|---|---:|---:|
| 适应后交叉组合 | 4 | 4 |
| 初始 Cell 对照 | 2 | 2 |
| 共计主要质量评价 | 6 | 6 |
| 路由熵与概率分析 | 1 组固定参考状态 | 1 组固定参考状态 |
| Cell 参数比较 | 6 个 Cell | 6 个 Cell |
| 训练轨迹复核 | 10 Step 已保存记录 | 10 Step 已保存记录 |
| 优化器更新 | 0 | 0 |

总计 12 次主要模型质量评价，加上必要的有限只读诊断。

所有主要评价可以串行进行，无需同时使用两个 GPU。

不进行新的反事实 Q 搜索。

不执行训练期 Gumbel 的大型 Monte Carlo 重复实验。

不新增与本轮归因无关的 Benchmark。

---

# 12. 实施结构

建议新建独立目录：

```text
research/projects/
  cellular-language-models/
    experiments/
      clm-001-qfirst-integrated/
        r3g/
          SPEC.md
          README.md
          config/
            frozen.json
          implementation/
            minicells_r3g/
              parent_restore.py
              branch_restore.py
              crossover.py
              parameter_diff.py
              routing_distribution.py
              trajectory_audit.py
              evaluation.py
              reports.py
              runner.py
          notebooks/
            kaggle-readonly.ipynb
          reports/
            INPUT_VERIFICATION.json
            BRANCH_RESTORE.json
            CROSSOVER_MATRIX.json
            CELL_PARAMETER_DIFF.json
            ROUTING_DISTRIBUTION.json
            TRAINING_TRAJECTORY_AUDIT.json
            ATTRIBUTION_SUMMARY.json
            SCIENTIFIC-FINDINGS.md
            RUN_MANIFEST.json
```

优先复用：

- R3f-v2 的 Checkpoint 身份验证。
- R3f-v2 的私有 Store 恢复。
- R3f-v2 的模型定义。
- R3f-v2 的 Calibration NLL 计算。
- 现有 GitHub 聚合结果发布器。

不复制模型实现或训练循环。

---

# 13. 自动执行流程

Codex 创建统一入口：

```bash
python scripts/run_r3g.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3g/config/frozen.json \
  --run-all
```

该命令是本轮需要新建的入口。

执行顺序：

1. 固定源码、配置和原始 R3f-v2 实验身份。
2. 从私有 HF 或保留的 Kaggle Output 恢复输入。
3. 验证 A-C / B-C Step 100 官方状态。
4. 恢复两组 Step 110 的完整 Old/New 配对状态。
5. 核验模型参数及所有 Store Receipt。
6. 分别计算每组的四种交叉 NLL。
7. 计算初始 Cell 与两种 Connection 的 NLL。
8. 验证历史 Calibration NLL 是否得到复现。
9. 比较所有 Cell 的参数与哈希。
10. 计算训练路由分布熵、动作概率和确定性匹配概率。
11. 复核 R3f-v2 已保存的 10 步训练路由记录。
12. 自动生成归因分析与科学判断。
13. 发布聚合结果到独立 GitHub Results 分支。
14. 保留输入身份、文件哈希与全部审计结果。

任何步骤都不得进行优化器更新。

两个 Arm 相互独立。

若其中一个 Arm 的配对分支缺失，应停止该 Arm 的归因分析，但允许另一个 Arm 完成。

缺失分支时，不得用训练或复制另一分支替代。

---

# 14. 结果文件规范

## 14.1 CROSSOVER_MATRIX.json

每个 Arm 至少包含：

```json
{
  "arm": "A-C",
  "seed": 151001301,
  "L_0O": null,
  "L_0N": null,
  "L_OO": null,
  "L_ON": null,
  "L_NO": null,
  "L_NN": null,
  "route_effect_old_cells": null,
  "route_effect_new_cells": null,
  "cell_effect_old_connection": null,
  "cell_effect_new_connection": null,
  "interaction": null,
  "historical_nll_reproduced": null
}
```

其中 `null` 是结构示例占位符，不是实际实验结果。

所有真实指标必须由验证过的历史 Checkpoint 计算。

## 14.2 CELL_PARAMETER_DIFF.json

每个 Cell 至少包含：

```text
cell_id
old_parameter_hash
new_parameter_hash
bitwise_equal
changed_parameter_count
l2_difference
max_abs_difference
rms_difference
```

同时输出全部六个 Cell 是否整体完全相同。

## 14.3 ROUTING_DISTRIBUTION.json

分别记录：

```text
arm
connection_version
action_space_size
raw_softmax_entropy
mixture_entropy
normalized_entropy
mean_top1_probability
mean_second_probability
hard_action_probability
expected_hard_sample_agreement
stage_breakdown
predecessor_breakdown
```

## 14.4 ATTRIBUTION_SUMMARY.json

至少包含：

```text
arm
route_only_benefit
cell_adaptation_benefit
interaction_effect
training_route_changed
cell_parameters_changed
quality_change_detected
numerical_resolution
attribution_status
```

归因状态建议使用：

- `ROUTE_DOMINANT`
- `CELL_ADAPTATION_DETECTED`
- `FAVORABLE_INTERACTION_DETECTED`
- `MIXED_OR_INCONCLUSIVE`
- `NO_DETECTABLE_CELL_ADAPTATION`
- `INVALID_OR_INCOMPLETE`

这些标签不是互斥的科学定理。最终应附上支持该标签的原始数值。

---

# 15. 验收条件

只有满足以下条件，R3g 才能判定为有效的只读机制审计。

### 工程验收

- 所有输入 Checkpoint 身份和哈希有效。
- Old/New 两个适应分支真实存在。
- A-C、B-C 各六种组合评价完成。
- 历史 Calibration NLL 可以复现。
- Cell 参数差异可以逐张量验证。
- 两组训练路由轨迹与历史记录一致。
- 路由分布熵和动作匹配概率计算完成。
- 原始模型和 Checkpoint 未修改。
- Optimizer Updates = 0。
- Development / Confirmation 未读取。
- 所有报告可回溯到源文件和代码版本。

### 科学验收

本轮不以正向发现作为成功标准。

即使最后发现：

- A-C 的收益几乎全部来自 Connection。
- B-C 的 Cell 权重完全相同。
- 两种训练策略接近均匀采样。
- 没有检测到有意义的 Cell 适应。

只要数据完整且归因正确，这就是有效的科学结果。

不能为了获得正向结论重新训练或调整评价条件。

---

# 16. 下一阶段决策规则

R3g 完成后，根据结果决定下一步。

## 情况 A：收益主要来自 Connection

表现为：

\[
\Delta_{\mathrm{route|old}}<0
\]

但：

\[
\Delta_{\mathrm{cell|new}}\approx0
\]

说明 Connection 能改善推理路径，但 Cell 没有检测到明显的独立适应收益。

**下一阶段优先研究训练与推理路由的一致性。**

候选方向包括：

- 稳定硬路由训练。
- 更低温度的训练采样。
- 按确定性路由提供更多 Cell 梯度覆盖。
- 与计算路径对齐的受控探索。

这些方案都必须在新的实验身份下实施，不属于本次 R3g。

## 情况 B：存在正向 Cell 适应收益

表现为：

\[
\Delta_{\mathrm{cell|new}}<0
\]

或者：

\[
I<0
\]

并且效果超过数值误差。

说明新的 Connection 可能使 Cell 形成有利的适应。

下一阶段应设计多 Seed、多个提案时刻的验证，研究这种收益是否能在更长训练过程中累积。

## 情况 C：两种收益均不明显

说明现有候选方向可能并没有足够的计算价值，不能仅依靠更新准入来改善模型。

下一阶段应重新研究：

- Connection 的反事实 Q 估计。
- 硬路径选择的训练目标。
- 当前任务信号是否足以形成 Cell 分工。
- 不同 Cell 是否具有可学习的互补能力。

## 情况 D：训练采样接近均匀，推理高度确定

这是最重要的潜在发现之一。

如果：

\[
H_{\mathrm{norm}}\approx1
\]

同时硬路由动作在训练分布中的概率接近 \(1/|\mathcal A|\)，则说明模型训练和推理之间存在明显的路由分布不匹配。

下一阶段应该优先验证这一训练机制问题，而不是扩大 Cell 图或增加模型参数。

---

# 17. 最终 Codex 实施指令

请在 `ArcheLabs/mini-cells` 中实施：

`CLM-001-R3g-CoAdaptation-Attribution-001`

本次任务是一次完整的只读 Cell–Connection 协同适应归因审计。

请复用 R3f-v2 已保存的真实 Checkpoint 和实验记录，不重新训练模型。

核心要求：

1. 从 R3f-v2 的私有 Store 恢复 A-C、B-C 各自的 Step 100 官方状态，以及 Step 110 的旧、新适应分支。
2. 使用对应事务中 K=10 的完整配对状态，不能只使用最终 Selected Checkpoint。
3. 构建 Old/New Cell × Old/New Connection 的 2×2 交叉评价。
4. 追加初始 Cell 在旧、新 Connection 下的即时质量评价。
5. 严格复现原始 R3f-v2 Calibration NLL。
6. 逐张量比较六个 Cell 的参数差异。
7. 特别验证 B-C 是否存在完全相同的 Cell 参数。
8. 复核 A-C 在 Step 108 的训练路径分叉。
9. 计算训练随机策略熵、确定性路由动作概率，以及训练—推理动作匹配概率。
10. 将 Connection 的直接收益与 Cell 适应收益分别归因。
11. 不修改历史模型，不产生任何 Optimizer 更新。
12. 不读取 Development 或 Confirmation。
13. 生成完整报告，并只将聚合结果发布到 GitHub。
14. 如果任何必要历史分支缺失，明确报告不可归因，不允许生成替代模型。

本次不自动启动下一阶段完整训练。

最终必须给出明确结论：

**R3f-v2 的改善究竟主要来自 Connection 推理路径的改变，还是来自 Cell 在新的 Connection 下产生了有益的训练适应？**

以及：

**MiniCells 当前训练时的随机 Connection，是否与推理时的确定性 Cell 组织存在显著分布不匹配？**

这两个问题将直接决定下一阶段应该优先研究训练路由机制、Connection 优化目标，还是进一步的 Cell 图组织演化。

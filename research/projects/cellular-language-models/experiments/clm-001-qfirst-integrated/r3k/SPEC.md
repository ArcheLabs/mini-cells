# MiniCells R3k — Connection Future Learning Value Study

**Connection 未来学习价值预测、跨数据泛化与长期反事实验证**

**实验 ID：** `CLM-001-R3k-Future-Learning-Value-001`  
**版本：** v1.0  
**仓库：** `ArcheLabs/mini-cells`  
**实验类型：** 机制研究 / 固定预测指标验证 / 历史配对续训  
**运行环境：** Kaggle T4 × 2  
**执行方式：** 一次启动，分阶段自动执行  
**科学范围：** `EXPLORATORY_DEVELOPMENT`  
**Confirmation：** `SEALED`  
**Connection Optimizer 更新：** 0

---

## 一、研究背景

### 1.1 已完成的研究

R3h 已证明，在当前 MiniCells 架构中，使训练路由更接近确定性推理路由，能够明显改善 Development NLL。

R3i 证明：

- Connection 可以产生真实且受 Trust Region 约束的局部更新。
- 48 次 Connection Proposal 全部通过稳定性准入。
- 8 次更新通过短期 Calibration 质量接受。
- Cell 参数能够在不同 Connection 下形成可测量的适应差异。
- 但四项终端质量假设均未获得支持。

R3j 对其中 12 个历史候选执行了配对长期续训。

结果：

| 指标 | R3j |
|---|---:|
| 历史已接受候选 | 8 |
| 历史已拒绝候选 | 4 |
| 成功恢复的配对模型 | 12/12 |
| +100 Step 正向案例 | 6 |
| +100 Step 负向案例 | 6 |
| 达到 +0.005 NLL 改善门槛 | 0 |
| Calibration 短期收益与长期收益 Spearman | −0.119 |
| Development +10 与 +100 收益 Spearman | +0.399 |

其中一个重要观察是：

在 12 个案例的 +100 Step 描述性均值中：

\[
R_{100}\approx-0.000467
\]

\[
C_{100}\approx+0.000655
\]

\[
V_{100}\approx+0.000188
\]

即：

- 新 Connection 的直接路由效果平均偏负。
- 新组织下的 Cell 适应效果平均偏正。
- 整体平均收益很小。

该结果提示：

**Connection 的价值可能不仅来自当前选对 Cell，还可能来自改变 Cell 后续学习的轨迹。**

但这些结果来自被选择的 12 个案例，不能据此证明这一机制普遍成立。

### 1.2 仍然缺少的证据

R3j 没有回答：

1. 全部 48 个候选中，这种现象是否普遍存在？
2. 历史已知的短期收益是否能够跨数据预测长期收益？
3. 是否可以构造不依赖未来训练结果的预测指标？
4. 哪些指标与未来 Cell 适应更相关？
5. Connection 的直接执行收益与未来学习收益，是否需要分别估计？
6. 是否存在足够可靠的预测信号，支持下一轮真正改变 Connection 更新机制？

R3k 将集中回答这些问题。

---

# 二、核心理论定义

## 2.1 Cell–Connection 状态

定义：

\[
M_s=(\theta_s,\phi_s)
\]

其中：

- \(\theta_s\)：当前 Cell 参数。
- \(\phi_s\)：当前 Connection 参数。
- \(s\)：主训练 Step。

给定一个真实的历史 Connection 候选：

\[
\phi'_s
\]

使用相同训练数据和外生随机流分别继续训练。

旧组织：

\[
\theta^O_{s+T}
=
\operatorname{Train}_T(\theta_s,\phi_s)
\]

新组织：

\[
\theta^N_{s+T}
=
\operatorname{Train}_T(\theta_s,\phi'_s)
\]

长期反事实价值：

\[
\boxed{
V_T=
L(\theta^O_{s+T},\phi_s)
-
L(\theta^N_{s+T},\phi'_s)
}
\]

当 \(V_T>0\)，新组织在该训练跨度后取得更低 NLL。

## 2.2 直接执行价值

在 Cell 参数固定时，改变 Connection 的效果：

\[
\boxed{
R_T=
L(\theta^O_{s+T},\phi_s)
-
L(\theta^O_{s+T},\phi'_s)
}
\]

它衡量 Connection 替换本身带来的直接推理质量变化。

## 2.3 Cell 适应价值

在新 Connection 固定时，更换为新分支训练出的 Cell：

\[
\boxed{
C_T=
L(\theta^O_{s+T},\phi'_s)
-
L(\theta^N_{s+T},\phi'_s)
}
\]

因此：

\[
\boxed{
V_T=R_T+C_T
}
\]

这是特定分解路径下的代数恒等式。

R3k 不假设 \(R_T\) 和 \(C_T\) 必然同号。

## 2.4 研究目标

R3k 不直接优化：

\[
V_{100}
\]

而研究是否存在一个能够在候选被正式接受前计算的预测函数：

\[
\boxed{
S(\theta_s,\phi_s,\phi'_s,\mathcal D_{\mathrm{probe}})
}
\]

使它能够预测：

\[
V_{100}
\]

其中 \(\mathcal D_{\mathrm{probe}}\) 不能是用于计算长期实验答案的评价数据。

---

# 三、研究假设

## H1：跨数据的短期质量评价具有预测能力

当前 R3i 使用 Calibration K=10 改善进行 Connection 接受。

R3k 检验：

在不同于历史 Calibration 的固定研究 Probe 上评价 K=10 分支，是否比原始 Calibration 分数更能预测 +100 Step 的真实收益。

## H2：早期 Cell 学习效应具有长期预测信息

如果新组织下的 Cell 在 K=10 已表现出有利适应，那么这种信号是否能够预测 +100 Step 的收益？

这里需要区分：

- Connection 的即时路由价值。
- Cell 在前十步的适应价值。
- 二者的总效果。

## H3：短期收益的变化趋势具有预测价值

当前即时路由收益未必等于未来学习收益。

因此，研究：

从 T=0 到 T=10，候选的相对价值是增长还是下降，能否预测 T=100 的长期结果。

## H4：真实长期效果可以与直接路由收益分离

R3j 中出现：

\[
R_{100}<0,\quad C_{100}>0
\]

的案例。

R3k 检验这种现象在全部 48 个候选中的分布、强度，以及是否跨架构和路由模式保持一致。

H4 属于机制假设，不等同于存在可用预测器。

---

# 四、实验矩阵

## 4.1 历史候选总体

使用 R3i 的全部 48 个真实候选。

组成：

- 3 个 Seed。
- 2 种 Cell 架构。
- 2 种训练路由。
- 4 个 Proposal Step。

\[
3\times2\times2\times4=48
\]

Seed：

```text
151001301
151001302
151001303
```

Arm：

```text
A-CH
A-CT
B-CH
B-CT
```

Proposal Step：

```text
100
200
300
400
```

所有候选都必须保留原始：

- Old Connection。
- New Connection。
- 原始 Cell 状态。
- K=10 Old/New Cell 状态。
- Optimizer。
- 训练游标。
- RNG。
- Scale Scan。
- 历史质量接受标签。

## 4.2 长期标签来源

| 候选 | 数量 | 处理 |
|---|---:|---|
| R3j 已长期续训 | 12 | 复用现有 T=10/50/100 Checkpoint |
| R3i 剩余候选 | 36 | 新增配对续训至 T=50/100 |
| **总计** | **48** | **全部形成长期评价结果** |

已完成的 12 个案例不允许重新训练。

如果新实现无法证明与 R3j 的训练延续一致，必须停止历史结果混合，而不是无条件重用。

## 4.3 候选组成限制

R3j 的 12 个案例已经包含全部八个历史已接受候选。

因此，R3k 新增长期续训的 36 个候选全部属于历史被拒绝候选。

这意味着：

- 新生成的长期训练结果主要检验被拒绝候选。
- 不能把这 36 个案例当作同时覆盖接受与拒绝的随机样本。
- 完整 48 个候选可以形成这个固定历史提案总体的描述性分析。
- 但不能推断未来所有 Connection Proposal 的成功率。

---

# 五、数据划分和隔离

本轮必须避免重犯 R3i 中 Calibration 被重复使用的问题。

## 5.1 固定历史数据

继续保持 R3i/R3j 的历史数据身份。

数据集：

```text
roneneldan/TinyStories
```

Revision：

```text
f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
```

Tokenizer SHA-256：

```text
612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff
```

已有范围：

| 角色 | Story Range |
|---|---|
| Historical Train | [0, 8192) |
| Historical Calibration | [50000, 50128) |
| Historical Admission | [50128, 50256) |
| Historical Development | [50256, 50512) |
| Reserved Boundary | 60000 |

不得更改历史分区，也不得访问正式 Confirmation。

## 5.2 新的预测参考数据

建立：

`R3K_PREDICTOR_PROBE`

预定 Story Range：

\[
[8192,8448)
\]

共 256 个 Story 索引。

用途：

- 计算冻结的 Connection 预测指标。
- 不更新模型。
- 不用于拟合可训练预测器。
- 不参与新的 Connection Proposal。
- 不参与长期结果评价。

使用同一 Tokenizer 和固定、可复现的评价窗口规则。

## 5.3 新的长期价值评价数据

建立：

`R3K_VALUE_HOLDOUT`

预定 Story Range：

\[
[8448,8960)
\]

共 512 个 Story 索引。

用途：

- 评价 T=10、50、100 的真实长期收益。
- 计算交叉归因。
- 检验预测指标的有效性。

不得用于：

- 构造或校准预测指标。
- 选择候选。
- 修改训练算法。
- Connection Proposal。
- 调整分数权重。

该 Holdout 是新的**探索性研究评价集**，不是原有的正式 Confirmation。

## 5.4 数据隔离门禁

以上范围是预注册建议，Codex 必须先验证：

- 数据集 Revision 完全一致。
- 与所有既有 Train、Calibration、Admission、Development 索引范围不重叠。
- 与可审计的历史训练数据内容不存在精确重复。
- 与既有正式 Confirmation 定义没有冲突。
- Tokenizer 和 Story 编码规则一致。
- Story/Token/Window 身份全部记录。

如果出现重复内容，按预先冻结的排除规则剔除精确重复项并报告，不得根据模型表现寻找替代 Story。

如果出现范围或用途上的实质冲突，停止实验并报告：

`RESEARCH_HOLDOUT_BOUNDARY_CONFLICT`

不得擅自读取正式 Confirmation 或临时选择更容易获得正向结果的数据。

---

# 六、预注册的 Connection 预测指标

这一部分是 R3k 的核心。

全部指标必须在新增长期结果评价之前冻结。

不训练新的神经网络预测器。

不根据已知 R3j 结果拟合权重。

## 6.1 S-Cal：历史 Calibration 短期分数

作为原始基线：

\[
\boxed{
S_{\mathrm{Cal}}
=
L^{\mathrm{Cal}}_{OO,10}
-
L^{\mathrm{Cal}}_{NN,10}
}
\]

这是 R3i 真实使用的质量接受依据。

不得重算后修改历史结果。

## 6.2 S-Direct：即时路由价值

在 Proposal Step \(s\)，Cell 参数固定：

\[
\boxed{
S_{\mathrm{Direct}}
=
L_{\mathrm{Probe}}(\theta_s,\phi_s)
-
L_{\mathrm{Probe}}(\theta_s,\phi'_s)
}
\]

它衡量候选 Connection 在改变 Cell 训练之前的直接效果。

要求：

- 使用历史正式 Cell 参数。
- 只替换 Connection。
- 使用相同固定 Probe。
- 统一确定性推理。
- 不发生 Optimizer 更新。

这是候选形成后、K=10 适应前即可计算的指标。

## 6.3 S-K10：独立 Probe 上的十步总收益

恢复真实历史 K=10 Old/New：

\[
\boxed{
S_{\mathrm{K10}}
=
L_{\mathrm{Probe}}(\theta^O_{s+10},\phi_s)
-
L_{\mathrm{Probe}}(\theta^N_{s+10},\phi'_s)
}
\]

与 S-Cal 的区别在于：

- S-Cal 使用历史 Calibration。
- S-K10 使用预先分离的 Predictor Probe。

二者都可以在决定是否接受候选之前通过配对 Shadow Adaptation 获得。

S-K10 不需要执行第 11 至第 100 步。

## 6.4 S-Cell：十步 Cell 适应价值

定义：

\[
\boxed{
S_{\mathrm{Cell}}
=
L_{\mathrm{Probe}}(\theta^O_{s+10},\phi'_s)
-
L_{\mathrm{Probe}}(\theta^N_{s+10},\phi'_s)
}
\]

这是在相同新 Connection 下比较旧、新 Cell 参数的效果。

它专门衡量早期 Cell 适应，而不是 Connection 的直接作用。

该指标不能被解释为长期能力专业化已得到证明。

## 6.5 S-Slope：早期价值变化

定义：

\[
\boxed{
S_{\mathrm{Slope}}
=
S_{\mathrm{K10}}-S_{\mathrm{Direct}}
}
\]

含义：

在同一 Probe 上，候选相对优势从初始时刻到完成 K=10 后改变了多少。

它是早期相对价值变化指标，不是数学上精确的未来学习梯度。

不能把它直接解释为对 \(d\theta_T/d\phi\) 的无偏估计。

## 6.6 机制监测项

还需要记录：

- 原始 Logit 改变量。
- Raw Policy KL。
- 实际 H/T 训练策略 KL。
- Hard Action Change。
- Full Path Change。
- 路由熵变化。
- Cell 使用比例变化。
- K=10 实际训练路径分叉率。
- Cell 梯度分布差异。
- Cell 参数差异。
- 原始 Connection 参数更新尺度。

这些先作为描述性特征。

本轮不允许从这些指标中事后搜索最佳预测公式。

---

# 七、预测指标的冻结协议

## 7.1 先预测，后观察长期结果

整个 R3k 必须严格遵循：

```text
Restore historical candidates
    ↓
Validate data identities
    ↓
Compute predictor scores
    ↓
Freeze all 48 predictions
    ↓
Publish immutable prediction manifest
    ↓
Restore/continue long-horizon branches
    ↓
Evaluate long-horizon holdout
    ↓
Compare predictions with outcomes
```

必须先生成：

`FROZEN_PREDICTIONS.json`

记录所有 48 个候选的五项分数：

```text
S_Cal
S_Direct
S_K10
S_Cell
S_Slope
```

同时冻结：

- 指标公式。
- 输入数据身份。
- 候选身份。
- 计算精度。
- 数值误差门槛。
- 主要预测比较。
- 统计方法。

## 7.2 结果信息隔离

R3j 的 12 个历史 Development 长期结果已经是已知研究信息。

因此，不能声称本轮预测公式的研究思路完全独立于这些结果。

但 R3k 仍可严格实施：

- 预测指标不能读取历史 R3j Development 值。
- 不基于这 12 个结果拟合参数。
- 先冻结对全部 48 个候选的预测。
- 新的 Holdout 结果只能在预测冻结后计算。
- 新增长期续训的 36 个案例不得提前评价并反馈修改指标。

必须在报告中明确标注：

`PREDECLARED_POST_R3J_EXPLORATORY_PREDICTION`

不能标记为真正的前瞻性独立 Confirmation。

## 7.3 不允许的操作

不得：

- 根据已知长期结果调节指标系数。
- 选择不同的 Probe 使相关性提高。
- 对不同 Arm 使用事后选择的公式。
- 只报告表现最好的指标。
- 将多个分数拟合为新的回归模型后声称得到了独立验证。
- 根据 R3k Holdout 的表现改变预测公式。

所有指标无论结果好坏都必须完整报告。

---

# 八、预注册预测比较

## 8.1 主要比较

重点比较：

**P1：跨数据短期分数**

\[
S_{\mathrm{K10}}
\quad\text{vs.}\quad
S_{\mathrm{Cal}}
\]

检验独立 Probe 是否优于重复使用的历史 Calibration。

**P2：Cell 适应分数**

\[
S_{\mathrm{Cell}}
\quad\text{vs.}\quad
S_{\mathrm{Direct}}
\]

检验早期 Cell 适应是否比当前直接路由效果更能预测长期质量。

**P3：早期学习趋势**

\[
S_{\mathrm{Slope}}
\quad\text{vs.}\quad
S_{\mathrm{Direct}}
\]

检验早期相对价值的变化，是否比单纯即时路由效果更有长期预测信息。

S-Cal、S-Direct、S-K10、S-Cell、S-Slope 的全部绝对表现都应报告。

## 8.2 预测评价指标

使用：

- Spearman Rank Correlation。
- 方向一致率。
- 正向预测覆盖率。
- 假阳性、假阴性。
- 弃权比例。
- 对长期收益的相对排序表现。
- 按 Seed、架构、路由模式的分层结果。

所有指标均使用固定长期目标：

\[
\boxed{
V_{100}^{\mathrm{Holdout}}
}
\]

也可以对 \(C_{100}^{\mathrm{Holdout}}\) 做单独的机制分析，但不得把它替换为主要目标。

## 8.3 不能直接比较数值量纲

S-Slope、S-Direct、S-Cell 都是 NLL 差值，但来自不同反事实比较。

本轮以排序和符号预测为主，不强行假设它们是无偏的长期收益估计。

如报告原始数值误差，应明确这是未经未来跨度校准的预测误差。

---

# 九、长期配对续训

## 9.1 保留 R3j 已完成的 12 个案例

从固定的 R3j Checkpoint Manifest 恢复：

```text
T=10
T=50
T=100
```

不重新训练。

必须验证：

- Model Hash。
- Cell Optimizer。
- Connection Hash。
- Data Cursor。
- RNG。
- 历史 Case ID。
- Training Step。
- R3j Source/Config/Run Identity。

## 9.2 新增 36 个案例

对尚未进入 R3j 长期续训的全部 36 个历史候选：

从 R3i 保存的真实 K=10 Old/New 分支出发。

执行：

```text
K10 historical states
    ↓
Continue 40 Cell Steps
    ↓
T=50 Checkpoint
    ↓
Continue 50 Cell Steps
    ↓
T=100 Checkpoint
```

保持：

- Old 使用原始 Connection。
- New 使用历史真实候选 Connection。
- 两条分支使用相同训练 Token。
- 相同主训练 Scheduler。
- 相同绝对 Step。
- 相同路由随机协议。
- 相同 Cell Optimizer 超参数。

禁止新 Connection Proposal。

## 9.3 新增训练预算

每个案例：

\[
2\times90=180
\]

次 Cell 更新。

36 个案例：

\[
36\times180=6480
\]

次 Cell 更新。

Token 预算：

\[
\boxed{
6480\times512=3,317,760
}
\]

这些是 R3k 的新增主续训 Token。

另外 12 个已完成案例的 2,160 次历史 R3j 更新不重复计费为新执行预算。

## 9.4 不根据中间结果停止

不允许根据 T=50 或训练 CE：

- 放弃某个候选。
- 提前接受新 Connection。
- 改变目标跨度。
- 修改训练配置。
- 为正向候选追加训练。

每个有效候选必须执行到 T=100。

---

# 十、长期价值评价

## 10.1 统一评价数据

全部 48 个案例必须在：

`R3K_VALUE_HOLDOUT`

使用相同评价协议。

不能把：

- 12 个 R3j 案例的 Development NLL；
- 36 个新案例的 R3k Holdout NLL；

直接混合成一个总体指标。

所有 48 个案例必须在同一个新 Holdout 上重新评价。

## 10.2 评价时点

每个候选：

\[
T\in\{10,50,100\}
\]

评价：

\[
V_T
\]

及对应：

\[
R_T,\ C_T,\ I_T
\]

其中：

\[
V_T=R_T+C_T
\]

交互项：

\[
I_T=L_{NN,T}-L_{NO,T}-L_{ON,T}+L_{OO,T}
\]

单独报告，不得重复相加。

## 10.3 数值正确性

每个评价时点：

- 至少重复确定性评价两次。
- 记录最大重复误差。
- 记录 float32 数值分辨率。
- 验证四格关系。
- 验证 Old/New 数据 Window 一致。
- 验证评价没有修改模型或 Optimizer。

如果：

\[
|V_T|
\]

低于预先定义的数值误差界，则标记：

`BELOW_NUMERICAL_RESOLUTION`

不得强行判定正负。

## 10.4 长期价值实用门槛

保持：

\[
\boxed{
V_{100}\ge0.005
}
\]

作为实际改善的描述性门槛。

但这个门槛不用于重新训练或筛选候选。

---

# 十一、统计与独立性问题

## 11.1 三 Seed 不等于 48 个独立训练实验

48 个 Proposal 来自：

- 三个初始 Seed。
- 每 Seed 四条联合训练轨迹。
- 每轨迹四个不同时间点。

相同轨迹中的多个 Proposal：

- 共享训练历史。
- 可能继承此前接受的 Connection。
- 后续数据窗口重叠。
- 不是统计独立的四次模型复制。

因此，不能把 48 个候选简单视为 48 个独立 Seed。

## 11.2 首要盲测集合

新增的 36 个长期续训结果，在预测指标冻结时尚未生成。

它们作为本轮主要新增结果集合：

`R3K_NEW36`

必须单独报告预测评价。

完整 48 个案例：

`R3K_ALL48`

作为固定历史候选总体的补充分析。

由于 NEW36 全部是历史被拒绝候选，其结论主要限定于被拒绝候选的价值预测，不能推广到所有接受/拒绝场景。

## 11.3 分层报告

至少提供：

- A-CH。
- A-CT。
- B-CH。
- B-CT。
- Seed 151001301。
- Seed 151001302。
- Seed 151001303。
- 历史 Accepted。
- 历史 Rejected。

仅在样本数充分时计算分层相关。

样本过少时只提供逐案例结果。

## 11.4 Bootstrap

沿用 2000 次 Story-cluster 配对 Bootstrap。

新增固定种子：

```text
bootstrap_seed = 2026102303
```

统计必须明确：

- Story Bootstrap 是单案例评价不确定性。
- 事件共享训练轨迹。
- 跨候选比较需考虑 Seed/Arm 聚类。
- 三个独立训练 Seed 不能支持很强的总体显著性声明。

主要预测比较统一报告多重比较调整后的探索性结果，不以未调整的单个最佳相关性作为成功证明。

---

# 十二、预测有效性判定

本轮预先采用三层判定。

## 12.1 工程可用性

`PREDICTOR_COMPUTABLE`

条件：

- 对 48 个候选均可计算。
- 不依赖未来长期训练结果。
- 数据身份有效。
- 数值有限。
- 跨 Session 可复现。

只满足这一项不代表预测具有科学价值。

## 12.2 描述性长期预测信号

对于 R3K_NEW36，某指标需同时达到：

1. Spearman 相关系数至少 0.30。
2. 在方向可分辨的案例中，符号准确率至少 70%。
3. 至少 20 个可判断方向的案例。
4. 三 Seed 中至少两个 Seed 呈现正向排序关系。
5. 结果不是由少数极端点单独决定。

达到后标记：

`PREDICTIVE_SIGNAL_EXPLORATORY`

这些是固定的探索性工作门槛，不意味着独立统计确认。

如果有效案例不足：

`PREDICTION_EVIDENCE_INCONCLUSIVE`

不得降低门槛重新判定。

## 12.3 相对于历史指标的改进

如果 S-K10、S-Cell 或 S-Slope 比 S-Cal、S-Direct 有更好的结果，应报告：

- 排序增益。
- 方向预测增益。
- 不确定性。
- 计算成本增量。
- 适用架构和路由。
- 是否在各 Seed 中一致。

不能仅因为相关系数略高，就认定新指标可以替代旧的正式质量接受策略。

---

# 十三、预测模型的边界

本轮不训练新的 Value Network。

原因：

- 目前只有 48 个历史候选。
- 仅三个独立 Seed。
- 多个候选高度相关。
- 原始 12 个 R3j 长期标签已被研究人员查看。
- 训练灵活预测器容易产生强烈过拟合。

因此，本轮优先比较明确的低复杂度预测指标。

如果发现较稳定的指标，下一阶段才能研究是否需要使用可训练的价值估计器。

任何后续训练出的价值模型，都需要来自新的 Connection 提案和独立训练轨迹的验证，不能在同一批 48 个历史候选上训练并声称已确认。

---

# 十四、计算成本与信息价值

本轮必须单独统计：

### 历史恢复成本

- R3i Checkpoint 下载。
- R3j Checkpoint 下载。
- Receipt 验证。
- Model/Optimizer 恢复。

### Predictor 计算成本

- S-Direct。
- S-K10。
- S-Cell。
- S-Slope。

### 长期续训成本

- 36 个 Old/New 续训。
- 6,480 次新增 Cell 更新。
- 3,317,760 个新增训练 Token。

### 评价成本

- 全部 48 个案例。
- 三个时点。
- 每时点四格交叉评价。
- 两次确定性重复评价。
- Bootstrap。
- 结果发布。

必须区分：

- 实际硬件时间。
- 执行算子数量。
- 估算 FLOPs。
- 存储与恢复开销。

**R3k 的主要价值指标不是每 GPU 小时降低多少 NLL，而是能否以可控制的额外计算成本获得关于 Connection 未来学习价值的可验证预测信息。**

---

# 十五、工程结构

建议新建：

```text
research/projects/
  cellular-language-models/
    experiments/
      clm-001-qfirst-integrated/
        r3k/
          SPEC.md
          README.md

          config/
            frozen.json

          implementation/
            minicells_r3k/
              parent_restore.py
              candidate_registry.py
              data_boundaries.py
              probe_features.py
              prediction_freeze.py
              continuation.py
              attribution.py
              evaluation.py
              statistics.py
              storage.py
              reports.py
              runner.py

          notebooks/
            kaggle-future-value.ipynb

          reports/
            INPUT_VERIFICATION.json
            HISTORICAL_CANDIDATES.json
            DATA_BOUNDARY_PROOF.json
            FROZEN_PREDICTIONS.json
            CONTINUATION_CHECKPOINTS.json
            LONG_HORIZON_VALUE.json
            CELL_LEARNING_ATTRIBUTION.json
            PREDICTIVE_SIGNAL_ANALYSIS.json
            COMPUTE_COST.json
            SCIENTIFIC-FINDINGS.md
            RUN_MANIFEST.json
```

复用：

- R3i 事务 Store。
- R3i Old/New 恢复。
- R3j 配对续训。
- R3j 长期交叉评价。
- R3h H/T 路由。
- 既有 Tokenizer/Data Loader。
- HF 私有 Checkpoint 存储。
- GitHub 聚合结果发布器。

不要复制模型网络定义。

---

# 十六、自动运行阶段

## Phase A：输入恢复

自动执行：

1. 固定 R3i Source/Config/Manifest。
2. 固定 R3j Source/Config/Manifest。
3. 恢复全部 48 个历史 Proposal。
4. 验证 K=10 真实 Old/New。
5. 恢复 R3j 已完成 12 个长期 Checkpoint。
6. 验证数据分区。
7. 生成不可变 Candidate Registry。

如果源 Checkpoint 缺失，停止相应案例并记录失败。

不允许重新生成历史 Connection Proposal。

## Phase B：预测冻结

执行：

1. 初始化固定 Predictor Probe。
2. 计算五项预测指标。
3. 记录全部 48 个预测结果。
4. 记录预测公式与数值误差。
5. 生成 `FROZEN_PREDICTIONS.json`。
6. 记录文件 SHA-256 和 Source Identity。
7. 在新增长期评价开始前完成持久化确认。

如果预测 Manifest 未成功保存，不允许进入正式新增长期训练阶段。

## Phase C：配对长期续训

对剩余 36 个候选：

- 从真实 K=10 状态恢复。
- 继续至 T=50。
- 保存。
- 继续至 T=100。
- 保存。

不中途读取 R3k Value Holdout 进行模型选择。

## Phase D：统一长期评价

冻结所有有效长期模型后：

1. 读取新 Value Holdout。
2. 评价全部 48 个候选。
3. 计算 T=10/50/100 四格 NLL。
4. 汇总长期收益。
5. 归因 Connection 与 Cell 效果。
6. 执行固定预测比较。
7. 计算不确定性。

## Phase E：结果发布

向私有 HF 发布完整 Checkpoint。

GitHub 仅发布：

- 配置与身份。
- 聚合评价。
- 案例级非敏感指标。
- 预测指标。
- 统计结果。
- 科学解释。
- Checkpoint 内容哈希引用。

不公开模型权重、私有存储凭据或原始训练数据。

远端发布必须验证 Commit SHA。

失败时允许仅重试发布，不重新执行训练。

---

# 十七、必须通过的测试

## 17.1 历史恢复

- 48 个候选身份正确。
- 全部 Scale 与历史记录一致。
- K=10 Old/New 模型匹配。
- Optimizer 恢复正确。
- Data Cursor 正确。
- 已提交 Connection 的历史链可验证。
- 被拒绝候选仍作为真实 Counterfactual New 分支保留。

## 17.2 Predictor

- 全部五项公式可复现。
- S-Direct 不执行 Optimizer 更新。
- S-K10 与 S-Cell 使用历史真实 K=10 状态。
- S-Slope 使用同一 Probe。
- 不读取 R3j 长期 Development 计算预测值。
- 不读取 R3k Value Holdout。
- 预测 Manifest 在新增长期训练前冻结。

## 17.3 训练

- Old/New 外生随机流相同。
- Train Token 相同。
- 绝对 Step 正确。
- Scheduler 不重新 Warmup。
- Connection 冻结。
- 不执行新的 Proposal。
- 不重复历史 K=10。
- 新增训练总预算正确。
- Checkpoint 可恢复。

## 17.4 评价

- 全部 48 个候选统一使用新 Holdout。
- 不混合 R3j Development 与 R3k Holdout。
- 2×2 归因满足代数关系。
- 重复评价误差在允许界内。
- Bootstrap 固定。
- 不利用长期结果调整预测指标。
- Confirmation 保持 SEALED。

---

# 十八、科学结果判定

R3k 必须分别给出以下结论。

### 1. 预测指标是否存在可用信号？

- `PREDICTIVE_SIGNAL_EXPLORATORY`
- `PREDICTIVE_SIGNAL_NOT_SUPPORTED`
- `PREDICTION_EVIDENCE_INCONCLUSIVE`

### 2. 是否存在可检测的长期学习价值？

- `LONG_HORIZON_VALUE_OBSERVED`
- `NO_DETECTABLE_LONG_HORIZON_VALUE`
- `INCONCLUSIVE`

### 3. 长期收益来自哪里？

- `DIRECT_ROUTE_DOMINANT`
- `CELL_ADAPTATION_DOMINANT`
- `MIXED_EFFECT`
- `NO_RESOLVED_COMPONENT`

这些必须依据定量归因，不能依据假设自动赋值。

### 4. 是否具备下一阶段优化依据？

只有当某个事前指标在新增 36 个案例中提供一致的探索性预测证据，并且其效果不能由单一案例或单一 Seed 解释时，才允许建议将其作为下一阶段 R3l 的候选决策信号。

即使达到这一条件，也只允许**提出新实验方案**，不得自动启动新的 Connection 优化训练。

---

# 十九、Codex 执行指令

请在 `ArcheLabs/mini-cells` 中实施：

`CLM-001-R3k-Future-Learning-Value-001`

本轮研究目标是：

**判断哪些在 Connection 候选被正式接受之前可以计算的指标，能够预测其对未来 Cell 学习的长期价值。**

严格要求：

1. 使用 R3i 的全部 48 个历史候选。
2. 复用 R3j 已完成的 12 个长期续训案例。
3. 新增剩余 36 个候选的长期配对续训。
4. 恢复真实历史 K=10 Old/New 分支，不重新生成候选。
5. 不训练 Connection，不修改 Cell 架构或路由语义。
6. 建立严格独立于长期评价的新 Predictor Probe。
7. 建立与所有既有数据角色隔离的 R3k Research Value Holdout。
8. 预先计算并冻结 S-Cal、S-Direct、S-K10、S-Cell、S-Slope。
9. 在预测冻结后继续训练新增 36 个候选。
10. 所有案例统一评价 T=10、50、100。
11. 使用同一 Holdout 执行四格 Cell–Connection 归因。
12. 重点比较短期总收益、直接路由价值和早期 Cell 适应价值的预测能力。
13. 对新增 36 个结果与完整 48 个结果分别报告。
14. 严格说明三个 Seed、事件相关性和历史结果已知的限制。
15. 不拟合高容量 Value Network。
16. 不使用 R3k 评价结果优化当前 Connection 接受门槛。
17. 不读取 Confirmation。
18. 维持可恢复的私有 HF Checkpoint 与完整身份哈希。
19. 发布全部正面与负面结果。
20. 不自动执行下一阶段训练。

最终必须回答三个核心问题：

**Q1：Connection 的即时执行收益，是否能够预测它在未来 100 Step 后的模型质量？**

**Q2：Cell 在 K=10 的早期适应效应，是否包含即时路由收益之外的长期预测信息？**

**Q3：是否存在一种不需要运行完整 100 Step 反事实训练，就能够识别未来更优计算组织的可靠信号？**

如果 R3k 找不到这样的预测信号，也属于有效的科学结果。

这将意味着 MiniCells 需要重新研究 Connection 如何获得长期信用分配，而不是继续使用当前短期局部 Q 来近似未来组织价值。

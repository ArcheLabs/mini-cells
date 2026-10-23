# MiniCells R3j — Connection Long-Horizon Value Validation

**Connection 长期反事实价值、短期接受预测能力与 Cell 持续适应验证**

**版本：** v1.0  
**实验 ID：** `CLM-001-R3j-Long-Horizon-Connection-Value-001`  
**仓库：** `ArcheLabs/mini-cells`  
**研究性质：** 历史只读审计 + 预注册配对续训  
**运行环境：** Kaggle T4 × 2  
**执行方式：** 一次启动，自动完成恢复、验证、续训、评价与发布  
**Confirmation：** `SEALED`  
**科学范围：** `EXPLORATORY_DEVELOPMENT`

---

# 1. 研究背景

## 1.1 R3i 的实验结论

R3i 成功完成稳定路由下的 Cell–Connection 联合训练。

正式结果：

| 指标 | R3i |
|---|---:|
| 新增联合训练模型 | 12 |
| Connection Proposal | 48 |
| Trust Region 通过 | 48 |
| Calibration 质量接受 | 8 |
| 可分辨的正向局部 Cell 效果事件 | 21 |
| 获得支持的终端质量假设 | 0/4 |

三 Seed 平均 Development NLL 差异：

| 主要比较 | Δ NLL（联合训练 − 冻结基线） |
|---|---:|
| A-CH vs A-FH | +0.000333 |
| A-CT vs A-FT | 0 |
| B-CH vs B-FH | +0.000180 |
| B-CT vs B-FT | +0.000210 |

正值表示联合训练略差。

这说明：

1. Connection 已经能够产生真实、受约束的更新。
2. Cell 已经能够在部分更新中形成可检测的适应效果。
3. 局部适应和短期质量接受，并没有转化为终端模型质量优势。
4. 当前 Connection 联合更新机制尚未获得额外模型质量收益的支持。

因此，本轮不再验证 Connection 能否更新，而是研究：

**现有 Connection 选择机制，是否能够识别对 Cell 的未来学习真正有利的组织。**

## 1.2 历史实验身份

必须使用 R3i 的固定历史结果。

```text
Experiment:
CLM-001-R3i-Stable-CoAdaptation-001

Source SHA:
80e49b071790c8bfb58dead19e6037259551ed61

Config SHA-256:
72de3cfb262a93ec6bb24b29b9f6c18d1d40030c76245cfda9a504852b30e630

Run ID:
ed08463119358da0f81b733786ab24bd89ac69c0f6b290dafc0f26311fe328a9
```

已发布的结果分支：

`results/r3i-4ef0eb15e8b35647-77b0f8dffd93`

结果 Commit：

`b5fce3287f8f1267e480deab9b60de1ad4956beb`

### 私有 Checkpoint 来源

```text
HF Repository:
libingjiang/minicells-research-checkpoints

HF Revision:
2fd527bf4958979f16da1fbc88ef5000bb70cc62

Manifest SHA-256:
25930ac73f54cb06cc1a61670e97aad7847f52a1aacca836d8595b64f7820a15
```

R3i Manifest 的：

`cross_session_verified=false`

意味着历史保存经过了即时回读，但尚不能直接声称独立 Kaggle Session 已成功恢复。

因此，R3j 必须首先执行真正的跨 Session 恢复验证。

---

# 2. 研究核心问题

本轮研究四个问题。

### Q1：短期价值是否具有持续性？

R3i 在完成 K=10 Cell 配对适应后，根据 Calibration 改善决定是否接受 Connection。

但是这种改善是否能够持续到：

\[
T=50
\]

和：

\[
T=100
\]

步，尚不清楚。

### Q2：当前质量接受决策是否具有预测价值？

对于 R3i 接受的 Connection：

- 长期质量是否仍优于保留旧 Connection？
- 早期收益是否消失？
- 是否存在反向损失？

对于拒绝的 Connection：

- 是否存在短期无益、长期有益的候选？
- 当前接受机制是否可能漏掉具有未来学习价值的计算组织？

### Q3：长期收益来自 Connection 还是 Cell？

即使新组织在 +100 Step 优于旧组织，也需要分辨：

- Connection 直接改善路由。
- Cell 在新 Connection 下学习得更好。
- Cell 与 Connection 形成有利的交互。

### Q4：现有反事实 Q 是否能预测长期学习价值？

R3i 的 Connection Proposal 基于当前 Cell 状态估计动作的反事实价值。

本轮检验这一短期目标与真实长期收益之间的关系。

**若历史候选没有保存可验证的数值预测，不允许重新生成预测并伪装为历史值。**

---

# 3. 核心定义：Connection 的长期学习价值

设历史提案发生在主训练 Step \(s\)。

当时的正式模型为：

\[
M_s=(\theta_s,\phi_s)
\]

其中：

- \(\theta_s\) 表示 Cell 参数。
- \(\phi_s\) 表示旧 Connection。

已保存的真实候选是：

\[
\phi'_s
=
\phi_s+\alpha_s\Delta\phi_s
\]

保持 Cell 训练设置不变，分别沿两种组织继续训练。

### Old Continuation

\[
\theta^{O}_{s+T}
=
\operatorname{Train}_{T}
(\theta_s,\phi_s)
\]

### New Continuation

\[
\theta^{N}_{s+T}
=
\operatorname{Train}_{T}
(\theta_s,\phi'_s)
\]

定义：

\[
\boxed{
V_T
=
L(\theta^{O}_{s+T},\phi_s)
-
L(\theta^{N}_{s+T},\phi'_s)
}
\]

解释：

- \(V_T>0\)：新 Connection 在该训练跨度和评价数据上更好。
- \(V_T=0\)：未检测到收益。
- \(V_T<0\)：新 Connection 更差。

核心观察时点：

\[
\boxed{
T\in\{10,50,100\}
}
\]

这三个时点全部相对于原始 Proposal Step 计算。

例如：

```text
Proposal at Step 200

T=10  → Step 210
T=50  → Step 250
T=100 → Step 300
```

不能把已经完成的 K=10 再额外计入 T。

---

# 4. 实验范围

## 4.1 两个阶段

本轮分为：

**Stage A：Historical Proposal Audit**

对 R3i 的全部 48 个 Proposal 做只读审计。

**Stage B：Long-Horizon Paired Continuation**

从历史 K=10 旧、新适应分支出发，对预注册的 12 个候选继续训练。

两个阶段属于同一个 R3j 实验。

Stage A 不得根据结果临时改变 Stage B 的选择规则。

## 4.2 原始模型保持不变

保留：

- 六个 Cell。
- A：四阶段 Top-2。
- B：八步顺序 Cell 图。
- 原始 Gate。
- Connection 修正结构。
- H：稳定硬路由 + ε。
- T：原始冻结低温路由 + ε。
- R3h 的随机流协议。
- Cell AdamW。
- 原始训练数据和 Tokenizer。

## 4.3 明确禁止

不允许：

- 生成新的 Connection Proposal。
- 重新优化 Connection 参数。
- 执行新的 Trust Region 扫描以改变原始候选。
- 改变历史接受或拒绝决定。
- 改变温度。
- 改变 ε 调度。
- 改变 Cell 网络。
- 增加 Cell 或改变图结构。
- 引入 Balance Loss。
- 根据 Development 结果追加候选。
- 使用 Confirmation。
- 覆盖 R3i 历史 Checkpoint。

---

# 5. Stage A：全部 48 次历史提案审计

## 5.1 数据来源

优先读取 R3i 已发布的：

```text
PROPOSAL_RESULTS.json
TRUST_REGION_RESULTS.json
ADAPTATION_RESULTS.json
CELL_CONNECTION_ATTRIBUTION.json
GRAPH_ORGANIZATION.json
RUN_MANIFEST.json
```

必要时回读对应私有 Store。

不允许仅根据发布报告反向猜测参数。

## 5.2 提案身份

每次提案使用：

```text
source_run_id
seed
arm
proposal_at_step
candidate_fingerprint
chosen_scale
transaction_receipt_sha256
```

建立唯一 Candidate ID。

历史 Proposal 时间：

\[
s\in\{100,200,300,400\}
\]

每个 Seed 对应：

```text
A-CH
A-CT
B-CH
B-CT
```

总计：

\[
3\times4\times4=48
\]

个 Proposal。

## 5.3 历史指标

对每个 Proposal 整理：

| 类别 | 指标 |
|---|---|
| Proposal | 完整参数哈希、选择尺度 |
| Trust Region | KL、动作变化、完整路径变化 |
| Short-Term Quality | K=10 Calibration Old/New NLL |
| Cell Attribution | Route Effect、Cell Effect、Interaction |
| Adaptation | 实际训练路径差异、Cell 梯度差异 |
| Decision | Accepted / Rejected |
| History | 提案时正式 Connection Version |
| Cost | Proposal、扫描和 K=10 适应成本 |

## 5.4 反事实 Q 预测信号

尽可能恢复原始 Proposal 计算过程中已经保存的：

- Counterfactual CE。
- 动作价值差异。
- Policy Objective。
- Proposal Optimization Loss。
- 候选参数修改。
- 任务驱动方向的信息。

必须区分：

1. 历史直接保存的预测值。
2. 根据完整历史数据进行确定性重算得到的指标。
3. 无法恢复的指标。

对于第二类，必须记录重算代码、输入哈希和公式。

不得把第三类填成零，也不得用事后长期效果构造所谓历史预测分数。

如果没有足以定义统一标量预测的历史信息，报告：

`HISTORICAL_VALUE_PREDICTION_NOT_RECOVERABLE`

不影响其他 R3j 实验继续完成。

## 5.5 阶段 A 的输出

生成：

`HISTORICAL_PROPOSAL_AUDIT.json`

至少包含 48 条完整记录。

还要统计：

- 质量接受率。
- 局部收益分布。
- 不同尺度的候选分布。
- 正向/负向 Cell 效果的比例。
- 直接路由收益与 Cell 效果的关系。
- 原始预测值与 K=10 实际收益的关系。

此阶段只描述原始数据，不声称已经验证长期价值。

---

# 6. Stage B：候选选择

## 6.1 固定 12 个候选

本轮选择：

- 8 个 R3i 正式接受的候选。
- 4 个 R3i 正式拒绝的候选。

总数：

\[
\boxed{12}
\]

### Accepted Group

纳入所有八个历史 Accepted Proposal。

不得根据其后续质量表现剔除其中任何一个。

### Rejected Group

从每个 Arm 选择一个被拒绝候选。

四个 Arm：

```text
A-CH
A-CT
B-CH
B-CT
```

选择规则：

1. 过滤 `accepted=false` 且 `trust_region_passed=true` 的候选。
2. 按 `seed` 从小到大排序。
3. 在同一 Seed 内，按 `proposal_at_step` 从小到大排序。
4. 选择第一个。

因此每个 Arm 恰好选一个 Rejected Proposal。

如果某个 Arm 没有符合条件的候选，不得从其他 Arm 临时替换；报告缺失并减少本轮可执行案例数。

## 6.2 冻结 Case Manifest

在实际读取长期评价指标之前，生成：

`CASE_SELECTION_MANIFEST.json`

记录：

- 12 个预选 Candidate ID。
- Accepted / Rejected。
- 原始 Proposal Step。
- Old/New 分支 Receipt。
- 筛选理由。
- 选择规则版本。
- Manifest SHA-256。

之后不得修改案例名单。

## 6.3 科学限制

这个案例集合不是全部 48 个候选的随机代表样本。

因此：

**R3j 可以评价 12 个案例各自的长期反事实效果，但不能据此无偏估计全部候选的长期成功率。**

Accepted 与 Rejected 的组间均值也不得直接解释为原始接受策略的总体准确率。

---

# 7. 历史分支恢复

这是本轮最重要的工程门禁。

## 7.1 R3i 实际事务结构

已核对 R3i 的：

`minicells_r3i/adaptation.py`

每个 Proposal 事务位于：

```text
private-checkpoints/
  jobs/
    seed-<seed>-arm-<arm>/
      transactions/
        proposal-<step>/
```

实际相对路径须以恢复后的私有 Store 为准。

R3i 对 K=10 的每次适应保存：

```python
tx["old"]
tx["new"]
tx["adapt_done"]
tx["rows"]
```

第十次适应完成后，两条分支的完整状态已经写入对应事务快照。

事务最终提交还保存：

```python
tx["selected"]
tx["event"]
```

因此，恢复时应读取具有完整 `old/new` 字段的 K=10 事务快照，而不是仅加载最终 Selected Checkpoint。

## 7.2 具体恢复要求

每个案例必须恢复：

### 原始正式状态

\[
(\theta_s,\phi_s)
\]

### K=10 Old

\[
(\theta^O_{s+10},\phi_s)
\]

### K=10 New

\[
(\theta^N_{s+10},\phi'_s)
\]

以及各自：

- Cell Optimizer。
- Scheduler 相关状态。
- Data Cursor。
- Route RNG。
- 数据 RNG。
- Model State。
- Connection Version。
- 事务身份。
- 已完成的主 Step。
- 已累计训练 Token。

## 7.3 事务恢复验证

要求：

```text
adapt_done = 10
old.step = proposal_at_step + 10
new.step = proposal_at_step + 10
```

Old/New 的：

- Data Cursor 相同。
- 已消费主训练 Token 相同。
- 初始 Cell 参数来自相同 Proposal 正式状态。
- 配对随机流和外生输入符合历史记录。

Connection 参数分别对应旧状态和历史候选。

## 7.4 Trial 与 Official 的区别

R3i 在 K=10 配对适应中，New Connection 是经过 Trial Authorization 装入新分支的。

**即使历史候选最终被拒绝，这个 New 分支也是真实存在的反事实模型。**

因此，R3j 可以继续运行该 New 分支。

但要保留：

- `historical_accepted=false`
- `counterfactual_branch=true`

不得为了恢复被拒绝分支，把它篡改成 R3i 当时实际接受的正式 Connection。

## 7.5 必须验证跨 Session

使用独立 Kaggle Session：

1. 从固定 HF Revision 下载 Manifest。
2. 验证 Manifest SHA。
3. 下载完整 Store 对象。
4. 验证所有 Receipt。
5. 恢复历史模型参数。
6. 恢复训练状态。
7. 对照 R3i 的 K=10 NLL、参数哈希和训练身份。

如果无法恢复完整 Old/New 分支：

`PAIRED_BRANCH_NOT_RECOVERABLE`

对应案例停止。

不允许通过重新执行历史十步适应来伪造原始分支。

若 8 个 Accepted 案例中存在缺失，必须明确标注本轮 Accepted 样本不完整，不得将完成案例的结果当作全部八次接受的结果。

---

# 8. 长期配对训练

## 8.1 从哪里继续？

本次不从 Proposal Step 重新训练 K=10。

直接从历史 K=10 的真实 Old/New 状态继续。

例如：

```text
Original Proposal: Step 100

Historical Old at Step 110
Historical New at Step 110

R3j:
  Continue both to Step 150
  Continue both to Step 200
```

这能够确保：

- 前十步与 R3i 历史实验一致。
- 不浪费已经完成的适应计算。
- 不重新生成任何 Connection 候选。
- 研究变量只有 Old/New 计算组织的持续影响。

## 8.2 两个续训终点

从已保存的 Step \(s+10\) 继续训练到：

\[
s+50
\]

和：

\[
s+100
\]

因此：

- 第一段新增 40 次 Cell 更新。
- 第二段新增 50 次 Cell 更新。

每条分支最多新增：

\[
40+50=90
\]

次 Cell 更新。

## 8.3 禁止后续 Connection 更新

本轮全部续训期间：

- 旧 Connection 始终保持旧状态。
- 新 Connection 始终保持历史候选状态。
- 不执行新的 Proposal。
- 不执行 Trust Region。
- 不执行新的质量接受。
- 不允许根据评价结果切换分支。

即使训练过程中经过原始的 Step 200、300 或 400 Proposal 时刻，也不能自动执行 R3i 的 Proposal 调度。

具体实现应该使用独立 R3j Continuation Runner。

不得简单调用包含原始 Proposal 调度的 R3i `run_job()`。

## 8.4 原始学习率与全局进度

必须保留 R3i 的：

- Cell LR。
- Weight Decay。
- Gradient Clip。
- Scheduler。
- Absolute Main Step。
- Epsilon 调度。
- H/T 训练路由。
- 原始固定 Temperature。

不能因为 R3j 是新实验而重新执行 Warmup。

例如，原始 Proposal 发生在 Step 400，恢复于 Step 410：

续训必须继续使用 Step 410–499 的历史全局训练调度。

不得重置为 Step 0–89。

## 8.5 配对数据与随机流

Old/New 必须使用：

- 完全相同的后续 Train Token。
- 完全相同的数据顺序。
- 相同的外生随机数。
- 相同的训练预算。
- 相同的 Cell Optimizer 配置。
- 相同的 Epsilon 进度。

允许由于 Connection 不同导致路由动作不同。

外生随机流必须沿用 R3h/R3i 的：

`KEYED_CPU_TORCH_STREAMS_V1`

不得重新设计采样随机数协议。

## 8.6 防止未来数据泄漏

每个历史案例的训练延续都应按其原始全局 Step 使用对应的 Train 数据。

不能把其他案例的评价窗口、Calibration 或 Development 数据混入续训。

多个案例之间可能使用重叠的历史训练区间，这是独立反事实案例的正常结果。

但后续统计必须考虑它们的相关性。

---

# 9. 质量评价

## 9.1 三个时点

对每个案例评价：

\[
T=10,50,100
\]

需要注意：

T=10 是已保存的历史适应终点。

T=50 和 T=100 是 R3j 新增的训练终点。

## 9.2 Calibration 与 Development 的不同角色

历史 T=10 的 Calibration：

用于恢复和复现 R3i 当时的接受判断。

后续新训练不得继续用它选择 Connection。

长期分析采用预先冻结的 Development 评价窗口，分别对 T=10、50、100 的 Old/New 状态进行评价。

因为 Development 在之前多轮实验中已被使用，所以：

**这些评价只能视为探索性长期分析，不能视为全新独立验证。**

必须保持：

- 相同 Tokenizer。
- 相同 Development Window。
- 相同计算精度。
- 相同稳定硬路由。
- 相同 Loss 定义。

## 9.3 统一终端评价屏障

建议先完成全部选定案例的 +100 续训、保存三个时点的 Checkpoint，再统一加载 Development 计算各时点 NLL。

不能根据 T=50 的 Development 结果提前停止或增加训练。

如果确实存在从未被此前实验用于训练、选择或评价的非 Confirmation 数据，可以单独预注册为附加审计 Holdout，但必须验证数据来源和排除重复内容。

没有这样一份数据，也不影响 R3j 的探索性分析。

绝对不能将 Confirmation 改名作为新的审计数据。

## 9.4 长期价值

每个时点计算：

\[
\boxed{
V_T=L^O_T-L^N_T
}
\]

分别记录：

\[
V_{10},V_{50},V_{100}
\]

定义描述性变化：

\[
\Delta V_{10\rightarrow100}
=
V_{100}-V_{10}
\]

它表示相对于第 10 步，候选优势在第 100 步是扩大还是减小。

不能只通过一个差值断言训练过程单调改进。

---

# 10. 长期 Cell–Connection 归因

本次除 Old/New 总 NLL 外，还必须继承 R3g 的交叉评价方法。

## 10.1 四格评价

在 T=10、50、100：

| Cell 参数来源 | Old Connection | New Connection |
|---|---|---|
| Old 续训分支 | \(L_{OO,T}\) | \(L_{ON,T}\) |
| New 续训分支 | \(L_{NO,T}\) | \(L_{NN,T}\) |

其中：

\[
V_T=L_{OO,T}-L_{NN,T}
\]

## 10.2 Connection 的直接效果

\[
\boxed{
R_T=L_{OO,T}-L_{ON,T}
}
\]

正值表示在旧 Cell 下，新 Connection 直接改善路由质量。

## 10.3 Cell 的适应效果

\[
\boxed{
C_T=L_{ON,T}-L_{NN,T}
}
\]

正值表示在新 Connection 下，新分支 Cell 比旧分支 Cell 更好。

总收益满足：

\[
\boxed{
V_T=R_T+C_T
}
\]

这是一个明确的、以旧 Cell 为路线分解起点的加法分解。

不得再将交互项额外相加到这个等式中。

## 10.4 非加性交互

另计算：

\[
\boxed{
I_T=
L_{NN,T}-L_{NO,T}-L_{ON,T}+L_{OO,T}
}
\]

如果 \(I_T<0\)，表明在该四格评价中，Cell 来源与 Connection 之间存在有利的非加性交互。

但它不能单独证明 Cell 已经形成新的语义能力。

## 10.5 研究目标

观察：

- \(R_T\) 是否持续为正。
- \(C_T\) 是否随训练扩大。
- \(I_T\) 是否出现稳定的有利趋势。
- Cell 参数分化是否累积。
- 路由分工是否发生持续变化。

尤其重要的是：

**如果 \(V_{10}>0\)，但 \(V_{100}<0\)，需要判断反转主要来自路由效果还是 Cell 适应效果。**

---

# 11. Cell 组织稳定性

每个案例在三个时点记录：

- 六个 Cell 的使用比例。
- Cell 调用次数最大/最小比。
- Cell 梯度更新覆盖。
- 平均 Gradient Norm。
- 确定性路径差异。
- 实际训练路径差异。
- H/T 路由熵。
- 推理动作匹配概率。
- Cell 参数差异。
- A 的 Pair 分布。
- B 的顺序转移矩阵。
- B 的自连接比例。

必须区分：

**组织改变了**，与 **这种组织变化有价值**。

不得把 Cell 使用差异、较低熵或某个转移矩阵变得更稀疏直接算作质量收益。

---

# 12. 统计与科学判定

## 12.1 不将历史选择标签当作随机分组

Accepted 候选是根据原始 K=10 Calibration 改善选择出来的。

Rejected 候选是其中未达到接受门槛的案例。

因此：

- Accepted 与 Rejected 不是随机分配。
- 两组的原始提案质量可能不同。
- 同一 Seed 中的候选互不独立。
- 同一 Arm 的多个候选可能共享历史 Connection 参数。
- 多个 Proposal 的后续训练区间可能重叠。

必须报告这些限制。

## 12.2 个别案例的长期结果

对每个案例定义：

`LONG_HORIZON_POSITIVE`

条件：

\[
V_{100}>0
\]

且大于实际重复评价数值误差。

`LONG_HORIZON_NEGATIVE`

条件：

\[
V_{100}<0
\]

且绝对值大于数值误差。

`BELOW_NUMERICAL_RESOLUTION`

长期收益不足以与数值噪声区分。

## 12.3 实际效果门槛

同时保留一个更严格的描述性门槛：

\[
\boxed{
V_{100}\ge0.005
}
\]

将达到这一门槛的案例标记：

`PRACTICAL_LONG_HORIZON_GAIN`

这沿用近期实验中的 0.005 NLL 实际改善量级。

未达到这个门槛，但 \(V_{100}\) 明显大于数值误差的结果，可以称为小幅正向效果。

不得二者混淆。

## 12.4 短期预测长期

对全部有结果的案例，记录：

- \(V_{10}^{Calibration}\)
- \(V_{10}^{Development}\)
- \(V_{50}^{Development}\)
- \(V_{100}^{Development}\)
- 历史 Accepted/Rejected
- 候选原始 Q 预测（若可恢复）

可以进行描述性 Spearman 相关、符号一致率及排序分析。

但不得把这 12 个经过选择的案例的相关性当作全部 48 个提案的无偏估计。

如果历史预测值缺失，明确标记该分析无法实施。

## 12.5 Bootstrap

使用固定：

```text
bootstrap_replicates = 2000
bootstrap_seed = 2026102302
```

以 Story 为聚类单位，计算案例内 Old/New 配对质量差异区间。

同时报告：

- 三个训练 Seed 的覆盖情况。
- 各 Seed 内事件数量。
- 同一 Seed 多次事件的依赖关系。
- 12 个案例的完整逐项结果。

Bootstrap 不能替代独立训练 Seed。

本轮不进行独立 Confirmation 声明。

---

# 13. 计算预算

## 13.1 原始 K=10

历史已经完成。

本轮不重新训练。

## 13.2 新增续训

每个案例：

\[
2\text{ Branches}\times90\text{ Updates}
\]

共：

\[
12\times2\times90=2160
\]

次 Cell 更新。

每次：

\[
512\text{ Tokens}
\]

新增训练 Token：

\[
\boxed{
1,105,920
}
\]

预算表：

| 项目 | 数量 |
|---|---:|
| 只读历史提案 | 48 |
| 长期候选 | 12 |
| 每候选分支 | 2 |
| 历史 K=10 复用 | 240 次已完成更新的分支状态基础 |
| 新增 Cell 更新 | 2,160 |
| 新增训练 Token | 1,105,920 |
| 新 Connection Proposal | 0 |
| 新 Connection Optimizer 更新 | 0 |
| 新 Trust Region 搜索 | 0 |

新增成本不含：

- 跨 Session HF 恢复。
- 校验和快照。
- 三个时点的质量评价。
- 2×2 归因。
- 路由监测。
- GitHub 结果发布。

所有实际成本独立记录。

---

# 14. 运行恢复与私有存储

## 14.1 独立实验身份

R3j 建立新的：

```text
experiment_id
source_sha
config_sha256
run_id
execution_id
parent_r3i_run_id
parent_manifest_sha256
case_selection_manifest_sha256
```

不修改 R3i 身份。

## 14.2 独立 Continuation Checkpoint

每个案例建立：

```text
case_id/
  old/
    historical-step-10
    horizon-50
    horizon-100

  new/
    historical-step-10
    horizon-50
    horizon-100
```

实际存储仍使用现有 Store 及内容寻址约定。

必须记录：

- 原始 R3i 分支 Receipt。
- 导入后的 R3j 身份。
- 每条分支的来源。
- Model Hash。
- Connection Hash。
- Optimizer Hash。
- Data Cursor。
- Main Step。
- RNG 状态。

## 14.3 恢复门禁

中断恢复必须保证：

- Old/New 不能交换。
- 历史 Accepted/Rejected 标签不能改变。
- 已完成的续训 Step 不重复执行。
- 训练 Token 不重复计入。
- 评价没有改变训练 RNG。
- Connection 参数保持冻结。
- 没有触发原始 R3i Proposal 调度。

## 14.4 跨 Session 复核

在新 Session 完成一次：

1. 从固定私有 HF Revision 读取 R3i。
2. 验证原始 Receipt。
3. 恢复两个分支。
4. 对照历史模型哈希。
5. 恢复训练游标与 Optimizer。
6. 完成初始非更新前向。
7. 保存并回读 R3j 导入 Checkpoint。

全部通过后才开始新增训练。

若失败，不允许默默重训历史分支。

---

# 15. 输出文件

建议目录：

```text
research/projects/
  cellular-language-models/
    experiments/
      clm-001-qfirst-integrated/
        r3j/
          SPEC.md
          README.md

          config/
            frozen.json

          implementation/
            minicells_r3j/
              parent_restore.py
              transaction_restore.py
              historical_audit.py
              case_selection.py
              continuation.py
              attribution.py
              organization.py
              evaluation.py
              statistics.py
              storage.py
              reports.py
              runner.py

          notebooks/
            kaggle-long-horizon.ipynb

          reports/
            PARENT_RESTORE_VERIFICATION.json
            HISTORICAL_PROPOSAL_AUDIT.json
            CASE_SELECTION_MANIFEST.json
            CONTINUATION_CHECKPOINTS.json
            LONG_HORIZON_NLL.json
            LONG_HORIZON_ATTRIBUTION.json
            ROUTING_EVOLUTION.json
            PREDICTIVE_VALUE_ANALYSIS.json
            COMPUTE_COST.json
            SCIENTIFIC-FINDINGS.md
            RUN_MANIFEST.json
```

优先复用：

- R3i 的 Model/Trainer。
- R3i 的 Store 与事务验证。
- R3h 的 H/T 路由实现。
- R3g 的交叉归因代码。
- 原始数据加载与 Evaluation。
- 现有 HF 和 GitHub 发布模块。

不要复制新的 Cell 模型实现。

---

# 16. 自动执行方案

Codex 创建：

```bash
python scripts/run_r3j.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3j/config/frozen.json \
  --run-all
```

这是本轮需要新增的命令。

Notebook 仅负责固定源码、准备环境并调用该入口。

## Stage A：输入与恢复

自动：

1. 检查 R3i Source、Config 和 Run Identity。
2. 从私有 HF 固定 Revision 恢复 Store。
3. 验证历史 Manifest 和 Checkpoint 哈希。
4. 恢复全部 48 个 Proposal 元数据。
5. 验证 K=10 Old/New 事务状态。
6. 形成历史审计。
7. 冻结 12 个案例名单。

没有完整候选的案例必须标记缺失。

## Stage B：继续训练

对每个有效案例：

1. 导入 K=10 Old/New。
2. 检查训练起点及随机流。
3. 继续训练 40 Step，到 T=50。
4. 保存两条完整分支。
5. 再继续训练 50 Step，到 T=100。
6. 保存两条完整分支。

严格禁止调用新的 Connection Proposal。

## Stage C：统一评价

全部可用案例完成后：

1. 读取冻结的 Development Window。
2. 评价 T=10、50、100。
3. 计算每个时点的 Old/New NLL。
4. 执行 2×2 Cell–Connection 交叉评价。
5. 分析 Cell 梯度、路径和组织变化。
6. 计算长期价值与短期预测关系。

没有经过原始选择的新评价数据不得用于训练或超参数选择。

## Stage D：科学判断

自动生成：

- 每案例长期价值。
- Accepted 与 Rejected 案例结果。
- 短期与长期符号一致率。
- 直接路由收益与 Cell 适应收益。
- 不同架构和 H/T 路由的描述性比较。
- 计算成本。
- 科学限制。

## Stage E：发布

私有 HF 保存完整 Checkpoint。

GitHub 仅发布聚合结果。

远端结果分支必须核验 Commit SHA。

发布失败时允许仅重新发布，不重新训练。

---

# 17. 科学结果分类

本次不以所有 Accepted 候选获得长期正向效果为成功要求。

允许任何科学负结果。

### 情况 A：短期接受具有持续收益

如果多数已接受案例的：

\[
V_{10}>0
\]

且：

\[
V_{100}>0
\]

说明当前质量选择可能已经具备一定的长期价值，但原始 R3i 完整训练未能积累可检测的优势。

下一步研究累计更新及组织间干扰。

### 情况 B：短期收益频繁反转

如果：

\[
V_{10}>0
\]

但：

\[
V_{100}<0
\]

在多例中出现，说明当前 K=10 质量选择不足以可靠预测更长训练跨度。

下一步应研究长期反事实接受机制。

### 情况 C：被拒绝候选具有长期收益

如果某些 Rejected Candidate：

\[
V_{10}\le0
\]

但：

\[
V_{100}>0
\]

说明短期质量接受可能错过未来有价值的组织。

下一步应考虑把未来学习价值纳入 Connection 决策，而不只是放宽 Calibration 阈值。

### 情况 D：长期收益主要来自直接路由

如果：

\[
R_{100}>0
\]

而：

\[
C_{100}\approx0
\]

说明 Connection 主要通过更好的推理路径产生价值。

此时不能宣称已经实现有益的 Cell–Connection 共同演化。

### 情况 E：出现长期 Cell 适应收益

如果：

\[
C_{100}>0
\]

并且超过数值误差，同时观察到真实训练路径及 Cell 参数分化，则为未来学习价值提供更直接的证据。

但仍需要在未用于设计决策的后续实验中重复验证。

---

# 18. 完成门禁

R3j 只有满足以下条件，才算有效完成：

- R3i 固定身份验证完成。
- 私有 Checkpoint 成功跨 Session 恢复。
- 48 次 Proposal 历史审计完成，或明确列出不可恢复项。
- 12 个案例按预注册规则冻结。
- K=10 Old/New 分支通过身份和模型校验。
- 不重新执行历史 K=10。
- Old/New 继续使用相同训练数据和外生随机流。
- 无新的 Connection Proposal。
- Connection 参数在续训期间不变。
- 有效案例完成 +50 和 +100。
- 每个时点的 NLL 可复现。
- 完成 2×2 交叉归因。
- 不根据中间评价提前停止训练。
- Development 未用于候选选择。
- Confirmation 始终 `SEALED`。
- 保存完整成本。
- 结果文件包含来源及内容哈希。
- GitHub 发布成功或保留可独立重试的回传状态。
- 不自动进入 R3k。

工程失败必须单独报告。

不能把恢复失败、数据不一致或数值错误解释成科学负结果。

---

# 19. 最终给 Codex 的实施指令

请在 `ArcheLabs/mini-cells` 中实现：

`CLM-001-R3j-Long-Horizon-Connection-Value-001`

这是基于 R3i 历史 Checkpoint 的长期反事实价值验证。

本轮不重新训练完整 R3i，也不重新优化 Connection。

要求：

1. 使用 R3i 的固定 Source SHA、Config SHA 和私有 HF Manifest。
2. 首先完成跨 Session Checkpoint 恢复验证。
3. 只读审计全部 48 个历史 Connection Proposal。
4. 冻结全部 8 个 Accepted 候选。
5. 按每 Arm 最早有效拒绝事件的规则选择 4 个 Rejected 候选。
6. 从历史 K=10 事务恢复每个候选的真实 Old/New 模型与 Optimizer 状态。
7. 在不执行后续 Connection Proposal 的前提下，配对续训至相对 +50、+100 Step。
8. 保留原始 H/T 路由、温度、全局 Step、Epsilon 调度与外生随机流。
9. 用统一的确定性推理对三个时点进行 Development NLL 评价。
10. 对每个时点执行 Cell × Connection 的四格归因。
11. 分析短期接受效果与长期学习价值之间的关系。
12. 区分 Connection 直接效果、Cell 适应效果和非加性交互。
13. 分析长期训练中的 Cell 组织与梯度分配变化。
14. 明确样本选择偏差、历史 Development 重复使用和 Seed 相关性。
15. 不读取 Confirmation。
16. 保留所有旧、新分支 Checkpoint 与实际计算成本。
17. GitHub 只发布聚合结果。
18. 不自动启动下一阶段训练。

最终必须回答：

**第一，R3i 的 Connection 更新在短期获得的质量收益，究竟能够持续多久？**

**第二，当前 K=10 的 Connection 接受机制，是否会接受长期无益的组织，或拒绝具有长期潜力的组织？**

**第三，Connection 的长期价值主要来自立即改变计算路径，还是来自改变 Cell 在后续训练中的学习过程？**

本次实验的价值不取决于是否发现正向收益。

**只要能够在相同决策 Checkpoint 下，可靠地测量 Connection 的长期反事实学习价值，就为下一阶段设计真正面向未来学习能力的 Connection 更新策略建立了基础。**

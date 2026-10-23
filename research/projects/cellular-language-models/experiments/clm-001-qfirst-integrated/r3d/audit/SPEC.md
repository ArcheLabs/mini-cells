# MiniCells R3d-A — Checkpoint Read-Only Mechanism Audit

**R3d 长期信用与联合训练失败机制分析**

实验 ID：`CLM-001-R3d-AUDIT-001`
实验类型：只读机制分析
目标仓库：`ArcheLabs/mini-cells`
分析基础：R3d 已完成的全部 Checkpoint
执行方式：一次性实现、直接运行、自动生成完整报告
数据权限：Train、Calibration、Development；Confirmation 保持 SEALED

---

## 1. 研究目的

### 1.1 背景

R3d 已完成三个 Seed、六个 Arm 的全部训练。

正式结果：

| Arm | 平均 Development NLL |
|---|---:|
| F0 — 原始冻结模型 | 4.550880 |
| L — 长期信用 Connection | 4.604461 |
| M — 即时信用 Connection | 4.596800 |
| F — 冻结 Router、训练 Cell | **4.513366** |
| P — 冻结已学习 Connection、训练 Cell | 4.514825 |
| J — Cell 与 Connection 联合训练 | 4.529020 |

R3d 的 H1–H4 均为 `NOT_SUPPORTED`。

主要观察：

1. 长期信用训练后的 Connection 明显劣于原始冻结 Router。
2. 长期信用未稳定优于即时信用。
3. Cell 与 Connection 联合训练劣于固定 Router。
4. Connection 的训练期候选排序准确率约为 47%–55%。
5. Connection 在训练期间改变了大量原始路由选择。
6. 实际训练的 Cell 没有再次出现明显的持续性零曝光问题。

其中，H4 未被支持，不代表发生了曝光坍缩。R3d 的 H4 判定规则包含 H3 成立这一前置条件。

### 1.2 本次需要回答的问题

本次分析集中回答四个问题：

**Q1：反事实价值是否真实、稳定，并且可以被 Connection 预测？**

**Q2：R3d 的 Connection 是否学会了有价值的动作排序？**

**Q3：性能恶化是否主要来自 Connection 对原始路由的过度修改？**

**Q4：Cell 与 Connection 联合训练的负结果，是错误选择造成的，还是二者协同适应过程造成的？**

本次分析必须区分这几种解释，不能简单地将结果归结为「Router 没有学会」。

---

## 2. 分析边界

### 2.1 严格只读

禁止：

- 重新训练 Cell。
- 重新训练 Connection。
- 更新任何模型参数。
- 创建新的 Optimizer 训练过程。
- 修改已有 Checkpoint。
- 修改 R3c、R3d 的正式结果。
- 启动新的 R3e 实验。
- 引入新的网络结构。
- 修改 Cell 数量、Stage 或 Top-2 预算。
- 访问 Confirmation。
- 根据 Development 结果调整模型或训练参数。

允许：

- 加载已有 Checkpoint。
- 执行确定性的模型前向。
- 在内存中临时替换 Cell 选择。
- 执行反事实路径。
- 对已有 Connection 的修正分数施加分析性缩放。
- 计算 NLL、排序、路由变化与相应统计量。
- 将分析结果写入独立输出目录。

所有分析必须在 `torch.inference_mode()` 或等效无梯度上下文中运行。

### 2.2 一次性执行

Codex 应一次实现分析程序，并直接执行完整分析。

不安排：

- 多轮人工审批。
- 单独训练 Pilot。
- GPU Smoke 训练。
- 新模型预训练。
- 额外的模型正确性测试套件。
- 大规模路由日志采集。
- 自动超参数搜索。

只保留保证分析正确性的必要一致性检查。

---

## 3. 数据与 Checkpoint 来源

### 3.1 源码身份

R3d 实际训练源码：

`ed6b0cf375c8224ed1246203aa2a3109325274b6`

R3d 正式结果分支：

`results/r3d-c0e594bce834e136-c79e28e90a78`

R3d 发布汇总：

`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/reports/published/c0e594bce834e13629d3f5a0cfb8bb6f100c0ded994bf95d8a1a2fc0fcc912c9.json`

分析必须以真实执行的源码和配置为准，不得使用后续未经身份确认的实现替代。

### 3.2 Checkpoint

三个 Seed：

```text
151001301
151001302
151001303
```

每个 Seed 加载：

| Arm | Step |
|---|---:|
| F0 | 0 |
| L | 200 |
| M | 200 |
| F | 500 |
| P | 500 |
| J | 500 |

全部共 18 个 Checkpoint。

优先使用已保留的 Kaggle Output：

```text
/kaggle/working/r3d-output-ed6b0cf375c8
```

若该目录不存在，则使用 R3d 原有的私有 HF Checkpoint 恢复机制。

R3d 源码中使用：

```text
r3d/<run_id>/latest.json
```

作为远端最新状态指针。应从实际运行身份、现有 `hf-reference.json` 或远端正式 Manifest 解析，不得自行猜测 run_id。

注意：R3d 发布汇总中的某些 HF 引用可能指向 R3c 父实验，而不是本次 R3d 的 18 个 Checkpoint。必须按照 Manifest 身份确认来源。

### 3.3 Checkpoint 身份校验

至少检查：

- Experiment ID。
- Arm 与 Seed。
- 完成 Step。
- 源码 SHA。
- Config SHA。
- Checkpoint SHA256。
- 模型参数结构。
- 状态完整性。

如果某个 Checkpoint 不可恢复，报告明确列出缺失对象，不得使用随机初始化模型替代。

本次如果成功从独立 Kaggle Session 恢复了特定 Checkpoint，可以如实记录这些对象的跨 Session 恢复成功。但不得自动将未验证的其他对象标记为已验证。

---

## 4. 核心分析 A：反事实价值是否存在？

### 4.1 科学问题

R3d 假设，在相同状态下执行不同 Cell 对，会产生不同的最终任务收益。

但此前并未充分证明：

- 这种差异是否足够显著；
- 是否可以预测；
- 是否超出目标标签和有限样本带来的噪声；
- 是否能够转化为真实闭环推理收益。

因此，需要重新评估反事实动作价值。

### 4.2 使用完整的 15 种候选动作

当前六个 Cell，每阶段执行两个不同 Cell。

候选动作数量：

\[
|\mathcal A|=\binom62=15
\]

本次不再像 R3d 训练期间那样仅比较最多三个动作。

对于选定的分析状态，评估全部 15 个动作，以便获得相对完整的价值排序。

这是只读分析，不改变实际推理预算。

### 4.3 定义长期动作损失

对于 Stage \(t\) 的状态 \(s_t\)，候选 Cell 对 \(a\)：

\[
s_{t+1}^{(a)}=F_a(s_t)
\]

其余 Stage 使用同一个固定参数版本的 Connection 重新选择动作，最终得到：

\[
Q_t^{\mathrm{long}}(s_t,a)
=
L_{\mathrm{CE}}(s_4^{(a)})
\]

必须遵守：

1. 所有候选从同一个状态出发。
2. 当前 Stage 只修改候选动作。
3. 后续 Stage 必须基于新的状态重新路由。
4. 不改变模型参数。
5. 使用相同输入和目标 Token。
6. 目标标签仅用于结果评分，不参与动作选择。

### 4.4 定义即时动作损失

\[
Q_t^{\mathrm{short}}(s_t,a)
=
L_{\mathrm{CE}}(s_{t+1}^{(a)})
\]

其中 CE 使用模型现有的最终 Head，包括原有 Final LayerNorm。

注意：这里是将中间状态送入模型 Head 形成的诊断性中间损失，不代表原始模型已经学会在每个 Stage 直接输出。

最后一个 Stage 上，长期与即时损失按定义应相同。

### 4.5 必须输出的指标

对每个 Seed、Stage 和 Checkpoint，计算：

- 15 个候选动作的最终 CE 分布。
- 最高与最低动作收益差异。
- 候选动作收益的中位数与分位数。
- 原始 D 动作的反事实排名。
- 已学习 Connection 动作的反事实排名。
- 长期排序与即时排序的一致程度。
- Stage 之间的差异。

特别关注：

\[
\operatorname{Gap}(s_t)
=
Q(s_t,a_{\mathrm{baseline}})
-
\min_a Q(s_t,a)
\]

但必须将它命名为：

**Hindsight Candidate Opportunity Gap**

它使用真实标签挑选最佳候选，是具有标签信息优势的事后机会指标，不是可以部署的 Oracle Router，也不是实际可实现质量收益的严格保证。

若 Gap 很大但模型无法根据推理状态预测哪一个动作最好，仍然不能说明 Connection 有实际可利用的改进空间。

---

## 5. 核心分析 B：Connection 是否真正学会动作价值排序？

### 5.1 读取真实的 Connection 分数

R3d Connection 的实际形式为：

\[
q_{ij}(s,t)=z_i(s,t)+z_j(s,t)+\Delta_{\phi,ij}(s,t)
\]

其中：

- \(z_i\) 来自冻结的原始 Gate。
- \(\Delta_\phi\) 是可训练的 15 动作修正。
- 分数越高，动作越优先。

实际 R3d 修正器是一个对归一化状态与 Stage One-hot 拼接输入的线性模块。

这意味着它的状态依赖修正基本为仿射形式，Stage One-hot 主要提供按阶段的偏置；它不具备独立的每 Stage 状态投影矩阵。

本次应分析该表示形式是否足以预测实际动作收益，但不修改结构。

### 5.2 评分质量

比较：

\[
q_a-q_b
\]

与：

\[
Q_b-Q_a
\]

是否具有一致的符号。

输出：

- Pairwise ranking accuracy。
- Spearman rank correlation。
- Kendall rank correlation。
- Top-1 动作的真实价值排名。
- Top-3 动作是否包含低损失候选。
- 实际选中动作相对事后最佳动作的 regret。
- 原始 Gate 分数与长期价值的排序关联。
- 修正后 Connection 分数与长期价值的排序关联。

当两个候选动作的 CE 差异极小时，排名可能缺乏实际意义。

因此，同时报告：

- 全部候选对的排序。
- 只包含具有明显价值间隔的候选对的排序。
- 排名准确率随候选价值间隔变化的关系。

这里的价值间隔分组应使用预先冻结的分位数，不得根据最终正负结果寻找有利阈值。

### 5.3 区分训练统计与独立分析统计

R3d 原报告中的候选排序准确率来自 Train 中参数更新前的已采样候选。

本次应分别报告：

- 历史 Train 排序统计。
- 冻结 Checkpoint 在 Calibration 上的全候选排序。
- 冻结 Checkpoint 在 Development 上的全候选排序。

不得将三者混为一谈。

不得使用 Development 排序结果训练一个新的价值评估器。

### 5.4 关键判断

如果长期价值差异明显，但 Connection 排序接近随机：

说明至少当前 Connection 没有有效提取长期决策信息。

如果原始 Gate 排序好于修正后的 Connection：

说明新增价值学习可能破坏了已有的路由信息。

如果 Connection 排序优于原始 Gate，但最终 NLL 变差：

则需要进一步分析闭环状态分布变化与策略更新幅度。

---

## 6. 核心分析 C：Connection 是否过度修改原始路由？

### 6.1 问题背景

R3d L/J 训练期间的动作变化比例约为 70%。

但训练时的参数更新范数限制，并不直接等同于策略动作变化幅度限制。

因此，需要在固定 Checkpoint 上单独分析：

**如果减小 Connection 修正强度，模型质量是否能够恢复？**

### 6.2 路由修正插值

保持所有模型参数不变。

定义：

\[
q_{ij}^{(\lambda)}
=
z_i+z_j+\lambda\Delta_{\phi,ij}
\]

固定分析系数：

```text
0.00
0.10
0.25
0.50
0.75
1.00
```

其中：

- \(\lambda=0\)：使用原始 D 路由，采用其稳定 Top-2 tie-breaking。
- \(\lambda=1\)：使用完整的已学习 Connection。
- 中间值：仅缩小已学习修正的强度。

这些系数只用于分析，不进行参数更新，也不根据结果选择新的部署方案。

### 6.3 比较对象

主要分析：

- L：冻结 Cell 下的 Connection。
- M：即时信用 Connection。
- P：Cell 已进一步训练，Connection 冻结。
- J：Cell 与 Connection 已联合训练。

每个 Checkpoint 保持自身的全部 Cell 与公共模型参数不变，只改变路由修正强度。

F0 和 F 作为原始路由参考。

### 6.4 闭环评估

对于每一个 \(\lambda\)，从输入重新执行完整模型。

不得使用 \(\lambda=1\) 的中间隐藏状态直接回放其他 \(\lambda\) 的路由，因为不同策略可能访问不同的状态。

输出：

- 最终 Development NLL。
- 相对于同一模型参数下原始路由的 NLL 差异。
- 各 Stage 动作变化率。
- 原始 Gate 的 Top-2 margin。
- 修正分数的幅度分布。
- 动作变化率与最终 NLL 的关系。

需要明确区分：

**同一模型参数下改变 Connection**，可以隔离路由决策的直接效果。

**不同 Arm 的最终 NLL 比较**，则同时受到 Cell 训练轨迹影响，不能直接归因于 Connection。

### 6.5 关键判断

如果很小的 \(\lambda\) 就能改善 NLL，而较大的 \(\lambda\) 明显恶化：

说明 R3d 可能存在策略更新过度的问题。

如果所有非零 \(\lambda\) 都使结果恶化：

说明当前修正方向整体缺乏证据支持。

如果较大的 \(\lambda\) 提升动作排序，却使闭环 NLL 恶化：

说明仅在固定状态上学到正确偏好，不足以保证新策略访问的状态分布也能够得到改善。

---

## 7. 核心分析 D：联合训练失败的归因

### 7.1 为什么需要单独分析？

R3d 中：

\[
NLL(P)=4.514825
\]

\[
NLL(J)=4.529020
\]

虽然 J 比 P 更差，但这并不能直接证明：

「持续更新 Connection 本身造成全部性能损失。」

因为 P 和 J 后续训练的 Cell 参数也已经发生不同的变化。

需要进行固定参数下的路由替换分析。

### 7.2 对 J Checkpoint

固定 J 的所有 Cell 与公共参数。

比较：

- J 原始冻结 Gate 路由。
- J 已学习 Connection 路由。
- J 不同修正强度的路由。

计算：

\[
\Delta_J^{\mathrm{route}}
=
NLL(J,\pi_{\mathrm{learned}})
-
NLL(J,\pi_D)
\]

其中两次推理使用同一个 J 模型参数状态。

### 7.3 对 P Checkpoint

采用完全相同的方法：

\[
\Delta_P^{\mathrm{route}}
=
NLL(P,\pi_{\mathrm{learned}})
-
NLL(P,\pi_D)
\]

比较 P 和 J 的结果，判断不同 Cell 训练轨迹下，对 Connection 的依赖是否发生变化。

### 7.4 结合 F Checkpoint

F 是冻结 Router 后继续训练 Cell 的基线。

应比较：

- F 使用原始 Router。
- P 使用原始 Router。
- J 使用原始 Router。

三者的 Cell 参数来自不同的训练路径，因此这一对照主要用于观察训练路径造成的最终模型能力差异，而不是证明其中某一个具体参数更新的因果效果。

### 7.5 结论边界

此次只读分析能够隔离给定 Checkpoint 上的路由效果。

但无法严格还原历史训练中每一次 Connection 更新对未来 Cell 参数轨迹的因果影响。

如果现有 Checkpoint 数量不足以支持某个历史因果判断，应明确写为 `NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS`。

不得假定重新评估最终 Checkpoint 就能够证明完整训练过程的因果关系。

---

## 8. 采样与计算预算

### 8.1 反事实采样

为避免不必要的大规模前向，本轮采用固定、有限的状态样本。

建议：

每个 Seed、每个主要 Checkpoint：

- Calibration：选择 32 个不同 Story Window。
- Development：选择 16 个不同 Story Window。
- 每个 Story Window、每个 Stage 选择一个 Token 状态。
- 每个状态评价全部 15 种候选 Cell 对。

因此：

\[
48\times4=192
\]

个状态 / Seed / Checkpoint。

重点执行完整反事实的 Checkpoint：

```text
F0
L
M
P
J
```

F 主要参与完整闭环路由比较。

所有 Story 和 Token 选择使用独立固定随机种子，且不依赖目标标签或模型评分。

需要明确记录此方案的实际反事实计算量，不允许把训练期或分析期的额外计算描述为推理开销。

### 8.2 计算控制

- 优先批量化相同 Stage 的候选动作。
- 缓存公共输入与 Context 计算。
- 保持每个候选后续路径重新路由。
- 不保存全部 Token 的完整隐藏状态。
- 只保存候选评分摘要与必要的可复查索引。
- 不做无关的硬件 profiling。
- 不重复执行已经具有可信哈希的历史模型测试。

分析样本预算应在启动前冻结。

如运行环境资源不足，应报告未完成状态，不得临时仅保留有利于某种假设的样本。

---

## 9. 正确性要求

只执行与科学结论直接相关的最小检查。

### 9.1 前向一致性

在少量固定样本上验证：

使用当前策略选中的 Cell 对调用独立 `advance()` 路径时，其最终状态及 Head 输出应与完整模型执行路径一致，允许预先声明的浮点误差范围。

特别注意：

- Final LayerNorm。
- 选中 Cell 的混合权重。
- Stage 顺序。
- 状态更新。
- 后续重新路由。
- 相同动作下的 tie-breaking。

如果这一检查失败，反事实价值不可信，必须停止相应分析。

### 9.2 参数不可变

检查开始和结束时的模型参数 Fingerprint 应相同。

包括 Cell、Gate、Connection、Embedding、Head 和其他公共参数。

### 9.3 数据边界

- 模型参数不更新。
- Confirmation 不读取。
- 目标标签不参与策略打分。
- 训练历史统计与只读评估统计分开。
- Development 结果仅用于探索性机制判断。

### 9.4 不引入多余测试

不新增独立训练测试。

不对历史 R3c、R3d 重复跑完整测试套件。

不运行 Smoke、Pilot 或超参数搜索。

---

## 10. 统计与报告

### 10.1 统计单位

使用 Story 作为主要重采样单位。

不同 Stage、Token 和候选动作属于同一 Story 内的相关观测，不能视为独立训练重复。

分别报告三个 Seed，不得将三个 Seed 与所有 Story 混合后宣称拥有大量独立训练实验。

### 10.2 主要结果

报告至少包含：

| 指标 | F0 | L | M | P | J |
|---|---|---|---|---|---|
| 原始闭环 NLL | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 候选长期价值分布 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Connection 排序质量 | 基线 | ✓ | ✓ | ✓ | ✓ |
| Hindsight Opportunity Gap | ✓ | ✓ | ✓ | ✓ | ✓ |
| 不同修正强度的闭环 NLL | — | ✓ | ✓ | ✓ | ✓ |
| 路由变化率 | 基线 | ✓ | ✓ | ✓ | ✓ |
| Stage 级差异 | ✓ | ✓ | ✓ | ✓ | ✓ |

F 额外提供固定 Router 训练后的闭环 NLL 与参数控制对照。

### 10.3 输出可视化

仅生成三个必要图表：

1. **长期动作价值与 Connection 排序能力**：展示各 Stage 的分布、排序准确率与价值间隔关系。
2. **路由修正强度与闭环 NLL**：展示 \(\lambda\) 和质量、动作变化比例的关系。
3. **固定 Checkpoint 的路由替换效应**：比较 L、M、P、J 在其自身参数状态下使用原始路由与已学习路由的差异。

不要生成大量逐 Token、逐 Cell 的装饰性图表。

---

## 11. 预定义结论矩阵

报告必须将观测结果归入以下情况之一或多个，并说明证据限制。

| 情况 | 观测 | 更可能指向的问题 |
|---|---|---|
| A | 候选损失几乎没有实质差异 | 当前 Cell 缺少明显可利用的条件互补性 |
| B | 存在价值差异，但 Connection 排序接近随机 | 表示能力、监督质量或偏好优化问题 |
| C | Connection 排序较好，但闭环质量恶化 | 策略分布变化、状态偏移或局部目标与全局目标不一致 |
| D | 小幅修正有益，大幅修正有害 | 策略更新幅度控制问题 |
| E | 冻结路由下 Cell 改善明显，改变路由后难以保持 | Cell 与既有路由形成强烈协同适应 |
| F | 部分 Seed 正向、部分 Seed 负向 | 机制不稳定或证据不足 |

不允许仅根据一个相关性指标直接确认机制因果关系。

### 11.1 对下一阶段的意义

分析报告最后只给出以下三类建议之一。

**ROUTING_LEARNING_REVISION**

当存在可利用动作差异，但 Connection 无法正确预测时，下一阶段重点研究价值监督、表示能力或策略优化。

**POLICY_STABILITY_REVISION**

当价值排序尚可，但策略偏离导致闭环性能恶化时，下一阶段重点研究约束式路由更新和渐进适应。

**CELL_ORGANIZATION_REVIEW**

当缺少稳定且可预测的动作价值差异时，暂不继续增加复杂信用学习机制，优先重新研究 Cell 的组织和组合方式。

如果证据不足以判断，则必须输出：

`INCONCLUSIVE`

不得强行选择某个发展方向。

---

## 12. 输出目录

新建独立分析目录，例如：

```text
research/projects/cellular-language-models/
  experiments/
    clm-001-qfirst-integrated/
      r3d/
        audit/
          SPEC.md
          README.md
          config/
            frozen.json
          implementation/
            analysis.py
            runner.py
          reports/
            AUDIT_SUMMARY.json
            COUNTERFACTUAL_VALUES.json
            POLICY_RANKING.json
            POLICY_INTERPOLATION.json
            CHECKPOINT_ATTRIBUTION.json
            SCIENTIFIC-FINDINGS.md
```

实际可按照已有包组织形式调整。

不得覆盖 R3d 原有报告。

全部私有样本、模型状态和中间信息应保存在私有输出空间。

公开 GitHub 只允许发布不含私有训练数据的汇总指标及科学结论。

---

## 13. 一次性执行要求

实现一个统一入口：

```bash
python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/audit/run.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/audit/config/frozen.json \
  --run-all
```

上述脚本由本次工作创建。

入口执行以下任务：

1. 恢复 R3d Checkpoint。
2. 验证模型与分析身份。
3. 自动完成所有 Seed 的只读反事实分析。
4. 计算 Connection 排序质量。
5. 运行预定义策略修正强度的闭环评估。
6. 计算固定参数下的路由替换效应。
7. 自动生成图表、JSON 和科学报告。
8. 保存私有原始分析记录。
9. 如仓库写权限可用，提交允许公开的独立分析结果。

不设置中途审批，不自动重新训练任何模型。

如已有私有远端存储只提供读权限，也应完成分析，不应因为没有模型写权限而阻塞。

---

## 14. 最终交付

必须返回：

**A. 运行身份**

- 源码 Commit。
- 原始 R3d Checkpoint 身份。
- 配置哈希。
- 运行环境。
- 实际使用的 Seed 和数据角色。

**B. 数据结果**

- 反事实长期价值分布。
- Connection 排序能力。
- 原始 Gate 与学习 Connection 的排序比较。
- 路由修正强度与 NLL 曲线。
- P/J 的固定参数路由替换结果。
- 关键的 Stage 差异。

**C. 科学结论**

逐项回答 Q1–Q4。

必须明确区分：

- 实际观测。
- 数学推断。
- 机制假设。
- 尚无法识别的因果关系。

**D. 下一阶段研究建议**

只选择最受当前证据支持的一个主要研究方向。

如证据不足，明确建议暂不进入 R3e。

---

# 15. 给 Codex 的最终指令

请基于 `ArcheLabs/mini-cells` 已完成的 R3d 实验，实施 `CLM-001-R3d-AUDIT-001`。

这是一次严格只读的 Checkpoint 机制分析，不是新训练实验。

目标是解释 R3d 中：

- Connection 长期信用学习未能改善 NLL；
- 训练期候选排序准确率接近随机；
- Connection 大幅改变原始路由；
- 冻结 Router 的 Cell 训练优于联合训练；

这些现象之间的关系。

请直接复用现有 R3d 的模型、数据、Checkpoint 与私有存储恢复代码。

完成全部分析程序后，在现有相同环境中一次性运行。

不得进行模型参数更新、超参数搜索或新增训练。

重点完成三个分析：

1. 全 15 个 Cell 对的真实长期反事实价值与 Connection 评分一致性。
2. 缩放已学习 Connection 修正强度后的真实闭环 NLL 变化。
3. 在固定 Cell 参数下，替换原始与已学习路由的直接质量影响。

保留模型前向一致性、数据隔离和 Checkpoint 完整性所需的最小检查。

不要安排额外 Smoke、Pilot、重复环境验证或人工中途审批。

分析完成后自动生成完整报告，并明确判断：

- 是否存在可利用的长期动作价值；
- Connection 是否能够预测这种价值；
- 路由改变是否损害闭环计算；
- 下一阶段应优先修改训练方法、策略稳定性，还是 Cell 组织形式。

**不要在没有证据的情况下将 R3d 的失败归因于任何单一机制。**

本次研究的最终目标，是为下一阶段 CLM 架构与联合训练方法的选择提供可靠依据，而不是取得新的训练成绩。

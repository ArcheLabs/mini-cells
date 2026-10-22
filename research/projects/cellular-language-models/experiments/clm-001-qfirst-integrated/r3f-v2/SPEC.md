# MiniCells R3f-v2 — Local Trust-Region Connection Update

**真实 Connection 候选重建、局部更新准入与 Cell 适应实验**

**版本：** v1.0  
**实验 ID：** `CLM-001-R3f-v2-Local-Connection-001`  
**项目：** MiniCells / Cellular Language Models  
**仓库：** `ArcheLabs/mini-cells`  
**实验性质：** 探索性机制验证  
**运行环境：** Kaggle T4  
**执行方式：** 一次实现、一次启动、自动完成本轮实验  
**Confirmation：** `SEALED`

---

# 1. 研究目标

## 1.1 背景

R3e 和 R3f 暴露了同一个关键问题：

**Connection 能够学习新的参数，却无法产生满足 Trust Region 的实际路由更新。**

R3e 的 36 次 Connection 提案全部被拒绝。

R3f 进一步修复了零修正一致性、路由 Margin、随机流匹配和事务保存。

但在 Seed `151001301` 的 Step 100：

| 指标 | A-C | B-C |
|---|---:|---:|
| 最小回溯比例 | 1/1024 | 1/1024 |
| KL | 约 1.76e-9 | 约 1.95e-9 |
| 硬动作变化率 | 14.31% | 13.53% |
| 完整路径变化率 | 46.48% | 49.80% |
| Trust Region | REJECTED | REJECTED |
| Cell 适应更新 | 0 | 0 |

当前准入门槛：

\[
D_{\mathrm{KL}}\le0.01
\]

\[
D_{\mathrm{action}}\le0.10
\]

\[
D_{\mathrm{path}}\le0.20
\]

KL 已经足够小，但硬动作和完整路径仍然变化过大。

这与 Cell 网络的多步组合有关：即使每一步的动作变化有限，整条路径中至少一处发生变化的概率仍可能较高。

因此，继续缩小更新不能简单依赖 KL。

## 1.2 本次要解决的问题

**Q1：现有 Connection 优化器生成的真实候选，是否存在满足三个 Trust Region 门槛的非空局部更新？**

**Q2：如果存在，更新幅度需要缩小到什么范围？**

**Q3：通过准入的更新，能否在真实训练期间改变 Cell 计算路径？**

**Q4：这样的 Connection 更新能否进入 K 步 Cell 适应训练，并完成正确的质量评价与事务提交？**

## 1.3 科学假设

### H1：Local Admission Feasibility

相对于 R3f 原有的 11 个回溯尺度，继续缩小更新幅度，能够找到满足 Trust Region 且实际改变计算路径的非空 Connection 更新。

### H2：Real Co-Adaptation Execution

通过准入的真实 Connection 更新，能够进入 Cell 适应流程，在规定预算内完成旧、新组织的配对训练与质量评价。

### H3：Local Improvement

在规定的 K 步 Cell 适应之后，新 Connection 在 Calibration 上表现优于相同起点、相同训练预算的旧 Connection。

H1、H2 属于机制可行性问题。

H3 属于模型质量问题。

不得把机制成功直接解释为科学质量收益。

---

# 2. 实验边界

本轮只使用两个 Arm：

- `A-C`：四阶段 Top-2 Cell 选择。
- `B-C`：八步串行 Top-1 Cell 图。

只使用一个真实 Seed：

`151001301`

只从 Step 100 官方状态开始。

本轮最多生成：

- A-C 的一个真实 Connection 提案。
- B-C 的一个真实 Connection 提案。

每个提案执行一次规定的策略优化，不因候选失败重复搜索新的随机方向。

## 2.1 不修改的内容

- Cell 总数：6。
- A 组：四阶段，每阶段两个 Cell。
- B 组：八步串行，每步一个 Cell。
- Cell 内部结构。
- 原始 Gate。
- Connection 线性修正结构。
- 反事实长期 Q 的定义。
- Connection 优化目标。
- Cell Optimizer 配置。
- Trust Region 三个原始阈值。
- K 步 Cell 适应协议。
- Train / Calibration / Development / Confirmation 的边界。

## 2.2 明确禁止

不得：

- 覆盖 R3e 或 R3f 历史结果。
- 将旧随机方向扫描当作真实候选。
- 声称重新生成的候选就是丢失的历史候选。
- 根据 Development 选择学习率或回溯范围。
- 放宽动作变化与路径变化门槛。
- 为了通过门禁无限减小更新。
- 将没有实际路径变化的更新算作成功。
- 自动启动完整三 Seed、六 Arm 的 500 Step 训练。
- 解封 Confirmation。

---

# 3. 源码与历史 Checkpoint 身份

## 3.1 历史运行来源

R3f 历史试运行：

```text
experiment:
r3f-connection-admission

source_sha:
98cf8f6c214f33af72d1badf28d80a0c202dc934

config_sha256:
c8d896bd7780a3aa104afca151d1e86937438eb1f4e9842ec9d7f8bbfb464e25

run_id:
0d1b8b91af485b9a923ecdf01f94885fce7d6a15745eccff97a8738c7939b1c5
```

此前已经完成跨 Session Checkpoint 恢复和身份核对。

本轮继续使用同一个来源的 Step 100 官方状态，但建立新的实验身份。

## 3.2 依赖的已修复代码

以包含以下提交的 R3f 代码为起点：

```text
eeb7985 — GitHub 结果回传修复
9c86edc — 候选存档与独立执行证据
39a1aa3 — Kaggle 只读恢复 Notebook
```

必须使用这些修复之后的源码作为开发基线。

不应重新引入旧版 `run_git(timeout=...)` 适配错误。

## 3.3 Checkpoint 导入

从已核实的私有 HF 历史 Manifest 恢复：

- Seed 151001301。
- A-C Step 100 官方状态。
- B-C Step 100 官方状态。

恢复对象必须包括：

- Cell 参数。
- Connection 参数。
- Cell Optimizer 状态。
- Data RNG。
- Route RNG。
- 数据游标。
- 已完成 Cell Step。
- Route Margin。
- 数值容差。
- 原始配置和身份信息。

严格确认这些状态来自 Step 100 已拒绝历史提案之后的正式模型，而不是某个尚未提交的影子适应分支。

如果只能恢复候选事务中的官方快照，也可以使用，但必须验证该快照与正式模型状态相符。

## 3.4 新运行身份

新运行以：

```text
new_source_sha
new_config_sha256
parent_source_sha
parent_run_id
parent_checkpoint_sha256
execution_id
```

建立独立身份。

历史 Checkpoint 仅作为明确记录的输入，不允许将新的实验身份伪装成历史运行的原生 Checkpoint 身份。

如果当前 Store 的身份检查不支持跨实验导入，应实现一次显式、可验证的 Parent Import，而不是绕过身份验证。

---

# 4. 实验第一阶段：恢复 Step 100 官方状态

## 4.1 只读恢复

分别恢复 A-C、B-C。

验证：

- Step = 100。
- Checkpoint SHA-256。
- 全部 Cell 权重。
- Connection 权重。
- Optimizer 状态。
- RNG 状态。
- 数据游标。
- Route Margin。
- 原始运行身份。

不进行任何训练更新。

## 4.2 计算基线指标

在固定参考状态上重新计算：

\[
D_{\mathrm{KL}}(\pi_0,\pi_0)=0
\]

\[
D_{\mathrm{action}}(\pi_0,\pi_0)=0
\]

\[
D_{\mathrm{path}}(\pi_0,\pi_0)=0
\]

并验证原始模型的确定性路径能够复现。

若同一个模型与自身比较仍有路径差异，停止实验。

输出：

`PARENT_RESTORE_VERIFICATION.json`

---

# 5. 第二阶段：生成新的真实 Connection 提案

这是本轮唯一允许的 Connection 参数优化阶段。

不是只读操作。

## 5.1 复用原始优化方法

使用 R3f 当前的 `policy_proposal.py`。

保留：

- Train 中的真实状态。
- 全候选长期反事实 CE。
- 后续 Stage 由当前策略重新路由。
- 中心化期望成本目标。
- AdamW。
- Gradient Clipping。
- 原有学习率和 Weight Decay。
- 8 次策略优化更新。

A-C：

\[
|\mathcal A|=15
\]

B-C：

\[
|\mathcal A|=6
\]

每个提案使用原有的 12 个反事实状态采样预算。

不增加训练样本或候选动作数量。

## 5.2 重要：先保存候选，再执行 Trust Region

优化结束立即保存：

\[
\phi_0
\]

和：

\[
\phi_{\mathrm{proposal}}
\]

同时保存：

- 官方模型快照。
- 原始 Connection 参数。
- 未缩放的候选 Connection 参数。
- 参数差分。
- 8 次优化损失。
- 优化器配置。
- 状态采样身份。
- RNG 状态。
- Counterfactual 计算成本。
- Gate Margin。
- Route Margin。
- Config SHA。
- Source SHA。
- Parent Checkpoint SHA。

**这一保存操作必须发生在第一次回溯尺度评价之前。**

任何 Trust Region 失败都不得删除完整候选。

## 5.3 候选保存格式

建议使用现有 Store，将候选保存至：

```text
private-checkpoints/
  proposals/
    seed-151001301/
      A-C/
        step-0100/
          official
          proposal
          receipt
      B-C/
        step-0100/
          official
          proposal
          receipt
```

实际目录可以遵守现有 Store 规范。

必须具备：

- 完整参数。
- 完整身份。
- 内容哈希。
- 即时回读验证。
- 不可变事务记录。

本次不得将候选参数上传至公开 GitHub。

---

# 6. 第三阶段：局部更新幅度分析

## 6.1 数学定义

设：

\[
\Delta\phi=
\phi_{\mathrm{proposal}}-\phi_0
\]

构造：

\[
\boxed{
\phi_\alpha=\phi_0+\alpha\Delta\phi
}
\]

其中：

\[
0\le\alpha\le1
\]

每次尺度评价均使用同一个完整候选。

不得在不同尺度重新优化 Connection。

## 6.2 固定尺度集合

原始 R3f 使用：

\[
\alpha=2^{-k}
\]

其中：

\[
k=0,\ldots,10
\]

本轮扩展至：

\[
\boxed{
k=0,\ldots,20
}
\]

另外增加：

\[
\alpha=0
\]

作为完全无变化对照。

所有尺度在运行前冻结。

不得自动继续扫描到 \(2^{-30}\) 或更小。

## 6.3 计算指标

对每一个尺度分别计算：

| 指标 | 目的 |
|---|---|
| Parameter Delta Norm | 真实参数修改幅度 |
| Centered Logit Delta RMS | 去除公共分数平移后的修正 |
| Maximum Absolute Logit Delta | 最大评分扰动 |
| KL | 概率策略变化 |
| Hard Action Change | 同状态硬路由变化 |
| Full Path Change | 完整执行路径变化 |
| Changed Path Count | 非空更新验证 |
| Baseline Gate Margin | 动作边界稳定性 |
| Interpolation Error | 参数插值正确性 |

### 基线路由不变性

严格检查：

\[
\alpha=0
\]

时：

```text
KL = 0
action_changed_count = 0
path_changed_count = 0
```

## 6.4 固定状态和闭环状态必须分开

硬动作变化：

在同一组原始隐藏状态上比较两个 Connection。

完整路径变化：

从相同原始输入重新运行两个模型的完整计算。

不能将局部动作变化直接当作完整路径变化。

## 6.5 数据分工

使用原有冻结数据划分。

- Train：生成 Connection 提案。
- Admission：计算更新尺度的正式准入指标。
- Calibration：评价候选的即时质量和 Cell 适应后质量。
- Development：本轮不读取。
- Confirmation：保持封存。

Admission 与 Calibration 必须严格按已有范围分离。

如果现有 `reference()` 默认从 Calibration 读取 Trust Region 样本，应为本轮显式切换至 Admission，并记录数据范围。此变更属于新实验协议，必须拥有新配置哈希。

不能将 Admission 阈值评价与 Calibration 质量选择混用。

---

# 7. 第四阶段：Trust Region 准入

## 7.1 保持原有门槛

| 条件 | 要求 |
|---|---:|
| KL | ≤0.01 |
| Hard Action Change | ≤10% |
| Full Path Change | ≤20% |

所有阈值固定。

不允许根据运行结果调整。

## 7.2 更新必须具有实际效果

为了避免 \(2^{-20}\) 附近出现数值意义上的假更新，候选除了通过三个门槛，还必须满足：

1. Centered Logit Delta 超过冻结的数值容差。
2. 至少一个 Admission 输入的确定性完整路径发生变化。
3. 硬动作变化与完整路径变化的统计记录完整。
4. 新、旧模型存在可验证的实际动作差异，而不仅是参数差异。

定义：

\[
N_{\mathrm{changed\ paths}}>0
\]

本轮额外记录：

\[
R_{\mathrm{path}}\ge1\%
\]

是否成立，作为更新具有实质覆盖率的描述性指标。

但不将 1% 设置为新的正式 Trust Region 下限，以免在样本有限时人为阻止有效的小幅探索。

## 7.3 接受尺度的选择

按既定顺序：

\[
1,\frac12,\frac14,\ldots,2^{-20}
\]

选择第一个同时满足：

- KL。
- Hard Action Change。
- Full Path Change。
- 非空有效更新。

的尺度。

如果多个尺度符合条件，不根据 Calibration 质量选择其中表现最好的一个。

后续 Calibration 只决定是否最终接受该组织。

## 7.4 无法通过的情况

明确区分：

`NO_TRUST_REGION_SCALE`

所有尺度均违反至少一个门槛。

`ONLY_NUMERIC_NOOP_SCALES`

满足门槛的尺度没有有效执行变化。

`NONFINITE_POLICY_PROPOSAL`

参数、Logit 或梯度出现非有限值。

`INVALID_INTERPOLATION`

实际评分变化与预期线性插值不一致。

`SOURCE_OR_CHECKPOINT_MISMATCH`

恢复的输入不符合身份要求。

这些结果均应停止当前 Arm 后续的 Cell 适应，但保留候选及全部诊断。

---

# 8. 第五阶段：真实 Cell 适应

只有通过第七章准入的候选才进入本阶段。

## 8.1 配对训练

从同一个官方 Cell 状态：

\[
\theta_0
\]

建立两条分支。

旧 Connection：

\[
\theta_K^{\mathrm{old}}
=
\operatorname{Adapt}_K(\theta_0,\pi_0)
\]

新 Connection：

\[
\theta_K^{\mathrm{new}}
=
\operatorname{Adapt}_K(\theta_0,\pi_\alpha)
\]

固定：

\[
K=10
\]

## 8.2 配对控制

必须保持：

- Cell 初始参数相同。
- Cell AdamW 初始状态相同。
- 输入批次相同。
- Train 数据顺序相同。
- Gumbel 随机数相同。
- Epsilon 探索随机流相同。
- 完整执行次数相同。
- Cell 更新次数相同。

Connection 不同所造成的动作差异属于实验效应。

随机数流消费不一致，则停止并报告：

`ADAPTATION_ROUTE_STREAMS_DIFFER`

不得把随机数流差异当作 Connection 的学习收益。

## 8.3 必须同时验证实际训练路径

本轮不仅要验证 Admission 的硬路由变化，还必须观察真实 Train 阶段的 Cell 执行。

在相同输入和配对随机数下记录：

- Old/New 训练动作差异。
- Cell 调用分布差异。
- 各 Cell 梯度覆盖。
- 实际完成的 Cell 更新。
- 完整 Token 数。
- 训练时 Connection 分布 KL。
- 最终推理时的确定性路由差异。

如果 Admission 发生确定性路径变化，但训练时两种 Connection 的随机策略几乎无差异，也应如实报告。

特别注意：当前 R3f 的训练采样使用原始 Logit 的 Gumbel-Max 加 epsilon，而确定性评价使用带 Margin 的硬路由。

这两种策略语义并不完全相同。

因此，必须分别判断：

- 硬推理路径是否改变。
- 随机训练分布是否改变。
- Cell 训练轨迹是否因此产生差异。

不能仅凭参数不同宣布完成有效联合训练。

## 8.4 质量评价

完成 K=10 步后，在 Calibration 上计算：

\[
L_{\mathrm{old}}
\]

和：

\[
L_{\mathrm{new}}
\]

定义：

\[
\Delta_K=L_{\mathrm{new}}-L_{\mathrm{old}}
\]

负值表示新 Connection 适应后质量更好。

继续沿用历史门槛：

\[
L_{\mathrm{old}}-L_{\mathrm{new}}\ge0.001
\]

满足则：

`FINAL_QUALITY_ACCEPTED`

否则：

`QUALITY_REJECTED_AFTER_ADAPTATION`

即使质量拒绝，K 步 Cell 适应已经真实完成，也必须将 H2 的执行结果单独记录。

---

# 9. 第六阶段：事务提交与恢复

继续沿用现有事务状态：

```text
PROPOSED
→ ADAPTING(k)
→ EVALUATED
→ COMMITTED
```

增加完整候选归档的身份关联。

每个事务应保存：

- Official state。
- Candidate Connection。
- Chosen scale。
- Trust Region 全尺度报告。
- Old/New Cell 适应状态。
- 两条分支的 Optimizer。
- 两条分支的 RNG。
- Data cursor。
- Calibration 分数。
- 最终接受结果。
- 已计入计算成本。
- Source/Config/Parent 身份。

必须确保：

1. 中断后不会重新优化已经归档的候选。
2. 中断后不会重复消耗已完成的 K 步适应。
3. 结果拒绝不会删除被拒绝的候选证据。
4. 新模型提交后，其 Connection 和 Cell 参数属于同一个被选中的状态。
5. 恢复后重新计算的指标与保存记录一致。

---

# 10. 实验预算

本轮只执行真实的机制验证，不执行完整模型质量矩阵。

| 工作 | 预算 |
|---|---|
| Seed | 1 |
| Arm | A-C、B-C |
| 起始主 Step | 100 |
| Connection 提案 | 每 Arm 1 次 |
| Connection 优化更新 | 每 Arm 8 次 |
| 反事实状态 | 每 Arm 12 个 |
| A-C 候选动作 | 15 |
| B-C 候选动作 | 6 |
| 回溯尺度 | 22 个（含 0） |
| Cell 适应 | 通过准入后 K=10 |
| Development 评价 | 0 |
| Confirmation 访问 | 0 |

必须分别记录：

- Connection 反事实前向成本。
- Connection 优化成本。
- Trust Region 扫描成本。
- Old/New 适应成本。
- Calibration 评价成本。
- 被丢弃分支成本。
- 总 GPU 时间。

不能将仅有的 10 次有效 Cell 更新当作完整实验成本。

## 10.1 停止规则

如果 A-C、B-C 全部未能产生有效准入更新：

状态：

`LOCAL_CONNECTION_ADMISSION_NOT_ESTABLISHED`

停止，不运行完整模型矩阵。

如果至少一个 Arm 通过准入，并完成真实 K 步适应与质量评价：

状态：

`LOCAL_COADAPTATION_MECHANISM_VERIFIED`

可以认为局部联合训练机制得到了工程验证。

但仍不代表 H3 的质量改善成立。

本轮运行到此即结束，不自动继续 500 Step ×3 Seed 的完整训练。

---

# 11. 结果输出

新增独立目录：

```text
research/projects/
  cellular-language-models/
    experiments/
      clm-001-qfirst-integrated/
        r3f-v2/
          SPEC.md
          README.md
          config/
            frozen.json
          implementation/
            minicells_r3f_v2/
              parent_import.py
              proposal.py
              scale_scan.py
              admission.py
              adaptation.py
              evaluation.py
              storage.py
              runner.py
          notebooks/
            kaggle-development.ipynb
          reports/
            PARENT_RESTORE_VERIFICATION.json
            PROPOSAL_MANIFEST.json
            LOCAL_SCALE_SCAN.json
            TRUST_REGION_RESULTS.json
            ADAPTATION_RESULTS.json
            TRAINING_ROUTE_DIFFERENCE.json
            COMPUTE_COST.json
            SCIENTIFIC-FINDINGS.md
            RUN_MANIFEST.json
```

优先复用 R3f 已有模块，而不是复制整套模型与训练逻辑。

## 11.1 保存真实候选

必须永久保存：

- 完整候选参数。
- 实际 Proposal Optimizer 更新记录。
- 完整尺度扫描。
- Checkpoint 身份。
- 保存 Receipt。

这样即使本轮仍无法通过 Trust Region，后续也可以在不重新训练 Connection 的情况下，对**同一个真实候选**进行机制研究。

这是本轮最重要的成果保存要求。

## 11.2 GitHub

公开发布：

- 冻结配置。
- 运行身份。
- 聚合诊断。
- 接受率。
- Cell 适应执行记录。
- 计算成本。
- 科学结论。

禁止公开：

- 私有 Checkpoint。
- 训练 Token 数据。
- 可恢复的完整模型权重。
- 私有 HF 访问凭据。

使用已修复的 GitHub 发布器。

结果推送后，必须验证远端 Branch Commit SHA 与本地提交一致。

如果推送失败，本地结果继续保留，并支持独立重新发布，不重新训练。

---

# 12. 自动执行要求

Codex 创建统一入口：

```bash
python scripts/run_r3f_v2.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-v2/config/frozen.json \
  --run-all
```

这个脚本在本次实施中创建，目前不是已经存在的命令。

Notebook 只负责准备环境、固定源码及调用 Runner。

Runner 自动执行：

1. 恢复并验证两个历史 Step 100 官方状态。
2. 新建 R3f-v2 独立运行身份。
3. 分别生成一个 A-C、B-C 真实 Connection 候选。
4. 在进行任何尺度评价前保存完整候选。
5. 执行全部 22 个固定尺度的 Trust Region 扫描。
6. 按预注册规则选择第一个有效准入尺度。
7. 对准入通过的候选自动执行 K=10 配对 Cell 适应。
8. 完成 Calibration 质量评价。
9. 正确提交或拒绝候选。
10. 自动保存私有状态和公开汇总。
11. 完成 GitHub 回传或记录可恢复的回传失败。
12. 输出最终科学状态。

不得因为某个 Arm 失败而跳过另一个独立 Arm。

不得增加与本轮机制问题无关的 Pilot、Smoke 或完整回归测试。

---

# 13. 科学结论标准

## H1：Local Admission Feasibility

`SUPPORTED_EXPLORATORY`：

至少一个真实候选存在满足全部 Trust Region 门槛且产生实际确定性路径变化的尺度。

否则：

`NOT_SUPPORTED`

如果恢复身份或前向一致性失败：

`INVALID_RUN`

## H2：Real Co-Adaptation Execution

`SUPPORTED_EXPLORATORY`：

至少一个真实、通过准入的候选完成全部 K=10 配对 Cell 更新与 Calibration 评价，状态和成本记录可验证。

否则：

`NOT_SUPPORTED`

H2 不要求最终质量改善。

## H3：Local Improvement

`SUPPORTED_EXPLORATORY`：

新 Connection 在相同起点和适应预算下，满足：

\[
L_{\mathrm{old}}-L_{\mathrm{new}}\ge0.001
\]

这只是预注册的局部质量门槛，不代表完整 Development 收益或统计确认。

如果没有任何候选进入配对适应：

`NOT_TESTED`

不得把未执行的质量实验标记为已证明失败。

---

# 14. 本轮可能出现的结论

| 结果 | 科学含义 | 后续方向 |
|---|---|---|
| 小尺度产生稳定有效更新，且质量提升 | 当前提案机制具有进一步研究价值 | 可以设计正式多 Seed 验证 |
| 小尺度有效准入，但质量不提升 | 稳定性问题初步解决，策略方向或价值目标仍可能有问题 | 研究动作价值与组织适应收益 |
| 仅极小尺度通过，执行效果近乎为零 | 当前更新方向缺乏实用的局部可行范围 | 改进 Connection 提案算法 |
| 所有尺度都违反路径约束 | 当前方向仍然过度扰动计算路径 | 研究局部、稀疏的路由提案 |
| KL 很小但实际硬路由大量变化 | 分类分布约束无法单独控制离散计算路径 | 保留硬路径约束，进一步研究 Margin |
| 通过准入，但配对适应失败 | 联合训练执行或事务机制存在问题 | 修复适应流程，而非宣称算法失败 |

## 后续候选算法研究

如果最终证明现有 Connection 提案方向无法生成实用的局部更新，下一阶段再研究：

**Margin-Aware Local Connection Proposal**

其潜在目标是：

\[
\min_{\Delta\phi}
\widehat L_{\mathrm{policy}}
+
\lambda R_{\mathrm{route}}
\]

其中 \(R_{\mathrm{route}}\) 对实际硬路由变化施加约束或可微近似惩罚。

还可以研究根据状态、原始 Gate Margin 和预期收益，只对少量高价值路径提出局部变更。

但这涉及新的候选生成算法。

**本轮不得提前混入该算法，否则无法判断问题来自缩步范围，还是优化方法改变。**

---

# 15. 最终交付要求

Codex 必须返回：

1. 新实验 Source SHA 与 Config SHA-256。
2. 完整历史 Parent Checkpoint 身份与恢复验证。
3. A-C、B-C 的真实 Connection 候选参数哈希。
4. 每个候选的 22 个尺度完整指标。
5. 首个通过准入的尺度（如果存在）。
6. Connection 硬路径变化数量与比例。
7. 真实训练时的随机路由分布变化。
8. Cell 适应是否执行，以及实际 K 值。
9. 配对适应的 Calibration NLL。
10. Connection 最终是否被质量准入接受。
11. 真实候选的私有保存位置与 Receipt。
12. 完整计算成本。
13. GitHub 结果分支及远端 SHA 验证。
14. H1、H2、H3 的独立科学判断。
15. 下一阶段是继续完整联合训练，还是先改进 Connection Proposal 的建议。

---

# 16. 给 Codex 的最终指令

请基于 `ArcheLabs/mini-cells` 当前已经修复的 R3f 实现，建立独立的 R3f-v2 实验。

本次目标不是重新训练完整模型，而是验证：

**现有的 Connection 优化器是否能通过更精细的局部更新产生有效、稳定、非空的计算路径变化，并使这样的更新进入真实 Cell 适应训练。**

具体要求：

- 从已经恢复验证的 Seed 151001301、Step 100 官方状态开始。
- 仅执行 A-C、B-C。
- 每个 Arm 只生成一个真实的 Connection 提案。
- 在任何回溯之前保存完整候选。
- 使用原有的策略优化目标。
- 采用固定的 22 档尺度（包含零更新对照，非零尺度为 \(2^0\) 至 \(2^{-20}\)）。
- 保留 KL、硬动作变化、完整路径变化三个门槛。
- 只有有效、非空的候选可以通过准入。
- 通过准入后自动执行 K=10 配对 Cell 适应。
- 完成 Calibration 质量评价。
- 不读取 Development 和 Confirmation。
- 不启动完整 500 Step 多 Seed 矩阵。
- 全部私有状态与结果可恢复。
- 自动生成科学报告并尝试 GitHub 回传。
- 回传失败不能导致重新训练。

**任何科学负结果都必须保留，并且不能为了得到一次成功准入而自动改变算法或阈值。**

本次实验的真正成功标准不是取得更好的 NLL，而是确认：我们是否能够生成、控制、保存并实际采用一个由任务信号驱动的 Connection 更新。

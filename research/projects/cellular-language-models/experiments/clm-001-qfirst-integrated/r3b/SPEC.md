# MiniCells — CLM-001-R3b 路由稳定性与 Cell 专业化纵向诊断实验

**文档类型：科研实验方案 + Codex 可执行工程规范**  
**版本：v1.0（待预注册冻结）**  
**编制日期：2026-10-16**  
**目标仓库：** `ArcheLabs/mini-cells`  
**实验代号：** `CLM-001-R3b`（暂定；与原始 R3、既有合成工程测试严格隔离）  
**当前授权状态：** 允许准备、实现、CPU 测试与低成本预检；**真实 500-Step Kaggle GPU 实验尚未授权启动**。

---

## 0. 当前仓库核查与是否可以开始

### 0.1 已确认的工程事实

2026-10-16 仓库 `main` 已包含 Hugging Face Checkpoint Storage 的合并实现（PR #261；合并提交 `ebdaad9ece3c890cb428deaa72621dcd9cd7d597`）。当前检索时 `main` HEAD 为 `2e47a28a27a0932912673a1140e533e12abcc00a`，已经包含后续工程结果分支的合并。

主要入口：

- `research/projects/cellular-language-models/engineering/hf-checkpoint-storage/README.md`
- `research/projects/cellular-language-models/engineering/hf-checkpoint-storage/notebook.ipynb`
- `research/projects/cellular-language-models/engineering/hf-checkpoint-storage/reports/engineering.json`
- `shared/python/minicells_artifacts/hf_storage.py`
- `shared/python/minicells_runtime/hf_run.py`
- `shared/python/minicells_runtime/r3.py`
- `tests/engineering/test_hf_storage.py`

**已通过离线测试：** 工程报告记录 CPU 81 项通过、0 项失败（36 项工程测试 + 45 项冻结 R3 回归）、Lint PASS、历史 R3 文件差异为空；报告对应实现源码 `660c0abcfcc137ccf423dd92e27c47d3746c6e96`。

**已看到的远端写入证据：** GitHub 当前已存在合成工程结果：

`research/projects/cellular-language-models/engineering/kaggle-continuation/artifacts/660c0abcfcc137ccf423dd92e27c47d3746c6e96/7e5556f9698a/`

其中：

- `checkpoint-index.json` 记录 `hf_repo=libingjiang/minicells-research-checkpoints`，HF revision `0a94ae00b19c88751b718f30e2ec6c0db43e38a8`，三个 Arm 都有 0–10 Step 及 Event Revision 的索引。
- `storage-status.json` 报告 `status=DURABLE_COMPLETE`。
- **同一个文件仍标记 `HF_CROSS_SESSION_RESUME=NOT_VERIFIED`。**
- `engineering-progress.json` 标记 `status=COMPLETE`，但 `KAGGLE_CROSS_SESSION_RESUME=NOT_VERIFIED`，`scientific_decision=false`；实验身份明确是 `MINICELLS-KAGGLE-CONTINUATION-SMOKE-V1`，不是正式 R3 或 R3b。

因此：**HF 写入与 GitHub 索引已有实际回传记录，但还不能据此认定“跨独立 Kaggle Session 的恢复已验收”。** 同时，现有 `hf-checkpoint-storage` Notebook 明确为合成 CPU 工程验收入口，不是可直接运行真实 R3b 的 GPU 训练程序。

### 0.2 当前 Go / No-Go 结论

| 阶段 | 当前判断 | 允许动作 |
|---|---|---|
| 已有代码与离线回归 | PASS（以仓库报告为准） | 开始编写 R3b 方案、配置与测试 |
| HF 私有仓库写入及 GitHub 双通道 | 有远端回传证据；需补可验证证据链 | 运行只读审计及小规模复核 |
| **跨 Kaggle Session A→B→C** | **NOT_VERIFIED** | 先完成真实跨 Session 验收 |
| 真实 R3b GPU Runner | **尚未实现、未审核** | 允许实现与 CPU/短 GPU Pilot |
| **完整 R3b 500 Step** | **HOLD** | 上述门禁通过并冻结科学协议后方可启动 |

> 本文所谓“可以进入下一步”，指**开始实施 R3b 科学实验的工程准备与科学预注册**，不是跳过存储验收后立刻运行正式 GPU 训练。

---

## 1. 研究背景：R3 为什么需要下一步

原 R3 固定来源源码：`52b1950b23debbcd3f808c10eb6e0d3ea2d317c6`；原始实验不修改、不追认。

原 R3 记录的主要事实：

| 指标 | Q-R3 | QC-R3 |
|---|---:|---:|
| Step 500 Greedy NLL | 4.684990 | 4.699344 |
| Step 500 Greedy 平均动作路径长度 | 2.9851 | 14.7386 |
| Step 500 强制结束率 | 2.185% | 74.194% |
| CTX 动作占比 | 约 0.491% | 约 27.624% |

R3 的局部信用方向审计显示局部与固定后缀偏好的一致性约 **92.35%**，但这是局部机制的有效性信号，**不能推出真实 Connection 路由能改善语言质量**。同一套 R3 结果还包含：Step 400 的 pre-admission 在 Development 集上的探索性 QC 优势，而 Step 500 的最终 Development 评估转为劣势。这可以提出**后期路由退化假设**，但目前缺少同一套冻结 Probe 上的连续时间曲线，不能直接宣称已证明具体退化机理。

R3 的 `evaluation.snapshots=[]` 且 `periodic_diagnostics_enabled=false`；最终专业化和因果自强化没有可用的完整纵向轨迹。R3 的 QS 使用**零竞争梯度**的 Sham，因此 Q/QS 相同结果不构成独立的随机方向控制；报告时不得把这两者作为独立显著性复制。

**下一轮的核心任务：找出 Cell 的局部优势、Connection 的路由选择与 STOP/CTX 决策为什么没有形成稳定的全局收益。** 不以新增 Cell、放大参数规模或推出模型性能为目标。

---

## 2. 实验目标、假设与分阶段设计

### 2.1 实验边界

**R3b-DIAG（首轮）**：冻结 R3 的模型结构、训练规则、数据边界和主要超参数；只允许必要的工程封装、可观察性与诊断代码变更。三个主 Arm 在**一致的真实训练条件**下重跑，并保留固定 Probe 的纵向证据。该轮所有结论均为**单 Seed 探索性开发结论**。

**R3c-MECHANISM（条件后续）**：只有 R3b 足以区分候选原因后，才冻结单一训练机制变更，进行受控比较。不得在 R3b 过程中偷偷修改 beta、STOP 训练或竞争 loss 再将结果称为同一次研究。

### 2.2 预注册假设

| ID | 假设 | 关键证据 | 不成立时解释 |
|---|---|---|---|
| H1 | 局部 LCG 方向有一定可重复性，但并不保证整体策略改善 | 预冻结候选对的局部/长程方向一致率及 CI，和真实 greedy 性能并列报告 | 局部代理本身可能失配或采样偏差大 |
| H2 | QC 的 STOP / CTX 与 Cell 动作价值发生校准问题 | 相同 Probe 的 action-type gap、STOP margin、forced_end、路径长随 Step 变化 | 可能主要是连续 Cell 效果差或深度成本估计错误 |
| H3 | 不同 Cell 能形成条件优势，但 Connection 未必利用该优势 | 条件交叉优势、router–oracle agreement、paired regret、group-specific exposure | Cell 尚未分化，或测量组定义不足 |
| H4 | 路由偏好变化与后续 Cell 收益变化存在时间顺序关系 | 先行优势/后行路由与先行曝光/后行优势的配对轨迹 | 无法据此声称因果自强化；需要后续匹配置换干预 |

H4 的**因果**部分不在首轮自动作出肯定结论；只有执行了预注册的匹配路由干预，并提供跨 Seed 证据，才能将状态从 `CAUSAL_FEEDBACK_NOT_TESTED` 升级。

---

## 3. 先完成云端真实工程验收（必须）

### 3.1 不新增 GPU 成本的 A/B/C 测试

使用已经提交到仓库的：

`research/projects/cellular-language-models/engineering/hf-checkpoint-storage/notebook.ipynb`

**GPU 设置必须是 Accelerator None**；在 Kaggle 中启用已经存在的 `HF_TOKEN`、`GITHUB_TOKEN`。沿用当前实际 HF repo `libingjiang/minicells-research-checkpoints`，不擅自修改私有性或创建第二个命名空间。

1. **A（新合成运行）**：设置 `ACTION="auto"`、`SEGMENT_END_STEP=6`，在新 Kaggle Session 中保存版本。核查真实 HF revision、完整 bundle readback、GitHub index 中各 Arm 的 Step 6 记录与 Event Revision。不得假定已有 Step 10 的 run-id 是新 Run，必要时显式创建新的工程 run-id。
2. **B（全新 Kaggle Session）**：保持同一 Source SHA、Run ID、HF repo、实验身份，只将段终点改为 10。必须在没有挂载 A Output、没有本地 Checkpoint 的情况下经 HF 恢复 Step 6，继续至 10。日志必须包含远端恢复位置、`STARTED_FROM_RANDOM_INIT=false` 的等价证据。
3. **C（另一全新 Kaggle Session）**：设置 `ACTION="evaluate"`，加载历史 Step 6（按精确 Event Revision / HF revision），完全不做 optimizer step；再与独立的合成连续 0→10 对照比较**模型权重、Optimizer、训练游标、RNG、Registry、Shadow Event 状态及共同的 Probe 身份**。同时通过 GitHub 发布的 checkpoint-index 引用验证 GitHub→HF→Kaggle 的恢复链路。

**验收材料**：三个不同 Session/Version 的 Kaggle 链接、Source SHA、HF repo/revision、对应 GitHub 结果 Commit、状态报告以及完整状态哈希比较。

**门禁标准：** 远端下载校验无误；B 恢复点恰好是 A 的 Step 6 且非随机初始化；分段和连续训练状态在既有精确性契约要求内一致；C 不改写原始状态；密钥未泄漏；GitHub 无 `.pt`。失败或缺少实际证据时保持 `NOT_VERIFIED`，不能自动转 PASS。

### 3.2 生产数据规模与并发验证

合成 CPU Checkpoint 只有很小的尺寸，不能说明正式模型状态上传可接受。启动真实研究前必须使用**非正式、极短 GPU Pilot**测量：

- 原始三 Arm 的实际 Checkpoint 大小与 Bundle 文件数；HF 初次上传、增量上传、提交及首次 readback 耗时。
- 两张 T4 上 Worker 独立运行时，单一发布者能否拿到**三 Arm 一致的提交快照**；不得把两个 GPU 的半完成状态误拼成一次可恢复原子状态。
- 上传超时、Token 权限不足、HF 不可达及 CAS 冲突是否会停止于最后安全提交点，并保留 `DURABLE_PARTIAL`。
- 记录上传占总预算比和 HF 仓库容量增长。生产快照间隔可以经审查调整，但必须在真实训练前冻结。

当前的 `hf_run.py` 是 **CPU synthetic fixture**，不可将其 `SEGMENT_END_STEP=500` 视为已实现正式 GPU 实验。

---

## 4. R3b 科学实验的冻结配置

### 4.1 必须从 R3 复用的部分

- 数据源：`roneneldan/TinyStories`，固定历史 revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
- Tokenizer：历史声明的 SHA256 `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`，同时验证当前资产 Manifest 中实际 Hash。
- 数据切分沿用 R3：Train `[0,8192)`、Calibration `[50000,50128)`、Admission `[50128,50256)`、Development `[50256,50512)`；Confirmation 保持封存。
- 模型：width 128；6 个 Accepted Cell，1 个 Reserved Cell；4 Heads；路径上限 16；硬路由，不引入 soft mixture。
- 训练：每 Arm 500 Step、effective batch 8、sequence length 64、256,000 Token；R3 的 Q/LCG 等更新规则与损失不改动。
- Oracle / Shadow：维持 R3 的预算、候选采样、Cell/Connection/Path 提案与 Step 400 Admission，保留 Event Cursor 事务语义。
- 各 Arm 同一 Seed 下初始化权重和数据顺序一致；检查前向/后向额外的诊断随机数不会消耗训练 RNG。

### 4.2 Arm 定义

| Arm | 训练 | 作用 | 解释边界 |
|---|---|---|---|
| `Q-R3b` | R3 Q-only | 长程 Q 监督对照 | 必须同 Seed 从零重训，不得只引用 R3 的旧 NLL |
| `QC-R3b` | Q + R3 原版 LCG | 核心处理组 | 不提前修复 STOP 或重新调参 |
| `QS-R3b` | Q + 同等候选前向 + 零 LCG 梯度 | 工程成本、伪处理负对照 | 不是独立随机梯度 Sham；可能与 Q 的权重完全相同 |

建议首轮仅 Seed `151001301`，并以新身份记录为 R3b 的探索性复现；其他预注册 Seed 留待机制显现之后再决定是否使用。**禁止把同一 Seed 内的 Story Cluster Bootstrap 当成多 Seed 显著性。**

原 R3 Pilot 自动确定竞争权重（历史结果 beta≈4552.793）。R3b 应延续**原算法与固定的 Pilot 选择规则**，在 CPU/小 GPU 门禁后一次性冻结新 Run 的运行配置及实际 beta。不可根据 Development NLL 事后调整 beta；若决定直接锁定历史 beta 而不运行 Pilot，必须作为事先登记的独立方案，不能混用。

### 4.3 唯一允许加入训练主链的改动

- 可靠的 HF Checkpoint 保存/恢复、源 SHA 固定、结果索引。
- 只读、RNG 隔离的固定 Probe 诊断，不改参数与梯度。
- 明确的训练段终点控制、设备调度与上传耗时记录。
- 与历史同语义的 Bug 修复仅在确认会影响正确性并记录科学差异后才允许；影响训练轨迹的修改应视为新处理，不得冒充完全等价的 R3b。

**禁止：** 修改 Q、LCG、梯度比例、Optimizer、Learning Rate Schedule、STOP/CTX 奖惩、候选 Cell 采样或 Shadow Admission 规则。这样的改变属于 R3c 假设检验。

---

## 5. 固定诊断计划：解决 R3 最严重的证据缺口

### 5.1 Checkpoint 与 Snapshot 日程

冻结以下测量点（独立的分析计划，不改变 `training.steps`）：

`0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500`

每个测量点要明确标注：

- `pre_event`：训练更新已完成，但本 Step Shadow 事件尚未执行；
- `post_event`：相关事件全部提交后；
- `event_revision`：具体事件提交序号（Step 400 特别重要）。

Step 400 必须同时保留 `pre-admission` 与 `post-admission` 的完整 Checkpoint 身份，并明确比较的状态。建议日常纵向曲线统一使用 Step `post_event`；Step 400 的原始 pre-admission 结果另单列，不混在 post 曲线内。

每个 Arm 保存完整的冻结 Probe Bank（只生成一次）。每次快照记录 `checkpoint_sha256 / probe_sha256 / runtime_config_sha256 / evaluation_protocol_sha256`，且与 HF Index 对应。

### 5.2 轻量指标（每 50 Step）

- 同一 Development 固定样本上的 greedy / sampled / uniform / constant 的 NLL、PPL 与归一化计算成本。
- `average_path_length`、实际 STOP 率、`forced_end_rate`、`context_call_fraction`、`context_read_before_stop`、self-loop、各 Cell 路由占比。
- Router：`router_top_cell`、动作类型 Logit/Q 差距、`STOP-vs-best-CELL` margin、`CTX-vs-best-CELL` margin、近似并列比例、深度分桶、CTX 已读/未读分桶。
- Q 校准：可识别配对比例、方向一致率、rank-correlation、paired regret、Cell 评分熵；单独跟踪 STOP 与 CTX 的 Q 排序，不用 Cell-only 排序替代它们。
- Credit：`||grad_Q||`、`||grad_LCG||`、缩放后的比例、两者 cosine、裁剪比例、有限性及空事件比例。
- 训练与发布：训练 wall-time、诊断 wall-time、Checkpoint 大小、HF upload/readback 耗时与总字节数。

### 5.3 高成本功能评估（建议冻结 Step 0/100/200/300/400/450/500）

在预冻结的 Story/Token Position/Depth/CTX 状态 Probe 网格上：

1. 按 Cell、按输入组测量冻结后缀的真实 Loss Advantage（不是只比较 Router 调用次数）。
2. Cell 对 `(i,j)` 在组 A/B 上的条件交叉优势：`D_ij(G)=E[L(j)-L(i)|G]`；记录置信区间、覆盖率、不可辨识比例。
3. 相同 Probe 的“实际最佳 Cell vs Router 选择”一致率、路由后悔值、优势先行/偏好后行与曝光先行/优势后行的时间曲线。
4. 预先固定有限 Cell 对和状态组，或进行多重比较修正；禁止事后挑选显著分化 Cell。
5. 若新增有序二步路径诊断，则与主单步 Cell Advantage 区分，避免将单 Cell 优势误读为组合价值。

**复杂评估的资源消耗必须记录**。如果 Pilot 发现评估大幅超出 Kaggle 时间预算，应在真实训练前固定精简频率；开始后不得因结果不理想而改变分析网格。

### 5.4 绝对不可误报的结论

- 无匹配路由干预及多 Seed，不得宣称已经证明“因果自强化”。
- 不能仅凭调用比例不均匀说“Cell 已经专业化”。
- 不能仅用 `local_agreement≈0.92` 证明全局路由正确。
- 不能忽略 QC 自身 Greedy 与 Uniform 的 NLL 比较。
- 不得用历史 R3 原始 Step 400/500 曲线替代新 R3b 的冻结一致评估。

---

## 6. 主要统计分析与结果判定

### 6.1 主要终点

沿用 R3 的**预声明主对比**：

`(QC greedy − QC own-uniform) − (Q greedy − Q own-uniform)`

分别对 **NLL** 和 **cost-aware utility** 做配对 Story Cluster Bootstrap，报告差异、95% CI、有效 Story 数、独立 Seed 数及训练计算开销。

此外强制报告：

- `QC greedy − Q greedy` 的绝对 NLL 与成本；
- `QC greedy − QC uniform` 的差值；
- Q/QC 训练末的算力、路径成本及 STOP/CTX 行为；
- 每个快照时间点的纵向主终点（作为探索性轨迹，不把重复窥视当正式显著性检验）。

**要点：** 整体质量改进与局部信用一致性是不同终点。优于自己 Uniform 才意味着路由学到额外作用，优于对照 Q 才意味着新增 LCG 提供系统性收益。

### 6.2 机制判别矩阵

| 观察 | 优先考虑的解释 | 下一轮可能的单一干预 |
|---|---|---|
| QC 前期有效，后期 STOP margin 持续下移、forced_end 上升 | Action-type 价值校准/目标不一致 | 单独试 STOP–Cell 统一价值尺度或约束 |
| Cell-only ranking 进步，STOP/CTX 排序失败 | LCG 不覆盖非 Cell 动作造成优化不对称 | 单独试 Action-type calibration，保留 hard path |
| 缩放后 LCG/Q 梯度比例急剧漂移 | 冻结 beta 与训练动态的尺度失配 | 单独试预注册 beta schedule 或 gradient balancing |
| Cell 条件优势有但 Router 未跟随 | Credit-to-Policy 转换失败 | 单独试 router preference / better paired targets |
| Router 有分流但 Cell 不具条件收益 | 分配不代表专业化 | 单独试促进可验证专业化的 credit 设计 |
| QC 与 Q 都退化 | 可能是数据预算/训练稳定性共同缺陷 | 先排除共享组件缺陷再改 LCG |

所有表格项都只是**待验证解释**，R3b 不能提前将其中之一作为已知事实。

### 6.3 决策规则

**R3b 首轮的成功定义是诊断信息完整，不是 QC NLL 必须改善。**

- `ENGINEERING_PASS`：HF + GitHub + 独立恢复以及完整保存均正常。
- `DIAGNOSTIC_COMPLETE`：预冻结所有必要快照与关键指标齐全，缺失有合理原因并显式记录。
- `H1_DEV_SIGNAL` / `H2_EXPLORATORY`：按预注册数据处理结果标记，不等于正式确认。
- `CAUSAL_FEEDBACK_NOT_TESTED`：未做匹配随机干预时保持此状态。
- `MULTISEED_NOT_TESTED`：Seed 2/3 未跑时保持此状态。
- `INCONCLUSIVE`：证据不足时合法且有价值，不得为了获得 PASS 事后改协议。

---

## 7. R3b 训练 Runner 与存储接口的实施要求

### 7.1 新增独立实验命名空间

建议增设：

```text
research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/
  r3b/
    README.md
    SPEC.md
    config/proposed.json
    implementation/clm001_r3b/
      cli.py
      runner.py
      instrumentation.py
      evaluation.py
      publication.py
    notebook.ipynb
    reports/
tests/research/clm001_r3b/
```

以上为**建议产物，不是当前仓库已具备的文件**。尽量复用 `clm001_r3` 已审核的模型/训练、`minicells_runtime` 续训适配器、`minicells_artifacts.hf_storage` 存储模块，避免复制完整训练器。

### 7.2 运行模式与示意命令

要求新增以下逻辑模式（名称为期望接口，Codex 需与实际 CLI 对齐；**现在直接运行这些示例不应被理解为已支持**）：

```bash
# 工程/权限预检，无真实训练
python r3b/run.py --mode preflight --storage huggingface

# 新建一轮冻结科学实验，仅第一训练段
python r3b/run.py --mode train --seed 151001301 --segment-end-step 200

# 新 Session 从 HF 精确恢复，继续至 Step 350
python r3b/run.py --mode resume --checkpoint-ref checkpoint-index.json --segment-end-step 350

# 只读评估指定历史模型
python r3b/run.py --mode evaluate --arm QC-R3b --step 400 --event-revision 0

# 明确的新研究 Fork，而不是伪装 Resume
python r3b/run.py --mode fork --parent-checkpoint-ref checkpoint-index.json --new-experiment-id CLM-001-R3c
```

对一个三 Arm Kaggle 运行，可预置总目标 **500 Step**，分段端点 `200 → 350 → 500`。分段端点**是执行预算，不进入冻结训练配置哈希**。每个 Arm 可进度不同，但 GitHub 只有在验证一致的 Run Bundle 后才写入受控的最新 Index。

### 7.3 Hugging Face 发布要求

- 按实验 / run-id / arm / step / event revision 做不可变索引；训练期间一旦本地安全提交即安排持久化。
- 复用已有的 `HFCheckpointStore`：不可变 SHA 对象、HF 精确 revision、单一 Publisher、CAS 防回退、恢复时 SHA256 全量核对。
- 训练进度不得越过需要远端已提交确认的安全边界；远端出错优先返回 `STORAGE_BLOCKED`，保留已确认的历史索引和本地状态。
- Worker 不得各自上传半完成三 Arm Bundle；实现 Host 进程上的**串行发布队列与状态协调**。并发运行与恢复一致性必须完成负面测试。
- 控制 HF commit 频率与网络开销：不牺牲 Event 原子性前提下，对频繁轻量状态变更允许缓冲，但不得把“尚未上传的 Checkpoint”标记为已持久化。
- HF 私有仓库中的原始 Probe / 训练资产仍可能包含有版权/授权约束的数据，只保存运行必需数据或经身份哈希验证的可重建资产；不要将原始训练文本发布到公开 GitHub。

### 7.4 GitHub 发布要求

继续复用历史 R3 结果分支机制与白名单方式，仅发布：

- `experiment-manifest.json`、`checkpoint-index.json`、`storage-status.json`；
- `metrics-summary.json`、`routing-longitudinal-summary.json`、`specialization-summary.json`、`gradient-audit-summary.json`；
- `statistical-contrast.json`、`resource-ledger.json`、`failure-index.json`、`publication.json`。

不得上传 `.pt`、模型/优化器/RNG、私有 Probe、Train Oracle 明细、HF/GitHub Token。对于还没上传成功的历史 Step，不得发布伪造 HF revision。远端 HF 存储成功与 GitHub push 成功分别记录，允许 `STORAGE_BLOCKED` 下保留有效的科学或工程失败证据。

---

## 8. 正式运行时间与算力预算

建议保持 R3 同样的两张 T4 设备和三个 Arm：

- `cuda:0`: Q → QS；
- `cuda:1`: QC。

首次真实 GPU Pilot 测定**实际**单 Step 耗时、上传开销与重评估开销后制定每个 Segment 的最大持续时间。不要复用合成 CPU 的 53 秒或纯训练 R3 的时间作为新纵向诊断的预算。真实实验不得因为预测时间超预算就悄悄省略预冻结的关键测量。

计算与存储预算单列：训练 GPU 秒、评价 GPU 秒、HF 上传/下载与等待秒数、候选额外 FFN 计算量、推理实际调用成本。`max_steps` 和样本数不得随着中途进度临时增加。

---

## 9. 预注册与数据隔离

1. 先在 GitHub 上提交冻结 `SPEC.md`、`config/registered.json`（或对应锁文件）、Source SHA、数据 Manifest SHA、探针生成规则/固定种子、诊断日程、统计分析脚本与版本。
2. Calibration 仅用于工程/度量校准；Development 为唯一探索性比较集；Admission 不得泄漏到主路由训练；Confirmation 保持封存。
3. 如果分析代码根据 Development 结果调整了路线，则后续独立确认必须另设实验，不能引用同一 Development 结果当作确认。
4. 记录全部成功/失败 Seeds 与 Checkpoint，不因结果不好而只发布最好一次。
5. 使用新的 R3b Experiment ID/Source SHA，即使参数与 R3 相同也不能共享 R3 原始 checkpoint 身份。

---

## 10. 测试计划与硬门禁

### Gate A — 代码与身份（CPU）

- R3 原始源码文件、原始结果分支和既有 Checkpoint 均不更改。
- 81 项已有测试仍 PASS，新 R3b 测试覆盖新增诊断和运行模式。
- 启用诊断与关闭诊断的相同训练步骤必须产生一致的训练 RNG/模型状态（给定项目既有确定性要求）。
- 事件在 Step 400 中断、上传失败后恢复，不会重复 Shadow 提案、Admission 或 Promotion。
- 只读 Evaluate 期间严禁调用 optimizer.step；源/数据/探针/事件版本错配必须拒绝加载。

### Gate B — 真实 HF I/O（小 CPU 工程 Run）

- 私有仓库写权限、Pinned Read、SHA256、仓库 CAS、回读完整性测试成功。
- GitHub Checkpoint Index 指向实际存在并可读取的 HF revision。
- 历史任意一步 / Event Revision 可独立获取，私有二进制数据不进入 GitHub。

### Gate C — **真实跨 Session A/B/C（必须）**

- Kaggle A/B/C 为不同执行 Session，B 无需挂载 A Output、无需用户本地下载；C 从 HF Pinned Revision 执行只读评估。
- 合成 0→6→10 与不中断 0→10 连续训练的完整状态一致；GitHub 回传的引用可复核。
- 报告 `HF_CROSS_SESSION_RESUME=PASS` 前必须保留上述证据；旧 `NOT_VERIFIED` 不因现有 `DURABLE_COMPLETE` 自动升级。

### Gate D — 真实 GPU Runner Pilot

- 一次短 T4 Pilot 在**新的 R3b 代码**上确认实际尺寸、上传时间、设备并发/串行发布协调、Snapshot 成本和恢复过程。
- 验证固定数据 Manifest 和预冻结科研流程，确认没有隐式改算法。
- 提交 Pilot 报告后再决定是否按 200→350→500 正式跑。

### Gate E — 协议冻结与人工授权

- 上述 A-D PASS；`SPEC.md`、冻结配置、统计计划、预计 GPU/存储成本和正式实验 Commit SHA 已提交；用户明确授权正式运行。
- 任意一项未满足，即使 Kaggle GPU 可用，也不能正式启动 R3b。

---

## 11. Kaggle Notebook UX 与最小人工操作

建议 R3b Notebook 首屏只暴露以下配置：

```python
EXPERIMENT = "CLM-001-R3b"
ACTION = "preflight"  # preflight / train / resume / evaluate / publish-only
SEGMENT_END_STEP = 200
HF_REPO = "libingjiang/minicells-research-checkpoints"
RUN_ID = None          # 程序自动固定，用户不得在 Resume 中重生 ID
CHECKPOINT_REF = None # 在 Resume/Evaluate 时填 GitHub 索引引用或选择 HF latest
```

所有研究配置、代码 SHA、Tokenizer SHA、HF URL、Bundle Hash、Seed 等应由锁定文件和服务端索引获得，用户不应手填哈希或恢复时改变科学参数。使用 Kaggle Secrets 获取访问令牌，日志不可打印 Token。

每次结束输出人可读摘要：

```text
EXPERIMENT=CLM-001-R3b
RUN_ID=<frozen>
SOURCE_SHA=<full-sha>
ARM_PROGRESS={Q:200, QC:200, QS:200}
HF_REPO=libingjiang/minicells-research-checkpoints
HF_REVISION=<full-commit-sha>
CHECKPOINT_INDEX_SHA256=<sha256>
STORAGE_STATUS=DURABLE_PARTIAL
GITHUB_RESULTS_BRANCH=<branch>
SCIENTIFIC_STATUS=EXPLORATORY_IN_PROGRESS
```

如果当前仍处于工程/合成模式，必须显示 `EXPERIMENT=MINICELLS-KAGGLE-CONTINUATION-SMOKE-V1`，严禁伪装为上述真实输出。

---

## 12. 异常、重试与安全边界

| 异常 | 正确行为 |
|---|---|
| HF Token 失效或写权限不足 | 拒绝进入真实训练；可发布匿名化工程失败信息 |
| HF 上传完成但 GitHub Push 失败 | 不重新训练，仅 `publish-only` 重试 GitHub |
| HF 在下一安全提交点上传失败 | 停止在最后可恢复状态并报告 STORAGE_BLOCKED |
| 新 Kaggle 会话无法找到请求的 Run | Fail closed，不从零开始 |
| HF revision / checkpoint SHA 不一致 | Fail closed，绝不尝试反序列化 |
| 旧 Worker 覆盖了较新 Step | CAS 阻止更新；不得回退 latest |
| 只有两个 Arm 成功 | 精确发布 PARTIAL 与各 Arm 进度，绝不声称完整配对结果 |
| Snapshot 漏采或 Probe 重建 | 显式声明数据缺失，不能补造当时的纵向记录 |
| Torch pickle 来自未受信任的来源 | 禁止加载；仅允许身份绑定的受控私有仓库与已审核训练代码 |

---

## 13. Codex 的执行计划及交付清单

### Phase 0：审计已有成果（先完成）

- 读取本方案第 0 节所列的真实工程报告与 `checkpoint-index.json`。
- 追踪刚刚发布的 `DURABLE_COMPLETE` 工程 Run 来源：确认是否跨 Kaggle Session；若未满足 A/B/C，不许将跨会话门禁标记 PASS。
- 如 A/B/C 未做：优先指导执行已有 HF 工程 Notebook，不追加新的同类系统。

**交付：** `engineering-acceptance.json`，写明 Gate A/B/C 状态、真实 Kaggle Version、HF revision、GitHub Commit 与异常。

### Phase 1：冻结 R3b 科研协议

- 建立独立 R3b 目录，继承 R3 主要科研条件与基线。
- 补齐诊断定义、样本冻结、统计分析和 Step 400 Event 时点契约。
- 明确所有 R3b 改动是否仅影响观测而不影响训练。

**交付：** `r3b/SPEC.md`、冻结配置、`experiment.lock.json`、设计审计。

### Phase 2：实现真实 R3b GPU Launcher

- 将已经合并的 `HFCheckpointStore`/`minicells_runtime` 接入**真实** R3 训练 Worker，不调用 synthetic CPU fixture 当作科研训练。
- 增加训练终止边界、统一发布队列、固定 Probe 诊断和双通道结果。
- 保持原 R3 的 hard path / Q / LCG / Shadow 事务语义。

**交付：** 新 Notebook/Runner，集成 CPU 测试，CI 通过。

### Phase 3：最小真实 GPU Pilot

- 小规模训练+HF 持久化与恢复回读，证明时间和带宽预算可行。
- 测试上传失败、Event 恢复、设备并发和诊断不干扰训练。

**交付：** `gpu-pilot.json`、Checkpoints HF 索引、GitHub 回传链接、预算复核。

### Phase 4：正式运行（仅明确授权后）

- 真实 Seed 1，Q/QC/QS 三 Arm，500 Step，总目标不变；Segment 200→350→500。
- 保存完整历史 Checkpoint、固定 Probe 功能诊断与 GitHub 聚合结果。
- 若负结果同样按计划输出；不得因负结果删除或改写实验。

**交付：** GitHub 结果分支、HF pinned indices、纵向图表/统计、问题分析、资源账本、下一轮单一机制变更建议。

---

## 14. 最终验收标准

达到以下全部条件才能认为“R3b 研究实验执行完毕”：

- [ ] HF 与 GitHub 双通道真实验收证据完整；跨 Session A/B/C PASS。
- [ ] R3b 新 Runner 使用真实模型、真实 TinyStories 固定数据与两张 T4；不是合成工程 Fixture。
- [ ] 冻结协议 / Source SHA / 数据与 Probe 身份全程一致。
- [ ] 三 Arm 的 500 Step + Event Cursor 均完整，或报告明确的 PARTIAL。
- [ ] 每个要求的诊断点都有 HF 可恢复 Checkpoint 与对应 GitHub 指针。
- [ ] STOP/CTX/成本、Cell 条件优势、路由后悔值、信用梯度及缺失率均公开汇总。
- [ ] 结果保留 `single-seed exploratory` 身份，不捏造因果或多 Seed 结论。
- [ ] 任何 HF/GitHub 失败都有可追溯状态，且无 Token 或模型二进制泄漏到公共 GitHub。

---

## 15. 给 Codex 的直接指令

> 请以 `ArcheLabs/mini-cells` 当前 `main` 为起点，先审计 PR #261 合并后的 Hugging Face + GitHub 双通道工程与最新工程结果。仓库中的小型 CPU 运行已经回传 `DURABLE_COMPLETE` 和真实 HF revision，但同一结果仍标记 `HF_CROSS_SESSION_RESUME=NOT_VERIFIED`；不得把“已上传成功”误判为“跨新 Kaggle Session 完整恢复已验收”。首先完成并复核真实 A/B/C 门禁。
>
> 然后根据本规范创建独立 `CLM-001-R3b` 路由稳定性与 Cell 专业化纵向诊断实验，复用冻结 R3 的真实训练算法和已合并的 HF 存储系统；补齐真正的双 T4 GPU Runner、固定 Probe、分段训练、Step 400 前/后事件诊断、单一远端 Publisher 和 GitHub 轻量汇总。
>
> 本阶段允许代码、CPU 测试与最小 GPU Pilot。**不要自行启动真实三 Arm 500 Step 训练；要在 Gate A–E PASS 后提交实施结果和预算，等待明确授权。** 不修改历史 R3、不覆盖旧 Checkpoint、不更改其科研结论、不下载 Kaggle 旧 Output、不发布私有 `.pt` 到 GitHub。

---

## 附录：本规范核对的关键仓库位置

1. PR #261：`https://github.com/ArcheLabs/mini-cells/pull/261`
2. HF 工程说明：`https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/engineering/hf-checkpoint-storage/README.md`
3. 工程 CPU 报告：`https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/engineering/hf-checkpoint-storage/reports/engineering.json`
4. 已发布 HF Checkpoint 索引：`https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/engineering/kaggle-continuation/artifacts/660c0abcfcc137ccf423dd92e27c47d3746c6e96/7e5556f9698a/checkpoint-index.json`
5. 已发布存储状态：`https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/engineering/kaggle-continuation/artifacts/660c0abcfcc137ccf423dd92e27c47d3746c6e96/7e5556f9698a/storage-status.json`
6. 原 R3 协议：`https://github.com/ArcheLabs/mini-cells/blob/main/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3/SPEC.md`

**说明：** 文档中引用的工程状态来自 GitHub 可见源码和结果记录；私有 Hugging Face 的真实字节内容以及 A/B/C Kaggle 独立会话证据尚未由本次核查直接读取。涉及 R3b 的路径、CLI 和目标输出均为待实现规范，不表示仓库目前已经支持。

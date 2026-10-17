# MiniCells — CLM-001-R3b Diagnostic Execution Optimization

**实验诊断执行优化与训练/分析解耦实施规范（Codex 执行版）**

- **版本**：v1.0，工程变更提案；经审查与测试后冻结。
- **日期**：2026-10-17。
- **目标仓库**：`ArcheLabs/mini-cells`。
- **审核基线**：`main` 合并 PR #263 后的提交 `bd051e6854d56e21655e304a6763eee0b369b59c`。Codex 开工时必须重新检查远端 HEAD，记录任何新增提交，并在新分支实施。不要把本文中的 SHA 冒充运行时的最终源码 SHA。
- **实验身份**：`CLM-001-R3b`；三组 `Q-R3b` / `QC-R3b` / `QS-R3b`。
- **本次变更性质**：**仅优化诊断的执行时机、资源使用、离线评估及证据发布**。保持冻结的 R3 科学算法、训练预算与数据边界不变。
- **当前授权**：允许设计、实现、CPU/模拟验证、针对已有 Pilot 产物的只读性能审计，以及经确认的小规模、严格限定的诊断基准；**不授权重新执行已经成功的完整 Pilot，不授权启动正式 500-Step 训练**。

> **最终交付目标**：训练不再同步执行绝大多数昂贵的 Development/Functional 评估；所有关键模型状态仍实时、可靠地持久化至私有 Hugging Face；历史 Checkpoint 的诊断可独立、可恢复、可追溯地执行，并把轻量研究结果回传 GitHub。**将训练关键路径控制在约 2 小时是待实测的工程目标，不是已有测量保证；完整研究（训练 + 所有离线诊断）的 GPU 总时间可能超过 2 小时。**

---

## 0. 背景、已知事实及本次需要解决的问题

### 0.1 已有 Pilot 结果

根据本次 Pilot 的运行反馈（由实验执行方报告；下列数字需在工程记录中绑定原始 HF/GitHub 产物，不得冒充已独立重测）：

| 项目 | 已报告结果 | 判定 |
|---|---:|---|
| 双 T4 分配 | GPU0：Q→QS；GPU1：QC | 编排成功 |
| 单次 Pilot | 约 **8 分 35 秒** | 工程链路完成 |
| 每个 Training Step | 约 **3 秒** | 仅短 Pilot 样本，需注明误差 |
| Calibration 快照 | 约 **156–163 秒** | **高成本热点** |
| 三组状态恢复 | 模型、Optimizer、CUDA RNG 等通过 | 保留证据，避免无效重复 |
| QC 局部一致率 | 约 **92.3%** | 局部诊断信号，非模型整体性能结论 |
| HF 完整回读 | 成功，最终约 **116 MB** | 不等于生产全量历史存储/上传预算 |
| GitHub 结果回传 | 成功 | 双通道已运作 |
| 当前完整流程估算 | **7.24 小时** | 估算，**不是**真实 500 Step 总耗时 |

Pilot 证明新工程链路在已测条件下可以运作，**并不代表 R3b 科学假设已得到验证**。

### 0.2 理论训练基线（仅用于容量规划）

以两个独立队列、每次更新约 3 秒计算：

```text
GPU0:  Q-R3b  500 step + QS-R3b 500 step ≈ 3000 s
GPU1:  QC-R3b 500 step                  ≈ 1500 s
理论训练更新 makespan ≈ 3000 s ≈ 50 min
```

该值**不包含**数据准备、Step 0 冻结探针、Shadow Proposal/Admission/Promotion 事务、必需的 Admission 评估、远端 Checkpoint ACK、最终 SHA 回读、分段恢复、进程启动及波动。不能把 50 分钟作为可靠完成时间承诺。

### 0.3 当前代码里的具体耦合

现有位置（合并 PR #263 后）：

- `r3b/implementation/clm001_r3b/runner.py::segment`：在 Step 0、每个诊断 Step、事件前/后等调用 `observe()`；`observe()` 直接调用 `instrumentation.snapshot()`，之后 `deliver()` 等待 HF Publisher ACK。
- `r3b/implementation/clm001_r3b/instrumentation.py::snapshot`：无论快照属于 Light 还是 Heavy，都在同步路径计算四种策略的 `evaluate`（`greedy/sampled/uniform/constant`）、路由评分、STOP/CTX 反事实校准；Heavy 时进一步执行 Calibration/Development 固定后缀与 Cell 功能对照。
- `r3b/implementation/clm001_r3b/analysis.py::aggregate`：假定大量 `diagnostics/snapshot-*.json` 已存在且每条均包含质量评价；已有 `plots.render` 也直接依赖这些数据。
- `r3b/implementation/clm001_r3b/evaluation.py::evaluate_checkpoint`：已有**只读的历史质量评估**入口，但尚未形成可批量执行 Quality/Functional 纵向诊断的独立发布管线。
- `r3b/implementation/clm001_r3b/pilot.py`：目前使用 Calibration 快照耗时估算高成本阶段；它没有直接测出全量 Development、Event 以及生产 Checkpoint 多次上传的完整真实成本。

**优化对象是 R3b 的观测执行路径，不是 R3 的 `Trainer.update()`。**

### 0.4 一个不可忽略的边界：Shadow Admission 的计算不能随意延期

原始 R3 `clm001_r3/events.py::process()` 的 `pre-admission`、`admit-*`、`promote` 是**训练算法语义的一部分**。其 Development/Admission 评价结果可能参与后续候选接纳判断，私有训练和 Promotion 会影响后续模型状态。

**严格要求：**

1. **不得**把 `events.process()` 中影响 Admission 决策的评估推迟到离线分析。
2. **不得**删除、跳过、替换其判定函数，或用低样本代理直接替代。
3. **不得**改变 Event 执行顺序、Event Cursor、每个事件原有的持久化边界及重试语义。
4. 可以把 `events.process()` 结束后的**额外解释性诊断**延后；不能把算法所需判定本身延后。
5. 性能报告必须独立列示 **Mandatory Event Cost**，不能把所有诊断耗时都说成“可删除”。

---

## 1. 范围与验收目标

### 1.1 P0 必须实现

1. **训练关键路径解耦**：训练期间不再同步运行一般 Development 全策略 Evaluation 和高成本 Functional Counterfactual。仅执行冻结的训练算法、必须的事件判定、必要的低成本训练日志、Checkpoint/事件持久化。
2. **历史 Checkpoint 独立诊断**：从私有 HF 的**精确 revision + bundle + SHA256** 恢复指定 Arm/Step/Event Revision；独立执行 Quality、Functional、Action-Type 校准等；不执行优化器更新、不修改原始运行状态。
3. **双通道发布**：模型/私有原始诊断保存 HF，GitHub 只发布白名单内的聚合结果和固定指针；训练索引与分析索引独立。
4. **可分段、可恢复**：训练恢复、离线诊断任务恢复均具备幂等性；运行中断不丢失已提交进度。
5. **审计与科学可比性**：同一模型 Checkpoint 使用原冻结 Data/Tokenizer/Probe Bank，记录研究版本、分析版本与每条观测的确切身份；不把缺失观测当作 0 或 PASS。
6. **改动不污染科学算法**：证据证明开启/关闭/延期观测后，训练模型、Optimizer、RNG、数据游标、Registry、Shadow Event 状态一致。

### 1.2 P1 争取实现

- 降低相同输入下的重复推理、CUDA host 同步、同一 Checkpoint 的重复下载与反事实计算。
- 支持一次分析 Session 连续评估多个 Checkpoint，且按任务/资源预算分片。
- 以历史快照结果生成图表，不触发模型再推理。
- 自动生成独立的训练关键路径预算和离线科学分析预算。

### 1.3 明确不做

- 不重新运行整个 8 分 35 秒 Pilot 来证明早已验证的双 T4、HF/GitHub、RNG 恢复。
- 不修改训练 Step=500、Seed、Token Budget、模型宽度、Cell 数、Q/LCG/STOP/CTX 算法、原始训练梯度、原始 Shadow Admission 判定、R3 历史代码。
- 不根据已有 R3b Development 结果调整研究指标以寻找“更好结论”。
- 不让正式 500-Step 自动训练绕过 Gate C/D/E 或将 Authorization 字段修改为 `true`。
- 不在 GitHub 提交 `.pt`、完整训练/Probe 数据、按故事原始数组、Oracle 私有标签或访问令牌。

---

## 2. 目标架构：Training 与 Analysis 分离

```text
                     GitHub — immutable training source + locked protocol
                                          │
                                  Kaggle Training Session
                                          │
                         Original R3 Trainer + mandatory events
                                          │
                      Small journal / snapshot reference (no heavy eval)
                                          │
                           Atomic local checkpoint + all event revisions
                                          │
                       Single Host HF Publisher / verified remote ACK
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
          Hugging Face Private Repo                   GitHub lightweight progress
       original weights + state + assets                 run / step / HF pointer
       immutable frozen banks + journal
                   │
                   ▼
                          Independent Analysis Session(s)
                   │
        select immutable checkpoint by arm + step + event_revision + revision
                   │
               ┌───┴───────────────┐
               ▼                   ▼
         Quality Tasks         Functional / STOP / Cell Tasks
               │                   │
               └─────────┬─────────┘
                         ▼
                  HF private analysis results
                         │
                         ▼
                 GitHub aggregate & figures
```

**不变式：** 训练端产出带完整恢复状态的“事实”，分析端从已持久化事实计算“解释”。分析失败不得回滚已经成功的训练，也不能更改训练端的 `latest.json` 指针。

### 2.1 三类计算必须分别标注

| 类别 | 例子 | 是否允许延期 | 所需证据 |
|---|---|---|---|
| **TRAINING_SEMANTIC** | `Trainer.update`、Oracle 实际训练标签、Shadow private updates、Admission/Promotion 判定及原始事件持久化 | **否** | 完全保持 R3 语义、数值和顺序 |
| **TRAINING_TELEMETRY** | Trainer Journal、当前 Step/事件、梯度范数（已在 Journal）、Checkpoint 身份、计时、训练曝光统计（已有记录） | 通常无须延期；不能新增大量 forward | 相同训练轨迹、额外成本实测 |
| **POSTHOC_ANALYSIS** | Development 全策略 NLL、Router Q Margin、STOP/CTX 反事实、固定后缀 Cell Regret/Rank/Crossover、Bootstrap/绘图 | **可以** | 不变的 Checkpoint/数据/Probe 与独立分析身份 |

**注意：** 目前 `instrumentation.snapshot()` 中的 `router_rows()`、`action_calibration()` 虽然可能比 Functional 测量便宜，但也使用额外模型计算。实施前分别 profile，只有被证明足够轻、且不影响关键路径目标的部分才可留在训练侧。最安全的默认值是训练侧仅作 Journal 汇总与 Checkpoint 身份记录。

---

## 3. 科学协议不变式：禁止“工程优化”改变实验结论

### 3.1 必须逐字节或语义严格等价的项目

- 原始 R3 `Model`、`Trainer.update()`、`Oracle`、竞争性信用 LCG、`Q/LCG` 梯度施加与裁剪、STOP/CTX、Shadow/Nursery/Admission/Promotion。
- 训练 Seed=`151001301`、三个 Arm、每个 Arm 500 更新；每个 Arm 有效 Token=256,000（不含独立统计的 Shadow private 更新）。
- `training.steps=500` 和原训练 Budget/Batch/Tokenizer/Data Splits 保持不变；分段终点仍是**执行控制**，不得改变冻结配置哈希。
- 原始 `Q-R3b → QS-R3b` 与 `QC-R3b` 双队列，避免并发状态污染。
- Q/QC/QS 的差异与负对照定义；尤其 QS 是**零应用 LCG 梯度**，不能重新解释为随机梯度对照。
- 每次原始事件的本地 Checkpoint 和远端存储 ACK；不得因移除额外分析而跳过事件后状态。
- Calibration/Development/Admission/Confirmation 数据角色原封不动；**Confirmation 始终封闭**。

### 3.2 两种不同的版本身份

**Training Identity（不允许通过改动分析脚本偷偷重定义）**：

`experiment_id + run_id + training_source_sha + frozen_runtime_config_sha + assets_manifest_sha + tokenizer_sha + protocol_sha + seed`。

**Analysis Identity（可独立演进，但必须可追踪）**：

`analysis_source_sha + analysis_protocol_sha + input_training_checkpoint_ref + frozen_probe_sha + dataset/window_sha + evaluation_seed + policy + step + event_revision + stage`。

必须把两者都写入诊断结果。`analysis_source_sha` 允许比训练源码更新；**不得因为新分析代码而把历史 Checkpoint 伪装成由新训练代码生成**。如果输入来自 R3b 新训练，从一开始必须使用经过验收的训练源码，而不是将旧 R3 Checkpoint 硬标记为新 R3b。

### 3.3 “延后诊断”与科学时间点的区别

- `measurement_time` 是实际执行离线测量的时间；
- `model_checkpoint_step/event_revision` 是被测模型的**历史状态**；
- `analysis_mode=POSTHOC_FROM_IMMUTABLE_CHECKPOINT`；
- 不许标记成 `LIVE_AT_TRAINING_STEP`；
- 不许根据最终 Step 500 模型重新生成 Step 200 的“历史模型状态”。

### 3.4 冻结探针的科学范围

原始 R3/R3b 的冻结 Probe Bank 包含 Step 0 捕获的状态；`rebind()` 会绑定当前 Registry 身份，但不会将初始 hidden state 自动变成“由当时新模型产生的在线状态”。

因此必须**明确区分**：

1. `FROZEN_INITIAL_STATE`：在**相同初始捕获状态**上对不同训练 Checkpoint 的 Router/Cell 反事实测量；可用于固定条件下的可比性。
2. `LIVE_RECOMPUTED_STATE`：在不同训练 Checkpoint 上按冻结输入窗口重新采集模型内部状态；能反映分布随模型变化的现象，但样本覆盖可能不同。

首轮离线优化必须**至少完整复现既有的 FROZEN_INITIAL_STATE 协议**；如增加 LIVE 重采样，必须声明它是额外探索指标、固定独立协议，不能在同一曲线上与 FROZEN 状态混画或混用置信区间。不能用未经验证的 `rebind()` 来声称已观察到训练期真实在线状态分布。

---

## 4. 建议冻结的诊断执行计划（R3b-DEO-v1）

> 以下为**实施提议**，必须在正式 500-Step 前经审查冻结，并以 JSON/Markdown 保存协议 SHA。数值计划是为了减少重复执行、保持 R3b 科学主问题的可检验性，不是声称“删掉诊断后指标精度完全不受影响”。

### 4.1 Checkpoint：不删减

保留所有原定的训练关键点以及事件版本：

```text
0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
```

- 每个 Step 保留**原始完整训练恢复状态**，不能只保存模型权重。
- Step 100/200/400 的事件链中，保存 `pre_event` 和每一个**已提交**的 Event Revision。
- Step 400 的 `revision=0`（前事件）与**最新已提交 revision**（完成后）都必须可直接选择。
- 每个 Checkpoint SHA256、事件游标、Frozen Probe Bank SHA256、训练 Source SHA 等必须索引齐全。
- 根据已有 HF 的内容寻址去重，不重复上传每个已存在对象；**不得**为了缩短上传耗时合并/丢弃原始事件提交点。

### 4.2 训练期轻量 Telemetry：全部 11 个 Step

| 时间点 | 保留内容 | 强制限制 |
|---|---|---|
| Step 0,50,...,500 | Checkpoint SHA、Step、Event、Model/Registry 版本、trainer journal 汇总、Q/LCG 梯度范数/余弦/裁剪、曝光统计、耗时和存储字节 | **不执行完整 Development Evaluation；不执行昂贵的 `Snapshot.relative`/固定后缀对照** |
| Step 100/200/400 所有 Event Revision | 精确事件索引；可选极轻量元数据 | `events.process` 中原本的语义性评估照常运行 |

如果 `training_exposure_by_group` 已在原 Journal 可获得，做只读聚合即可；不要为了补齐某个新指标再做一次完整训练 rollout。

目标是将新增 Telemetry 的执行时间压缩到训练更新开销的一小部分。把新增 P50/P90/P99 时间纳入 Ledger；**在尚未实测之前，不以具体秒数作为已达成事实**。

### 4.3 Quality Evaluation：离线，全量 Development，预注册主对照

必须支持对相同 Development window + seed + story 分组进行下列测量：

- **Primary policies**：`greedy`、`uniform`（用于主 DID 与各 Arm 内 Greedy-vs-Uniform 差）。
- **Secondary policies**：`sampled`、`constant`（只在指定关键时间点执行，或按预算延期，不可省略但报告为 COMPLETE）。
- 记录质量 NLL/PPL、实际 STOP/forced-end、CTX、non-STOP 路径长度、规范化计算成本和 per-story 配对身份。不能用低样本 Pilot 质量值代替正式 Development。

**建议第一批完整 Evaluation 时间点（全部 Arm、全量 Development，Greedy+Uniform）：**

```text
0, 100, 200, 300, 350, 400[pre_event], 400[post_event], 450, 500
```

**Secondary 全策略（全部 Arm，Sampled+Constant）：**

```text
0, 400[pre_event], 400[post_event], 500
```

其余 50/150/250 等时间点仅保存训练期轻量指标，不伪造 NLL。以后如确实有研究需要，可以从原始 Checkpoint **追加独立 Analysis Run**，但不得在看到结果后将新增时间点重新称作预注册检验点。

**Step 400 必须按具体 Checkpoint `(step, event_revision, checkpoint_sha256)` 选取，不允许只用 `step=400` 的 `latest` 代替前事件模型。**

### 4.4 Functional / Cell Specialization：离线、独立有预算

**建议首批重点时间点：**

```text
0, 200, 350, 400[pre_event], 400[post_event], 450, 500
```

每个高成本任务测量：

- 已冻结的 Initial-State Probe 下，Cell 对的条件优势 `D_ij(G)=E[L(j)-L(i)|G]`，与原 R3 `fixed_suffix` 数学及方向约定一致。
- Cell 完整 Rank、Router Top Cell、Oracle-best Cell、paired regret、group conditional log-preference。
- STOP/CTX 与合法 Cell 的相对 Q margin；在预定 Calibration Probe 上执行基于共同随机数的 action-type 校准。
- 少量预定义 Pair `(0,1),(2,3),(4,5)` 的 Crossover、正确的故事簇 Bootstrap、样本数与置信区间；空样本/不可辨识必须显式报告。
- 从原训练 Journal 读取梯度源信息，不为了诊断对训练模型进行**新的 Training backward**。
- H2/H3/H4 均保持探索性；`CAUSAL_FEEDBACK_NOT_TESTED`、`MULTISEED_NOT_TESTED` 必须维持。

**不要**把原来的 `mc_samples=8` 直接减半。任何调整 MC 采样/Probe 个数、步骤数量、显著性阈值，必须先完成**只看计算成本与可靠性的独立诊断协议评估**，书面注册新的分析版本；不得静默改变原科学定义。第一阶段优先通过**延期、复用和批处理**提效。

### 4.5 可选增量层

完成首批离线 Quality 与 Functional 后，若仍需要更精细定位退化时间，可用新增 `analysis_extension` 对 Step 50/150/250 增补同一方案的诊断，并标记 `EXPLORATORY_EXTENSION_AFTER_INITIAL_RUN`。不得因此升级显著性结论或隐瞒事后扩展。

---

## 5. 训练 Runner 的具体改造要求

目标文件：

```text
r3b/implementation/clm001_r3b/runner.py
r3b/implementation/clm001_r3b/instrumentation.py
r3b/implementation/clm001_r3b/coordinator.py
r3b/implementation/clm001_r3b/common.py
```

### 5.1 用显式调度计划取代 `observe -> full snapshot`

建议增加模块（名称可在实现中优化，但需保持能力）：

```text
r3b/implementation/clm001_r3b/diagnostic_schedule.py
r3b/config/diagnostic-execution-v1.json
```

职责：

- 对 `(step, phase, event_revision)` 计算应执行的 `TRAINING_TELEMETRY` 与应**仅登记**的待分析任务。
- 默认实际运行时所有昂贵的 Quality/Functional 调用都返回 `DEFERRED` 任务清单，**不进行模型前向**。
- 原始事件执行流程不归它管理；不可拦截或跳过原始 `events.process()`。
- 输出计划必须可哈希、可冻结、可扩展而不会无声修改已有运行身份。

示意（**接口草案，非现有 API**）：

```python
# Trainer / original event semantics unchanged.
if safe_boundary:
    cp.save(folder, trainer, nursery, data_rng, identity)
    write_training_telemetry(folder, trainer, checkpoint_ref=read(latest))
    enqueue_deferred_analysis_manifest(folder, step, phase, checkpoint_ref)
    # local checkpoint + deferred manifest -> frozen handoff -> verified HF ACK
    deliver(phase)

if pending(c, step):
    # Original synchronous semantic admission/evaluation/promotion remains exactly here.
    events.process(...)
```

实际代码必须保持 R3b 已有的 `pre_event` / 每个事件提交后的发布顺序；伪代码不能替代对 `runner.segment` 的逐路径审计。

### 5.2 初始 Frozen Bank 不能删除

`instrumentation.banks(...)` 创建 Calibration/Development 两套 Frozen Probe。虽然建库本身也有成本，但不应为了省时而删除或在下一会话重建。要求：

- Step 0 首次创建并持久化，绑定训练模型与原始数据身份。
- 恢复时只校验、读取**原字节**，不得重新抽样。
- `FROZEN_PROBE_MISSING` / 校验不符时 fail closed。
- 单独记录 Step 0 Bank capture 开销；不得把它当作“被删除的诊断”。

### 5.3 事件提交/故障隔离

- 对每个事件版本，**先原始 checkpoint commit，再把包含事件状态的不可变 handoff 提交到 HF 并 ACK**。
- HF 失败时仍按现有机制安全停止；保留上一个有效 HF Index。
- Event Revision 和 SHA 在远端索引中保持严格单调，防止旧 Worker 回退。
- 不允许因新离线任务失败导致 `STORAGE_BLOCKED`（除非连基本训练 Checkpoint 都无法提交）；应显示 `ANALYSIS_BLOCKED`，与训练 Storage 状态分离。

### 5.4 不重复发布相同状态

当前 Runner 在 Segment 起始也会 `observe(post_event)`、`deliver(post_event)`。对于已经持久化过且没有新内容的 Checkpoint，尽量将交付实现为**内容一致时的幂等验证/指针复用**；不要因启动新 Kaggle Session 对相同 checkpoint 重新执行长诊断或产生重复新 Commit。

---

## 6. 独立历史分析入口与任务协议

目标新增模块：

```text
r3b/implementation/clm001_r3b/analysis_runner.py
r3b/implementation/clm001_r3b/analysis_store.py
r3b/implementation/clm001_r3b/diagnostic_schedule.py
```

扩展既有 `cli.py`，或添加 `r3b/analyze.py` 独立入口。优先保留现有 `evaluate` 的向后兼容行为。

### 6.1 建议 CLI（需要 Codex 实现，**不是现成参数**）

```bash
# 枚举可选检查点，必须给出 Step/Event Revision/完整哈希。
python r3b/run.py --mode list-checkpoints \
  --checkpoint-ref checkpoint-index.json --output /kaggle/working/r3b-analysis

# 单个历史模型的高成本功能诊断。
python r3b/run.py --mode analyze-checkpoint \
  --checkpoint-ref checkpoint-index.json \
  --arm QC-R3b --step 400 --event-revision 0 \
  --stage functional \
  --analysis-plan r3b/config/diagnostic-execution-v1.json \
  --output /kaggle/working/r3b-analysis

# 从同一保存包展开全部预注册 Quality 任务；按预算分片执行。
python r3b/run.py --mode analyze-batch \
  --checkpoint-ref checkpoint-index.json \
  --stage quality --max-tasks 3 \
  --analysis-plan r3b/config/diagnostic-execution-v1.json \
  --output /kaggle/working/r3b-analysis

# 仅按已完成、已校验的观测汇总，不再运行模型。
python r3b/run.py --mode aggregate-analysis \
  --analysis-index <PINNED_ANALYSIS_INDEX> \
  --output /kaggle/working/r3b-report
```

命令行详细参数须结合现有 `HFCheckpointStore.fetch_verified_checkpoint`、`MountedInputStore.select` 和 Kaggle Notebook 约束落实；禁止假设 `--checkpoint-ref` 能直接接受任意 URL。现有安全源允许的可信 GitHub index 规则必须保留。

### 6.2 分析对象契约

每个任务的逻辑主键：

```text
(training_run_id, arm, training_source_sha,
 exact_checkpoint_sha256, step, event_revision, phase,
 analysis_protocol_sha256, analysis_source_sha, stage, policy_set,
 role, frozen_probe_sha256, dataset_window_sha256)
```

逻辑任务主键 hash 后作为不可变目录名，禁止只用 `arm+step`，因为 Step 400 会存在多个 Event Revision，且以后可能升级分析算法。

**每个任务**包含 `PLANNED / RUNNING / COMPLETE / FAILED / BLOCKED / SKIPPED_BUDGET` 之一；只有产物哈希验证通过才标为 COMPLETE。丢失、污染、网络失败必须出具 machine-readable failure_code。

### 6.3 只读评估的操作准则

- 输入只能是可信 HF 私有仓库中经完整 SHA256 验证的 Checkpoint + 资产/Probe Bundle。
- 与原模型和对应 Event Revision 精确加载，验证 `source_sha, run_id, config_sha, assets_sha, tokenizer_sha, protocol_sha`。
- 输入 `.pt` 使用 PyTorch 反序列化时必须来自可信、验证过的项目私有来源；**SHA256 仅保证字节未变，不能为恶意 Pickle 提供来源可信性**。
- 不调用 `Trainer.update()`、`optimizer.step()`、Shadow Proposal/Admission、Promotion 或会改变训练语义的函数。
- 包裹相应 RNG 隔离（Python / NumPy / Torch CPU/CUDA / 模块 train/eval 状态、Trainer streams）；任务前后检查全状态 fingerprint、Optimizer hash、RNG、Probe hash 及原 Checkpoint/Manifest 未变化。
- 必须区分 `fixed_suffix` 中用于测量导数的局部 `autograd` 与真正的 Trainer 参数更新；**不能为了“只读”而对所有 Functional 任务套用 `torch.no_grad()`，以免破坏需要梯度的科学定义**。
- 同一分析任务完成后可释放内存并继续下一个 Step；不可使上一个模型的 CUDA state 污染下一个评估。

### 6.4 数据角色及配对一致性

- Calibration 用于方法审计/运行预算选择；Development 用于已预注册的探索性质量报告；Confirmation 封闭。
- Greedy/Uniform 主对照必须在同一 Story/window/seed/成本规范上配对；保留原 `paired_identity` 与用于 cross-policy 匹配的共同 pairing key。
- Cell 功能对照的组别（目前 `initial input token ID % 2`）、Cell 对、样本缺失规则和 Bootstrap 以冻结协议为准。
- 固定 Probe 与 Live-recomputed Probe 的时序证据不得混合。
- 后处理使用同一模型状态、固定随机流与有明确 revision 的 Registry；杜绝读取“最新权重”覆盖过去时点。

### 6.5 按任务逐步持久化

离线分析不能把“整个 Batch 完成”当作唯一持久化点。每项任务：

1. 精确读取并校验远端 Checkpoint。
2. 执行对应诊断，写入本地临时目录。
3. 计算内容哈希、观测样本数、资源消耗。
4. **提交到独立 HF Analysis Prefix**，取得精确 HF Revision；完整性检查通过后才更新 `analysis-index.json`。
5. 需要时将该任务的**白名单聚合数据**回传 GitHub。
6. Session 中断后跳过已经 COMPLETE 且哈希一致的任务；仅重试未完成任务。

**不能**让分析结果写回既有 Training `runs/<run-key>/latest.json` 或覆盖训练 bundle。若在同一 HF Model Repo 里写入 `analysis/...`，必须使用独立路径、CAS 并确保历史 Training revision 和 per-arm pointers 绝不回退。也允许使用另一个私有分析仓库，但不能未经同意擅自创建一堆仓库或改变私有可见性。

---

## 7. 研究分析产物与聚合兼容性

### 7.1 建议存储布局（逻辑形式）

```text
Hugging Face: libingjiang/minicells-research-checkpoints [PRIVATE]
├── objects/sha256/...                  # 原有内容寻址对象
├── runs/<training-key>/...             # 原有完整训练 bundle；不可改写
└── analyses/<analysis-version>/<training-run>/
    ├── manifests/<task-key>.json
    ├── private/<task-key>/...           # per-story、固定后缀、完整功能探针结果
    └── index.json                      # 独立分析索引；不可回退

GitHub (metadata-only):
research/projects/cellular-language-models/experiments/
  clm-001-qfirst-integrated/r3b/artifacts/
    <training-source-sha>/<result-run-key>/
      experiment-manifest.json
      checkpoint-index.json             # 训练指针，未改
      training-resource-ledger.json
      analysis-index.json               # 新增：每个任务的精确 HF revision
      metrics-summary.json
      routing-longitudinal-summary.json
      specialization-summary.json
      gradient-audit-summary.json
      statistical-contrast.json
      diagnostic-resource-ledger.json
      failure-index.json (按需)
```

不要为了复用原 Publisher 把巨大的原始 `snapshot.json`、`nll[]/cost[]/groups[]`、Optimizer、RNG 或私有 Probe 推到公开 GitHub。

### 7.2 分析任务 Manifest 示例

下面是**期望 schema 样例，不代表仓库已有字段**：

```json
{
  "schema": "minicells.r3b.analysis-task.v1",
  "task_status": "COMPLETE",
  "experiment_id": "CLM-001-R3b",
  "training_run_id": "<RUN_ID>",
  "arm": "QC-R3b",
  "checkpoint": {
    "training_source_sha": "<40_HEX>",
    "hf_repo": "libingjiang/minicells-research-checkpoints",
    "hf_revision": "<IMMUTABLE_COMMIT_SHA>",
    "step": 400,
    "event_revision": 0,
    "phase": "pre_event",
    "checkpoint_sha256": "<SHA256>"
  },
  "analysis": {
    "analysis_mode": "POSTHOC_FROM_IMMUTABLE_CHECKPOINT",
    "analysis_source_sha": "<40_HEX>",
    "analysis_protocol_sha256": "<SHA256>",
    "stage": "functional",
    "role": "development",
    "policies": [],
    "frozen_probe_sha256": "<SHA256>",
    "window_identity_sha256": "<SHA256>",
    "optimizer_updates": 0,
    "sample_count": 0,
    "missing_count": 0,
    "elapsed_seconds": 0
  },
  "private_result": {
    "hf_revision": "<ANALYSIS_COMMIT_SHA>",
    "hf_path": "analyses/<...>/private/<TASK_ID>/result.json",
    "sha256": "<SHA256>"
  },
  "scientific_decision": false,
  "confirmation": false
}
```

字段 `elapsed_seconds: 0` 只是格式占位，真实运行中必须使用测量值；`sample_count=0` 必须导致适当的 `INSUFFICIENT_COVERAGE`，不能被误认为合格研究证据。

### 7.3 修改 `analysis.aggregate()` 的关键点

目前聚合代码几乎默认所有历史 `snapshot-*.json` 含四种策略和 Functional 字段。改造后必须：

- 从**经过验证的 Analysis Task Manifest 列表**读取不同阶段的数据，不依赖目录 glob 顺序。
- 缺失某策略/某 Step/某 Event Revision 时使用明确 `NOT_EVALUATED / INCOMPLETE`，而非 `KeyError`、零值或者用其他时间点替代。
- Primary DID 必须要求**同 Step、同 Phase、同事件语义、三个 Arm 的 Greedy/Uniform 都已完成并通过配对身份检查**，否则输出 `PARTIAL`/`INCOMPARABLE`。
- 不可跨 Arm 把 Step400 `pre_event` 和 `post_event` 混为一个样本。
- Quality+Functional 的发布状态要和 Training 的 `DURABLE_COMPLETE` **分开**。若模型完成而分析只完成 4/21 项，应正确显示 `TRAINING_COMPLETE_ANALYSIS_PARTIAL`。
- 在未完成 500 的运行中，不能把 `500` 时点的 `PARTIAL` 占位当作最终科学系统判定。
- 图表允许不完整时间序列（缺失点断线并标注），不能做数值插值冒充已观察数据；绘图不调用模型评估。
- 每条公开聚合指标包含样本量（尤其独立 Story 数）和比较范围；单 Seed 永远不能冒充多 Seed 重复。

### 7.4 与当前 GitHub 发布器的兼容性

保留 `publication.py` 已有文件白名单、单 JSON 容量上限、拒绝敏感 payload 逻辑；仅在必要时添加 `analysis-index.json` / `diagnostic-resource-ledger.json` 等白名单文件。

独立 `analysis_source_sha` 和 namespace，避免后处理 GitHub Push 覆盖训练结果分支；publish-only 失败只能重新发布聚合，不触发训练或重新计算已经落地的私有数据。

---

## 8. 性能优化：先剖析，再有控制地调整

### 8.1 必须测量的分项耗时

对每个 Arm、Step、阶段记录：

| 分类 | 测量指标 | 说明 |
|---|---|---|
| `training_update_seconds` | P50/P90/P99、总时长 | GPU 同步边界正确，区分 host/wall |
| `probe_bank_capture_seconds` | Calibration/Development Bank 各自耗时 | 只建一次 |
| `mandatory_event_seconds` | propose/pre-admission/admit/promote 私有更新、正式判定 | **不能删除** |
| `training_telemetry_seconds` | Journal/索引/轻量计算 | 新增开销应很低 |
| `checkpoint_serialize_seconds` | 模型/Optimizer/Shadow 打包 | 包含事件变大后的情形 |
| `hf_commit_seconds` | 每次上传、按新增对象字节统计 | `wire_bytes` 未计量须记 `UNKNOWN` |
| `hf_full_readback_seconds` | 最终完整回读 | 可能是尾部重成本 |
| `analysis_quality_seconds` | Greedy、Uniform、Sampled、Constant 分别计时 | Development 与 Calibration 严格分开 |
| `analysis_action_calibration_seconds` | STOP/CTX 共同随机数校准 | 不与质量 eval 合并 |
| `analysis_functional_seconds` | Cell fixed-suffix/Oracle/crossover | 逐子任务计时 |
| `analysis_aggregation_seconds` | Bootstrap、报表和绘图 | CPU 为主 |

把原 Pilot 的 156–163 秒拆出各函数占比；如果旧 Pilot 产物未提供精确子项计时，应写 `NOT_MEASURED`，**不要通过推测填入伪精确数字**。

### 8.2 优化优先级

**P0：零算法风险的执行时机调整**：把 Quality/Functional 从 `runner.segment` 同步路径移走，改为 HF Checkpoint 离线分析。这首先解决训练关键路径。

**P1：避免重复工作**：

- 同一 `checkpoint_sha + protocol_sha + policy + role + seed` 的评估如已通过验证，可直接复用结果。
- 同一 Checkpoint 与 Frozen Bank 只下载一次，在同一分析进程的多个任务中共享只读数据（注意模型/随机流隔离）。
- Step 400 Pre/Post 必须是不同 Checkpoint，不能错误缓存合并。
- 复用算法已经产生的可证明等价的 `pre-admission` 评价时，必须比较**全部** Role/Model SHA、策略、数据窗口、评价 Seed/算法、数据分组与成本口径；无法证明完全等价就重新评估，不得靠 Step 相同就复用。

**P2：保持数值语义的计算优化**：

- 在**无需导数**的推理子路径中使用 `torch.inference_mode`/`no_grad`；在 `fixed_suffix` 梯度诊断中保留必要 autograd。
- 对能证明 Batch 不改变随机状态和硬路径决策的部分进行批量化；任何微小差异对统计定义的影响必须测试，而非简单宣称“数学等价”。
- 尽量将每次测量的 CUDA 同步合并到准确的阶段边界，避免频繁 `.item()`/CPU 取值强制同步。
- 对固定模型、相同训练已冻结的后缀计算使用内容寻址缓存；不得跨不同 Registry/Cell 状态误命中。
- 符合成本目标后再考虑 Monte Carlo 数量优化；**不先减少科学采样来迎合 2h 目标**。

### 8.3 预算拆分与执行目标

**训练关键路径**（默认要求）：

```text
T_training_makespan
  = max(T_Q + T_QS, T_QC) + scheduling/waiting
  + mandatory Shadow/event cost
  + initial frozen-bank capture
  + committed Checkpoint serialization
  + HF ACK / readback / retry
  + training telemetry
```

实际多 GPU 情况必须测总 wall time，不能对并行时间简单相加。短 Pilot 的 3 秒/step 只给出约 50 分钟更新基线；**在拿到 mandatory events、Bank、上传实际耗时之前不能认定两小时达标**。

**Offline Analysis**：独立记录 `T_quality + T_functional + T_action_calibration + T_bootstrap + T_remote_IO`。用户要求“整个研究 2 小时”的话，这个预算必须另行审计、可能需要更多资源或缩减预注册的研究任务，不能通过拆进第二个 Session 来宣称“总时间只有 2 小时”。

验收要求：

- 提供训练 makespan 的实测或有上下界的保守预算，目标 ≤7200 秒；若证据不足，状态为 `BUDGET_NOT_VERIFIED`。
- 另出分析队列的总耗时估计、首批任务数、优先级、延迟与可恢复性；允许跨独立 Session 实施。
- 不因超时擅自停止正在提交的事件、删掉必需的事件判定或调整科学冻结参数。

---

## 9. 恢复、并发与失败语义

### 9.1 训练侧

- 原有一主 Publisher、两 Worker 队列和不可变 handoff 不变。
- 训练更新与事件仍等待远端安全边界的 HF ACK；不能为性能擅自把远端确认变成完全无保护的 fire-and-forget。
- Step 200/350/500 继续使用执行控制参数，不能修改 `training.steps`。
- 若分析服务宕机，训练继续；若训练 Checkpoint 无法安全发布，训练按原机制停在安全边界。
- 不允许在新 Session 找不到原 Run 时静默从 Step 0 重训。

### 9.2 分析侧

- 分析 Session 可单独恢复；任务完成由独立索引控制。
- 超出资源预算可停止在任务边界，不修改模型状态，不丢失已提交诊断。
- 同一任务被多个 Worker 执行：使用 `CAS parent_commit` 或单 Publisher 防止覆盖；结果不同 SHA 时标记冲突，而不是随机选择一个。
- Analysis HF 发布失败后可以重试；不应使训练 Storage Index 从 `DURABLE_COMPLETE` 退回失败状态。
- 重复提交同一结果应幂等。不同 `analysis_protocol_sha` 必须生成**新任务身份**。
- `ANALYSIS_PARTIAL` 不能自动升级 H1/H2/H3 科学结论。

### 9.3 需要测试的异常场景

- 网络中断、Token 不可用、仓库私有权限变化、损坏的 HF SHA/远端返回不完整。
- Step 400 同 Step 多 Event Revision，明确选择 `0` 与完成后版本。
- 先完成 QC、尚未完成 Q/QS：聚合正确地标记 `PARTIAL`。
- Analysis 在独立新 Session 中从 HF 已完成任务恢复，不能再次跑已经验证完成的任务。
- 重复任务并发提交时索引 CAS 冲突处理，不得污染 Training latest。
- 无法访问训练数据或 Frozen Bank：直接 `MISSING_ASSET`/`FROZEN_PROBE_MISMATCH`，不可自行重新生成并冒充原样本。

---

## 10. 配置/Source SHA/Notebook 改造

### 10.1 建议新增配置

建议 `r3b/config/diagnostic-execution-v1.json`，表达如下含义（需 Codex 校验并最终生成严格 schema）：

```json
{
  "schema": "minicells.r3b.diagnostic-execution.v1",
  "mode": "DEFER_EXPENSIVE_ANALYSIS",
  "training": {
    "journal_telemetry_steps": [0,50,100,150,200,250,300,350,400,450,500],
    "preserve_every_original_event_revision": true,
    "mandatory_semantic_event_evaluation": "UNCHANGED",
    "save_frozen_probe_banks_at_step_zero": true
  },
  "posthoc": {
    "quality_primary_steps": [0,100,200,300,350,400,450,500],
    "quality_secondary_steps": [0,400,500],
    "quality_event_compare_at_400": ["pre_event","post_event"],
    "functional_steps": [0,200,350,400,450,500],
    "functional_event_compare_at_400": ["pre_event","post_event"],
    "primary_policies": ["greedy","uniform"],
    "secondary_policies": ["sampled","constant"],
    "required_roles": ["calibration","development"],
    "confirmation": "SEALED",
    "sampling": "INHERIT_FROZEN_R3B_PROTOCOL"
  },
  "analysis_publication": {
    "private_raw_results": "HUGGING_FACE",
    "public_aggregates": "GITHUB",
    "separate_from_training_latest": true
  }
}
```

注：表达了 Step 400 双阶段要求。真正的任务展开器必须从 Event Index 解析 pre-event revision0 和 post-event 最后提交 revision，而不是把 `400` 这个整数当成唯一状态。建议另加 Schema 检查，确保不出现未声明的 policy/role。

### 10.2 Notebook 分离

保持原 `r3b/notebook.ipynb` 作为训练/预检入口，新增：

```text
r3b/analysis-notebook.ipynb  # 只读评估与独立分析
```

两个 Notebook 均应：

- 明确显示 `TRAINING_SOURCE_SHA` 和 `ANALYSIS_SOURCE_SHA`、协议哈希及实际 HF repo/revision。
- 源码按**确切 commit SHA** 获取，不按可变分支 HEAD。
- 从 Kaggle Secrets 获取 `HF_TOKEN`（分析若需 GitHub 回传再读取 `GITHUB_TOKEN`），禁止在 Notebook 中打印 Token。
- 分析 Notebook 不持有正式训练启动入口或授权凭证，不会意外运行 `trainer.update`。
- 不使用 Kaggle `Add Input → Notebook Output` 作为 Checkpoint 持久化方案。
- 使用 Kaggle 工作目录存放短期计算，跨会话权威存储为私有 HF。

### 10.3 源码冻结与实验身份回顾

本次优化会改变 `runner.py` 和分析代码，因此必须建立新源码 SHA，并明确区分：

- **新的正式 R3b Training Run**：在启动前固定优化后的 Training Source SHA、配置、Assets、Beta、分析计划并正式授权。
- **此前成功的 Pilot**：仍然属于它原来的源码/配置身份；不允许通过篡改旧报告将其当作新源码上的完整 GPU Pilot。
- 如变更被证明只是观测执行调度变化，可对受影响路径进行**最小针对性 GPU 验证**（新旧相同冻结模型状态/训练更新对比、HF 读取与至少一个快照任务），而不是无意义地重复整轮 Pilot。但任何新源码的正式发布授权必须记录这种差异验证及负责人审批。

---

## 11. 完整测试矩阵与门禁

### Gate 0：源码及科学语义不变（CPU）

- 原始 `r3/implementation`、`r3/config`、`r3/SPEC.md` 相对冻结 SHA 无变化。
- `Trainer.update`、原 `events.process` 文件哈希无变化。
- Q/QC/QS 的模型更新在诊断 `enabled/disabled/deferred` 三种设置下，**模型+Optimizer+RNG+data_position+Registry+Nursery+Event Cursor** 按现有精确度策略一致。
- 包含发生真实原事件的合成短程案例（100/200/400 可在 synthetic CPU fixture 中缩放事件 Step，但事件操作本身保持原语义）。

**验收**：PASS，否则 BLOCKED。

### Gate 1：Checkpoint 和冻结探针闭环（CPU+HF 模拟）

- 每个 Milestone、每个 Event Revision 都能恢复，连同对应冻结 Bank。
- Step 400 `revision=0` vs 最终 Revision 可区分，不能被 latest 覆盖。
- HF index、bundle、event cursor、SHA 完整。
- 分析请求不改变 Training `latest` 与运行 Hash。

**验收**：PASS，否则 BLOCKED。

### Gate 2：独立诊断的数值等价（最重要）

在**同一份合成/已保存可信 Checkpoint**上，比较旧 `instrumentation.snapshot()` 与新 Offline Quality/Functional 输出：

- 对相同 Role、Policy、Probe、Event Revision，NLL/Cost/Router Q Margin/Cell preference/Regret/Long-horizon gap 使用规定容差（明确 FP32 CPU/GPU 行为）比较。
- 故意给出不同 Event Revision、不同 Frozen Bank、不同 Analysis Plan，必须拒绝误配。
- 新评估执行前后训练参数、Optimizer、global+Trainer RNG、Nursery 以及输入文件 Hash 完全不变。
- **禁止**仅对照聚合平均值；需对少量相同故事/状态的**样本级私有观测**作准确对照，然后把完整原始数据留在 HF 或测试工作区。
- 对不同采样策略执行独立随机流和共同配对 key 校验。

**验收**：PASS 或数值偏差有书面、经审批的科学影响分析；不允许静默通过。

### Gate 3：断点重入、失败与安全（CPU）

- 离线任务中断再恢复，已完成任务不重复高成本计算。
- 私有 HF 分析文件和索引 CAS 冲突可处理；Train/Analysis 两份最新指针互不干扰。
- GitHub 仅有 aggregate JSON，包含必要审计哈希、样本数和失败状态，无敏感原始数组/`.pt`。
- 错误身份、损坏 SHA、缺少冻结 Bank、任意 GitHub URL、重复任务键碰撞均 fail closed。
- 不完整分析只标 `PARTIAL`、图表缺失点正确显示。

**验收**：PASS，否则 BLOCKED。

### Gate 4：针对性性能验收（真实受控执行；不重复既有完整 Pilot）

采用已完成 Pilot 的可信 HF Checkpoint，优先复用已有数据。验证：

1. 在当前 Kaggle/GPU 环境只读读取并恢复该 Pilot 权重/Registry/Bank。
2. **仅**执行具有上限的 Quality/Action/Functional 子任务计时，区分 Calibration 与 Development；如需额外 GPU 时间，实施前批准明确的最大任务数量或运行预算。
3. 若用于生产训练源码已更改，额外做最小训练无扰动对照（例如 1–3 Step，并保存为新的**工程测试 Run**，不视为正式 R3b 科学样本）。
4. 形成按真实体量评估的训练关键路径资源账单：更新/银行/事件/Checkpoint/HF/Telemetry，尤其包含 Step 400 Admission 的不可延期成本。
5. 审核后给出 `≤2h`、`>2h` 或 `NOT_VERIFIED` 结论，附测量依据和安全系数。

**验收**：实际真实环境证据齐备方可标 `PERFORMANCE_GATE_PASS`；若两小时目标尚不满足，输出下一层性能改进建议而不是修改原科学算法。

### Gate 5：科学协议冻结与启动授权

- 审核者接受新诊断执行计划、需要的主要 Quality/Functional Step 和统计定义。
- Runtime、Assets、Tokenizer、Pilot、训练 Source SHA、分析 Source SHA、HF repo 与仓库权限、事件语义均已绑定并锁定。
- 只有明确授权才能将正式 500-Step 训练开关升级；Codex **不得自行开启**。

---

## 12. 实施顺序（建议拆分提交与 PR）

### Phase A — Audit / Baseline（不启动 GPU）

1. 拉取当前主分支并记录 HEAD，审计 PR #263 的后续更改。
2. 将已成功 Pilot 的 HF revision、GitHub 结果 commit、三个 Arm 的资源明细录入只读 `baseline-evidence.json`；缺失细项标 `NOT_MEASURED`。
3. 逐函数标识 `Training Semantic / Telemetry / Posthoc`，特别核实所有 `events.process` 路径。
4. 新增独立的 Diagnostic Execution Protocol 与 Schema，出版本化 SHA。

**交付**：`AUDIT.md`、`baseline-evidence.json`、`diagnostic-execution-v1.json`。

### Phase B — Train-side deferral（CPU）

1. 修改 `runner.segment` 的观测调度，保留原事件、Bank、Checkpoint、HF ACK。
2. 训练侧只记录 Journal-based Telemetry 和不可变待分析任务清单。
3. 对照开启/关闭新采集模式完成无扰动测试。
4. 明确 train-side 的“低成本”实际范围，避免隐藏 `router_rows`/Monte Carlo 等昂贵操作。

**交付**：代码、CPU 测试、完整训练状态比对 JSON。

### Phase C — Offline Analysis（CPU + HF 模拟）

1. 复用 R3b 历史选择器、Frozen Bank 和已有 Instrumentation 数学，新增质量/功能/校准单独入口。
2. 设计独立不可变任务键、Hash Manifest、Analysis HF 持久化、独立索引。
3. 扩展 `analysis.aggregate()` 和 `plots.render`，处理稀疏、延期、不完整数据。
4. 扩展 GitHub 元数据 Publisher，训练/分析 namespace 隔离。
5. 完成旧/新数值等价与失败模式测试。

**交付**：`analysis_runner.py`、`analysis_store.py`、`analysis-notebook.ipynb`、单测和示例 Artifact。

### Phase D — 性能剖析及有限验证

1. 不重复已成功的完整 Pilot；优先使用 HF 已保存的权威 Checkpoint。
2. 真实测出至少一组 Quality/Functional 的关键成本分项，并对现有 160 秒快照开销作可证实归因。
3. 只在必要时做新版本的微型 GPU 无扰动验证；必须使用单独工程身份。
4. 给出训练 makespan 与离线分析时间的**分别预算**，包含保守误差。

**交付**：`resource-budget.json`、`performance-report.md`、HF/Session 证据链接。

### Phase E — Release preparation（不自动启动正式实验）

1. 固定新的源码、方案、Schema、源数据和观察任务身份。
2. 完成 CI，独立 PR；提交 Codex 成果和 GitHub Actions 证据。
3. 形成是否批准下一次正式训练的 Go/No-Go 建议；Authorization 继续 `false`。

**交付**：全部 Commit SHA、测试日志、性能表、Known Limits、解锁正式训练所需签字事项。

---

## 13. 建议新增/修改的仓库文件

```text
research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/
├── DIAGNOSTIC_EXECUTION_OPTIMIZATION.md      # 本文或与本文等价的仓库版
├── config/
│   └── diagnostic-execution-v1.json          # 经审查的采集与离线任务计划
├── implementation/clm001_r3b/
│   ├── diagnostic_schedule.py                # 新增
│   ├── analysis_runner.py                    # 新增
│   ├── analysis_store.py                     # 新增/复用已有私有 HF 底层
│   ├── cli.py                                # 扩展模式
│   ├── runner.py                             # 移除同步重诊断，不动算法/事件
│   ├── instrumentation.py                    # 拆分重任务，复用原数学
│   ├── analysis.py                           # 新 Manifest 驱动汇总
│   ├── plots.py                              # 稀疏/部分结果绘图
│   ├── publication.py                        # 双通道、独立索引
│   ├── coordinator.py                        # 如必要仅增加幂等性/资源计时
│   └── pilot.py                              # 原 Pilot 不重跑；必要时只读剖析入口
├── analysis-notebook.ipynb                    # 新增，独立诊断
├── notebook.ipynb                             # 原训练入口保持/最小更新
└── reports/
    └── diagnostic-optimization/              # 运行证据和资源估算

tests/research/clm001_r3b/
├── test_diagnostic_schedule.py
├── test_deferred_training_equivalence.py
├── test_posthoc_numerical_equivalence.py
├── test_analysis_checkpoint_identity.py
├── test_analysis_resume_publish.py
├── test_partial_aggregate.py
└── test_mandatory_admission_unchanged.py

.github/workflows/clm001-r3b-cpu.yml           # 扩展 CI 检查
```

**实施要求**：优先复用现有函数、同一数学与快照封装；不要平行复制 R3 模型或写一套结果定义稍有不同的“快速评估器”。如某个文件名与项目既有约定冲突，可以改变文件名，但不能省略功能或测试。

---

## 14. 主要风险与避免方式

| 风险 | 影响 | 必须采取的措施 |
|---|---|---|
| 把 Admission 判定误删或延期 | **训练结果和科学问题改变** | 逐调用路径审计；事件前后状态精确对照 |
| 将 Frozen 初始 hidden state 宣称为当前在线状态 | **虚假纵向结论** | 固定/重算 Probe 两套标签，分开统计 |
| 新分析代码改变训练 Source Identity | 恢复失败/伪造历史 | Training SHA / Analysis SHA 分离，明确 provenance |
| Step400 `latest` 覆盖 Pre-Event | 无法区分退化是否由 Promotion 触发 | 精确 Event Revision + Checkpoint SHA 定位 |
| 分析结果写回 Training latest | 训练状态污染和回滚 | 独立 Analysis Prefix、单 Publisher/CAS |
| 以低样本试运行 NLL 替代正式 Development | 科学结论不可比 | 主要 Development 数据/窗口不变；小样本仅成本基准 |
| 先降 MC 数再申报局部一致性“通过” | 统计辨识率变化 | 先保留原样本定义；变化另开分析版本 |
| 后处理失败导致训练标 FAILED | 结果状态混淆 | `TRAINING_DURABLE` 与 `ANALYSIS_PARTIAL` 独立 |
| GitHub 误上传 per-story 原始数组 | 泄露/仓库膨胀 | 白名单、深层敏感数据拒绝、2 MB 文件上限 |
| 因异步改造跳过 HF ACK | 中断时丢失训练恢复状态 | 保留远端持久化门禁，不做无保护后台上传 |
| 只优化训练 wall time，宣称所有研究 <2h | 预算误导 | 分开列训练和分析时长，报告完整成本 |
| 大量新增离线任务缺少预算 | 分析仍可能无限耗时 | 任务优先级、最大任务量、断点续跑与资源账本 |

---

## 15. 验收报告应当长什么样

Codex 最终必须提交一份内容可核查的 `reports/diagnostic-optimization/acceptance.json`，至少包括：

```json
{
  "schema": "minicells.r3b.deo-acceptance.v1",
  "implementation_source_sha": "<NEW_SHA>",
  "historical_r3_unchanged": true,
  "training_semantics_equivalence": "PASS_OR_BLOCKED",
  "mandatory_event_semantics": "PASS_OR_BLOCKED",
  "training_checkpoint_integrity": "PASS_OR_BLOCKED",
  "posthoc_numeric_equivalence": "PASS_OR_BLOCKED",
  "training_rng_optimizer_identity": "PASS_OR_BLOCKED",
  "separate_hf_analysis_index": "PASS_OR_NOT_VERIFIED",
  "github_metadata_only": "PASS_OR_NOT_VERIFIED",
  "training_critical_path_budget_seconds": null,
  "training_budget_status": "NOT_VERIFIED",
  "analysis_total_budget_seconds": null,
  "analysis_budget_status": "NOT_VERIFIED",
  "prior_successful_pilot_repeated": false,
  "formal_500_step_training_authorized": false,
  "scientific_decision": false,
  "ci_url": null,
  "hf_evidence_revision": null,
  "remaining_blockers": []
}
```

**占位字段不可硬编码成 PASS**。没有真实计时或 HF 回读证据时写 `NOT_VERIFIED`。日志与 CI 报告必须可对应提交 SHA；对实际测试与模拟测试分别标记。

同时提交一份面向人阅读的 `PERFORMANCE_REVIEW.md`，包含**现状 7.24h 估算的来源、实际新测 Training critical path 与 Analysis cost、比原 R3b 运行前同步方案省掉的计算、仍不可延期的 Admission/Event 成本、剩余风险**。

---

## 16. 交给 Codex 的最终实施指令（可直接复制）

> 请在 `ArcheLabs/mini-cells` 基于合并 PR #263 的最新 `main` 新建独立工作分支，执行《MiniCells — CLM-001-R3b Diagnostic Execution Optimization》全部工程规范。先确认当前源码与已保存 Pilot 的真实 HF/GitHub 证据，不要重复已经成功的完整 Pilot。
>
> 
> 本次只优化 **R3b 诊断执行架构**：从训练 Runner 同步路径移走昂贵的 Development/Functional/STOP-CTX 解释性评估，保留原始 R3 `Trainer.update()` 和所有 `events.process()` 必需的 Shadow Admission 判定与事件 Checkpoint；在训练期间只保留低开销 Journal/Checkpoint Telemetry 和冻结 Bank。所有关键 Step/Event Revision 仍由单一 Publisher 经私有 Hugging Face 可靠提交和 ACK。
>
> 
> 新增从精确 HF Checkpoint 读取的只读、可恢复的离线 Quality/Functional 纵向分析入口，保留 Greedy/Uniform 主 DID、指定时间点的 Secondary 策略、Cell Conditional Advantage、STOP/CTX、冻结 Probe 与 Bootstrap 统计定义。使用独立 Analysis Task Index/Revision，不修改 Training Latest；私有原始分析文件留在 HF，公开 GitHub 仅回传聚合及可验证索引。
>
> 
> 强制实施训练无扰动比对、Step400 事件版选择、离线数值等价、跨 Session 幂等恢复、存储失败及隐私发布测试。真实资源测试优先复用已保存 Pilot 的权威 Checkpoint；新 GPU 工作只在必要时作为有明确上限的**工程性微型验证**，不能把它当作正式 R3b 实验，不得运行新的三组完整 Pilot 或 500-Step。
>
> 
> 输出 GitHub Commit/PR、CI、两类（Training 与 Analysis）资源预算、实际测量证据、完整 Go/No-Go 清单。所有真实验收证据缺失时标记 `NOT_VERIFIED`；默认 `formal_500_step_authorized=false`，不要自行升级。

---

## 17. 这次工程工作的完成定义

这次任务**不是**“让 R3b 看上去从 7.24h 变成 50min”。真正的 Done 定义是：

1. 三个 Arm 的**训练算法和历史事件判定未变化**。
2. 训练安全边界仍能够在 Kaggle→HF→新 Session 下恢复，分析任务在训练之后独立运行。
3. 每个被测模型都有确切的 Step、Event Revision、Checkpoint SHA、HF revision；分析版本与训练版本可分别追踪。
4. 标准的 Greedy/Uniform、STOP/CTX、Cell 功能评价和故事簇推断仍然可执行、可比较、可复核。
5. 训练关键路径和完整诊断各自有**独立、真实、带误差说明的资源预算**；如未达两小时，不允许假报 PASS。
6. 全部安全、数值与 CI 门禁通过或明确报告阻塞；新的正式训练仍等待用户明确批准。

**建议批准本工程实施；暂不批准正式 500-Step 训练。**

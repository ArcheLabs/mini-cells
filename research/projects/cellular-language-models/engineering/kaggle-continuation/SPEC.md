# MiniCells — Kaggle 实验持久化、恢复与独立评估工程实施规范

**版本：** Engineering Gate v1.0  
**日期：** 2026-10-16  
**交付对象：** Codex / MiniCells 开发者  
**目标仓库：** `ArcheLabs/mini-cells`  
**参照实现：** CLM-001-R3，源提交 `52b1950b23debbcd3f808c10eb6e0d3ea2d317c6`  
**实施性质：** 仅工程基础设施；不启动新科学实验，不重新训练 R3，不改动既有 R3 的实验结果或科学结论。

## 1. 任务结论与优先级

先把 Kaggle 训练改造成**可分段、可恢复、可独立评估、可验证持久化**的实验系统，然后再开展 R3b。暂时不依赖 Google Cloud、外部对象存储，也不要求下载旧 R3 Checkpoint。

实施顺序：

1. **P0：跨 Kaggle Session 恢复**。从已保存 Notebook 版本的 Output（作为下一 Notebook 的只读 Input）恢复完整实验状态。
2. **P0：安全的分段训练**。同一冻结实验可在 0→200→350→500 Step 的独立保存版本中继续，不能改变实验身份或重复训练。
3. **P0：实际跨会话验收**。不能仅靠本地目录复制测试，必须完成一次真实的 Kaggle Save & Run All → Add Input → Resume。
4. **P1：历史 Checkpoint 独立评估与 Fork**。允许选择任意有效历史 Step，只读评估；Fork 必须创建独立实验身份。
5. **P1：纵向诊断工程**。新实验预先冻结诊断计划，保存固定探针、路由及 Cell 专业化轨迹。

**禁止：** 在恢复功能通过真实 Kaggle 验收前，执行完整 R3b GPU 训练或修改信用分配、STOP/CTX、Connection 的科学算法。

## 2. 现状与确定的问题（基于 R3 源码）

现有组件可复用：

- `r3/implementation/clm001_r3/checkpoint.py`：保存模型、Optimizer、数据位置、RNG、Registry、Shadow Candidate 等状态；使用身份与 SHA256 验证；提供 `export_portable()`。
- `r3/implementation/clm001_r3/runner.py`：只在当前 `output/latest.json` 存在时调用 `restore()`。
- `r3/implementation/clm001_r3/automatic.py`：一次性 CPU Gate → GPU Pilot → 对齐检查 → 冻结 beta → 三 Arm 训练 → 聚合及发布。
- `r3/implementation/clm001_r3/monitoring.py`：Frozen Probe Bank、Snapshot、Longitudinal 诊断。
- `r3/notebook.ipynb`：每次从 GitHub 拉取源码，在新的 `/kaggle/working/` 路径执行 `--mode auto`。

已发现的工程缺口：

1. Kaggle 新 Session 不继承上一 Session 的 `/kaggle/working`；旧 Output 不会自动成为新 `output`。
2. 现有 `restore()` 只能按照本地 `latest.json` 恢复，缺少显式远端输入和任意历史 Step 选择。
3. `--mode auto` 没有明确的分段边界和 `PAUSED` 状态；预检、训练和发布耦合。
4. `portable/` 仅导出最新 Checkpoint，不能代替历史 Checkpoint 库；各 Arm 内重复复制 Assets 会膨胀 Output。
5. `r3/config/proposed.json` 中 `evaluation.snapshots=[]` 且 `periodic_diagnostics_enabled=false`，造成纵向证据缺失。
6. Notebook 按可移动分支获取源码，如果分支前进，旧 Checkpoint 的 Source SHA 校验可能失败。
7. 缺少“上一次结果实际已成为下一次可读取 Input”的端到端门禁。

**注意：** 保存到 `/kaggle/working` 并不等于运行中已有远端持久化。Kaggle 的 Notebook 版本成功保存后，Output 可作为之后 Notebook 的 Input；但当前运行中途异常、未成功产生可访问版本时，不能保证恢复到最后一个本地 Step。没有外部存储时，设计必须承认这个限制。

## 3. 工程架构与职责边界

```text
GitHub（冻结源码、协议、配置；汇总与公开结果）
    │
    ▼
Kaggle Notebook：Bootstrap → Preflight → Train Segment
    │                                  │
    │                                  ▼
    │                          /kaggle/working/（可保存输出）
    │                                  │
    │                           Save & Run All 成功
    │                                  │
    ▼                                  ▼
下一 Kaggle Notebook ← Add Input ← 上一版本的 Notebook Output
    │                     （只读 /kaggle/input/...）
    ├── Resume：完整状态，继续同一实验
    ├── Evaluate：只读历史 Checkpoint，不训练
    └── Fork：创建新实验身份后继续训练
```

- **GitHub**：源码、协议、配置、允许发布的汇总报告；不存 `.pt`、原始训练数据、逐条 Oracle 标签、凭据。
- **Kaggle Working**：本次运行产出的可保存 Checkpoint 和诊断；保证完成后输出尺寸符合 Kaggle 限制。
- **Kaggle Input**：上一已保存版本的只读快照；绝不原地修改，不将它当作正在运行的持久化磁盘。
- **未来对象存储**：预留 `CheckpointStore` 抽象，但本次只实现 `LocalOutputStore + MountedInputStore`。不得为了未来 GCS 引入必须配置的云端依赖。

## 4. 标准目录和续跑包

建议保持既有模型内部保存格式，新增**实验级恢复包**，而不是在每个 Arm 反复复制 Assets。

```text
/kaggle/working/minicells-run/<source-sha12>/
├── candidate-config.json
├── runtime-config.json                 # 真正冻结的 beta 和运行参数
├── experiment.lock.json                # source、protocol、config、assets、tokenizer hash
├── run-status.json                     # 状态及阶段信息
├── continuation.json                   # 已发布的恢复入口（见 §5）
├── assets/                             # 一份共享数据，含 manifest/tokenizer/npy/windows
├── arms/
│   ├── Q-R3b/
│   │   ├── identity.json
│   │   ├── latest.json
│   │   ├── checkpoint-step-000050.pt
│   │   ├── checkpoint-step-000050.sha256.json
│   │   ├── ...
│   │   ├── frozen-probe-bank.pt
│   │   ├── frozen-probe-bank.json
│   │   ├── snapshots/
│   │   └── diagnostics/
│   ├── QC-R3b/ ...
│   └── QS-R3b/ ...
├── evaluations/                        # 新评估报告，和训练历史分离
└── continuation-manifest.json          # 恢复包文件清单与哈希
```

`Q-R3b/QC-R3b/QS-R3b` 为示例：工程实现必须按运行配置的实际 Arm 名称生成，不能硬编码 R3b 的名称，也不能悄悄替换 R3 的已有实验身份。

**必须保留**：每个 Arm 的完整训练状态（不仅是权重）、每个有效 Checkpoint 的校验侧车文件、每个 Arm 的 `identity.json` 和固定探针、一次共享 Assets、冻结配置及协议身份、已完成的 Shadow Lifecycle / Event 状态。需要调试或复核的原始 `diagnostics/` 继续保留在私有产物中。

**体积约束**：Kaggle 官方文档说明 Notebook 可保存 `/kaggle/working` 输出上限约 20 GB。生成恢复包前统计文件尺寸，并保留安全余量（工程默认目标 ≤16 GiB，超过时失败并给出文件体积分解；该余量为本项目策略，不是 Kaggle 官方硬限制）。不得静默删除历史 Checkpoint、探针或证据来压缩大小。可在保留完整性和随机访问能力的前提下显式压缩或使用分层输出包。

## 5. Manifest、恢复状态和不可变性

新增一份可读、可校验的 `continuation.json`，最少包含：

```json
{
  "schema_version": 1,
  "experiment_id": "CLM-001-R3B-ENGINEERING-SMOKE",
  "run_id": "<stable-run-identifier>",
  "source_sha": "<full-git-sha>",
  "protocol_sha256": "<sha256>",
  "candidate_config_sha256": "<sha256>",
  "runtime_config_sha256": "<sha256>",
  "assets_manifest_sha256": "<sha256>",
  "tokenizer_sha256": "<sha256>",
  "status": "PAUSED",
  "arms": {
    "Q-R3b": {
      "last_committed_step": 200,
      "checkpoint": "arms/Q-R3b/checkpoint-step-000200.pt",
      "checkpoint_sha256": "<sha256>",
      "checkpoint_revision": 0,
      "probe_bank_sha256": "<sha256>",
      "event_cursor": "<validated-state-or-reference>"
    }
  },
  "files_manifest": "continuation-manifest.json",
  "complete": false
}
```

以上 JSON **仅为新增格式示意**，字段按实际可校验的数据结构落地，不得伪造 `event_cursor` 等值。

规则：

- 每个 `.pt` 文件路径不可变；哈希文件对应当前精确字节内容。
- 原子写入：Checkpoint → 校验文件 → 完成 Event/Snapshot 边界 → 文件总清单 → 最后提交 `continuation.json`。只有最后一步成功，恢复包才可声明 `READY/PAUSED`。
- 支持 Arm 进度不一致；分别记录最后一个可靠 Step，不要用单个全局 Step 猜测三 Arm 的进度。
- 同一 Step 存在事件版本时，使用清单中的**精确文件路径和版本**，不能简单 glob 取第一个或仅依赖数字 Step。
- 检查相对路径、目录穿越、符号链接、源身份、模型结构、Tokenizer、数据清单、探针 SHA、完整配置。
- 输入缺失、文件损坏或身份不匹配时 **fail closed**；不得退回随机初始化，也不得自动重建 Frozen Probe Bank。
- 旧版本输入必须只读，所有新文件写入当前 `/kaggle/working`；保留父版本的明确来源元数据。
- 对已有 `latest.json` 和原有 `.pt` 格式尽量向后兼容；不要更改 R3 历史数据和哈希。

## 6. CLI 命令与语义

在现有 `r3/run.py` / `clm001_r3.cli` 体系中抽象公共工程组件，或新建专门的实验运行入口；以下是**目标 CLI**，不是现有命令。

```bash
# 一：不训练，仅检查已挂载的恢复包
python run.py --mode verify-checkpoint \
  --checkpoint-input /kaggle/input/<previous-output>/minicells-run/<source-sha12>

# 二：从零开始训练，但当前版本只运行到 Step 200
python run.py --mode auto \
  --config config/frozen.json \
  --output /kaggle/working/minicells-run/<source-sha12> \
  --stop-after-step 200 \
  --require-resume false

# 三：从上一个已保存版本恢复，继续至 Step 350
python run.py --mode auto \
  --config config/frozen.json \
  --checkpoint-input /kaggle/input/<previous-output>/minicells-run/<source-sha12> \
  --output /kaggle/working/minicells-run/<source-sha12> \
  --stop-after-step 350 \
  --require-resume true

# 四：只读检查指定历史 Step，不更新任何模型/优化器
python run.py --mode evaluate-checkpoint \
  --checkpoint-input /kaggle/input/<previous-output>/minicells-run/<source-sha12> \
  --arm QC-R3b --step 200 \
  --evaluation-config config/evaluation.json \
  --output /kaggle/working/evaluations/diagnostic-001

# 五：Fork 新实验（先建立新 run_id，再允许更新训练状态）
python run.py --mode fork-experiment \
  --checkpoint-input /kaggle/input/<previous-output>/minicells-run/<source-sha12> \
  --arm QC-R3b --step 200 \
  --new-config config/fork.json \
  --output /kaggle/working/minicells-fork/<new-run-id>
```

要求：

- `--require-resume true` 且输入缺失：立即报错；**不能从零训练**。
- `--require-resume false` 且输入已存在：默认报错，避免意外重复或覆盖（必要时提供显式 `--mode verify-checkpoint`）。
- `--stop-after-step` 是当前会话的执行上界，不进入训练配置 Hash；冻结的总目标 `training.steps` 不能变。
- `stop-after-step <= committed_step`：不得重新训练；返回当前已满足上界的状态。
- 仅允许在 Step 的训练更新、指定 Event、Checkpoint 和本阶段必要快照全部处理并一致后正常停止。
- 返回码 0 + `status=PAUSED` 代表有意完成当前 Segment；不能当成最终 `COMPLETE`，也不能输出正式科学结论。
- `evaluate-checkpoint` 不接触 Optimizer.step，不更改输入 Checkpoint，不触发预检训练、不重新下载训练数据，评估结果必须单独命名。
- `fork-experiment` 记录 `parent_experiment_id`、`parent_run_id`、`parent_arm`、`parent_step`、`parent_checkpoint_sha256`、新 Source/Config/Protocol Hash；**禁止伪装成原实验 Resume**。

## 7. 冻结源码、预检、训练与发布的生命周期

把原来的 `auto` 拆成可持久化状态机：

```text
NEW → PREFLIGHTING → PREFLIGHT_PASSED → TRAINING
                              │                 │
                              │                 ├→ PAUSED → RESUMING → TRAINING
                              │                 │
                              │                 └→ COMPLETE → ANALYZED → PUBLISHED
                              └→ PREFLIGHT_FAILED
任何阶段 → FAILED（保留可校验的最后安全状态）
```

具体要求：

1. **首次运行**执行 CPU Gate、真实 GPU Pilot、Credit/Alignment Gate，得到运行时冻结配置（包括 beta）、锁文件及 Assets。
2. **Resume** 不重新运行可能改变实验身份的 Pilot/自动 beta 冻结；必须验证原始 Gate 证据和运行时配置后直接继续。
3. **Source Pinning**：第一次运行锁定完整 Git SHA；后续运行必须检出同一 Commit，禁止 `SOURCE_REF=branch-tip` 漂移。修改任何训练代码必须创建新实验或采用明确审查过的迁移流程。
4. **Shadow Events**：当前 R3 在 Step 100/200/400 存在 proposal/admission 等生命周期事件；暂停必须在已完成并持久化相应 Event 状态后发生。重启时事件不能重复执行或遗漏。
5. **三 Arm 调度**：保留原有 T4×2 设备队列（Q 与 QS 串行、QC 独立）的可比性。某 Arm 已完成本 Segment 时应跳过训练；其他 Arm 可继续完成自己的区间。不要把 PAUSED 当作 Worker 失败。
6. **Results**：未完成时只产生进度与工程审计；不得生成最终 H1/H2 判定、最终成本比较或伪造 COMPLETE。
7. **GitHub Publish**：只发布白名单汇总元数据；GitHub 认证/发布失败不能破坏已完成的本地 Checkpoint。保留已有 `publish-only` 重试路径，但要防止未完成版本被错误解释为完整实验。

## 8. Kaggle Notebook 启动与版本交接

新 Notebook 顶部仅暴露可配置入口（示意）：

```python
ACTION = "train"              # train | evaluate | verify
SOURCE_SHA = "<frozen-full-commit>"
RESUME_INPUT = None            # /kaggle/input/<mounted-previous-version>/...
STOP_AFTER_STEP = 200
REQUIRE_RESUME = False
```

不得在执行中悄悄切换 `SOURCE_SHA` 或自动选择最新 GitHub 分支。Notebook 只负责 Bootstrap 和 CLI 调度；训练逻辑必须留在仓库模块，并允许在 CPU 环境运行单元测试。

**真实操作流程：**

1. 第一个 Notebook 版本：固定 SHA，完成 Preflight 和 0–200 Step，检查 `run-status.json` 为 `PAUSED` 且 `continuation.json` 可校验。
2. 使用 **Save & Run All** 成功保存该版本，并核实 Output 中恢复包真实存在。
3. 创建下一次运行的 Notebook（或在 Kaggle 支持的情况下使用更新版本），通过 **Add Input → Notebook Output Files** 将**刚保存成功的指定版本**挂载为只读 Input。
4. 明确填写挂载的真实路径、`REQUIRE_RESUME=True`、`STOP_AFTER_STEP=350`；新 Session 首先执行 `verify-checkpoint`，再恢复训练。
5. 重复直到 500 Step。每次确认下一版本准确指向预期父版本，防止误用旧 Output。

**禁止把“本地脚本拷贝目录后恢复成功”当作 Kaggle 持久化成功。** 如果 Kaggle 界面无法建立 Notebook→自身历史版本的输入引用，采用两个交替 Notebook 或私有 Kaggle Dataset 进行接力；不需要人工下载文件。这里是平台兼容的备选路径，须通过实际操作验证。

**平台限制：** 没有外部存储时，不能承诺 Save & Run All 正在执行中的瞬时故障也能恢复最近 Step。必须为单次 Segment 留足执行时间余量，避免达到 Kaggle Session 上限；只有成功保存并可被下一个 Session 挂载的版本才算真正持久化。

## 9. Evaluate / Resume / Fork 三种语义严格分离

| 模式 | 加载内容 | 是否更新模型 | 是否保留原 Run 身份 | 是否可改变算法 |
|---|---|---:|---:|---:|
| Resume | 模型、Optimizer、RNG、Data Cursor、Shadow/Event、探针 | 是 | 是 | 否 |
| Evaluate | 模型、Registry、固定数据和探针；评估所需最小状态 | 否 | 原 Run 只读引用 | 仅可新建独立评估协议 |
| Fork | 指定历史模型；按新协议决定继承或重置 Optimizer 等 | 是 | **否** | 是，需新身份 |

评估必须同时记录：`checkpoint_sha256`、源 Run 身份、评估代码 Git SHA、评估配置 Hash、数据/探针身份、随机种子、指标来源和是否为事后探索性分析。

从 R3 旧 Checkpoint 评估属于 **retrospective/exploratory**，不得替代 R3 已冻结的原生 Snapshot 或宣称已有预注册纵向观测。

## 10. 纵向诊断基础设施（工程预备，不改变训练算法）

新实验应在首次运行前冻结 Snapshot 计划，例如 `[0,50,100,150,200,250,300,350,400,450,500]`；必须设置 `periodic_diagnostics_enabled=true`。这是下一轮实验的拟议配置，Codex 本次只实现支持和测试，不擅自启动正式运行。

每个 Snapshot 至少保存：

- 冻结探针上的 `greedy / sampled / uniform` NLL 与计算成本；
- STOP rate、forced-end rate、CTX 调用比例、路径长度、self-loop、Cell 调用分布；
- Q 的候选排序、校准/后悔值与 STOP 相对 Cell/CTX 的评分差距；
- Cell 在固定探针子组的条件优势与交叉优势、变化方向、稳定性；
- Q 与 LCG 梯度范数、缩放后比例和余弦；
- Snapshot 生成时间、耗时、样本数、配置与 Checkpoint Hash。

区分**轻量常规指标**与**高成本反事实评估**：后者可只在冻结的关键 Step 与固定样本上执行；预先做资源预估，必要时在执行前调小探针规模，并将该选择锁入配置。不能在见到 Development 结果后临时调整测量方法。

Snapshot 只在相应 Step 状态稳定后生成；所有报告采用原子写入与版本/哈希校验。恢复后既不能覆盖既有 Snapshot，也不能重复计入时间序列。固定探针必须在 Step 0 创建并保存，不允许从训练后的权重重新生成。

## 11. 可靠性和安全策略

- 启动时拒绝软连接逃逸、路径穿越、未授权绝对路径、不匹配的 Checkpoint Manifest。
- `torch.load` 可能反序列化复杂 Python 对象；只加载本项目可信 Kaggle 私有 Output，先校验来源及 SHA256，不接受用户未知来源 `.pt`。评估路径尽可能使用受约束的模型权重格式或严格的允许类型。
- GitHub Token/HF Token 等仅从 Kaggle Secrets/运行时环境读取；不得写入 `continuation.json`、日志、Notebook 或公开结果。
- `diagnostics/` 中逐条 Oracle、原始训练迹线、敏感数据保持私有；GitHub 返回白名单摘要，不改变现有安全边界。
- 现有实验身份中的 `protocol_sha256`、Assets/Tokenizer Hash、Source SHA 必须验证。不可用弱检查绕过失败。
- 错误中输出清楚的 `failure_code`：如 `RESUME_INPUT_MISSING`、`CHECKPOINT_HASH_MISMATCH`、`SOURCE_SHA_MISMATCH`、`FROZEN_PROBE_MISSING`、`EVENT_STATE_INCONSISTENT`、`OUTPUT_SIZE_BUDGET_EXCEEDED`。
- 断点被破坏时拒绝重训；若要重新开始，必须明确选择新 Run，并保留原始失败证据。

## 12. 测试矩阵与验收标准

### A. 本地 / CI 自动测试（必须）

1. **CPU 等价性**：固定合成数据，`连续 N 步` 与 `A 步 → 保存 → 新进程恢复 → N-A 步` 的模型参数、Optimizer、RNG、数据游标、Registry 和事件记录一致；在可完全确定的 CPU 条件下要求逐元素相等。
2. **非标准边界**：在普通 Step、Shadow proposal、Admission Event 前后分别暂停恢复；事件不重放、不丢失。
3. **多 Arm 不同步**：Q 已完成、QC 未完成、QS 尚未启动时能单独恢复，最终汇总正确。
4. **负向校验**：缺少 Checkpoint、篡改 `.pt`、改变 Runtime Config、Source SHA、Tokenizer/Assets Manifest、丢失 Probe Bank 时全部 fail closed。
5. **指定历史 Step**：选 Step 100，而 `latest.json` 指向 200，确实加载 100 且不会写入原数据。
6. **Fork 身份**：Fork 生成新 Run ID 且完整记录父来源，不可与 Resume 混淆。
7. **重复执行**：同一已完成 Segment 再执行不会重复训练、重复事件或重复 Snapshot。
8. **输出容量**：超预算有显式失败并指出最大文件；默认正常产物不重复存多份 Assets。
9. **失败注入**：在 Checkpoint 写入、校验文件写入、Manifest 提交和 Snapshot 前后模拟中断；只允许恢复最后一个完整提交。
10. **发布隔离**：GitHub 发布失败或 Secret 缺失不造成已完成 Checkpoint 损坏；不得将原始 `.pt`/Oracle 标签推到 GitHub。
11. **资源门禁**：启用 Snapshot 的耗时/体积进入预估；不可默认静默关闭诊断。
12. **回归**：已有 R3 单测继续通过；R3 历史 Source SHA、协议及结果文件不变。

### B. 真正的 Kaggle 跨会话测试（P0，必须有可核验证据）

使用**全新、明显区分于 R3 的工程 Smoke ID**，微型合成数据，不触发真实 R3b 正式训练：

- Version A：运行 0→6 步，保存 `PAUSED` 与完整恢复包；Save & Run All 成功。
- Version B：从 A 的 Output 作为 `/kaggle/input` 读取，运行 6→10 步，并输出明确的 `RESTORED_FROM_STEP=6`、`STARTED_FROM_RANDOM_INIT=false`、父版本身份。
- Version C：把该 Checkpoint 作为 Input，执行 `evaluate-checkpoint --step 6`；校验输入无变化，记录评估结果。
- 对照：同一合成实验直接执行 0→10 步，比较恢复结果。在 GPU 不保证 bitwise determinism 时报告可接受数值偏差与随机性原因；不以宽松容差掩盖恢复失败。
- **关键证据**：A/B/C 的 Kaggle Version 标识、真实只读 Input 路径、恢复前后 Step、Checkpoint SHA256、测试日志、结果 Manifest。

仅当以上完成，状态才能写 `KAGGLE_CROSS_SESSION_RESUME=PASS`。没有实际 Kaggle 验证时必须保持 `NOT_VERIFIED`，即使所有本地测试通过。

## 13. 交付物与实现顺序

建议按独立 Commit/PR 交付：

**Commit/PR 1 — Checkpoint Contract**
- 公共 `CheckpointStore`/`ContinuationManifest` 接口、实验级共享 Assets、精确选择历史 Step、哈希/身份/路径校验。
- 本地 CPU 单测与故障注入测试。

**Commit/PR 2 — Segmented Runner**
- `--stop-after-step`、`--checkpoint-input`、`--require-resume`；Preflight 缓存验证、Arm 独立游标、PAUSED 状态、事件恢复。
- `resume`/`evaluate`/`fork` 语义分离；既有 R3 回归通过。

**Commit/PR 3 — Kaggle Notebook & Diagnostic Gate**
- 固定 Source SHA 的轻量 Notebook、Input 路径配置、预先冻结的纵向 Snapshot 支持、体积预算报告、README/操作指南。
- 真实跨 Kaggle Session Smoke 验收，并将证据回传 GitHub（仅元数据）。

最终提供：变更文件列表、Commit SHA、测试命令与 PASS/FAIL 结果、实际 Kaggle Version/Output 证据、已知限制、下次启动 R3b 的最小操作手册。**没有真实 Kaggle 验证或无法获得 Kaggle 运行权限时，必须明确 BLOCKED / NOT_VERIFIED，不得宣称工程门禁通过。**

## 14. 本阶段 Done / Not Done

**Done 当且仅当：**

- 原始 Kaggle Output 能在另一个独立会话作为 Input 读取，并恢复到确切 Step；
- Resume 后状态验证通过，连续训练对照成立；
- 指定历史 Checkpoint 可在不训练、不覆盖文件的情况下评估；
- Preflight 不重复改变实验身份，Shadow/Event 不重复，PAUSED 不冒充 COMPLETE；
- 正式恢复包包含所有必要资产且容量可控；
- 固定探针/纵向 Snapshot 在新实验中默认受门禁保障；
- GitHub 白名单发布及原 R3 回归未破坏；
- 有真实 Kaggle A→B→C 验收证据。

**Not Done：** 仅写了恢复代码、只做了本地测试、只生成了 `/kaggle/working` 文件，或仅检查了 Kaggle Output 页面但没有下一会话实际恢复。

## 15. 明确的执行指令

请 Codex 先审阅当前 `ArcheLabs/mini-cells` 仓库和 R3 实际实现，将上述规范映射到现有模块，避免复制新的训练框架、破坏既有 R3 实验。先实施工程功能与自动化测试，提交 GitHub；随后使用独立微型合成 Smoke 实验验证 Kaggle 版本间持久化。不得自动运行正式 R3b、改变 R3 的科学协议或上传模型权重到 GitHub。

如发现本规范与仓库实际架构冲突，优先保持原有 Checkpoint 身份校验和研究隔离原则；在 PR 说明中记录必要的偏差及原因，不得静默放宽验证。完整结果应包括“已实现”“已通过测试”“尚未在 Kaggle 验证”三个独立状态，避免把代码完成误报成端到端验证完成。

# CLM-001-R3b：完整离线诊断与实测性能报告实施规范

**交付对象：** Codex\
**仓库：** `ArcheLabs/mini-cells`\
**目标结果分支：** `results/clm001-r3-a7792ef30582-33734fdd1558`\
**训练源码 SHA：** `a7792ef30582dfcaf324cca7e5245c8430df48b2`\
**Run ID：** `33734fdd1558`

## 0. 总指令：一次性完成，不设置人工阶段门槛

请检查当前远端代码与结果，然后**一次性完成**本规范中的
Notebook、执行器、任务恢复、结果发布、实测性能报告和测试。不要拆成需要用户逐阶段批准的
Phase A/B/C、Gate 1/2/3，也不要在每个阶段结束后停下来等待确认。

程序内部可以根据 Kaggle Session
时间限制自动分批；这是可靠性机制，不是人工审批门槛。每次 Session
重启都必须从私有 Hugging Face
的权威分析索引恢复，复用已经校验完成的任务。

本次只分析已经完成的训练产物。**禁止重训、禁止重跑 Pilot、禁止改动训练
Checkpoint、禁止修改冻结科学协议或采样预算、禁止打开 Confirmation。**

如果当前 Codex 环境可以操作
Kaggle，则直接执行并完成全量分析。如果不能操作 Kaggle
Session，也必须完成仓库代码、Notebook、测试、基于真实账本的性能报告和
GitHub 发布；最后只列出一个准确的 Kaggle
启动动作，并如实说明尚未产生的运行数据。不能假装已执行。

## 1. 目标交付

一次性完成以下全部工作：

1.  修正 Kaggle 分析 Notebook，使其默认分析本次 Run，而不是旧 Pilot。
2.  实现完整的 102 项 Posthoc Diagnostic 计划、自动分批和可恢复执行。
3.  将原始诊断结果与权威任务索引幂等提交至私有 Hugging
    Face，并验证远端内容。
4.  将允许公开的聚合结果、图表、覆盖账本及性能报告发布到 GitHub。
5.  生成本次 Run 专属的 Markdown 和 JSON 性能报告。
6.  逐项统计 102 个计划任务的状态。每项必须为
    `COMPLETE`、`UNAVAILABLE`、`BLOCKED` 或
    `PENDING`，并附可核验的理由；全量执行结束时不得残留 `PENDING`。
7.  执行测试与 CI，并提交变更及最终交付总结。

不要只写文档、只改
Notebook，或只生成报告；应在执行权限允许时把整个链路跑完。

## 2. 输入身份必须精确固定

已知目标身份：

-   结果分支：`results/clm001-r3-a7792ef30582-33734fdd1558`
-   训练源码 SHA：`a7792ef30582dfcaf324cca7e5245c8430df48b2`
-   Run ID：`33734fdd1558`
-   训练臂：`Q-R3b`、`QC-R3b`、`QS-R3b`

本次核心文件：

-   `research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/artifacts/a7792ef30582dfcaf324cca7e5245c8430df48b2/33734fdd1558/checkpoint-index.json`
-   同目录下的 `experiment-manifest.json`
-   同目录下的 `resource-ledger.json`

必须从该 Run 的清单、Checkpoint 索引和私有 Hugging Face
发布引用中重新校验 `hf_repo`、不可变 `hf_revision`、bundle 标识和
SHA256。不能从旧 `baseline-evidence.json` 继承旧 Pilot 的 HF
revision，不能用可变分支名或"最新 revision"替代精确引用。

生成专属 `analysis-input-reference.json`，至少记录 training source
SHA、run ID、runtime config SHA、assets manifest SHA、tokenizer
SHA、training protocol SHA、HF repo/revision、Checkpoint index
SHA、analysis source SHA、analysis protocol
SHA。训练身份与分析身份必须分开，分析代码升级不能改变模型的训练身份。

如果实际读取到的训练 SHA 或 Run ID
与本文不一致，停止对该输入执行诊断，并报告身份不匹配；不得静默切换到其他运行。

## 3. 必须修正的现有问题

### 3.1 Notebook 默认只列出 Checkpoint

当前 `r3b/analysis-notebook.ipynb` 默认
`ACTION="list-checkpoints"`，只列出状态，不会执行完整任务。改为单一、明确的全量分析入口：验证身份、读取索引、执行未完成任务、汇总结果、生成报告和发布。

### 3.2 Notebook 引用旧 Pilot

当前 Notebook 从 `baseline-evidence.json` 读取 Checkpoint
引用。必须改为本次 Run 的精确 immutable HF revision。不得静默回退旧
Pilot。

### 3.3 `max_tasks=1` 不能成为全量执行限制

当前 Notebook 固定 `max_tasks=1`。应改为自动批处理：在 Kaggle Session
时间预算内尽可能执行任务，接近时间限制时安全提交状态并退出；下一次启动自动恢复。批次大小不得改变任务数、样本量、策略、数据角色或统计口径。

### 3.4 Development 资产缺失必须如实报告

如果 Development Probe Bank 缺失、SHA 不匹配或其角色实际是
Calibration/Pilot，不能重标记，也不能拿 Calibration 代替
Development。相关计划任务必须标记 `UNAVAILABLE` 或
`BLOCKED`，写清错误码、缺少的资产及影响范围；不生成假数据、不填
0、不缩小 102 项的分母。

### 3.5 发布状态要与远端一致

检查 `publication.py`。不得只在本地写 `PUBLISHED`
而不提交对应回执，也不能让 GitHub 永远停留在
`PREPARED`。分析发布状态必须独立于训练端
`latest.json`；分析失败不得改变训练状态。

## 4. 102 项诊断执行契约

冻结计划为：

  Stage           计划数 内容
  ------------ --------- -------------------------------------------
  Quality             39 质量与策略对照
  Functional          42 Cell 功能、优势、Regret、专业化与纵向变化
  Action              21 Router、STOP/CTX 等动作校准
  **合计**       **102** 完整计划矩阵

必须从 `diagnostic-execution-v1.json` 和 `diagnostic_schedule.py`
生成任务身份，不能手写一份不同的替代矩阵。102 是计划任务数，不等于 102
次训练，也不等于 GPU 小时数。

执行规则：

-   每个任务从指定 immutable Checkpoint 恢复模型；不得执行优化器更新。
-   使用冻结的数据、Tokenizer、Probe Bank、采样协议和策略。
-   保留 Step 400 的 pre-event/post-event 区别及准确 event revision。
-   每项记录 identity、开始/结束时间、设备、耗时、样本覆盖、结果 SHA、HF
    revision、状态与错误原因。
-   任务结果先幂等提交到私有 HF，验证远端索引后才可发布聚合结果。
-   已被权威私有索引验证为 `COMPLETE` 的任务直接复用，不重复计算。
-   `RUNNING`、本地文件存在或上传请求已发出，都不能视为完成。
-   失败任务保留错误码并可重试；不能转成零分或默认成功。
-   最终必须满足
    `complete + unavailable + blocked + pending = 102`；全量执行结束时
    `pending=0`。

如果当前冻结计划展开后无法生成 102
个可执行任务，仍须逐项记录缺失计划单元和原因，不得悄悄缩小分母。

## 5. Kaggle 执行与恢复要求

Notebook 使用现有 Kaggle Secrets 读取 `HF_TOKEN` 和
`GITHUB_TOKEN`；凭证不能写入源文件、日志、异常或产物。

单一 Notebook 入口必须自动完成：

1.  校验分析源码 SHA 和工作目录 HEAD。
2.  校验本次 Run 的 immutable HF 输入引用。
3.  验证 Checkpoint index、Checkpoint SHA、配置和资产身份。
4.  从私有 Analysis Index 读取已完成与待办任务。
5.  自动分批执行尚未完成的任务。
6.  逐项幂等提交 HF 结果并验证远端索引。
7.  重新读取权威索引，生成聚合结果、图表、覆盖账本与性能报告。
8.  发布允许公开的 GitHub 产物。
9.  输出清楚的 `COMPLETE`、`PARTIAL` 或 `BLOCKED` 状态及 102 项计数。

重复运行必须安全：不得重训、不得创建新的训练 Run ID、不得更换输入
revision、不得重算已验证完成的任务。不得依赖 Kaggle 本地工作目录在下一次
Session 仍然存在；不得要求用户手工下载 `.pt`；不得把 Kaggle Notebook
Output 当作 Checkpoint 的权威存储。

Hugging Face 是模型和私有原始诊断结果的权威存储；GitHub
用于源码、公开聚合数据、图表、覆盖清单与性能报告。

## 6. 科学与安全不变式

禁止：

-   修改任何训练 Checkpoint；
-   调用 `Trainer.update()` 或执行优化器更新；
-   重生成训练数据、Token、Probe Bank 或 Oracle 标签；
-   改动质量指标、采样数量、策略、Seed、窗口或 Bootstrap 口径；
-   互换 Calibration 与 Development；
-   用 Confirmation 数据进行选择、调参、补齐或验证；
-   因结果不理想而修改任务计划；
-   将缺失观测标成 0、PASS、阴性结论或科学失败；
-   将原始逐样本数据、私有标签、模型权重、令牌或完整私有 payload 发布到
    GitHub。

继续使用现有只读隔离与状态指纹验证。每项任务执行前后验证输入 Checkpoint
与训练状态未改变。不能证明只读的任务不得标记为 `COMPLETE`。

分析完成前不得宣称科学假设成立或失败。`scientific_decision` 与
Confirmation 必须保持关闭。

## 7. 本次 Run 的性能报告

新增独立报告，不覆盖旧历史报告，例如：

-   `r3b/reports/runs/33734fdd1558/CLM-001-R3b-POSTHOC-PERFORMANCE.md`
-   `r3b/reports/runs/33734fdd1558/performance-summary.json`

旧报告记录的是当时尚未执行新 GPU Run 的状态，应保留为历史证据。

### 7.1 训练阶段

从本次 `resource-ledger.json` 和发布记录读取：

-   三个训练臂的更新步数、CUDA 时间、P50/P99；
-   Mandatory Admission/Promotion/Event 成本；如果账本没有记录，明确写
    `NOT_RECORDED`；
-   Checkpoint 序列化、HF 上传、远端 ACK、最终回读耗时；
-   逻辑 Bundle、提交
    payload、新增内容对象大小，并说明它们不是同一个指标；
-   训练端到端墙钟时间。只有起止时间可信时才可标记为实测，否则
    `NOT_MEASURED`；
-   两小时训练关键路径判断：仅在完整墙钟实测可用时给出
    `PASS`/`FAIL`，否则 `NOT_VERIFIED`。不得用 CUDA 时间相加推断通过。

### 7.2 离线诊断阶段

从每项任务的计时记录和 HF 索引生成：

-   每类任务计划数、完成数、不可用数、阻塞数；
-   Quality、Functional、Action
    的耗时中位数、P90/P95/P99、最小值、最大值；
-   按设备、arm、step、role、stage 的耗时；
-   模型恢复/下载、计算、上传、提交和远端验证时间；
-   新计算与复用任务数量；
-   实际 GPU 时间和分析阶段墙钟时间；
-   未完成任务的原因与剩余成本估算。估算必须与实测分开。

### 7.3 报告口径

分别报告：

1.  `Training critical path`：训练端到端耗时；
2.  `Posthoc diagnostic cost`：离线分析 GPU 时间与墙钟时间；
3.  `Total study resource cost`：训练和诊断可加总的实际资源；
4.  `Total elapsed wall-clock`：只有起止时间与先后/并行关系可信时才报告；
5.  `Two-hour verdict`：只评估训练关键路径，不能把移至另一个 Session
    当作减少了 GPU 资源。

每个数字标注 `MEASURED`、`DERIVED`、`ESTIMATED` 或
`NOT_MEASURED`，并写明来源路径。缺失的实测值不能杜撰。

## 8. GitHub 公开产物

发布允许公开的聚合文件、图表、102 项任务覆盖账本、分析输入引用、analysis
source/protocol SHA、HF revision 与公开 commit 的关系，以及性能
Markdown/JSON。按当前协议更新：

-   `experiment-manifest.json`
-   `metrics-summary.json`
-   `gradient-audit-summary.json`（仅在协议要求且可从真实数据生成时）
-   `routing-longitudinal-summary.json`
-   `specialization-summary.json`
-   `statistical-contrast.json`

空结果或覆盖不完整必须明确标注
`PARTIAL`/`INCONCLUSIVE`。发布后验证远端分支确实包含产物，并扫描凭证、私有标签和原始
payload。最终回执须提供真实 commit SHA 与文件路径。

## 9. 必须执行的测试

以下均属于本次一次性实施工作，不是需要用户逐项批准的阶段门槛：

1.  Notebook/CLI 静态检查、导入测试。
2.  训练 SHA、HF revision 或 Checkpoint SHA 不匹配时失败关闭。
3.  输入 Checkpoint 被修改时失败，不能提交 `COMPLETE`。
4.  证明分析不执行训练更新，且模型、Optimizer、RNG、事件游标和输入文件状态不变。
5.  已完成任务重复执行时复用结果，不重复计算。
6.  上传/提交/Session 中断后可恢复。
7.  Development 资产缺失或角色不符时记为不可用，不得替换成 Calibration。
8.  Confirmation 始终封闭。
9.  GitHub 白名单有效，不发布私有数据或凭证。
10. 任务计数恒等式成立，结束时 `pending=0`。
11. 性能报告严格区分实测、推导、估算和未测。
12. 原有相关测试与 CI 通过，并记录真实命令、结果与链接。

不得为了让测试通过而减少任务、缩小样本、改动数据角色或冻结协议。

## 10. 最终验收与 Codex 执行指令

最终交付报告必须一次性说明：

-   Notebook 是否指向正确 Run 与 immutable HF revision；
-   102 项任务的完成/不可用/阻塞数量及逐项原因；
-   私有 HF 的索引、revision 和内容是否验证；
-   GitHub 聚合结果、图表、覆盖账本、性能报告是否发布；
-   训练端到端时间是否实测；两小时目标为 `PASS`、`FAIL` 还是
    `NOT_VERIFIED`；
-   离线诊断 GPU 时间、墙钟时间及总资源成本；
-   训练状态和科学协议是否保持不变；
-   所有测试/CI 的实际结果；
-   修改文件、提交 SHA 与结果链接；
-   若不能执行 Kaggle，唯一尚需用户执行的准确 Notebook 路径与启动方式。

**请完整阅读本文后直接实施整个工作。不要设置需要用户逐步批准的门槛。**
一次性完成仓库修改、可运行验证、尽可能完成全部诊断与发布，并在最终报告中如实区分已完成、不可用、阻塞和未实测事项。没有执行证据的内容不得声称完成。

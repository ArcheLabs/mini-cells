CLM-001-R3b：从 Kaggle 历史 Output 恢复并完成 102 项诊断
交付对象：Codex / MiniCells 工程执行者
目标仓库：ArcheLabs/mini-cells
文档性质：一次性工程实施 + Kaggle 执行规范（不是分阶段审批计划）
训练实验：CLM-001-R3b
本次研究目标：复用既有训练 Checkpoint 与已完成分析，补齐冻结的 102 项事后诊断，给出研究汇总和实际资源报告
绝对禁止：重新训练已完成的 R3b、修改冻结科学协议、打开 Confirmation、用未验证结果填补缺失。
0. 给 Codex 的总指令
请直接检查当前 ArcheLabs/mini-cells 仓库与目标 Kaggle 历史版本，一次性完成以下整个工作：将 R3b 的 full-analysis 改造成 Kaggle Output 优先、完全离线读取历史训练状态、优先复用旧结果、按任务安全持久化、自动完成全部可执行诊断、发布研究总结和实测性能报告 的流程。
不要设置“先提交 Phase A，等待批准才能做 Phase B”之类人工门槛。 输入完整性验证、失败关闭、任务自动分批、Kaggle Session 时间边界，是软件可靠性约束，不是人工审批。工程实施、测试与可执行的 Kaggle 分析应在一个工作请求中连续推进。因 Kaggle 平台单 Session 时间限制而分成多个实际执行 Session 是允许的，但每个 Session 必须自动保存可继承的持久化结果，不能让用户重新运行训练或逐项配置任务。
如当前 Codex 无权实际操作 Kaggle UI/API：仍一次性完成全部仓库代码、Notebook、自动恢复、测试、报告生成逻辑和使用说明；最后仅列出用户需要执行的准确 Save & Run All 操作，不能谎称 102 项诊断已经完成，也不能以额外人工 Gate 延误代码实施。
原则：Checkpoint 来源与诊断结果存储解耦。Kaggle Saved Output 成为本轮分析的优先、不可变训练数据来源；Hugging Face 可作为旧结果回收来源及低频备份，但绝不是读取 Checkpoint 的必经路径，也不能阻断已经保存到 Kaggle Output 的研究数据。
1. 现有事实及不可混淆的身份
对象	已知值	约束
训练 Source SHA	a7792ef30582dfcaf324cca7e5245c8430df48b2	永不重标记
训练公开目录键	33734fdd1558	与内部 Run ID 不同
内部训练 Run ID	r3b-a7792ef30582-ea3ea51c	必须校验
三个训练臂	Q-R3b、QC-R3b、QS-R3b	每臂已完成 500 Step
原始训练 HF immutable revision	828585f1ff93420aa4485c725211ed28f70d15f3	仅用于校验原训练事实或可选回收，不是本方案强制下载来源
先前分析引擎 Source SHA	ed2accbc90f541325ca2f6c53229e9c81475768d	既有 42 项结果可能以此参与任务身份哈希
既有分析协议 SHA256	8f9bd833ab371ce8c624693222825269889da1fde1860f8be7a73f2bb2eb90d0	重新从冻结配置检验，不能静默变化
Kaggle Notebook	bingjiangli/minicells	历史 Output 来源账户/slug
用户提供的页面	https://www.kaggle.com/code/bingjiangli/minicells/notebook?scriptVersionId=359735394	scriptVersionId 不是 CLI Version Number
用户报告的最新 CLI Version Number	65	是候选历史来源，必须核对对应运行身份、时间和 Output 内容
既有 Kaggle 日志	42/102 COMPLETE，余 60 因 HF HTTP 429 BLOCKED	仅证明运行期计数，不能直接等于远端有效结果数
初始诊断计划	Quality 39 + Functional 42 + Action 21	102 个逻辑任务
训练输出状态	用户已检查 Kaggle Output 含完整 Checkpoint	仍要程序化核对所有 57 个记录、冻结资产和完整性


重要： 版本 65 是用户报告的“最新版本”，目前没有证据证明它必然是训练输出版本、42 项诊断版本，或与 scriptVersionId=359735394 一一对应。请通过版本 API、Output 文件清单、日志中的 Source SHA / Run ID 和实际文件哈希核对。训练 Checkpoint 来源版本与中断的分析结果版本可以不同，分别锁定 TRAINING_OUTPUT_VERSION 和 PRIOR_ANALYSIS_OUTPUT_VERSION；不得将一个编号硬编码为两者。
历史训练公开清单：
research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/reports/runs/33734fdd1558/checkpoint-index.json
既有报告：
research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/reports/runs/33734fdd1558/
本次实现必须从当前实际 Git HEAD 重新读上述文件，不依赖本文记录的路径推测文件是否存在。
2. 最终目标与范围
完成以下所有交付：
1. Kaggle Saved Output 数据来源：通过版本号直接下载，或从受信任、只读的 /kaggle/input 历史版本挂载。优先从完整历史 Output 读取所有训练状态，不以 HF 能否提供全部 Checkpoint 为前提。
2. 完整性核查：校验三个 Arm、57 个预期 Checkpoint 记录及其 SHA256/事件修订、初始冻结探针、Development Bank、训练配置、Tokenizer、Assets、Protocol、Source SHA 和训练 Run ID。
3. 回收旧分析：在 Kaggle 历史 Output 和旧 HF 私有分析前缀中寻找已完成结果，按来源身份与内容完整性核验；复用通过验证的结果，不猜测 42 项全部存在。
4. 102 项诊断：在本地优先持久化结果的模式下运行真正缺失的任务，保留 Quality 39 / Functional 42 / Action 21 和全部采样定义；不重新训练。
5. 跨 Kaggle Session 自动延续：每个运行都产生可读取的 Saved Output 数据包，后续 Session 读取固定历史输出和前一份分析包，继续执行，而不是依赖已结束 Session 的内存和 /kaggle/working。
6. 科学聚合：在经过验证的任务结果上产生 Quality、Functional、Action、Router、STOP/CTX、专业化、纵向曲线及主 DID 对照。缺失仍是缺失。
7. 真实资源报告：汇总训练期 Ledger 和新旧分析期已记录的真实资源数据；未测不得称实测。
8. 发布结果：仅允许公开的聚合、图表、Coverage、签名/身份和性能报告提交到 GitHub。私有模型、逐样本原始诊断和完整训练快照留在私有 Kaggle/HF 存储。
9. 测试与审计：提供 CLI/Notebook 可运行入口、负面测试、CI、完整任务清单、历史版本引用、输出哈希及最终结果链接。
研究完成的定义是得到完整可靠的分析证据，不是 QC 模型必须优于 Q 模型；H1/H2/H3/H4 的状态必须由真实观测决定。independent_seeds=1，MULTISEED_NOT_TESTED，CAUSAL_FEEDBACK_NOT_TESTED，formal_confirmation=false 继续保持。
3. 历史 Kaggle Output 的精确发现与获取
3.1 必须先确认 Version 65 所指向的数据
仅列出 Output，不启动 GPU 推理、训练或分析。对 Kaggle CLI 使用明确的版本引用：
kaggle kernels files bingjiangli/minicells/65 --page-size 200
如分页存在，列出所有页；不能只看第一页，也不能只检查 HTML。要检查文件路径、大小与扩展名，寻找如下对象：
- /kaggle/working/minicells-r3b-analysis-ed2accbc90f5/analysis/...（若为分析版本）；
- continuation.json、experiment.lock.json、runtime-config.json、protocol.md；
- arms/Q-R3b/、arms/QC-R3b/、arms/QS-R3b/ 的 Checkpoint；
- arms/<arm>/frozen-probe-bank.pt 和 development-probe-bank.pt；
- assets/、Tokenizer、冻结 Probe 的元数据；
- analysis/coverage-ledger.json、analysis/public/*.json、analysis/result.json、图表、日志；
- __results__.html（仅日志/展示证据，不替代原始诊断结果）。
对所有潜在版本建立 kaggle-version-inventory.json，记录 Kaggle Notebook ref、CLI Version Number、网页 scriptVersionId（若可核对）、运行状态、Output 文件数、路径/大小、身份匹配结论和检索时间。
若版本 65 实际不含完整训练快照：不能悄悄回退 HF 或把 65 当成完整来源。 在用户已保存的其他历史 Kaggle 版本中查找原训练 Saved Output，固定正确版本；此前 42 项分析应从对应分析版本独立查找。若无法确定，报告 KAGGLE_TRAINING_OUTPUT_VERSION_UNRESOLVED 并保留清单，不能执行错误模型分析。
3.2 正式下载训练数据
下载命令示意（实际版本必须来自校验结果）：
kaggle kernels output bingjiangli/minicells/65 \
  --page-size 200 \
  -p /kaggle/working/minicells-r3b-source-v65
如 Kaggle 数据大，应使用 --file-pattern 精确下载整个有用目录所需文件类型，而不是只取 .pt。下载全部 Output 时无需解压模型文件，避免无谓空间复制。
为避免误取最新版本，使用包含正确处理指定历史版本的 Kaggle CLI；安装完成后通过 kaggle --version、kaggle kernels files .../65 的固定结果与本地文件校验确认它确实读到了正确版本。安装 CLI 的某个 Git ref 要记录确切 commit；不得未经核验永久固定此前对话给出的短 SHA。官方 CLI 对历史版本用 owner/slug/<version-number>，不是 scriptVersionId。
禁止： 使用不带 /版本号 的可变最新 Output 作为冻结研究输入；禁止覆盖旧输出目录；禁止将版本号变化解释为科学实验变化。
3.3 安全、容量和可见性
- 只能访问用户授权的 Notebook 版本及该用户已有的 Kaggle API 凭证（优先 KAGGLE_API_TOKEN Secret；遵循当前 CLI 支持的认证方式）。
- 校验可用磁盘空间、文件数、完整下载字节数。大文件下载失败需要可恢复或重新验证；不能把部分 .pt 当完整快照。
- 路径遍历、恶意 ZIP 条目、符号链接逃逸、同名覆盖必须失败关闭。研究代码应在受信任源码中使用 PyTorch Checkpoint；文件 SHA 一致也不意味着任意 Pickle 文件安全。
- Kaggle Notebook/Output 是否公开，需要实际核对；如 Output 包含私有模型或逐样本数据，不得在公开 Notebook Output 中发布。保持历史可见性边界或采用实际可验证的私有仓库存储。不能简单假设“用户自己的 Kaggle Output 天然是私有的”。
- 原始凭证不得打印、写入 Output 或 GitHub。
4. 必须检查的 Checkpoint 完整性：不仅是 .pt
仓库原始 Checkpoint Index 记录 57 条：3 个 Arm × 每臂 19 条历史版本（11 个基本 Step + 对 Event 的额外修订）。至少核查：
- 基本 Step：0,50,100,150,200,250,300,350,400,450,500；
- Step 100 的事件前及最终已提交 Event Revision；
- Step 200 的事件前及逐次已提交修订；
- Step 400 的 pre_event (revision=0) 与所有已提交事件修订，尤其最终 post_event；
- 训练终点为每 Arm Step 500，并校验 training_state_sha256、event_state_sha256、Checkpoint 哈希；
- 完整 Frozen Calibration 和 Development Probe Bank、其身份文件与哈希；
- Assets / Tokenizer / Runtime Config / Protocol / Seed 与原训练身份一致；
- runner.verify(root) 与 MountedInputStore(root).verify()、MountedInputStore(root).select() 等既有校验通过。
不要从 .pt 文件名臆造 revision，或将某一步的最终模型冒充事件前状态。 若某个历史修订不存在，准确标记它对应的逻辑任务 UNAVAILABLE；不替代为 Step 500、相邻 Step 或 HF latest。
程序需新建 kaggle-output-input-reference.json，例如：
{
  "schema": "minicells.r3b.kaggle-output-input.v1",
  "training_source_sha": "a7792ef30582dfcaf324cca7e5245c8430df48b2",
  "training_run_id": "r3b-a7792ef30582-ea3ea51c",
  "public_run_directory": "33734fdd1558",
  "kaggle_owner": "bingjiangli",
  "kaggle_kernel": "minicells",
  "kaggle_version_number": null,
  "kaggle_script_version_id": null,
  "output_inventory_sha256": null,
  "checkpoint_index_sha256": null,
  "original_hf_checkpoint_ref": {},
  "verified_checkpoint_records": 0,
  "asset_and_probe_verification": "PENDING",
  "storage_transport": "KAGGLE_SAVED_OUTPUT",
  "training_updates": 0
}
其中 null 是schema 占位，不能原样当成验证通过的真实输入。original_hf_checkpoint_ref 可继续保存原清单中的科学引用（为任务身份兼容），但实际读取必须来自经过验证的 Kaggle 本地只读目录，不能偷偷下载 HF 以通过验证。
如果只有 .pt 而没有冻结探针、Assets、Tokenizer 或事件状态，不能宣称完整分析可运行。允许再从另一个已锁定并校验的 Kaggle 历史版本补齐完全相同身份的文件，记录每个文件来源与哈希；不允许临时重建/替换 Development Probe 或使用 Confirmation。

5. 修改当前分析架构：输入来源与科学任务身份分离
当前主要代码位置：
r3b/analysis-notebook.ipynb
r3b/analyze.py
r3b/implementation/clm001_r3b/full_analysis.py
r3b/implementation/clm001_r3b/analysis_runner.py
r3b/implementation/clm001_r3b/analysis_store.py
r3b/implementation/clm001_r3b/analysis_aggregate.py
r3b/implementation/clm001_r3b/posthoc_performance.py
r3b/implementation/clm001_r3b/diagnostic_schedule.py
r3b/implementation/clm001_r3b/runner.py
当前 full_analysis._run() 默认执行：
HFCheckpointStore(...).fetch_verified_checkpoint(...) → runner.verify → run_tasks → HF AnalysisStore.commit → HF AnalysisStore.save_coverage。
新方案应实现：
VerifiedKaggleOutputStore → runner.verify → ExistingFrozenAnalysisEngine → DurableLocalResultStore → Output Snapshot + Optional Batch HF Backup → Aggregate → GitHub Public Report。
至少提供新入口：
--mode full-analysis
--input-source kaggle-output
--training-output-version <verified-int>
--prior-analysis-output-version <verified-int or none>
--output /kaggle/working/minicells-r3b-posthoc
--device cuda:0
--session-seconds 6900
--publish-github
参数名称可按当前 CLI 最小变更调整，但必须有一键 Run All 默认走 Kaggle Output 的正式入口。原 --mode full-analysis 的 HF 模式可保留为向后兼容显式选项，不允许继续在新默认路径偷偷走 HF Checkpoint 下载。
5.1 切勿让更换储存来源破坏历史任务身份
这是本次实施中最容易出错的地方：在原代码里，analysis_runner.identity() 把 input_checkpoint_ref 和 analysis_source_sha 纳入任务 identity。直接把 HF ref 换成 Kaggle ref、或把 analysis_source_sha 换成新修改的 Git commit，会使旧的 42 项无法按同一 Key 被检索，即使原始计算完全相同。
实施要求：
1. 科学模型身份保持不变：Checkpoint 的内容 SHA、训练源码 SHA、事件修订与冻结协议不能重写。
2. 记录实际物理来源：Kaggle Kernel/version/output hash 放在单独的 storage_provenance，不得伪装成新的训练身份。
3. 保留原来固定的 input_checkpoint_ref 作为历史任务身份字段，但在运行证明中声明 actual_input_transport=KAGGLE_OUTPUT_VERIFIED；该字段仅保留与先前任务一致的引用身份，并不意味着本次又从 HF 下载。
4. 科学评估引擎与执行编排源码分别固定：如果要复用旧版 ed2accbc... 产生的 42 项结果，应让后续新计算继续使用经过源码哈希检验的同一版科学诊断引擎（可把旧引擎作为不可变 checkout，新的下载/保存/发布编排为另一套独立代码），分别记录 measurement_engine_source_sha 和 execution_orchestrator_source_sha。不得仅为了复用结果就把新分析代码伪称是旧 Source SHA。
5. 如必须修改科学指标/计算函数，则新分析应有新 analysis_source_sha 和新的逻辑任务身份；旧结果只能列为历史探索证据，未经可复核数值等价性审核不能混进新主统计。
6. 如旧 42 项与新产物在数据、引擎、Protocol、Checkpoint、角色、政策任一方面不一致，不直接复用，保留冲突记录并仅重新执行该具体任务。
5.2 旧版已执行的 42 项并非当然存在于本地 Output
根据原始 analysis_runner.execute()，其返回内存中的原始 result 字典；原 full_analysis.run_tasks() 随后提交 AnalysisStore.commit(result) 到 HF。原代码每任务未必单独把完整原始结果 JSON 写到本地 analysis/tasks。
因此日志的 complete=42 是运行时证据，不等于 Kaggle Output 必然有 42 份原始结果。必须扫描完整 Output 和旧 HF 私有前缀：
- 先找 Kaggle 保存的 analysis 目录内是否存在完整任务结果 JSON；
- 再查旧 HF Analysis Index、孤儿结果路径和每项结果的具体 revision/sha；
- 逐项验证 identity、task_status=COMPLETE、optimizer_updates=0、training_state_unchanged=true、SHA 和结果实际样本覆盖；
- 两侧均缺失才将任务标记为需重算。
禁止以旧 Kaggle 覆盖账本中的 COMPLETE 计数直接替换严格的结果文件校验。也不要因一次限流导致的 BLOCKED 报告覆盖之前已经可靠保存的 COMPLETE 记录。
6. 本地优先分析结果存储与跨 Session 保存
6.1 不再让每项任务受 HF commit 限流约束
已核对的旧源码中，每项任务通常产生：
1. HF 提交原始 result.json；
2. HF 提交 index.json；
3. HF 提交更新后的 coverage.json。
42 项约 126 次提交，因此接近此前 128 commits/hour 的限制是可以从代码解释的。新方案禁止维持三次 HF commit/任务的默认行为。
建立 LocalResultStore / KaggleOutputResultStore（具体名称自定），使用：
- 每项任务的单独不可变内容寻址 JSON 文件；
- 原子写入（先临时文件、flush/fsync、rename），结果 SHA256；
- 独立 analysis-index.json、coverage-ledger.json；
- 崩溃后通过完整文件和哈希重建任务状态；
- 输出文件与 task ID 永远不允许悄悄覆盖为不同内容；
- 任务计算结果持久落盘之后才标记本地验证完成；
- 研究跨 Session 的真正 COMPLETE 还需要存在于已保存的 Kaggle Output 版本或独立验证的私有存储中，避免把一次性 /kaggle/working 当成长期持久化。
日志应区分 COMPUTED_LOCAL、KAGGLE_VERSION_PERSISTED、HF_VERIFIED、VERIFIED_REUSED，而不是只用一个模糊的 COMPLETE。科学聚合可以使用已验证的本地原始观测；最终可恢复状态必须绑定一个可再下载的 Kaggle Saved Output 或精确 HF revision。
6.2 Session 保存、续接与快照发现
Kaggle 的 /kaggle/working 只在当前执行环境里可写。用户通过 Save & Run All 保存版本后，完成的版本 Output 才可以用 CLI/API 取回；简单调用 Path.write_text() 不能保证旧 Session 结束后仍有文件。
实现单一自动流程：
训练历史 Saved Output（固定版本）
         + 既有分析 Saved Output（固定版本，可缺失）
         + 旧 HF 原始任务（可选，仅一次验证读取）
                  │
                  ▼
          精确身份 / 文件验证
                  │
                  ▼
          已验证任务结果合并
                  │
                  ▼
        执行缺失的 102 计划任务
                  │
                  ▼
    每项落盘 + Coverage + 输出数据包
                  │
                  ▼
       Kaggle 保存新的历史版本 Output
                  │
                  ▼
  下一独立 Session 固定该版本继续，直到完成
可执行选项（按运行环境选用其一，不混用）：
- 方案 A，建议用于零人工循环：由受权的外部编排器通过 Kaggle API/CLI kernels_push 启动 Save & Run All 版本，等待结束，再用 kernels_output 下载/验证其 Output，记录版本号，并提交下一次版本运行，直到 102 项完成或明确不可用。编排器仅传入可信固定版本号，不在“最新版本”上盲读。须存在外部执行环境、Kaggle API 凭证与可用配额；这不是在 Kaggle Notebook 内凭空创建一个可以控制自身 Save Version 的权限。
- 方案 B，只有手工 Kaggle 页面可用时：Codex 把自动发现、下载、恢复和诊断全部封装为同一个 Notebook。每次在 Kaggle 选择 Save & Run All 自动执行当次 Session 的全部任务；如果由于时间预算暂停，则同一 Notebook 下次 Run All 自动读取先前确认的 Saved Output 版本。每次运行只要求一次点击，不要求用户挑选 Stage/Step/Arm、修改科学参数或再审批。
硬约束： 如果没有独立编排器或其他确实可用的持久化/定时服务，Notebook 不可能保证在当前 Session 关闭后自行启动下一 Session。这属于平台边界。实现必须明确报告 NEXT_RUN_REQUIRED 及可恢复版本，不得宣称“一次 Run All 就能无条件跨多个新 Session 运行完”。
6.3 避免重复复制 1GB 以上训练快照
训练历史 Output 按固定版本读取。新分析 Output 只保存新产生的诊断结果、分析索引、Coverage、报告和数据来源引用，不得每次又复制所有历史训练 .pt，除非 Kaggle 跨版本不可读取来源且确实没有其他可验证方式。
跨 Session 引用冻结的训练 Saved Output + 上次分析 Saved Output，减少下载与存储成本。若必须下载全部 Checkpoint，记录重用缓存策略、实际下载耗时和流量；绝不能省略关键事件版本。
对于多次分析 Output，还要定义累计结果包或祖先版本链二选一：
- 累计包：新版本保存先前的全部任务 JSON 加新结果，下一次只需下载最新分析包；
- 祖先链：只保存增量结果，但清单列出所有固定上游版本，恢复时逐个校验。
不得只保存最近一批结果，导致再下一次版本无法获取最初结果。
6.4 中断语义及结果文件安全
- SIGTERM、预算到期、断网、GPU OOM、HF 429 等需分别记录。
- 时间预算在任务边界检查；不为赶时限缩减采样或半途输出“完整结果”。
- 单任务计算完先落盘，即使上传失败，本地当前运行仍保有结果；若当前 Saved Version 最终保存失败，则该任务跨 Session 按未持久化处理。
- 单个任务结果 JSON 带原始科学 Engine SHA、真实物理输入存储来源、输入验证摘要、测量时间、设备、样本数、实际耗时、只读保证和 SHA。
- 仅在预先已知的受信任文件目录加载 .pt，输入目录和输出目录严格隔离；分析不得对输入源写入 latest.json、事件状态或模型文件。
7. 102 项科学任务不可变
唯一合法的任务展开来源：当前冻结 r3b/config/diagnostic-execution-v1.json 与 diagnostic_schedule.full_matrix()。固定覆盖：
Stage	计划数	主要工作
Quality	39	Development 上 Greedy/Uniform 主比较，Sampled/Constant 次比较；所有指定历史 Step/Event Phase
Functional	42	Calibration/Development 冻结 Probe 上的 Cell 条件优势、固定后缀、Regret、交叉分化、纵向变化
Action	21	Calibration 上 Router、STOP/CTX 和相关动作价值校准
合计	102	三臂同科学协议，单 Seed Development


强制保留 Step 400 的 pre_event / post_event 分界，所有指定 Event Revision 都来自实际 MountedInputStore.select 的已提交训练状态。不得为了性能调优而删减 Cell 对、Story Window、Bootstrap、采样或统计规则。
冻结边界：
- 训练源码与资产身份完全不变，不得调用 Trainer.update / optimizer.step；
- Functional 原协议允许为求导执行局部 Autograd，不能粗暴套一层 torch.no_grad() 改科学定义；
- Calibration、Development、Admission、Confirmation 的角色边界不变；
- Confirmation 继续 SEALED；
- 研究仍为单 Seed 探索性结果，不能宣称已确认因果自强化；
- 路由分化不能只根据使用次数或可视化不均衡来判断，必须检查 Cell 条件 Advantage 和 Router 实际选择与 Regret；
- Quality 主 DID、QC greedy 与自己 uniform、QC greedy 与 Q greedy 比较要满足原匹配条件。
任务状态必须满足：
COMPLETE + UNAVAILABLE + BLOCKED + PENDING = 102
其中 COMPLETE 必须区分 result_data_verified 和 cross_session_persistence_verified。如果只有结果被计算但没有保存到新版本、HF 私有存储或同等可靠的持久化环境，不得对外宣称完整交付。
8. 从旧 42 项中尽可能恢复，而非直接重做
旧 Kaggle 日志：最后 complete=42, blocked=60, pending=0，核心阻塞为 HfHubHTTPError，HTTP 429、每小时 128 次 commit。旧执行日志第 42 项完成时间约 3425 秒，整体在约 3488 秒时被限流，最终发布了一个部分结果分支。
恢复程序应建立下列严格事实层级：
1. KAGGLE_RESULT_VERIFIED：从精确历史 Output 解出原始任务 result.json，与旧 Task Identity、Scientific Engine、Input Checkpoint/Role、SHA、只读证明匹配。
2. HF_RESULT_VERIFIED：在用户已有的 HF 私有分析前缀中，通过精准 revision 和 index + 结果 SHA 验证。允许回收“已上传结果但索引提交失败”的孤儿结果；不能因为索引孤儿就丢弃有效结果，也不能仅因孤儿存在就信任错误身份。
3. AGGREGATE_ONLY：旧 GitHub 汇总、Kaggle HTML 或日志有数字但缺少可验证原始观测。可做背景参考，不能直接当本次原始任务 COMPLETE。
4. MISSING：没有有效原始观测，才调度重新计算。
生成 recovery-ledger.json：102 个任务各自包括 Source Version、文件 SHA、旧 HF revision（若有）、是否复用、是否重新计算、是否存在冲突、是否已跨 Session 保存以及明确错误码。
对于旧 HF 的 429：读取已存在数据不需要创建仓库提交，恢复时禁止为了“修复索引”而每个任务创建新 HF commit。可以只读批量下载已有结果，将它们合并存入新的 Kaggle Output；若将来需要 HF 备份，则按批次单 commit，不在任务循环内部提交 Coverage。
8.1 关键源码身份与旧结果复用
旧版分析 analysis_source_sha = ed2accbc...，新代码提交 SHA 必然不同。应采用可审计的冻结科学引擎/新编排器双版本身份，将旧版 R3b 的 analysis_runner.execute 及其影响结果计算的依赖锁定为原分析源码，新的 Kaggle 下载与存储代码放在新的编排层。实际运行必须能证明旧引擎字节与 pinned SHA 所指代码一致，而不是调用新引擎同时把身份字段手填成旧 SHA。
如果实施者选择真正升级科学引擎，则 102 项需要新的 Analysis Identity。历史 42 项不得未经等价性证明混入新引擎结果；最终报告应明确研究数据的引擎来源范围。
9. 建议 Notebook 用户体验：一个入口自动完成
建议新增 r3b/kaggle-output-analysis-notebook.ipynb，或合理更新原 r3b/analysis-notebook.ipynb，默认模式：
EXPERIMENT = "CLM-001-R3b-POSTHOC"
ACTION = "full-analysis-from-kaggle-output"
KERNEL = "bingjiangli/minicells"
SOURCE_CANDIDATE_VERSION = 65   # 必须验证版本身份，不成功则自动寻找可信历史版本或失败关闭
PRIOR_ANALYSIS_VERSION = None  # 通过精确版号/Output 身份自动发现；不得猜测
TRAINING_SOURCE_SHA = "a7792ef30582dfcaf324cca7e5245c8430df48b2"
TRAINING_RUN_DIRECTORY = "33734fdd1558"
SCIENTIFIC_ENGINE_SHA = "ed2accbc90f541325ca2f6c53229e9c81475768d"
SESSION_SECONDS = 6900
DEVICE = "cuda:0"
PUBLISH_PUBLIC_AGGREGATES = True
HF_INPUT_REQUIRED = False
注意以上只是目标接口示例，65 是待确认的候选版本，不能不经验证直接当成原训练完整 Saved Output。Notebook 首次运行只需用户授权访问已有 Kaggle API Token 和必要 GitHub Secret，运行不应要求手工指定 102 个任务、挑选 Step 或填写二十几个 SHA。若自动发现有歧义，报告具体候选清单，不在可能是错误训练产物的文件上继续科学计算。
Notebook 显示至少以下状态：
R3B_TRAINING_IDENTITY=VERIFIED | BLOCKED
KAGGLE_TRAINING_OUTPUT_VERSION=<exact_version>
KAGGLE_PRIOR_ANALYSIS_OUTPUT_VERSION=<exact_version_or_none>
CHECKPOINT_RECORDS_VERIFIED=<n>/57
PROBE_AND_ASSET_IDENTITY=VERIFIED | BLOCKED
PRIOR_RESULTS_REUSED=<n>
HF_OLD_RESULTS_READ_ONLY_RECOVERED=<n>
NEW_TASKS_COMPUTED=<n>
TASK_COVERAGE=COMPLETE:<n>,UNAVAILABLE:<n>,BLOCKED:<n>,PENDING:<n>
CURRENT_ANALYSIS_OUTPUT=<exact_saved_version_if_known>
NEXT_RUN_REQUIRED=true|false
TRAINING_UPDATES=0
CONFIRMATION=SEALED
SCIENTIFIC_DECISION=false
如果同一 Notebook 无法在运行时获知它结束后才生成的版本号，打印 OUTPUT_VERSION_PENDING_SAVE，由外部编排器在 Save 完成后查询并写入续跑索引；不要提前伪造版本号。
10. 完整结果和性能报告
分析必须生成以下全部真实、可验证的研究产物（具体路径应遵循仓库白名单和已有 publication.py）：
r3b/reports/runs/33734fdd1558/
  kaggle-version-inventory.json
  kaggle-output-input-reference.json
  checkpoint-validation.json
  recovery-ledger.json
  coverage-ledger.json
  analysis-index.json
  session-resource-ledger.json
  analysis-resource-ledger.json
  metrics-summary.json
  gradient-audit-summary.json
  routing-longitudinal-summary.json
  specialization-summary.json
  statistical-contrast.json
  performance-summary.json
  CLM-001-R3b-POSTHOC-PERFORMANCE.md
  SCIENTIFIC-FINDINGS.md
  figures/
对于 GitHub：上述文件只有公开白名单且确认不会泄漏私有逐样本信息的子集可发布。analysis-index.json、恢复账本和详细版本 inventory 中可能包含私有路径；需要先做公开投影/字段脱敏。原始 per-story NLL、成本、Probe、Cell 反事实、完整 Checkpoint 及输出 ZIP 保留在私有的 Saved Output 或 HF，绝不直接 GitHub 提交。
10.1 必须给出的科学结论
1. Quality：Q/QC/QS 在对应 Step/Phase 的 Greedy / Uniform NLL 和路径成本、QC greedy 相对 QC own-uniform、QC greedy 相对 Q greedy；只有匹配数据完整时计算主 DID 及 95% CI。
2. Functional：Cell 条件优势、跨组逆转、Router-Oracle Agreement、Paired Regret、随时间变化；明确冻结初始 Probe，而非在线重新采样状态。
3. Action：STOP/CTX Margin、forced-end、CTX 读取及 Router Action-Type 校准是否随 Step 退化。
4. Step 400：Event pre/post 的明确修订及可比性（不可与普通纵向曲线混成一个点）。
5. Q/QS：QS 为零竞争梯度 Sham，不等于独立随机方向梯度负对照。
6. 结论状态：INCONCLUSIVE、EXPLORATORY_ONLY、NOT_TESTED 或有数据支持的更具体探索性信号；单 Seed 不能升级为 Formal Confirmation。
10.2 性能报告的实测口径
沿用原训练 resource-ledger.json，已知训练三 Arm 的 CUDA 测量约为：
Arm	Step	CUDA 秒（原账本）	更新 P50/P99 秒（原账本）
Q-R3b	500	1384.929964	2.7293 / 3.1446
QC-R3b	500	1472.399612	2.9010 / 3.3432
QS-R3b	500	1379.131448	2.7133 / 3.1449


三组 CUDA 更新资源时长合计约 4236.461024 s；不能把这个和 Training End-to-End Wall 当成同一个指标。 已有训练账本还记录 HF 上传/回读成本，但不包含可靠训练端到端时钟和完整语义事件成本，所以两小时训练关键路径应保持 NOT_VERIFIED，直至有可信时间戳证据。
新分析阶段应记录：
- Output 版本发现、文件枚举、下载、完整性校验耗时与字节；
- 每个任务恢复模型/加载 Probe/计算/持久化的独立耗时；
- GPU CUDA 事件测量、Session 墙钟、Task P50/P90/P95/P99；
- 复用旧任务数量、重新执行数量、丢失/失败/未完成数量；
- Kaggle Save Version 输出包大小、跨 Session 再下载时长；
- HF 旧结果只读查询、可选批量备份的独立耗时与 Commit 数量；
- GitHub 发布耗时；
- 训练成本、分析成本和完整研究 GPU 资源总计（不可覆盖遗漏项）。
对 MEASURED、DERIVED、ESTIMATED、NOT_MEASURED 分类，不得把 Output 下载等待时间算成 GPU 运算，也不得用连续两个 Session 的 GPU 活跃时间求“实际端到端日历耗时”。
11. 明确不做的事项
- 不重新训练 Q-R3b、QC-R3b、QS-R3b，不执行 Pilot，不改变 500 Step 和固定 Seed；
- 不改变历史 Checkpoint、Tokenizer、数据角色、探针、模型、Optimizer、RNG 或 Event 修订；
- 不进行新方法调参，不重新打开 Confirmation；
- 不因 HF 限流再中断模型的本地离线分析；
- 不以“用户报告全套 Checkpoint 存在”代替代码逐文件校验；
- 不以 __results__.html 或 42/102 日志计数替代原始任务 result 的内容校验；
- 不混淆 Kaggle scriptVersionId 与 Version Number；
- 不默认将 Kaggle Output 视为私有/不可删除的永久归档；
- 不在没有真实外部控制器的情况下承诺自动创建后续 Kaggle Session；
- 不向 GitHub 发布私有 Checkpoint、原始样本或访问令牌。
12. 一次性测试与回归要求
Codex 应在一个实施工作里完成（不需要用户逐项批准）：
1. 版本精确性：65 与 scriptVersionId=359735394 不相符或无法确认时，不把当前最新版本误识别为训练版本；分页文件清单完整。
2. 输入完整性：三臂 57 Checkpoint，Source、Run、Protocol、Tokenizer、Assets、Probe、Event 修订逐项哈希；缺少一个指定 revision 必须阻止相关任务。
3. 只读保证：研究输入文件前后哈希不变，模型/Optimizer/RNG/Event Cursor 不被任何分析步骤修改，training_updates=0。
4. HF 断开仍可算：已验证 Kaggle Output 完整的情况下，禁用 HF_TOKEN/模拟 HF 429，仍可以运行已冻结分析并将结果安全保存到当前 Output。HF 不能是输入必经路径。
5. 本地幂等：同一 Task ID 再跑时能复用原始结果，内容哈希冲突时报错；不可因为时间戳变化覆盖不同 bytes。
6. 旧结果回收：模拟 42 个原始完成记录、孤儿结果、只有 Coverage 的记录、HF 结果缺失、结果哈希错、旧 Engine SHA 错等情形，准确决定复用/重算。
7. 跨 Session：模拟上一 Saved Output + 下一 Session，验证所有已持久化结果被复用、剩余任务续跑；禁用原本 /kaggle/working 目录也能恢复。
8. 完整任务矩阵：冻结 Quality39 / Functional42 / Action21；严格 Event pre/post 对照、数据角色、Policy 集、样本预算不变；计数总和始终 102。
9. 科学结论安全：只有 42 项且三臂覆盖不全时主 DID 不被错误计算，Development 不被 Calibration 替代，Confirmation 始终封闭。
10. 敏感数据边界：私有结果和模型永不进入 GitHub 或公共 Kaggle 输出；凭证不会进入日志、异常和产物。
11. 发布可信：GitHub 结果分支与具体 Commit、白名单、文件 SHA 校验完成，公开覆盖状态与真实状态一致。
12. 性能口径：未测墙钟/语义事件耗时严格 NOT_MEASURED；所有新分析资源成本来自实测。
13. 真实运行：如 Kaggle 权限可用，做真实 Version Inventory + 精确 Output 下载 + SHA 校验，运行全部可执行诊断，提供 Saved Output 版本和结果回执；不可用时明确唯一缺少的执行动作。
建议新增测试文件到 tests/research/clm001_r3b/ 或现有对应测试包，至少覆盖以上异常路径。原有冻结 R3/R3b 测试不得回退。
13. 最终验收：一次交付全部状态
最终报告必须回答这些具体问题：
1. 真正的训练 Kaggle Output 版本号是多少？ Version 65 和 scriptVersionId=359735394 分别对应哪次运行？证据链接？
2. 57/57 Checkpoint 是否逐项验证？ 三臂 Frozen/Development Banks、Assets/Tokenizer/Protocol 是否齐全？
3. 旧 42 项中验证回收了几项？ 分别来自 Kaggle 原始任务文件和 HF 已完成任务/孤儿结果；哪些只有日志没有原始数据？
4. 实际重新计算了多少项？ 102 项的最终 COMPLETE/UNAVAILABLE/BLOCKED/PENDING 和每项原因？
5. 科学 Engine 是否仍是精确冻结的 ed2accbc...？ 如果更换了，旧结果是否被隔离并解释？
6. Saved Output 是否持久化成功？ 精确 Kernel/version、Output 文件列表与关键哈希；如果多 Session，明确版本链。
7. HF 限流是否不再阻塞分析？ 实际新增 HF Commit 数；即使 HF 不可用是否仍能通过本地分析+Kaggle 保存？
8. Quality / Functional / Action 有什么真实观测？ 哪些结论探索性支持、否定、不确定？
9. 真实成本是多少？ 训练 CUDA 资源、诊断 GPU 资源、Session 墙钟、Output 下载与保存成本、HF 可选备份成本，严格标注测量口径。
10. 提交与测试证据？ GitHub Commit/PR、CI 链接、实际 Notebook 版本、产物与安全审查。
只有原始结果文件及哈希可验证的已完成任务才能进入正式聚合。 若一次 Session 无法完成，交付覆盖与可用的持久化版本，并让自动化编排器或同一 Notebook 下一次 Session 无损续跑；不需要新的人为科研协议批准。
14. 可直接粘贴给 Codex 的执行指令
请检查 ArcheLabs/mini-cells 的当前 R3b 分析代码及已有 Kaggle Saved Output，以本规范为唯一一次性实施请求。固定原训练 SHA a7792ef30582dfcaf324cca7e5245c8430df48b2、Run 33734fdd1558、原诊断引擎 ed2accbc90f541325ca2f6c53229e9c81475768d 和冻结 102 项任务计划。优先精确识别和读取 bingjiangli/minicells 的历史 Output（Version 65 为待核实候选；scriptVersionId=359735394 不得直接当 Version Number）。完整校验 57 个历史 Checkpoint 以及 Frozen/Development Probe、Assets、Tokenizer、Event Revision，改造 full-analysis 使用 Kaggle Output 作为只读输入，无需 HF 下载 Checkpoint。分别检查 Kaggle 旧分析 Output 和 HF 私有历史结果，回收真正已验证的旧 42 项中的可用结果，避免更换执行编排源码破坏原任务身份；缺失任务才重算。替换原每任务三次 HF commit 的默认链路，改为每任务本地原子落盘、Kaggle Saved Output 跨 Session 保存、可选低频批量 HF 备份和 GitHub 白名单聚合发布。执行冻结的 Quality39 / Functional42 / Action21，补齐纵向诊断、主对照、科学发现和实际性能报告，运行全部相关测试并记录真实资源。不得重训、不得改科学协议、不得开放 Confirmation、不得公开私有权重/逐样本结果。一次性实现、执行并提交所有可执行成果；如果 Kaggle 环境不允许完成实际分析，则交付可运行 Notebook、精确版本来源和唯一必要的 Save & Run All 操作，不得声称未执行的项目已完成。

15. 关键代码与平台依据
仓库基线（须实施时重新核查）：
- 旧版本 full_analysis.py
- 旧版本 analysis_store.py
- 旧版本 analysis_runner.py
- 冻结任务计划展开 diagnostic_schedule.py
- 训练结果报告目录
Kaggle 文档：
- Kaggle CLI：kernels files 与 kernels output —— 支持明确的历史 owner/slug/version-number 和文件名筛选。
- Kaggle Notebooks：版本与 Output —— Save & Run All 产生新版本及关联 Output；Quick Save 不等价于一次可靠完整运行。
以上为实施规范，不是本次已经完成的诊断运行回执。
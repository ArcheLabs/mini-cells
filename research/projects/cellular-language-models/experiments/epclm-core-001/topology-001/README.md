# EPCLM Core-001 Topology-001

运行 [notebook.ipynb](notebook.ipynb)：Kaggle Tesla T4、Internet、已有 GITHUB_TOKEN 与 KAGGLE_API_TOKEN Secrets，Save & Run All。代码分支 `codex/epclm-topology-001`。不需要挂载历史 Output。

## 实验范围

R1 → IC → R2 → SR → T4，分别全新训练 250000 有效预测目标、977 更新，Seed 173018021，16×128 状态、K=4、Alpha=0.1、FP32。所有组复制同一份内存初始参数；邻接索引为非持久 Tensor Buffer，不进入参数字典。R1/IC 的事件注入和参数结构来自现有 BroadcastCore，Readout 与 Update Network 不变。Degree-4 只在 Projection 前按方向取邻居 Mean。

`protocol.json` 为本次开发协议。训练沿用 `trainer.train` 的文档采样、Python Random(seed) 顺序、lane/tick 遍历、文档重置、持久状态 TBPTT、有效目标 Mask、Loss Reduction、AdamW 和 LR 调度。特别保留原调度器 `lr_multiplier(step,977,97)`，初始 step=0；不借用 Transformer 的另一种步号惯例。历史实现不修改。

数据来自 Kaggle Output 版本 142，仅下载 Tokenizer 与数据，沿用 GC001 的完整 Validation（16 文档/3322 目标）及 Development Holdout（205 文档/47403 目标）。Independent Test 不访问。没有模型恢复、周期 Checkpoint、Recovery Bundle、文件 Hash 校验或独立 Smoke/FLOPs 门禁。

训练在更新 250/500/750/977 记录完整 Validation Progression，最终验证复用本组刚测得的 977 更新结果，避免重复。训练数值记录在每次 TBPTT 更新的最后一个 Token 上，对四轮内部演化分别记录每 Cell State Norm、Delta 和 Left/Center/Right Message Norm；这是明确的抽样诊断，不声称记录了所有 Token 的内部状态。

## 诊断和比较

每组完整 Holdout；共享 Document Cluster 抽样、2000 次 Bootstrap、主指标为 Token 加权 Delta NLL，另报文档等权均值与正负差异比例。相同抽样集合用于所有比较，包含 IC/R2/SR/T4 对 R1，以及 SR 对 R2、T4 对 R2/SR。单 Seed 文档置信区间不估计训练 Seed 方差。

每组 16 篇 Validation × 三个位置，共 48 个一次性 State Pulse，Source Cell 0/4/8/12 固定轮换，方向 Seed 713；在事件已注入后、内部演化前施加扰动。记录每一轮内部更新及未来 0/1/2/4/8 Event 的状态、NLL 与 JS 差异。锚点最多前移以保留八个未来 Event，不跨文档，保留请求分位与实际位置。显著影响同时要求绝对差异 >1e-6、相对差异 >1e-6；最小扰动 1e-6，其他情况为注入后 Source State Norm 的 0.001 倍。理论因果范围按已完成内部更新数计算，孤立组源 Cell 外不允许传播。

Diversity 保存完整 Cell Pairwise Cosine、各 Cell Norm、状态方差和四轮 Delta/Message Norm。图统计由邻接表 BFS 计算，不用 NetworkX 执行训练。理论投影 FLOPs、Mean 算术量和邻居元素读取分别报告，不视为实际 GPU DRAM 流量或完整模型 FLOPs。Batch-1 推理预热 5 次、独立测量 30 次，报告单微步、四轮演化、含注入/Readout 的单 Token Decode、32 Token Prefill 的 Median/P95。辅助 Generation 为 GC001 前八个固定 Prompt、Greedy、最多 128 Token，保留文本与指标。

筛选要求 Holdout PPL 至少改善 1%、配对 CI 上界 <0、无因果越界。文档未定义“明显恶化”和“延迟合理”的数值，本实现运行前公开取 Validation PPL 比率 ≤1.01、Decode Median 比率 ≤2.0，均可在原始报告中核查；这只是开发筛选，不是科学确认。证据缺失时返回 INCOMPLETE_SCREEN，不伪装为 NO_CLEAR_TOPOLOGY_WINNER。不自动进入下一阶段。

## 预算、失败与已有结果

5400 秒执行（含五组训练/评估）、600 秒发布、1200 秒余量，总上限两小时。真实五组运行能否在预算内完成取决于 T4 速度，不预先宣称可以完成。执行器用已完成组的实际总耗时加 5% 余量判断是否还应启动下一组；不足时标记 NOT_STARTED_BUDGET，避免把剩余时间浪费在必然中断的训练上。时间耗尽时保留每组状态和已有报告，不扩大预算；失败诊断或失败组不阻断其他尚可执行的项目。

每组完成训练后保存一次 `<arm>-final-model.pt`，路径与 Run ID 记录在 decision 中；权重只留 Kaggle Private Output，不进入 Git。下一次启动仅复用同一 Source/协议下已经完成训练的**报告记录**，不下载或加载任何旧模型/优化器/持久状态；已完成训练但诊断失败的组也不会被自动重新训练。未完成训练组从头开始。报告通过现有固定 Source 的结果分支找回，非 Checkpoint Recovery；每个完成模型的获取位置保持为该组原 Kaggle Output。源码变更是新的实验身份，不自动复用旧源码结果。

结果分支 `kaggle/epclm-core-001-topology-001-<source12>`，目录 `artifacts/experiments/epclm-core-001/topology-001/`。只发布报告/协议。执行结束后发布器核实远端提交，才在本地和日志中标记发布完成；远端提交内 publication_status=PENDING 是提交时尚未完成远端核验的真实状态。

## 最小本地验证

仅三个紧凑 CPU 测试覆盖文档列出的十类必要检查，外加微型 TBPTT 尾批和配对统计。没有本地正式训练、CPU Benchmark、50 Update Smoke，也没有访问旧 CP-001 或其他 Cell Graph 实验的模型结果。真实五组训练与测量由本 Notebook 在 T4 上执行。

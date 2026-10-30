# Stage 2 预注册准备 001

结论：工程提交一致性问题已补充验证；**科学筛查继续 NO-GO**。本轮不新增 Optimizer 更新，不做质量评价、数据重选、GPU 或 Confirmation 质量访问。阶段 1 PASS 和历史 C-sync NO-GO 保持原样。

## 1. Connection 更新证据

`implementation/connection_commit.py` 提供只观察、不改变决策的 VerifiedEngine。它调用原有 interpolation_audit.Engine，由原 FAST Engine 完成实际参数加载。调用前复制 Connection，原提交完成后、写入 Checkpoint 前检查：

- 接受：用私有候选包的 original/full 参数与选定 scale 重新计算原 FP32 插值，逐字节哈希同时匹配实际参数和 proposal.selected_connection_sha256；要求非零变化。
- 拒绝或没有候选：更新前后 Connection 哈希必须相同。
- 写盘后：读取实际 Checkpoint，再核对 Connection 哈希和保存的 receipt。
- 收据写入 curve/Checkpoint；不改变 Q 优化、原准入、尺度选择、主 Cell 更新或已有统计字段。

`formal_connection_commits: 0` 是 scan.run 的计数：扫描本身只评估模型副本，不提交主模型。外层 Engine 收到 result 后才 load_state_dict。新增独立的 `outer_engine_connection_commits`，不会把工程提交解释为科学授权，也不改写旧报告的 scan 计数。

对保留的阶段 1 完成 Checkpoint 与 candidate.pt 进行了只读核验。其完整 curve 与已发布 stage1-engineering-002 的 CLM curve 相同：选中 `0.00390625`，更新前 `a078e830…`，实际更新后及插值目标均为 `afb9d4a0…`；对应扫描行的 hash、passed、nonempty-path 及 formal_selected_scale 也一致。见 `reports/stage2-preparation-001/RETAINED_CHECKPOINT_PROOF.json`。这证明该次工程接受已经落到真实状态，不证明学习方向的质量收益。

接受/拒绝两条控制流测试使用原外层 Engine，但注入已知候选，并冻结 Cell。没有候选优化或 Cell Optimizer 调用；测试确认原 Engine 真正加载/不加载参数，以及实际保存文件一致。纯张量反例还验证错误起点、错误接受参数、拒绝后变更都会阻断。测试中使用的人工方向不冒充学习候选；真实学习候选证据来自上述保留 Checkpoint。

## 2. Trust 上限与执行输入

新 manifest 显式记录 action_change_max=0.1、KL max=0.01、path_change_max=0.2；同时固定旧 SCREENING_MANIFEST.json 的完整 SHA-256。`legacy_configuration()` 在**每次读取**时验证原始字节哈希，并检查三上限与新 manifest 相等。

两个读取点（模型校准前恢复上限、CLM 工程 Engine 配置构造）均通过此函数。旧文件仍仅是冻结的工程运行模板，不是本轮科学注册清单。来源文件、原 FAST Engine、stageb_engine、观察器、新 manifest 和测试文件均纳入本轮 SOURCE_MANIFEST 输入哈希。篡改源文件的测试会直接失败。

## 3. 与原协议的关系

本轮是依据 `architecture-benchmark/PROTOCOL.v1.zh-CN.md` 执行摘要、第 2 节及第 11 节和后续阶段 1 授权建立的**独立三架构探索实验**。旧 CLM003 根目录五组协议继续保留，其研究目标没有被声明完成。

- 本轮回答共同初始化、共同任务与固定预算下 CLM/MoE/Dense 的质量及成本差异。
- 本轮不回答学习 Connection 是否优于冻结随机或随机重连，也不将三组工程测试当作旧 A/B/C 的替代结果。
- 旧 `CLM003-ABC-SCREEN-001` 的父权重、数据、Seed、预算和 Confirm 均不构成本轮科学注册。
- C-sync NO-GO 未解除；按已批准的新架构方案，它不是三架构基准前置条件。
- 本轮不得宣称完整的五组 Connection 因果研究、非劣、效率优势或大规模 LLM 竞争力已经成立。

## 4. 已记录的人类选择与仍需审阅的算法修订

用户已选择：三组共同训练公共参数；固定训练 Token，分别记录实际成本，不宣称计算效率优势。

**公共参数选择尚不等于算法修订通过。** 当前工程实现保留 frozen-public 行为，未擅自解冻。将其变为 trainable-public 需要独立审阅：

1. 明确公共组包含 Embedding/tied head、position、init、memory、context、Norm/final；CLM frozen Gate 是否仍属于架构专属固定 scorer，须单列，不能静默划入公共组。
2. 三组公共组件应使用同一规范初始化、优化器/调度/clip、有效标签及相同恢复语义；tied Embedding/head 只能注册一次。
3. CLM 原 Q 反事实优化会固定 Cell/公共组件；公共参数只能在主更新改变，候选优化、准入评估前后不得改变它们。主更新的参数所有权、冻结指纹与 Checkpoint 需要修订。
4. 旧 Engine 的 Cell-only 主优化器、mask、frozen_sha、公有参数不变断言及当前计费均须同步修改，不能简单 requires_grad=True 就称为有效联合训练。
5. 需要新的参数曝光和恢复测试。**这些变更本轮没有实施或运行**，也不使用 frozen-public 工程 NLL 支持未来 trainable-public 结论。

固定 Token 方案已记录为拟用比较模式，不预先冻结具体 Token 数、步数或 Seed。Cell 调用 8/16/48 及 CLM 候选成本分别报告；共同数据预算不等同计算匹配，完整效率测量作为独立门禁。

## 5. 未闭合项与停止边界

`STAGE2_PREPARATION.json` 逐项记录状态。以下仍为空或待审阅：科学模型规模；Seed 列表；Token/步数/更新计划；失败运行处理；Development 推进阈值；非劣界限；最小效率收益阈值；新科学 Train/Development/Calibration/封存 Confirmation 身份及内容哈希；自然语言语料选择。

未据工程 NLL 推导阈值、自动选范围、创建新的科学数据或继承旧 Confirm。若本次仅作筛查、暂不设非劣界限，必须明确禁止非劣结论；这仍需在注册中写明。不能伪造“最终科学清单已冻结”。新旧 manifest 均保持 scientific_frozen=false、stage2_authorized=false。

本轮执行 5 项零 Optimizer 工程测试并读取保留状态，测试最终全通过；不重复阶段 1 训练冒烟或旧 Phase 0。新的 CI 仅运行这组准备测试及纯损失/确认门禁测试，不自动触发训练。停止并等待研究参数及公共参数算法修订审阅，阶段 2 仍未授权。

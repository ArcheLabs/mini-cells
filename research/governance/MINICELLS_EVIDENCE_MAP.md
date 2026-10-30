# MiniCells Evidence Map

维护日期：2026-10-30（Asia/Shanghai）。**历史迁移正在进行；自动发现覆盖不等于逐项科学结论认证。** 固定 main 快照、路径与源哈希见[完整性审计](audits/EVIDENCE_MAP_COMPLETENESS.json)。本文件是统一视图，原始协议/结果仍是事实来源。未经合并的分支不作为 main 的已验证成果。

## 1. Project Research Overview

长期目标：探索 Cell 组合、局部计算、动态结构及持续学习是否能构成有效模型。当前问题：功能是否可寻址与保护、模块路径是否有任务价值、实际计算成本是否改善。实际研究方向以[项目 catalog](catalog.yaml)为依据，含 cellular-computing、continual-learning、cellular-language-models、safe-model-evolution、clm-native；方向名称不是已实现能力。

## 2. Executive Research Summary

**观察事实：** Core005 在固定线性可写、固定特征及显式路由的合成环境报告 3/3 Seed 通过；Core009B-1 confirmation 报告三个 Seed 的 carrier causal sufficiency。Core009C discovery 的两个 Seed 未找到满足全部门槛的稀疏/局部效应配置。Native M3W-0 是 Checkpoint-only 恢复诊断，并非新模型训练。见下方各 Claim 的直接结果链接。

**研究解释：** 已获得的窄范围机制证据，不足以证明真实语言模型的无回放持续学习、自然模块边界或端到端硬件优势。部分局部结构可测量，不意味着其可部署地寻址、重用和保护。早期实验、后续正式 Core、实际模型诊断的证据层级不可并列计作独立成功。

**研究决策：** 本视图不新批准研究路线或资源投入。当前独立 sparse-path 研究的 README 说明它不依赖已暂停的 CLM-Bridge 昂贵后续训练；这一原始状态不因建立 Map 改变。大部分历史目录的逐阶段结果、注册时序及演进关系仍待核实，不能将自动索引中的 UNKNOWN 填成成功。

## 3. Evidence Overview

|Claim ID|Research Claim|Status|Evidence|Scope / Limitations|
|---|---|---|---|---|
|EM-C001|Core005 的固定线性可写合成设置达到了其子空间证书/分裂门槛|ESTABLISHED|[Core005 RESULTS](../projects/continual-learning/experiments/core-005-subspace-certified-mitosis/artifacts/RESULTS.md)|报告 3/3；不是任意非线性 Cell 或真实 LM 证书|
|EM-C002|Core009B-1 冻结确认设置下 carrier causal sufficiency 得到支持|SUPPORTED|[009B-1 confirmation](../projects/continual-learning/experiments/core-009b1-carrier-causal-sufficiency/artifacts/confirmation/RESULTS.md)|81011/81012/81013；结果明确不证明 reuse/addressability/growth|
|EM-C003|Core009C 该轮固定稀疏/局部候选能在全部 discovery Seed 通过门槛|CONTRADICTED|[009C discovery](../projects/continual-learning/experiments/core-009c-sparse-local-effect-geometry/artifacts/discovery/RESULTS.md)|仅该搜索范围、81201/81202；不是不可能性定理|
|EM-C004|M3W-0 的恢复诊断证明了端到端 Native CLM 训练成功|NOT_ESTABLISHED|[M3W-0 RESULTS](../projects/cellular-language-models/experiments/native-clm-v0-m3w0-write-drift-restoration/artifacts/RESULTS.md)|记录 Native training=False、scientific decision=False|
|EM-C005|当前项目已验证通用持续语言学习或数量级 LLM 加速|NOT_ESTABLISHED|[历史研究账本](audits/RESEARCH_LEDGER.md)、[native sparse-path 定位](../projects/clm-native/README.md)|历史账本明确边界；新 sparse-path 不声称 LM/10× 效率|

|EM-C006|Core006 的全部三个正式 Seed 达到持续可塑性系统门槛|CONTRADICTED|[Core006 decision](../projects/continual-learning/experiments/core-006-real-representation-continual-plasticity/artifacts/decision.json)|0/3；证书降低遗忘与几何子假设为 true，不把系统失败写成所有机制失败|

状态全集：ESTABLISHED、SUPPORTED、PARTIALLY_SUPPORTED、NOT_ESTABLISHED、CONTRADICTED、UNTESTED、SUPERSEDED。Outcome 不等于 Claim Status；本次未凭自动状态字符串新增 ESTABLISHED。UNKNOWN/NOT_RECORDED 是缺失信息标记，CONFLICT_REVIEW_REQUIRED 是证据冲突标记，不能替代上述 Claim 状态。

## 4. Research Lines

### Cellular Computing

#### 1. Research Objective
稀疏 Cell、组织、路由与增长。来源：[项目 README](../projects/cellular-computing/README.md)。

#### 2. Original Hypothesis
早期可构造模块机制是否形成有用计算。这是对项目问题的摘要；未找到可认证的最早版本时，原始提出日期为 UNKNOWN。

#### 3. Research Evolution
|Experiment / records|Objective|Outcome|Contribution|
|---|---|---|---|
|[完整工作单元索引](audits/EVIDENCE_MAP_INVENTORY.md)|稀疏 Cell、组织、路由与增长|逐项 Outcome 见原始结果；尚未核定者 UNKNOWN|历史 001–026 的原始文档及后续治理账本|

真实日期/直接因果关系未核实者不编排为确定时间线。编号顺序不是演进证据。

#### 4. Established Findings
仅采用 Evidence Overview 中明确限定的主张；未有对应 Claim 的记录为 UNKNOWN，不凭项目 README 判定充分验证。

#### 5. Partially Supported Findings
早期探索证据强度异质，尚不能把机制构造当作通用学习成功。证据来源见本线原始记录及下方 Source Records。

#### 6. Failed or Unresolved Hypotheses
宽泛项目目标没有因局部机制通过自动成立。具体失败与缺失结果均保留在完整盘点中；失败原因无来源时 UNKNOWN。

#### 7. Known Limitations
合成/线性、工程/科学、部分/完整、参数/实际时延必须区分；不跨模型、数据和硬件外推。

#### 8. Current Interpretation
【研究解释】早期探索证据强度异质，尚不能把机制构造当作通用学习成功；不作新增资源决策。

#### 9. Open Questions
是否可超出原设置复现？与其他方向的真实依赖是否有原始声明？参见第 8 节。

#### 10. Related Research Lines
与其余四线 RELATED；未有原始记录支撑时不标 EXTENDS/REPLICATES/SUPERSEDES。

#### 11. Source Records
[项目入口](../projects/cellular-computing/README.md)、[实验完整索引](audits/EVIDENCE_MAP_INVENTORY.md)、[历史综合账本](audits/RESEARCH_LEDGER.md)。具体结果优先于综合判断。

### Continual Learning

#### 1. Research Objective
知识写入、功能保护、分裂、重用和可部署寻址。来源：[项目 README](../projects/continual-learning/README.md)。

#### 2. Original Hypothesis
局部参数/效应是否允许受控更新而不损坏旧功能。这是对项目问题的摘要；未找到可认证的最早版本时，原始提出日期为 UNKNOWN。

#### 3. Research Evolution
|Experiment / records|Objective|Outcome|Contribution|
|---|---|---|---|
|[完整工作单元索引](audits/EVIDENCE_MAP_INVENTORY.md)|知识写入、功能保护、分裂、重用和可部署寻址|逐项 Outcome 见原始结果；尚未核定者 UNKNOWN|Core006 decision 明确 parent_experiment 为 Core005（EXTENDS）；Core009 后续关系以账本为线索，未核实的不按编号推断|

真实日期/直接因果关系未核实者不编排为确定时间线。编号顺序不是演进证据。

#### 4. Established Findings
仅采用 Evidence Overview 中明确限定的主张；未有对应 Claim 的记录为 UNKNOWN，不凭项目 README 判定充分验证。

#### 5. Partially Supported Findings
Core005 窄设置通过；009B-1 支持 carrier；009C 未找到满足门槛的局部配置。证据来源见本线原始记录及下方 Source Records。

#### 6. Failed or Unresolved Hypotheses
宽泛项目目标没有因局部机制通过自动成立。具体失败与缺失结果均保留在完整盘点中；失败原因无来源时 UNKNOWN。

#### 7. Known Limitations
合成/线性、工程/科学、部分/完整、参数/实际时延必须区分；不跨模型、数据和硬件外推。

#### 8. Current Interpretation
【研究解释】Core005 窄设置通过；009B-1 支持 carrier；009C 未找到满足门槛的局部配置；不作新增资源决策。

#### 9. Open Questions
是否可超出原设置复现？与其他方向的真实依赖是否有原始声明？参见第 8 节。

#### 10. Related Research Lines
与其余四线 RELATED；未有原始记录支撑时不标 EXTENDS/REPLICATES/SUPERSEDES。

#### 11. Source Records
[项目入口](../projects/continual-learning/README.md)、[实验完整索引](audits/EVIDENCE_MAP_INVENTORY.md)、[历史综合账本](audits/RESEARCH_LEDGER.md)。具体结果优先于综合判断。

### Cellular Language Models

#### 1. Research Objective
原生与 Hybrid LM、路径/路由和端到端验证。来源：[项目 README](../projects/cellular-language-models/README.md)。

#### 2. Original Hypothesis
受控 Cell 机制能否迁移到语言建模。这是对项目问题的摘要；未找到可认证的最早版本时，原始提出日期为 UNKNOWN。

#### 3. Research Evolution
|Experiment / records|Objective|Outcome|Contribution|
|---|---|---|---|
|[完整工作单元索引](audits/EVIDENCE_MAP_INVENTORY.md)|原生与 Hybrid LM、路径/路由和端到端验证|逐项 Outcome 见原始结果；尚未核定者 UNKNOWN|早期 routing 001–006、CLM-001/002/003 及 Native 诊断，各阶段需分别解释|

真实日期/直接因果关系未核实者不编排为确定时间线。编号顺序不是演进证据。

#### 4. Established Findings
仅采用 Evidence Overview 中明确限定的主张；未有对应 Claim 的记录为 UNKNOWN，不凭项目 README 判定充分验证。

#### 5. Partially Supported Findings
M3W-0 只读恢复提供根写入主导证据，不是新 Native LM 训练结论。证据来源见本线原始记录及下方 Source Records。

#### 6. Failed or Unresolved Hypotheses
宽泛项目目标没有因局部机制通过自动成立。具体失败与缺失结果均保留在完整盘点中；失败原因无来源时 UNKNOWN。

#### 7. Known Limitations
合成/线性、工程/科学、部分/完整、参数/实际时延必须区分；不跨模型、数据和硬件外推。

#### 8. Current Interpretation
【研究解释】M3W-0 只读恢复提供根写入主导证据，不是新 Native LM 训练结论；不作新增资源决策。

#### 9. Open Questions
是否可超出原设置复现？与其他方向的真实依赖是否有原始声明？参见第 8 节。

#### 10. Related Research Lines
与其余四线 RELATED；未有原始记录支撑时不标 EXTENDS/REPLICATES/SUPERSEDES。

#### 11. Source Records
[项目入口](../projects/cellular-language-models/README.md)、[实验完整索引](audits/EVIDENCE_MAP_INVENTORY.md)、[历史综合账本](audits/RESEARCH_LEDGER.md)。具体结果优先于综合判断。

### Safe Model Evolution

#### 1. Research Objective
预训练模型修改、压缩、组合、回滚的可审计安全性。来源：[项目 README](../projects/safe-model-evolution/README.md)。

#### 2. Original Hypothesis
修改边界是否可定位并保护已有功能。这是对项目问题的摘要；未找到可认证的最早版本时，原始提出日期为 UNKNOWN。

#### 3. Research Evolution
|Experiment / records|Objective|Outcome|Contribution|
|---|---|---|---|
|[完整工作单元索引](audits/EVIDENCE_MAP_INVENTORY.md)|预训练模型修改、压缩、组合、回滚的可审计安全性|逐项 Outcome 见原始结果；尚未核定者 UNKNOWN|PCU、conversion、functional-boundary 等按各自原始记录索引；完整先后仍 UNKNOWN|

真实日期/直接因果关系未核实者不编排为确定时间线。编号顺序不是演进证据。

#### 4. Established Findings
仅采用 Evidence Overview 中明确限定的主张；未有对应 Claim 的记录为 UNKNOWN，不凭项目 README 判定充分验证。

#### 5. Partially Supported Findings
工程通过不能推出安全演化已普遍解决。证据来源见本线原始记录及下方 Source Records。

#### 6. Failed or Unresolved Hypotheses
宽泛项目目标没有因局部机制通过自动成立。具体失败与缺失结果均保留在完整盘点中；失败原因无来源时 UNKNOWN。

#### 7. Known Limitations
合成/线性、工程/科学、部分/完整、参数/实际时延必须区分；不跨模型、数据和硬件外推。

#### 8. Current Interpretation
【研究解释】工程通过不能推出安全演化已普遍解决；不作新增资源决策。

#### 9. Open Questions
是否可超出原设置复现？与其他方向的真实依赖是否有原始声明？参见第 8 节。

#### 10. Related Research Lines
与其余四线 RELATED；未有原始记录支撑时不标 EXTENDS/REPLICATES/SUPERSEDES。

#### 11. Source Records
[项目入口](../projects/safe-model-evolution/README.md)、[实验完整索引](audits/EVIDENCE_MAP_INVENTORY.md)、[历史综合账本](audits/RESEARCH_LEDGER.md)。具体结果优先于综合判断。

### Bounded Native Sparse Paths

#### 1. Research Objective
状态依赖非局部 Cell 通信的质量与真实成本。来源：[项目 README](../projects/clm-native/README.md)。

#### 2. Original Hypothesis
相对信息匹配的 recurrent/centralized controls 是否有收益。这是对项目问题的摘要；未找到可认证的最早版本时，原始提出日期为 UNKNOWN。

#### 3. Research Evolution
|Experiment / records|Objective|Outcome|Contribution|
|---|---|---|---|
|[完整工作单元索引](audits/EVIDENCE_MAP_INVENTORY.md)|状态依赖非局部 Cell 通信的质量与真实成本|逐项 Outcome 见原始结果；尚未核定者 UNKNOWN|独立 native-sparse-path；README 明确不是 CLM-Bridge 后续依赖|

真实日期/直接因果关系未核实者不编排为确定时间线。编号顺序不是演进证据。

#### 4. Established Findings
仅采用 Evidence Overview 中明确限定的主张；未有对应 Claim 的记录为 UNKNOWN，不凭项目 README 判定充分验证。

#### 5. Partially Supported Findings
该研究为 synthetic computation，不声称 LM 能力或 10× LLM 效率。证据来源见本线原始记录及下方 Source Records。

#### 6. Failed or Unresolved Hypotheses
宽泛项目目标没有因局部机制通过自动成立。具体失败与缺失结果均保留在完整盘点中；失败原因无来源时 UNKNOWN。

#### 7. Known Limitations
合成/线性、工程/科学、部分/完整、参数/实际时延必须区分；不跨模型、数据和硬件外推。

#### 8. Current Interpretation
【研究解释】该研究为 synthetic computation，不声称 LM 能力或 10× LLM 效率；不作新增资源决策。

#### 9. Open Questions
是否可超出原设置复现？与其他方向的真实依赖是否有原始声明？参见第 8 节。

#### 10. Related Research Lines
与其余四线 RELATED；未有原始记录支撑时不标 EXTENDS/REPLICATES/SUPERSEDES。

#### 11. Source Records
[项目入口](../projects/clm-native/README.md)、[实验完整索引](audits/EVIDENCE_MAP_INVENTORY.md)、[历史综合账本](audits/RESEARCH_LEDGER.md)。具体结果优先于综合判断。

## 5. Experiment Registry

[完整历史盘点附录](audits/EVIDENCE_MAP_INVENTORY.md)登记所有扫描发现的工作目录、原始身份字符串、协议/结果链接及待核定字段；它是本 Map 的长表附录，不是第二套注册机制。原有 [registry](registry.yaml) 保持不变。

迁移均标 RETROSPECTIVE，只说明本次事后整理，不伪造原实验注册时点。目录与身份声明不等同独立实验，重复身份/别名待审阅。确认为例行软件测试者不登记为科研实验；用于改变研究主张的验证记录必须保留。

## 6. Cross-Research Findings

【研究解释】可测量局部机制与完整可部署系统之间存在证据缺口：[Core005 的固定线性边界](../projects/continual-learning/experiments/core-005-subspace-certified-mitosis/artifacts/RESULTS.md)与[009B-1 不证明 addressability 的边界](../projects/continual-learning/experiments/core-009b1-carrier-causal-sufficiency/artifacts/confirmation/RESULTS.md)共同支持应分层解释，不支持通用不可能性。

【研究解释】恢复/归因诊断与模型训练结论必须分开：[M3W-0](../projects/cellular-language-models/experiments/native-clm-v0-m3w0-write-drift-restoration/artifacts/RESULTS.md)与[项目历史账本](audits/RESEARCH_LEDGER.md)均限定了证据层级。其他跨线因果判断尚 UNKNOWN。

## 7. Known Limitations

- 线性固定特征证书不能外推为任意非线性组合（EM-C001）。
- Carrier sufficiency 不等于重用、可寻址、增长或完整 CLM（EM-C002）。
- 一轮局部几何搜索失败不证明局部结构绝不存在（EM-C003）。
- Checkpoint-only counterfactual restoration 不证明新训练成功（EM-C004）。
- 稀疏结构/参数指标不能直接推断实际 LLM 延迟（EM-C005 的来源边界）；本 Map 不新增硬件实验。

## 8. Open Research Questions

|Question ID|Research Question|Related Claims|Existing Evidence|Knowledge Gap|
|---|---|---|---|---|
|EM-Q001|线性证书机制能否迁移到非线性 LM？|EM-C001|Core005|跨设置确认|
|EM-Q002|Carrier 是否可部署地寻址和复用？|EM-C002|009B-1、009C|几何与可部署机制之间的桥接|
|EM-Q003|局部路径能否改善质量且降低真实成本？|EM-C005|各路由研究原始索引|公平控制与硬件证据|
|EM-Q004|恢复诊断可否支持有效新学习？|EM-C004|M3W-0|训练与功能保护证据|
|EM-Q005|历史修订之间哪些是真正因果后续？|全部|完整盘点/历史账本|日期、别名和关系逐项审阅|

问题不是已批准实验；本文件不创建训练任务。

## 9. Current Research Assessment

【研究解释】短期可依赖的是有明确边界的机制结果、测量和工程资产；不能从长期研究价值推断短期产品价值。当前瓶颈包括功能寻址/重用、从受控设置迁移、实际成本可比性，以及历史证据的语义整理欠账。

【研究决策】维持原始授权/停止状态，不因 Evidence Map 重新分配资源。没有足够信息的路线先标待审阅，不自动判继续、暂停或放弃。未经合并的新提案不是 main 成果。

## 10. Research Decision History

|Decision ID|Date|Decision|Evidence|Rationale|Revisit Condition|
|---|---|---|---|---|---|
|EM-D001|UNKNOWN|账本保留早期探索与现代正式验证的证据边界|[2026-09-03 账本](audits/RESEARCH_LEDGER.md)|受控结果不外推|出现对应跨设置直接证据|
|EM-D002|UNKNOWN|native-sparse-path 作为独立研究，昂贵 CLM-Bridge 后续仍暂停|[项目 README](../projects/clm-native/README.md)|原始项目声明；不是本次新决定|以原研究决策的后续修订为准|
|EM-D003|2026-10-30|建立统一证据视图，历史资料不改写|用户建设规范与本次 Git 提交|保持长期研究记忆|主要研究阶段结束时复核|

关系标签采用 EXTENDS/MODIFIES/REPLICATES/TESTS_FAILURE_MODE_OF/CONTRADICTS/SUPERSEDES/INDEPENDENT/RELATED；没有原文支持时仅 RELATED 或 UNKNOWN。取代旧判断必须保留旧 Claim 和原来源。

## 11. Update History

|Date|Change|Boundary|
|---|---|---|
|2026-10-30|建立统一结构、main 快照盘点、6 条初始 Claim、5 条研究线、5 个开放问题|语义历史迁移尚未完成；统计与覆盖审计见附录，不宣称最终验收通过|

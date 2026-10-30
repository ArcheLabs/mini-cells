# MiniCells Evidence Map
## 建设规范与实施要求

**Document Type:** Research Governance Specification  
**Project:** ArcheLabs / MiniCells  
**Target Directory:** `research/governance/`  
**Primary Document:** `MINICELLS_EVIDENCE_MAP.md`  
**Status:** Proposed  
**Purpose:** Establish a unified, traceable and continuously maintained research evidence system.

---

# 1. 建设背景

MiniCells 已经建立了较为完整的研究治理体系，包括实验设计、预注册、验证、审计和正式结果记录。

但随着研究方向和实验数量不断增加，出现了一个新的问题：

**虽然每项实验都有自己的记录，但缺少一个能够反映整个项目研究状态的统一、清晰、可读的综合文档。**

这导致：

1. 研究结论分散在不同实验目录和报告中，难以快速理解。
2. 不同阶段的实验结果容易被并列讨论，忽略它们之间的因果和演进关系。
3. 早期实验的失败、后续方案调整和新的实验结果没有形成连续的研究脉络。
4. 局部实验成功容易被扩大解释，超出原始实验实际支持的范围。
5. 研究方向发生变化后，难以准确还原当时的证据、判断和决策依据。
6. 新研究启动时，容易重复已经开展过的工作，或者忽略已有实验暴露的问题。

因此，需要建立 MiniCells Evidence Map。

它不是新的实验治理系统，而是现有研究治理体系之上的统一证据视图。

# 2. 核心目标

Evidence Map 必须实现以下目标。

## 2.1 统一研究事实

能够从一个入口了解：

- MiniCells 已经开展过哪些研究。
- 每项研究验证了什么。
- 哪些实验通过、失败或部分通过。
- 哪些技术能力已经得到验证。
- 哪些研究假设尚未成立。
- 哪些问题仍然没有答案。

## 2.2 保持研究连续性

能够理解每条研究路线的完整演进：

原始假设 → 实验设计 → 实验结果 → 失败原因 → 方案调整 → 后续验证 → 当前结论。

不能将不同阶段的实验简单并列，然后脱离历史背景进行整体判断。

## 2.3 防止结论漂移

所有重要研究结论都必须能够追溯到原始实验记录。

当新的实验改变了原有判断时，必须明确说明：

- 原来的判断是什么。
- 新证据是什么。
- 为什么需要修改判断。
- 原有结论是否仍然在特定条件下成立。

不得通过覆盖历史文档的方式消除旧结论。

## 2.4 支持未来研究决策

Evidence Map 应当帮助研究者回答：

- 我们真正已经证明了什么？
- 哪些机制仍然只是理论假设？
- 当前最重要的未知问题是什么？
- 哪些实验已经回答了部分问题？
- 哪些研究路线值得继续？
- 哪些研究路线应该暂停或重新设计？

Evidence Map 不负责自动决定研究方向，但必须提供可靠的决策依据。

# 3. 基本设计原则

## 3.1 原始实验记录是事实来源

Evidence Map 不得取代现有实验协议、正式结果、审计报告和工程记录。

原始记录继续保留在现有目录中。

Evidence Map 负责：

- 摘要。
- 分类。
- 关联。
- 解释。
- 证据状态。
- 研究判断。

不得为了建立 Evidence Map 而重写原始实验结果。

## 3.2 严格区分三个层次

所有重要内容必须明确区分：

**A. Observed Facts — 观察事实**

实验实际发生了什么？

例如：

- 使用了什么模型。
- 使用了什么数据集。
- 执行了多少个 Seed。
- 实际指标是多少。
- 是否达到预注册门槛。

只能引用原始记录。

**B. Research Interpretation — 研究解释**

实验结果意味着什么？

例如：

- 某种保护机制没有达到预期效果。
- 某种拓扑结构可能存在功能局部性问题。
- 参数稀疏并没有转化为实际延迟收益。

研究解释必须说明依据和适用范围，不得伪装成实验事实。

**C. Research Decision — 研究决策**

根据现有证据，我们决定：

- 继续。
- 修改。
- 暂停。
- 放弃。
- 进一步调查。

研究决策必须有明确的证据依据，并保留历史版本。

## 3.3 不允许扩大实验结论

每项实验都存在适用范围。

例如：

- 线性算子的顺序组合通过，不代表任意非线性 Cell 都可以组合。
- 一个受控任务通过，不代表真实语言模型中的持续学习已经成立。
- 参数量减少，不代表 GPU 推理速度提高。
- Cell 能够局部执行，不代表它能够形成有效的自组织网络。

Evidence Map 必须记录每项证据的适用范围和不能支持的结论。

## 3.4 失败实验同样重要

以下内容必须纳入：

- 正式实验通过。
- 正式实验失败。
- 部分通过。
- 未达到预注册门槛。
- 实验无效。
- 实验中止。
- 探索性实验。
- 工程验证。
- 已被后续证据修正的历史结论。

不得只登记成功的实验。

## 3.5 不追求形式上的完整

Evidence Map 应当优先保证准确性、可读性和研究价值。

不要为了填满表格而猜测缺失信息。

无法确认的内容统一标记：

`UNKNOWN` 或 `NOT_RECORDED`

存在相互矛盾的记录时：

`CONFLICT_REVIEW_REQUIRED`

不得自行选择其中一个版本作为事实。

# 4. 文件与目录规划

保持现有 `research/governance/` 结构，避免建立第二套治理系统。

建议采用以下结构：

```text
research/
└── governance/
    ├── README.md
    │
    ├── MINICELLS_EVIDENCE_MAP.md
    │
    ├── validations/
    │
    └── audits/
```

其中：

### README.md

现有治理体系的入口。

需要补充：

- Evidence Map 的用途。
- Evidence Map 的位置。
- 实验预注册和关闭时必须更新 Evidence Map 的要求。
- 原始实验记录与 Evidence Map 的关系。

保留现有 README 的其他内容，不进行无关重构。

### MINICELLS_EVIDENCE_MAP.md

整个 MiniCells 项目的统一研究证据总览。

所有研究方向、重要实验、当前结论和未解决问题都应该能够从这里找到。

不在本阶段创建大量重复的实验数据库或额外治理系统。

如果未来文档规模确实无法维持可读性，再考虑拆分。

# 5. Evidence Map 的内容结构

`MINICELLS_EVIDENCE_MAP.md` 必须采用以下章节结构。

## 5.1 Project Research Overview

简要说明 MiniCells 的研究目标和当前研究定位。

需要区分：

- Long-term Research Goals。
- Current Research Questions。
- Current Research Directions。

不要把尚未验证的长期愿景写成已经实现的能力。

这一部分应当保持简洁。

## 5.2 Executive Research Summary

这是整个文档最重要的部分。

应当让没有参与近期实验的研究者，在几分钟内理解项目的当前状态。

至少回答：

1. 当前已经获得哪些确定性成果？
2. 当前最重要的技术限制是什么？
3. 哪些核心假设尚未成立？
4. 哪些问题已经被实验否定或显著削弱？
5. 哪些研究问题仍然开放？
6. 当前研究方向有什么重要变化？

必须以实际证据为依据。

不要将历史成果简单罗列为项目进展。

## 5.3 Evidence Overview

建立统一的研究主张与证据状态表。

建议使用以下结构：

| Claim ID | Research Claim | Status | Evidence | Scope / Limitations |
|---|---|---|---|---|

每条 Claim 必须是一个边界明确、可以被证据支持或反驳的研究主张。

避免使用过于宽泛的表述。

例如：

不建议：

`CLM can perform continual learning.`

建议：

`The tested fixed-topology Native CLM configuration did not satisfy the registered continual-learning retention threshold.`

每条 Claim 必须具有稳定 ID，不因文档重新组织而改变。

### 证据状态定义

使用以下统一状态：

| Status | 含义 |
|---|---|
| ESTABLISHED | 在明确限定的条件和范围内得到充分验证 |
| SUPPORTED | 已有证据支持，但仍需进一步验证 |
| PARTIALLY_SUPPORTED | 部分机制成立，完整主张尚未成立 |
| NOT_ESTABLISHED | 已经进行相关研究，但尚未证明该主张 |
| CONTRADICTED | 当前证据在明确范围内与该主张相矛盾 |
| UNTESTED | 尚无直接实验验证 |
| SUPERSEDED | 原有判断已被后续证据或更准确的判断取代 |

注意：

- NOT_ESTABLISHED 不等于证明不可能。
- CONTRADICTED 必须明确限定实验条件。
- SUPERSEDED 必须保留原始判断及其历史依据。
- ESTABLISHED 不代表在所有模型、任务和条件下成立。

## 5.4 Research Lines

这是 Evidence Map 的核心章节。

按照研究方向组织全部实验，展示各方向的演进过程。

初始分类由历史实验盘点决定，不得仅根据当前讨论预设所有分类。

可以考虑以下方向：

- Native CLM。
- Constructive CLM。
- Hybrid CLM。
- Cellization / Expert Conversion。
- Cell Routing。
- Emergent Topology。
- Cell Subgraph / Compositional Computation。
- Continual Learning。
- Safe Model Evolution。
- Hardware-aware Inference。
- 其他实际存在的研究方向。

这些名称仅作为候选分类，不要求全部建立独立研究线。

每条 Research Line 必须包含以下内容。

### Research Line Template

```markdown
## [Research Line Name]

### 1. Research Objective

该研究方向试图解决什么问题？

### 2. Original Hypothesis

最初的研究假设是什么？

### 3. Research Evolution

按照时间顺序记录重要实验：

| Experiment | Objective | Outcome | Contribution |
|---|---|---|---|

### 4. Established Findings

已经得到充分验证的结论。

### 5. Partially Supported Findings

部分成立但尚未完全验证的结论。

### 6. Failed or Unresolved Hypotheses

失败或仍未解决的核心假设。

### 7. Known Limitations

当前已经发现的技术限制。

### 8. Current Interpretation

结合全部历史证据，对当前研究状态进行综合解释。

### 9. Open Questions

尚未回答的问题。

### 10. Related Research Lines

与其他研究方向的关系。

### 11. Source Records

所有重要原始实验和验证记录的链接。
```

研究演进必须按照真实时间和因果关系组织。

不能仅仅按照实验编号排列，也不能将后续修订实验与原始实验视为相互独立的成果。

## 5.5 Experiment Registry

建立所有实验的统一索引。

每项实验至少记录：

| 字段 | 说明 |
|---|---|
| Experiment ID | 原始实验 ID |
| Name | 实验名称 |
| Research Line | 所属研究方向 |
| Registration Type | 注册类型 |
| Experiment Type | 实验类型 |
| Date | 实验时间 |
| Protocol | 原始协议链接 |
| Result | 正式结果链接 |
| Outcome | 实际执行结果 |
| Related Experiments | 相关实验 |
| Main Finding | 核心发现 |
| Current Relevance | 对当前研究的意义 |

### Registration Type

- `PROSPECTIVE`：实验开始前完成预注册。
- `RETROSPECTIVE`：历史实验，事后整理。
- `UNREGISTERED`：没有找到有效预注册记录。

历史实验不得被伪装成事前注册。

### Experiment Type

- `FORMAL_EXPERIMENT`
- `EXPLORATORY_EXPERIMENT`
- `ENGINEERING_VALIDATION`
- `BENCHMARK`
- `AUDIT`
- `THEORETICAL_STUDY`

如果某项工作具有多个性质，选择主要类型，并在说明中补充。

### Experiment Outcome

- `PASS`
- `FAIL`
- `PARTIAL`
- `INCONCLUSIVE`
- `INVALID`
- `ABORTED`
- `NOT_EXECUTED`
- `UNKNOWN`

注意：实验 Outcome 与 Research Claim Status 是两个不同概念。

一次实验 PASS，不代表整个研究主张已经 ESTABLISHED。

## 5.6 Cross-Research Findings

记录跨研究方向的重要发现。

例如：

- Cell 局部更新与知识遗忘之间的关系。
- 局部连接与整体计算表达能力之间的关系。
- 稀疏执行与硬件效率之间的关系。
- Cell 组合与能力复用之间的关系。
- 动态增长与旧能力保持之间的关系。
- Cell 子图与持续学习之间可能存在的关系。

每条跨研究发现必须引用至少两个相关证据来源，或者明确说明它是理论推断。

不能仅凭研究者的直觉形成事实性结论。

## 5.7 Known Limitations

集中整理已经发现的限制。

例如：

- Cell 数量增加不一定提高模型能力。
- 减少理论 FLOPs 不一定降低实际推理延迟。
- 参数局部化不等于功能局部化。
- 保护参数更新不等于保护模型功能。
- 结构涌现不等于功能涌现。
- 受控线性组合不等于任意非线性组合。
- 增加模块不等于获得有效的持续学习能力。

每条限制必须关联具体证据，并标明适用范围。

## 5.8 Open Research Questions

集中记录当前尚未解决的问题。

建议使用：

| Question ID | Research Question | Related Claims | Existing Evidence | Knowledge Gap |
|---|---|---|---|---|

这里记录问题，而不是直接创建新的实验方案。

Open Questions 不应被误认为已批准的研究任务。

## 5.9 Current Research Assessment

记录当前整体研究判断。

包括：

- 已经成立的研究基础。
- 当前主要技术瓶颈。
- 值得进一步调查的研究机会。
- 暂时不值得投入的方向。
- 仍然缺乏证据的长期假设。

短期目标与长期目标必须分开评估。

不能因为某项技术具有长期研究价值，就默认它具备短期工程价值。

也不能因为某项技术暂时没有产品价值，就直接否定其基础研究价值。

本章节是当前研究判断，不是永久结论。

## 5.10 Research Decision History

记录重要研究决策。

建议采用：

| Decision ID | Date | Decision | Evidence | Rationale | Revisit Condition |
|---|---|---|---|---|---|

需要保留：

- 为什么启动某条研究路线。
- 为什么修改原始假设。
- 为什么暂停或终止某项研究。
- 为什么选择新的研究方向。
- 哪些新证据可能改变原来的决定。

不得删除已经被后续决策取代的历史记录。

## 5.11 Update History

记录 Evidence Map 本身的重要修改。

例如：

- 新增历史实验。
- 修订研究结论。
- 增加研究方向。
- 修正实验关联。
- 修改当前研究判断。

不需要记录每一次文字润色。

---

# 6. 实验与证据之间的关系

Evidence Map 必须建立以下逻辑关系：

```text
Research Line
    |
    +-- Research Questions
    |       |
    |       +-- Research Claims
    |
    +-- Experiments
    |       |
    |       +-- Registration
    |       +-- Execution
    |       +-- Results
    |       +-- Audits
    |
    +-- Evidence Synthesis
    |       |
    |       +-- Established Findings
    |       +-- Limitations
    |       +-- Open Questions
    |
    +-- Research Decisions
```

一个 Experiment 可以支持多个 Claim。

一个 Claim 也可以由多个 Experiment 共同支持或反驳。

必须保留这种多对多关系。

### 实验之间的关系类型

建议使用以下关系标签：

- `EXTENDS`：扩展之前的实验。
- `MODIFIES`：修改之前的方案。
- `REPLICATES`：重复验证之前的结果。
- `TESTS_FAILURE_MODE_OF`：针对之前实验暴露的问题进行验证。
- `CONTRADICTS`：产生与已有结论相矛盾的证据。
- `SUPERSEDES`：在明确范围内取代之前的实验判断。
- `INDEPENDENT`：相互独立的实验。
- `RELATED`：存在研究关联，但没有明确的直接演进关系。

不要因为两个实验属于同一研究方向，就自动认定它们存在因果关系。

---

# 7. 新实验的预注册与关闭规范

Evidence Map 必须与现有实验治理体系衔接。

不得建立第二套重复的实验注册机制。

## 7.1 实验启动前

所有正式研究实验必须在执行前完成预注册。

至少记录：

- Experiment ID。
- Research Line。
- Research Question。
- Primary Hypothesis。
- Existing Evidence。
- Experimental Design。
- Baselines。
- Metrics。
- Success Criteria。
- Failure Criteria。
- Random Seeds。
- Compute Budget。
- Dataset / Model。
- Code Revision。
- Planned Analysis。
- Stopping Rules。

探索性实验可以采用简化协议，但必须记录研究问题、实验目的、执行范围和预算。

例行 CI、普通单元测试不必全部作为研究实验注册；但凡是用于支持或改变研究主张的测试、测量和模型实验，都必须进入证据体系。

## 7.2 实验执行过程中

如果发生以下变化，必须记录：

- 实验设计修改。
- 模型或数据变化。
- Seed 变化。
- 指标变化。
- 预算变化。
- 训练中断。
- 代码版本变化。

原始协议不可覆盖。

变更必须保留：

- 变更时间。
- 变更内容。
- 变更原因。
- 是否发生在结果可见之前。
- 对实验有效性的影响。

如果已经看到实验结果后才修改分析方法，必须明确标记为事后分析。

## 7.3 实验结束后

所有实验必须提交结果记录。

即使失败、无效或中止，也必须记录。

关闭时至少包含：

- 实际执行配置。
- 实际执行 Seed。
- 实际结果。
- 与预注册门槛的对比。
- 所有偏差。
- 失败原因或未知原因。
- 主要发现。
- 结论边界。
- 原始数据与报告链接。
- 后续建议。

## 7.4 Evidence Map 更新

实验正式关闭时，必须同步更新：

1. Experiment Registry。
2. 对应 Research Line。
3. 相关 Research Claims。
4. Open Questions（如有变化）。
5. Current Research Assessment（如有实质变化）。

不要求每次实验都修改所有章节。

但不能只增加一个实验链接，而不更新其对当前研究判断的影响。

---

# 8. 历史实验迁移规范

本次建设的第一项主要工作，是整理现有历史实验。

历史实验的登记方式必须与未来实验的预注册区分。

## 8.1 盘点范围

Codex 应检查当前 `main` 分支中的完整研究记录。

至少检查：

- `research/` 下的所有实验目录。
- 所有正式实验协议。
- 所有正式结果。
- 所有验证与审计报告。
- 所有探索性实验和工程验证。
- 相关研究方向的 README。
- 与实验相关的历史总结和决策记录。

不得仅依靠文件名包含 `experiment` 的目录进行检索。

需要通过研究文档之间的引用关系，发现可能遗漏的实验。

对于当前 main 中无法确认的历史工作，单独列出待核查项。

不要擅自将未合并分支的内容当作 main 中的正式成果。

## 8.2 历史实验登记

每个发现的实验都必须：

1. 保留原始 Experiment ID。
2. 保留原始名称。
3. 找到协议和结果来源。
4. 记录实际 Outcome。
5. 记录所属 Research Line。
6. 关联前置和后续实验。
7. 提炼主要发现。
8. 明确结论边界。

没有预注册记录的历史实验，使用 `RETROSPECTIVE` 或 `UNREGISTERED`。

不能补写一个看似完整的预注册协议，并声称它是在实验开始前存在的。

## 8.3 历史冲突处理

如果发现：

- 不同报告的数字不一致。
- README 与正式结果不一致。
- 不同版本的协议互相冲突。
- 后续总结改变了原始实验结论。

不得直接覆盖。

必须登记：

`CONFLICT_REVIEW_REQUIRED`

并记录：

- 冲突内容。
- 相关文件。
- 各自的来源和时间。
- 目前无法确认的原因。

留待人工审阅。

## 8.4 不得擅自补全

以下信息没有来源时必须标记为 UNKNOWN：

- Seed 数量。
- 模型参数量。
- 训练数据量。
- 实际运行时间。
- 硬件配置。
- 具体失败原因。
- 真实推理收益。
- 统计显著性。

不能根据上下文猜测。

---

# 9. 研究结论的引用规范

Evidence Map 的每个重要研究结论都必须附带原始来源。

优先引用：

1. 正式实验结果。
2. 正式预注册协议。
3. 正式验证或审计报告。
4. 实际执行日志和数据。
5. 探索性记录。
6. 研究讨论和理论分析。

引用时优先使用 GitHub 仓库内的相对路径，必要时提供稳定的 GitHub 链接。

对重要结果，应尽可能指向具体文件和章节，而不是只链接整个实验目录。

如果某个研究结论来自多个实验，必须列出所有关键证据。

如果研究解释与实验事实不同，必须分别引用，不得混合引用。

外部论文可以作为理论依据，但不能替代 MiniCells 自身的实验结果。

---

# 10. Evidence Map 的维护责任

Evidence Map 是持续维护的研究文档，不是一次性整理成果。

必须遵守以下规则：

### 规则 A：新实验进入研究体系

新实验必须在预注册时关联现有 Research Line，或者明确提出新 Research Line。

### 规则 B：实验关闭时同步更新

实验正式关闭的 PR 必须包含对应的 Evidence Map 更新。

### 规则 C：重要结论变化时及时更新

如果新证据改变了已有 Claim 的状态，必须更新 Evidence Overview 和对应 Research Line。

### 规则 D：保留历史

不得删除旧实验、旧 Claim 和旧 Decision。

可以标记为 SUPERSEDED，但必须保留历史依据。

### 规则 E：定期整体复核

建议每个主要研究阶段结束时进行一次 Evidence Map 整体复核。

复核重点：

- 是否存在过时结论。
- 是否存在重复研究。
- 是否有实验没有关联到研究主张。
- 是否有结论缺少证据。
- 是否有研究方向需要重新评估。

---

# 11. 文档质量要求

Evidence Map 必须满足以下要求。

## 11.1 可读性

- 使用中文作为主要说明语言。
- 技术名称、指标和实验 ID 保留原始英文。
- 优先使用简洁、准确的表达。
- 不复制整篇实验报告。
- 避免重复解释相同机制。
- 表格用于索引，正文用于解释研究关系。

## 11.2 准确性

- 不得修改原始实验数据。
- 不得将工程验证写成正式研究成果。
- 不得将探索性结果写成已证明的结论。
- 不得混淆实验 Outcome 和研究 Claim Status。
- 不得隐去失败结果。
- 不得推断不存在的实验关联。

## 11.3 可追溯性

所有重要结论都必须能追溯到：

Claim → Experiment → Result → Protocol / Artifacts。

## 11.4 稳定性

- Claim ID 稳定。
- Experiment ID 保持原始值。
- 不因目录调整而重新编号。
- 文档结构变化不能破坏已有引用。
- Git 历史作为文档变更历史保存。

---

# 12. Codex 实施计划

本规范批准后，按以下阶段执行。

## Phase 1：建立 Evidence Map 框架

任务：

1. 检查现有 `research/governance/README.md`。
2. 创建 `research/governance/MINICELLS_EVIDENCE_MAP.md`。
3. 按照本规范建立完整章节结构。
4. 在治理 README 中增加入口和更新规则。
5. 不修改现有实验记录。
6. 不启动任何实验或训练。

交付：

- Evidence Map 初始文档。
- README 最小修改。
- 变更摘要。

## Phase 2：完整历史实验盘点

任务：

1. 扫描当前 main 中的研究记录。
2. 建立全部实验清单。
3. 确认每项实验的原始 ID、协议、结果和状态。
4. 区分正式实验、探索性实验、工程验证和审计。
5. 识别遗漏、重复和冲突。
6. 不修改原始实验结果。

交付：

完整的历史实验盘点。

## Phase 3：建立 Research Lines

任务：

1. 根据历史实验形成合理的研究方向分类。
2. 还原每条研究路线的演进过程。
3. 关联原始假设、后续修订和实验结果。
4. 识别研究路线之间的依赖和交叉关系。
5. 提炼已经成立的结论、失败的假设和未知问题。

特别注意：

不得为了让研究路线看起来连贯而虚构实验之间的因果关系。

## Phase 4：形成整体证据判断

任务：

1. 建立 Evidence Overview。
2. 建立 Research Claims。
3. 统一 Claim Status。
4. 建立 Known Limitations。
5. 建立 Open Research Questions。
6. 整理 Research Decision History。
7. 形成当前 Research Assessment。

这一阶段必须保留不确定性。

如果历史记录不足以支持某项判断，就明确标记为待审阅。

## Phase 5：完整性审计

对最终 Evidence Map 进行审计。

必须检查：

- 所有发现的实验是否已经登记。
- 是否遗漏失败实验。
- 是否遗漏探索性实验。
- 所有重要 Claim 是否有来源。
- 所有正式结果是否链接正确。
- 实验之间的关系是否有依据。
- 是否存在结论扩大解释。
- 是否存在原始记录与 Evidence Map 不一致。
- 是否存在未经说明的 UNKNOWN 或冲突。
- 是否正确区分历史实验与事前注册实验。

发现问题时优先修正索引和解释，不得擅自修改原始实验记录。

---

# 13. 最终验收标准

只有同时满足以下要求，Evidence Map 建设才算完成。

### A. 覆盖完整

- 当前 main 中发现的所有研究实验均已登记。
- 正式实验、失败实验、探索性实验和工程验证均得到适当分类。
- 所有无法确认的历史记录均已列出。

### B. 研究连续性

- 每条主要研究路线都有清晰的演进记录。
- 重要实验之间的关系能够被理解。
- 原始假设和后续修订能够追溯。
- 失败实验的后续影响得到记录。

### C. 证据可信度

- 重要 Claim 均有原始证据。
- 观察事实、研究解释和研究决策明确分离。
- 实验结果没有被扩大解释。
- 不存在将历史实验伪装成事前预注册的情况。

### D. 可读性

一个没有参与近期工作的研究者，应当能够通过 Evidence Map 回答：

1. MiniCells 已经证明了什么？
2. 哪些研究尝试失败了？
3. 为什么失败？
4. 后续实验如何调整？
5. 当前有哪些重要的未知问题？
6. 为什么当前研究方向形成了现在的状态？

### E. 可持续维护

- 新实验可以按照现有结构登记。
- 实验关闭流程能够触发 Evidence Map 更新。
- 不需要建立第二套实验治理系统。
- 文档能够通过普通 Markdown 和 Git 正常维护。

---

# 14. 本阶段明确不做的事情

本次工作仅建设研究证据管理和综合视图。

不得：

- 启动新的模型训练。
- 运行新的研究实验。
- 修改模型架构。
- 修改现有实验协议。
- 修改历史实验的原始结果。
- 为历史实验补造预注册。
- 重新设计 MiniCells 的研究路线。
- 根据历史实验盘点结果擅自决定未来资源投入。
- 大规模重构现有 research 目录。
- 建立与现有治理体系重复的系统。

如果发现历史研究存在重大矛盾，应当记录并报告，而不是在本次任务中自行作出新的技术判断。

---

# 15. 最终交付

Codex 应提交：

1. `research/governance/MINICELLS_EVIDENCE_MAP.md`
2. 必要的 `research/governance/README.md` 修改。
3. 完整历史实验登记。
4. Research Lines 及其演进关系。
5. Evidence Overview。
6. Open Research Questions。
7. 历史冲突与缺失信息清单。
8. 最终完整性审计报告。

最终报告必须说明：

- 共发现多少项研究实验和验证记录。
- 其中正式实验、探索性实验、工程验证分别多少。
- 有多少项通过、失败、部分通过、无效或未执行。
- 有多少项历史记录缺少完整协议或结果。
- 有多少项存在待审阅的冲突。
- 有多少条 Research Claim 已建立。
- 有多少个 Open Research Questions。
- 哪些研究路线的历史关系仍然无法确认。

所有统计数字必须来自实际盘点，不得预先填写。

---

# 16. 核心原则

MiniCells Evidence Map 的最终目的，不是让项目拥有一份更漂亮的研究总结。

而是建立一套长期有效的研究记忆机制。

**每个实验都必须成为整个研究体系的一部分，而不是一个孤立的结果。**

**每个研究结论都必须能够追溯到证据，而不是依赖研究者对过去讨论的记忆。**

**每次研究方向变化都必须能够解释其依据，而不是在新的讨论中丢失历史背景。**

我们希望未来任何时候，都能够准确回答：

- 我们从哪里开始？
- 我们尝试过什么？
- 哪些事情已经证明可行？
- 哪些事情已经证明存在问题？
- 我们为什么形成当前判断？
- 还有什么值得研究？

这就是 MiniCells Evidence Map 必须长期承担的职责。
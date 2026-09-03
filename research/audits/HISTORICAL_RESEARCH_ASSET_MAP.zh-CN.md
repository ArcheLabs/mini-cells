[English](HISTORICAL_RESEARCH_ASSET_MAP.md) | 中文

# 历史研究资产地图

审计日期：**2026-09-03**

本文重新定位 Core / Native formal 体系之前的 notebook 研究资产，严格区分：

```text
今天仍然能成立的科学结论
          !=
即使旧解释失效后仍值得保留的工程 primitive
```

本文不是新的科学实验，不改变任何冻结 protocol、formal decision 或 consumed seed 状态。

## 分类

| 分类 | 含义 |
|---|---|
| **HISTORICAL EXPLORATORY** | 历史探索/假设形成，有方法与失败边界价值，但不是当前 continual-learning formal evidence |
| **HISTORICAL MECHANISTIC EVIDENCE** | 对局部机制有重复或可复现实证，但低于后续 formal system-level 标准 |
| **ENGINEERING PRECURSOR EVIDENCE** | 旧科学解释有限，但工程结构直接延续到当前安全模型演化方向 |
| **RETIRED / SUPERSEDED PROTOCOL LINEAGE** | 协议与方法资产仍有价值，但科学问题已由后续更直接的 canonical 路线取代 |

Notebook 所在目录不决定证据强度。

---

# 1. `research/notebooks/01-foundations`

**当前分类：HISTORICAL EXPLORATORY**

这一阶段从 Echo、native trainability、quantization localization、tiny arithmetic、consumer language，一直推进到 1D/2D latent tissue、adaptive halting、settling dynamics、stabilization cost 等问题。

它今天仍有价值的结论主要是：

- cellular/local recurrent computation 在受控语言任务中可以训练；
- local state、repeated computation、halting/settling 可以被明确测量；
- 2D/NCA-like tissue 本身没有产生真正 continual learning；
- scaling、稳定 settling 和训练成本是实际约束；
- “看起来像 NCA”不是能力证明。

**它没有证明：**

- 语言模型应该使用 literal 1D/2D NCA topology；
- 2D 是 CLM 的必要条件；
- local dynamics 解决 catastrophic forgetting；
- early Cell topology 对应自然知识原子。

**仍值得复用的工程资产：**

- explicit local state；
- repeated/iterative computation；
- adaptive stopping probes；
- 数值稳定性与 mechanistic instrumentation；
- trainability/scaling historical baselines；
- quantization/localization diagnostics。

**仓库处理：KEEP as historical research assets。**

保留其历史与方法价值，但退出 active Native-CLM scientific evidence path。

---

# 2. `research/notebooks/02-self-organization`

**当前分类：HISTORICAL MECHANISTIC EVIDENCE**

这一阶段研究 emergent sparse topology、reaction-diffusion-like plasticity、growing cellular LM、localized learning、conditional/pressure recruitment、proposal utility、capability specificity、conflict differentiation、trait genesis、probationary genesis 等。

最重要的正面资产不是“已经发现自然 Cell”，而是证明了若干局部组织机制可以被诱导、测量和控制：

- sparse/local organization under pressure；
- conditional recruitment；
- local mutability；
- conflict/overload 后 differentiation；
- permanent growth 前 probation；
- multi-seed/null-mode 诊断。

同时这一阶段留下了一个非常重要的负面边界：

```text
interesting emergence
!=
validated functional boundary
!=
validated continual learning
```

**它没有证明：** natural Cell ontology、bounded language-scale growth、local pressure 可以决定“模型应该学什么”、或者自组织 specialization 在长期全局评估中保持安全。

**仍值得复用的工程资产：**

- conditional recruitment；
- pressure/conflict 作为 candidate proposal signal；
- probationary growth；
- localized mutability；
- sparse topology；
- multi-seed/null-mode discipline。

**工程规则：** pressure/recruitment 只能提出 candidate，不能作为 global quality oracle。

**仓库处理：KEEP, selective canonicalization。**

后续可把冗余或纯展示性 variant 移到 archive，但必须先做引用审计。

---

# 3. `research/notebooks/03-routing-and-growth`

**当前分类：ENGINEERING PRECURSOR EVIDENCE**

这是四组历史 notebook 中对当前工程路线价值最高的一组。

核心历史转变是：

```text
Cell as biological analogy
        ->
Cell as routed computational unit
        ->
Cell as independently mutable model state
```

包括 CLM-0.1、CLM-0.3 progressive growth、marginal growth utility、counterfactual/probationary mitosis、upcycling、CLM-v2 handoff、program conditionality、sparse runtime 等。

后续 Core/Native 已经证明：旧 routing heuristic、semantic address、global-pool growth 不能被当作最终正确机制。

但这一阶段形成的工程结构仍然直接有价值：

- sparse routed computation；
- explicit Cell identity/state；
- independently mutable module；
- versionable Cell state；
- append/expand；
- counterfactual candidate；
- probationary mitosis；
- marginal-utility accounting；
- inherit-then-differentiate/upcycling；
- structural routing provenance。

这些 primitive 可以重新解释为：

```text
accepted model
    -> fork/shadow candidate
    -> local Cell/module mutation or append
    -> provenance + dependencies
    -> global evaluation
    -> commit | rollback
```

**它没有证明：** old router 是 functional address、local utility 等价于 global model improvement、online mitosis 可以直接永久提交、Cells 优于 LoRA/MoE/module 作为 fork/merge granularity。

**仓库处理：KEEP as active engineering heritage。**

不能整组 archive；未来只迁移重复 variant，并保留 canonical engineering-heritage notebooks 的可见性。

---

# 4. `research/notebooks/05-language-validation`

**当前分类：RETIRED / SUPERSEDED PROTOCOL LINEAGE**

CLM-0.4-mini 的目标是把 dependency-scoped transactional growth 从 synthetic function world 转移到真正 autoregressive token-level model，同时故意保持一个显式稳定 addressing plane。

它的科学状态必须非常严格：

- v1 development path 在 base prerequisite 阶段停止；
- v2 修改了 data/admission alignment；
- v2 仍处于 pre-formal protocol/data-lock 路线；
- 因此不能声称 CLM-0.4 formal supported，也不能声称正式 falsified；
- 后来的 Native CLM Stage 06 更直接测试 trained token-predictive continual learning，因此科学上取代 CLM-0.4-mini 作为 active path。

但它留下了很强的方法/工程资产：

- M0 software smoke / M1 science / M2 scale gate 分离；
- frozen protocol + seed discipline；
- speculative candidate before commit；
- dependency-scoped validation；
- rollback；
- zero-output growth birth；
- atomic Cell/route commit；
- Cell registry/version state；
- transaction journal/checkpoint replay；
- dense/equal-compute controls；
- stable certification plane 与 learned semantic router 分离。

**它没有证明：** CLM-0.4 continual learning success/failure、对 dense/MoE 的 superiority、explicit metadata addressing 是产品方案、或 foundation-scale safe growth。

**仓库处理：KEEP, scientifically retired/superseded。**

不要为了“补完历史”重新消耗 GPU 跑旧 formal sequence；应复用 protocol/transaction primitives，并以 Native Stage 06 作为 canonical trained-model evidence。

---

# 跨阶段资产结论

| 资产 | 当前处理 |
|---|---|
| NCA/2D topology 作为主要范式 | 历史研究，不作为工程依赖 |
| local mutable state | 保留 |
| sparse computation | 保留 |
| pressure/conflict signal | 保留为 proposal signal |
| automatic local acceptance | 不可信任为 global decision |
| probation/counterfactual candidate | 强保留 |
| append/growth | 保留，但需要全局验收 |
| semantic/natural Cell ontology | 未建立 |
| rollback/transaction | 强保留 |
| versioned registry/journal | 强保留 |
| true continual-language evidence | 以 Native M2/M3/M3R 为准 |

# 产品/工程提取规则

历史 primitive 进入产品前必须满足：

1. 与旧 scientific narrative 解耦；
2. local signal 默认只产生 candidate；
3. accepted model 在阶段边界做 global evaluation；
4. acceptance 前 change 必须 versionable/reversible；
5. 后续 negative evidence 优先于旧解释；
6. Cells 必须最终与 LoRA、adapter、MoE module、external memory、model merge/version tooling 做成本与能力对比。

# Notebook 清理规则

未来每个历史 notebook 可标为：

```text
CANONICAL_HISTORICAL
ENGINEERING_HERITAGE
SUPPORTING_HISTORICAL
ARCHIVE_CANDIDATE
SUPERSEDED_PROTOCOL
```

不要批量删除或移动。物理迁移必须先做 reference audit，并在 published reproduction path 需要时保留 compatibility reference/shim。

我们的目标不是让目录视觉上最短，而是让 **scientific authority、historical value、engineering reuse** 三者永远不会再被混淆。

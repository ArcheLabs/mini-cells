# MiniCells Research History & Evidence Map

2026-10-30 修订。历史阶段依据 historical-migration-map，证据依据原始报告及研究账本。本次将已存在的研究综合分析整合到单一阅读入口；没有新增模型实验。**历史仍有未核实部分，不宣称全面验收完成。**

## 第一部分：MiniCells Research Story

MiniCells 最初探索的不是一种已经证明有效的持续学习系统，而是一个问题：计算能否由可组合、可路由、可增长的 Cell 承担？早期 001–026 建立了稀疏执行、分化和增长的实验语言。这些构造能力给后续研究提供了组件，但没有证明自然的 Cell 边界或真实语言学习中的知识局部性。

研究随后把“局部”从参数位置推进到功能影响。Core001/002 系列研究写入效应与可寻址性；Core003 将修改视为需要验证后提交的事务；Core004 在旧行为 oracle 保护下探索增长恢复可塑性。它们逐步把问题从“能否增加模块”改成“什么时候允许更新，更新损坏谁”。这些属于研究问题的演进；没有原始声明的具体先后关系仍不按编号强行推断。

Core005 给出了清晰但狭窄的正结果：固定特征、线性可写 Cell、显式路由和受控冲突下，局部子空间证书可替代学习器的原始历史回放，并在饱和时通过受控分裂恢复学习。Core006 随后把相近问题带入真实预训练语言模型表示：保护降低了遗忘，增长出的孩子也被复用，但正式系统门槛 0/3 通过，分裂覆盖比例和冲突缓解不达标。因此“某种保护有效”与“持续学习系统成立”必须分开。

另一条 Constructive CLM 路线把模块组合本身变成研究对象。从坐标形成、结构跟踪，到受保护的 Cell、模型级多模块计算和学习控制平面，受控线性残差世界获得了若干正结果。它们证明可组合计算可以被构造和学习部分控制，却没有消除已提供安全几何、受控算子和路由支架的条件。这解释了为什么真实语言模型迁移仍须另做实验。

Native CLM M0/M1 建立了约 12.15M 参数模型的端到端可训练性；这给更强假设提供了可被否证的载体。M2 中保护降低遗忘，但 A/TinyStories regression 约 43.9%，未达到不超过 20% 的目标。R0/R0b 查明投影梯度不保证 AdamW 实际更新安全，并把验证推进到最终参数事务；这一工程修正没有重写 M2 的科学失败。M3 增长并复用了新 Cell，却因全局 Top-K 插入改变读取所有权而仍失败。M3R 保留根读取所有权后改善约 2.5 个百分点，但仍远未达标。Address Diagnostic/M3L 又把问题缩小到：离线可恢复的局部边界，能否由实际路由和有界历史摘要解码？

由此形成的当前解释是：**安全写入、容量增长和读取寻址是三个不同问题。** 可训练性不是持续学习；事务正确不是功能覆盖充分；孩子被复用不是旧能力得到保护；离线几何存在不是部署路由能使用它。Core009 的 carrier 与 sparse/local effect 研究进一步区分了“承载因果效应”与“能够局部寻址和重用”。不能因为其中一项成立而宣布整个 CLM 成立。

以上历史分析来自 [研究账本](audits/RESEARCH_LEDGER.md)，下面在同一文档保留逐家族结果与边界。2026-09-03 账本不是所有较新结果的自动认证：之后的研究和未归属记录仍须逐项回填，本轮不把未合并分支当作 main 成果。

## 第二部分：Master Research Timeline

历史阶段是研究组织方式，不等于严格日期或全部直接因果链。下列详细条目保留来源账本的结果、正负证据与适用条件；其英文状态原文不替换为新的成功标准。日期无直接依据时 UNKNOWN。

### 01 Foundations / 02 Self-Organization / 03 Routing and Growth

#### Legacy experiments 001–026 — foundations, self-organization, routing and growth


**研究问题（原综述）**  
Establish early feasibility of sparse Cell computation, self-organization pressure, routing, differentiation and growth before the formal continual-learning core.

**原始来源**  
`research/projects/cellular-computing/experiments/` through `research/projects/cellular-language-models/experiments/`, plus `research/governance/catalog.yaml`.

**正向结果**
- supplied the early architectural and experimental vocabulary used by later Core validations;
- demonstrated that sparse modular computation and growth mechanisms can be constructed;
- produced useful implementation components and negative controls for later work.

**负向结果与限制**
- many experiments predate the later formal seed/protocol discipline;
- local or synthetic behavior from this period should not be promoted into current Native-CLM claims;
- the old catalog is incomplete relative to the modern validation tree.

**原记录状态**  
Historical evidence, heterogeneous strength.

**证据上限**  
Architecture exploration and mechanism discovery.

**不能证明的内容**  
Natural Cell ontology, replay-free continual language learning, globally optimal local learning decisions, or bounded long-horizon growth.

**获得的技术组件**  
Sparse modular execution, explicit routing/growth scaffolds, early experiment harnesses.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

### 04 Continual Learning

#### Core 001 / 001B / 002 / 002B / 002C — knowledge subsumption and write-addressability foundations


**研究问题（原综述）**  
Probe whether knowledge/write effects can be isolated, assembled, or recovered in controlled systems and whether superposition prevents useful write addressability.

**原始来源**  
`research/governance/validations/core-001-*`, `core-002-*`.

**正向结果**  
Established controlled settings in which local write/effect structure is measurable and supplied the foundations for later transaction and certificate experiments.

**负向结果与限制**  
Controlled write-addressability does not establish that mature LLM checkpoints expose a stable deployable Cell boundary.

**原记录状态**  
Foundational controlled evidence; consult per-validation frozen result for exact claim.

**证据上限**  
Mechanism-level write geometry.

**不能证明的内容**  
Task-free continual learning or natural modularity in pretrained Transformers.

**获得的技术组件**  
Local-effect measurement and transaction-oriented experimental framing.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 003 — dependency-scoped transactional learning


**研究问题（原综述）**  
Treat model changes as scoped transactions rather than unconstrained global mutation.

**正向结果**  
Helped establish the useful distinction between a candidate local update and its accepted/committed effect.

**证据上限**  
Scoped transaction mechanics do not decide whether the complete model is globally better.

**不能证明的内容**  
A local acceptance signal is a global model-quality oracle.

**获得的技术组件**  
Candidate/commit separation and dependency-scoped validation.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 004 — growth-restored plasticity


**研究问题（原综述）**  
Test whether adding capacity can restore learning when safe mutation of existing capacity is blocked.

**正向结果**  
Growth restores plasticity in the registered controlled setting when the old-behavior evaluator rejects unsafe writes.

**负向结果与限制**  
Uses a controlled old-behavior oracle; later Native M3 shows that new capacity can still damage old behavior through read/routing ownership.

**原记录状态**  
**SUPPORTED — controlled**.

**证据上限**  
Capacity rescue, not safe system-level growth.

**不能证明的内容**  
That growth is sufficient for continual learning or that the correct split/routing boundary is known.

**获得的技术组件**  
Append/expand as an escape from exhausted writable freedom.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 005 — replay-free subspace-certified mitosis


**研究主张（原综述）**  
Can bounded Cell-local subspace state replace raw historical replay for safe writes and saturation-triggered growth in a controlled continual-learning system?

**原始来源**  
[research/projects/continual-learning/experiments/core-005-subspace-certified-mitosis/validation/core-005-subspace-certified-mitosis/README.md](../../research/projects/continual-learning/experiments/core-005-subspace-certified-mitosis/validation/core-005-subspace-certified-mitosis/README.md) and frozen protocol/artifacts.

**正向结果**
- `SUBSPACE_CERTIFIED_MITOSIS_SUPPORTED` on formal seeds `80501/80502/80503`;
- certificate-growth produced 48/48 effective commits with zero cumulative positive historical regression in the registered history;
- no learner old-sample/old-label/replay access;
- growth rescued all registered forced collisions and children were reused;
- wrong-certificate causal control produced false-safe events and regression.

**负向结果与限制**
- linear writable Cell;
- fixed feature coordinates;
- explicit routing;
- certificate is constructed from registered activations;
- synthetic conflict construction.

**原记录状态**  
**SUPPORTED — controlled**.

**证据上限**  
Exact protection of the registered linear history plus controlled capacity rescue.

**不能证明的内容**  
A fixed sufficient certificate exists for arbitrary Transformer weights, language-scale activation spans stay low-rank, semantic routing is solved, or an opaque checkpoint contains reconstructable certificates.

**获得的技术组件**  
Explicit functional contracts, safe projected local writes, saturation detection, and growth as a transactional fallback.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 006 — real-representation continual plasticity


**研究主张（原综述）**  
Can the Core-005 style bounded state support replay-free continual learning on real pretrained language-model representations without rapid saturation or growth explosion?

**原始来源**  
[research/projects/continual-learning/experiments/core-006-real-representation-continual-plasticity/validation/core-006-real-representation-continual-plasticity/README.md](../../research/projects/continual-learning/experiments/core-006-real-representation-continual-plasticity/validation/core-006-real-representation-continual-plasticity/README.md).

**正向结果**
- real Pythia representations did not immediately fill the 64-D projected space;
- functional reuse increased;
- certificate writes reduced registered-history forgetting relative to unsafe writes;
- candidate retained substantial new-learning gain relative to replay;
- spawned children were reused.

**负向结果**
- `REAL_REPRESENTATION_CONTINUAL_PLASTICITY_NOT_SUPPORTED`, 0/3 formal seeds;
- spawned Cells covered ~65.6–68.8% of addresses, above the registered bound;
- address-based splits failed the registered conflict-relief requirement;
- detailed Kaggle artifacts were partly lost after session closure and were not reconstructed.

**原记录状态**  
**NOT SUPPORTED**, with positive sub-results.

**证据上限**  
Real hidden-state low-dimensional/reusable geometry is useful, but routing address is not equivalent to a writable functional boundary.

**不能证明的内容**  
That semantic addresses are Cells, that growth remains bounded, or that the certificate mechanism alone solves continual learning.

**获得的技术组件**  
Real-model functional regression measurement and certificate-based damage reduction.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 007 — functional-boundary discovery


**研究主张（原综述）**  
Can write-demand/interference geometry provide a better split boundary than semantic address identity, and can that identity be recovered from inference-visible representations?

**原始来源**  
[research/projects/continual-learning/experiments/core-007-functional-boundary-discovery/validation/core-007-functional-boundary-discovery/README.md](../../research/projects/continual-learning/experiments/core-007-functional-boundary-discovery/validation/core-007-functional-boundary-discovery/README.md).

**正向结果**
- discovery selected `interference_cut` under the frozen mechanism-selection score;
- discovery showed high soft-Top2 oracle coverage and useful z/oracle agreement.

**负向结果与限制**
- discovery is not confirmation;
- the first confirmation infrastructure lost a completed seed before durable checkpointing, forcing seed retirement;
- the main README still records later boundary work as an unresolved/negative integration area.

**原记录状态**  
The checked-in validation README is **confirmation incomplete / amended protocol**, not a supported scientific confirmation.

**证据上限**  
Promising discovery geometry only.

**不能证明的内容**  
A stable deployable functional boundary, safe nonlinear updates, or autonomous CLM growth.

**获得的技术组件**  
Resumable per-seed checkpoint/publish discipline and explicit separation of discovery from confirmation.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 008 family — certified functional atoms / equivalence / postmortem capacity


**研究问题（原综述）**  
Interrogate whether increasingly local functional units can be certified and whether apparent Cell evidence survives causal/control analysis.

**原始来源**  
`research/governance/validations/core-008-*` and associated stage documents.

**正向结果**  
Produced useful causal controls, functional-equivalence preflights and capacity diagnostics that narrowed which local effects deserve later testing.

**负向结果与限制**  
The family did not establish a universal natural Cell atom or a trained-language continual-learning solution. Some results are diagnostic/postmortem rather than formal model milestones.

**原记录状态**  
Mixed family; preserve per-validation statuses.

**证据上限**  
Mechanism selection and causal diagnostics.

**不能证明的内容**  
That a local causal effect is a globally useful or stable Cell boundary.

**获得的技术组件**  
Stronger preflight/control methodology before expensive formal runs.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Core 009 family — factorized/effect geometry and deployable addressability


**研究问题（原综述）**  
Probe whether mature model write/effect geometry can supply sparse, persistent, compositional and deployable functional coordinates.

**原始来源**  
`research/governance/validations/core-009a-*`, `core-009b*`, `core-009c-*`, `core-009d-*`, plus branch-published artifacts where applicable.

**正向结果**  
The family provides bridge evidence that pretrained-model representations/effects contain useful local and factorized signals worth decoding.

**负向结果与限制**  
The checked-in tree contains multiple sequential/revised 009 hypotheses, and not all branch-published outcomes are normalized into the main validation directories. The accumulated evidence has not established a stable, low-growth natural Cell boundary that transfers into Native continual-language success.

**原记录状态**  
**MIXED / family-level audit**; do not collapse the individual 009 decisions into one positive claim.

**证据上限**  
Foundation-interface diagnostics.

**不能证明的内容**  
That sparse local effect geometry is the correct Cell ontology, or that effect addressability solves read/write ownership in a trained CLM.

**获得的技术组件**  
Functional-effect probes and a more disciplined interface between foundation-model diagnostics and CLM experiments.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

### 05 Language Validation — Constructive CLM

#### Constructive CLM-001 / 001B — coordinate formation and latent discovery


**研究问题（原综述）**  
Can a persistent addressable coordinate system form from experience, including under the registered additive superposition scaffold?

**正向结果**  
Both registered constructive claims are supported on their frozen formal seeds.

**原记录状态**  
**SUPPORTED — controlled**.

**证据上限**  
Singleton-exposure and registered additive pair-superposition worlds.

**不能证明的内容**  
Arbitrary blind source separation or natural coordinates in pretrained LLMs.

**获得的技术组件**  
Learned coordinate/address formation in an engineered substrate.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Constructive CLM-002 — long-horizon structure-tracking growth


**研究问题（原综述）**  
Can growth track reusable latent structure rather than simply spawn one Cell per observation?

**正向结果**  
Supported to the registered finite horizon: `N_final=4096`, 30 latent factors, 30 final Cells, finite-horizon fitted growth exponent ~0.663.

**原记录状态**  
**SUPPORTED — controlled finite horizon**.

**证据上限**  
Finite-horizon evidence only.

**不能证明的内容**  
An asymptotic sublinear-growth theorem or language-scale bounded growth.

**获得的技术组件**  
Structure-tracking growth controller under known constructive assumptions.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Constructive CLM-003 — protected learned/growing Cells


**研究问题（原综述）**  
Integrate learned/growing Cells with replay-free protected writes.

**正向结果**  
`PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED` on seeds `90511/90512/90513`.

**原记录状态**  
**SUPPORTED — controlled**.

**证据上限**  
Learned root plus linear protected-write integration.

**不能证明的内容**  
Nonlinear Transformer Cell safety or Native continual-language success.

**获得的技术组件**  
Protected append/growth composition in an explicit modular substrate.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Constructive CLM-004 — model-level multi-Cell computation


**研究主张（原综述）**  
Can multiple learned Cells participate sparsely in nontrivial model-level simultaneous and order-sensitive sequential computation while preserving a protected mutation invariant?

**原始来源**  
[research/projects/cellular-language-models/experiments/constructive-clm-004-model-level-multicell-computation/validation/constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md](../../research/projects/cellular-language-models/experiments/constructive-clm-004-model-level-multicell-computation/validation/constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md).

**正向结果**
- `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`;
- seeds `90611/90612/90613`;
- all 17 registered gates pass;
- held-out simultaneous and sequential composition, sparse execution, route support/order recovery and protected mutation all pass.

**原记录状态**  
**SUPPORTED — controlled**.

**证据上限**  
Controlled linear residual operators.

**不能证明的内容**  
Arbitrary nonlinear Transformer Cells, language generation, learned/endogenous routing/growth/write control, LLM-scale endogenous CLM or JAM execution.

**获得的技术组件**  
Sparse multi-module composition as a model-level structural primitive.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Constructive CLM-005 — learned control-plane transition


**研究主张（原综述）**  
Can routing/write/growth scaffolds transition to learned controllers while retaining earlier constructive invariants?

**原始来源**  
[research/projects/cellular-language-models/experiments/constructive-clm-005-scaffold-removal/validation/constructive-clm-005-scaffold-removal/FORMAL_RESULT.md](../../research/projects/cellular-language-models/experiments/constructive-clm-005-scaffold-removal/validation/constructive-clm-005-scaffold-removal/FORMAL_RESULT.md).

**正向结果**
- `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`;
- seeds `90811/90812/90813`;
- all 20 registered gates pass;
- learned router/growth/write heldout meta accuracy 1.0 in the registered world;
- bounded reusable children, sparse composition and replay-free protected behavior retained.

**负向结果与限制**  
Core-005 certificate basis and constrained update solver remain fixed safety primitives.

**原记录状态**  
**SUPPORTED — controlled**.

**证据上限**  
Learned control plane over a supplied safety geometry in a controlled linear-residual world.

**不能证明的内容**  
Fully learned safety geometry, arbitrary nonlinear Cells, language-scale continual learning, asymptotic growth, or foundation-scale Native CLM.

**获得的技术组件**  
Learned candidate-control policies can coexist with deterministic safety/validation primitives.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

### 06 Native CLM

#### Native CLM v0 M0/M1 — architecture and next-token trainability


**研究问题（原综述）**  
Can the Native CLM architecture execute and train end-to-end from next-token loss?

**正向结果**
- Stage-06 M0/M1 complete;
- canonical ~12.15M-parameter model trained successfully;
- validation loss `5.723429 -> 0.788535` and perplexity `305.9523 -> 2.2002` in the registered run;
- sparse active Cell fraction `2/8`.

**原记录状态**  
**SUPPORTED — trainability**.

**证据上限**  
End-to-end trainability of a small token-predictive Native CLM.

**不能证明的内容**  
Continual learning, retention, autonomous growth or superiority to dense/MoE baselines.

**获得的技术组件**  
A real trainable Native-CLM test substrate for falsifying stronger mechanisms.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Native CLM v0 M2 — fixed-topology replay-free continual language


**研究主张（原综述）**  
Are fixed-topology protected Cells sufficient for replay-free continual language learning?

**正向结果**
- certificate projection reduced mean forgetting from ~0.2790 to ~0.2115;
- protected path preserved roughly 96% of unsafe plasticity in the registered run.

**负向结果**
- `NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED`;
- formal seeds `73211/73212/73213`;
- A/TinyStories regression ~43.9% against the registered <=20% target.

**原记录状态**  
**NOT SUPPORTED**, with partial causal value for protection.

**证据上限**  
Protection helps, but fixed topology and current certificate semantics do not provide sufficient continual-language retention.

**不能证明的内容**  
Replay-free continual language learning or that better protection alone will solve the problem.

**获得的技术组件**  
Certificates as bounded regression guards, not as a complete learner.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Native M2-R0 / M2-R0b — optimizer/update invariant audit


**研究问题（原综述）**  
Determine whether the registered safe-gradient constraint survives the optimizer/parameter transaction.

**原始来源**
- R0 published commit `c1f9e132b026efe24a0238e6ea333bdd2ae5fbdb`;
- R0b published commit `3addfea3e98c792bf487432f0ab22b787e2a4d49`.

**正向结果**
- R0 exposed that canonical AdamW grad projection does not imply safe realized update;
- R0b classifies the SGD reference issue as finite-precision parameter-transaction roundoff;
- R0b identifies `ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT`;
- projecting the final realized AdamW update reduces committed p95 violation to ~`5.44e-6`, at the matched numerical floor.

**Important threshold discipline**  
The SGD+weight-decay arm is not post-hoc promoted to a material structural violation: its excess factor (~3.95) remains below the preregistered material threshold 16.

**原记录状态**  
**SUPPORTED — mechanics/diagnostic**. R0b explicitly has `scientific_decision=false` and `native_clm_training_milestone=false`.

**证据上限**  
Optimizer transaction correctness.

**不能证明的内容**  
Certificate semantic coverage, old-function sufficiency or continual-learning success; it does not change the historical M2 decision.

**获得的技术组件**  
`raw optimizer proposal -> realized-update projection/validation -> commit`.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Native CLM v0 M3 — global-pool growth-restored continual language


**研究主张（原综述）**  
Does adding fresh writable Cells restore continual-language retention after M2?

**原始来源**  
[research/projects/cellular-language-models/experiments/native-clm-v0-m3-growth-restored-continual-language/validation/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md](../../research/projects/cellular-language-models/experiments/native-clm-v0-m3-growth-restored-continual-language/validation/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md).

**正向结果**
- growth occurs and remains within its registered topology bound;
- children are reused 100% under the registered reuse metric;
- growth-phase plasticity and sparse compute survive;
- zero learner replay.

**负向结果**
- `NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED` on seeds `73411/73412/73413`;
- growth A regression ~0.484–0.494, worse than matched fixed protected topology ~0.429–0.442;
- global Top-K insertion changes read ownership even when a new child begins as an exact clone.

**原记录状态**  
**NOT SUPPORTED**.

**证据上限**  
Fresh capacity is not enough; read ownership must also be preserved.

**不能证明的内容**  
That growth is generally useless. It falsifies the registered global-pool growth mechanism.

**获得的技术组件**  
A strong invariant for future expansion: **safe write growth requires safe read-address growth**.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### Native CLM v0 M3R — read-preserving / lineage-isolated growth


**研究主张（原综述）**  
Does preserving top-level root read ownership make growth safe enough for continual retention?

**正向结果**
- root-route probe identities and birth root ownership are conserved;
- growth reaches 16 Cells with 100% registered child reuse;
- sparse compute, zero replay and plasticity remain;
- A retention improves by ~2.5 percentage points relative to matched global growth.

**负向结果**
- `NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED`;
- A regression remains ~0.469–0.472, far above the <=20% absolute target;
- child execution remains broad across old and new domains.

**原记录状态**  
**NOT SUPPORTED**, with a positive causal sub-result for root read conservation.

**证据上限**  
Top-level read ownership can be conserved; lineage-local parent/child functional addressing remains unresolved.

**不能证明的内容**  
That hierarchical routing is sufficient for continual learning.

**获得的技术组件**  
Lineage isolation as a safer expansion topology than unrestricted global insertion.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### M3R Address Diagnostic — local query geometry


**研究主张（原综述）**  
Within a stable lineage, is the parent/child functional split present in current frozen query geometry or write/effect geometry?

**正向结果**
- 24/24 valid lineage edges in the canonical diagnostic;
- current cosine median AUC ~0.5315;
- free affine query probe median AUC ~0.9623;
- classification recorded in the research index as `QUERY_GEOMETRY_SEPARABLE`.

**原记录状态**  
**SUPPORTED — diagnostic only**, no new continual-learning seeds.

**证据上限**  
The representation contains a recoverable offline local boundary; the deployed cosine rule does not decode it.

**不能证明的内容**  
That the boundary can be learned online without historical state, or that using it restores continual retention.

**获得的技术组件**  
Offline address diagnostics can distinguish representation failure from decoder/routing-rule failure.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### M3L Query-Sketch Gate


**研究主张（原综述）**  
Can the useful lineage-local affine boundary be recovered without replaying old tokens/queries, using a compact rank-16 historical Gaussian query sketch plus the current conflict stream?

**正向结果**
- offline oracle median AUC remains high (~0.9281);
- rank-16 sketch is a near miss at median AUC ~0.8968;
- edge-floor, normalized-oracle-recovery, old-FPR and current-TPR gates pass.

**负向结果**
- frozen classification `QUERY_SKETCH_GATE_NOT_FEASIBLE` because registered median AUC >=0.9000 is missed.

**原记录状态**  
**NOT SUPPORTED — diagnostic**, no new continual-language seed claim.

**证据上限**  
The registered compact sketch is insufficient under its frozen threshold; the remaining question is capacity/family, not whether query geometry contains signal.

**不能证明的内容**  
That all compact historical state is insufficient, or that raw replay is necessary.

**获得的技术组件**  
Historical address summaries can be audited explicitly for information capacity before another expensive training run.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

#### M3L-1 / later address-state diagnostics


**研究问题（原综述）**  
Separate low-rank capacity limits from Gaussian-family/model-family limits using the exact frozen M3L samples and oracle.

**原记录状态**  
**ACTIVE / diagnostic in the checked-in research index**. Do not infer a new continual-learning milestone.

**证据上限**  
Checkpoint-only mechanism diagnosis.

**不能证明的内容**  
Any improvement to M2/M3/M3R unless a later full registered model run transfers the mechanism.

**获得的技术组件**  
Escalate representation capacity only after a fixed-sample diagnostic identifies the actual bottleneck.

---


来源综述：[RESEARCH_LEDGER](audits/RESEARCH_LEDGER.md)。设计细节/日期缺失不补造；后续关系只采用原文明确陈述，未声明者 UNKNOWN。

### 07 Safe Model Evolution / 08 Hybrid CLM / 新增独立 sparse-path

这三部分的完整实验级历史尚未整合，不能以研究线 README 替代实验结果。main 中的 PCU、conversion、functional-boundary、Hybrid 以及 sparse-path 记录保持在原目录，待从原协议/结果补齐问题、设计、失败与调整；不凭名称构造因果链。[safe-model-evolution](../projects/safe-model-evolution/README.md)、[language models](../projects/cellular-language-models/README.md)、[clm-native](../projects/clm-native/README.md)。

## 第三部分：Research Line Evolution

持续学习的问题由“局部写入是否可保护”转到“受限保护后还能否持续增长”，再转到“功能边界是否可部署寻址”。Core005 的受控正结果与 Core006 的系统失败共同促成这种区分；Native M3/M3R 进一步表明读取所有权不能由写入保护自动保证。Constructive CLM 则保留另一条正向证据：在提供的安全几何中，多模块组合与学习控制可以成立。两条路线不是互相否定，而是条件不同。

## 第四部分：Cross-Experiment Evidence

|研究判断|正向证据|反向/限制证据|当前解释|
|---|---|---|---|
|保护可以降低旧功能回归|Core005、Core006 子结果、M2|Core006 与 M2 系统门槛失败|有条件支持，不是完整持续学习|
|增加容量足以解决持续学习|Core004 受控恢复|M3、M3R 未达标|不足；读取寻址仍关键|
|Cell 能组成计算|Constructive004/005|线性残差/安全几何条件|受控支持，非任意非线性 LM|
|安全梯度就是安全实际更新|投影提供保护意图|R0/R0b AdamW 审计|需验证最终参数事务|
|局部几何可被实际路由利用|Address 诊断的离线探针|cosine 解码与摘要门槛问题|尚未建立部署可用性|

以上均是带条件的研究解释，来源为第二部分对应原记录，非新的实验决策。

## 第五部分：Complete Experiment History — 迁移尚未完成

本文件尚不是完整历史的验收版本。下面保留旧版 Claim 和待核查索引作为附录，不把 141 个工作目录当作 141 个独立实验。后续必须按 Experiment / Family / Run / Seed / Diagnostic / Engineering Validation / Proposal / Governance Record 分层，原 ID 保留命名空间；未分类记录属于 NOT_YET_REVIEWED，而非已核实没有结果。

## 第六部分：Current Research Assessment

已经支持的是受控机制和明确设置下的可训练性；仍缺的是功能保护、可塑性、读取寻址和实际成本同时成立的系统证据。新研究问题必须由这些差距驱动，而不是将局部通过累加为整体成功。这里不批准新训练或资源投入。

## 状态映射与不确定性修订

Core005 原账本的 SUPPORTED—controlled 对应本 Map 的 SUPPORTED（明确受控范围），不继续使用 ESTABLISHED。Core006 的 NOT SUPPORTED 是系统 Outcome FAIL、总体主张 NOT_ESTABLISHED；正向子假设仍单独保留。原先 CONTRADICTED 若指“全部 Seed 通过”只是狭窄事实否定，不得扩展成机制不可能。旧标记保存在 Git 历史中，此处记录修订原因。

待核查状态分为 NOT_YET_REVIEWED、RESULT_NOT_FOUND_AFTER_REVIEW、INSUFFICIENT_EVIDENCE、CONFLICT_REVIEW_REQUIRED、NOT_AN_INDEPENDENT_EXPERIMENT。不得通过批量替换把未检查者说成已检查。

## 旧版索引与主张（保留历史，不作为本次全面验收）

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
|EM-C001|Core005 的固定线性可写合成设置达到了其子空间证书/分裂门槛|SUPPORTED|[Core005 RESULTS](../projects/continual-learning/experiments/core-005-subspace-certified-mitosis/artifacts/RESULTS.md)|报告 3/3；不是任意非线性 Cell 或真实 LM 证书|
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

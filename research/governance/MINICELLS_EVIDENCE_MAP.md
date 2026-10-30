# MiniCells Research History and Evidence Map

Updated 2026-10-30. This is a research history, not a count of successful directories. The inventory covers main snapshot `3a70e57e6fbf2c515fda239aae03641a66ff6d74`; historical synthesis below draws on the existing Research Ledger. Newer records still require source-level integration. **Full historical acceptance remains incomplete.** Original experiments and results are unchanged.

## 1. MiniCells Research Story

MiniCells began by asking whether computation could be organized as interacting, reusable Cells rather than a fixed monolithic model. Early foundations, self-organization, routing and growth experiments established implementable mechanisms and experimental vocabulary. They did not establish a natural Cell ontology, safe language learning, or hardware superiority.

The next question concerned function rather than parameter location: which existing behavior does a write affect? Core001/002 studied write effects and addressability; Core003 separated proposed updates from validated commits; Core004 investigated growth when safe writable freedom was exhausted. These are related research questions, not an assertion that experiment numbering proves chronology.

Core005 obtained a controlled positive result: bounded subspace certificates supported protected writes and growth without learner replay in a fixed-feature, linear-writable, explicitly routed synthetic setting. Core006 moved the question to real pretrained language-model representations. Protection and reuse had positive sub-results, but the system-level claim failed on all three formal seeds. The distinction between a useful protection mechanism and a working continual learner became central.

Constructive CLM pursued the complementary question of whether multiple Cells could actually compose computation. Its controlled linear-residual experiments supported composition and parts of a learned control plane. Supplied safety geometry and constrained operators remained essential conditions; these results did not automatically transfer to nonlinear language models.

Native M0/M1 demonstrated small-model next-token trainability. M2 then tested replay-free continual language learning: protection reduced forgetting, but approximately 43.9% A/TinyStories regression exceeded the registered 20% limit. R0/R0b showed why projecting gradients did not guarantee safe realized AdamW updates. Correcting the update transaction did not reverse M2's scientific outcome.

M3 added reusable capacity, but inserting children into global Top-K routing changed read ownership and retention still failed. M3R preserved root ownership and improved retention by about 2.5 percentage points relative to matched global growth, yet remained far from its absolute target. Address diagnostics and M3L narrowed the question further: can useful offline boundaries be decoded by the deployed router and a bounded historical summary?

**Current interpretation:** safe writes, added capacity, and safe read addressing are distinct requirements. Trainability is not continual learning; transaction correctness is not functional coverage; child reuse is not retention; offline geometry is not deployed addressability. Core009's carrier and sparse/local-effect work reinforces these distinctions. This interpretation is a synthesis of the cited studies, not a newly approved research direction.

The detailed historical summaries below retain both positive and negative evidence. They are adapted from the [Research Ledger](audits/RESEARCH_LEDGER.md), whose 2026-09-03 synthesis is not itself certification of every later experiment.

## 2. Master Research Timeline

Historical stages follow the [ownership migration map](historical-migration-map.md). Stage order is organizational, not a verified date or causal ordering. Experiment dates remain UNKNOWN where the source does not supply them. Original result labels are quoted rather than automatically converted into project-wide success.

### 01 Foundations / 02 Self-Organization / 03 Routing and Growth

#### Legacy experiments 001–026 — foundations, self-organization, routing and growth


**Claim / role**
Establish early feasibility of sparse Cell computation, self-organization pressure, routing, differentiation and growth before the formal continual-learning core.

**Evidence source**
`research/projects/cellular-computing/experiments/` through `research/projects/cellular-language-models/experiments/`, plus `research/governance/catalog.yaml`.

**Positive evidence**
- supplied the early architectural and experimental vocabulary used by later Core validations;
- demonstrated that sparse modular computation and growth mechanisms can be constructed;
- produced useful implementation components and negative controls for later work.

**Negative evidence / limitation**
- many experiments predate the later formal seed/protocol discipline;
- local or synthetic behavior from this period should not be promoted into current Native-CLM claims;
- the old catalog is incomplete relative to the modern validation tree.

**Formal status**
Historical evidence, heterogeneous strength.

**Known ceiling**
Architecture exploration and mechanism discovery.

**What it does NOT prove**
Natural Cell ontology, replay-free continual language learning, globally optimal local learning decisions, or bounded long-horizon growth.

**Engineering primitive obtained**
Sparse modular execution, explicit routing/growth scaffolds, early experiment harnesses.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

### 04 Continual Learning

#### Core 001 / 001B / 002 / 002B / 002C — knowledge subsumption and write-addressability foundations


**Claim / role**
Probe whether knowledge/write effects can be isolated, assembled, or recovered in controlled systems and whether superposition prevents useful write addressability.

**Evidence source**
`research/governance/validations/core-001-*`, `core-002-*`.

**Positive evidence**
Established controlled settings in which local write/effect structure is measurable and supplied the foundations for later transaction and certificate experiments.

**Negative evidence / limitation**
Controlled write-addressability does not establish that mature LLM checkpoints expose a stable deployable Cell boundary.

**Formal status**
Foundational controlled evidence; consult per-validation frozen result for exact claim.

**Known ceiling**
Mechanism-level write geometry.

**What it does NOT prove**
Task-free continual learning or natural modularity in pretrained Transformers.

**Engineering primitive obtained**
Local-effect measurement and transaction-oriented experimental framing.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 003 — dependency-scoped transactional learning


**Claim / role**
Treat model changes as scoped transactions rather than unconstrained global mutation.

**Positive evidence**
Helped establish the useful distinction between a candidate local update and its accepted/committed effect.

**Known ceiling**
Scoped transaction mechanics do not decide whether the complete model is globally better.

**What it does NOT prove**
A local acceptance signal is a global model-quality oracle.

**Engineering primitive obtained**
Candidate/commit separation and dependency-scoped validation.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 004 — growth-restored plasticity


**Claim / role**
Test whether adding capacity can restore learning when safe mutation of existing capacity is blocked.

**Positive evidence**
Growth restores plasticity in the registered controlled setting when the old-behavior evaluator rejects unsafe writes.

**Negative evidence / limitation**
Uses a controlled old-behavior oracle; later Native M3 shows that new capacity can still damage old behavior through read/routing ownership.

**Formal status**
**SUPPORTED — controlled**.

**Known ceiling**
Capacity rescue, not safe system-level growth.

**What it does NOT prove**
That growth is sufficient for continual learning or that the correct split/routing boundary is known.

**Engineering primitive obtained**
Append/expand as an escape from exhausted writable freedom.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 005 — replay-free subspace-certified mitosis


**Claim**
Can bounded Cell-local subspace state replace raw historical replay for safe writes and saturation-triggered growth in a controlled continual-learning system?

**Evidence source**
[research/projects/continual-learning/experiments/core-005-subspace-certified-mitosis/validation/core-005-subspace-certified-mitosis/README.md](../../research/projects/continual-learning/experiments/core-005-subspace-certified-mitosis/validation/core-005-subspace-certified-mitosis/README.md) and frozen protocol/artifacts.

**Positive evidence**
- `SUBSPACE_CERTIFIED_MITOSIS_SUPPORTED` on formal seeds `80501/80502/80503`;
- certificate-growth produced 48/48 effective commits with zero cumulative positive historical regression in the registered history;
- no learner old-sample/old-label/replay access;
- growth rescued all registered forced collisions and children were reused;
- wrong-certificate causal control produced false-safe events and regression.

**Negative evidence / limitation**
- linear writable Cell;
- fixed feature coordinates;
- explicit routing;
- certificate is constructed from registered activations;
- synthetic conflict construction.

**Formal status**
**SUPPORTED — controlled**.

**Known ceiling**
Exact protection of the registered linear history plus controlled capacity rescue.

**What it does NOT prove**
A fixed sufficient certificate exists for arbitrary Transformer weights, language-scale activation spans stay low-rank, semantic routing is solved, or an opaque checkpoint contains reconstructable certificates.

**Engineering primitive obtained**
Explicit functional contracts, safe projected local writes, saturation detection, and growth as a transactional fallback.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 006 — real-representation continual plasticity


**Claim**
Can the Core-005 style bounded state support replay-free continual learning on real pretrained language-model representations without rapid saturation or growth explosion?

**Evidence source**
[research/projects/continual-learning/experiments/core-006-real-representation-continual-plasticity/validation/core-006-real-representation-continual-plasticity/README.md](../../research/projects/continual-learning/experiments/core-006-real-representation-continual-plasticity/validation/core-006-real-representation-continual-plasticity/README.md).

**Positive evidence**
- real Pythia representations did not immediately fill the 64-D projected space;
- functional reuse increased;
- certificate writes reduced registered-history forgetting relative to unsafe writes;
- candidate retained substantial new-learning gain relative to replay;
- spawned children were reused.

**Negative evidence**
- `REAL_REPRESENTATION_CONTINUAL_PLASTICITY_NOT_SUPPORTED`, 0/3 formal seeds;
- spawned Cells covered ~65.6–68.8% of addresses, above the registered bound;
- address-based splits failed the registered conflict-relief requirement;
- detailed Kaggle artifacts were partly lost after session closure and were not reconstructed.

**Formal status**
**NOT SUPPORTED**, with positive sub-results.

**Known ceiling**
Real hidden-state low-dimensional/reusable geometry is useful, but routing address is not equivalent to a writable functional boundary.

**What it does NOT prove**
That semantic addresses are Cells, that growth remains bounded, or that the certificate mechanism alone solves continual learning.

**Engineering primitive obtained**
Real-model functional regression measurement and certificate-based damage reduction.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 007 — functional-boundary discovery


**Claim**
Can write-demand/interference geometry provide a better split boundary than semantic address identity, and can that identity be recovered from inference-visible representations?

**Evidence source**
[research/projects/continual-learning/experiments/core-007-functional-boundary-discovery/validation/core-007-functional-boundary-discovery/README.md](../../research/projects/continual-learning/experiments/core-007-functional-boundary-discovery/validation/core-007-functional-boundary-discovery/README.md).

**Positive evidence**
- discovery selected `interference_cut` under the frozen mechanism-selection score;
- discovery showed high soft-Top2 oracle coverage and useful z/oracle agreement.

**Negative evidence / limitation**
- discovery is not confirmation;
- the first confirmation infrastructure lost a completed seed before durable checkpointing, forcing seed retirement;
- the main README still records later boundary work as an unresolved/negative integration area.

**Formal status**
The checked-in validation README is **confirmation incomplete / amended protocol**, not a supported scientific confirmation.

**Known ceiling**
Promising discovery geometry only.

**What it does NOT prove**
A stable deployable functional boundary, safe nonlinear updates, or autonomous CLM growth.

**Engineering primitive obtained**
Resumable per-seed checkpoint/publish discipline and explicit separation of discovery from confirmation.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 008 family — certified functional atoms / equivalence / postmortem capacity


**Claim / role**
Interrogate whether increasingly local functional units can be certified and whether apparent Cell evidence survives causal/control analysis.

**Evidence source**
`research/governance/validations/core-008-*` and associated stage documents.

**Positive evidence**
Produced useful causal controls, functional-equivalence preflights and capacity diagnostics that narrowed which local effects deserve later testing.

**Negative evidence / limitation**
The family did not establish a universal natural Cell atom or a trained-language continual-learning solution. Some results are diagnostic/postmortem rather than formal model milestones.

**Formal status**
Mixed family; preserve per-validation statuses.

**Known ceiling**
Mechanism selection and causal diagnostics.

**What it does NOT prove**
That a local causal effect is a globally useful or stable Cell boundary.

**Engineering primitive obtained**
Stronger preflight/control methodology before expensive formal runs.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Core 009 family — factorized/effect geometry and deployable addressability


**Claim / role**
Probe whether mature model write/effect geometry can supply sparse, persistent, compositional and deployable functional coordinates.

**Evidence source**
`research/governance/validations/core-009a-*`, `core-009b*`, `core-009c-*`, `core-009d-*`, plus branch-published artifacts where applicable.

**Positive evidence**
The family provides bridge evidence that pretrained-model representations/effects contain useful local and factorized signals worth decoding.

**Negative evidence / limitation**
The checked-in tree contains multiple sequential/revised 009 hypotheses, and not all branch-published outcomes are normalized into the main validation directories. The accumulated evidence has not established a stable, low-growth natural Cell boundary that transfers into Native continual-language success.

**Formal status**
**MIXED / family-level audit**; do not collapse the individual 009 decisions into one positive claim.

**Known ceiling**
Foundation-interface diagnostics.

**What it does NOT prove**
That sparse local effect geometry is the correct Cell ontology, or that effect addressability solves read/write ownership in a trained CLM.

**Engineering primitive obtained**
Functional-effect probes and a more disciplined interface between foundation-model diagnostics and CLM experiments.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

### 05 Language Validation: Constructive CLM

#### Constructive CLM-001 / 001B — coordinate formation and latent discovery


**Claim / role**
Can a persistent addressable coordinate system form from experience, including under the registered additive superposition scaffold?

**Positive evidence**
Both registered constructive claims are supported on their frozen formal seeds.

**Formal status**
**SUPPORTED — controlled**.

**Known ceiling**
Singleton-exposure and registered additive pair-superposition worlds.

**What it does NOT prove**
Arbitrary blind source separation or natural coordinates in pretrained LLMs.

**Engineering primitive obtained**
Learned coordinate/address formation in an engineered substrate.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Constructive CLM-002 — long-horizon structure-tracking growth


**Claim / role**
Can growth track reusable latent structure rather than simply spawn one Cell per observation?

**Positive evidence**
Supported to the registered finite horizon: `N_final=4096`, 30 latent factors, 30 final Cells, finite-horizon fitted growth exponent ~0.663.

**Formal status**
**SUPPORTED — controlled finite horizon**.

**Known ceiling**
Finite-horizon evidence only.

**What it does NOT prove**
An asymptotic sublinear-growth theorem or language-scale bounded growth.

**Engineering primitive obtained**
Structure-tracking growth controller under known constructive assumptions.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Constructive CLM-003 — protected learned/growing Cells


**Claim / role**
Integrate learned/growing Cells with replay-free protected writes.

**Positive evidence**
`PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED` on seeds `90511/90512/90513`.

**Formal status**
**SUPPORTED — controlled**.

**Known ceiling**
Learned root plus linear protected-write integration.

**What it does NOT prove**
Nonlinear Transformer Cell safety or Native continual-language success.

**Engineering primitive obtained**
Protected append/growth composition in an explicit modular substrate.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Constructive CLM-004 — model-level multi-Cell computation


**Claim**
Can multiple learned Cells participate sparsely in nontrivial model-level simultaneous and order-sensitive sequential computation while preserving a protected mutation invariant?

**Evidence source**
[research/projects/cellular-language-models/experiments/constructive-clm-004-model-level-multicell-computation/validation/constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md](../../research/projects/cellular-language-models/experiments/constructive-clm-004-model-level-multicell-computation/validation/constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md).

**Positive evidence**
- `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`;
- seeds `90611/90612/90613`;
- all 17 registered gates pass;
- held-out simultaneous and sequential composition, sparse execution, route support/order recovery and protected mutation all pass.

**Formal status**
**SUPPORTED — controlled**.

**Known ceiling**
Controlled linear residual operators.

**What it does NOT prove**
Arbitrary nonlinear Transformer Cells, language generation, learned/endogenous routing/growth/write control, LLM-scale endogenous CLM or JAM execution.

**Engineering primitive obtained**
Sparse multi-module composition as a model-level structural primitive.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Constructive CLM-005 — learned control-plane transition


**Claim**
Can routing/write/growth scaffolds transition to learned controllers while retaining earlier constructive invariants?

**Evidence source**
[research/projects/cellular-language-models/experiments/constructive-clm-005-scaffold-removal/validation/constructive-clm-005-scaffold-removal/FORMAL_RESULT.md](../../research/projects/cellular-language-models/experiments/constructive-clm-005-scaffold-removal/validation/constructive-clm-005-scaffold-removal/FORMAL_RESULT.md).

**Positive evidence**
- `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`;
- seeds `90811/90812/90813`;
- all 20 registered gates pass;
- learned router/growth/write heldout meta accuracy 1.0 in the registered world;
- bounded reusable children, sparse composition and replay-free protected behavior retained.

**Negative evidence / limitation**
Core-005 certificate basis and constrained update solver remain fixed safety primitives.

**Formal status**
**SUPPORTED — controlled**.

**Known ceiling**
Learned control plane over a supplied safety geometry in a controlled linear-residual world.

**What it does NOT prove**
Fully learned safety geometry, arbitrary nonlinear Cells, language-scale continual learning, asymptotic growth, or foundation-scale Native CLM.

**Engineering primitive obtained**
Learned candidate-control policies can coexist with deterministic safety/validation primitives.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

### 06 Native CLM

#### Native CLM v0 M0/M1 — architecture and next-token trainability


**Claim / role**
Can the Native CLM architecture execute and train end-to-end from next-token loss?

**Positive evidence**
- Stage-06 M0/M1 complete;
- canonical ~12.15M-parameter model trained successfully;
- validation loss `5.723429 -> 0.788535` and perplexity `305.9523 -> 2.2002` in the registered run;
- sparse active Cell fraction `2/8`.

**Formal status**
**SUPPORTED — trainability**.

**Known ceiling**
End-to-end trainability of a small token-predictive Native CLM.

**What it does NOT prove**
Continual learning, retention, autonomous growth or superiority to dense/MoE baselines.

**Engineering primitive obtained**
A real trainable Native-CLM test substrate for falsifying stronger mechanisms.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Native CLM v0 M2 — fixed-topology replay-free continual language


**Claim**
Are fixed-topology protected Cells sufficient for replay-free continual language learning?

**Positive evidence**
- certificate projection reduced mean forgetting from ~0.2790 to ~0.2115;
- protected path preserved roughly 96% of unsafe plasticity in the registered run.

**Negative evidence**
- `NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED`;
- formal seeds `73211/73212/73213`;
- A/TinyStories regression ~43.9% against the registered <=20% target.

**Formal status**
**NOT SUPPORTED**, with partial causal value for protection.

**Known ceiling**
Protection helps, but fixed topology and current certificate semantics do not provide sufficient continual-language retention.

**What it does NOT prove**
Replay-free continual language learning or that better protection alone will solve the problem.

**Engineering primitive obtained**
Certificates as bounded regression guards, not as a complete learner.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Native M2-R0 / M2-R0b — optimizer/update invariant audit


**Claim / role**
Determine whether the registered safe-gradient constraint survives the optimizer/parameter transaction.

**Evidence source**
- R0 published commit `c1f9e132b026efe24a0238e6ea333bdd2ae5fbdb`;
- R0b published commit `3addfea3e98c792bf487432f0ab22b787e2a4d49`.

**Positive evidence**
- R0 exposed that canonical AdamW grad projection does not imply safe realized update;
- R0b classifies the SGD reference issue as finite-precision parameter-transaction roundoff;
- R0b identifies `ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT`;
- projecting the final realized AdamW update reduces committed p95 violation to ~`5.44e-6`, at the matched numerical floor.

**Important threshold discipline**
The SGD+weight-decay arm is not post-hoc promoted to a material structural violation: its excess factor (~3.95) remains below the preregistered material threshold 16.

**Formal status**
**SUPPORTED — mechanics/diagnostic**. R0b explicitly has `scientific_decision=false` and `native_clm_training_milestone=false`.

**Known ceiling**
Optimizer transaction correctness.

**What it does NOT prove**
Certificate semantic coverage, old-function sufficiency or continual-learning success; it does not change the historical M2 decision.

**Engineering primitive obtained**
`raw optimizer proposal -> realized-update projection/validation -> commit`.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Native CLM v0 M3 — global-pool growth-restored continual language


**Claim**
Does adding fresh writable Cells restore continual-language retention after M2?

**Evidence source**
[research/projects/cellular-language-models/experiments/native-clm-v0-m3-growth-restored-continual-language/validation/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md](../../research/projects/cellular-language-models/experiments/native-clm-v0-m3-growth-restored-continual-language/validation/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md).

**Positive evidence**
- growth occurs and remains within its registered topology bound;
- children are reused 100% under the registered reuse metric;
- growth-phase plasticity and sparse compute survive;
- zero learner replay.

**Negative evidence**
- `NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED` on seeds `73411/73412/73413`;
- growth A regression ~0.484–0.494, worse than matched fixed protected topology ~0.429–0.442;
- global Top-K insertion changes read ownership even when a new child begins as an exact clone.

**Formal status**
**NOT SUPPORTED**.

**Known ceiling**
Fresh capacity is not enough; read ownership must also be preserved.

**What it does NOT prove**
That growth is generally useless. It falsifies the registered global-pool growth mechanism.

**Engineering primitive obtained**
A strong invariant for future expansion: **safe write growth requires safe read-address growth**.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### Native CLM v0 M3R — read-preserving / lineage-isolated growth


**Claim**
Does preserving top-level root read ownership make growth safe enough for continual retention?

**Positive evidence**
- root-route probe identities and birth root ownership are conserved;
- growth reaches 16 Cells with 100% registered child reuse;
- sparse compute, zero replay and plasticity remain;
- A retention improves by ~2.5 percentage points relative to matched global growth.

**Negative evidence**
- `NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED`;
- A regression remains ~0.469–0.472, far above the <=20% absolute target;
- child execution remains broad across old and new domains.

**Formal status**
**NOT SUPPORTED**, with a positive causal sub-result for root read conservation.

**Known ceiling**
Top-level read ownership can be conserved; lineage-local parent/child functional addressing remains unresolved.

**What it does NOT prove**
That hierarchical routing is sufficient for continual learning.

**Engineering primitive obtained**
Lineage isolation as a safer expansion topology than unrestricted global insertion.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### M3R Address Diagnostic — local query geometry


**Claim**
Within a stable lineage, is the parent/child functional split present in current frozen query geometry or write/effect geometry?

**Positive evidence**
- 24/24 valid lineage edges in the canonical diagnostic;
- current cosine median AUC ~0.5315;
- free affine query probe median AUC ~0.9623;
- classification recorded in the research index as `QUERY_GEOMETRY_SEPARABLE`.

**Formal status**
**SUPPORTED — diagnostic only**, no new continual-learning seeds.

**Known ceiling**
The representation contains a recoverable offline local boundary; the deployed cosine rule does not decode it.

**What it does NOT prove**
That the boundary can be learned online without historical state, or that using it restores continual retention.

**Engineering primitive obtained**
Offline address diagnostics can distinguish representation failure from decoder/routing-rule failure.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### M3L Query-Sketch Gate


**Claim**
Can the useful lineage-local affine boundary be recovered without replaying old tokens/queries, using a compact rank-16 historical Gaussian query sketch plus the current conflict stream?

**Positive evidence**
- offline oracle median AUC remains high (~0.9281);
- rank-16 sketch is a near miss at median AUC ~0.8968;
- edge-floor, normalized-oracle-recovery, old-FPR and current-TPR gates pass.

**Negative evidence**
- frozen classification `QUERY_SKETCH_GATE_NOT_FEASIBLE` because registered median AUC >=0.9000 is missed.

**Formal status**
**NOT SUPPORTED — diagnostic**, no new continual-language seed claim.

**Known ceiling**
The registered compact sketch is insufficient under its frozen threshold; the remaining question is capacity/family, not whether query geometry contains signal.

**What it does NOT prove**
That all compact historical state is insufficient, or that raw replay is necessary.

**Engineering primitive obtained**
Historical address summaries can be audited explicitly for information capacity before another expensive training run.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

#### M3L-1 / later address-state diagnostics


**Claim / role**
Separate low-rank capacity limits from Gaussian-family/model-family limits using the exact frozen M3L samples and oracle.

**Formal status**
**ACTIVE / diagnostic in the checked-in research index**. Do not infer a new continual-learning milestone.

**Known ceiling**
Checkpoint-only mechanism diagnosis.

**What it does NOT prove**
Any improvement to M2/M3/M3R unless a later full registered model run transfers the mechanism.

**Engineering primitive obtained**
Escalate representation capacity only after a fixed-sample diagnostic identifies the actual bottleneck.

---


Synthesis source: [Research Ledger](audits/RESEARCH_LEDGER.md). Unstated chronology and successor relationships remain UNKNOWN.

### 07 Safe Model Evolution / 08 Hybrid CLM / Independent Native Sparse Paths

Experiment-level integration of these histories is still pending. Their project records exist, but neither project names nor README summaries substitute for verified results: [Safe Model Evolution](../projects/safe-model-evolution/README.md), [Cellular Language Models](../projects/cellular-language-models/README.md), [CLM Native](../projects/clm-native/README.md). No missing causal chain is invented here.

## 3. Research Line Evolution

Continual-learning research progressed from scoped writes to protection under constrained writable capacity, then to growth and functional addressing. Core006 explicitly identifies Core005 as its parent. The Native M2/M3/M3R sequence, as described in the Ledger, separates optimizer correctness, capacity rescue, root read ownership and lineage-local addressing. Constructive CLM provides controlled positive evidence for composition; it does not invalidate the failures observed in different, less scaffolded settings.

## 4. Cross-Experiment Evidence and Stable Claims

| Claim ID | Bounded claim | Status | Evidence and limitation |
|---|---|---|---|
| EM-C001 | Core005 supports certificate-protected writes and growth in its registered linear setting | SUPPORTED | [Core005 results](../projects/continual-learning/experiments/core-005-subspace-certified-mitosis/artifacts/RESULTS.md); not arbitrary nonlinear Cells |
| EM-C002 | Core009B-1 supports carrier causal sufficiency in its confirmation setting | SUPPORTED | [Confirmation results](../projects/continual-learning/experiments/core-009b1-carrier-causal-sufficiency/artifacts/confirmation/RESULTS.md); not reuse/addressability/growth |
| EM-C003 | The tested Core009C candidate family met all discovery gates | CONTRADICTED | [Discovery results](../projects/continual-learning/experiments/core-009c-sparse-local-effect-geometry/artifacts/discovery/RESULTS.md); restricted search, not impossibility |
| EM-C004 | M3W-0 establishes successful new Native CLM training | NOT_ESTABLISHED | [M3W-0 results](../projects/cellular-language-models/experiments/native-clm-v0-m3w0-write-drift-restoration/artifacts/RESULTS.md); checkpoint-only diagnosis |
| EM-C005 | General continual language learning or order-of-magnitude LLM speedup is established | NOT_ESTABLISHED | [Ledger](audits/RESEARCH_LEDGER.md), [native project scope](../projects/clm-native/README.md); controlled mechanisms do not establish this |
| EM-C006 | Core006 establishes system-level real-representation continual plasticity | NOT_ESTABLISHED | [Formal decision](../projects/continual-learning/experiments/core-006-real-representation-continual-plasticity/artifacts/decision.json); 0/3 passes, positive sub-hypotheses retained |

Protection has positive evidence in Core005/Core006/M2, while Core006 and M2 fail their system-level goals. Capacity rescue in Core004 does not make Native M3/M3R successful. Constructive004/005 support controlled composition, not arbitrary nonlinear language computation. These are cross-study interpretations, not extra experimental observations.

### Status mapping and revision history

Experiment Outcome, Claim Status and review state are separate. A recorded formal failure maps to Outcome FAIL; the broader positive research hypothesis may remain NOT_ESTABLISHED rather than impossible. Original labels such as SUPPORTED—controlled remain quoted in the history.

This revision corrects EM-C001 from ESTABLISHED to SUPPORTED to match the controlled evidential scope. EM-C006 previously used CONTRADICTED for the narrower proposition that all seeds passed; the current formulation records the unresolved system-level positive hypothesis while preserving its 0/3 failure fact. Earlier wording remains in Git history; these are interpretation corrections, not changes to experimental results.

## 5. Complete Experiment History: Migration Status

The detailed families above are integrated narrative, not a certification of complete coverage. The [discovery inventory](audits/EVIDENCE_MAP_INVENTORY.md) remains an audit appendix. Its 141 directories are work units, not 141 independent experiments. Families, runs, seeds, diagnostics, engineering validations, proposals and governance records must be distinguished before experiment totals can be certified. IDs require namespaces; an unqualified `001` is not globally unique.

Review states are NOT_YET_REVIEWED, RESULT_NOT_FOUND_AFTER_REVIEW, INSUFFICIENT_EVIDENCE, CONFLICT_REVIEW_REQUIRED and NOT_AN_INDEPENDENT_EXPERIMENT. The 137 unclassified work units are NOT_YET_REVIEWED, not evidence that their results are absent. The 39 unowned identity declarations and two potential same-run status differences remain explicit audit tasks.

## 6. Current Research Assessment

The supported foundation consists of controlled mechanisms, bounded diagnostics and trainability in identified settings. Functional protection, plasticity, addressing and actual cost have not yet been established together as a general language-learning system. Parameter count and sparse execution are not measured latency gains. Long-term research value does not establish short-term product value, and a current product limitation does not disprove basic research potential.

### Open research questions

| ID | Question | Evidence gap |
|---|---|---|
| EM-Q001 | Do linear certificates transfer to nonlinear language models? | Cross-setting confirmation |
| EM-Q002 | Can causal carriers be deployed, addressed and reused? | Sufficiency versus addressability |
| EM-Q003 | Do local paths improve quality and actual cost? | Matched controls and hardware measurement |
| EM-Q004 | Can restoration diagnostics enable safe new learning? | Diagnosis versus a trained learner |
| EM-Q005 | Which historical revisions have documented causal dependencies? | Source-level chronology and identity resolution |

Questions are not approved experiments. No resource or training authorization is created by this Map.

### Decision history

- EM-D001, date UNKNOWN: preserve controlled versus system-level evidence boundaries, as recorded in the Ledger.
- EM-D002, date UNKNOWN: the native-sparse-path project describes itself as independent of paused expensive CLM-Bridge follow-up; see its project README. This is an existing statement, not a new decision.
- EM-D003, 2026-10-30: establish a unified evidence view without rewriting experiment records.

### Update history

2026-10-30: initial main discovery index; subsequent research-history restructuring; this audit removes duplicated old template sections, switches the primary narrative to English, reconciles claim labels and refreshes structural/link checks. Full historical acceptance remains pending, especially stages 07/08 and unreviewed work units.

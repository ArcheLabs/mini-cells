# Historical Research Asset Map

Audit date: **2026-09-03**

This document classifies the pre-Core and pre-Native notebook lineages by their **current scientific authority** and by the **engineering primitives that remain useful even when the original interpretation was later weakened, falsified, or superseded**.

It is not a new scientific result. It does not change any frozen protocol or formal decision.

## Why this map exists

MiniCells accumulated several generations of research before the later Core / Constructive / Native formal discipline. Those notebooks still contain valuable mechanisms, negative lessons, training infrastructure, and design ideas, but their current value is easy to misread in two opposite ways:

1. **over-promotion** — treating an exploratory or historical result as current evidence for Native continual learning;
2. **over-deletion** — discarding a useful engineering primitive because the stronger scientific interpretation later failed.

The audit therefore separates two questions:

```text
What does this work scientifically establish today?
                    !=
What engineering primitive from it is still worth reusing?
```

## Status vocabulary

| Classification | Meaning |
|---|---|
| **HISTORICAL EXPLORATORY** | Useful hypothesis formation / feasibility history; not current formal evidence for continual learning |
| **HISTORICAL MECHANISTIC EVIDENCE** | Reproducible or repeated local mechanism evidence, but below the later formal system-level standard |
| **ENGINEERING PRECURSOR EVIDENCE** | Historical scientific interpretation may be limited, but the implementation/design primitive directly survives into current safe-model-evolution engineering |
| **RETIRED / SUPERSEDED PROTOCOL LINEAGE** | Important experimental/protocol asset whose scientific question is now tested more directly by a later canonical lineage |
| **CANONICAL FORMAL EVIDENCE** | Reserved for later frozen registered validations; notebook location alone never grants this status |

The notebook directory is an execution/archive location, not an evidence rank.

---

# 1. `research/notebooks/01-foundations`

## Current classification

**HISTORICAL EXPLORATORY**

Primary historical question:

> Can cellular/NCA-like local recurrent computation provide a useful language-model substrate, and which of its dynamics remain trainable and controllable?

Representative lineage includes Echo/basic trainability, quantization localization, native continual-learning previews, tiny arithmetic, consumer-language bridge/ablation/scaling, early 30M work, optimizer search, 1D/2D latent tissue, adaptive halting, settling/stabilization and related controls.

Canonical stage summary: `research/stages/01-foundations/README.md`.

## Durable scientific value

This lineage established useful **feasibility and failure boundaries**:

- local/recurrent cellular computation can be trained in controlled language-like tasks;
- local state and repeated computation are technically usable modelling ingredients;
- halting/settling/stabilization are measurable rather than purely metaphorical concepts;
- 2D/NCA-inspired topology does not by itself produce continual learning;
- scaling and stable settling can be expensive or fragile;
- visually interesting cellular dynamics are not a substitute for a registered capability claim.

## What it does NOT prove

It does not establish:

- that language is best represented by a literal 1D/2D NCA tissue;
- that spatial Cell topology is necessary for CLM;
- that cellular recurrence solves catastrophic forgetting;
- that local settling gives a globally better model;
- that any early Cell boundary corresponds to a natural knowledge atom.

## Surviving engineering primitives

| Primitive | Current value |
|---|---|
| explicit local state | useful for modular/adaptive runtime state |
| repeated computation | reusable where iterative refinement is beneficial |
| adaptive stopping / halting probes | useful evaluation/runtime concept, not a CL proof |
| controlled numerical/mechanistic probes | high methodological value |
| trainability/scaling baselines | useful historical regression references |
| quantization/localization instrumentation | potentially reusable for deployment diagnostics |

## Engineering reuse rule

Reuse the **mechanism or instrumentation**, not the old NCA/biological interpretation. A product component should not inherit scientific authority from this notebook family.

## Repository action

**KEEP as historical research assets.**

Do not keep this directory on the active Native-CLM evidence path. Individual notebooks may later be tagged `canonical-historical`, `supporting`, or `archive-candidate`, but should not be deleted merely because the original architecture direction changed.

---

# 2. `research/notebooks/02-self-organization`

## Current classification

**HISTORICAL MECHANISTIC EVIDENCE**

Primary historical question:

> Can useful Cell organization, recruitment, differentiation and growth emerge from local pressure rather than being completely hand-designed?

The lineage includes multi-seed stabilization, emergent sparse topology, reaction-diffusion-like plasticity, local-substrate topology, growing cellular language models, localized learning, conditional/pressure recruitment, proposal utility, recruitment response curves, capability specificity, conflict-driven differentiation, trait genesis and probationary genesis.

Canonical stage summary: `research/stages/02-self-organization/README.md`.

## Durable scientific value

This family supplied repeated evidence that **local adaptive organization can be induced and measured**, and it helped discover several mechanisms that later reappeared in more disciplined forms:

- sparse/local organization under pressure;
- conditional recruitment;
- local mutability;
- differentiation after conflict or overload;
- probation before permanent structural growth;
- the need for multi-seed controls and anti-null-mode checks.

Equally important, the lineage exposed a durable negative lesson:

```text
interesting emergence
!=
validated functional boundary
!=
validated continual learning
```

## What it does NOT prove

It does not establish:

- a natural Cell ontology;
- that emergent topology aligns with writable functional independence;
- that recruitment/growth remains bounded at language scale;
- that local pressure is sufficient to decide what a model should learn;
- that self-organized specialization survives global evaluation after many updates.

## Surviving engineering primitives

| Primitive | Current value |
|---|---|
| conditional recruitment | useful candidate-allocation policy |
| pressure/conflict trigger | useful signal for proposing, not automatically accepting, structural change |
| probationary growth | **high value** for shadow/fork-before-commit workflows |
| localized mutability | high value for scoped model changes |
| sparse topology | useful implementation structure when explicit |
| multi-seed / null-mode discipline | high methodological value |

## Engineering reuse rule

Pressure/recruitment signals may **propose** a Cell/fork/update. They must not be treated as a global quality oracle. Acceptance belongs to stage-level/global evaluation.

## Repository action

**KEEP, with selective canonicalization.**

The strongest mechanism notebooks should remain easy to discover; visually interesting or redundant variants can later move to `research/archive/` after references are audited. Archiving must not rewrite the historical conclusion.

---

# 3. `research/notebooks/03-routing-and-growth`

## Current classification

**ENGINEERING PRECURSOR EVIDENCE**

This is the most directly reusable of the four historical notebook groups.

Primary historical transition:

```text
Cell as biological analogy
        ->
Cell as routed computational unit
        ->
Cell as independently mutable model state
```

The lineage includes CLM-0.1, CLM-0.3 progressive growth, release benchmarks, marginal growth utility, counterfactual/probationary mitosis, upcycling, program conditionality, CLM-v2 handoff/closed-loop work, routing/growth variants and sparse-runtime investigations.

Canonical stage summary: `research/stages/03-routing-and-growth/README.md`.

## Durable scientific value

The old routing/growth experiments do **not** establish the final correct router or Cell ontology. Later Core/Native results show that semantic/address and global-pool read geometry can fail badly.

However, the stage established a durable architectural fact: a Cell can be engineered as an explicit, sparse, independently mutable computation/state unit. That is useful even if it is not a natural knowledge atom.

## What it does NOT prove

It does not establish:

- that historical routing heuristics are correct functional addresses;
- that marginal local utility equals global model improvement;
- that mitosis should be accepted online without global evaluation;
- that growth is sufficient for retention;
- that Cell granularity is superior to LoRA/module/expert granularity for merge/fork;
- that any CLM-0.x result supersedes later formal Native negatives.

## Surviving engineering primitives

| Primitive | Current value |
|---|---|
| sparse routed computation | **high** |
| explicit Cell identity/state | **high** |
| independently mutable module | **high** |
| versionable Cell state | **high** |
| progressive append/expand | **high**, but acceptance must be global |
| counterfactual candidate | **high** for branch/shadow evaluation |
| probationary mitosis | **high** for commit/rollback architecture |
| marginal-utility accounting | useful as one metric, never a sole acceptance rule |
| upcycling / inherit-then-differentiate | potentially useful for fork/merge experiments |
| structural routing metadata | useful provenance even when semantic meaning is not trusted |

## Current engineering interpretation

This stage should be read as the origin of a **model-evolution substrate**, not as proof of a new biological learning paradigm.

The strongest reusable pattern is:

```text
accepted model state
      |
      +-- create local candidate / branch
      |       |
      |       +-- mutate or append explicit Cell/module state
      |       +-- preserve provenance and dependencies
      |       +-- evaluate counterfactually
      |
      +-- global acceptance gate
              |
              +-- commit
              +-- rollback/discard
```

This maps directly onto the current safe-model-evolution engineering direction.

## Repository action

**KEEP as active engineering heritage.**

Do not archive the family wholesale. Later cleanup should make canonical engineering-heritage notebooks easy to find and move only redundant historical variants after reference analysis.

---

# 4. `research/notebooks/05-language-validation`

## Current classification

**RETIRED / SUPERSEDED PROTOCOL LINEAGE**

Primary historical question:

> Can the dependency-scoped transactional growth loop transfer from controlled synthetic functions to a real autoregressive token-level model under an explicit stable addressing plane?

The directory contains CLM-0.4-mini M0/M1 infrastructure, calibration, preview and release notebooks.

Canonical protocol lineage:

- `research/validations/clm-0.4-mini-language-validation/`
- `research/validations/clm-0.4-mini-m1-v2-language-validation/`

## Scientific status

The lineage must **not** be described as either a successful or failed formal CLM-0.4 continual-learning result.

The original formal experiment was not completed. The v1 development path stopped at a base prerequisite failure; v2 explicitly revised data/admission alignment and remained in a pre-formal protocol/data-lock state. Later Native CLM Stage 06 tests the trained token-predictive continual-learning question more directly and therefore supersedes CLM-0.4-mini as the active scientific path.

## Durable methodological value

CLM-0.4-mini introduced several disciplined design elements that remain highly useful:

- explicit M0 software smoke vs M1 scientific decision vs M2 engineering scale rehearsal;
- frozen protocol and seed discipline;
- candidate model changes before commit;
- dependency-scoped validation;
- rollback;
- zero-output growth initialization;
- atomic Cell/route commit;
- Cell registry/version state;
- transaction journal/checkpoint replay;
- dense/equal-compute controls;
- separation of a stable certification address from a diagnostic semantic router.

These are engineering/protocol assets, not evidence that the scientific hypothesis passed.

## What it does NOT prove

It does not establish:

- CLM-0.4 continual-learning success;
- CLM-0.4 continual-learning failure under a completed formal protocol;
- superiority over dense or MoE baselines;
- that explicit metadata addressing is a product solution;
- that growth/transaction logic transfers safely to foundation-scale models.

## Surviving engineering primitives

| Primitive | Current value |
|---|---|
| smoke -> formal -> scale-gate separation | **high methodological value** |
| speculative candidate + rollback | **high** |
| transaction journal | **high** for reproducibility/provenance |
| atomic structural commit | **high** |
| explicit versioned Cell registry | **high** |
| zero-output/function-preserving birth goal | **high**, though later routing still needs global validation |
| dense/equal-compute controls | high experimental value |
| stable evaluator/control plane separated from learned router | high safety value |

## Repository action

**KEEP, but mark scientifically retired/superseded.**

Do not spend GPU time resurrecting the old CLM-0.4-mini formal sequence merely to close it. Reuse its protocol/transaction ideas in the current engineering line or in new explicitly registered experiments. Native Stage 06 remains the canonical trained-model evidence path.

---

# Cross-family asset matrix

| Asset / idea | 01 Foundations | 02 Self-Organization | 03 Routing/Growth | 05 Language Validation | Current disposition |
|---|---|---|---|---|---|
| cellular/NCA topology as primary paradigm | explored | explored | de-emphasized | not central | **research history only** |
| local mutable state | early | strengthened | explicit Cell state | versioned registry | **retain** |
| sparse computation | early | emergent | explicit routed primitive | controlled Cell-FFN | **retain** |
| pressure/conflict signal | limited | central | used for growth | transaction trigger | **retain as proposal signal** |
| automatic local acceptance | implicit risk | explored | explored | constrained by validator | **do not trust globally** |
| probation/counterfactual candidate | precursor | developed | explicit | speculative transaction | **retain strongly** |
| append/growth | explored | developed | explicit | atomic growth bundle | **retain with global validation** |
| semantic/natural Cell interpretation | speculative | speculative | unresolved | deliberately bypassed | **not established** |
| rollback/transaction | limited | precursor | explicit precursor | explicit protocol | **retain strongly** |
| protocol/seed discipline | early | improving | improving | strong | **retain** |
| true continual-language evidence | none | none | none | incomplete | **use Native M2/M3/M3R instead** |

---

# Product / engineering extraction rule

A historical primitive may move into a product design only if all of the following are true:

1. **The primitive is separated from its old scientific narrative.**  
   Example: probationary growth may be reused without claiming autonomous mitosis is scientifically solved.

2. **Its local decision is treated as candidate generation unless a later registered result proves more.**  
   Local utility, pressure, routing score, certificate fit or novelty are not global model-quality oracles.

3. **The accepted model is evaluated globally at a defined stage boundary.**

4. **The change remains versionable and reversible until acceptance.**

5. **Later negative evidence overrides earlier interpretation.**  
   M2/M3/M3R limits apply even if an older notebook looked positive.

6. **Engineering value is measured against existing alternatives.**  
   Cells must eventually earn their complexity against adapters, LoRA, MoE modules, external memory and model-merging/versioning baselines.

---

# Cleanup policy for historical notebooks

Do not mass-delete or mass-move these notebooks.

Future cleanup should assign each notebook one of:

```text
CANONICAL_HISTORICAL
ENGINEERING_HERITAGE
SUPPORTING_HISTORICAL
ARCHIVE_CANDIDATE
SUPERSEDED_PROTOCOL
```

Physical movement requires the same reference-audit discipline as research scripts: update allowed links and leave compatibility references when a published reproduction path depends on the old location.

The objective is not a visually minimal tree. The objective is a tree in which **evidence authority, historical value and reusable engineering value cannot be confused**.

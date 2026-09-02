[English] | [中文](README.zh-CN.md)

# MiniCells Research

MiniCells separates the **product architecture** from the stronger **endogenous / Native CLM research hypothesis**.

```text
mature pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / Native CLM
```

The external product path may use engineered persistent Cells, sparse routing, replay-free protection, growth, versioning and rollback. Native-CLM research asks how far those mechanisms can become learned, compositional and endogenous.

## Native CLM core progress

This table is the stable high-level research scoreboard. When a registered experiment or milestone is completed, update the evidence and status here instead of redefining the roadmap.

Status legend:

- 🟢 **Supported / complete** — formal evidence or a completed engineering milestone that should be reused rather than re-tested.
- 🟡 **Partial evidence** — useful evidence exists, but the stronger Native-CLM claim remains open.
- 🔵 **Active** — current main experiment or training milestone.
- ⚪ **Planned** — later gate in the stable sequence.
- 🔴 **Not supported / blocked** — the registered hypothesis failed or should not be used as the main path.

| # | Native CLM proposition | Current evidence | Status |
|---:|---|---|---|
| 1 | Functional organization can emerge under pressure | Experiments 014–024 | 🟡 Strong emergence evidence; not itself continual-learning proof |
| 2 | Sparse Cells can be independently mutable computational units | 025/026, CLM-0.1–0.3 | 🟢 Reusable mechanism |
| 3 | Conflict can trigger differentiation / growth | 021–024, Core 004 | 🟢 Reusable mechanism |
| 4 | Growth can restore plasticity | Core 004 | 🟢 Formally supported |
| 5 | Historical behavior can be protected without learner-side replay | Core 005; real-representation bridge in Core 006 | 🟢 Certificate principle supported |
| 6 | Mature LLMs expose a useful writable interface | Core 006, 009A, 009B-1 | 🟢 Strong foundation-interface evidence; **not** a natural Cell ontology |
| 7 | Reusable Cell coordinates / read addresses can form from experience | Constructive CLM-001 and 001B | 🟢 Controlled constructive formation supported, including no-singleton superposition discovery |
| 8 | Long-horizon Cell growth can track reusable structure rather than transaction count | Constructive CLM-002 | 🟢 Finite-horizon structure-tracking growth supported; not an asymptotic theorem |
| 9 | Learned/growing Cells can support replay-free protected continual writes | Constructive CLM-003 | 🟢 `PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED`, 3/3 formal seeds |
| 10 | Multiple learned Cells can perform stable model-level computation/composition | Constructive CLM-004 | 🟢 `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`, 3/3 formal seeds |
| 11 | Router/write/growth scaffolds can be removed toward endogenous control | Constructive CLM-005 | 🟢 `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`, 3/3 formal seeds |
| 12 | Train the first Small Native CLM v0 from next-token loss | **Stage 06 M0/M1** | 🔵 **Active** |

The important negative boundary remains:

```text
pretrained semantic/routing address
!=
automatically correct functional Cell boundary
```

Core 006 and the 002/009 natural-geometry line prevent us from returning to that assumption.

## Current main program

### Native CLM v0 — M0 + M1

Constructive CLM mechanism validation is now closed. The next question is no longer whether a controlled Cell system can be constructed. It is:

> Can a real token-predictive neural model train end-to-end while its internal computation is sparsely routed through persistent Cells?

The first implementation intentionally separates variables:

```text
M0  architecture / execution smoke
M1  ~12M next-token training
M2  continual language stream
M3  autonomous Cell growth
M4  Cell ontology / specialization analysis
M5  Dense Transformer / static-MoE comparison
```

M1 does not absorb M2/M3. The first language run keeps Cell count fixed so a training failure can be attributed to token modeling / learned routing rather than mixed with continual-learning and growth-policy failures.

Canonical Stage-06 roadmap: [Native CLM](stages/06-native-clm/README.md).

## Constructive CLM sequence — closed

```text
G1a  CLM-001   addressable learned coordinate formation      🟢
G1b  CLM-001B  latent discovery under superposition          🟢
G2   CLM-002   long-horizon structure-tracking growth        🟢
G3   CLM-003   protected learned/growing Cells                🟢
G4   CLM-004   model-level multi-Cell computation             🟢
G5   CLM-005   scaffold removal / endogenous transition       🟢
                                                          ↓
                                              Native CLM v0
```

The canonical evidence-reuse/no-repeat policy is frozen in the [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md), with a machine-readable companion at [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml).

See the [Continual-Learning Research Roadmap](validations/CONTINUAL_LEARNING_ROADMAP.md) for the frozen constructive evidence chain and transition boundary.

## Two research tracks

### Track A — Foundation Interface Research

Core 006–009 characterize a mature pretrained LLM as a writable substrate for an external CLM layer.

Current conclusion:

```text
useful low-dimensional/factorized write interface
!=
ready-made natural Cell ontology
```

Core 009D remains a non-blocking operator-geometry diagnostic. Track-A negatives do not invalidate the product path or the now-completed Constructive CLM evidence chain.

### Track B — Constructive CLM Research

The controlled Native-CLM feasibility sequence is **closed** through CLM-005. Formal parent evidence includes:

- **CLM-001** — `LEARNED_COORDINATE_FORMATION_SUPPORTED`, seeds `90111/90112/90113`.
- **CLM-001B** — `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`, seeds `90211/90212/90213`.
- **CLM-002** — `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`, seeds `90411/90412/90413`.
- **CLM-003** — `PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED`, seeds `90511/90512/90513`.
- **CLM-004** — `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`, seeds `90611/90612/90613`.
- **CLM-005** — `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`, seeds `90811/90812/90813`; all 20 registered gates passed on all three formal seeds.

Do not create cosmetic CLM-005B/006 synthetic extensions. New work should name a real-model integration or scaling variable.

## Stage 06 target model

The canonical M1 model is deliberately around 12M parameters rather than 30M:

```text
byte vocabulary             256
context                     256
d_model                     384
shared Transformer blocks     6
Cellular Layers               1
initial Cells                 8
active Cells/token            2
parameters                  ~12.15M
```

The Cellular Layer executes only selected Cell operators and keeps per-Cell route key, certificate, usage and lineage state. M0 separately verifies dynamic spawn and dynamic checkpointing; autonomous growth remains M3.

## Research stages

1. [Foundations](stages/01-foundations/README.md) — Echo, NCA language dynamics, 1D/2D tissue, settling, and training mechanics.
2. [Self-Organization](stages/02-self-organization/README.md) — sparse topology, recruitment, differentiation, and trait genesis.
3. [Routing and Growth](stages/03-routing-and-growth/README.md) — routed, independently mutable Cell state and capacity growth.
4. [Continual-Learning Core](stages/04-continual-learning-core/README.md) — write-addressability failures, growth-restored plasticity, replay-free certificates, real-representation constraints, foundation-interface geometry, and Constructive CLM.
5. [Language Validation](stages/05-language-validation/README.md) — historical token-level transfer/scale-readiness work.
6. [Native CLM](stages/06-native-clm/README.md) — real token-predictive Native CLM training, continual streams, autonomous growth, ontology analysis and scaling comparisons.

## Research assets

- [Experiment implementations](experiments/README.md) are stage-aligned adapters that import reusable code from `src/minicells/`.
- [Notebook assets](notebooks/README.md) preserve historical workflows and Stage-06 GPU training orchestration.
- [Validations](validations/) contain frozen protocols, evidence maps and decision documents.
- [Canonical artifacts](../artifacts/experiments/) are immutable scientific evidence once formal runs are published; model-training milestones also publish incomplete runs when informative.
- Machine-readable historical paths/outcomes remain in [`catalog.yaml`](catalog.yaml).

## Current boundary

The repository now supports controlled constructive feasibility through learned routing/write/growth control. It still does **not** establish general natural-language continual learning, an asymptotic `K(N)=o(N)` theorem, arbitrary latent-source discovery, arbitrary nonlinear Cell safety, language-scale autonomous growth, or LLM-scale endogenous CLM.

The current task is therefore concrete: train and evaluate the first **Small Native CLM v0**, beginning with M0/M1 rather than returning to indefinite synthetic mechanism validation.

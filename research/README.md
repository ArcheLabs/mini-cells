[English] | [中文](README.zh-CN.md)

# MiniCells Research

MiniCells separates the **product architecture** from the stronger **endogenous / Native CLM research hypothesis**.

```text
mature pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / Native CLM
```

## Native CLM core progress

Status legend:

- 🟢 **Supported / complete** — reusable formal evidence or completed engineering milestone.
- 🟡 **Partial evidence** — useful evidence exists, but the stronger claim remains open.
- 🔵 **Active** — current registered experiment.
- ⚪ **Planned** — later gate.
- 🔴 **Not supported / blocked** — registered hypothesis failed or should not remain the main path.

| # | Native CLM proposition | Current evidence | Status |
|---:|---|---|---|
| 1 | Functional organization can emerge under pressure | Experiments 014–024 | 🟡 Strong emergence evidence |
| 2 | Sparse Cells can be independently mutable computational units | 025/026, CLM-0.1–0.3 | 🟢 Reusable mechanism |
| 3 | Conflict can trigger differentiation / growth | 021–024, Core 004 | 🟢 Reusable mechanism |
| 4 | Growth can restore plasticity | Core 004 | 🟢 Formally supported |
| 5 | Historical behavior can be protected without learner-side replay | Core 005; Core 006 bridge | 🟢 Certificate principle supported |
| 6 | Mature LLMs expose a useful writable interface | Core 006, 009A, 009B-1 | 🟢 Strong foundation-interface evidence; not a natural Cell ontology |
| 7 | Reusable Cell coordinates can form from experience | Constructive CLM-001 / 001B | 🟢 Controlled constructive formation supported |
| 8 | Long-horizon growth can track reusable structure | Constructive CLM-002 | 🟢 Finite-horizon structure tracking supported |
| 9 | Learned/growing Cells can support protected continual writes | Constructive CLM-003 | 🟢 Formally supported |
| 10 | Multiple learned Cells can perform model-level composition | Constructive CLM-004 | 🟢 Formally supported |
| 11 | Router/write/growth scaffolds can transition toward learned control | Constructive CLM-005 | 🟢 Formally supported |
| 12 | A real next-token Native CLM can train end-to-end | Stage 06 M0/M1 | 🟢 `NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS` |
| 13 | Fixed-topology protected Cells are sufficient for replay-free continual language | Stage 06 M2 | 🔴 `...M2...NOT_SUPPORTED`; protection itself has strong partial evidence |
| 14 | Dynamic Cell growth restores protected continual-language capacity | **Stage 06 M3** | 🔵 **Active / frozen before formal run** |

## Current main experiment — Native CLM v0 M3

M1 trained the canonical 12,154,368-parameter Native CLM from next-token loss while retaining `2/8 = 25%` sparse Cell execution. Its checkpoint is pinned at:

```text
HF repo   archelabsxyz/native-clm-v0
file      final-model.pt
SHA-256   91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 then tested fixed-topology replay-free continual language. Certificate projection consistently reduced forgetting (~0.2115 protected vs ~0.2790 unsafe) while preserving ~96% of unsafe plasticity, but final TinyStories-A regression remained ~43.9% against the registered <=20% gate. M2 is therefore a valid frozen negative result; formal seeds `73211/73212/73213` are consumed.

M3 asks the next causal question:

> Starting from the same exact M1 checkpoint, can autonomous context-addressed Cell growth restore old-domain retention beyond a matched fixed-topology protected control while preserving new-domain plasticity and zero learner replay?

Because the original M2 local data manifest was lost with the terminated Kaggle session, M3 creates a new exact Hub-revision-pinned A/B/C/D snapshot and compares both arms on that same snapshot:

```text
GPU0  fixed_protected     8 Cells forever
GPU1  growth_protected    8 -> at most 16 Cells
```

The growth controller may inspect only current learner-visible pressure: training loss, Cell route hits, certificate rank, projected/raw gradient ratio, and frozen-router query vectors. It cannot see domain/phase labels, evaluation metrics, hidden novelty labels, or old training samples.

New children exactly clone the parent operator at birth, receive a context-derived frozen route key, start with an empty certificate, and must demonstrate post-birth route reuse.

Formal seeds are untouched:

```text
73411 / 73412 / 73413
```

Canonical documents:

- [Stage 06 — Native CLM](stages/06-native-clm/README.md)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)
- [M2 formal closure](stages/06-native-clm/M2_CLOSURE.md)
- [M3 frozen protocol](validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json)
- [M3 validation README](validations/native-clm-v0-m3-growth-restored-continual-language/README.md)

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  fixed-topology replay-free continual language      🔴
M3  growth-restored continual language                 🔵
M4  Cell ontology / specialization                     ⚪
M5  Dense Transformer / static-MoE comparison          ⚪
```

Do not scale to 30M before M3 closes. If M3 is supported, the next step is M4 at the same scale before a scaling reproduction.

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

The canonical evidence-reuse/no-repeat policy remains frozen in the [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md), with machine-readable companion [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml).

## Research stages

1. [Foundations](stages/01-foundations/README.md)
2. [Self-Organization](stages/02-self-organization/README.md)
3. [Routing and Growth](stages/03-routing-and-growth/README.md)
4. [Continual-Learning Core](stages/04-continual-learning-core/README.md)
5. [Language Validation](stages/05-language-validation/README.md)
6. [Native CLM](stages/06-native-clm/README.md) — **active** real token-predictive continual-learning/growth line.

## Current boundary

The repository supports the controlled constructive mechanism chain and a successfully trained 12.15M Native CLM v0. It has evidence that certificate-projected writes reduce real-language forgetting, but fixed 8-Cell continual learning failed the registered absolute-retention gate. It does **not** yet establish growth-restored continual language, semantic Cell ontology, Dense/MoE superiority, asymptotic `K(N)=o(N)`, or LLM-scale endogenous CLM. M3 is the registered experiment for the first of those open boundaries.

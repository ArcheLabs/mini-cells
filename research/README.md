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
| 13 | The trained Native CLM can learn a replay-free continual language stream with protected Cell-local writes | **Stage 06 M2** | 🔵 **Active** |

The important negative boundary remains:

```text
pretrained semantic/routing address
!=
automatically correct functional Cell boundary
```

## Current main experiment — Native CLM v0 M2

M1 is closed. The canonical 12,154,368-parameter model trained successfully from next-token loss while retaining `2/8 = 25%` sparse Cell execution:

```text
initial validation loss   5.723429
final validation loss     0.788535
initial perplexity         305.9523
final perplexity           2.2002
```

Canonical checkpoint:

```text
HF repo   archelabsxyz/native-clm-v0
file      final-model.pt
SHA-256   91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 now asks:

> Starting from that exact checkpoint, can protected sparse Cell-local writes learn sequential language distributions with zero learner-side replay while retaining prior language behavior better than identical unsafe writes?

Registered M2 stream:

```text
A TinyStories         evaluation-only retention anchor
B WikiText-2 raw      train
C CodeParrot code     train
D Databricks Dolly    train
```

Training is `B -> C -> D`; old training data never return to the learner. The shared substrate and learned router are frozen so M2 isolates Cell-local write protection. Growth stays disabled until M3.

Two Kaggle GPUs are used as concurrent causal arms rather than DDP:

```text
GPU0 protected certificate-projected writes
GPU1 unsafe identical writes without projection
```

Formal seeds: `73211 / 73212 / 73213`.

Canonical Stage-06 roadmap and protocol:

- [Stage 06 — Native CLM](stages/06-native-clm/README.md)
- [M2 frozen protocol](validations/native-clm-v0-m2-continual-language/protocol.json)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  replay-free continual language                    🔵
M3  autonomous Cell growth                            ⚪
M4  Cell ontology / specialization                    ⚪
M5  Dense Transformer / static-MoE comparison         ⚪
```

Do not scale to 30M before M2/M3 close. The open problem is continual behavior and topology adaptation, not basic next-token trainability.

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
6. [Native CLM](stages/06-native-clm/README.md) — **active** real token-predictive training/continual-learning line.

## Research assets

- [Experiment implementations](experiments/README.md) import reusable code from `src/minicells/`.
- [Notebook assets](notebooks/README.md) preserve historical workflows and Stage-06 Kaggle orchestration.
- [Validations](validations/) contain frozen protocols and scientific decisions.
- [Canonical artifacts](../artifacts/experiments/) become immutable evidence after publication.
- Binary model checkpoints live in the separate Hugging Face model repository; Git stores exact SHA/revision provenance and lightweight evidence.

## Current boundary

The repository supports the controlled constructive mechanism chain and a successfully trained 12.15M Native CLM v0. It does **not** yet establish replay-free continual natural-language learning, autonomous language-scale growth, semantic Cell ontology, Dense/MoE superiority, asymptotic `K(N)=o(N)`, or LLM-scale endogenous CLM. M2 is the registered experiment that now tests the first of those remaining boundaries.

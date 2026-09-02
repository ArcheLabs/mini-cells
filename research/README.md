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

- 🟢 supported / complete
- 🟡 partial evidence
- 🔵 active blocking gap / next registered design
- ⚪ later milestone
- 🔴 registered hypothesis not supported

| # | Native CLM proposition | Evidence | Status |
|---:|---|---|---|
| 1 | Functional organization can emerge under pressure | Experiments 014–024 | 🟡 |
| 2 | Sparse Cells can be independently mutable computational units | 025/026, CLM-0.1–0.3 | 🟢 |
| 3 | Conflict can trigger differentiation / growth | 021–024, Core 004 | 🟢 |
| 4 | Growth can restore plasticity in the controlled CLM loop | Core 004 | 🟢 formal |
| 5 | Historical behavior can be protected without learner replay | Core 005; Core 006 bridge | 🟢 principle |
| 6 | Mature LLMs expose a useful writable interface | Core 006, 009A, 009B-1 | 🟢 interface evidence |
| 7 | Reusable Cell coordinates can form from experience | Constructive CLM-001 / 001B | 🟢 formal |
| 8 | Long-horizon growth can track reusable structure | Constructive CLM-002 | 🟢 formal finite horizon |
| 9 | Learned/growing Cells can host protected continual writes | Constructive CLM-003 | 🟢 formal |
| 10 | Multiple learned Cells can compose at model level | Constructive CLM-004 | 🟢 formal |
| 11 | Router/write/growth scaffolds can transition toward learned control | Constructive CLM-005 | 🟢 formal |
| 12 | A real next-token Native CLM can train end-to-end | Stage 06 M0/M1 | 🟢 complete |
| 13 | Fixed-topology protected Cells are sufficient for replay-free continual language | Stage 06 M2 | 🔴 not supported; protection has partial causal value |
| 14 | Global-pool context-addressed growth restores continual-language retention | Stage 06 M3 | 🔴 not supported |
| 15 | Read-preserving / lineage-isolated growth can restore continual retention | Stage 06 M3R | 🔵 next blocking design |

## Trained-model evidence

### M1 — trainability supported

Canonical 12,154,368-parameter Native CLM trained successfully from next-token loss:

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

Canonical checkpoint:

```text
HF repo   archelabsxyz/native-clm-v0
file      final-model.pt
SHA-256   91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

### M2 — protection helps, fixed topology fails

Formal decision:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate projection reduced mean forgetting from ~0.2790 to ~0.2115 while preserving ~96% of unsafe plasticity, but final A/TinyStories regression remained ~43.9% versus the registered <=20% gate.

### M3 — fresh capacity grows, but read geometry is unsafe

Formal decision:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
protocol = 9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658
seeds = 73411 / 73412 / 73413
```

All three seeds grew from 8 to 16 Cells, achieved 100% registered child reuse, preserved sparse compute, zero learner replay and new-domain plasticity — yet A retention became consistently **worse** than the matched fixed-topology protected control:

```text
seed 73411  fixed A reg 0.4416  -> growth 0.4938
seed 73412  fixed A reg 0.4293  -> growth 0.4838
seed 73413  fixed A reg 0.4351  -> growth 0.4889
```

The new scientific boundary is therefore not merely writable capacity:

```text
safe write growth requires safe read-address growth
```

Post-formal artifact analysis shows strong route leakage. In seed 73411, the four children born during B already captured about 40% of Cell routing mass on **every** A/B/C/D evaluation domain; after C, the eight children captured about 50% of A routing mass. Raw child reuse is therefore not equivalent to correct address reuse.

## Current blocking gap — M3R

The next experiment must introduce a genuinely new mechanism: **function-preserving, lineage-isolated read growth**.

Preferred invariant:

```text
old root router selects the same root lineages before and after growth
                           ↓
a lineage-local gate selects parent vs child
```

At birth, parent gate mass should be conserved within the lineage so that an exact parent clone creates zero forward-function drift. New children must not immediately enter global Top-K competition with unrelated roots.

M3R should register:

- birth-time function invariance;
- old-context root-lineage route invariance;
- child selectivity, not route-hit reuse alone;
- bounded non-cap-saturating growth;
- zero replay and protected writes;
- restoration of the same absolute A-retention boundary that failed M2/M3.

Do **not** advance to M4 ontology analysis or 30M scaling until this gap is closed.

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  fixed-topology replay-free continual language      🔴
M3  global-pool growth-restored continual language     🔴
M3R read-preserving / lineage-isolated growth          🔵
M4  Cell ontology / specialization                     ⚪ BLOCKED
M5  Dense Transformer / static-MoE comparison          ⚪
```

## Canonical documents

- [Stage 06 — Native CLM](stages/06-native-clm/README.md)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)
- [M2 closure](stages/06-native-clm/M2_CLOSURE.md)
- [M3 formal result](validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)
- [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md)

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

Constructive support remains reusable evidence. The M2/M3 trained-model negatives do not invalidate those controlled mechanisms; they identify the missing integration invariant in a real token-predictive model.

## Product boundary

External CLM remains a separate near-term product path. Native-CLM trained-model failures do not invalidate engineered persistent Cells, routing, certificates, growth, versioning and rollback on top of mature pretrained LLMs.

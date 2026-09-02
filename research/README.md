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
- 🔵 active diagnostic / blocking gap
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
| 14 | Global-pool growth restores continual-language retention | Stage 06 M3 | 🔴 not supported |
| 15 | Read-preserving / lineage-isolated growth restores continual retention | Stage 06 M3R | 🔴 not supported; root read conservation has partial causal value |
| 16 | The local parent/child functional split is recoverable from current query geometry | M3R Address Diagnostic | 🟢 `QUERY_GEOMETRY_SEPARABLE` |
| 17 | A compact replay-free historical query sketch is sufficient to recover the local affine gate | M3L Query-Sketch Gate | 🔴 `QUERY_SKETCH_GATE_NOT_FEASIBLE`; single-gate near miss |
| 18 | The M3L shortfall is rank/capacity-limited rather than Gaussian-family-limited | M3L-1 Address-State Capacity | 🔵 active diagnostic |

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

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate projection reduced mean forgetting from ~0.2790 to ~0.2115 while preserving ~96% of unsafe plasticity, but A/TinyStories regression remained ~43.9% against the registered <=20% gate.

### M3 — fresh capacity grows, but global read geometry is unsafe

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73411 / 73412 / 73413
```

All three seeds grew from 8 to 16 Cells with 100% registered child reuse and preserved new-domain plasticity, yet A retention became worse than the matched fixed-topology protected control. Post-formal analysis showed newly spawned global candidates stealing old-context Top-K ownership.

The resulting invariant was:

```text
safe write growth requires safe read-address growth
```

### M3R — root read conservation works, lineage-local address still fails

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
protocol = c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627
seeds = 73611 / 73612 / 73613
artifact commit = 986b043a5d2f5ee9140cf35b14f68aacc3b7a942
HF revision = a23b521e137a7e44616809895d44d87cc7d6f87f
```

M3R kept the original eight M1 roots as the only top-level candidates and routed children only inside their root lineage.

What transferred successfully:

- A/B/C/D root-route probe hashes stayed invariant throughout B -> C -> D;
- birth root ownership and gate probabilities were conserved;
- growth reached 16 Cells and child reuse stayed at 100%;
- sparse compute, zero replay and plasticity were preserved;
- A retention improved by a small but stable ~2.5 percentage points relative to matched global growth.

What did not transfer:

```text
seed 73611  global A reg 0.4967 -> lineage 0.4722
seed 73612  global A reg 0.4947 -> lineage 0.4691
seed 73613  global A reg 0.4961 -> lineage 0.4713
```

The absolute <=20% A-regression gate remained far away. Final child execution share was still broad across domains (~52% A, ~55% B, ~59% C, ~54% D), so the local cosine rule did not form a selective functional boundary.

The new blocking gap is therefore narrower than M3:

```text
root read ownership can be conserved
but lineage-local parent/child functional address remains unresolved
```

## Completed address diagnostics and current blocker

### M3R Address Diagnostic — 🟢 `QUERY_GEOMETRY_SEPARABLE`

Checkpoint-only analysis over 24/24 M3R lineage edges found that the current parent/child cosine rule was nearly random (median AUC ~0.5315), while a free affine probe on the same frozen query representation reached median AUC ~0.9623. The functional split is therefore present in query geometry, but is not decoded by centroid/cosine addressing.

### M3L Query-Sketch Gate — 🔴 `QUERY_SKETCH_GATE_NOT_FEASIBLE`

M3L tested whether that affine boundary could be recovered without old-token/query replay in gate fitting, using only a rank-16 Gaussian historical query sketch plus the current conflict stream. The result was a frozen single-gate negative:

```text
valid edges                  24/24
offline oracle median AUC    0.9281
rank-16 sketch median AUC    0.8968   (required >=0.9000)
edge-floor fraction          0.7500   PASS
normalized oracle recovery   0.9356   PASS
old FPR                      0.1855   PASS
current TPR                  0.8204   PASS
```

The miss is concentrated most strongly in first differentiation A -> B; B -> C and A+B -> C are substantially stronger.

### M3L-1 Historical Address-State Capacity — 🔵 ACTIVE

Before another continual-language formal run, M3L-1 keeps the exact M3L samples, temporal ownership semantics, sequence-group split, thresholds and oracle fixed while sweeping the historical address state:

```text
diagonal / rank 0
rank 8
rank 16  (must reproduce M3L exactly)
rank 32
rank 64
rank 128
full dense covariance
offline linear oracle
```

The diagnostic distinguishes `LOW_RANK_CAPACITY_SUFFICIENT`, `FULL_COVARIANCE_REQUIRED`, and `GAUSSIAN_FAMILY_LIMITED`. It is checkpoint-only, consumes no new formal seeds, and cannot retroactively change M3L/M3R/M3.

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  fixed-topology replay-free continual language      🔴
M3  global-pool growth-restored continual language     🔴
M3R read-preserving / lineage-isolated growth          🔴
M3R Address Diagnostic                                 🔵
M4  Cell ontology / specialization                     ⚪ BLOCKED
M5  Dense Transformer / static-MoE comparison          ⚪
```

Do **not** advance to M4 ontology analysis or 30M scaling until the local functional-address mechanism is selected and validated.

## Canonical documents

- [Stage 06 — Native CLM](stages/06-native-clm/README.md)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)
- [M2 closure](stages/06-native-clm/M2_CLOSURE.md)
- [M3 formal result](validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)
- [M3R Address Diagnostic](validations/native-clm-v0-m3r-address-diagnostic/README.md)
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

Constructive support remains reusable evidence. The M2/M3/M3R trained-model negatives do not invalidate those controlled mechanisms; they progressively isolate the missing integration invariants in a real token-predictive model.

## Product boundary

External CLM remains a separate near-term product path. Native-CLM trained-model failures do not invalidate engineered persistent Cells, routing, certificates, growth, versioning and rollback on top of mature pretrained LLMs.

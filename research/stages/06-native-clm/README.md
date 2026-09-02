[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE — ONLINE HISTORICAL ADDRESS-STATE INTEGRATION**

Stage 06 moves the formally supported Constructive CLM mechanisms into a real token-predictive model.

## Stable roadmap

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
      certificate protection reduced forgetting          🟡 PARTIAL EVIDENCE
  M3  global-pool growth-restored continual language     🔴 NOT SUPPORTED
  M3R read-preserving / lineage-isolated growth          🔴 NOT SUPPORTED
      root read ownership was preserved                  🟡 PARTIAL EVIDENCE
  M3R Address Diagnostic                                 🟢 QUERY GEOMETRY SEPARABLE
  M3L query-sketch lineage gate                           🔴 NOT FEASIBLE
  M3L-1 historical address-state capacity                🟢 LOW_RANK_CAPACITY_SUFFICIENT
      minimum passing low-rank state = 32
  M3L-2 online historical address-state integration      🔵 FROZEN / UNRUN
  M4  Cell ontology / specialization analysis            ⚪ BLOCKED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

Do not scale to 30M and do not advance to M4 until M3L-2 has a valid formal decision.

## Canonical substrate

```text
parameters                  12,154,368 at M1 start
vocab                       256 UTF-8 bytes
context                     256
shared width                384
shared blocks               6
attention heads             6
FFN width                   1536
Cellular Layers             1
initial Cells               8
active Cells/token          2
Cell operator               384 × 384 linear residual
certificate max rank        64
```

Canonical M1 checkpoint:

```text
Hugging Face  archelabsxyz/native-clm-v0
file          final-model.pt
SHA-256       91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

## M0 — Architecture + execution — 🟢

M0 established sparse routing, Cell-local gradients, certificate projection, dynamic spawn, optimizer enrollment, dynamic checkpoint round-trip and generation.

## M1 — Real next-token training — 🟢

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

See [M1_CLOSURE.md](M1_CLOSURE.md).

## M2 — Fixed-topology continual language — 🔴 NOT SUPPORTED

Formal decision:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate protection was causally useful:

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

But protected A regression remained ~43.9% against the registered <=20% ceiling. See [M2_CLOSURE.md](M2_CLOSURE.md).

## M3 — Global-pool growth — 🔴 NOT SUPPORTED

Formal decision:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73411 / 73412 / 73413
```

Growth reached 16 Cells with 100% child reuse and preserved plasticity, but A regression worsened from roughly 43–44% in the fixed control to roughly 48–49% under growth.

Post-formal analysis showed that children inserted into the global Top-K pool captured approximately half of old-domain Cell execution. The learned boundary was:

```text
safe write growth requires safe read-address growth
```

See the [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md).

## M3R — Read-preserving / lineage-isolated growth — 🔴 NOT SUPPORTED

Formal decision:

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
protocol = c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627
seeds = 73611 / 73612 / 73613
artifact commit = 986b043a5d2f5ee9140cf35b14f68aacc3b7a942
HF revision = a23b521e137a7e44616809895d44d87cc7d6f87f
```

M3R kept the original eight M1 roots as the only top-level routing candidates and allowed children to compete only inside their root lineage.

### What worked

Across all three seeds:

- root-route probe hashes for A/B/C/D were invariant from initial through after-B/C/D;
- birth root Top-K ownership and root probabilities were preserved;
- growth reached 16 Cells;
- child reuse remained 100%;
- B/C/D plasticity, zero replay and sparse execution passed;
- lineage routing improved A retention by a stable ~2.5 percentage points versus the matched global-growth control.

### What still failed

| seed | global A regression | lineage A regression | A advantage | lineage forgetting |
|---:|---:|---:|---:|---:|
| 73611 | 0.4967 | 0.4722 | 0.0245 | 0.2106 |
| 73612 | 0.4947 | 0.4691 | 0.0256 | 0.2089 |
| 73613 | 0.4961 | 0.4713 | 0.0248 | 0.2098 |

The registered absolute retention target remained <=20% A regression, so M3R remained far outside the target.

Children were still not sufficiently selective. Final child execution share was approximately:

```text
A   ~52.3%
B   ~55.1%
C   ~59.5%
D   ~53.6%
```

The lineage mechanism therefore solved **global root ownership**, but not the **parent/child functional boundary inside a lineage**. M3R still chose parent vs child using:

```text
q · k_child > q · k_parent
```

This is consistent with the earlier Core 006/007 warning that representation/query similarity is not automatically a safe functional mitosis address.

## M3R Address Diagnostic — 🟢 QUERY GEOMETRY SEPARABLE

The completed checkpoint-only diagnostic found 24/24 valid lineage edges. Current cosine addressing had median AUC ~0.5315, while a free affine query probe reached ~0.9623. Query geometry contains the boundary; centroid/cosine decoding does not recover it.

## M3L — Replay-Free Query-Sketch Gate — 🔴 NOT FEASIBLE

Using the stricter temporal parent-lifetime ownership and sequence-group-heldout split, the offline affine oracle remained separable (median AUC 0.9281). A rank-16 Gaussian historical query sketch recovered median AUC 0.8968, narrowly below the frozen >=0.90 gate while passing every other registered feasibility metric. M3L therefore remains a valid negative mechanism diagnostic.

## M3L-1 — Historical Address-State Capacity — 🟢 LOW_RANK_CAPACITY_SUFFICIENT

M3L-1 kept the M3L data, edge ownership, splits, oracle and feasibility thresholds fixed and swept diagonal/rank-8/16/32/64/128/full-covariance Gaussian historical address states.

The completed diagnostic classified:

```text
LOW_RANK_CAPACITY_SUFFICIENT
minimum passing rank = 32
publish commit = 5ace6faf344b1b805752a33ffb861aeaf34dad6e
```

This closes the checkpoint-level capacity selection question: the registered Gaussian second-order family did not require dense covariance, but rank 16 was too small for the frozen feasibility rule. This does **not** establish online continual-learning success.

## M3L-2 — Online Historical Address-State Integration — 🔵 FROZEN / UNRUN

M3L-2 is the new registered formal experiment. It compares the exact M3R lineage-cosine algorithm against the same protected-write/growth algorithm with a persistent rank-32 historical query state and affine lineage-local parent/child gate.

Registered address state:

```text
rank                         32
query width                  384
maximum persistent bytes     52,360 / Cell
current queries / leaf/batch <= 256
bootstrap batches            160
```

Because the historical M1 checkpoint predates address sidecars, M3L-2 explicitly registers a one-time pre-continual bootstrap from 10,000 TinyStories `train` documents at the pinned TinyStories revision. It performs no optimizer, parameter, certificate or growth update. Its one-shot data handle is released before B starts; A retention continues to use the separate TinyStories `validation` split.

After continual start, learner replay remains zero. At each 50-step growth check, current routed-query sufficient statistics are either merged into an unsplit leaf's rank-32 state or become the child state if that leaf splits. The pre-window parent history freezes at split. Child operators are exact parent clones, so birth remains function-preserving.

New untouched formal seeds:

```text
74211 / 74212 / 74213
```

Canonical details and execution surface are recorded in [M3L2_REGISTRATION.md](M3L2_REGISTRATION.md).

## Evidence boundary

M2/M3/M3R formal seeds are consumed and must not be reused as untouched evidence. M3R Address Diagnostic, M3L and M3L-1 are checkpoint-only mechanism diagnostics and cannot retroactively change any formal M3R gate or decision.

M3L-2 formal seeds remain untouched until the canonical two-GPU Kaggle formal runner is deliberately executed. M4 ontology analysis and 30M scaling remain blocked until M3L-2 receives a valid formal decision.

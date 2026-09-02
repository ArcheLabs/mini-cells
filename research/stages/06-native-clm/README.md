[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE — LINEAGE-LOCAL FUNCTIONAL ADDRESS DIAGNOSIS**

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
  M3R Address Diagnostic                                 🔵 ACTIVE
  M4  Cell ontology / specialization analysis            ⚪ BLOCKED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

Do not scale to 30M and do not advance to M4 until the lineage-local functional-address gap is understood.

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
- lineage routing improved A retention by a small but extremely stable ~2.5 percentage points versus the matched global-growth control.

### What still failed

| seed | global A regression | lineage A regression | A advantage | lineage forgetting |
|---:|---:|---:|---:|---:|
| 73611 | 0.4967 | 0.4722 | 0.0245 | 0.2106 |
| 73612 | 0.4947 | 0.4691 | 0.0256 | 0.2089 |
| 73613 | 0.4961 | 0.4713 | 0.0248 | 0.2098 |

The registered absolute retention target remained <=20% A regression, so M3R remained far outside the target.

More importantly, children were still not sufficiently selective. Final child execution share was approximately:

```text
A   ~52.3%
B   ~55.1%
C   ~59.5%
D   ~53.6%
```

The lineage mechanism therefore solved **global root ownership**, but not the **parent/child functional boundary inside a lineage**. M3R still chose parent vs child using the scalar comparison:

```text
q · k_child > q · k_parent
```

where the child key is the mean query of the pressure window that caused birth.

This is consistent with the earlier Core 006/007 warning that representation/query similarity is not automatically a safe functional mitosis address.

## M3R Address Diagnostic — 🔵 ACTIVE

The next stage is deliberately diagnostic, not another formal continual-learning run.

It reuses the already-published M3R lineage checkpoints and the exact pinned M3R A/B/C/D snapshot. No Native CLM parameters are updated and no new formal seeds are consumed.

For every actual M3R `parent -> child` edge, samples are conditioned on reaching that edge's root/ancestor path before the local decision. The diagnostic compares domain A against the child's birth domain using:

```text
current cosine margin
frozen query q
Cell write input x
downstream write-left factor dL/dh_cell_out
normalized write pair [x, dL/dh_cell_out]
parent-certificate residual
```

The registered diagnostic classification is one of:

```text
QUERY_GEOMETRY_SEPARABLE
WRITE_EFFECT_GEOMETRY_SEPARABLE
NO_CLEAR_LOCAL_BOUNDARY
INCONCLUSIVE_COVERAGE
```

Interpretation:

- query separable -> learn a better lineage-local read gate;
- query fails but write/effect separates -> separate read and write addressing;
- neither separates -> investigate a richer learned functional coordinate before another router heuristic.

See [M3R Address Diagnostic protocol](../../validations/native-clm-v0-m3r-address-diagnostic/protocol.json).

## Evidence boundary

M2/M3/M3R formal seeds are consumed and must not be reused as untouched evidence. The address diagnostic is checkpoint-only and cannot retroactively change any formal M3R gate or decision.

M4 ontology analysis remains blocked until the functional-address mechanism is selected and validated in a new registered experiment.

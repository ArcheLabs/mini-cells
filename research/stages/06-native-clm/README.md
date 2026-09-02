[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE — READ-GEOMETRY GAP**

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
  M3R read-preserving / lineage-isolated growth          🔵 NEXT ACTIVE DESIGN
  M4  Cell ontology / specialization analysis            ⚪ BLOCKED ON M3R
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

Do not scale to 30M and do not advance to M4 until the read-geometry gap exposed by M3 is closed.

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

M1 trained the 12.15M Native CLM from real next-token loss:

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

M2 trained only Cell operators over the zero-replay stream `B -> C -> D`, with A/TinyStories evaluation-only and the shared substrate/router frozen.

Protection was causally useful:

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

But protected A regression remained ~43.9% against the registered <=20% ceiling. The M2 formal seeds are consumed.

See [M2_CLOSURE.md](M2_CLOSURE.md).

## M3 — Global-pool growth-restored continual language — 🔴 NOT SUPPORTED

Formal decision:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
protocol = 9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658
seeds = 73411 / 73412 / 73413
```

M3 compared, on the same pinned data snapshot and seed:

```text
fixed_protected   8 Cells forever
vs
growth_protected 8 -> at most 16 Cells
```

Both arms retained zero learner replay, frozen shared substrate/original router, two active Cells/token and certificate-projected Cell writes.

### Formal outcome

| seed | fixed A reg | growth A reg | growth advantage | fixed forgetting | growth forgetting | growth Cells | child reuse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 73411 | 0.4416 | 0.4938 | -0.0522 | 0.2137 | 0.2201 | 16 | 1.000 |
| 73412 | 0.4293 | 0.4838 | -0.0545 | 0.2107 | 0.2170 | 16 | 1.000 |
| 73413 | 0.4351 | 0.4889 | -0.0539 | 0.2123 | 0.2186 | 16 | 1.000 |

The failing registered gates on every seed were:

```text
growth_A_retention_advantage
growth_absolute_A_retention
growth_mean_forgetting
```

Growth itself worked mechanically: all seeds reached 16 Cells, children were reused, sparse compute survived, B/C/D plasticity passed, and zero replay remained true. Therefore the negative result is not explained by failure to allocate fresh capacity.

### Post-formal diagnosis: read-address leakage

The registered child key was the mean frozen-router query of current conflict contexts, and each child was inserted into the same global Top-K candidate pool as the original roots.

For seed `73411`, the four children born during B already received approximately:

```text
A route mass  40.33%
B route mass  40.01%
C route mass  41.52%
D route mass  39.32%
```

After C, all eight children received approximately:

```text
A route mass  50.30%
B route mass  51.79%
C route mass  57.21%
D route mass  50.59%
```

This is high reuse but poor address selectivity. New Cells steal substantial read traffic from old contexts.

M3 also reached the maximum eight children at steps `50/150/250/350/450/550/650/750`, before phase D. The growth rule therefore behaved close to cooldown-limited repeated spawning under persistent pressure.

The key learned boundary is:

```text
fresh writable capacity
!=
safe continual expansion
```

More specifically:

```text
safe write growth requires safe read-address growth
```

See the frozen [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md).

## M3R — Read-preserving / lineage-isolated growth — 🔵 NEXT ACTIVE DESIGN

M3R must be a new integration experiment, not a threshold-tuned M3 rerun.

Preferred computational invariant:

```text
root router selects the same original root lineages as before growth
                         ↓
within each selected lineage, a local gate chooses parent vs child
```

A child must not immediately enter a global competition with unrelated roots.

A stronger birth invariant is gate-mass-preserving mitosis. If the original parent receives gate mass `g_p`, after birth that same mass is split only inside the lineage:

```text
g_p * W_parent
        ↓ birth
g_p * [(1-alpha) W_parent + alpha W_child]
```

with `W_child = W_parent` at birth. Then the forward function is exactly unchanged for any `alpha`, while later child divergence can be restricted to contexts routed into that lineage.

The next frozen protocol should therefore test:

- near-zero/logit-exact functional drift at birth;
- root-lineage route invariance for old contexts;
- child selectivity rather than raw route-hit reuse alone;
- bounded non-cap-saturating growth;
- zero replay and protected writes;
- restoration of the same A-retention gate that failed M2/M3.

M4 ontology analysis remains blocked until this mechanism works in the trained token-predictive model.

## Evidence

- [M3 protocol](../../validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json)
- [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)
- canonical artifacts: `artifacts/experiments/native-clm-v0-m3-growth-restored-continual-language/`
- artifact commit: `e8b6a40f68862d6f01f67b125afdaeec97e6c45c`
- Hugging Face evidence revision: `4bc1e73518f09039335a368d4352ff0201cee06c`

The M3 formal seeds are consumed and must never be reused as untouched evidence.

[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE**

Stage 06 moves the formally supported Constructive CLM mechanisms into a real token-predictive model.

## Stable roadmap

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
      protection causally reduced forgetting             🟡 PARTIAL EVIDENCE
  M3  growth-restored continual language                 🔵 ACTIVE
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

Do not scale to 30M before M3 closes. The active uncertainty is whether topology growth restores protected continual capacity.

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

M2 started from the exact M1 checkpoint and trained only Cell operators through the replay-free stream:

```text
B WikiText-2 raw
  ↓
C cleaned Python CodeParrot
  ↓
D Databricks Dolly
```

A/TinyStories remained evaluation-only. Shared substrate and learned router were frozen.

Formal seeds `73211 / 73212 / 73213` all produced the same decision:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

Protection was nevertheless causally useful:

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

The only failed registered gate was absolute A retention: protected TinyStories regression remained ~43.9% against the pre-registered <=20% ceiling.

See [M2_CLOSURE.md](M2_CLOSURE.md). Those formal seeds are consumed and must never be reused as untouched evidence.

## M3 — Growth-Restored Continual Language — 🔵 ACTIVE

M3 does not change the failed M2 gate or tune on its seeds. It asks a new causal question:

> Does autonomous context-addressed Cell growth restore replay-free old-domain retention beyond an otherwise identical fixed-topology protected control while preserving new-domain plasticity?

Because the original M2 local data manifest was lost when the Kaggle session terminated, M3 freezes a new exact Hub-revision-pinned data snapshot and runs both causal arms on that same snapshot:

```text
GPU0  fixed_protected
      8 Cells forever

GPU1  growth_protected
      starts at 8 Cells
      may grow to at most 16 Cells
```

Both arms use:

- the same exact M1 checkpoint;
- identical B -> C -> D data and seed schedule;
- zero learner replay;
- certificate-projected Cell-local gradients;
- frozen shared Transformer, query projection, norm and original eight route keys;
- two active Cells per token.

### Autonomous growth signal

Every 50 learner steps, the growth arm may inspect only current-training quantities:

```text
window train loss
Cell route hits
certificate rank
projected/raw Cell-gradient ratio
frozen-router query vectors
```

No domain ID, phase name, evaluation metric, hidden novelty label or historical training example is visible to the growth decision.

When persistent protected-write pressure is detected, M3 chooses the Cell maximizing:

```text
route_hits * (1 - projected/raw gradient ratio)
```

and creates a child with:

```text
W_child        = exact W_parent clone
route_key      = mean current conflict-query vector
certificate    = empty / rank 0
parent_id      = lineage pointer
```

The exact operator clone is intended to avoid a large functional discontinuity at birth; the child then supplies fresh writable directions and must prove post-birth reuse.

### Registered M3 gates

All three untouched formal seeds must independently pass, including:

- fixed control remains 8 Cells and exposes >=30% A regression;
- growth uses learner-visible signals only;
- 1–8 children are created, final Cells <=16;
- >=75% of children receive >=512 post-birth routed token hits;
- active Cell compute remains <=30% of dense-all-Cell compute;
- each B/C/D phase improves >=5%;
- growth A regression <=20%;
- growth improves A retention by >=10 percentage points vs matched fixed control;
- growth mean forgetting <=15%;
- growth retains >=80% of fixed-control new-domain plasticity.

Development seeds:

```text
73301 / 73302 / 73303
```

Untouched formal seeds:

```text
73411 / 73412 / 73413
```

Positive status:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_SUPPORTED
```

Negative status:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

Frozen protocol: [`../../validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json`](../../validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json)

Canonical Kaggle notebook: [`../../notebooks/06-native-clm/native-clm-v0-m3-growth-restored-continual-language-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m3-growth-restored-continual-language-kaggle.ipynb)

The notebook preflights two GPUs and Hugging Face model-repository write permission **before** formal training, uploads all six final fixed/growth checkpoints to `archelabsxyz/native-clm-v0`, and Git-publishes only lightweight evidence.

## Advance rule

If M3 is supported, move to M4 Cell ontology/specialization analysis at the same scale before any 30M reproduction. If M3 is not supported, preserve the negative result and diagnose growth addressing/trigger/certificate lifecycle rather than tuning the formal seeds post hoc.

[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE**

Constructive CLM mechanism validation is closed through CLM-005. Stage 06 trains a real token-predictive model whose internal computation is routed through persistent Cells.

## Stable roadmap

Status legend remains repository-wide:

- 🟢 complete / supported
- 🟡 partial evidence
- 🔵 active
- ⚪ planned
- 🔴 blocked / not supported

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🔵 IMPLEMENTED / GPU RUN PENDING
  M2  continual language stream                          ⚪ PLANNED
  M3  autonomous Cell growth                             ⚪ PLANNED
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

The sequence is stable. M1 does **not** absorb M2/M3: first establish ordinary next-token trainability, then add continual-learning pressure, then make growth a scientific variable.

## Native CLM v0 definition

The first model is not an external memory layer. Tokens pass through a learned sparse Cellular Layer before the LM head:

```text
UTF-8 bytes
   ↓
Token + position embedding
   ↓
shared causal Transformer blocks
   ↓
learned sparse Cellular Layer
   ↓
remaining shared blocks
   ↓
LM head
   ↓
next-token loss
```

For Cell `i`, M0/M1 use the registered linear residual operator family:

```text
g_i(h) = W_i h
h' = h + Σ gate_i · g_i(h),  i ∈ TopK(router(h))
```

Cells carry persistent operator, route-key, certificate, usage and lineage state. The runtime supports dynamic Cell creation; autonomous growth-policy evaluation is intentionally deferred to M3.

## M0 — Architecture + execution — 🟢 COMPLETE

M0 is an engineering gate, not a scientific decision. GitHub CI passes the complete registered execution smoke. The runtime now verifies:

- next-token forward/backward;
- gradients reach router and Cell parameters;
- sparse top-k Cell execution;
- bounded Cell certificate updates;
- certificate-nullspace Cell-gradient projection;
- dynamic child spawn and optimizer enrollment;
- sparse execution after topology change;
- dynamic Cell-count checkpoint save/reload;
- generation after reload.

Canonical runner:

```bash
python scripts/research/run_native_clm_v0_m0.py
```

M0 smoke checkpoint weights are discarded after round-trip verification; lightweight decision artifacts are retained.

## M1 — First next-token Native CLM — 🔵 ACTIVE

M1 asks only:

> Can a real token-predictive model with a learned sparse Cellular Layer train end-to-end from next-token loss at a nontrivial but repeatedly trainable scale?

It does **not** yet claim continual learning, autonomous growth, ontology quality, or superiority to Dense/MoE baselines.

Canonical configuration:

```text
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
parameters                  ≈ 12.15M
```

Heterogeneous plasticity:

```text
shared LR   = 2e-4
router LR   = 4e-4
Cell LR     = 8e-4
```

Each Cell contains `W_i`, `route_key_i`, bounded certificate `Q_i`, `usage_count_i`, and `parent_id_i`. Before optimizer commit, Cell weight gradients are projected through the current certificate nullspace:

```text
dW <- dW (I - QᵀQ)
```

Canonical M1 keeps the Cell count fixed at eight. M0 proves the runtime can grow; M2 introduces continual pressure; M3 tests autonomous growth.

## M1 data and gates

Canonical Kaggle orchestration creates a local cache from public TinyStories using a byte tokenizer:

```text
train documents       50,000
validation documents   2,000
```

M1 is marked `NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS` only when all registered engineering gates hold:

- 10M–15M total parameters;
- requested optimizer steps complete with finite loss;
- validation loss improves by at least 5% from initialization;
- Cell execution fraction <=30% (`2/8 = 25%` canonically);
- router receives nonzero gradient;
- Cells receive nonzero gradient;
- generation executes;
- exactly one Cellular Layer is used;
- Cell count remains fixed so autonomous growth is not silently claimed.

`scientific_decision=false`: M1 is a real-model training milestone, not yet a formal continual-learning result.

## Checkpoints and repository artifacts

Kaggle keeps binary checkpoints in runtime storage. Git receives lightweight evidence only:

```text
summary.json
metrics.csv
run-config.json
sample.txt
RESULTS.md
data-manifest.json
```

`summary.json` records final checkpoint SHA-256 and byte size so model identity is preserved without turning Git into a weight registry.

## Kaggle notebook

Canonical notebook:

[`../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb)

Run it top-to-bottom with a Kaggle GPU and `GITHUB_TOKEN`. It clones the branch, runs M0, prepares TinyStories, trains canonical M1, prints every gate and a generation sample, then publishes lightweight M0/M1 evidence back to the branch. Passing and incomplete M1 runs are both publishable.

## Stop / advance rule

If M1 passes, do not scale immediately to 30M. Advance to **M2 — continual language stream** at the same ~12M scale. The first 30M run belongs after continual/growth behavior is understood, as a scaling confirmation rather than a debugging environment.

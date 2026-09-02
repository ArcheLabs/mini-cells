[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE**

Constructive CLM 001–005 closed the controlled mechanism-feasibility sequence. Stage 06 trains and evaluates those mechanisms inside a real token-predictive model.

## Stable roadmap

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
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  replay-free continual language stream              🔵 ACTIVE
  M3  autonomous Cell growth                             ⚪ PLANNED
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

Do not scale to 30M before M2/M3 close. The current scientific uncertainty is continual behavior, not basic trainability.

## Native CLM v0 substrate

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

The canonical substrate remains unchanged through M2:

```text
parameters                  12,154,368
vocab                       256 UTF-8 bytes
context                     256
shared width                384
shared blocks               6
attention heads             6
FFN width                   1536
Cellular Layers             1
Cells                       8
active Cells/token          2
Cell operator               384 × 384 linear residual
certificate max rank        64
```

For Cell `i`:

```text
g_i(h) = W_i h
h' = h + Σ gate_i · g_i(h), i ∈ TopK(router(h))
```

## M0 — Architecture + execution — 🟢 COMPLETE

M0 established the runtime: next-token forward/backward, sparse routing, router/Cell gradients, bounded certificate updates, certificate-nullspace gradient projection, dynamic child spawn, optimizer enrollment, dynamic-topology checkpoint round-trip, and generation after reload.

## M1 — First next-token Native CLM — 🟢 COMPLETE

M1 established that the real token-predictive architecture can train end-to-end from next-token loss while retaining sparse Cell execution.

Canonical result:

```text
status              NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS
parameters          12,154,368
Cells               8
active Cells/token  2
initial val loss    5.7234292984008786
final val loss      0.7885352313518524
initial perplexity  305.9523278270837
final perplexity    2.200171322843134
active fraction     0.25
initial route H     0.6929467022418976
final route H       0.5747594475746155
```

All ten registered engineering gates passed. `scientific_decision=false`: M1 is a real-model training milestone, not yet a continual-learning result.

See [M1_CLOSURE.md](M1_CLOSURE.md).

### Canonical M1 checkpoint

```text
Hugging Face: archelabsxyz/native-clm-v0
file:         final-model.pt
SHA-256:      91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 resolves the current Hub commit but trusts the checkpoint only if this exact SHA-256 matches.

## M2 — Replay-free continual language — 🔵 ACTIVE

Frozen question:

> Starting from the exact trained M1 checkpoint, can protected sparse Cell-local updates learn sequential language distributions without learner-side replay while retaining prior language behavior better than unsafe writes?

M2 introduces only the continual-time variable and a direct causal safety control. It does **not** enable growth or change the 12.15M architecture.

### Registered stream

```text
A  TinyStories validation        evaluation-only M1 retention anchor

B  WikiText-2 raw                continual training phase 1
        ↓
C  CodeParrot codecomplex        continual training phase 2
        ↓
D  Databricks Dolly              continual training phase 3
```

Old training data are never supplied to later phases. Historical domains remain available only to the evaluator:

```text
learner replay bytes = 0
```

At each boundary M2 records the full loss/perplexity matrix over A/B/C/D.

### Why shared substrate and router are frozen

M2 is deliberately a **Cell-local write test**. The M1 shared substrate and learned router are frozen; only `W_i` is writable.

This gives both experimental arms the same learned read-address geometry. Routing inputs occur before Cell execution, so protected and unsafe arms have the same route policy while their Cell writes diverge. That isolates the certificate intervention rather than conflating it with router drift or shared-model forgetting.

Each phase starts a fresh Cell-only AdamW optimizer. Optimizer state is not carried across domain boundaries.

### Protected vs unsafe

Protected:

```text
dW <- dW (I - QᵀQ)
```

Unsafe control:

```text
dW <- dW
```

Everything else is registered identically: M1 parent, data, seed, batch schedule, routing, topology, LR, and phase order.

### Two-GPU strategy

For this 12.15M model, DDP would add synchronization overhead with little benefit. Kaggle's two GPUs are instead used for the causal comparison itself:

```text
GPU0  protected arm
GPU1  unsafe arm
```

The two arms run concurrently for each seed. Formal seeds then execute sequentially:

```text
73211
73212
73213
```

This approximately halves wall-clock time for the registered comparison while avoiding distributed-gradient complexity.

### Formal gates

All gates must pass on all formal seeds:

- exact same canonical M1 checkpoint;
- Cell-only writes;
- shared substrate/router unchanged bit-for-bit;
- zero learner replay;
- fixed 8-Cell topology;
- sparse execution <=30%;
- protected gain >=5% on each new B/C/D phase;
- final protected A regression <=20%;
- unsafe mean forgetting >=3% so interference is actually exposed;
- protected mean forgetting improves over unsafe by >=2 percentage points;
- protected mean plasticity remains >=80% of unsafe plasticity.

Positive decision:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_SUPPORTED
```

Negative decision is preserved as:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

Frozen protocol: [`../../validations/native-clm-v0-m2-continual-language/protocol.json`](../../validations/native-clm-v0-m2-continual-language/protocol.json)

## Kaggle M2 notebook

Canonical notebook:

[`../../notebooks/06-native-clm/native-clm-v0-m2-continual-language-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m2-continual-language-kaggle.ipynb)

Required Kaggle Secrets:

```text
HF_TOKEN
GITHUB_TOKEN
```

The notebook:

1. verifies two GPUs;
2. downloads the exact M1 checkpoint from `archelabsxyz/native-clm-v0` and verifies SHA-256;
3. prepares A/B/C/D local UTF-8 corpora;
4. runs protected and unsafe arms concurrently on GPU0/GPU1 for each frozen formal seed;
5. writes the registered scientific decision whether positive or negative;
6. uploads all six final M2 arm checkpoints to Hugging Face;
7. pushes only lightweight evidence to Git.

## Stop / advance rule

If M2 is supported, advance to **M3 — autonomous Cell growth** at the same model scale. If M2 is not supported, preserve the negative result and diagnose whether the failure is insufficient protected capacity, certificate transfer, or fixed-topology limits before changing the mechanism.

Do not use formal seeds for tuning and do not turn a failed frozen M2 into a post-hoc threshold edit.

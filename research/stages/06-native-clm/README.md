[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE**

Constructive CLM mechanism validation is closed through CLM-005. Stage 06 stops asking whether the registered Cell mechanisms can exist in controlled worlds and starts training a real token-predictive model whose internal computation is routed through persistent Cells.

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
  M0  architecture + execution                           🔵 ACTIVE
  M1  ~12M next-token training                           🔵 IMPLEMENTED / GPU RUN PENDING
  M2  continual language stream                          ⚪ PLANNED
  M3  autonomous Cell growth                             ⚪ PLANNED
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

The sequence is stable. M1 does **not** absorb M2/M3 merely to make the first run more impressive: first establish ordinary next-token trainability, then add continual-learning pressure, then make growth a scientific variable.

## Native CLM v0 definition

The first model is not an external memory layer. Tokens flow through the Cellular Layer before the LM head:

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
```

The learned router chooses only `k` active Cells per token and executes only those Cell operators:

```text
h' = h + Σ gate_i · g_i(h),  i ∈ TopK(router(h))
```

This is already different from a static MoE in two important ways: Cells carry persistent certificate/lineage state and the runtime supports dynamic Cell creation. Autonomous growth policy evaluation is intentionally deferred to M3.

## M0 — Architecture + execution

M0 is an engineering gate, not a scientific decision.

It must prove in one cheap CPU-capable smoke run that the runtime can:

- perform next-token forward/backward;
- deliver gradients to both router and Cell parameters;
- execute sparse top-k Cells rather than all Cells;
- update bounded Cell certificate state;
- project Cell gradients through the certificate nullspace;
- spawn a child Cell and add its parameters to an optimizer;
- keep sparse execution after the Cell set changes;
- save and reload a checkpoint whose Cell count is dynamic;
- generate tokens after reload.

Canonical runner:

```bash
python scripts/research/run_native_clm_v0_m0.py
```

M0 output is lightweight and checkpoint weights are discarded after round-trip verification.

## M1 — First next-token Native CLM

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

Parameter groups deliberately use heterogeneous plasticity:

```text
shared LR   = 2e-4
router LR   = 4e-4
Cell LR     = 8e-4
```

The intent is `Cell > router > shared` plasticity without making M1 depend on a continual-learning claim.

### M1 safety state

Every Cell contains:

```text
W_i                mutable operator
route_key_i        learned read address
Q_i                bounded certificate basis
usage_count_i       runtime state
parent_id_i         lineage state
```

Before an optimizer step, Cell weight gradients are projected into the current certificate nullspace:

```text
dW <- dW (I - QᵀQ)
```

M1 updates the certificate slowly (one representative routed context per Cell every registered interval) so the mechanism is exercised without making historical retention itself the M1 claim.

### Why growth is off in canonical M1

The runtime can spawn Cells in M0, but canonical M1 keeps the Cell count at eight. Otherwise a failed first language run would conflate:

```text
language-model optimization
+
routing
+
continual-learning pressure
+
growth policy
```

M2 introduces the continual stream. M3 then tests autonomous growth under that pressure.

## M1 data

Canonical Kaggle orchestration uses a deterministic local cache made from the public TinyStories dataset and a byte tokenizer:

```text
train documents       50,000
validation documents   2,000
```

The model trainer itself only consumes local UTF-8 text files. Dataset acquisition is kept in a separate script so later corpora do not require an architecture rewrite.

## M1 engineering gates

A canonical M1 run is marked `NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS` only when all of these hold:

- total parameters are between 10M and 15M;
- requested optimizer steps complete with finite loss;
- validation loss improves by at least 5% from initialization;
- only <=30% of available Cell operators execute per token (`2/8 = 25%` in the canonical model);
- router receives nonzero gradient;
- Cells receive nonzero gradient;
- generation executes;
- exactly one Cellular Layer is used;
- Cell count remains fixed in M1, so autonomous growth is not silently claimed.

`scientific_decision=false`: M1 is a real-model training milestone, not yet a formal continual-learning result.

## Checkpoints and repository artifacts

Kaggle keeps binary checkpoints in the runtime output. Git receives only lightweight evidence:

```text
summary.json
metrics.csv
run-config.json
sample.txt
RESULTS.md
data-manifest.json
```

`summary.json` records the final checkpoint SHA-256 and byte size. This preserves model identity without turning the Git repository into a weight registry.

## Kaggle notebook

Canonical notebook:

[`../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb)

Run it top-to-bottom. It:

1. clones the M0/M1 branch;
2. installs LM dependencies;
3. runs M0;
4. prepares the registered TinyStories cache;
5. trains the canonical ~12M M1 model;
6. prints all M1 gates and a generation sample;
7. publishes M0/M1 lightweight results back to the branch with `GITHUB_TOKEN`.

Positive or incomplete M1 runs are both publishable; failed scientific-looking cherry-picking is not allowed.

## Stop / advance rule

If M1 passes, do not scale immediately to 30M. Advance to **M2 — continual language stream** at the same ~12M scale. The first 30M run belongs after continual/growth behavior is understood, as a scaling confirmation rather than a debugging environment.

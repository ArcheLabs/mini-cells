# Native CLM v0 M3L-2 — Online Historical Address-State Integration

Status: **FROZEN / UNRUN**

M3L-2 is the next formal continual-language experiment after the M3R → Address Diagnostic → M3L → M3L-1 mechanism-selection chain.

## Registered question

Starting from the exact canonical M1 checkpoint, can a persistent rank-32 historical query address state and a lineage-local affine parent/child gate turn the checkpoint-level addressability result into actual replay-free continual-language retention beyond the matched M3R lineage-cosine algorithm?

This is a new formal experiment. It does not reinterpret any M3, M3R, M3L, or M3L-1 result.

## Parent evidence

```text
M3R       NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
Address   QUERY_GEOMETRY_SEPARABLE
M3L       QUERY_SKETCH_GATE_NOT_FEASIBLE
M3L-1     LOW_RANK_CAPACITY_SUFFICIENT
           minimum passing low-rank Gaussian state = 32
```

M3L-1 publish commit:

```text
5ace6faf344b1b805752a33ffb861aeaf34dad6e
```

## Causal arms

```text
GPU0 / control
  exact M3R immutable-root lineage routing
  parent vs child = frozen cosine-key comparison

GPU1 / treatment
  same M1 checkpoint
  same protected Cell writes
  same growth pressure controller
  same immutable root router
  + persistent rank-32 historical query state
  + lineage-local affine parent/child gate
```

The scientific variable is the lineage-local address mechanism. M3 growth thresholds and the B → C → D schedule remain unchanged.

## Address state

For each active lineage leaf, M3L-2 maintains a bounded Gaussian second-order state over normalized frozen-router queries:

\[
A_i=(n_i,\mu_i,U_i,\lambda_i,\sigma^2_{i,\mathrm{res}}),\qquad \operatorname{rank}(U_i)\le 32.
\]

Registered persistent budget:

```text
rank                         32
router query width           384
maximum bytes / Cell         52,360
regularization               1e-4
target historical FPR        0.10
```

Raw historical token/query replay is not retained.

Current-stream sufficient statistics are ephemeral float32 first/second moments. At most 256 deterministically ordered routed queries per leaf per training batch enter the accumulator. At each 50-step growth check, an unsplit leaf merges the current sufficient statistics into its historical sketch and truncates back to rank 32.

## Function-preserving mitosis

If the frozen M3 pressure rule selects a leaf parent:

1. the pre-window parent historical sketch is frozen;
2. the current-window sketch becomes the child's initial address state;
3. the child operator is an exact clone of the parent operator;
4. an affine Gaussian-LDA edge gate is derived from parent-history vs current-window moments;
5. the root-level Top-K set and root probability mass are unchanged.

The local decision is:

\[
w^Tq+b>\tau.
\]

Because parent and child operators are identical at birth, installing the gate must not materially change logits at the birth probe.

## Explicit bootstrap boundary

The canonical M1 checkpoint predates query-address sidecars. M3L-2 therefore registers a one-time **pre-continual bootstrap**, rather than pretending that this state existed in M1.

```text
TinyStories train @ f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
10,000 documents
160 batches
sampling seed 74001
        ↓
construct rank-32 sidecars for the original 8 roots
        ↓
release the one-shot bootstrap handle
        ↓
B → C → D begins
```

Bootstrap constraints:

- separate TinyStories `train` split from the A retention `validation` split;
- zero optimizer steps;
- zero Native CLM parameter updates;
- zero certificate updates;
- zero growth;
- raw bootstrap queries are discarded;
- bootstrap data is inaccessible to the learner after continual phase B starts.

The claim is therefore **zero learner replay after continual start**, not bootstrap-free M1 state.

## Formal seeds

```text
development  74101 / 74102 / 74103
formal       74211 / 74212 / 74213
```

The formal seeds remain untouched until the canonical Kaggle formal runner is deliberately executed.

Consumed M2/M3/M3R formal seeds are explicitly forbidden.

## Main formal gates

A positive decision requires every registered gate on every formal seed, including:

```text
control A regression              >= 30%
treatment A regression            <= 20%
A retention advantage             >= 10 percentage points
treatment mean forgetting         <= 15%
B/C/D phase gain                  >= 5% each
treatment/control plasticity      >= 0.80
active fraction vs dense          <= 0.30
spawned children                  1..8
child reuse fraction              >= 0.75
address-state rank                <= 32
address bytes / Cell              <= 52,360
```

It also requires zero replay, exact M1 identity, matched seed/data, immutable root read function, function-preserving birth, address-state checkpoint round-trip, and one affine gate per spawned child.

## Canonical execution surface

```text
research/notebooks/06-native-clm/
  native-clm-v0-m3l2-online-address-state-kaggle.ipynb

scripts/research/
  prepare_native_clm_v0_m3l2_data.py
  run_native_clm_v0_m3l2.py
  publish_native_clm_v0_m3l2.py
```

Publication is HF-first: all six final checkpoints plus the formal decision must be uploaded to `archelabsxyz/native-clm-v0` before lightweight Git evidence is pushed.

## Decision boundary

M4 remains blocked until M3L-2 has a valid formal decision. No 30M scale-up is licensed by the checkpoint-only M3L-1 result.

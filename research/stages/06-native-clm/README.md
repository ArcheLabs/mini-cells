[English] | [中文](README.zh-CN.md)

# Stage 06 — Native CLM

Status: **ACTIVE — M2 SAFE-FUNCTIONAL-WRITE REOPENED**

Stage 06 has one scientific objective: replay-free continual learning in a real token-predictive model. M1 established trainability. M2 was the first true continual-learning milestone and remains **NOT SUPPORTED**.

M3, M3R, the Address Diagnostic, M3L, M3L-1, M3L-2 and M3W-0 are therefore retained as **M2 failure-decomposition evidence**, not as milestones that progressed beyond M2.

Canonical reopened roadmap:

- [M2_REOPENED_ROADMAP.md](M2_REOPENED_ROADMAP.md)
- [M2_REOPENED_ROADMAP.zh-CN.md](M2_REOPENED_ROADMAP.zh-CN.md)

## Current roadmap

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
        ↓
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
        ↓ reopened failure analysis
  M2-R0 actual optimizer-update invariant audit          🔵 FROZEN / UNRUN
        ↓
  M2-R1 functional certificate reconstruction            ⚪ BLOCKED ON R0
        ↓
  M2-R2 fixed-topology replay-free continual language    ⚪ BLOCKED ON R0/R1
        ↓ only after formal PASS
  growth / mitosis reopened                              ⚪ BLOCKED

Historical M2 failure decomposition:
  M3   global-pool growth                                🔴 NOT SUPPORTED
  M3R  lineage-isolated read routing                     🔴 NOT SUPPORTED
  Address Diagnostic                                     🟢 QUERY GEOMETRY SEPARABLE
  M3L  rank-16 query-sketch gate                         🔴 NOT FEASIBLE
  M3L-1 address-state capacity                           🟢 LOW_RANK_CAPACITY_SUFFICIENT (rank 32)
  M3L-2 online address integration                       🔴 NOT SUPPORTED / partial retention benefit
  M3W-0 write-drift restoration                          🟡 ROOT_WRITE_DOMINANT_TRANSFER_GAP

  M4   Cell ontology / specialization                    ⚪ BLOCKED
  30M scale-up                                            ⚪ BLOCKED
```

Hard rule: **no new Native growth milestone, M4 work or 30M scale-up before M2-R2 formally passes.**

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

## M0 / M1 — 🟢

M0 established sparse routing, Cell-local gradients, certificate projection, dynamic spawn, optimizer enrollment, checkpoint round-trip and generation.

M1 established real next-token trainability:

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

See [M1_CLOSURE.md](M1_CLOSURE.md).

## M2 — 🔴 NOT SUPPORTED / the active unfinished milestone

Formal decision:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
formal seeds = 73211 / 73212 / 73213  (consumed)
```

Observed:

```text
protected mean forgetting     ~0.211502
unsafe mean forgetting        ~0.278987
protected A regression        ~0.438682
registered A regression       <=0.20
protected/unsafe plasticity    ~0.964
```

Certificate projection was causally useful, but it did not close the end-to-end retention gap.

The original fixed M2 boundary remains the baseline for re-establishing the continual-write primitive:

```text
exact M1 start
8 Cells fixed
active Cells = 2
shared Transformer frozen
router / route keys frozen
growth disabled
B -> C -> D
learner raw replay = 0
```

## Why the M3 series is now failure-decomposition evidence

Later work exposed real but secondary mechanisms:

- M3: global children disrupted old-domain read ownership;
- M3R: root ownership was preserved but lineage-local cosine addressing remained poor;
- Address Diagnostic: the frozen query representation actually contained a separable boundary;
- M3L-1: rank-32 historical address state was sufficient to represent that boundary;
- M3L-2: online address state improved A regression by ~5 pp versus cosine control while preserving ~99% plasticity, but A regression remained ~42%;
- M3W-0: checkpoint-only 2×2 restoration attributed ~94.4–95.2% of residual A operator damage to final original-root operator drift.

M3W-0 registered:

```text
ROOT_WRITE_DOMINANT_TRANSFER_GAP
publish commit = 7bd8d554fea89fc44ef16287453c97f90a4e3f06
scientific_decision = false
```

Simply freezing roots is not the solution: restoring roots to M1 retained only ~49–50% of B gain, ~67–68% of C gain and ~33% of D gain. The roots contain both useful plasticity and destructive interference.

The active question is therefore:

\[
\boxed{\text{Which realized parameter transactions can be safely committed to an old Cell?}}
\]

## M2-R0 — 🔵 Protected Update Invariant Audit

Before changing certificate semantics, M2-R0 audits the mathematical invariant the original M2 implementation intended to enforce.

The historical protected path projects the raw gradient:

\[
G_p=G(I-Q^TQ),
\]

then lets AdamW apply elementwise moment preconditioning and decoupled weight decay. M2-R0 directly measures the post-optimizer parameter delta for each Cell/step:

\[
\rho=\frac{\|\Delta WQ^T\|_F}{\|\Delta W\|_F+10^{-12}}.
\]

Frozen matched arms:

```text
1. canonical AdamW + gradient projection + wd=.01
2. AdamW + gradient projection + wd=0
3. SGD + gradient projection + wd=0            (algebraic reference)
4. AdamW + gradient projection + final-delta projection
```

The fourth arm lets AdamW form its complete proposal and commits only:

\[
U=U_{raw}(I-Q^TQ).
\]

M2-R0 uses the exact M1 checkpoint and only the pinned WikiText B-train stream as a real gradient source. It runs 64 steps/arm, keeps certificates frozen, performs no growth, reads no old-A replay, consumes no formal seed and writes no model checkpoint. `scientific_decision=false`.

Canonical protocol:

```text
research/validations/native-clm-v0-m2r0-update-invariant-audit/protocol.json
```

Canonical Kaggle notebook:

```text
research/notebooks/06-native-clm/native-clm-v0-m2r0-update-invariant-audit-kaggle.ipynb
```

## M2-R1 — ⚪ Functional Certificate Reconstruction

After R0 closes optimizer-level ambiguity, R1 studies certificate semantics in the fixed final-M1 representation rather than immediately launching another continual formal run.

Roadmap candidates:

```text
A. current top-1 mean basis                 historical baseline
B. all-active probability-weighted SVD      activation coverage
C. importance-weighted activation subspace  soft/scaled protection
D. Jacobian/Fisher functional sketch         old-function protection
```

For a Cell:

\[
\delta z\approx J_i(x)p_i\Delta W_i h,
\]

so the desired quantity is functional damage rather than activation similarity:

\[
D_i(\Delta W)=\mathbb E_{old}\|J_i p_i\Delta W_i h\|^2.
\]

Persistent approximations should prioritize low-rank/Kronecker forms such as:

\[
F_i\approx A_i\otimes B_i,
\]

\[
A_i=\mathbb E[p_i^2hh^T],\qquad B_i=\mathbb E[J_i^TJ_i].
\]

R1 must validate certificate fidelity against held-out old-function drift and report explicit storage/rank/capacity. Raw old replay is not allowed as the final mechanism.

## M2-R2 — ⚪ the next true continual-learning formal

Only after R0/R1 mechanism validation will a new fixed-8-Cell B→C→D formal experiment run with untouched new formal seeds.

Only independently validated changes may enter:

1. realized-update constrained optimizer;
2. selected functional certificate.

Minimum end-to-end gates remain:

```text
A absolute regression <= 20%
mean forgetting <= 15%
B/C/D phase gain >= registered plasticity floor
plasticity >= 80% matched control
shared/router frozen
zero learner replay
fixed 8-Cell topology
```

Only a formal M2-R2 pass supports:

\[
\boxed{\text{Native CLM basic replay-free continual-write primitive supported.}}
\]

## When growth reopens

Future growth is derived from safe-write infeasibility rather than `loss high + rank pressure + cooldown`.

Define a Cell safe-update set:

\[
\mathcal S_i(\epsilon)=\{\Delta W:D_i(\Delta W)\le\epsilon\}.
\]

Reuse an existing Cell when sufficient new-domain gain is achievable inside its safe set. Reopen capacity allocation / mitosis only when no eligible existing Cell can absorb the write safely enough.

## Evidence boundary

- M2/M3/M3R/M3L-2 formal seeds are consumed and cannot be reused as untouched evidence;
- Address Diagnostic, M3L, M3L-1 and M3W-0 are diagnostics and cannot change historical formal decisions;
- M2-R0/R1 are also diagnostics;
- M2-R2 must use untouched new formal seeds;
- M4, growth milestones and 30M scale-up remain blocked until R2 passes.

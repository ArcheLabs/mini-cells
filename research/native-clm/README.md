# Native CLM Architecture Lab

`research/native-clm/` is the long-lived architecture laboratory for **native cellular language models**. The immediate goal is not to train one CLM for as long as possible. It is to use a small, fixed budget to discover better CLM architecture choices before scaling them.

The canonical working surface remains one living notebook:

- `native_clm_lab.ipynb`

Execution code lives beside it:

- `native_clm_runtime.py` — frozen C0/T1 data, model, training and evaluation primitives;
- `run_native_clm.py` — historical C0/T1 baseline runner;
- `native_clm_candidates.py` — Phase-A single-factor C1–C4 candidates;
- `run_native_clm_search.py` — resumable candidate sweep, dual-GPU scheduling and leaderboard generation.

## Research question

Primary question:

> Under a fixed ~2M-parameter / 10M-token search budget, which Native CLM architectural choices improve next-token prediction relative to the frozen C0 anchor?

The search distinguishes quality/parameter, quality/training-compute and quality/inference-compute. Equal parameters do not imply equal FLOPs, so compute, throughput and VRAM are reported separately rather than collapsed into one score.

## Frozen anchors

The first real accelerator development run is immutable evidence under:

```text
research/native-clm/results/dev/baseline-seed-91001/
```

At 10M TinyStories training tokens:

- **C0 historical scaled CLM**: 1,899,576 params, PPL `16.8387103`;
- **T1 modern Transformer**: 1,894,080 params, PPL `12.1907091`;
- C0/T1 PPL ratio: `1.3812741`;
- estimated C0/T1 training-FLOP ratio: `3.3046`.

These measurements are anchors. Phase A does **not** retrain T1 and does not spend additional development or confirmation seeds merely to reproduce the known 10M gap.

## Frozen data and optimization protocol

All Phase-A candidates use the same protocol as the anchor:

- dataset: `roneneldan/TinyStories`;
- immutable revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`;
- deterministic ByteLevel BPE trained on the first 20,000 pinned training stories;
- vocabulary 2,048;
- context 128;
- train sequence length 125;
- batch 8 = exactly 1,000 consumed tokens/optimizer step;
- AdamW, betas `(0.9, 0.95)`, weight decay `0.1`;
- LR `3e-4`, warmup then cosine decay;
- gradient clipping `1.0`;
- FP16 autocast + GradScaler on CUDA;
- identical deterministic training-batch schedule for the same seed;
- identical fixed validation examples for the same seed.

Architecture-search decision budget:

```text
profile = baseline
seed    = 91001
tokens  = 10,000,000 / candidate
params  = within 1% of frozen T1
```

Untouched confirmation seeds `91101–91103` are not consumed during Phase A. Development seeds `91002–91003` are also held back until a candidate first demonstrates a causal single-factor improvement on `91001`.

## Phase A: orthogonal architecture search

Phase A changes one factor at a time.

### C1 — residual gated update

Question: is C0's GRU state replacement harming recurrent optimization?

C1 replaces the GRU update with an identity-preserving write:

```text
proposal = attention_delta + ffn_delta
gate     = sigmoid(W[state, proposal] + b)
state'   = state + gate * proposal
```

The gate starts with bias `-2`, roughly mirroring C0's `+2` carry bias while preserving a direct identity path. Width is increased only enough to keep the total parameter count matched.

- dim 188, heads 4, FFN 752;
- params: **1,901,808** (`1.00408×` T1).

### C2 — shared-wide phase Cell

Question: should cross-phase parameter sharing be exchanged for a wider recurrent Cell?

C2 uses one Cell for all 12 recurrent steps instead of three independent stage parameter sets. It retains the historical `8×4 → 32×4 → 128×4` communication pattern and GRU update, adds learned step embeddings plus zero-initialized three-phase affine modulation, and spends the saved parameters on width.

- dim 270, heads 6, FFN 1080;
- params: **1,909,170** (`1.00797×` T1).

C2 has materially higher FLOPs; it is explicitly a quality/parameter experiment, not a compute win claim.

### C3 — progressive receptive-field schedule

Question: is the fixed `8×4 → 32×4 → 128×4` schedule optimal?

C3 retains C0 dimensions, GRU update, stage parameterization and total parameter count. Only the per-step windows change:

```text
stage 1: 8, 16, 32, 64
stage 2: 16, 32, 64, 128
stage 3: 32, 64, 128, 128
```

- params: **1,899,576** (identical to C0).

### C4 — sparse communication

Question: must every recurrent compute step perform attention communication?

C4 keeps C0's parameters and GRU/FFN update, but performs attention only on recurrent steps 1 and 4 inside each four-step stage. Intermediate steps are computation-only.

- params: **1,899,576** (identical to C0);
- estimated forward cost is lower than C0 because six of twelve attention calls are removed.

## Evaluation

Every candidate is evaluated at the same checkpoints:

```text
1M, 2.5M, 5M, 7.5M, 10M tokens
```

Two quality statistics are primary:

1. final validation PPL at 10M;
2. validation-NLL area under the curve over **log training tokens**.

The runner records both raw AUC and the log-width-normalized mean NLL. The latter captures learning efficiency across the fixed budget rather than only the last checkpoint.

Also reported independently:

- exact parameter count and T1 ratio;
- estimated training FLOPs;
- estimated inference FLOPs/token;
- measured tokens/s;
- elapsed time;
- peak VRAM.

Do not combine these into an arbitrary composite score.

## Dual-GPU search

`run_native_clm_search.py sweep` treats GPUs as independent workers. On a two-T4 Kaggle machine:

```text
round 1: GPU0=C1, GPU1=C2
round 2: GPU0=C3, GPU1=C4
```

T1 is never retrained. C0/T1 metrics and curves are loaded from the immutable `baseline-seed-91001` evidence and inserted into the generated architecture leaderboard.

The runner supports resume. Large token caches, optimizer states and checkpoints remain under `/kaggle/working/native-clm`; only small evidence is copied back into Git.

## Decision rule

Phase A is deliberately causal:

```text
C0/T1 frozen anchors
    ↓
C1, C2, C3, C4 single-factor runs on seed 91001
    ↓
identify real wins in 10M PPL and/or log-token NLL AUC
    ↓
only then combine winning mechanisms
    ↓
only promising combined candidates receive additional development seeds
    ↓
confirmation seeds remain untouched until interpretation is frozen
```

A candidate that merely changes compute without improving quality is still informative, but is not promoted as a better CLM architecture. A candidate that improves PPL while greatly increasing compute is a quality/parameter result and must be labelled as such.

## Evidence policy

The notebook is mutable; evidence is additive.

Frozen baseline development evidence:

```text
research/native-clm/results/dev/baseline-seed-91001/
```

Phase-A evidence:

```text
research/native-clm/results/dev/architecture-search-baseline-seed-91001/
```

Promoted results later use immutable `NCLM-XXX/` directories. Existing run records are never overwritten to fit a later interpretation.

This phase still excludes continual learning, growth/mitosis, replay-free writes, rollback, MoE conversion and JAM execution/consensus concerns. Those mechanisms return only after the static Native CLM architecture is competitive.
# Native CLM Architecture Lab

`research/native-clm/` is the long-lived architecture laboratory for native cellular language models. The immediate objective is architecture discovery under a fixed small budget, not long-run convergence of one model.

The canonical working surface remains one living notebook:

- `native_clm_lab.ipynb`

Execution code:

- `native_clm_runtime.py` — frozen data/training/evaluation primitives and C0/T1 anchors;
- `native_clm_candidates.py` / `run_native_clm_search.py` — completed Phase-A search;
- `native_clm_phase_b.py` / `run_native_clm_phase_b.py` — completed Phase-B modernization/native split;
- `native_clm_phase_c.py` / `run_native_clm_phase_c.py` — Phase-C winner cross-over;
- `native_clm_visualize.py` — dependency-free SVG evidence visualizations.

## Frozen search protocol

Architecture search uses:

- TinyStories at immutable revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`;
- deterministic ByteLevel BPE, vocab 2,048;
- context 128, train sequence 125, batch 8 = exactly 1,000 consumed tokens/step;
- AdamW `(0.9, 0.95)`, weight decay `0.1`, base LR `3e-4`, warmup + cosine;
- gradient clipping `1.0`, CUDA FP16 + GradScaler;
- development seed `91001` for architecture discovery;
- 10M train tokens/candidate;
- approximately 1.9M parameters, within 1% of T1;
- fixed validation samples and deterministic batch schedule for a given seed.

Held-back development seeds `91002–91003` and confirmation seeds `91101–91103` remain untouched until the Phase-C interpretation is frozen.

## Frozen anchors

At 10M tokens:

| model | role | PPL | params |
|---|---|---:|---:|
| T1 | modern Transformer anchor | 12.1907 | 1,894,080 |
| C0 | historical CLM anchor | 16.8387 | 1,899,576 |
| C1 | residual gated CLM | 14.8185 | 1,901,808 |

T1 is already modernized with RMSNorm, RoPE, SwiGLU and GQA. C0 retained the historical LayerNorm + GELU + learned-position + GRU design.

## Phase-A result

Phase A isolated four CLM factors. C1 was the only clear quality win. The main design lesson was that an identity-preserving residual state path is materially better than C0's GRU state replacement at this scale/budget.

Phase-A evidence:

```text
research/native-clm/results/dev/architecture-search-baseline-seed-91001/
```

## Phase-B result: modernization vs Native CLM structure

Phase B deliberately split the remaining gap into two axes.

### M-series — transferable modern LM technology

| model | change on C1 | PPL @ 10M | result |
|---|---|---:|---|
| M1 | RMSNorm | 14.8272 | neutral/slightly worse |
| M2 | SwiGLU | 13.4815 | strong gain |
| M3 | RoPE | 13.0448 | stronger gain |
| M4 | RMSNorm + SwiGLU + RoPE | 12.0274 | development-seed quality parity/slight win vs T1 |

The evidence does **not** support the claim that every modern Transformer component automatically helps CLM. The transferable gains are concentrated in SwiGLU and RoPE; RMSNorm is nearly neutral in the current design.

### N-series — CLM-native structure

| model | change on C1 | PPL @ 10M | result |
|---|---|---:|---|
| N1 | fixed residual-only update | 14.3396 | best native structural gain |
| N2 | low-rank dual-gate ARU | 14.5926 | small gain, weaker than N1 |
| N3 | step-conditioned write gate | 14.7968 | nearly neutral |
| N4 | soft adaptive communication | 14.9432 | regression |

N1 changes the interpretation of C1. The useful mechanism is not the learned write gate itself. Current evidence favors:

```text
simple identity-preserving residual refinement
    > learned write gate
    > more complex recurrent gating
```

N1 also improves throughput and memory relative to C1 because it removes the dense learned write gate.

Phase-B evidence:

```text
research/native-clm/results/dev/phase-b-baseline-seed-91001/
```

## Phase C: winner cross-over

Phase C asks one narrow question:

> Do the independently evidenced Native-CLM winner (N1 residual refinement) and modernization winners (SwiGLU / RoPE) compose?

No new speculative mechanism is introduced.

### X1 — N1 + SwiGLU

Tests transfer of the M2 gain onto the simpler residual-only state transition.

```text
N1 + SwiGLU + learned absolute positions + LayerNorm
```

SwiGLU hidden width is `624`, chosen to keep total parameters matched to T1.

### X2 — N1 + RoPE

Tests transfer of the M3 gain onto N1.

```text
N1 + GELU + RoPE + LayerNorm
```

GELU FFN width is `954`, spending the removed learned-position/gate capacity on active FFN parameters.

### X3 — N1 + SwiGLU + RoPE

Primary Phase-C candidate:

```text
fixed residual refinement + SwiGLU + RoPE + LayerNorm
```

This is the clean combination of the strongest independently evidenced native and modernization mechanisms.

### X4 — X3 + RMSNorm

RMSNorm was neutral as a single factor on C1, but M4 slightly beat T1. X4 therefore re-tests it only in the combined state:

```text
fixed residual refinement + SwiGLU + RoPE + RMSNorm
```

X4 is not used to retroactively claim RMSNorm was useful in Phase B; it tests interaction only.

### Parameter matching

Expected trainable parameters:

| model | params |
|---|---:|
| X1 | 1,893,912 |
| X2 | 1,893,578 |
| X3 | 1,893,536 |
| X4 | 1,893,912 |
| T1 | 1,894,080 |

All four are well inside the ±1% protocol boundary.

## Phase-C visualization evidence

The Phase-C runner produces three dependency-free SVGs as first-class evidence:

```text
phase-c-final-ppl.svg
phase-c-learning-curves.svg
phase-c-quality-compute.svg
```

They are generated from the same committed JSON/CSV measurements, not manually edited figures.

1. **Final PPL** — zero-based 10M PPL bars for T1/C0/C1/M4/N1/X-series.
2. **Learning curves** — validation NLL at the frozen checkpoints on a logarithmic token x-axis.
3. **Quality / compute** — final PPL versus estimated training FLOPs normalized to T1; lower-left is better.

A companion `phase-c-visualizations.json` records the visualization contract.

## Evaluation and decision rule

Every Phase-C candidate is evaluated at:

```text
1M, 2.5M, 5M, 7.5M, 10M tokens
```

Primary quality signals:

1. validation PPL at 10M;
2. validation NLL AUC over log training tokens;
3. normalized mean NLL over that interval.

Reported separately:

- exact parameter count;
- ratio to T1 parameters;
- PPL ratio to T1, M4 and N1;
- estimated training/inference FLOPs;
- measured tokens/s;
- peak VRAM.

No arbitrary composite score is used.

With two GPUs the default Phase-C sweep is:

```text
round 1: X1 / X2
round 2: X3 / X4
```

C0, T1, C1, M4 and N1 are frozen anchors and are not retrained.

Phase C remains a development-seed experiment. Even if X3/X4 beat T1 at seed `91001`, that is only a promotion signal. The next step is held-back development seeds `91002–91003`; only after the interpretation is frozen should confirmation seeds `91101–91103` be consumed.

## Evidence policy

The notebook is mutable; evidence is additive and existing records are never rewritten.

```text
research/native-clm/results/dev/baseline-seed-91001/
research/native-clm/results/dev/architecture-search-baseline-seed-91001/
research/native-clm/results/dev/phase-b-baseline-seed-91001/
research/native-clm/results/dev/phase-c-baseline-seed-91001/
```

Promoted runs later use immutable `NCLM-XXX/` directories.

This architecture program still excludes continual learning, growth/mitosis, replay-free writes, rollback, MoE conversion and JAM execution/consensus constraints. Those mechanisms return only after the static Native CLM architecture is stable across held-back seeds.

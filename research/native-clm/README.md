# Native CLM Architecture Lab

`research/native-clm/` is the long-lived architecture laboratory for native cellular language models. The immediate objective is architecture discovery under a fixed small budget, not long-run convergence of one model.

The canonical working surface remains one living notebook:

- `native_clm_lab.ipynb`

Execution code:

- `native_clm_runtime.py` — frozen data/training/evaluation primitives and C0/T1 anchors;
- `run_native_clm.py` — historical C0/T1 baseline runner;
- `native_clm_candidates.py` / `run_native_clm_search.py` — completed Phase-A C1–C4 search;
- `native_clm_phase_b.py` / `run_native_clm_phase_b.py` — Phase-B modernization/native two-track search.

## Frozen protocol

Unless a run is explicitly marked smoke/debug, architecture search uses:

- TinyStories `roneneldan/TinyStories` at immutable revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`;
- deterministic ByteLevel BPE, vocab 2,048;
- context 128, train sequence 125, batch 8 = exactly 1,000 consumed tokens/step;
- AdamW `(0.9, 0.95)`, weight decay `0.1`, base LR `3e-4`, warmup + cosine;
- gradient clipping `1.0`, CUDA FP16 + GradScaler;
- development seed `91001` for architecture discovery;
- 10M train tokens/candidate;
- approximately 1.9M parameters, within 1% of T1;
- fixed validation samples and the same deterministic training-batch schedule for a given seed.

Held-back development seeds `91002–91003` and confirmation seeds `91101–91103` are not consumed until an interpretation and candidate are worth confirming.

## Frozen anchors and Phase-A result

Immutable evidence:

```text
research/native-clm/results/dev/baseline-seed-91001/
research/native-clm/results/dev/architecture-search-baseline-seed-91001/
```

At 10M tokens:

| model | role | PPL | interpretation |
|---|---|---:|---|
| T1 | modern Transformer anchor | 12.1907 | RMSNorm + RoPE + SwiGLU + GQA |
| C0 | historical CLM anchor | 16.8387 | LayerNorm + GELU + learned positions + GRU update |
| C1 | residual gated CLM | 14.8185 | clear Phase-A quality win |
| C2 | shared-wide phase Cell | 15.1677 | quality gain but very high compute |
| C3 | progressive receptive field | 17.8528 | rejected |
| C4 | sparse communication | 17.6817 | efficiency signal, quality regression |

C1 is the parent for Phase B. It improved both final PPL and log-token NLL AUC, so the result is not a last-checkpoint accident. The principal Phase-A design lesson is that an identity-preserving write path is better than C0's GRU state replacement at this scale/budget.

## Why Phase B has two tracks

T1 is already a modernized decoder, while C1 still uses older generic LM components. We therefore separate two questions:

1. **M-series — transferable modern LM technology:** how much of the remaining gap can be closed by components that are not CLM-specific?
2. **N-series — Native CLM structure:** what improvements come from recurrence, state update and communication mechanisms that do not exist in an ordinary feed-forward decoder?

Conceptually:

```text
Native CLM quality = generic modern LM technology + CLM-native mechanisms
```

Do not conflate an M-series gain with evidence for a novel cellular mechanism, and do not deny a useful generic modernization merely because it also benefits Transformers.

## M-series: modernization controls

All M-series candidates keep C1's three independent recurrent stages, local windows `(8, 32, 128)`, four updates/stage and residual write gate. Only the stated generic LM component changes.

### M1 — C1 + RMSNorm

Single-factor normalization control:

```text
LayerNorm -> RMSNorm
```

Everything else remains C1.

### M2 — C1 + SwiGLU

Single-factor activation/FFN control:

```text
GELU MLP -> SwiGLU
```

SwiGLU hidden width is parameter-matched rather than copied from T1 blindly.

### M3 — C1 + RoPE

Single-factor position control:

```text
learned absolute position embedding -> RoPE on local q/k
```

The learned position table is removed. One unmatched odd head dimension is left unrotated by the existing RoPE primitive; attention projections and local causal windows are unchanged.

### M4 — modernized C1

Combination test after the three single-factor controls:

```text
C1 + RMSNorm + SwiGLU + RoPE
```

M4 answers how competitive C1 becomes after receiving the same class of mature generic LM components. It is not used to infer which component caused a gain; M1–M3 provide that attribution.

GQA is intentionally not included in this first modernization group. At this scale it is primarily an attention/KV efficiency intervention and changes the local recurrent attention parameterization. It can be tested separately after the quality effects above are understood.

## N-series: Native CLM structural optimization

All N-series candidates retain C1's older generic LM components so that changes are attributable to CLM-native mechanisms rather than modernization.

### N1 — residual-only gate ablation

Tests whether C1 wins because of the identity residual path alone or because the learned gate is important:

```text
state' = state + alpha * proposal
alpha = sigmoid(-2)
```

The fixed alpha matches C1's initial write magnitude. Parameters released by removing the gate are spent on active FFN capacity, not inert padding parameters.

### N2 — dual-gate ARU

Separates retention and writing:

```text
retain, write = gates(state, proposal)
state' = retain * state + write * proposal
```

A low-rank two-output gate keeps the total parameter budget matched. Initialization strongly preserves identity (`retain bias +6`) while keeping writes conservative (`write bias -2`).

### N3 — recurrent-step-conditioned write

Keeps C1's gate but adds a learned per-recurrent-step write bias:

```text
gate_logits' = gate_logits + step_gate_bias[step]
```

The new bias starts at zero, so the initial function is C1. This directly tests whether different recurrent phases should learn different write intensity.

### N4 — adaptive communication

Adds a state-conditioned soft communication gate to attention output before the C1 update:

```text
strength = sigmoid(g(state))
attention_delta' = strength * attention_delta
```

The gate starts near one. This experiment tests representational value of adaptive communication. It does **not** skip the attention kernel, so it must not be reported as a realized FLOP saving. A later hard-routing experiment would be required for that claim.

## Evaluation

Every Phase-B candidate is evaluated at:

```text
1M, 2.5M, 5M, 7.5M, 10M tokens
```

Primary quality signals:

1. validation PPL at 10M;
2. validation NLL AUC over log training tokens;
3. normalized mean NLL over that log-token interval.

Also reported separately:

- exact parameter count and ratio to T1;
- PPL ratio to C1 and T1;
- estimated training/inference FLOPs;
- measured tokens/s;
- peak VRAM.

No arbitrary composite score is used.

## Dual-GPU Phase-B execution

`run_native_clm_phase_b.py sweep` treats each GPU as an independent complete-model worker. With two GPUs and the default eight candidates:

```text
round 1: M1 / M2
round 2: M3 / M4
round 3: N1 / N2
round 4: N3 / N4
```

The runner supports `--track modernization` and `--track native` for separate group execution, or `--track all` for the full experiment. C0, T1 and C1 are never retrained; their immutable evidence is inserted into the Phase-B leaderboard.

## Decision rule

Phase B remains a development-seed architecture search:

```text
C0/T1 frozen anchors
       ↓
C1 frozen Phase-A parent
       ↓
M1 M2 M3          N1 N2 N3 N4
       \            /
        M4 combination
             ↓
attribute generic-modernization gains separately from native-CLM gains
             ↓
only then combine proven mechanisms
             ↓
held-back development seeds -> frozen interpretation -> confirmation seeds
```

A candidate is a quality improvement only if its PPL and learning-curve evidence support the claim. Compute/throughput/VRAM changes remain separate dimensions.

## Evidence policy

The notebook is mutable; evidence is additive and existing records are never rewritten.

```text
research/native-clm/results/dev/baseline-seed-91001/
research/native-clm/results/dev/architecture-search-baseline-seed-91001/
research/native-clm/results/dev/phase-b-baseline-seed-91001/
```

Promoted results later use immutable `NCLM-XXX/` directories.

This architecture program still excludes continual learning, growth/mitosis, replay-free writes, rollback, MoE conversion and JAM execution/consensus constraints. Those mechanisms return only after the static Native CLM architecture is competitive.

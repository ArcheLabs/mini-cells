# Experiment 026 — 30M Cell Granularity Differentiation

## Question

Does the current CLM fail to develop clear cellular specialization partly because one program cell is too coarse?

Experiment 026 isolates **cell granularity** from model capacity and growth. The same retained 30M TextNCA source is function-preservingly upcycled into the same fixed 12-root CLM. Each existing FFN expert is then interpreted as a tissue and exactly partitioned along its hidden dimension into:

- G=1 micro-cell per tissue;
- G=2;
- G=4;
- G=8.

All four arms have the same initial parameter count, the same tissue-level router, the same source experience, and—up to floating-point summation order—the same age-zero function. Persistent birth is disabled.

The experiment therefore asks a narrow causal question:

> Holding capacity, routing architecture, initialization, optimizer, data and training budget fixed, does finer cell granularity produce stronger and more stable functional differentiation?

## Why Experiment 025 is not reused directly

Experiment 025 used a 90% arithmetic / 10% TinyStories developmental shift. That environment was intentionally strong but structurally simple: the organism could benefit from a largely global adaptation toward arithmetic. It is therefore a weak environment for testing whether smaller cells spontaneously form distinct phenotypes.

Experiment 026 uses a persistent four-domain environment instead:

- **Story** — TinyStories;
- **Math** — the existing six-family synthetic integer arithmetic generator;
- **Symbolic** — deterministic sequence reversal, sorting and elementwise-offset transformations;
- **Facts** — deterministic synthetic key/value question-answer examples.

The latter three are controlled synthetic environments. They are not claimed as broad mathematical reasoning, symbolic reasoning, or factual-knowledge benchmarks.

## Fixed developmental environment

Each arm receives 20M continuation tokens after the common 100M Story source experience.

The four domains are exactly balanced over every four training steps. Within each four-step block, the order is deterministically shuffled from a frozen seed. Therefore every arm sees the same domain sequence and the same sampled examples.

Evaluation checkpoints:

- 0;
- 1M;
- 2M;
- 5M;
- 10M;
- 15M;
- 20M continuation tokens.

## What is held fixed

Across G={1,2,4,8}:

1. source checkpoint;
2. ProgressiveGrowthCLM 12-root upcycle;
3. tissue-level hierarchical router;
4. total initial parameter count;
5. age-zero model function;
6. 20M deterministic four-domain schedule;
7. AdamW optimizer and LR schedule;
8. batch size and sequence length;
9. next-token cross-entropy objective;
10. no persistent growth and no juvenile LR multiplier.

This is intentionally **not** an autonomous-mitosis experiment.

## Measurements

### Performance

At each checkpoint:

- per-domain validation NLL/PPL;
- balanced four-domain NLL;
- geometric-mean PPL;
- elapsed time and peak VRAM.

### Cell differentiation

For each micro-cell, a fixed diagnostic probe records its first tissue invocation on a held-out batch from each domain. The four resulting contribution RMS values form a domain profile.

Cell specialization is:

\[
D_i = 1 - \frac{H(p_i)}{\log 4},
\]

where the contribution profile is normalized across the four domains. `0` means a uniform profile; values closer to `1` indicate stronger domain selectivity.

The experiment also measures:

- output-projection gradient phenotype cosine between domains;
- gradient-conflict proxy `0.5 * (1 - mean cosine)`;
- profile cosine stability versus the same cell at age zero;
- within-tissue pairwise domain-profile cosine as a redundancy measure.

### Stress diagnostic

The developmental-tissue v0 stress equation is evaluated diagnostically from normalized contribution load, validation residual loss, change from the age-zero domain profile, gradient conflict, and local neighbor capacity.

It does **not** modify training and cannot trigger division in Experiment 026. A stress trend therefore cannot by itself be reported as evidence of autonomous developmental pressure.

## Preregistered decision

G=1 is the baseline.

A finer-granularity arm qualifies only when, at the final 20M checkpoint:

1. median micro-cell specialization is at least **0.05 higher** than G=1;
2. balanced NLL ratio versus G=1 is at most **1.02**;
3. mean cell-profile cosine versus age zero is at least **0.80**.

If at least one G>1 arm satisfies all three conditions, and parameter/function parity checks pass, the status is:

`GRANULARITY_DIFFERENTIATION_SIGNAL`

Otherwise:

`NO_GRANULARITY_DIFFERENTIATION_SIGNAL`

This decision does not claim that the qualifying G is globally optimal. It only tests whether finer granularity provides a useful developmental substrate under the frozen environment.

## Formal outputs

The formal result directory is:

`results/experiment-026-cell-granularity/`

It contains:

- `protocol.json` — frozen protocol copied into the run bundle before training;
- `run-provenance.json`;
- `worker-summary.json`;
- per-arm `metrics.csv`;
- per-arm `cell-diagnostics.csv`;
- per-arm `tissue-diagnostics.csv`;
- per-arm `age-zero-parity.json`;
- combined `granularity-trajectory.csv`;
- combined `cell-diagnostics.csv`;
- combined `tissue-diagnostics.csv`;
- `granularity-final.csv`;
- `decision.json`;
- `performance-by-granularity.png`;
- `differentiation-by-granularity.png`;
- `granularity-frontier.png`.

## Kaggle target

The formal job targets **Tesla T4 ×2** with an 8-hour hard wall budget and a 30-minute finalization reserve. Two granularity arms run concurrently; incomplete arms are rotated through worker slices and resume from checkpoints.

Canonical command:

```bash
python scripts/run_experiment_026_cell_granularity.py \
  --total-wall-hours 8 \
  --finalization-reserve-minutes 30 \
  --worker-slice-hours 2.25
```

The canonical notebook is:

`research/kaggle/experiment-026-cell-granularity.ipynb`

The notebook is intended for unattended **Save Version → Run All** execution and publishes only after all four arms and the formal report are complete.

## Interpretation boundary

A positive result would support the hypothesis that the previous `Expert = Cell` abstraction was too coarse and that `Expert = Tissue` is a better substrate for developmental organization.

It would **not** show that routing is endogenous, that morphology self-organizes, or that cells autonomously divide. Those are subsequent experiments. The natural next stage after a positive 026 is to enable stress-driven function-preserving micro-cell fission while keeping the 026 environment frozen, so the effect of autonomous growth can be tested separately.

# Native CLM cross-seed replication

Architecture search is frozen after Phase C. This stage is replication, not another candidate search.

## Frozen experiment

Models:

```text
T1  T1-modern-transformer
M4  M4-c1-modernized
X3  X3-n1-swiglu-rope
X4  X4-n1-swiglu-rope-rmsnorm
```

Development replication seeds:

```text
91002
91003
```

Confirmation seeds remain untouched:

```text
91101
91102
91103
```

Every model is retrained independently on every replication seed. In particular, T1 and M4 are not copied from seed 91001. The training budget, TinyStories revision, tokenizer, context length, optimizer, checkpoint schedule and ~1.9M-parameter matching remain identical to the frozen architecture-search protocol.

Initialization semantics are also preserved rather than silently normalized: historical T1 uses the baseline runtime's `model_seed = seed + 1`; M4/X3/X4 retain the candidate-runner convention `model_seed = seed`.

## Dual-T4 schedule

For each seed:

```text
round 1: T1 / M4
round 2: X3 / X4
```

Seed 91002 completes before seed 91003. The runner is resumable.

## Decision statistics

For each seed, report separately:

- final validation PPL at 10M tokens;
- final validation NLL;
- log-token validation NLL AUC;
- exact parameters;
- analytical train FLOPs;
- inference FLOPs/token;
- measured tokens/s;
- peak VRAM;
- paired PPL delta against the T1 trained on the same seed.

Across seeds, report:

- individual seed points;
- mean and min–max range;
- mean paired ΔPPL versus same-seed T1.

No composite score is used. Quality/parameter and quality/compute remain separate claims.

## Visual evidence

Each seed emits:

```text
replication-seed-<seed>-final-ppl.svg
replication-seed-<seed>-learning-curves.svg
replication-seed-<seed>-quality-compute.svg
```

The cross-seed root emits:

```text
replication-cross-seed-ppl.svg
replication-paired-delta-vs-t1.svg
```

All SVGs are dependency-free and generated directly from the same JSON/CSV evidence.

## Promotion rule

The replication stage answers whether the seed-91001 advantage survives held-back development seeds. It does not consume confirmation seeds and does not promote a model automatically.

Only after the 91002/91003 results are interpreted should one architecture be frozen for untouched confirmation on 91101/91102/91103.

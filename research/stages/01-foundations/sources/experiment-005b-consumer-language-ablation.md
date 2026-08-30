# Experiment 005B — Recurrent Optimization Factorial Ablation

## Purpose

Experiment 005 showed that the 1.17M-parameter MiniTextNCA-S+ reduced 500K-token validation perplexity from the structural TextNCA control's ~254.7 to ~140.7, while a parameter-matched Transformer reached ~112.3. The S+ treatment changed three things at once:

1. LayerNorm -> RMSNorm;
2. GRU update-gate carry bias 0 -> +2;
3. no intermediate supervision -> auxiliary stage losses 0.1 / 0.2.

005B localizes the contribution of these changes before a larger scaling run.

## Design

This is a complete 2^3 factorial experiment. Every combination of three binary factors is trained from scratch:

| Code | Factor OFF | Factor ON |
| --- | --- | --- |
| R | LayerNorm | RMSNorm |
| C | GRU carry bias 0 | GRU carry bias +2 |
| A | final-stage LM loss only | auxiliary stage losses 0.1 / 0.2 + final loss |

The eight cells are:

- `ln-c0-a0`
- `rms-c0-a0`
- `ln-c2-a0`
- `ln-c0-aux`
- `rms-c2-a0`
- `rms-c0-aux`
- `ln-c2-aux`
- `rms-c2-aux`

`ln-c0-a0` reproduces the Experiment 005 TextNCA structural control. `rms-c2-aux` reproduces MiniTextNCA-S+.

## Fixed conditions

All cells use the same:

- TinyStories corpus identity and tokenizer SHA-256 as Experiment 005;
- ~2K ByteLevel BPE vocabulary;
- context length 128;
- d_model 128, 4 heads, FFN 512;
- hierarchical windows `[8, 32, 128]`;
- recurrent iterations `[4, 4, 4]`;
- tied token embedding / LM head;
- AdamW, base LR `3e-4`, warmup 50 steps, cosine schedule;
- model initialization seed `55005`;
- training schedule seed `5005`;
- exactly 500,000 supervised tokens per cell;
- evaluation checkpoints at 125K, 250K and 500K.

Replacing LayerNorm with RMSNorm changes the parameter count slightly because RMSNorm has no bias. The runner records the total parameter spread; all other factor switches add no learned parameters.

## Factorial response

The primary response is 500K-token validation NLL, not perplexity, because NLL is additive. With standard +/-1 coding, every main or interaction effect is:

`effect = 2 * mean(response * factor_sign)`

A negative effect improves NLL. The report also converts the effect back to a perplexity multiplier with `exp(effect)`.

Reported terms:

- main: `R`, `C`, `A`;
- pairwise: `RC`, `RA`, `CA`;
- three-way: `RCA`.

Because this is a single-seed deterministic factorial run, the effects are treatment localization measurements, not statistical confidence intervals. A later multi-seed confirmation can be run if a decision hinges on a small difference.

## Replication gate

Before using 005B to select Experiment 006, both corner conditions must reproduce Experiment 005 within 5% relative PPL:

- `ln-c0-a0` vs 005 `textnca-s`;
- `rms-c2-aux` vs 005 `minitextnca-s-plus`.

If either drifts by more than 5%, the run is marked `NEEDS_ITERATION / EXPERIMENT_005_REPLICATION_DRIFT` and no 006 variant is recommended.

If replication passes, the run is marked `PASS / RECURRENT_OPTIMIZATION_FACTORS_LOCALIZED`, and the lowest 500K validation-PPL configuration becomes `recommended_006_variant`.

## Multi-GPU execution

The parent process prepares the corpus once. On a Kaggle 2xT4 session it launches two independent Python workers at a time with separate `CUDA_VISIBLE_DEVICES` values. Each GPU therefore trains one complete factorial cell; no DDP is used, and the training batch/schedule is unchanged from Experiment 005.

With one CUDA GPU the same script runs all eight cells sequentially.

## Curated outputs

- `decision.json`
- `task-spec.json`
- `corpus-manifest.json`
- `tokenizer.json`
- `model-configs.json`
- `factorial-results.csv`
- `factorial-effects.csv`
- `checkpoints.csv`
- `replication.csv`
- `generation-samples.json`
- `generation-progression.md`
- `best-500k.pt`
- `factorial-ppl.png`
- `factorial-learning-curves.png`
- `main-effects.png`
- `interaction-effects.png`
- `triple-interaction.png`
- `replication.png`

Only the best 500K checkpoint is curated. The other seven checkpoints are intentionally deleted after aggregation because they are regenerable ablation intermediates and are not suitable continuation checkpoints for a clean 10M scaling run.

## Kaggle

Use `research/notebooks/01-foundations/experiment-005b-consumer-language-ablation.ipynb` with a T4 accelerator and Internet enabled. The notebook refreshes `main`, installs the LM dependencies, runs the language bridge + ablation invariant tests, executes 005B, displays the reports, and leaves publication disabled until the outputs have been reviewed.

# Experiment 007 — MiniCells-30M v0

## Purpose

Experiment 007 tests whether the MiniCells language architecture that remained competitive with a parameter-matched Transformer at ~1.17M parameters also remains competitive after a ~25x parameter increase. It also retains the first MiniCells model intended to survive beyond a single experiment run.

This is a parameter-scaling experiment, not a context-length experiment. Context remains 128 tokens so the main architectural change relative to Experiment 006 is model size.

## Models

### MiniCells-30M v0

- vocabulary: 2,048
- context: 128
- hidden dimension: 720
- attention heads: 8
- FFN dimension: 2,880
- hierarchical causal windows: `[8, 32, 128]`
- recurrent iterations: `[4, 4, 4]`
- normalization: LayerNorm
- GRU carry bias: +2
- auxiliary stage losses: disabled
- tied token embedding / LM head
- parameters: 29,602,800

### Transformer-30M control

- vocabulary: 2,048
- context: 128
- hidden dimension: 512
- attention heads: 8
- FFN dimension: 2,048
- layers: 9
- normalization: RMSNorm
- tied token embedding / LM head
- parameters: 29,458,432

Relative parameter error is ~0.49%, below the 1% Experiment 007 gate.

## Data

Dataset: `roneneldan/TinyStories`.

Experiment 007 reuses the exact tokenizer from Experiment 006. It materializes:

- 120,000,000 training tokens
- 1,000,000 validation tokens

The token cache is stored as raw `uint16` memmaps to avoid the multi-gigabyte overhead of Python integer lists or int64 tensors. Before training starts, the first 15M training tokens and first 200K validation tokens are re-hashed using the legacy int64 representation and must exactly reproduce Experiment 006.

## Training

Each model is trained from random initialization with:

- target consumed tokens: 100,000,000
- batch size: 8
- sequence length: 125
- supervised tokens per optimizer step: 1,000
- total optimizer steps: 100,000
- AdamW
- learning rate: 3e-4
- betas: `(0.9, 0.95)`
- weight decay: 0.1
- warmup: 1,000 steps
- cosine decay over the full 100M-token schedule
- gradient clipping: 1.0
- FP16 autocast + GradScaler on CUDA

Evaluation checkpoints:

- 10M
- 25M
- 50M
- 75M
- 100M

With two T4 GPUs, MiniCells and Transformer run concurrently in independent processes, one model per GPU. This is not DDP and does not change the global batch size.

## Exact resume

A full resume checkpoint is overwritten every 5M consumed tokens at:

`results/consumer-language-30m-v1/resume/<model>-latest.pt`

It includes:

- model state
- AdamW state
- GradScaler state
- optimizer step / consumed tokens
- batch RNG state
- Torch CPU RNG state
- CUDA RNG state
- accumulated metrics and generation samples
- elapsed training time
- peak VRAM

The LR schedule always targets 100M tokens. Stopping at 25M or 50M and resuming later therefore does not create a warm restart.

To stop deliberately while preserving an exact continuation point:

```bash
python scripts/research/run_consumer_language_30m.py --stop-after-tokens 25000000
```

On Kaggle, save the two `*-latest.pt` files as notebook output or a Kaggle Dataset. In a later session, mount that output and run:

```bash
python scripts/research/run_consumer_language_30m.py \
  --resume-input /kaggle/input/<saved-output>/resume
```

The runner also accepts `MINICELLS_30M_RESUME_INPUT` and `MINICELLS_30M_STOP_AFTER` environment variables.

## Decision

GREEN requires all of:

- MiniCells / Transformer PPL at 100M <= 1.15x
- MiniCells learning slope / Transformer learning slope >= 0.90
- PPL-ratio deterioration from 10M to 100M <= 0.05

YELLOW requires:

- PPL ratio at 100M <= 1.30x
- slope ratio >= 0.80
- ratio deterioration <= 0.10

Otherwise the result is RED.

## Retained model

At 100M tokens, the worker writes:

`results/consumer-language-30m-v1/minicells-30m-v0-fp16.pt`

This artifact contains FP16 inference weights and the architecture configuration. Full optimizer/resume state is intentionally not published to Git.

After results are merged, generate text with:

```bash
python scripts/research/generate_minicells_30m.py \
  --prompt "Once upon a time there was a small robot"
```

The model is TinyStories-only. It should be evaluated as a small story language model, not as an instruction-following assistant or general-knowledge model.

## Kaggle execution

Use T4 x2 and Internet ON.

The reproducible notebook is:

`research/notebooks/01-foundations/experiment-007-minicells-30m.ipynb`

The notebook:

1. updates the repository;
2. installs `.[lm]`;
3. checks CUDA / GPU count;
4. runs language bridge, ablation, scaling, and 30M tests;
5. starts or resumes Experiment 007;
6. shows partial progress or final results;
7. publishes only after manual review.

For a first attempt, run the default 100M target. If Kaggle session length becomes limiting, use a staged stop such as 25M or 50M, preserve the resume checkpoints, and continue in the next session.

## Publication

After a complete run and review:

```bash
python scripts/research/publish_experiment_007_results.py --push
```

This publishes curated results to:

`kaggle/experiment-007-results`

The publisher includes the FP16 MiniCells-30M artifact, but excludes the large optimizer/resume checkpoints and the Transformer inference checkpoint.

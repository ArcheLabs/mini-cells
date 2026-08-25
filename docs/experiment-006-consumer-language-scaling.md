# Experiment 006 — 10M Language Scaling

## Question

Does the MiniCells-v2 language architecture selected by Experiment 005B close, preserve, or widen its gap to a parameter-matched Transformer as training scales from 500K to 10M consumed tokens?

Experiment 006 deliberately excludes PVM/JAM constraints. It is a language-architecture scaling experiment.

## Source decision

Experiment 005B localized the dominant optimization factor to GRU carry-biased initialization and selected `ln-c2-a0`:

- LayerNorm;
- GRU update-gate carry bias `+2`;
- no RMSNorm;
- no auxiliary stage loss;
- TextNCA hierarchy `[8, 32, 128]`;
- four recurrent iterations per stage.

Experiment 006 locks this configuration as `minicells-v2`.

## Models

### MiniCells-v2

- `d_model = 128`
- 4 attention heads
- FFN width 512
- causal local windows `[8, 32, 128]`
- recurrent iterations `[4, 4, 4]`
- LayerNorm
- GRU carry bias `+2`
- learned step embeddings
- tied token embedding / LM head
- no auxiliary stage supervision

### Transformer-S

The Transformer is automatically parameter-matched to MiniCells-v2 and the run aborts if relative parameter error exceeds 5%.

## Data identity and scale

The exact Experiment 005 tokenizer artifact is reused rather than retrained.

Experiment 006 builds a separate TinyStories cache containing:

- 15,000,000 training BPE tokens;
- 200,000 validation BPE tokens.

Before training, the runner verifies that:

1. the tokenizer SHA-256 equals the Experiment 005 tokenizer SHA-256;
2. the first 800K training tokens reproduce the Experiment 005 training-token SHA-256;
3. the first 100K validation tokens reproduce the Experiment 005 validation-token SHA-256.

This prevents tokenizer or upstream-data drift from being mistaken for a scaling effect.

## Training budget

Each model consumes exactly 10,000,000 supervised tokens.

Shared training settings:

- batch size 8;
- sequence length 125;
- 1,000 supervised tokens per optimizer step;
- 10,000 optimizer steps;
- AdamW;
- learning rate `3e-4`;
- 100 warmup steps;
- cosine decay over the complete 10M-token run;
- gradient clipping at 1.0;
- same deterministic training schedule.

Both models start from random initialization. Experiment 005B checkpoints are not resumed.

## Checkpoints

Evaluation and fixed-prompt generation are recorded at:

- 500K;
- 1M;
- 2M;
- 5M;
- 10M consumed tokens.

The primary trajectory is:

`PPL(MiniCells-v2) / PPL(Transformer-S)`.

Learning slope is estimated from validation NLL across all five checkpoints.

## Parallel execution

With two Kaggle T4 GPUs:

- physical GPU 0 runs MiniCells-v2;
- physical GPU 1 runs Transformer-S.

These are independent processes, not DDP. Each model therefore preserves the single-model batch and optimization semantics.

If only one CUDA GPU is available, the runner executes the two models sequentially.

## Decision

### GREEN

- 10M PPL ratio `<= 1.15`;
- MiniCells/Transformer learning-slope ratio `>= 0.90`;
- PPL-ratio change from 500K to 10M `<= +0.05`.

Interpretation: no meaningful scaling disadvantage is visible through 10M tokens.

### YELLOW

- 10M PPL ratio `<= 1.30`;
- slope ratio `>= 0.80`;
- ratio change `<= +0.10`.

Interpretation: a gap remains but the architecture is still viable for another scale/architecture round.

### RED

Anything outside those bounds.

Interpretation: a scaling disadvantage emerges before 10M tokens and should be understood before increasing model/data scale.

## Curated outputs

- `decision.json`
- `task-spec.json`
- `corpus-manifest.json`
- `tokenizer.json`
- `model-configs.json`
- `checkpoints.csv`
- `model-summary.csv`
- `relative-gap.csv`
- `generation-samples.json`
- `generation-progression.md`
- `minicells-v2-10m.pt`
- `transformer-s-10m.pt`
- `ppl-scaling.png`
- `nll-scaling.png`
- `relative-gap.png`
- `throughput.png`

Kaggle results publish to `kaggle/experiment-006-results` and are curated under `artifacts/experiments/006-consumer-language-scaling` after merge.

# Experiment 005 — 500K Consumer Language Model Bridge

## Purpose

Experiment 005 is the first MiniCells capability-scaling experiment. It deliberately removes PVM/JAM constraints and asks a narrower architectural question:

> At a very small but exactly controlled language-model training budget, does a hierarchical TextNCA-style model learn in the same regime as a parameter-matched Transformer, and do low-risk MiniCells training optimizations improve that trajectory?

The experiment is **not** intended to prove consumer-grade language quality at 500K tokens. Its purpose is to decide whether scaling to 2M, 5M, and 10M+ tokens is empirically justified.

## Models

All models use the same 2,048-token ByteLevel BPE tokenizer, TinyStories token stream, context limit, optimizer family, data order, token budget, and validation positions.

### `textnca-s`

Structural control:

- `d_model = 128`
- 4 attention heads
- FFN width 512
- three causal local-attention stages
- windows `[8, 32, 128]`
- iterations `[4, 4, 4]`
- stage-local recurrent weights shared across the four iterations
- GRU state update
- learned step embeddings
- learned position embeddings
- LayerNorm
- final-stage LM loss only
- token embedding / LM head weight tying

This is a small structural control, not a claim of exact reproduction of the paper's full training setup.

### `minitextnca-s-plus`

Same core architecture plus three deliberately low-risk recurrent-training changes:

- pre-RMSNorm instead of LayerNorm
- GRU update-gate carry bias of `+2.0`
- deep supervision after stages 1 and 2 with weights `0.1` and `0.2`

The final-stage LM loss keeps weight `1.0`.

### `transformer-s`

A causal Transformer using the same `d_model=128`, heads, tokenizer, context, and weight tying. The runner searches layer count and FFN width and refuses to run if the closest configuration is more than 5% away from the `minitextnca-s-plus` trainable parameter count.

## Data

Source: `roneneldan/TinyStories` through Hugging Face Datasets streaming mode.

Tokenizer:

- ByteLevel BPE
- requested vocabulary size: 2,048
- special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- trained deterministically from the first 20,000 non-empty training stories

Model streams:

- 800,000 cached training tokens
- 100,000 cached validation tokens
- the model training budget is independent of the cached stream size

The optional dependencies are installed with:

```bash
pip install -e '.[lm]'
```

Kaggle Internet must be enabled for the first dataset/tokenizer preparation. Once the cache exists in the current session, reruns reuse it.

## Exact token budget

Training uses:

- batch size 8
- sequence length 125
- exactly 1,000 supervised next-token targets per optimizer step
- exactly 500 optimizer steps

Therefore every model consumes exactly:

```text
500 × 1,000 = 500,000 supervised training tokens
```

The same deterministic list of stream start positions is reused for all three models.

Evaluation checkpoints are exactly:

- 125,000 tokens
- 250,000 tokens
- 500,000 tokens

Validation uses a separate fixed set of TinyStories validation positions with context 128.

## Optimization

Shared defaults:

- AdamW
- learning rate `3e-4`
- betas `(0.9, 0.95)`
- weight decay `0.1`
- 50-step warmup
- cosine decay
- gradient norm clipping at `1.0`
- CUDA FP16 autocast + GradScaler on T4

## Quantitative outputs

For every checkpoint and model:

- training loss
- validation NLL
- validation perplexity
- elapsed wall time
- tokens/second
- gradient norm

The report also computes:

- `TextNCA / Transformer` PPL ratio
- `MiniTextNCA+ / Transformer` PPL ratio
- early learning slope `alpha` from `log(NLL) ~ -alpha log(tokens)`
- peak CUDA memory
- parameter count

## Decision policy

The key candidate is `minitextnca-s-plus`.

### GREEN

At 500K tokens:

```text
MiniTextNCA+ PPL / Transformer PPL <= 1.25
```

Interpretation: continue directly to at least 2M tokens.

### YELLOW

The 500K ratio is `1.25–1.60`, but either:

- MiniTextNCA+ learning slope is at least 80% of the Transformer's slope, or
- the relative PPL ratio improves by at least `0.05` from 125K to 500K.

Interpretation: the scaling signal remains worth extending, but architecture/training changes should remain under observation.

### RED

The gap is above the YELLOW range or is not closing fast enough.

Interpretation: change the architecture/training recipe before spending a larger token budget.

These are pre-registered engineering decision thresholds, not claims of universal scaling laws.

## Qualitative progression

At every checkpoint each model generates from the same five fixed prompts using:

- temperature `0.8`
- top-k `40`
- 32 new tokens
- deterministic generation seeds

The samples are recorded in both JSON and Markdown. They are qualitative evidence only and do not replace validation perplexity.

## Generated report

The runner produces:

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
- three resumable 500K model/optimizer checkpoints
- `training-curves.png`
- `ppl-scaling.png`
- `relative-gap.png`
- `learning-slope.png`
- `throughput.png`
- `consumer-readiness-summary.png`

## Kaggle procedure

The notebook `research/kaggle/experiment-005-consumer-language-bridge.ipynb` contains a self-recovering bootstrap. If `/kaggle/working/mini-cells` disappeared after a session reset, it clones `main` again automatically.

The equivalent manual sequence is:

```bash
cd /kaggle/working
git clone --depth 1 https://github.com/ArcheLabs/mini-cells.git
cd mini-cells
pip install -e '.[lm]'
python -m pytest tests/test_language_bridge.py -q
python scripts/run_consumer_language_bridge.py
```

After reviewing the figures and `decision.json`:

```bash
python scripts/publish_experiment_results.py 005 --push
```

The curated result branch is:

```text
kaggle/experiment-005-results
```

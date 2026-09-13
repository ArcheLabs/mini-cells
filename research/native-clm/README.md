# Native CLM Architecture Lab

`research/native-clm/` is the long-lived architecture laboratory for **native cellular language models**.

The purpose of this research is deliberately narrower than the continual-learning program:

> Find the best next-token-prediction architecture for a native CLM before adding growth, mitosis, online writes, rollback, replay-free learning, JAM integration, or on-chain constraints.

The canonical working surface is a single living notebook:

- `native_clm_lab.ipynb`

Execution infrastructure shared by that notebook lives beside it:

- `native_clm_runtime.py` — frozen data/model/training/evaluation primitives;
- `run_native_clm.py` — resumable single-model worker and paired one/two-GPU launcher.

The notebook may evolve as new architecture ideas are added. Completed run records must not be rewritten. Each promoted run is identified as `NCLM-XXX` and its immutable metrics/configuration is exported under `results/` before a later notebook change supersedes the working candidate.

## Research question

Primary question:

> Can a parameter-matched native CLM outperform a modern small Transformer on next-token prediction?

The initial scale is approximately 2M parameters because it is intended for fast architecture search. Candidates that survive this stage should be re-tested at approximately 8M, 30M, and then larger scales.

The comparison distinguishes three claims:

1. **quality / parameter** — same train tokens and approximately the same parameter count;
2. **quality / training compute** — approximately the same total training FLOPs;
3. **quality / inference compute** — account for recurrent CLM steps rather than treating recurrence as free depth.

A win in (1) must not be reported as a win in (2) or (3).

## Frozen initial baselines

### C0 — historical Native CLM, scaled to ~2M

C0 is a scaled reconstruction of the architecture used by MiniCells Experiment 007, not the earlier moving-average engineering scaffold.

The historical structure is retained:

- three independent NCA stages;
- local causal self-attention in each stage;
- GELU FFN;
- `GRUCell` recurrent update;
- learned per-step embeddings;
- four recurrent iterations per stage;
- hierarchical windows `(8, 32, 128)`;
- LayerNorm;
- GRU carry/update bias `+2`;
- tied token embedding / LM head.

The historical 30M model used `d=720`, `FFN=2880`, eight heads. The ~2M baseline uses `d=168`, `FFN=672`, eight heads, preserving the 4× FFN ratio while parameter-matching T1.

Current trainable parameters with vocabulary 2,048:

- C0: **1,899,576**
- T1: **1,894,080**
- C0/T1: **1.002902×** (about 0.29% mismatch)

### T1 — modern ~2M decoder Transformer

T1 uses:

- four decoder blocks;
- `d_model=192`;
- six query heads / two KV heads (GQA);
- RMSNorm;
- RoPE;
- SwiGLU with FFN width 480;
- pre-norm residual blocks;
- tied token embedding / LM head.

T1 is the modern architecture baseline, not a claim that this one 2M configuration is globally optimal.

## Frozen data and training protocol

The training runtime pins:

- dataset: `roneneldan/TinyStories`;
- revision: `f54c09fd23315a6f9c86f9e14f9f4f06615b22e3`;
- tokenizer: deterministic ByteLevel BPE trained from the first 20,000 pinned training stories;
- vocabulary: 2,048 including `<pad>`, `<unk>`, `<bos>`, `<eos>`;
- context: 128;
- training sequence length: 125;
- batch size: 8, therefore exactly 1,000 consumed training tokens per optimizer step;
- optimizer: AdamW, betas `(0.9, 0.95)`, weight decay `0.1`;
- base LR: `3e-4`, linear warmup then cosine decay;
- gradient clipping: `1.0`;
- CUDA precision: FP16 autocast + GradScaler;
- validation: identical fixed examples for C0 and T1 within each seed;
- C0 and T1 use the same deterministic batch schedule within a paired seed.

Profiles:

| profile | tokens/model | train stream | validation stream | purpose |
|---|---:|---:|---:|---|
| `smoke` | 100K | 200K | 50K | plumbing only |
| `dev` | 1M | 1.2M | 250K | fast architecture development |
| `baseline` | 10M | 12M | 1M | baseline decision runs |

Development seeds are `91001–91003`. Untouched confirmation seeds are `91101–91103`.

## Dual-GPU execution

For two visible GPUs the paired runner follows the useful pattern from Experiment 007: it trains **one complete model per GPU concurrently** rather than applying DataParallel to a ~2M model.

- GPU 0: C0
- GPU 1: T1

With one GPU the models run sequentially. CPU is rejected for normal training and is available only through an explicit debug flag.

Large token caches, Hugging Face caches, optimizer states and resume checkpoints are written under `/kaggle/working/native-clm` on Kaggle. They are not committed to Git.

The notebook reads `HF_TOKEN` and `GITHUB_TOKEN` from environment variables or Kaggle Secrets without printing their values. The TinyStories dataset is public, so `HF_TOKEN` is optional for correctness but is reused when available.

## Compute accounting

Every paired result reports:

- exact parameter count;
- validation NLL and PPL;
- consumed training tokens;
- analytical forward FLOPs/token;
- estimated training FLOPs (`~3 × forward`, explicitly labelled as an estimate);
- measured tokens/s;
- elapsed time;
- peak VRAM.

The initial C0 recurrence executes materially more compute than T1 at equal parameter/token budget. Therefore the first baseline can support a **quality/parameter** statement only; it cannot establish a quality/compute win.

The FLOP estimator counts the intended local-attention algorithm. Wall-clock throughput is retained because the current masked SDPA implementation may not realize ideal sparse-local attention cost on every backend.

## Running the baseline

Open `native_clm_lab.ipynb` and run all cells. Its default configuration is a real `baseline` development run at seed `91001`, i.e. 10M tokens per model.

The notebook:

1. finds or clones `research/native-clm-lab`;
2. installs the repository's existing `.[lm]` dependencies;
3. reuses Kaggle/Hugging Face cache conventions and secrets;
4. audits parameter/FLOP budgets;
5. prepares the pinned tokenizer/corpus with hashes;
6. launches C0 and T1 concurrently when two GPUs are available;
7. resumes from compatible checkpoints after interruption;
8. prints a paired leaderboard;
9. copies only small development evidence into `research/native-clm/results/dev/`;
10. optionally commits/pushes development evidence to the same branch when `GITHUB_TOKEN` is available.

Confirmation results are deliberately excluded from the convenience auto-push path and are reviewed before promotion.

## Historical evidence boundary

Existing MiniCells work contains strong evidence about **continual-learning mechanisms**, but much less direct evidence about static native-CLM architecture quality.

Results such as growth-restored plasticity, replay-free/subspace-certified mitosis, routing/growth engineering, functional-boundary experiments and CLM-0.4 continual-learning validation constrain this search but are not PPL architecture wins.

In particular, real-representation work warns against assuming that address-based specialization is equivalent to functional specialization.

## Living notebook, immutable evidence

`native_clm_lab.ipynb` is intentionally mutable. It is the current laboratory, not an immutable scientific artifact.

Development evidence is additive under:

```text
research/native-clm/results/dev/<profile>-seed-<seed>/
```

Promoted runs are immutable:

```text
research/native-clm/results/
  NCLM-001/
    config.json
    metrics.json
    environment.json
    README.md
```

A later experiment may supersede the interpretation of an older run, but must not overwrite its recorded configuration or metrics.

## Initial optimization order

Do not start architecture search from one seed. First complete the frozen baseline development and confirmation sets. Then change one architectural factor at a time:

1. `C1`: gated/update dynamics;
2. `C2`: phase/step-conditioned recurrence;
3. `C3`: small phase-specific modulation while retaining shared recurrent parameters;
4. `C4`: receptive-field schedule;
5. later: dual state, adaptive recurrence, compute/communication separation, functional Cell specialization.

The initial Native CLM lab does not validate continual learning, Cell growth/mitosis, transaction rollback, replay-free safety, MoE-to-CLM conversion, or JAM execution/consensus constraints. Those mechanisms can be reintroduced after a strong static Native CLM is established.

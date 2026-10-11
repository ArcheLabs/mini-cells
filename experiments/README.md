# MiniCells Experiment 001

Experiment 001 trains a fixed dense Transformer and a Global Router language
model on TinyStories. Routing uses eight FFN Cells per block with Top-2
activation. Both models have six causal attention blocks, width 512, eight
attention heads and a tied 8,192-token embedding/output head.

The dense FFN has hidden width 8,192; each routed Cell has width 1,024.
This matches total parameters at approximately 61M, with small router overhead.
Active parameters and compute differ: this is a capacity-matched dense/sparse
comparison, not an equal-compute claim or evidence for an internal Cell router.
Cell internal routing and STOP are later experiments.

## Registered training protocol

Configuration: `configs/routing/experiment-001.json`.

- Development seed: 92001; does not consume Native CLM's reserved seeds.
- TinyStories revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- First 50,000 training and 1,024 validation stories; validation is never used
  to train the tokenizer. ByteLevel BPE is trained on the first 10,000 training
  stories. Tokenizer, text and token hashes are saved for audit.
- Same seeded random token windows with replacement for both models, with EOS
  delimiters between stories. Validation windows are fixed and held out.
- Exactly 10M consumed training tokens per model: 5,000 steps, 16 sequences
  per batch, 125 predicted tokens per sequence.
- AdamW, LR 3e-4, betas (0.9, 0.95), weight decay 0.1, 200-step warmup,
  cosine decay to 10% of base LR, gradient clipping 1.0, FP16 + GradScaler.
- Global Router: normalized Top-2 weights, no token dropping/capacity cap;
  load balance coefficient 0.01. Validation loss excludes the auxiliary loss.
- Evaluate at initialization and every 1M training tokens. Save the latest
  model/optimizer/scaler checkpoint and learning curve after evaluation.

The first run is a development comparison. One seed and 10M tokens cannot
establish architectural superiority or convergence. No result is claimed until
the GPU run completes. Existing `research/native-clm/` evidence and protocols
remain a separate research track.

## Execute

Actions → **Kaggle Experiment** → **Run workflow**. Only manual dispatch submits
GPU jobs; merging code does not launch or repeat smoke tests. The repository
needs only the `KAGGLE_API_TOKEN` secret for the `bingjiangli` account.

The dedicated notebook is `bingjiangli/minicells-experiment-001`, titled
"MiniCells Experiment 001". Actions embeds experiment source and config from
the checked-out commit into the uploaded script. Kaggle internet is enabled
for the pinned dataset and missing data dependencies; no credential is embedded.

Local CUDA execution from repository root:

```sh
python -m pip install torch numpy datasets tokenizers
python -m experiments.runners.train --output experiment-output
```

## GPU budget and evidence

The two models train concurrently in independent processes: Fixed on physical
GPU 0, Global Router on physical GPU 1. Each worker sees one GPU through
`CUDA_VISIBLE_DEVICES` and uses local `cuda:0`. This uses both T4s for independent
experiments rather than splitting one model with DDP. Data and tokenizer are
prepared once; workers share identical memory-mapped token arrays and sampling
schedules. A single-GPU session is rejected instead of silently running the
pair sequentially.

Each model has at most 45 minutes for initialization, training, evaluation and
intermediate checkpoints. The whole runner has a shared 55-minute deadline
starting before dependency setup, with two minutes reserved for checkpoint
saving. The server's hard session timeout is one hour. Setup time reduces the
training window for both workers. These are ceilings, not runtime promises;
Kaggle's quota display determines the actual charge for the dual-GPU session.

If either model fails or runs out of time, outputs are preserved and
`comparable_at_full_budget` is false. Such a run must not be reported as a
completed equal-token comparison. The workflow downloads outputs from the
actual URL and version returned by the Kaggle push API, validates completion,
and uploads evidence even if this validation fails.

Artifacts include the full config, commit/environment provenance, tokenizer,
data fingerprints, per-model checkpoint/metrics/learning curve, validation
Cell utilization and entropy. FLOPs and activated parameters are analytic
estimates with a stated scope; tokens/s and peak allocated GPU memory are
measured. The router's path is one parallel selection per FFN; it is not a
learned sequential path and does not test reasoning generalization.

Official Kaggle references:
[authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication),
[metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

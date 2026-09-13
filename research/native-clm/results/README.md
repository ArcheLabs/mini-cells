# Native CLM result records

This directory contains two evidence classes.

## Development evidence

The living notebook may copy small paired-run evidence to:

```text
dev/<profile>-seed-<seed>/
  README.md
  pair-summary.json
  protocol.json
  C0-summary.json
  T1-summary.json
  C0-checkpoints.csv
  T1-checkpoints.csv
```

These records are useful for architecture decisions but are **not promoted scientific results**. Large token caches, model/optimizer states and resume checkpoints remain outside Git.

Development seeds are `91001–91003`. Convenience GitHub push is permitted for these records.

## Promoted evidence

Use one immutable directory per promoted run:

```text
NCLM-001/
  README.md
  config.json
  metrics.json
  environment.json
```

Confirmation seeds are kept out of the notebook's convenience auto-push path and must be reviewed before promotion.

Minimum promoted metrics for a direct CLM/Transformer comparison should include:

- `validation_nll`
- `validation_ppl`
- `parameter_count_total`
- `parameter_count_trainable`
- `train_tokens`
- `context_length`
- `estimated_train_flops`
- `estimated_inference_flops_per_token`
- `tokens_per_second`
- `peak_memory_bytes`
- `seed`
- corpus/tokenizer hashes and pinned dataset revision

A promoted result directory is append-only. If a run is invalidated, document why in its README or in a later superseding run; do not silently rewrite old metrics.

The living `../native_clm_lab.ipynb` may change freely. Promoted directories are the durable scientific record.

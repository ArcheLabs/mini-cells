# Native CLM result records

This directory stores promoted, immutable Native CLM experiment records.

Use one directory per promoted run:

```text
NCLM-001/
  README.md
  config.json
  metrics.json
  environment.json
```

Minimum `metrics.json` fields for a direct CLM/Transformer comparison should include:

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

A result directory is append-only once promoted. If a run is invalidated, document why in its README or in a later superseding run; do not silently rewrite the old metrics.

The living `../native_clm_lab.ipynb` may change freely. This directory is the durable scientific record.

# CLM-0.4 Release

CLM-0.4 Release promotes the successful Preview configuration into a guarded release workflow.

## Frozen model/data recipe

The first release intentionally keeps the Preview recipe unchanged:

- decoder-only Transformer, 4 layers, `d_model=256`, 8 heads
- L1/L2 dense FFN hidden 768
- L3/L4 shared FFN hidden 256 + 32 sparse base Cells/layer
- Cell hidden 32
- 8,192-token digit-aware BPE
- sequence length 256
- base mixture: 60% TinyStories / 30% controlled Math / 10% controlled Story
- AdamW base training defaults from `BaseTrainConfig`
- continual direct optimizer: AdamW, lr 0.003, 32 steps
- growth optimizer: AdamW, lr 0.003, 64 steps
- 192 continual transactions

Base CLM parameters: **5,273,088**.

The release benchmark also trains one ordinary dense Transformer with hidden 1032 and **5,273,120 parameters**, a difference of only 32 parameters. It uses exactly the same data assets, tokenizer, seed, base optimizer, and continual curriculum. The dense continual baseline is full-model always-commit fine-tuning.

## Mandatory two-stage execution

### Stage A — 1M end-to-end smoke

Profile: `smoke-1m`

This is not a capability benchmark. It must execute the entire release path:

1. generate 1M-token assets with the frozen tokenizer/data generator;
2. train CLM base;
3. run all 192 CLM continual transactions;
4. train the equal-parameter dense base;
5. run all 192 dense continual transactions;
6. generate CLM telemetry and visualizations;
7. generate CLM-vs-Dense comparison telemetry and visualizations;
8. generate report artifacts;
9. produce `release-readiness.json`.

A successful smoke emits:

```text
status = READY_FOR_30M
```

The readiness artifact locks:

- release pipeline SHA-256;
- hashes of the critical source files used by training/runtime/report/publishing;
- tokenizer payload identity;
- full 192-transaction completion for both CLM and Dense;
- expected CLM/Dense parameter counts;
- required analysis artifacts.

No minimum Math/Story quality threshold is imposed on the 1M smoke because its purpose is engineering validation, not model-quality selection.

### Stage B — 30M release

Profile: `release-30m`

The 30M runner refuses to start unless a `READY_FOR_30M` smoke artifact is supplied. It also verifies that:

- the release pipeline identity is unchanged;
- all critical source-file hashes are unchanged;
- the 30M tokenizer payload hash matches the 1M smoke tokenizer;
- the 1M smoke completed all 192 transactions.

If any critical source file changes after the smoke, the 1M smoke must be rerun before 30M can start.

## Kaggle

Use:

```text
research/notebooks/05-language-validation/clm-0.4-release.ipynb
```

The notebook runs the 1M stage first. `START_30M = False` by default. Inspect the smoke output and only then set it to `True`.

Recommended accelerator: **T4 x2**.

## Publication branches

1M engineering smoke:

```text
kaggle/clm-0.4-release-1m-smoke-results
```

30M release:

```text
kaggle/clm-0.4-release-30m-results
```

Curated evidence includes JSON/JSONL/CSV/Markdown/PNG. Raw token shards, tokenizer payloads, and model checkpoints are excluded from Git.

## Interpretation

The 1M output is a pipeline/readiness artifact and should not be marketed as CLM-0.4 capability.

The 30M output is the release benchmark and includes equal-data/equal-token/equal-parameter CLM-vs-Dense comparison plus the longitudinal Cell-growth and continual-learning telemetry inherited from Preview.

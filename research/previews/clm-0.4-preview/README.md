# CLM-0.4 Preview

CLM-0.4 Preview is the product-facing successor to the frozen M1-v1/v2 development validations. It is designed to ship quickly, learn continuously, and expose longitudinal public telemetry. Preview metrics are observational and must not be described as formal scientific validation.

## Architecture

The first two Transformer blocks use dense FFNs. Blocks 3 and 4 use:

```text
attention residual
+ shared FFN residual
+ sparse Top-2 Cell residual
+ optional owner-private growth Cell residual
```

The shared FFN is trained during base training and frozen during continual learning. Continual transactions may mutate only routed base Cells or committed owner-private growth Cells.

Default Preview configuration:

- 4 decoder blocks
- 256 model width, 8 attention heads
- 8192 byte-BPE vocabulary
- 256-token maximum sequence length
- dense FFN hidden 768 in blocks 1–2
- shared FFN hidden 256 in blocks 3–4
- 32 base Cells in each of blocks 3–4
- Cell hidden 32, deterministic Top-2 address routing
- ~5.273M base parameters

## Base data

Preview keeps a 30M-token base budget but reallocates the controlled mix:

- 60% pinned TinyStories language carrier
- 30% aligned controlled Math QA
- 10% aligned context-conditioned Story QA

The Preview tokenizer is trained on carrier + controlled templates and exposes contiguous decimal numbers as individual digits before BPE. Decoding joins those digit boundaries again.

## Product execution policy

Base Math/Story capability metrics are public quality metrics, not execution gates. Preview continues when assets/checkpoints are valid and numeric execution is healthy. Dependency-scoped validation still controls continual commits and growth.

Default continual optimizer:

- direct Cell update: AdamW, batch 32, LR 0.003, 32 steps
- growth/private update: AdamW, batch 32, LR 0.003, 64 steps

These are product defaults and may be versioned as Preview evolves; every public result records the actual run configuration.

## Public telemetry

Every run emits analysis-sized evidence:

### Product/market metrics

- total transactions and effective commits
- acceptance and rollback rates
- base, grown, and total Cell counts
- Cell births over time
- growth attempts, growth commits, growth rescue rate
- private Cell reuse and reuse acceptance
- Math and Story capability snapshots
- protected knowledge/probe count
- cumulative training and validation wall time
- parameter count and growth overhead

### Research/engineering metrics

- per-transaction new-task gain
- local and hidden-global regression
- dependency-scope coverage
- touched parameter fraction
- false-safe rate
- structural escape rate
- Cell activation/update/rejection counts
- Cell birth transaction and owner address
- dependency probe count per Cell
- state hashes and transaction decisions

### Files

```text
results/clm-0.4-preview/
├── decision.json
├── summary.json
├── dashboard.json
├── PUBLIC_METRICS.md
├── RESULTS.md
├── base/
│   ├── checkpoint.pt              # local only; publisher excludes it
│   └── base-metrics.json
├── checkpoints/                   # local only; publisher excludes it
├── telemetry/
│   ├── timeline.csv
│   ├── transactions.jsonl
│   ├── cell-snapshots.jsonl
│   └── cell-registry-final.jsonl
└── visualizations/
    ├── cells-growth.png
    ├── learning-decisions.png
    ├── learning-vs-regression.png
    ├── dependency-coverage.png
    ├── compute-cost.png
    ├── parameter-growth.png
    ├── capability-over-time.png
    └── cell-activity-top20.png
```

`dashboard.json` and `PUBLIC_METRICS.md` are intended to be directly consumable by public web surfaces.

## Kaggle flow

Use `research/notebooks/05-language-validation/clm-0.4-preview.ipynb` with two T4 GPUs.

The notebook:

1. updates the repository;
2. installs MiniCells;
3. deterministically rebuilds Preview data if `/kaggle/working` was cleared;
4. trains or resumes the Preview base checkpoint;
5. runs/resumes continual transactions;
6. regenerates public charts/report;
7. publishes curated telemetry to `kaggle/clm-0.4-preview-results`.

The result publisher excludes raw 30M-token shards, tokenizer payloads, and model checkpoints.

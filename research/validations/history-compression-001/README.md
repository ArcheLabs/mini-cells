# History Compression 001

Status: **PROTOCOL FROZEN — FORMAL GPU RUNS PENDING**

This validation follows Functional Boundary Oracle 001. The Oracle result showed that explicit frozen-base historical supervision can identify and safely train a 32-channel aligned sub-expert coordinate on Granite 3.1 1B-A400M under the frozen calibration protocol.

History Compression 001 keeps the substrate, writable granularity, optimizer family, safety gates, and withheld historical evaluation fixed while reducing only the learner-visible historical calibration prompt budget:

```text
32 -> 8 -> 2 -> 0
```

The experiment asks for the smallest observed historical prompt budget that remains supported in at least 2/3 formal seeds, with `full_32` serving as a required positive control.

Canonical assets:

- [`protocol.json`](protocol.json) — machine-readable frozen protocol
- [`PROTOCOL.md`](PROTOCOL.md) — scientific rationale and decision rules
- formal notebook: `research/notebooks/07-safe-model-evolution/history-compression-001-kaggle.ipynb`
- implementation: `scripts/research/history_compression_001/`
- durable results after execution: `artifacts/experiments/history-compression-001/`

This test does **not** mix replay-budget compression with Fisher, low-rank gradient, activation, or other historical-certificate mechanisms. Those mechanisms should be tested separately after this ladder establishes the replay-budget boundary.

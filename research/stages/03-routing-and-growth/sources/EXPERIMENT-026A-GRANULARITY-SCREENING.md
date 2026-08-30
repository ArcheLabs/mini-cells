# Experiment 026a — Granularity Screening

Experiment 026a is the low-cost mechanism screen that precedes the longer Experiment 026 confirmation run.

## Question

Before spending the full 026 budget, does finer micro-cell granularity produce an early, measurable increase in local differentiation under the same normalized local-plasticity rule?

## Arms and budget

- G=1, G=4, G=8 micro-cells per tissue.
- 12 fixed root tissues in every arm.
- 5M continuation tokens per arm.
- 15M total training tokens.
- Balanced Story / Math / Symbolic / Facts continuation.
- Persistent mitosis disabled.
- T4x2 target; two arms run concurrently and incomplete arms resume from checkpoints.

The screen reuses the tested Experiment 026 worker and data pipeline. It changes only the arm set, token budget, output directory, frozen protocol, and report gate.

## Screening gate

G=1 is the reference. A finer arm justifies the larger confirmation run when all are true at 5M tokens:

1. specialization gain (final minus its own age-zero baseline) exceeds the G=1 gain by at least 0.02;
2. balanced NLL is no worse than 1.03x G=1;
3. non-zero cell-local plasticity variance is actually observed.

A passing screen emits `PROCEED_TO_026_CONFIRMATION`. A failing screen emits `DO_NOT_PROCEED_TO_026_CONFIRMATION`.

This is deliberately a decision gate, not a confirmatory scientific claim.

## Kaggle

Use `research/notebooks/03-routing-and-growth/experiment-026a-granularity-screening.ipynb` with Tesla T4x2 and Internet enabled. The notebook clones `codex/experiment-026a-granularity-screening`, runs regression/mechanism tests and the granularity smoke check, then starts the screening runner.

Canonical local/Kaggle command:

```bash
python scripts/research/run_experiment_026a_granularity_screening.py
```

Formal outputs are written to `results/experiment-026a-granularity-screening/`.

# Experiment 019 Numerical Recovery

The first Kaggle execution completed all three GPU workers and all six trained donor tissues per replicate, then failed during CPU estimator postprocessing because an estimator fold produced all-NaN predictions. Pandas subsequently returned `NaN` from `Series.idxmax()`, which surfaced as `KeyError: nan`.

This is a postprocessing failure, not a donor-training failure. Worker outputs remain reusable.

## Cause

The original estimator pipeline assumed every recorded floating-point feature was finite. A non-finite proposal observable can propagate through fold standardization into ridge/MLP predictions. The old selection helper then hid the real cause by calling `idxmax()` on an all-NaN prediction group.

## Recovery policy

`run_proposal_utility_discovery_resumable.py` now:

1. Detects complete `r0/r1/r2` worker outputs and skips GPU retraining.
2. Writes `finite-audit.csv` for both oracle values and all proposal features.
3. Treats non-finite oracle values as a hard experiment failure; oracle values are never imputed.
4. Repairs feature-only non-finite values using the finite **training-fold median** for that feature. Held-out statistics are never used.
5. Replaces a feature that is entirely non-finite in the training fold with neutral zero for that fold and records the audit.
6. Standardizes only after finite repair and clips diagnostic z-scores to ±20 to prevent the small MLP diagnostic from numerical blow-up.
7. Explicitly rejects any non-finite estimator prediction before selection metrics.
8. Uses positional `argmax` inside each candidate group rather than depending on pandas `idxmax()` behavior for missing values.

The scientific oracle, leave-one-family-out split, feature definitions, estimators, thresholds, and pre-registered decision criteria are unchanged.

## Kaggle recovery

After pulling the hotfix, an interrupted run with complete worker outputs only needs:

```bash
python scripts/run_proposal_utility_discovery_resumable.py
```

The script prints `reusing completed Experiment 019 workers` when no GPU donor retraining is performed.

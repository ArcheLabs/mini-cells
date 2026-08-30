# Core Validation 002 Results

- Status: `WRITE_ADDRESSABILITY_NOT_SUPPORTED`
- Primary seed passes: `0/3`
- Core comparison: identical learned sparse representation with inferred local write vs global writer SGD.
- Causal control: read-address inference preserved while write destination is permuted.
- Mechanistic prediction: off-support squared latent activation predicts Write Leakage.
- Scope: synthetic additive sparse-functional world; no language, replay, growth, or encoder updates during edits.

See `protocol.json`, `decision.json`, `seed-summary.csv`, and `edit-records.csv`.

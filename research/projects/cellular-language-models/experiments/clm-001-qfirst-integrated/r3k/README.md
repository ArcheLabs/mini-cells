# R3k — Future Learning Value

Import `notebooks/kaggle-future-value.ipynb` into the original Kaggle Notebook. Use a **new cloud session**, **T4 x2**, Internet, and existing **HF_TOKEN / GITHUB_TOKEN**. Run All once. No source hash, version, separate pilot or test-suite entry is required.

## Fixed study

- All 48 real R3i candidates, exactly 12 reused R3j cases and 36 new historical rejected cases.
- Restore original K10 full Old/New state, preserve absolute optimizer/scheduler/data/RNG semantics and self-connections. No new proposals or Connection optimization.
- Audit exact raw indices/content, fixed TinyStories revision, historical shard and tokenizer hashes. Exclude exact duplicates without replacements. Predictor [8192,8448), research Value Holdout [8448,8960); formal Confirmation remains SEALED.
- Freeze 128 evaluation windows per research role with the existing story-grouped sampler; audit all256/512 raw Story indices. This fixed evaluation count matches the historical Development window count and avoids multiplying evaluation cost.
- Compute all five specified scores on the independent Predictor Probe. No R3j Development metric reader is used for prediction.
- Persist and verify all48 frozen predictions before importing historical future states or continuing NEW36. Continue only90 new Cell updates per branch; 6480 updates / 3317760 new tokens maximum.
- Freeze all48 long-horizon models before mapping the new Holdout for evaluation. Evaluate horizons10/50/100 using the inherited deterministic H/T inference, all four Cell/Connection combinations and two repeats.
- Report V=R+C, interaction separately, fixed Story bootstrap, all five predictors, NEW36/ALL48 and stratified results. Three primary comparisons receive Holm-adjusted exploratory score-label permutation results with explicit trajectory-dependence limitations. No trainable predictor, weight fitting or automatic R3l.

## Persistence and recovery

Reuse R3j incremental Store publication, under a separate R3k HF pointer namespace. New immutable objects are verified and uploaded; unchanged exports do not create commits. The original paired continuation checkpoints each completed update and retains horizon states plus the newest pair. Rerunning the same pinned Notebook restores the verified HF segment and skips already committed work. Save Version Output remains an additional copy.

`--report-existing` only regenerates and republishes saved reports. It does not train. GitHub receives the explicit aggregate allowlist only, using the existing authentication and main-descended result publisher, followed by remote commit verification.

## Validation scope

CPU tests use synthetic data and an offline HF backend: prediction formulas/immutable freezing, complete matrix, data exclusion, actual90-step paired continuation and restart, Connection freeze, no new proposals, no retraining reused cases, blocked premature Holdout evaluation, fixed historical Store restore, two-session transport simulation, repeated four-grid evaluation, full48 reports and worker CLI compatibility.

Real private HF access, raw-story data proof and T4 runtime are checked on launch. Synthetic CPU success does not claim real Kaggle cross-session verification or scientific support. Cost reports distinguish measured time, inherited executed operator rows and unavailable hardware FLOPs; no FLOP speed claim is inferred from wall time.

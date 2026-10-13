# Core 008 Postmortem Status

Current state: `IMPLEMENTATION_COMPLETE_UNRUN`

This branch adds a non-confirmatory diagnostic bridge over the already-observed Core Validation 008 seeds.

Implemented:

- frozen diagnostic protocol;
- exact pinned Pythia/SlimPajama identity and manifest verification;
- per-write truncated-SVD intrinsic-rank curves;
- train-fitted global PCA heldout capacity curves;
- PCA sparsity curves for top-k coefficient retention;
- 32-rank-unit offline factorized dictionary controls for rank-1/2/4/8 atoms;
- comparison against the published Core 008 adaptive residuals;
- resumable three-seed Kaggle orchestration;
- report and artifact publisher that preserves `scientific_decision=false` and does not modify Core 008 artifacts;
- unit tests for rank bounds, PCA reconstruction, sparsity monotonicity, budget accounting, and protocol boundary.

No scientific claim is emitted by this bridge. GPU execution on seeds `80821`, `80822`, and `80823` is still required.

# Core Validation 009A Status

Status: `IMPLEMENTED_DISCOVERY_UNRUN`

Branch: `codex/core-validation-009a-factorized-functional-coordinates`

## Implemented

- Frozen geometry-only protocol.
- Fresh discovery seeds `80901/80902` and untouched confirmation seeds `80911/80912/80913`.
- Left/right covariance eigenspaces from normalized frozen-Pythia write demands.
- Left-only, right-only and heldout two-sided residual diagnostics.
- Full `(m,n)` geometry landscape.
- Budget-matched `m+n=64` split selection under the 4096-scalar Core 008 basis budget.
- Per-write rank-1 oracle identity reference.
- Deterministic winner selection and viability stop rule.
- Committed winner-lock requirement before confirmation.
- Partial/resumable reporting and per-seed Kaggle publication.
- Unit tests for synthetic tensor-product recovery, one-sided geometry, seed isolation, budget invariants and discovery selection.

## Not yet claimed

No discovery seed, confirmation seed, or scientific Core 009A decision has been run by this implementation commit. GPU/Kaggle results must be published before any geometry conclusion is stated.

## Stop boundary

If no discovery split reaches heldout median local-action residual `<= 0.45` independently on both discovery seeds, confirmation is forbidden by the frozen protocol.

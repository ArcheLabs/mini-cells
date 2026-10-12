# Core Validation 009A Bridge Status

Status: `IMPLEMENTATION_COMPLETE_GPU_DIAGNOSTIC_PENDING`

Scientific decision: `false`

Source Core 009A status: `FACTORIZED_FUNCTIONAL_COORDINATES_SUPPORTED` (immutable)

Source commit: `8290d4d674a8ec9ce98d4de129043526841e5f95`

## Implemented

- frozen post-confirmation protocol;
- exact raw 009A `(56,8)` source-reproduction guard;
- raw / centered / whitened / mean-direction-removed controls;
- token-normalized and token-energy-weighted write spectra;
- sequence-level right/left spectra;
- heldout right-only and `(56,n)` two-sided residuals for `n={1,2,4,8}`;
- raw right-PC1 alignment diagnostics;
- right-PC1 functional ablation and residual spectrum;
- per-seed checkpointing;
- report/CSV/PNG generation;
- Kaggle publishing and hydration;
- one-cell Kaggle notebook;
- mechanism/protocol tests.

## Pending

Run diagnostic seeds `80911, 80912, 80913` on Kaggle GPU. Until those artifacts exist, no claim should be made about why the right side collapsed.

The bridge can only choose the hypothesis for a later 009B. It cannot alter the completed 009A scientific result.

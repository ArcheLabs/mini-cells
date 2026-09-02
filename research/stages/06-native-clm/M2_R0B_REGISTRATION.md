# Native CLM v0 — M2-R0b Numerical Reference Audit Registration

## Status

`FROZEN_UNRUN`

M2-R0b is a diagnostic continuation of M2-R0. It is **not** a new continual-language experiment, does not consume a new formal seed, does not alter the historical M2 decision, and does not establish a Native CLM training milestone.

Parent evidence:

- M2-R0 publish commit: `c1f9e132b026efe24a0238e6ea333bdd2ae5fbdb`
- M2-R0 classification: `INCONCLUSIVE_REFERENCE_FAILURE`
- Historical M2 decision remains: `NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED`

## Why R0b exists

M2-R0 found strong evidence that projecting the raw gradient is not equivalent to constraining the realized parameter transaction under AdamW. It also found that final realized-update projection nearly restores the registered nullspace invariant.

However, the algebraic SGD/no-decay reference failed the strict relative `rho` threshold. Its first-step update norms were around `1e-6`, while absolute residuals were around `1e-7`, making relative `rho` large even when the residual could be explained by parameter-dtype add/subtract roundoff.

R0b therefore asks one narrow question:

> Is the M2-R0 reference failure explained by finite-precision parameter transaction roundoff, or is there still a structural nullspace violation in the supposedly safe references?

## Frozen measurement stack

For every certificate-ranked Cell and audited step, R0b records:

1. gradient-level analytic transaction `-lr * g` after the canonical certificate projection and gradient clipping;
2. fp64 projection of that transaction onto the exact span represented by the stored certificate rows;
3. simulated parameter-dtype commit `fl(W + Delta_safe) - W`;
4. raw optimizer-realized update before any final-update repair;
5. an fp64 matched-safe projection of that optimizer update;
6. a matched-safe parameter-dtype commit at the same update scale;
7. the actually committed update after the arm's optional realized-update projection.

The fp64 certificate reference is obtained by QR-orthonormalizing the frozen stored certificate rows. QR changes numerical orthogonality but not the represented span.

## Machine-floor control

For each Cell/step, define the empirical matched-safe float-commit residual and a conservative dtype roundoff bound:

`8 * eps(dtype) * (||W||_F + ||Delta_safe||_F)`

The frozen machine-floor envelope is the maximum of:

- the empirical matched-safe float-commit residual;
- the conservative dtype roundoff bound;
- `1e-30`.

The primary excess statistic is:

`committed_excess_factor = committed_violation_norm / machine_floor_envelope`.

This prevents tiny but algebraically safe transactions from failing solely because a relative invariant divides by an update norm close to the float32 transaction floor.

## Frozen thresholds

Before the canonical R0b run:

- minimum audited nonzero Cell updates per arm: `128`
- roundoff bound multiplier: `8`
- reference p95 excess factor: `<= 2`
- reference max excess factor: `<= 4`
- fp64 matched-safe ideal rho max: `<= 1e-10`
- structural material excess p95: `>= 16`
- structural material committed rho p95: `>= 1e-4`

No threshold may be changed after observing the canonical R0b result.

## Frozen arms

The audit preserves the five M2-R0 arms and identical data order:

- `current_adamw_grad_projection`
- `adamw_no_decay_grad_projection`
- `sgd_no_decay_grad_projection`
- `sgd_with_decay_grad_projection`
- `adamw_final_update_projection`

Certificate state, router/shared parameters, topology, and growth remain frozen. Certificate updates are zero. Learner replay bytes are zero.

## Registered classifications

Primary numerical-reference classification:

- `INCONCLUSIVE_SGD_NUMERICAL_REFERENCE_FAILURE`
- `INCONCLUSIVE_FINAL_PROJECTION_NUMERICAL_REFERENCE_FAILURE`
- `R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF`

Only after both numerical references pass may the existing optimizer-mechanics decomposition be reported:

- `CURRENT_UPDATE_INVARIANT_HOLDS_AFTER_NUMERICAL_CONTROL`
- `WEIGHT_DECAY_BREAKS_UPDATE_INVARIANT`
- `ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT`
- `BOTH_PRECONDITIONER_AND_WEIGHT_DECAY_BREAK_UPDATE_INVARIANT`
- `MIXED_OR_INTERACTION_UPDATE_INVARIANT_VIOLATION`

## M2-R1 gate

M2-R1 is unblocked **only** when the primary R0b classification is:

`R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF`.

This gate means only that the protected-update measurement/mechanics layer is numerically closed. It does **not** mean that the certificate has sufficient functional coverage, and it does not reopen M2-R2 formal continual-language evaluation by itself.

## Canonical execution

Kaggle notebook:

`research/notebooks/06-native-clm/native-clm-v0-m2r0b-numerical-reference-audit-kaggle.ipynb`

Runner:

`python scripts/research/run_native_clm_v0_m2r0b.py --checkpoint <M1-final-model.pt> --data-dir <verified-m2r0-data> --device cuda`

Publisher:

`python scripts/research/publish_native_clm_v0_m2r0b.py --branch codex/native-clm-v0-m2r0b-numerical-reference-audit`

Only lightweight JSON/CSV/Markdown evidence is publishable. No checkpoint is produced by R0b.

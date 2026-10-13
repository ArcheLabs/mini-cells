# Native CLM v0 M2-R0b — Numerical Reference Audit

- Classification: `R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF`
- Mechanics diagnosis: `ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT`
- M2-R1 unblocked: `True`
- Scientific decision: `False`
- Native CLM training milestone: `False`
- New formal seeds consumed: `False`
- Protocol SHA-256: `c5dc053c7c587ebb178dccd4e012cf1f5786bf6f707c0e6d513919a56041bf54`
- Data manifest SHA-256: `7467aa08c4fd407af584910f431be7c0d2942bfd23e68fbaa58b93c54d76dfb7`

| arm | n | grad analytic p95 rho | grad float-commit p95 rho | optimizer raw p95 rho | matched-safe float p95 rho | committed p95 rho | excess p95 | excess max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_adamw_grad_projection | 512 | 7.77322e-07 | 0.015486 | 0.155416 | 5.37789e-06 | 0.155416 | 1243.63 | 1491.69 |
| adamw_no_decay_grad_projection | 512 | 7.75115e-07 | 0.015471 | 0.155417 | 5.45463e-06 | 0.155417 | 1243.63 | 1491.64 |
| sgd_no_decay_grad_projection | 512 | 7.55219e-07 | 0.0153145 | 0.0153145 | 0.0113999 | 0.0153145 | 0.0105339 | 0.0105855 |
| sgd_with_decay_grad_projection | 512 | 7.55612e-07 | 0.0152994 | 0.425481 | 0.00984636 | 0.425481 | 3.9528 | 3.96589 |
| adamw_final_update_projection | 512 | 7.77061e-07 | 0.0154942 | 0.155362 | 5.43402e-06 | 5.43598e-06 | 0.0105284 | 0.0106351 |

Boundary: M2-R0b is a numerical/optimizer-mechanics diagnostic only. It does not change the historical M2 decision and does not establish certificate coverage or continual-learning success.

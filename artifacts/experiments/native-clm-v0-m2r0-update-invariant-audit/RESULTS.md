# Native CLM v0 M2-R0 — Protected Update Invariant Audit

- Classification: `INCONCLUSIVE_REFERENCE_FAILURE`
- Scientific decision: `False`
- New formal seeds consumed: `False`
- Protocol SHA-256: `96d435f49f35e8f4baa28a4bb4c2b9aad37048d4b7565558434d0aa6aca487aa`
- Data manifest SHA-256: `7467aa08c4fd407af584910f431be7c0d2942bfd23e68fbaa58b93c54d76dfb7`

| arm | audited updates | skipped tiny/zero | rank min/max | mean rho | p95 rho | max rho |
|---|---:|---:|---:|---:|---:|---:|
| current_adamw_grad_projection | 512 | 0 | 49/60 | 0.08976 | 0.155416 | 0.24016 |
| adamw_no_decay_grad_projection | 512 | 0 | 49/60 | 0.089761 | 0.155417 | 0.240162 |
| sgd_no_decay_grad_projection | 512 | 0 | 49/60 | 0.00423737 | 0.0153145 | 0.0902347 |
| sgd_with_decay_grad_projection | 512 | 0 | 49/60 | 0.324492 | 0.425478 | 0.434843 |
| adamw_final_update_projection | 512 | 0 | 49/60 | 1.89853e-06 | 5.43655e-06 | 1.5326e-05 |

Boundary: M2-R0 audits optimizer mechanics only. It does not change the historical M2 decision and does not establish certificate coverage or continual-learning success.

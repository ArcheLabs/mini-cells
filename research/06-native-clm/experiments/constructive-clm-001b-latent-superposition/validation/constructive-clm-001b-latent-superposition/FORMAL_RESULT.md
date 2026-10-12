# Constructive CLM-001B — Formal Result

Status: **SUPPORTED**  
Scientific decision: `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`

## Frozen identity

- Formal result commit: `55071bc7fd01e7c61df02846cd8f4205b906814f`
- Protocol SHA-256: `de234ea4c75773c41012818deaba1d95622737398d5324afb8bfeea5b56063a8`
- Formal seeds: `90211`, `90212`, `90213`
- Completed seeds: 3/3
- Missing seeds: none

Canonical artifacts:

- `artifacts/experiments/constructive-clm-001b-latent-superposition/decision.json`
- `artifacts/experiments/constructive-clm-001b-latent-superposition/gate-summary.csv`
- `artifacts/experiments/constructive-clm-001b-latent-superposition/RESULTS.md`

## Formal summary

| seed | Cells | prototypes | pair MSE | triple MSE | pair recall | triple recall | key cosine | effect cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90211 | 6 | 12 | 0.000001739 | 0.000001357 | 1.0 | 1.0 | 0.999958 | 0.999997 |
| 90212 | 6 | 12 | 0.000001486 | 0.000001163 | 1.0 | 1.0 | 0.999962 | 0.999998 |
| 90213 | 6 | 12 | 0.000001333 | 0.000001328 | 1.0 | 1.0 | 0.999960 | 0.999996 |

All registered gates passed on all formal seeds. Training contained no singleton transactions. Six latent Cells were recovered from twelve pair-superposition prototypes, and x-only routing generalized to held-out unequal-weight pairs and never-seen triples.

The learned decomposition also materially beat both registered controls: nearest transaction memory and shuffled effect addressing.

## Frozen interpretation

001B closes **G1b** for the registered additive pair-superposition family:

```text
no singleton factor exposure
+ correlated/non-orthogonal latent factors
+ repeated unlabeled pair superpositions
-> relational latent decomposition
-> reusable Cell keys/effects
-> x-only unseen pair/triple addressability
```

It does **not** establish arbitrary blind source separation, unknown nonlinear mixing, language-scale latent discovery, replay-free write protection integration, or asymptotic growth behavior.

The next active gap is **G2 — Long-Horizon Growth Law**. Constructive CLM-002 must test whether Cell state tracks reusable latent structure rather than transaction count as the continual stream grows.

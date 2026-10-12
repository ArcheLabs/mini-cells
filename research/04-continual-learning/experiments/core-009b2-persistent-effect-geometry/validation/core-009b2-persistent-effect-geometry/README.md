# Core Validation 009B-2 — Persistent Effect Geometry

Status: **implementation complete; GPU discovery unrun**

Core 009B-1 established carrier causal sufficiency. 009B-2 reduces each normalized write to the natural-magnitude effect vector `a_i = Ghat_i r` in R^64 and asks whether effects admit a compact reusable shared coordinate system `a_i ≈ V beta_i` whose online incremental span stops growing.

009B-2 intentionally does not test sparse coefficients, a deployable router, certificates, continual mutation, replay, whole-model causal NLL, or a confirmed CLM architecture.

Frozen parent: result commit `f2691daf5738eac0232866a46d079db3aa61b60a`, status `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`, locked causal scale `rho=0.01`.

## Discovery

Fresh seeds: `81101, 81102`.

Fit an uncentered origin-preserving PCA/eigenspace from train effects only. Diagnostic dimensions are `1,2,4,8,12,16,24,32,40,48,56,64`; compact candidates are `1,2,4,8,12,16,24,32`.

A dimension is viable independently on both discovery seeds only if heldout median residual <= `0.25`, heldout p90 residual <= `0.50`, train→heldout median residual gap <= `0.10`, and dimension <= `32`. Choose the smallest viable dimension. Dimensions above 32 are diagnostic only.

If no compact dimension is viable: `EFFECT_GEOMETRY_DISCOVERY_NO_COMPACT_SUBSPACE`; confirmation is forbidden. A viable discovery publishes committed `basis-lock.json`.

## Confirmation

Untouched seeds: `81111, 81112, 81113`.

Start with an empty orthonormal basis. For each training effect, append its normalized residual only when `||e_i||/||a_i|| > 0.25`; never rotate/refit old coordinates. Run `canonical`, `sha-0`, `sha-1`, `sha-2` stream orders.

Every seed must pass: offline median <=0.25; offline p90 <=0.50; generalization gap <=0.10; worst online final dimension <=32; online/locked dimension <=2.0; online heldout median <=0.25; online p90 <=0.50; late-half growth <=5 coordinates/100 writes; independent-memory compression N/K >=4.

Positive: `PERSISTENT_EFFECT_GEOMETRY_SUPPORTED`. Negative: `PERSISTENT_EFFECT_GEOMETRY_NOT_SUPPORTED`.

The 64D ambient ceiling is never accepted as bounded-growth evidence.

## Kaggle

Discovery:

```bash
python scripts/research/orchestrate_core_validation_009b2.py --phase discovery --branch codex/core-validation-009b2-persistent-effect-geometry --device cuda --push-results
```

Run confirmation only after `confirmation_allowed=true`, then refresh/re-clone the branch containing `basis-lock.json`:

```bash
python scripts/research/orchestrate_core_validation_009b2.py --phase confirmation --branch codex/core-validation-009b2-persistent-effect-geometry --device cuda --push-results
```

Use `research/notebooks/04-continual-learning-core/core-009b2-persistent-effect-geometry.ipynb` for the two-cell Kaggle workflow. The runner emits stage-level heartbeat logs.

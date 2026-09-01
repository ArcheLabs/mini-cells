# Core Validation 009B-2 — Persistent Effect Geometry

Status: **implementation complete; GPU discovery unrun**

## Question

Core 009B-1 established on three untouched confirmation seeds that the train-fitted common carrier preserves roughly 98% of full-write target gain while the non-carrier residual contributes only about 2%.

009B-2 therefore reduces the write object from a 64x64 matrix to the natural-magnitude carrier effect

\[
a_i = \hat G_i r \in \mathbb{R}^{64}.
\]

This experiment asks:

> Do these effects admit a compact reusable shared coordinate system \(a_i\approx V\beta_i\), and does an online incremental basis stop growing fast enough to be a plausible persistent-memory substrate?

If the answer is no, **do not run 009B-3 addressability**.

## What this experiment does not test

009B-2 intentionally does **not** test sparse coefficients, a deployable router, certificates, continual mutation, replay, whole-model causal NLL, or a confirmed CLM architecture.

## Frozen parent

- 009B-1 result commit: `f2691daf5738eac0232866a46d079db3aa61b60a`
- parent status: `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`
- parent locked causal scale: `rho=0.01`

The runner refuses to execute if the published parent decision does not match these pins.

## Effect definition

For each sequence:

\[
G_i=\operatorname{mean}_t[(U^\top\nabla_{h_t}L)z_t^\top],
\qquad
\hat G_i=\frac{G_i}{\|G_i\|_F}.
\]

Fit the carrier from training tokens only:

\[
r=\frac{\operatorname{mean}_{train}(z)}{\|\operatorname{mean}_{train}(z)\|}.
\]

Then define \(a_i=\hat G_i r\). `a_i` is not renormalized. Primary reconstruction error is \(\|a_i-VV^\top a_i\|/\|a_i\|\).

## Discovery

Fresh seeds: `81101, 81102`.

Fit an uncentered origin-preserving PCA/eigenspace from train effects only. Diagnostic dimensions are `1,2,4,8,12,16,24,32,40,48,56,64`; only `4,8,12,16,24,32` can become a compact winner.

A dimension is viable independently on both discovery seeds only if heldout median residual <= `0.25`, heldout p90 residual <= `0.50`, train→heldout median residual gap <= `0.10`, and dimension <= `32`. Choose the smallest viable dimension.

If no compact dimension is viable, status is `EFFECT_GEOMETRY_DISCOVERY_NO_COMPACT_SUBSPACE` and confirmation is forbidden. A viable discovery publishes `basis-lock.json`; no confirmation seed may be opened before that lock is committed.

## Confirmation

Untouched seeds: `81111, 81112, 81113`.

Start with an empty orthonormal basis. For each training effect, compute \(e_i=(I-VV^\top)a_i\). If \(\|e_i\|/\|a_i\|>0.25\), append the normalized residual as a new coordinate. Existing coordinates are never rotated or refit.

Run the online algorithm under four frozen stream orders: `canonical`, `sha-0`, `sha-1`, `sha-2`.

Every confirmation seed must pass:

- offline heldout median residual <= `0.25`;
- offline heldout p90 residual <= `0.50`;
- offline train→heldout median gap <= `0.10`;
- worst online final dimension <= `32`;
- worst online final dimension / locked offline dimension <= `2.0`;
- worst online heldout median residual <= `0.25`;
- worst online heldout p90 residual <= `0.50`;
- worst late-half growth <= `5` new coordinates / 100 writes;
- worst independent-memory compression ratio \(N/K\) >= `4`.

Positive: `PERSISTENT_EFFECT_GEOMETRY_SUPPORTED`.

Negative: `PERSISTENT_EFFECT_GEOMETRY_NOT_SUPPORTED`.

## Why the growth gates are strict

The effect ambient dimension is only 64, so an unconstrained span will eventually stop growing even if every effect is unrelated. That is not CLM evidence. 009B-2 therefore requires compact heldout reconstruction, `K<=32`, `K<<N`, low late growth, and robustness across stream orders.

## Kaggle execution

Use `research/notebooks/04-continual-learning-core/core-009b2-persistent-effect-geometry.ipynb`.

Discovery:

```bash
python scripts/research/orchestrate_core_validation_009b2.py \
  --phase discovery \
  --branch codex/core-validation-009b2-persistent-effect-geometry \
  --device cuda \
  --push-results
```

Run confirmation only if discovery ends with `confirmation_allowed=true`. Refresh/re-clone the branch so the committed `basis-lock.json` is present.

Confirmation:

```bash
python scripts/research/orchestrate_core_validation_009b2.py \
  --phase confirmation \
  --branch codex/core-validation-009b2-persistent-effect-geometry \
  --device cuda \
  --push-results
```

The runner prints stage-level heartbeat messages around foundation loading, cache reuse, projection and signature extraction. After write extraction, effect geometry/growth is CPU-safe and GPU utilization may drop.

## Interpretation

A positive result supports \(a_i\approx V\beta_i\) as a compact reusable effect representation in this frozen-Pythia regime. Only then may the program open Core 009B-3 — Deployable Effect Addressability.

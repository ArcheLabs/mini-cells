# Native CLM v0 — M3L-1 Historical Address-State Capacity

M3L-1 is a **checkpoint-only mechanism diagnostic** following the canonical M3L result:

```text
QUERY_SKETCH_GATE_NOT_FEASIBLE
rank-16 sketch median AUC = 0.8968305
registered positive floor  = 0.9000000
```

It does not retrain Native CLM, update Cells/router/certificates, grow topology, or consume a new continual-learning formal seed.

## Question

The parent M3L diagnostic showed that a compact rank-16 Gaussian historical query sketch recovered most of the offline affine boundary but narrowly missed the frozen feasibility gate. M3L-1 asks:

> Is that shortfall mainly **address-state capacity/rank**, or does the whole Gaussian second-order summary family remain insufficient even at full covariance?

## Frozen capacity curve

All candidates use the exact same M3L temporal parent-lifetime ownership, sequence-group train/test split, raw edge samples, and offline logistic oracle.

```text
rank 0    diagonal Gaussian moments
rank 8    low-rank Gaussian + residual diagonal
rank 16   exact M3L parent candidate / identity anchor
rank 32
rank 64
rank 128
full       dense 384x384 covariance Gaussian LDA
oracle     raw-query linear logistic upper bound only
```

The historical address state records storage bytes for every candidate. Current-domain moments are current-stream observations and are not persistent replay state.

## Same feasibility gates as M3L

A candidate passes only if all of the original M3L gate requirements hold:

- median AUC >= 0.90;
- >=75% of edges have AUC >= 0.85;
- median normalized oracle-excess recovery >= 0.85;
- median heldout old FPR <= 0.20;
- median heldout current TPR >= 0.70;
- edge coverage and offline-oracle separability remain valid.

No threshold is relaxed because the parent M3L miss was small.

## Registered classifications

Exactly one classification is emitted:

- `INCONCLUSIVE_COVERAGE`
- `ORACLE_NOT_SEPARABLE`
- `LOW_RANK_CAPACITY_SUFFICIENT`
- `FULL_COVARIANCE_REQUIRED`
- `GAUSSIAN_FAMILY_LIMITED`

`LOW_RANK_CAPACITY_SUFFICIENT` reports the minimum passing rank in the frozen grid. `FULL_COVARIANCE_REQUIRED` means ranks through 128 fail but dense covariance passes. `GAUSSIAN_FAMILY_LIMITED` means even dense Gaussian LDA fails while the linear oracle remains separable.

## Parent identity guard

The rank-16 point must reproduce the already-published M3L aggregate within the separately frozen `identity.json` tolerance. A capacity run is rejected rather than published if the M3L rank-16 identity drifts.

## Transition reporting

Capacity curves are reported both globally and separately for:

```text
A -> B
B -> C
A+B -> C
```

This is important because the parent M3L failure was concentrated most strongly in first differentiation `A -> B`.

## Interpretation boundary

A positive low-rank/full-covariance capacity classification does **not** establish continual-language success. It only selects the historical address-state family to integrate into a future newly registered online M3L continual-language experiment.

Likewise, a Gaussian-family-limited result does not negate query separability; it says the stored historical summary is not rich enough to reconstruct the affine boundary without raw old-query replay.

Canonical protocol: [`protocol.json`](protocol.json)

Rank-16 identity contract: [`identity.json`](identity.json)

Implementation is validated by a read-only CI workflow; no CI path executes the consumed continual-language formal runners.

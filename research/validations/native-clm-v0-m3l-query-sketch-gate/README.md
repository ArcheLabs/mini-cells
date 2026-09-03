# Native CLM v0 — M3L Query-Sketch Gate Diagnostic

## Mission

The M3R Address Diagnostic found a strong local boundary in the frozen router-query representation while the actual M3R parent-vs-child cosine rule was near chance:

```text
classification               QUERY_GEOMETRY_SEPARABLE
valid lineage edges          24 / 24
current cosine median AUC    0.5315
free linear query probe AUC  0.9623
```

M3L asks the next narrower question:

> Can the useful linear boundary be recovered **without replaying old tokens or old query samples**, using only a compact historical query sketch plus the current conflict stream?

This stage is deliberately checkpoint-only. It does not train Native CLM, does not update Cell weights, router weights or certificates, does not grow new Cells, and does not consume new formal continual-learning seeds.

## Why this is not ordinary replay

For each lineage parent, historical query observations are reduced once to a fixed low-rank Gaussian sketch:

```text
count
mean
rank-16 covariance eigenvectors/eigenvalues
residual diagonal variance
```

After the sketch is built, raw historical queries are not passed to the gate estimator.

The gate uses:

```text
historical sketch moments
        +
current conflict-query moments
        ↓
regularized pooled low-rank LDA
        ↓
score(q) = w^T q + b
```

No old pseudo-samples are generated and no gradient optimization replays old representations.

For the canonical d=384 Native CLM query and rank 16 sketch, one historical address sketch is approximately 27.7 KiB in float32 storage.

## Edge-local temporal ownership

M3L does not reuse the previous diagnostic's universal `A vs birth-domain` comparison for every edge. It follows the intended lineage ownership semantics.

Examples:

```text
root4 -> child8, child8 born in B
old ownership = A
current       = B

child8 -> child12, child12 born in C
parent8 itself was born in B
old ownership = B
current       = C

root0 -> child13, child13 born in C
root0 has existed since A
old ownership = A + B
current       = C
```

Samples are conditioned on reaching the relevant parent through the frozen root and ancestor lineage decisions before the local gate is evaluated.

## Train/test isolation

A token-level random split could overstate separability because adjacent tokens in one 256-byte sequence are strongly correlated. M3L therefore splits at the `ByteSequenceDataset` sequence-row level:

```text
one source sequence -> entirely train or entirely heldout
```

The historical sketch is built only from old-domain training groups. The current query moments are built only from current-domain training groups. Heldout old/current groups are used only for evaluation.

## Registered gate

The canonical gate is a single affine decision boundary:

```text
score(q) = w^T q + b
```

`w` is derived from a pooled low-rank Gaussian covariance using the Woodbury identity. Class priors are equal. The execution threshold is calibrated using only the historical sketch distribution to target a 10% historical false-positive rate.

The same heldout groups are also evaluated with:

- the old M3R-style cosine margin;
- an offline linear logistic oracle trained with raw old/current training queries.

The oracle is a comparator only and is never used by the sketch gate.

## Registered classifications

Exactly one aggregate diagnostic classification is produced:

```text
INCONCLUSIVE_COVERAGE
EDGE_LOCAL_QUERY_GEOMETRY_NOT_SEPARABLE
QUERY_SKETCH_GATE_FEASIBLE
QUERY_SKETCH_GATE_NOT_FEASIBLE
```

`QUERY_SKETCH_GATE_FEASIBLE` requires all of the following under sufficient coverage:

```text
sketch-gate median AUC                   >= 0.90
fraction of edges with AUC >= 0.85      >= 0.75
median normalized oracle recovery        >= 0.85
median heldout historical FPR            <= 0.20
median heldout current-domain TPR        >= 0.70
```

The offline oracle must independently confirm that the edge-local temporal boundary remains linearly separable.

## Interpretation

A positive result does **not** establish continual-learning success. It establishes a much narrower mechanism fact:

```text
compressed historical address state
+
current conflict queries
can recover the local decision boundary
without old-sample replay
```

That would license the next full Native CLM experiment: maintain query sketches online, create an exact-clone child, learn the lineage-local gate from the parent sketch and current conflict stream, and only then transfer read/write ownership.

A negative result means the earlier offline query separability depended on information that the registered compact sketch cannot preserve, and another continual-language formal run should not be started yet.

## Canonical inputs

M3R lineage checkpoints are read from the already published revision:

```text
HF repo      archelabsxyz/native-clm-v0
revision     a23b521e137a7e44616809895d44d87cc7d6f87f
seeds        73611 / 73612 / 73613
```

Those seeds are already consumed M3R formal seeds. M3L does not reuse them as new formal evidence; it only performs frozen-checkpoint mechanism diagnostics.

The exact M3R A/B/C/D corpus snapshot is reconstructed and SHA-verified before execution.

## Frozen protocol

See [`protocol.json`](protocol.json). Do not change the sketch rank, ownership semantics, thresholds or classification rule after observing the canonical result.

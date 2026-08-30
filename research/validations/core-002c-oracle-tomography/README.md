# Core Validation 002C — Oracle Sparse-Assembly Representation Tomography

## Question

Core Validation 002 found that one learned latent coordinate was highly local but not sufficiently writable. Core Validation 002B tested the natural rescue hypothesis that a true functional atom might instead be a small non-oracle sparse assembly. The frozen 002B result was `SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED`: wider inferred assemblies generally worsened fidelity and leakage.

002C is the final diagnostic needed to distinguish two explanations:

1. **Inference-limited** — a good sparse assembly exists in the learned representation, but eight edit examples are insufficient to discover it.
2. **Representation-limited** — the current learned latent code does not contain a small stable linear sparse assembly for the true functional atom.

002C is evaluator-only tomography. It is not a deployable editor and it performs no sequential writes.

## Frozen parents

- Core Validation 002: `WRITE_ADDRESSABILITY_NOT_SUPPORTED`
- Core Validation 002B: `SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED`

002C cannot relabel or rescue either frozen outcome.

## Representation

The synthetic world remains unchanged:

\[
x = As,\qquad y = Vs,
\]

with \(F=512\), \(d=128\), \(k=4\), and the same learned sparse encoder used in 002/002B:

\[
z=E(x),\qquad M=1024,\qquad \operatorname{TopK}=8.
\]

The encoder is frozen after the identical sparse pretraining objective.

## Oracle tomography

Unlike 002B, the tomography evaluator is explicitly allowed to see the true sparse coefficients \(s\) and feature ids. On a large iid oracle train probe stream it accumulates

\[
G=\mathbb E[zz^\top],\qquad C=\mathbb E[zs^\top].
\]

For every true feature \(j\), deterministic Gram-matrix OMP fits

\[
h_j=z^\top a_j,\qquad \|a_j\|_0\le r,
\]

for

\[
r\in\{1,2,4,8,16\}.
\]

All reported fidelity/locality metrics use an independent held-out probe stream.

A dense linear oracle reference is also fitted:

\[
B=(G+\lambda I)^{-1}C,
\]

with frozen \(\lambda=10^{-4}\). This reference is interpretive only; it cannot make the sparse-assembly gate pass.

## Metrics

Affected functional fidelity:

\[
U_{\rm repr}(j)=
\frac{\mathbb E[(h_j-s_j)^2\mid s_j\ne0]}
{\mathbb E[s_j^2\mid s_j\ne0]}.
\]

Off-support leakage:

\[
L_{\rm repr}(j)=
\frac{\mathbb E[h_j^2\mid s_j=0]}
{\mathbb E[s_j^2\mid s_j\ne0]}.
\]

002C also records unconditional normalized MSE, context-ratio variance, featurewise success fraction, and

\[
\frac{U_{\rm repr}(r)}{U_{\rm repr}(1)}.
\]

## Frozen interpretation gate

A seed supports a sparse linear assembly only if one fixed \(r>1\) satisfies all of:

- median held-out \(U_{\rm repr}\le0.10\),
- median held-out \(L_{\rm repr}\le0.005\),
- median relative fit error versus width 1 \(\le0.60\),
- at least 50% of features jointly satisfy the fidelity/locality thresholds,
- at least 50% of features individually improve by at least 40% versus width 1.

All three frozen seeds `74201`, `74202`, and `74203` must pass.

Formal statuses:

- `ORACLE_SPARSE_ASSEMBLY_PRESENT`
- `ORACLE_SPARSE_ASSEMBLY_NOT_SUPPORTED`
- `ORACLE_TOMOGRAPHY_INVALID` only if sparse base quality invalidates the diagnostic.

Smoke runs emit only `SMOKE_ONLY` and are never scientific decisions.

## Interpretation

If sparse tomography passes, 002B was primarily an address-inference/sample-complexity failure.

If sparse tomography fails but the dense linear oracle reaches the same fidelity/locality thresholds, the true atoms remain linearly recoverable but are distributed over a wider latent subspace.

If both sparse and dense linear oracle decoders fail, the current learned \(z\) is not a stable linearly writable basis for the true atoms. At that point the 002 series should stop; the next research question is how to train writable representations, or whether transactional continual learning removes the need for write-addressability as a prerequisite.

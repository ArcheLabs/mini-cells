# Core Validation 002B — Sparse Functional Write Assemblies

## Status

002B is a new frozen mechanism test. It does **not** modify the formal result of Core Validation 002, which remains:

`WRITE_ADDRESSABILITY_NOT_SUPPORTED`

The question is narrower:

> Did 002 fail because a writable functional atom is not generally represented by one learned latent coordinate, but by a small sparse latent assembly?

## Mathematical object

Core Validation 002 used a single address `q` and a one-column writer update. 002B replaces that assumption with

\[
h(x)=z(x)^\top a,\qquad \|a\|_0\le r
\]

and the rank-1 writer update

\[
\Delta W=\delta a^\top.
\]

The induced behavioral change is

\[
\Delta \hat y(x)=h(x)\delta.
\]

The formal address widths are

\[
r\in\{1,2,4,8\}.
\]

The editor never receives the true sparse code `s`, target feature id, mixing matrix `A`, behavior matrix `V`, or evaluator affected/invariant sets.

## Non-oracle inference

For the eight edit examples, compute the current residual matrix

\[
R=Y_{edit}-\hat Y_{edit}.
\]

Because the synthetic edit is rank-1 in the ideal world, extract the dominant sample-space singular direction of `R`. Deterministic OMP then selects a support in frozen learned latent space. Alternating least squares refines the selected coefficients `a` and output direction `delta` while keeping the support fixed.

This gives one common algorithm for `r=1,2,4,8`; only the permitted support width changes.

## Representation diagnostics

Ground truth is evaluator-only and is used after the write to diagnose representation geometry.

For an inferred assembly `a`, define

\[
h=z^\top a.
\]

002B records:

1. **Assembly fit error**: best linear reconstruction error of `s_j` from `h` on unseen affected combinations.
2. **Off-support energy**:
   \[
   \frac{E[h^2\mid s_j=0]}{E[h^2\mid s_j\ne0]}.
   \]
3. **Context invariance**: variance of the fitted ratio `beta h / s_j` on affected combinations.
4. **Target correlation** between `h` and `s_j`.
5. Selected support and realized support size.

These values must never influence address inference.

## Oracle latent reference

002B includes evaluator-only `z=s` with exact target-column writes. In the additive synthetic world it must satisfy

\[
U\approx0,\qquad L\approx0.
\]

If this fails, the implementation is invalid.

## Matched global baseline

The 002 global SGD writer had median `U≈0.83`, so its low leakage was near-no-op behavior and was not a valid locality comparator.

002B replaces that decisive comparison with a pre-registered deterministic full-writer ridge curve:

\[
\Delta W^\top=\gamma Z^\top(ZZ^\top+\lambda I)^{-1}R,
\]

with frozen `lambda=1e-4` and scales

`[0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`.

For each formal seed, the best sparse-width candidate is matched to the global curve point with nearest sequence-level median `U`. If the median `U` gap is larger than `0.05`, the matched-global gate fails. This prevents an unmatched or no-op baseline from establishing locality superiority.

## Frozen formal gates

Every one of seeds `74201`, `74202`, and `74203` must pass all gates:

- sparse pretrained base normalized MSE `<= 0.20`;
- oracle latent maximum `U <= 1e-10` and maximum `L <= 1e-12`;
- best `r<=8` median `U <= 0.10`;
- best width must be greater than one and its median `U` must be `<= 0.60 ×` the `r=1` median `U`;
- best median `L <= 0.005`;
- matched-global median `U` gap `<= 0.05`;
- candidate leakage `<= 0.50 ×` matched-global leakage;
- repeated-target mean `U <= 0.20` and mean `L <= 0.01`;
- best assembly median evaluator-only fit error `<= 0.10`.

Dense MLP and standard MoE remain contextual controls. Their base quality cannot veto the sparse mechanism gate.

Formal statuses are only:

- `SPARSE_WRITE_ASSEMBLY_SUPPORTED`
- `SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED`

Smoke mode emits only `SMOKE_ONLY` with no scientific decision.

## Interpretation

A positive result would support the narrower claim

\[
\boxed{\text{a writable knowledge address can be a small sparse functional subspace}}
\]

rather than the stronger 002 assumption

\[
\boxed{\text{one functional atom}\leftrightarrow\text{one learned latent coordinate}}.
\]

A negative result in this favorable Gaussian additive world would be evidence against this frozen rank-1 sparse-assembly mechanism for `r<=8`, not against all possible addressable continual-learning architectures.

## Run

CPU execution check:

```bash
python -m pytest -q tests/test_core_validation_002b.py
python scripts/run_core_validation_002b.py --smoke --device cpu
python scripts/report_core_validation_002b.py
```

Formal CUDA run:

```bash
python scripts/run_core_validation_002b.py --device cuda
python scripts/report_core_validation_002b.py
```

Formal outputs are written to:

`results/core-validation-002b-sparse-write-assembly/`

Use the Kaggle notebook at:

`research/kaggle/core-validation-002b-sparse-write-assembly.ipynb`

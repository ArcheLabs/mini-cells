# Core Validation 005 — Replay-Free Subspace-Certified Mitosis

Status: **PROTOCOL_FROZEN_UNRUN**

## Decision question

Can bounded Cell-local subspace state replace historical replay for deciding safe writes and saturation-triggered mitosis in a controlled continual-learning system?

Core Validation 004 showed that growth can restore plasticity when an old-behavior oracle rejects unsafe local updates. Core 005 removes that oracle from the learner. The learner may retain only `(W, Q, route)` for each Cell; it may not replay or retain old examples, labels, hidden activations, or protected probes.

## Exact writable-Cell invariant

Each writable Cell is linear in its mutable parameters:

\[
y = Wz,
\]

where feature coordinates `z` are fixed during continual learning. `Q` is an orthonormal basis spanning all committed activation rows registered to that Cell. Future writes obey

\[
\Delta WQ=0.
\]

If an old activation is `z_old = Qa`, then

\[
(W+\Delta W)z_{old}=Wz_{old}
\]

exactly up to floating-point tolerance. Thus the registered historical activation set is compressed into a finite-dimensional sufficient statistic rather than replayed.

For current transaction matrix `Z` and desired residual `R`, define

\[
P_{free}=I-QQ^T,
\]

and solve

\[
\min_A \|ZP_{free}A-R\|_F^2.
\]

The normalized minimum residual is the saturation certificate. If it is below the frozen threshold, the Cell commits a projected write. If it is above threshold, the existing Cell cannot express the requested change while preserving its registered history.

## Mitosis

On an infeasible direct write to an address that has no private Cell, the candidate:

1. leaves the base Cell unchanged;
2. creates a zero-output private Cell with empty `Q`;
3. adds an exact-address monotonic route;
4. fits the current residual in the new Cell;
5. registers the current activation in both the base certificate and private certificate.

Later writes to that address update only the private Cell and are subject to its own `Q`. No second private Cell is spawned for an already-private address in this frozen experiment.

## Curriculum

The formal world has four base Cells, eight addresses per base Cell, 12 feature dimensions, three output dimensions, and 48 transactions. Every base Cell receives the same deterministic transaction template containing:

- free-space writes;
- near-saturation writes using a weak remaining free direction;
- forced collisions whose residual lies wholly inside the protected base span;
- repeated writes to previously spawned children.

The collision construction is mathematically infeasible for the protected base Cell but feasible for a new empty Cell.

## Variants

- `unsafe_always`: ignores certificates and always applies an unconstrained write.
- `certificate_no_growth`: uses the exact certificate but rejects infeasible writes.
- `certificate_growth`: primary candidate; safe projected writes plus mitosis.
- `wrong_certificate`: same mechanism and certificate rank, but `Q` is row-shifted before decisions, destroying its geometry while preserving its size.

The wrong-certificate arm is a causal control. If certificate geometry is irrelevant, it should perform like the primary candidate.

## Hidden evaluator

The evaluator retains the complete history only to measure:

- global historical regression;
- the true full-history constrained-feasibility decision;
- certificate/full-history decision equivalence.

It is never passed to the learner decision function and cannot control commit, rollback, growth, routing, or thresholds.

## Frozen formal gates

Every formal seed must independently satisfy all gates:

- learner old-sample/old-label/replay accesses = `0`;
- false-safe count = `0`;
- certificate-vs-full-history feasibility mismatches = `0`;
- regression damage `<= 0.05x unsafe_always`;
- committed new-learning gain `>= 0.95x unsafe_always`;
- growth rescue rate `>= 0.95`;
- child reuse acceptance `>= 0.95`;
- spawned Cells / effective commit `<= 0.30`;
- growth must improve committed gain over `certificate_no_growth`;
- `wrong_certificate` must produce at least one false-safe and at least `0.05` cumulative regression damage.

Formal seeds are:

```text
80501
80502
80503
```

There is no development seed, calibration, majority-vote rescue, or post-hoc threshold tuning.

Formal statuses are only:

```text
SUBSPACE_CERTIFIED_MITOSIS_SUPPORTED
SUBSPACE_CERTIFIED_MITOSIS_NOT_SUPPORTED
```

Smoke mode emits only `SMOKE_ONLY`.

## Interpretation boundary

A positive result would establish the complete replay-free mechanism only in the registered linear-writable, fixed-feature, explicit-routing setting. It would not establish that arbitrary Transformer weights have a fixed sufficient subspace certificate, that language-scale activation spans remain low-rank, that semantic routing is solved, or that missing certificates can be reconstructed from an opaque pretrained checkpoint.

The intended scientific decision is narrower: whether bounded local state can replace raw historical replay for exact registered-history write protection, saturation detection, and growth control.

## Run

CPU formal run:

```bash
python scripts/research/run_core_validation_005.py --device cpu
python scripts/research/report_core_validation_005.py
```

CPU smoke:

```bash
python scripts/research/run_core_validation_005.py --smoke --device cpu
python scripts/research/report_core_validation_005.py
```

The branch workflow runs tests and the full three-seed CPU formal experiment automatically and records formal outputs under:

```text
artifacts/experiments/core-validation-005-subspace-certified-mitosis/
```

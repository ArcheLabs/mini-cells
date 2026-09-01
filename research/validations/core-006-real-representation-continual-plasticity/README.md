# Core Validation 006 — Real-Representation Continual Plasticity

Status: **PROTOCOL_FROZEN_UNRUN**

## Decision question

Can bounded dependency-aware subspace state support replay-free continual learning on **real pretrained language-model representations** without rapid saturation or growth explosion?

Core Validation 005 established certificate → safe write → saturation → growth in a registered linear-writable synthetic world. Core 006 removes the synthetic representation while preserving the part of the architecture for which the certificate mathematics remains auditable.

The frozen experiment uses:

- `EleutherAI/pythia-160m` at `step143000`, permanently frozen;
- pinned `DKYoon/SlimPajama-6B` real text from seven sources;
- 64-D projected Pythia hidden states;
- a fixed K-means address router;
- linear writable Cells;
- per-address covariance/dependency certificates;
- dependency-partitioned mitosis;
- actual next-token NLL plus a replay baseline.

No raw dataset text is committed. Results retain source labels and SHA-256 identities only.

## Writable Cell

For frozen final Pythia hidden state

\[
h\in\mathbb R^{768},
\]

a seeded orthonormal projection

\[
U\in\mathbb R^{768\times64}
\]

defines

\[
z=U^Th.
\]

A Cell contributes

\[
\Delta h=UAz,
\]

where only

\[
A\in\mathbb R^{64\times64}
\]

is mutable. The Pythia foundation and LM head are frozen.

For current real text, standard next-token cross entropy supplies

\[
g_h=\partial L_{NLL}/\partial h,
\]

and the local functional target is

\[
R=-\eta U^Tg_h.
\]

The experiment fits a bounded local `Delta A` rather than updating foundation weights.

## Dependency-aware certificate

Every fixed address `a` accumulates only bounded state:

\[
\Sigma_a=\sum_{z\in a}zz^T,
\]

plus token/sequence counts. A Cell certificate is

\[
\Sigma_c=\sum_{a:owner(a)=c}\Sigma_a.
\]

The smallest eigenspace covering 99.5% of `Sigma_c` energy is `Q_c`. Future safe writes use

\[
P_c=I-Q_cQ_c^T,
\qquad
\Delta A=BP_c,
\]

so

\[
\Delta A Q_c=0.
\]

The learner never receives old samples for the candidate variants.

The experiment records exact rank, 99%-energy rank, 99.5%-certificate rank, participation rank, entropy rank, covariance trace, dependency tokens/sequences, address count, and

\[
\eta_c=\frac{N_{tokens,c}}{r_{participation,c}}
\]

as functional reuse density.

## Certificate conflict

For the current transaction, the experiment solves both unrestricted and safe least-squares functional writes. Their normalized residual gap is the **certificate conflict fraction**.

This distinguishes:

- inability of the Cell parameterization to express the requested update at all; from
- inability caused specifically by already-protected historical geometry.

## Dependency-partitioned mitosis

Core 006 does not attach an empty additive side Cell.

If a current address conflicts with its parent Cell certificate:

1. clone the parent's current `A` into a child;
2. move that fixed address to the child;
3. move the address covariance/dependency state with it;
4. recompute the parent's certificate from its remaining addresses;
5. recompute safe-write feasibility.

Because child `A` initially equals parent `A`, the split itself is function-preserving.

The dependency state is exactly partitioned:

\[
\Sigma_{parent}^{before}
=
\Sigma_{parent}^{after}+
\Sigma_{child}^{after}.
\]

This directly tests whether a heavily reused/high-rank Cell can regain plasticity by separating dependency sets rather than forgetting them.

V1 does not recursively subdivide an address whose owner has only one address; such saturation is recorded as `blocked_saturation`.

## Fixed real router

A disjoint bootstrap partition of SlimPajama is encoded by frozen Pythia, projected to 64-D, mean pooled and deterministically clustered into 32 addresses. The addresses are greedily load-balanced across eight base Cells.

After bootstrap:

- centroids never move;
- Pythia never moves;
- address assignment is fixed by the frozen representation;
- mitosis changes only address ownership.

Router learning and router drift are intentionally outside this experiment.

## Real continual stream

Sources are processed in this frozen order:

1. Wikipedia
2. GitHub
3. ArXiv
4. Books
5. StackExchange
6. C4
7. CommonCrawl

Per source:

- 8 router-bootstrap sequences;
- 32 continual train sequences;
- 8 disjoint heldout sequences;
- sequence length 128.

The 32 train sequences form eight transactions of four sequences, giving 56 total continual transactions.

## Variants

- `unsafe`: current data only, unrestricted Cell writes, no growth.
- `certificate_no_growth`: replay-free safe writes, no growth.
- `certificate_mitosis`: replay-free safe writes plus dependency-partitioned mitosis.
- `replay`: unrestricted writes with an explicit old-sequence buffer.

The important comparison is therefore

\[
certificate\_mitosis\stackrel{?}{\approx}replay
\]

without learner-side history.

## Hidden evaluator and causal diagnostics

The hidden evaluator may retain committed sequences and their post-commit NLL only to measure later regression. Candidate learner update functions never receive those sequences again.

At the end, each active Cell is ablated on heldout sequences currently routed to it:

\[
C_c=L_{NLL}(A_c=0)-L_{NLL}(full).
\]

The report compares routing/dependency frequency, effective rank, reuse density, and causal ablation effect. A monotonic dependency→causality relation is **not** assumed and is not a positive gate.

## Frozen gates

Every formal seed must pass all gates:

1. candidate old-sample and old-label accesses are exactly zero;
2. median 99%-energy rank at stream midpoint is at most 90% of Cell dimension;
3. median reuse density at midpoint is at least 1.25× its quarter-stream value;
4. final positive registered-history regression is at most 0.50× `unsafe`;
5. cumulative new-learning gain is at least 0.80× `replay`;
6. mitosis gains more than `certificate_no_growth`;
7. median split conflict reduction is at least 0.15;
8. spawned Cells are at most 50% of the 32 addresses;
9. at least four later transactions reuse spawned children;
10. at least one active Cell has a non-zero heldout causal ablation effect.

Formal seeds:

```text
80611
80612
80613
```

The only scientific statuses are:

```text
REAL_REPRESENTATION_CONTINUAL_PLASTICITY_SUPPORTED
REAL_REPRESENTATION_CONTINUAL_PLASTICITY_NOT_SUPPORTED
```

Formal results may not be used to retune the frozen gates.

## Interpretation boundary

A positive result would establish only that the Core 005 mechanism survives a first contact with real frozen Pythia representations and real next-token loss under a fixed router.

It would not establish safe nonlinear foundation updates, autonomous semantic routing, router drift, certificate recovery from opaque historical checkpoints, or full-scale CLM continual learning.

A negative result is intended to be actionable. Rapid effective-rank saturation, poor plasticity relative to replay, or near-linear Cell growth should be treated as a No-Go before investing in a full CLM training run.

## Run

Formal Kaggle/GPU run:

```bash
python -m pip install -e ".[lm]"
python scripts/research/run_core_validation_006.py --device cuda
python scripts/research/report_core_validation_006.py
```

Reduced real-data smoke:

```bash
python scripts/research/run_core_validation_006.py --smoke --device cuda
python scripts/research/report_core_validation_006.py
```

Formal outputs can be copied into the canonical artifact tree with:

```bash
python scripts/research/publish_core_validation_006.py
```

# Constructive CLM-001B — Latent Coordinate Discovery under Superposition

Status: **PROTOCOL_FROZEN_UNRUN**

## Why 001B exists

Constructive CLM-001 formally passed on untouched seeds `90111/90112/90113`, but its hidden factors first appeared as singleton transactions. That means a newborn Cell could be initialized from an unusually clean `(x, y)` prototype.

001B removes that scaffold.

The decision question is:

> Can the learner recover reusable latent Cell keys/effects when **no training transaction ever exposes a hidden factor alone**, and can those Cells be addressed from `x` alone on combinations never shown during discovery?

This is not another growth/certificate experiment. Core 004/005 remain reused parent mechanisms.

## Registered hidden world

Six latent factors generate correlated, non-orthogonal context and effect atoms.

Training contains only equal-weight pair superpositions:

```text
x_ij = 0.5 c_i + 0.5 c_j + noise
y_ij = 0.5 e_i + 0.5 e_j + noise
```

There are 15 possible pairs. A seeded perfect matching of three pair types is held out entirely; the remaining twelve pair types appear over eight shuffled discovery cycles.

Training therefore contains:

```text
96 transactions
12 distinct observed pair types
0 singleton transactions
0 triple transactions
```

The learner is never given:

- factor IDs;
- pair IDs;
- the hidden factor count;
- heldout support identities.

## Relational learner

001B does not create a permanent Cell from a transaction mean.

Instead it learns a small bank of stable mixture prototypes online, then uses their **overlap geometry**.

For pair midpoints, two observed mixtures that share one latent factor are systematically nearer than two disjoint mixtures. The learner therefore:

```text
current (x,y)
  -> online mixture prototypes
  -> pairwise distance classes
  -> learned overlap graph
  -> maximal star cliques
  -> learned incidence matrix B
  -> solve P ≈ B D
  -> latent joint Cell atoms D
  -> split into read keys + effect values
```

The overlap graph is the line graph of the hidden pair graph, but the learner is not handed that hidden graph. The largest maximal cliques are inferred from prototype geometry.

Each observed mixture prototype must belong to exactly two learned stars. Once that structural constraint is satisfied, latent Cell state is recovered by least squares:

```text
P ≈ B D
```

where `P` is the learned mixture-prototype matrix and `D` is the recovered Cell matrix.

## Inference boundary

Training may use `(x,y)` because this is a write/learning experiment. Heldout routing is stricter:

```text
x only
  -> choose <=3 learned Cell keys
  -> simplex least-squares coefficients
  -> apply the same coefficients to Cell effects
```

No heldout `y`, factor identity or support identity is available to the router.

## Heldout composition tests

Two tests are registered.

### Completely unseen pairs

The three heldout pair types are never shown during training. Evaluation also changes the coefficients away from the discovery-time `0.5/0.5` rule:

```text
alpha in [0.25, 0.75]
x = alpha c_i + (1-alpha)c_j
y = alpha e_i + (1-alpha)e_j
```

### Completely unseen triples

Training contains no triples. Evaluation tests all 20 triples with fresh random convex weights.

A positive result therefore requires the recovered latent coordinates to support new compositions rather than nearest-transaction recall.

## Controls

### Transaction memory

Nearest observed training context prototype predicts its associated effect prototype. This baseline can reuse transactions but cannot explicitly synthesize latent unseen compositions.

### Shuffled effect address

Context prototypes are preserved while their effect prototypes are deterministically deranged. This tests whether a stable learned read/effect relation is causally necessary.

## Formal gates

Every untouched formal seed must pass all registered gates, including:

- zero singleton training exposures;
- exactly 12 learned mixture prototypes;
- exactly 6 recovered latent Cells;
- six largest star cliques of size four;
- each mixture prototype belongs to exactly two latent stars;
- near/far distance-class ratio `>= 1.25`;
- mean matched key cosine `>= 0.98`;
- mean matched effect cosine `>= 0.98`;
- all six hidden factors covered post hoc;
- heldout pair and triple MSE `<= 0.001`;
- pair/triple route recall `>= 0.95`;
- exact support rate `>= 0.90`;
- main MSE `<= 0.10x` transaction-memory baseline;
- heldout pair MSE `<= 0.10x` shuffled-effect baseline;
- late Cell-coordinate similarity `>= 0.995`;
- transaction/Cell compression `>= 12x`.

Hidden factor identities are used only after learning for scientific alignment/recall metrics.

## Seed discipline

Observed during implementation and permanently development-only:

```text
201 202 203 204 205 206 207 208 209 210
```

Untouched formal seeds:

```text
90211 90212 90213
```

Do not run the formal seeds individually before the registered formal decision.

## Run

CPU is sufficient.

Development smoke run:

```bash
python scripts/research/run_constructive_clm_001b.py --seed 201
```

Frozen formal decision:

```bash
python scripts/research/run_constructive_clm_001b.py --formal
```

Artifacts are written to:

```text
artifacts/experiments/constructive-clm-001b-latent-superposition/
```

## Interpretation boundary

A positive result would support:

```text
no singleton exposure
  + repeated structured superposition
  -> relational latent factor discovery
  -> persistent Cell keys/effects
  -> x-only unseen composition
```

It would **not** establish arbitrary blind source separation. The registered discovery rule still assumes pairwise additive midpoint structure during discovery. Unknown arity, nonlinear mixing, language-scale representations and endogenous foundation plasticity remain open.

If positive, the next main experiment is Constructive CLM-002: long-horizon growth law. 001B should not be repeatedly strengthened after looking at formal seeds; harder superposition families belong in a separately frozen stress-test series.

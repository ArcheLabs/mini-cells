# MiniCells Continual-Learning Research Roadmap

Status: **causal funnel frozen; updated after Core 009B-2 discovery negative**  
Current evidence frontier: `EFFECT_GEOMETRY_DISCOVERY_NO_COMPACT_SUBSPACE`  
Current experiment: **Core 009C — Sparse / Local Effect Geometry**

## Mission

The research program does not assume that knowledge is naturally partitioned into semantic Cells, hard functional modes, or one small fixed global linear dictionary.

The working product hypothesis is:

\[
\boxed{\text{slow/frozen foundation}+\text{fast persistent effect memory}+\text{learned addressability}+\text{safety certificates}+\text{adaptive growth}}
\]

A later stage is not opened when an earlier prerequisite fails. Until the full chain is established, repository documentation must use **candidate CLM substrate**, not “confirmed CLM architecture”.

## Evidence established so far

1. Core 005/006 support replay-free local subspace certificates and growth-restored plasticity.
2. Core 006/007 do not support semantic or hard discrete functional Cell boundaries.
3. Core 008 rejects a small fixed shared matrix basis; individual writes are nevertheless nearly rank-1.
4. Core 009A supports raw asymmetric factor geometry.
5. The 009A right-collapse bridge shows the dominant right direction is mainly the train-token activation mean/representation carrier, not sequence averaging.
6. Core 009B-1 formally supports **carrier causal sufficiency**: with `rho=0.01` locked using full-write discovery only, 3/3 untouched confirmation seeds passed. Carrier-only writes preserved roughly 97.6–98.2% of full target gain, residual-only writes contributed roughly 1.9–2.5%, and excess unrelated harm was negligible.
7. Core 009B-2 discovery rejects a **single compact global linear carrier-effect dictionary**. On both discovery seeds, 32D left heldout median residual around 0.64–0.67; even 56D remained around 0.31. The spectrum was broad and highly reproducible. Confirmation is forbidden.

The active causal decomposition remains:

\[
G_i\approx a_i\mu^\top,\qquad a_i=\hat G_i r\in\mathbb R^{64}.
\]

What failed is the stronger hypothesis that all `a_i` share one compact global linear basis. This does **not** rule out sparse overcomplete composition, a union of local subspaces, operator-level bilinear composition of `G_i`, or engineered adaptive modules.

## Evidence funnel

### Core 009B-1 — Carrier Causal Sufficiency

**Status: formally supported.**

Result: `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`.

### Core 009B-2 — Persistent Effect Geometry

**Status: discovery negative; closed.**

Rejected hypothesis:

\[
\boxed{a_i\approx V\beta_i,\quad \dim(V)\le32}
\]

under a single train-fitted, origin-preserving global linear subspace with heldout residual/generalization gates. No confirmation may be run.

Interpretation boundary: 009B-2 rejects **compact global linear effect geometry**, not every persistent effect dictionary and not the carrier-effect object itself.

### Core 009C — Sparse / Local Effect Geometry

**Status: current frontier; discovery protocol frozen.**

009C tests two alternative explanations for the high global rank.

**H1 — sparse overcomplete composition**

\[
\boxed{a_i\approx Dc_i,\qquad \|c_i\|_0\ll K}
\]

with train-only dictionary learning, OMP coding, `K in {64,96,128}` and sparsity `s in {1,2,4,8}`. Capacity-matched deterministic random dictionaries are mandatory controls.

**H2 — union of local subspaces**

\[
\boxed{a_i\approx V_{z_i}\beta_i}
\]

with train-only spherical clustering, small origin-preserving PCA charts, and heldout assignment by train centroids only. Semantic/source labels and eval-assisted fitting are forbidden. Capacity-matched random chart assignments are mandatory controls.

A discovery-positive family must use one fixed configuration that independently passes both discovery seeds. Primary gates are heldout residual, improvement over the known 32D global baseline, improvement over matched nulls, and per-write/local complexity. Discovery cannot be rescued by confirmation-time growth.

If both families fail, return to the full operator `G_i` and freeze a new bilinear/compositional representation hypothesis before any deployable routing experiment.

### Deployable Effect Addressability — blocked until 009C positive

If 009C locks a representation, the next experiment must learn an inference-visible map from context to the locked sparse support/coefficients or local chart/coordinates. Gradients, targets, future tokens, oracle effects and replay targets are forbidden router inputs. Primary metric is causal effect recovery, not coefficient MSE.

### Certified Persistent Effect Memory — blocked until addressability positive

For locked persistent coordinates and historical dependencies, compare unsafe mutation, certificate/no-growth, certificate+growth and replay oracle. Growth is triggered only when the desired write is infeasible under existing dependency constraints.

### Transfer and Product Validation

Only after representation, deployable addressability and certified mutation are positive: test another dense modern LM, at least one MoE, multiple insertion layers, longer streams and comparisons with LoRA/adapters, replay and conventional MoE adaptation. A Pythia-only positive is not sufficient for product architecture.

## Product fallback path

Even if natural sparse/local effect structure is not supported, the already-established certificate/growth evidence keeps an engineered product path open:

\[
\boxed{\text{frozen/slow foundation}+\text{engineered sparse mutable modules}+\text{router}+\text{certificate}+\text{growth}}
\]

This path does not claim that natural Cells were discovered. It treats Cells as persistent lifecycle-managed adaptation modules whose writes are protected and whose capacity can grow when constrained plasticity is exhausted.

## Experimental discipline

- Every confirmatory result uses untouched seeds.
- Discovery and confirmation are separated by a committed lock artifact.
- Failed scientific gates are preserved; no silent seed replacement.
- Diagnostic bridges never overwrite source scientific decisions.
- Causal interventions take priority over normalized geometry when they disagree.
- Semantic/source labels cannot create a functional boundary unless the protocol explicitly tests semantics.
- No deployable router/certificate/growth mechanism is added before the representation prerequisite it depends on is confirmed.
- Ambient-dimensional saturation is not called bounded growth.
- Overcomplete dictionaries must report per-write sparsity and description-length costs; lowering residual by adding atoms is not sufficient evidence.
- Local-subspace models must route heldout effects using train-fitted information only.
- Selection/reuse counters are never primary scalability evidence.
- Online growth must be robust to multiple frozen stream orders when it becomes a scientific gate.

# MiniCells Continual-Learning Research Roadmap

Status: **carrier-effect vector route closed; operator-level compositional test frozen**  
Current evidence frontier: `SPARSE_LOCAL_EFFECT_GEOMETRY_NOT_FOUND`  
Current experiment: **Core 009D — Compositional Operator Geometry**

## Mission

The research program does not assume that knowledge is naturally partitioned into semantic Cells, hard functional modes, one small fixed global linear dictionary, or a sparse code in any representation chosen post hoc.

The working product hypothesis is:

\[
\boxed{\text{slow/frozen foundation}+\text{persistent mutable modules}+\text{learned addressability}+\text{safety certificates}+\text{adaptive growth}}
\]

A stronger research hypothesis remains open only while evidence supports a reusable natural substrate inside pretrained LMs. A later stage is not opened when an earlier prerequisite fails. Until the full chain is established, repository documentation must use **candidate CLM substrate**, not “confirmed CLM architecture”.

## Evidence established so far

1. Core 005/006 support replay-free local subspace certificates and growth-restored plasticity.
2. Core 006/007 do not support semantic or hard discrete functional Cell boundaries.
3. Core 008 rejects a small fixed shared matrix basis; individual writes are nevertheless nearly rank-1.
4. Core 009A formally supports raw asymmetric factor geometry. The locked split is `left_dim=56`, `right_dim=8`, with heldout local-action residual around 0.30–0.31 on confirmation.
5. The 009A right-collapse bridge shows the dominant right direction is mainly the train-token activation mean/representation carrier, not sequence averaging.
6. Core 009B-1 formally supports **carrier causal sufficiency**: with `rho=0.01` locked using full-write discovery only, 3/3 untouched confirmation seeds passed. Carrier-only writes preserved roughly 97.6–98.2% of full target gain, residual-only writes contributed roughly 1.9–2.5%, and excess unrelated harm was negligible.
7. Core 009B-2 discovery rejects a **single compact global linear carrier-effect dictionary**. On both discovery seeds, 32D heldout median residual was around 0.64–0.67; even 56D remained around 0.31. Confirmation is forbidden.
8. Core 009C discovery rejects the frozen **sparse overcomplete carrier-effect dictionary** and **centroid-routed local-subspace** hypotheses. Across discovery seeds, sparse dictionaries generalized poorly: the strongest `K=128,s=8` configuration fit train effects to roughly 0.21 median residual but remained around 0.64–0.65 on heldout effects and improved only a few percent over matched random dictionaries. Local chart models remained near 0.90+ heldout residual. Confirmation is forbidden.
9. The combined 009B-2/009C evidence closes the hypothesis that `a_i = Ghat_i r` itself is a useful persistent global/sparse/local organizational coordinate under the frozen linear models. Carrier causal sufficiency is therefore treated as **execution sufficiency**, not structural sufficiency.

The active operator-level observation is:

\[
\hat G_i \approx \sigma_i\,\ell_i r_i^\top,
\]

with a broad left/effect side and compact right/condition side. Core 009D tests whether the joint pair contains reusable structure that was destroyed by the carrier projection.

## Evidence funnel

### Core 009B-1 — Carrier Causal Sufficiency

**Status: formally supported.**

Result: `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`.

Interpretation: the dominant carrier is sufficient to execute most of the tested causal write. This does not imply that the carrier-compressed effect vector preserves the write's reusable identity.

### Core 009B-2 — Persistent Effect Geometry

**Status: discovery negative; closed.**

Rejected hypothesis:

\[
\boxed{a_i\approx V\beta_i,\quad \dim(V)\le32}
\]

under a single train-fitted, origin-preserving global linear subspace with heldout residual/generalization gates. No confirmation may be run.

### Core 009C — Sparse / Local Effect Geometry

**Status: discovery negative; closed.**

Rejected frozen carrier-effect hypotheses:

\[
\boxed{a_i\approx Dc_i,\qquad \|c_i\|_0\le 8,\quad K\le128}
\]

under the registered MOD+OMP dictionary learner and matched random dictionaries, and:

\[
\boxed{a_i\approx V_{z_i}\beta_i}
\]

under train-only spherical clustering, local PCA charts and centroid-based heldout assignment.

The local result must not be over-generalized to all possible union-of-subspaces algorithms: 009C specifically tested centroid-routed charts, not residual-optimized k-subspaces. The sparse result is stronger because the frozen optimizer substantially fit train effects while heldout residual remained high.

The scientific decision is not altered by later diagnostics. No 009C confirmation may be run.

### Core 009D — Compositional Operator Geometry

**Status: current frontier; discovery protocol frozen.**

009D returns to the full normalized operator:

\[
\hat G_i\in\mathbb R^{64\times64}
\]

and inherits the independently supported 009A train-only factor subspace:

\[
U_{56},\qquad V_8.
\]

It tests two pre-registered mechanisms.

#### H1 — Sparse tensor-product operator coordinates

For:

\[
C_i = U_{56}^\top \hat G_i V_8\in\mathbb R^{56\times8},
\]

test whether a compact set of coordinate pairs is enough:

\[
\boxed{
\hat G_i\approx U_{56}\,\operatorname{TopS}(C_i)\,V_8^\top,
\qquad s\in\{1,2,4,8,16,32\}
}
\]

with `s<=16` required for a compact discovery-positive result.

Because `U` and `V` are orthonormal, top-|coefficient| selection is the exact best `s`-term Frobenius approximation in this fixed 448-atom tensor-product dictionary. There is no OMP/dictionary-optimizer failure mode.

The matched null performs deterministic independent orthogonal rotations **inside the same learned 56D and 8D subspaces**. Dense projection is therefore identical under the null; any improvement isolates coordinate sparsity rather than subspace quality.

Train-only vectorized PCA of `vec(Ghat)` is reported both at active-matched dimensions and at a shared-storage-matched 1D baseline.

#### H2 — Right-conditioned operator organization

For each write's oriented rank-1 factors:

\[
\hat G_i\approx \sigma_i\ell_i r_i^\top,
\]

use only the compact right-side coordinate:

\[
x_i = V_8^\top r_i
\]

to predict the broad left-side coordinate:

\[
y_i = U_{56}^\top \ell_i.
\]

The predictor is centered ridge regression with its regularization selected by deterministic **train-only 4-fold CV**. Heldout `y_i` is never used for prediction.

Baselines are:
- mean train left factor;
- nearest train right-address neighbor;
- deterministically permuted train left targets passed through the same ridge/CV procedure.

This is intentionally an **operator-structure oracle**, because the heldout right factor is extracted from the heldout `Ghat`. A positive result would show that right-side variation contains organizational information lost by `Ghat r`; it would not yet establish deployable routing.

#### Rank-1-in-factor-subspace compression guard

009D separately measures:

\[
C_i\rightarrow \operatorname{rank1}(C_i).
\]

If the 56x8 factor core can be reduced to rank-1 with no material additional heldout residual, a write can be represented by roughly `56+8+1=65` factor coefficients rather than a dense 448-coefficient core. This is useful operator compression but **does not count as cross-write reuse** and cannot by itself open confirmation.

#### 009D decision hierarchy

1. Rank-1 core guard must pass on both discovery seeds.
2. Prefer the smallest fixed sparse tensor `s<=16` that passes every frozen gate on both seeds.
3. Only if sparse tensor reuse fails may the single frozen right-conditioned procedure be locked, and only if it passes all gates on both seeds.
4. Rank-1 compression alone produces `OPERATOR_FACTOR_COMPRESSION_ONLY` and stops.
5. If no reusable family passes, confirmation and deployable routing are forbidden.

Discovery seeds: `81301, 81302`.  
Confirmation seeds (only after committed lock): `81311, 81312, 81313`.

### Deployable Operator Addressability — blocked until 009D reusable positive

If sparse tensor coordinates are locked, the next experiment must predict the locked coordinate-pair support/coefficients using inference-visible context only.

If right-conditioned organization is locked, the next experiment must first recover the right/address coordinate from inference-visible context without gradients, targets, future tokens or oracle operators.

Primary metrics must include causal operator/effect recovery; coefficient prediction alone is insufficient.

### Certified Persistent Operator Memory — blocked until addressability positive

For locked persistent operator coordinates and historical dependencies, compare unsafe mutation, certificate/no-growth, certificate+growth and replay oracle. Growth is triggered only when the desired write is infeasible under existing dependency constraints.

### Transfer and Product Validation

Only after representation, deployable addressability and certified mutation are positive: test another dense modern LM, at least one MoE, multiple insertion layers, longer streams and comparisons with LoRA/adapters, replay and conventional MoE adaptation. A Pythia-only positive is not sufficient for product architecture.

## Product fallback path

Even if 009D finds only compression or is fully negative, the already-established certificate/growth evidence keeps an engineered product path open:

\[
\boxed{\text{frozen/slow foundation}+\text{engineered rank-1/low-rank mutable modules}+\text{router}+\text{certificate}+\text{controlled growth}}
\]

This path does not claim that natural Cells were discovered. It treats Cells as persistent lifecycle-managed adaptation modules whose writes are protected and whose capacity can grow when constrained plasticity is exhausted.

A rank-1 operator representation is still useful in this fallback because it reduces the state required for each engineered write, even when there is no cross-write natural dictionary.

## Experimental discipline

- Every confirmatory result uses untouched seeds.
- Discovery and confirmation are separated by a committed lock artifact.
- Failed scientific gates are preserved; no silent seed replacement.
- Diagnostic bridges never overwrite source scientific decisions.
- Causal interventions take priority over normalized geometry when they disagree.
- Semantic/source labels cannot create a functional boundary unless the protocol explicitly tests semantics.
- No deployable router/certificate/growth mechanism is added before the representation prerequisite it depends on is confirmed.
- Carrier causal sufficiency is never treated as proof that the carrier-compressed effect is a sufficient organizational representation.
- Ambient-dimensional saturation is not called bounded growth.
- Sparse representations must report per-write active coordinates and description-length costs; lowering residual by increasing representation capacity is not sufficient evidence.
- Matched operator nulls must preserve the relevant shared factor subspace whenever the hypothesis concerns coordinates inside that subspace.
- Right-conditioned 009D evaluation may use an oracle-extracted heldout right factor only as a representation diagnostic and must never be described as deployable routing.
- Rank-1-per-write compression is not evidence of cross-write reuse.
- Selection/reuse counters are never primary scalability evidence.
- Online growth must be robust to multiple frozen stream orders when it becomes a scientific gate.

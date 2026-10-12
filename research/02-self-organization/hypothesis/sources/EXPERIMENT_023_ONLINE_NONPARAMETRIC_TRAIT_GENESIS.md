# Experiment 023 — Online Nonparametric Trait Genesis

## Motivation

Experiment 022 produced `EMERGENT_TRAIT_BIFURCATION_SIGNAL`: a task-label-free two-mode phenotype-gradient field was stable in 9/9 calibration windows, geometry-driven descendants formed strong functional identity in 3/3 replicates, and a strictly stratified capacity control formed identity in 0/3.

The remaining structural prior is important: Experiment 022 still compared `K=1` with a researcher-specified `K=2` and then performed a researcher-specified fork. A developmental organism should not be told how many traits exist or when to create them.

Experiment 023 therefore asks whether one persistent TextNCA organism can perform online structural model selection and permanently create new phenotype traits only when the reduction in developmental distortion is worth the structural complexity cost.

The causal chain under test is:

```text
online phenotype-gradient field
        -> penalized unknown-K model selection
        -> persistent mode evidence
        -> endogenous trait genesis
        -> functional identity
```

This experiment deliberately does **not** add rewiring, merging, pruning, inference-time recruitment, or a local replacement for the gradient oracle. Those would confound the structural question.

## Question

Can one TextNCA organism decide online **when** and **how many** persistent phenotype traits to create from an unlabeled developmental field, while avoiding false growth on a unimodal stream and avoiding over-segmentation when one distribution is given two posthoc labels?

## Fixed substrate inherited from 021/022

The shared language genome is the validated TextNCA configuration:

- context 128;
- hidden dimension 128;
- four attention heads;
- FFN 512;
- windows `[8, 32, 128]`;
- recurrent iterations `[4, 4, 4]`;
- LayerNorm;
- GRU carry bias `+2`;
- tied embedding/head.

The first two NCA stages are a shared stem. A 128-dimensional phenotype vector is broadcast before the final shared NCA stage. The shared genome is frozen during online development.

Experiment 023 reserves at most four latent phenotype slots for bounded evaluation, but only the first `K_active` are physically active. The experiment begins with:

\[
K_{active}=1.
\]

The bounded maximum is an evaluation safety bound, not the selected model order.

## Shadow developmental sensor

Each incoming microbatch produces a phenotype gradient at the frozen parent phenotype:

\[
g_t = \frac{\partial L_t}{\partial m_{parent}}.
\]

The normalized gradient is:

\[
\hat g_t = \frac{g_t}{\lVert g_t\rVert+\epsilon}.
\]

The latest 96 gradients form a rolling developmental buffer:

\[
\mathcal G_t = \{\hat g_{t-95},\ldots,\hat g_t\}.
\]

The sensor is evaluated every 32 online steps.

The fixed parent-gradient sensor is intentionally an oracle-quality shadow sensor. It keeps gradients from different times and descendants in one comparable coordinate system. Replacing it with a purely local cellular observable is outside Experiment 023.

## Unknown-K structural model selection

For each sensor window, deterministic farthest-first K-means is fit for:

\[
K\in\{1,2,3,4\}.
\]

Let:

\[
R_K = \min_{\mu_1,\ldots,\mu_K,z_i}
\sum_i \lVert \hat g_i-\mu_{z_i}\rVert^2.
\]

Model order is selected using:

\[
\boxed{
J_K = \frac{R_K}{R_1} + \lambda_{structure}(K-1)
}
\]

with preregistered:

\[
\lambda_{structure}=0.20.
\]

This value is inherited from Experiment 022's minimum meaningful bifurcation-gain scale: a new mode should explain roughly 20% additional single-mode distortion before paying for another persistent trait.

Clusters containing less than 12% of the sensor window are structurally invalid and receive a prohibitive complexity score.

The selected order is:

\[
K^*_t = \arg\min_K J_K.
\]

No task/domain label enters `R_K`, `J_K`, cluster fitting, model-order selection, or proposed branch routing.

## Hysteresis / persistent evidence

A single `K^*_t > K_active` observation cannot grow the organism.

The candidate next order is:

\[
K_{candidate}=K_{active}+1.
\]

Its centroid set must remain stable across three consecutive sensor evaluations. For two equal-cardinality centroid sets, cluster permutation is optimized away and stability is the mean cosine similarity of aligned centroids.

Preregistered threshold:

\[
S_{mode}\ge0.65.
\]

A permanent genesis event occurs only after:

\[
3
\]

consecutive stable evaluations supporting additional structure.

Thus the organism cannot permanently grow from a single noisy minibatch or a single transient K-means partition.

## Genesis mechanics

### `1 -> 2`

The first bifurcation is initialized from the existing phenotype using the discovered two-mode axis:

\[
m_0 = m + \epsilon v,
\qquad
m_1 = m - \epsilon v,
\]

with:

\[
\epsilon=0.02.
\]

### `K -> K+1`, for `K >= 2`

New centroids are aligned to existing mode order by minimum total centroid distance. Exactly one centroid remains unmatched. The newborn phenotype copies the nearest existing phenotype and receives a small symmetry-breaking displacement toward the new centroid direction.

Existing phenotype identities are not reset when a later trait is born.

## Online routing after genesis

Once multiple traits exist, each incoming shadow gradient is assigned to the nearest currently aligned mode centroid. The corresponding phenotype receives the learning update.

No domain ID, task ID, expert ID, or stage ID enters this routing function.

Stage/domain labels exist only in the benchmark harness for constructing controlled streams and for posthoc scientific validation.

## Developmental curriculum

The organism sees one continuous online process. There is no model reset between stages.

### Stage A — `A_STORY_ONLY`

192 online steps of TinyStories only.

Required behavior:

\[
K:1\rightarrow1.
\]

Any genesis is a false positive.

### Stage B — `B_EMERGING_MATH`

Arithmetic appears gradually without exposing a boundary signal to the model.

Subphases:

- 128 steps: 115 STORY + 13 ARITHMETIC (~10% arithmetic);
- 128 steps: 90 STORY + 38 ARITHMETIC (~30% arithmetic);
- 128 steps: 64 STORY + 64 ARITHMETIC (50% arithmetic).

Required behavior:

\[
K:1\rightarrow2.
\]

The resulting two active traits must exhibit functional Story/Arithmetic identity.

### Stage C — `C_DUPLICATE_CONTROL`

192 steps:

- 64 STORY;
- 64 `ARITH_A`;
- 64 `ARITH_B`.

`ARITH_A` and `ARITH_B` are separate schedules over the **exact same deterministic arithmetic token distribution**. They are both collapsed to family `ARITHMETIC` for every posthoc structural analysis.

This is a hard over-segmentation control.

Required behavior:

\[
K:2\rightarrow2.
\]

The model must not create a third trait merely because the benchmark harness assigns two names to one distribution.

### Stage D — `D_THIRD_MODE`

384 steps:

- 128 STORY;
- 128 ARITHMETIC;
- 128 TRANSFORM.

TRANSFORM is a deterministic six-digit reverse-sequence language corpus encoded with the same tokenizer.

Required behavior:

\[
K:2\rightarrow3.
\]

The resulting three active traits must exhibit permutation-invariant functional identity across Story, Arithmetic, and Transform.

## Functional identity

For a stage containing `D` genuine capability families and at least `D` active traits, evaluate every family on every active phenotype.

The optimal one-to-one domain/branch assignment minimizes total NLL. For every matched domain, define the margin against its best alternative branch and normalize by the frozen-parent baseline NLL.

A stage passes identity only when all matched normalized margins satisfy:

\[
M_d^{norm}\ge0.01.
\]

Branch numbering is permutation-invariant.

## Posthoc routing purity

Task labels are allowed only after training for scientific validation.

For each stage after genesis, branch/family purity is computed by assigning each active branch its majority posthoc family. Required purity:

\[
\ge0.75.
\]

## Replicates

Three independent replicates are used.

Up to two Kaggle GPUs run replicates concurrently.

## Strong positive criteria

`ONLINE_NONPARAMETRIC_TRAIT_GENESIS_SIGNAL` requires all of the following:

1. **Unimodal null:** Stage A has no genesis and ends at `K=1` in **3/3** replicates.
2. **Two-mode genesis:** Stage B ends at `K=2`, contains exactly one `1->2` genesis, passes two-family functional identity, and passes routing purity in **>=2/3** replicates.
3. **Duplicate negative control:** Stage C has no genesis and remains at `K=2` in **3/3** replicates.
4. **Three-mode genesis:** Stage D ends at `K=3`, contains exactly one `2->3` genesis, passes three-family functional identity, and passes routing purity in **>=2/3** replicates.
5. **No overgrowth:** no replicate ever reaches `K=4`.

## Decision statuses

- `ONLINE_NONPARAMETRIC_TRAIT_GENESIS_SIGNAL`
- `FALSE_POSITIVE_GENESIS_ON_UNIMODAL_STREAM`
- `DUPLICATE_MODE_OVERSEGMENTATION`
- `TWO_MODE_GENESIS_WITHOUT_THIRD_MODE`
- `THIRD_MODE_WITHOUT_STABLE_TWO_MODE`
- `NO_ONLINE_TRAIT_GENESIS`

## Checkpointing

Each replicate stores:

- one parent checkpoint;
- one checkpoint after each of the four developmental stages.

Expected total:

\[
3\times5=15
\]

checkpoints.

Stage checkpoints include active phenotype state, optimizer state, rolling sensor buffer, current centroid alignment, structural-evidence hysteresis state, routing history, genesis history, and evaluation history. This allows stage-level recovery without retraining prior development.

The `.pt` files remain Kaggle-local. Curated CSV/JSON/PNG artifacts are publishable.

## Interpretation boundaries

A positive result would support:

> Under this controlled TextNCA substrate, an unlabeled phenotype-gradient field plus an explicit structural complexity cost can online-select an unknown number of persistent functional traits and trigger their genesis without task IDs.

It would **not** yet prove:

- a purely local NCA developmental sensor;
- autonomous topology rewiring;
- automatic pruning or merging;
- inference-time expert recruitment;
- general scaling to natural multi-domain corpora;
- that the synthetic Transform domain is a pure computational trait rather than a mixture of computation and distributional style.

Those are subsequent questions.

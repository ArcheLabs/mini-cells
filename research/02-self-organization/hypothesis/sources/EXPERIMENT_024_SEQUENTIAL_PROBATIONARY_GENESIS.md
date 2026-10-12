# Experiment 024 — Sequential Probationary Genesis

## Research question

Can one continuous TextNCA organism repeatedly reject or commit temporary task-label-free trait births, preserving accepted descendants and their optimizer history, so that a single organism grows from one persistent phenotype to two and then three only when future utility pays for each additional trait?

Experiment 022 established that task-label-free phenotype-gradient geometry can induce strong functional differentiation beyond a strictly balanced capacity control. Experiment 023 showed that hard discrete-cluster validity is too conservative as an online birth criterion. Experiment 023b then established a cleaner local rule: a temporary two-child geometry fork can be accepted or rejected by prospective prequential utility, rejecting Story-only and duplicated-Story controls while accepting Story+Arithmetic in 3/3 replicates.

Experiment 024 tests the missing sequential claim. The organism is never reset between developmental stages. A committed Geometry shadow becomes the incumbent for the next stage, including its phenotype state and Adam moments. A rejected proposal is discarded and the continuously adapted incumbent survives.

## Scope

024 deliberately tests only repeated birth valuation and persistence.

It does **not** claim:

- a purely local NCA developmental sensor;
- dynamic physical memory allocation (a bounded latent trait pool is preallocated for failure diagnostics);
- topology rewiring;
- pruning or merging;
- inference-time recruitment;
- end-to-end language quality competitive with large LLMs.

The shared TextNCA genome remains frozen after Story pretraining. Only persistent phenotype vectors adapt during the developmental sequence.

## Organism

The model reuses `OnlineTraitTextNCA` from Experiment 023. The active organism at stage `s` has:

- one frozen TextNCA genome;
- one fixed pretrained parent phenotype used only as the shadow gradient sensor;
- `K_s` active persistent phenotype vectors;
- an AdamW state for the phenotype matrix;
- an ordered set of active gradient-space centroids used for task-label-free microbatch routing.

Inactive phenotype slots have no forward or update cost. Up to four slots are allocated so an erroneous early birth can be recorded rather than crashing the experiment. A positive Experiment 024 result must still end at exactly `K=3`.

## Sequential stages

Each stage has 64 proposal probes followed by four chronological 64-step probation windows (256 online updates). The model receives no stage/task boundary token.

### A — Story null

- 256 Story updates.
- Start: `K=1`.
- A temporary `1 -> 2` proposal is intentionally opened.
- Preregistered result: reject and remain `K=1`.

### B — Arithmetic birth

- 128 Story + 128 Arithmetic updates.
- Expected start: `K=1`.
- Temporary `1 -> 2` proposal.
- Preregistered result: accept, form Story/Arithmetic identity, end `K=2`.

### C — Duplicate Arithmetic negative control

- 128 Story + 64 `ARITH_A` + 64 `ARITH_B`.
- `ARITH_A` and `ARITH_B` are independent schedules over the exact same arithmetic token distribution.
- Expected start: `K=2`.
- Temporary `2 -> 3` proposal.
- Preregistered result: reject, preserve two-trait identity, remain `K=2`.

### D — Weak Transform negative control

- 115 Story + 115 Arithmetic + 26 Transform updates (~10% Transform).
- Expected start: `K=2`.
- Temporary `2 -> 3` proposal.
- Preregistered result: reject, preserve Story/Arithmetic identity, remain `K=2`.

This tests whether an incipient but still low-value third pressure is enough to justify permanent structural cost.

### E — Transform birth

- 86 Story + 85 Arithmetic + 85 Transform updates.
- Expected start: `K=2`.
- Temporary `2 -> 3` proposal.
- Preregistered result: accept, form three-way Story/Arithmetic/Transform identity, end `K=3`.

The full expected committed trajectory is therefore:

```text
START  A  B  C  D  E
  1 ->1->2->2->2->3
```

## Proposal geometry is not the birth gate

For a stage beginning with `K` active traits, the fixed pretrained sensor produces normalized phenotype gradients on 64 proposal microbatches.

We fit:

- a `K`-mode field for incumbent routing;
- a `(K+1)`-mode field for the temporary candidate.

For `K>1`, incumbent centroids are aligned to the previous stage's trait order. The `(K+1)` candidate is aligned to the `K` incumbent centroids, identifying exactly one unmatched newborn mode and its nearest existing parent trait.

No silhouette threshold, model-order selection, task ID, or semantic label decides birth. Residual gain, cluster fraction, and posthoc purity are diagnostics only.

## Three aligned probation arms

Every stage forks three counterfactual trajectories from the exact same committed organism state and optimizer state.

### Incumbent

The current `K` traits continue online adaptation. Each microbatch is routed to the nearest aligned incumbent centroid using only the frozen-parent phenotype gradient.

### Capacity shadow

A `(K+1)`th phenotype is initialized identically to the Geometry candidate. Existing routes are unchanged except when the incumbent routes a microbatch to the specific parent trait selected for division. Only those parent-bound microbatches may alternate between the old parent and newborn.

The control uses benchmark stream identity only to keep the parent/newborn split locally balanced. It is not a candidate routing mechanism.

### Geometry shadow

The same newborn initialization is used, but each microbatch is routed to the nearest of the `(K+1)` proposal centroids. Neither routing nor birth uses task/domain labels.

## Newborn optimizer inheritance

A new phenotype must not gain or lose merely because it has a younger optimizer.

For each candidate arm:

- existing Adam state is cloned from the incumbent;
- the newborn phenotype is initialized from the selected parent trait plus the proposal direction;
- the newborn row inherits the selected parent's Adam first and second moments when those moments already exist.

For the first `1 -> 2` bifurcation, both descendants therefore share the parent's optimizer history.

## Prospective utility

At every online step, each arm is evaluated on the current microbatch **before** it updates on that microbatch.

For probation window `w`:

\[
U_w^{candidate}
=
\frac{L_w^{incumbent}-L_w^{candidate}}
{|L_w^{incumbent}|+\epsilon}
-0.005.
\]

The 0.5% term is the explicit structural cost inherited from Experiment 023b.

Geometry is accepted only if:

1. the final two windows have positive net utility;
2. the mean Geometry net utility of the final three windows is positive;
3. the mean Geometry-minus-Capacity advantage of the final three windows is at least 0.005.

The commit rule never reads expected stage outcome, semantic family, task label, or posthoc identity.

## Sequential commit semantics

This is the central change from 023b.

If the Geometry shadow is accepted:

- its final phenotype matrix becomes the next organism;
- its final Adam state becomes the next organism optimizer;
- its `(K+1)` centroids become the active routing geometry;
- `K <- K+1`.

If rejected:

- Geometry and Capacity shadows die;
- the continuously adapted Incumbent becomes the next organism;
- incumbent centroids remain active;
- `K` is unchanged.

Thus all five stages form one causal developmental history.

## Functional identity

After every stage the committed organism is evaluated on held-out Story, Arithmetic, and Transform streams.

When `K=2`, permutation-invariant Story/Arithmetic identity must satisfy the existing normalized per-domain margin threshold (>=1% parent NLL).

When `K>=3`, three distinct branches must minimize Story, Arithmetic, and Transform losses with all normalized margins >=1% parent NLL.

Identity is posthoc scientific validation only; it does not enter the commit decision.

## Routing purity

Geometry-shadow probation routes are compared with semantic labels only after the run.

The preregistered threshold remains:

\[
\text{purity} \ge 0.75.
\]

For the first birth this is two-family purity. For the final birth it is three-family purity.

## Strong positive gate

`SEQUENTIAL_PROBATIONARY_GENESIS_SIGNAL` requires:

1. Stage A Story-null rejection in 3/3 replicates.
2. Stage B `1 -> 2` Arithmetic birth with functional identity and routing purity >=0.75 in at least 2/3 replicates.
3. Stage C duplicate-Arithmetic rejection with retained Story/Arithmetic identity in 3/3 replicates.
4. Stage D weak-Transform rejection with retained Story/Arithmetic identity in 3/3 replicates.
5. Stage E `2 -> 3` Transform birth with three-way functional identity and routing purity >=0.75 in at least 2/3 replicates.
6. At least 2/3 replicates end at exactly `K=3`.

Preregistered statuses:

- `SEQUENTIAL_PROBATIONARY_GENESIS_SIGNAL`
- `FALSE_POSITIVE_SEQUENTIAL_BIRTH`
- `NO_FIRST_PROBATIONARY_BIRTH`
- `DUPLICATE_SIGNAL_CAUSES_EXTRA_BIRTH`
- `WEAK_SIGNAL_CAUSES_EARLY_BIRTH`
- `FIRST_BIRTH_WITHOUT_SECOND_TRAIT_GENESIS`

The aggregate order is conservative: false-positive or unnecessary early births dominate a later successful birth.

## Replicates and compute

- 3 replicates.
- Up to 2 Kaggle GPUs in parallel.
- 300 Story pretraining steps per replicate.
- 5 sequential stages.
- 256 probation updates/stage/arm.
- Genome frozen during all five stages.

Each replicate stores:

- one parent checkpoint;
- five stage checkpoints, each containing only the committed organism state/Adam state plus the counterfactual measurements required for auditing.

Expected Kaggle-local checkpoint count:

\[
3 \times (1 + 5) = 18.
\]

Model checkpoints are intentionally excluded from GitHub publication.

## Provenance

024 requires merged curated decisions from:

- Experiment 022: `EMERGENT_TRAIT_BIFURCATION_SIGNAL`;
- Experiment 023: corrected `NO_ONLINE_TRAIT_GENESIS`;
- Experiment 023b: `PROBATIONARY_TRAIT_GENESIS_SIGNAL`.

This preserves the research chain:

```text
022: geometry can make functional traits
023: hard cluster validity is not a good online birth gate
023b: one candidate birth can prove prospective value
024: can the same organism repeat that process sequentially?
```

## Interpretation boundaries

A strong positive would support the following limited claim:

> A persistent TextNCA phenotype population can sequentially acquire additional functional traits when temporary task-label-free geometry proposals demonstrate sustained prospective utility beyond matched extra capacity, while rejecting redundant and weak proposals.

It would **not** yet establish a fully autonomous growing MoE. The remaining major steps would be local developmental sensing, topology differentiation, pruning/merging, and inference-time endogenous recruitment.

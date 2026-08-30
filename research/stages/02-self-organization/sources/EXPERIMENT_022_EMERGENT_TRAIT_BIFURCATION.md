# Experiment 022 — Emergent Trait Bifurcation

## Motivation

Experiment 021 produced a strong functional split under task-label-free gradient routing, but it did **not** validate the narrower preregistered hypothesis that differentiation requires persistent bidirectional destructive interference.

021 instead showed:

- differentiation identity in 3/3 replicates;
- mean routing purity about 0.93;
- a much larger identity margin under geometry-aware routing than under the capacity control;
- destructive two-way interference in only 1/9 calibration windows;
- weak capacity-only identity in 2/3 replicates.

This suggests that the relevant developmental signal may be broader than negative gradient cosine. A cell may need to differentiate when its local learning field becomes persistently **multimodal**, even if the modes share a useful common component.

Experiment 022 therefore tests:

```text
persistent multimodal learning field
    -> bifurcation
    -> differential experience
    -> functional trait identity
```

The experiment remains deliberately minimal. It uses the fixed-topology TextNCA substrate and does not introduce autonomous topology rewiring or inference-time recruitment.

## Question

Can a story-trained TextNCA detect that its phenotype-gradient field is better explained by two stable modes than one, fork a shared-genome phenotype into two descendants, and develop functional identity beyond a strictly stratified capacity control — without using task labels in the trigger or geometry branch selection?

## Domains

The benchmark remains the 021 dual-ability setup:

1. `STORY`: TinyStories token stream.
2. `ARITHMETIC`: deterministic textual add/subtract/multiply/solve-x examples.

Both domains use the same TinyStories byte-level BPE tokenizer.

Task/domain labels exist only because this is a scientific benchmark. They are not supplied to the multimodality estimator or the geometry routing mechanism.

## Developmental sequence

### Phase 1 — one parent population

Train one TextNCA parent on STORY for 300 steps using the validated carry-biased recurrent language architecture.

The model has one persistent 128-dimensional population phenotype `m` injected before the final shared NCA stage.

### Phase 2 — measure the local learning field

Introduce mixed STORY and ARITHMETIC calibration microbatches.

For every microbatch compute only:

```math
g_k = \frac{\partial L_k}{\partial m}.
```

Normalize each gradient direction:

```math
\hat g_k = \frac{g_k}{\|g_k\|+\epsilon}.
```

The bifurcation estimator receives only the unlabeled set `{\hat g_k}`.

## K=1 versus K=2 gradient-field model

### One mode

Fit the centroid:

```math
\mu = \frac{1}{K}\sum_k \hat g_k
```

and residual:

```math
R_1 = \sum_k \|\hat g_k-\mu\|^2.
```

### Two modes

Fit two centroids with deterministic PCA-seeded Lloyd updates:

```math
R_2 = \min_{\mu_1,\mu_2,z_k}
\sum_k \|\hat g_k-\mu_{z_k}\|^2.
```

Define bifurcation gain:

```math
D = \frac{R_1-R_2}{R_1+\epsilon}.
```

This generalizes the 021 conflict criterion. Two modes may be opposing, orthogonal, or share a large common component.

## Window preregistration

Use three independent calibration windows per replicate.

A window supports bifurcation when:

```text
K=2 residual-fit gain >= 0.20
smaller cluster fraction >= 0.25
```

The fitted two-centroid axis is:

```math
v = \frac{\mu_1-\mu_2}{\|\mu_1-\mu_2\|}.
```

Because cluster numbering is arbitrary, cross-window stability is measured with absolute cosine:

```math
S_{ab}=|v_a^T v_b|.
```

Use the minimum pairwise window stability.

A replicate has **persistent multimodality** only if:

- at least 2/3 windows pass gain and balance;
- minimum pairwise axis stability >= 0.50;
- the combined calibration field also passes the gain/balance gate.

STORY/ARITHMETIC labels are used only afterward to report cluster purity. Purity is not part of the trigger.

## Fork initialization

Both fork controls receive the same symmetry break around the parent phenotype:

```math
m_0=m+0.02v
```

```math
m_1=m-0.02v.
```

The TextNCA genome is frozen after fork. Only the two 128-dimensional child phenotypes learn.

No model weights are duplicated.

## Arms

### 1. Unified

One phenotype receives all 400 mixed-domain updates.

This measures the non-forked baseline.

### 2. Stratified Capacity Fork

Two children receive the same initialization and the same one-child-per-microbatch update budget as the geometry fork.

This is a deliberately stronger negative control than Experiment 021.

The benchmark domain labels are used **only inside this control** to enforce exact exposure matching:

```text
child 0: 100 STORY + 100 ARITHMETIC updates
child 1: 100 STORY + 100 ARITHMETIC updates
```

Within each domain, assignment alternates deterministically between children.

This control is not a proposed routing mechanism. Its sole purpose is to remove finite-sample domain imbalance as an explanation for spontaneous capacity specialization.

### 3. Geometry Bifurcation Fork

For each new training microbatch:

1. compute its phenotype gradient at the frozen parent state;
2. normalize the gradient;
3. measure distance to the two fixed unlabeled calibration centroids;
4. update only the nearest child.

No task label, domain ID, expert ID, or learned router is supplied.

## Posthoc routing purity

After training, compare discovered branch assignments with benchmark domains only for scientific interpretation.

Purity is permutation-invariant:

```math
P=\max(P_{direct},P_{swapped}).
```

Routing PASS requires:

```text
purity >= 0.75
```

in at least 2/3 replicates for a strong positive result.

## Functional identity

Evaluate both fork children on both domains.

For each replicate form the 2 x 2 NLL matrix:

```text
                child 0   child 1
STORY              ...       ...
ARITHMETIC         ...       ...
```

Child numbering is permutation-invariant.

Functional identity requires:

- STORY and ARITHMETIC prefer opposite children;
- each normalized preference margin is at least 1% of the parent-domain NLL.

The identity margin is the mean of the two normalized margins.

## Geometry advantage over capacity

Experiment 021 showed that a weak capacity identity can emerge from finite-sample experience differences.

022 therefore preregisters an additional paired requirement:

```math
A_r = M^{geometry}_r-M^{capacity}_r
```

where `M` is normalized identity margin.

A replicate has meaningful geometry advantage only if:

```text
A_r >= 0.05
```

A strong positive result requires this in at least 2/3 replicates.

## Combined quality

Report:

- unified mean domain NLL;
- oracle best-child mean domain NLL for each fork arm;
- parent NLL;
- phenotype-distance growth.

The oracle best-child score is diagnostic only. Experiment 022 does not claim an inference-time recruitment mechanism.

## Hard invariants

1. All K=1/K=2 metrics must be finite.
2. Exactly 3 calibration windows per replicate.
3. The stratified capacity fork must give each child exactly the same number of STORY updates.
4. The stratified capacity fork must give each child exactly the same number of ARITHMETIC updates.
5. Both fork arms use the same parent checkpoint, symmetry-break axis, epsilon, post-fork steps, LR, and one-child-per-microbatch update budget.
6. Geometry trigger and geometry routing do not receive task labels.
7. The shared TextNCA genome is frozen after fork.

## Decision statuses

### `EMERGENT_TRAIT_BIFURCATION_SIGNAL`

Requires all of:

- persistent multimodality in >=2/3 replicates;
- geometry functional identity in >=2/3;
- geometry routing purity PASS in >=2/3;
- geometry identity-margin advantage >=0.05 in >=2/3;
- stratified capacity identity in fewer than 2/3 replicates.

### `STRATIFIED_CAPACITY_ALONE_SPECIALIZES`

The exactly balanced capacity control independently forms identity in >=2/3 replicates.

This takes precedence over a positive geometry claim.

### `MULTIMODALITY_WITHOUT_FUNCTIONAL_BIFURCATION`

Persistent multimodality exists in >=2/3, but the downstream geometry fork does not satisfy the full functional/routing/advantage gate.

### `FUNCTIONAL_BIFURCATION_WITHOUT_PERSISTENT_MULTIMODALITY`

Geometry functional identity appears in >=2/3 but the preregistered persistent K=2 gradient-field signal is absent.

### `NO_PERSISTENT_GRADIENT_MULTIMODALITY`

Neither persistent multimodality nor replicated functional bifurcation is established.

## Scope of a positive result

A positive 022 result would support:

```text
one local learning field can become stably multimodal;
its unlabeled geometry can define differential developmental experience;
that differential experience can create functional descendant identity.
```

It would **not** yet establish:

- autonomous inference-time recruitment;
- autonomous topology rewiring;
- arbitrary-many-mode growth;
- a production MoE architecture.

Those remain later steps.

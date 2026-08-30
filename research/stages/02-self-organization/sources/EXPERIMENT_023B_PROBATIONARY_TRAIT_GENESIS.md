# Experiment 023b — Probationary Trait Genesis

## Motivation

Experiment 022 produced `EMERGENT_TRAIT_BIFURCATION_SIGNAL`: a task-label-free phenotype-gradient geometry produced strong two-trait functional identity, while strictly matched capacity duplication did not.

Experiment 023 then tested online unknown-`K` structural model selection. Its corrected run rejected false Story-only growth but also rejected every genuine Story/Arithmetic and Story/Arithmetic/Transform birth. Inspection showed that the added lower-tail silhouette gate, not the structural penalty, blocked all multi-mode candidates. This established a useful negative result:

> better hard-cluster separability is not yet a justified proxy for whether another persistent phenotype deserves to exist.

023b therefore isolates the missing causal question:

> If a temporary fork is allowed to develop on future data, can it prove that its extra structure is worth keeping?

The experiment deliberately returns to `1 -> 2` only. It does not test unknown `K`, repeated growth, local sensing, rewiring, merging, pruning, or inference-time recruitment.

## Hypothesis

A structural proposal should not survive because it looks like multiple clusters. It should survive only if its counterfactual future utility exceeds the cost of the extra structure.

For a probation window `w`, define normalized fork utility:

\[
U_w^{fork}
=
\frac{L_w^{parent}-L_w^{fork}}
{|L_w^{parent}|+\epsilon}
-c_{structure}.
\]

Preregistered structural cost:

\[
\boxed{c_{structure}=0.005}
\]

or 0.5% of contemporaneous parent prequential NLL.

This is intentionally scale-normalized across conditions.

The geometry fork must also beat a matched-compute capacity fork:

\[
A_w=U_w^{geometry}-U_w^{capacity}.
\]

Preregistered recent-window advantage:

\[
\boxed{
\operatorname{mean}(A_{last\ 3})\ge0.005
}
\]

so extra capacity alone cannot explain birth.

## Fixed neural substrate

023b reuses the validated 021/022 substrate:

- TextNCA shared genome;
- context 128;
- hidden dimension 128;
- four attention heads;
- FFN 512;
- windows `[8, 32, 128]`;
- recurrent iterations `[4, 4, 4]`;
- GRU carry bias `+2`;
- tied embedding/head;
- one 128-dimensional persistent phenotype broadcast before the final shared NCA stage.

The parent is first trained on TinyStories for the same 300-step schedule used in 021/022. During probation the shared genome is frozen; only the parent phenotype or temporary child phenotypes can adapt.

## Shadow developmental sensor

For each incoming microbatch:

\[
g_t=\frac{\partial L_t}{\partial m_{parent}}
\]

is measured at the frozen parent phenotype.

This remains an oracle-quality sensor. Replacing it with a local cellular signal is explicitly out of scope.

No task/domain label enters this gradient.

## Proposal stage

Every condition intentionally opens exactly one temporary two-child proposal.

This is a scientific intervention: proposal quality is not under test in 023b. A false-looking proposal is allowed to enter probation so the utility gate must reject it on its own.

64 held-out proposal microbatches are used to fit the same task-label-free two-mode gradient geometry validated in Experiment 022.

The proposal records:

- `K=1 -> K=2` residual-fit gain;
- split balance;
- centroid separation;
- posthoc routing purity when two semantic families are present.

None of these metrics is a commit gate.

The two temporary children are initialized symmetrically around the parent:

\[
m_0=m+0.02v,
\qquad
m_1=m-0.02v.
\]

## Three counterfactual arms

All arms start from exactly the same pretrained parent checkpoint and consume exactly the same future probation stream.

### Parent

One phenotype continues online adaptation.

### Capacity shadow fork

Two temporary children are initialized with the same `±0.02v` symmetry break as the geometry fork.

Each stream key is sent alternately to child 0 and child 1 according only to occurrence count. This benchmark control uses stream identity solely to guarantee exact per-stream exposure balance. It is not a proposed routing mechanism.

Therefore both children receive exactly matched data exposure and the same number of active updates.

### Geometry shadow fork

Each incoming microbatch is routed only by distance of its frozen-parent phenotype gradient to the two proposal centroids.

No task/domain label is available to this routing function.

## Prequential evaluation

Probation lasts:

\[
4\times64=256
\]

online steps.

At every step and for every arm:

1. compute loss **before** that arm updates on the current microbatch;
2. record the prequential loss;
3. update only the active phenotype for that arm.

Thus utility cannot be created by evaluating on training examples only after adaptation.

Each 64-step block defines one independent chronological probation window.

## Commit rule

Let `U_w^G` and `U_w^C` be geometry and capacity normalized net utility.

A geometry fork is committed only if all three conditions hold:

### Sustained positive utility

\[
U_{3}^{G}>0,
\qquad
U_{4}^{G}>0.
\]

### Positive recent cumulative utility

\[
\operatorname{mean}(U_{2:4}^{G})>0.
\]

### Geometry-specific advantage

\[
\operatorname{mean}(U_{2:4}^{G}-U_{2:4}^{C})\ge0.005.
\]

No benchmark label enters this rule.

A fork can therefore be rejected even when its gradient geometry looks strongly bimodal, and accepted even when hard cluster separation would be modest, provided its future functional utility consistently pays for its cost.

## Controlled conditions

Each replicate runs four independent conditions from the same pretrained parent checkpoint.

### A — `STORY_ONLY`

256 TinyStories updates.

Preregistered expectation: **reject**.

This tests whether arbitrary two-way quantization of a single learning field can falsely earn structural existence.

### B — `DUPLICATED_STORY`

128 updates from `STORY_A` and 128 from `STORY_B`.

`STORY_A` and `STORY_B` are independent schedules over the exact same TinyStories token distribution.

Preregistered expectation: **reject**.

The model is never given the A/B labels.

### C — `STORY_ARITHMETIC`

128 TinyStories and 128 arithmetic updates.

Preregistered expectation: **accept** if the mechanism from Experiment 022 has prospective value.

### D — `WEAK_ARITHMETIC`

230 TinyStories and 26 arithmetic updates (~90/10).

This condition is discovery-only. There is no preregistered accept/reject target.

Its purpose is to reveal whether counterfactual utility naturally defines a practical capability-strength threshold without a semantic domain threshold.

## Post-commit functional validation

The commit decision is already complete before benchmark labels are used.

For mixed Story/Arithmetic conditions, each geometry child is then evaluated on both validation families.

Functional identity is permutation-invariant and uses the same 021/022 criterion:

\[
\boxed{\text{normalized identity margin}\ge0.01}
\]

for both matched families.

Posthoc geometry-routing purity must satisfy:

\[
\boxed{purity\ge0.75}.
\]

A positive utility result without functional identity is reported separately; it is not considered successful trait genesis.

## Strong positive gate

Across three replicates, `PROBATIONARY_TRAIT_GENESIS_SIGNAL` requires:

1. `STORY_ONLY`: reject in 3/3;
2. `DUPLICATED_STORY`: reject in 3/3;
3. `STORY_ARITHMETIC`: probation commit in >=2/3;
4. committed `STORY_ARITHMETIC` forks form 2x2 functional identity in >=2/3;
5. geometry routing purity >=0.75 in >=2/3;
6. geometry recent utility advantage over capacity >=0.005 in >=2/3.

`WEAK_ARITHMETIC` is reported but does not affect the strong status.

## Decision statuses

- `PROBATIONARY_TRAIT_GENESIS_SIGNAL`
- `FALSE_POSITIVE_PROBATIONARY_BIRTH`
- `UTILITY_WITHOUT_FUNCTIONAL_IDENTITY`
- `UTILITY_WITHOUT_STABLE_GEOMETRY_ROUTING`
- `CAPACITY_EXPLAINS_PROBATION_UTILITY`
- `NO_PROBATIONARY_BIRTH_SIGNAL`

## Interpretation boundary

A positive result would support:

\[
\boxed{
proposal
\rightarrow
shadow development
\rightarrow
counterfactual future utility
\rightarrow
persistent trait birth
}
\]

It would not yet establish autonomous proposal generation, repeated unknown-`K` growth, local NCA sensing, topology rewiring, or inference-time sparse recruitment.

The intended broader principle is:

> A structural change survives only if its counterfactual utility exceeds its cost.

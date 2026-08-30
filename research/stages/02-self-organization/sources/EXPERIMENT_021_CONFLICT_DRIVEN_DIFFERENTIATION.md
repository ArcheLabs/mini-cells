# Experiment 021 — Conflict-Driven Differentiation

## Motivation

Experiments 017–020 established that learned capability state can be localized, causally ablated, and transplanted, but adding one or three newborn cells did not reliably create capability identity. Experiment 020 ended with `NO_CAPABILITY_TISSUE_IDENTITY`.

The missing variable may be **differential learning pressure** rather than capacity. Two descendants that receive the same objective distribution have little reason to become different, even if they have separate phenotype memory.

Experiment 021 therefore tests the smallest causal chain needed for an emergent expert:

```text
persistent conflict -> fork -> differential experience -> functional differentiation
```

It deliberately returns to the stable 1-D TextNCA language substrate and does not include autonomous topology growth or a learned router.

## Question

Can a story-trained TextNCA discover that story and arithmetic learning pressures are incompatible, split one population phenotype into two descendants, and develop two functionally distinct cellular populations **without using task labels to choose a branch**?

## Cellular interpretation

The TextNCA genome is shared by all conditions. The first two NCA stages form a shared cellular stem. A small persistent phenotype vector is broadcast into the final 1-D token-cell population before the third shared NCA stage.

Before fork:

```text
shared stem -> parent 1-D cellular population
```

After fork:

```text
                    -> child population 0
shared stem --------|
                    -> child population 1
```

The NCA rule, embeddings, recurrent stage weights, normalization and language head remain shared. Forking duplicates only the phenotype vector, not an LLM or expert network.

## Abilities

### STORY

TinyStories, using the existing Experiment 005 byte-level BPE and cached token stream.

### ARITHMETIC

A deterministic synthetic language corpus encoded by the **same tokenizer**, containing four forms:

```text
Calculate a + b. Answer c.
Calculate a - b. Answer c.
Calculate a * b. Answer c.
Solve x + a = b. Answer x = c.
```

No domain token or expert token is added.

## Phase 0 — Parent

Each of three replicates starts from a fresh locked MiniCells-v2-style TextNCA:

- dim 128
- heads 4
- FFN 512
- windows 8 / 32 / 128
- recurrent iterations 4 / 4 / 4
- LayerNorm
- GRU carry bias +2
- tied embedding/head

The parent is trained for 300 steps on STORY only. This models the proposed developmental sequence: an existing language population encounters a new ability later.

## Phase 1 — Conflict calibration

After story pretraining, three independent calibration windows are collected. Each contains eight STORY and eight ARITHMETIC microbatches.

For every microbatch, Experiment 021 asks a counterfactual question at the parent phenotype `m`:

```text
g_k = dL_k / dm
```

The routing mechanism receives only the gradient vectors. It does **not** receive the domain label.

Gradients are unit-normalized and centered. PCA/SVD gives the first conflict axis `v`. The label-free conflict geometry records:

```text
directional cancellation = 1 - ||mean(unit(g_k))||^2
PC1 variance ratio
split balance = min(P(proj > 0), P(proj < 0))
```

A label-free geometry gate requires:

```text
directional cancellation >= 0.15
PC1 variance ratio >= 0.20
split balance >= 0.25
```

Domain labels are then used **only for posthoc scientific validation**. A normalized phenotype step in the STORY gradient direction must increase ARITHMETIC NLL by at least 0.001, and vice versa. A calibration window is conflict-positive only if the unlabeled geometry gate and both cross-interference tests pass.

Persistent dual-ability conflict requires at least 2/3 conflict-positive windows in a replicate.

The fixed routing geometry for the post-fork phase is learned from all calibration gradients together, still without labels.

## Phase 2 — Three matched arms

All arms clone the exact same parent checkpoint and consume the exact same 50/50 mixed STORY/ARITHMETIC microbatch schedule for 400 steps.

After fork, the shared TextNCA genome and parent phenotype are frozen. Only child phenotype vectors are trainable.

### Unified

One child phenotype is initialized exactly from the parent and receives every microbatch.

### Capacity Fork

Two child phenotypes receive the same symmetry break used by Differentiation Fork:

```text
m0 = m + epsilon * v
m1 = m - epsilon * v
```

with `epsilon = 0.02`.

Each incoming microbatch updates **exactly one** child according to a deterministic 50/50 schedule based only on step and replicate. The control does not inspect tokens, domain labels, loss, gradients or the conflict axis. Thus both forked arms have the same one-child-per-microbatch update budget, while the capacity children see the same mixed distribution in expectation.

This is the critical capacity control: extra phenotype capacity + symmetry breaking + matched update count, but no conflict-aware differential experience.

### Differentiation Fork

Initialization and per-step update budget are identical to Capacity Fork.

For every incoming microbatch, the parent phenotype gradient is measured and projected onto the fixed unlabeled conflict axis:

```text
score = <unit(g) - mean_unit_gradient, v>
```

Then:

```text
score >= 0 -> update child 0 only
score < 0  -> update child 1 only
```

The routing function receives no task label, domain ID, expert ID or learned router output.

## Functional identity

After training, each forked child is evaluated on both abilities, producing a 2 x 2 loss matrix. Child numbering is arbitrary, so identity is scored under the better of the two possible branch permutations.

For the chosen assignment, define:

```text
story margin      = L_story(other) - L_story(story-child)
arithmetic margin = L_math(other)  - L_math(math-child)
```

Normalize each margin by the parent loss on that domain.

A replicate has functional identity only when:

```text
story and arithmetic prefer opposite children
normalized story margin >= 0.01
normalized arithmetic margin >= 0.01
```

The differentiation arm must pass identity in at least 2/3 replicates.

Posthoc routing purity compares discovered branch assignment with the hidden STORY/ARITHMETIC labels, allowing the global branch permutation. It must be >= 0.75 in at least 2/3 replicates.

## Controls and decision

A strong positive result requires:

1. persistent conflict in >=2/3 replicates;
2. Differentiation Fork functional identity in >=2/3;
3. Differentiation Fork routing purity in >=2/3;
4. Capacity Fork must **not** produce identity in >=2/3.

Decision statuses:

- `CONFLICT_DRIVEN_DIFFERENTIATION_SIGNAL`
- `CAPACITY_ALONE_SPECIALIZES`
- `CONFLICT_WITHOUT_DIFFERENTIATION`
- `DIFFERENTIATION_WITHOUT_CONFIRMED_CONFLICT`
- `NO_DUAL_ABILITY_CONFLICT`

A positive result is evidence for `conflict -> division -> differentiation`, not yet for autonomous MoE routing. Inference-time recruitment and connectivity differentiation remain later experiments.

## Recovery

Each replicate saves:

```text
rN-parent.pt
rN-unified.pt
rN-capacity-fork.pt
rN-differentiation-fork.pt
```

for 12 checkpoints total. Completed parent/arm checkpoints are reused after interruption and are intentionally excluded from curated GitHub artifacts.

## Run

```bash
python -m pytest tests/test_language_conflict_differentiation.py -q
python scripts/run_conflict_driven_differentiation.py
```

The runner uses up to two GPUs concurrently across the three replicates.

# Native CLM v0 M3R — Read-Preserving / Lineage-Isolated Growth

Status: **FROZEN — READY FOR FORMAL RUN**

M3R follows the valid M3 negative result. It does not tune M3 thresholds or reuse M3 formal seeds.

## Question

> Starting from the exact canonical M1 checkpoint and keeping the M3 protected-write pressure controller numerically unchanged, does a read-preserving growth topology restore replay-free continual-language retention relative to the matched M3 global-pool growth algorithm?

The new variable is **read topology during mitosis**.

## Why M3R exists

M3 showed that autonomous growth was operational but unsafe at the read boundary:

- every formal seed grew from 8 to 16 Cells;
- children were heavily reused;
- plasticity and sparse execution survived;
- nevertheless TinyStories-A retention became worse than the matched fixed-topology control.

The reason is structural: M3 inserted every child directly into the same global Top-K pool. Freezing router parameters and old route keys did not freeze the routing function because the candidate key set changed.

M3R restores the hierarchical invariant that was already successful in Constructive CLM-003:

```text
frozen original-root router
          ↓
selected root lineages
          ↓
lineage-local parent/descendant selection
```

## Matched causal arms

```text
GPU0  global_growth_control
      exact frozen M3 growth algorithm
      dynamic global Top-K over all Cells

GPU1  lineage_growth
      same M3 growth-pressure thresholds
      immutable Top-K over the original 8 roots
      lineage-local concrete Cell selection
```

Both arms start from the exact M1 checkpoint, see the same freshly pinned data snapshot and seed, use the same B -> C -> D stream, use zero learner replay, train only Cell operators, and retain the same Core-005-style certificate projection.

## Function-preserving mitosis

The top-level root probability is conserved. A root lineage always consumes exactly the route mass assigned to its original root. A child therefore cannot steal another root's sparse slot.

Within a lineage, Cells form a parent-to-child chain. Each node may have at most one direct child. A query descends from parent to child only if the child's frozen route key has strictly higher cosine compatibility; ties retain the parent.

A new child is an exact operator clone of the currently pressured **leaf** parent:

```text
W_child = W_parent
Q_child = empty
```

At the instant of birth, switching parent -> child therefore preserves the lineage contribution exactly up to floating-point error, while preserving the top-level root gate mass.

Each birth records:

- model-logit max absolute drift;
- model-logit MSE;
- root Top-K match;
- root probability max drift.

## Growth controller is not retuned

M3R copies the frozen M3 numerical pressure rule exactly:

```text
growth check interval                  50
growth cooldown                       100
max new Cells                           8
max final Cells                        16
minimum route hits/window             512
minimum parent certificate rank         8
maximum projected/raw grad ratio       0.9
minimum window train loss              1.5
inherit scale                           1.0
```

The only additional eligibility constraint is `parent must be a lineage leaf`, which is required by the registered chain topology and is not a pressure-threshold change.

## Registered read-safety gates

In addition to zero replay, plasticity, retention and sparse-compute gates, M3R requires:

1. every birth has logits max-absolute drift <= `1e-5` and logits MSE <= `1e-10`;
2. every birth preserves root Top-K exactly and root-probability drift <= `1e-7`;
3. root-route probe hashes for A/B/C/D are identical before and after every continual phase;
4. A child-execution share is at least 10 percentage points lower than the matched global-growth control;
5. the lineage treatment has at least a 10-point child selectivity margin between its strongest new-domain child share and A child share;
6. lineage A regression <= 20%;
7. lineage A retention improves >=10 percentage points over global growth;
8. lineage mean forgetting <=15%;
9. lineage new-domain plasticity stays >=80% of global growth.

All registered gates must pass independently on all three formal seeds.

## Seed discipline

Development-only:

```text
73501 / 73502 / 73503
```

Untouched formal:

```text
73611 / 73612 / 73613
```

Permanently consumed and forbidden as M3R evidence:

```text
M2  73211 / 73212 / 73213
M3  73411 / 73412 / 73413
```

## Decisions

Positive:

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_SUPPORTED
```

Negative:

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
```

A negative formal result is preserved without post-hoc threshold changes or seed reuse.

Canonical protocol: [`protocol.json`](protocol.json)

Canonical Kaggle notebook: [`../../notebooks/06-native-clm/native-clm-v0-m3r-read-preserving-growth-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m3r-read-preserving-growth-kaggle.ipynb)

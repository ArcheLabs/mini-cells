# Native CLM v0 M3 Closure

## Decision

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

Formal seeds: `73411 / 73412 / 73413`  
Protocol SHA-256: `9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658`  
Data manifest SHA-256: `38197be7396d292106700b94208cdc65cf935809889f3759caa2f2ff5e390e16`  
Artifact commit: `e8b6a40f68862d6f01f67b125afdaeec97e6c45c`  
HF revision: `4bc1e73518f09039335a368d4352ff0201cee06c`

M3 is a valid frozen negative result. The seeds are consumed.

## What succeeded

Across all three seeds:

- the fixed arm remained at 8 Cells and exposed the expected retention failure;
- the growth arm reached 16 Cells;
- registered child reuse was 100%;
- sparse execution survived growth;
- B/C/D plasticity passed;
- growth plasticity stayed comparable to fixed control;
- learner replay remained exactly zero;
- shared substrate and original router state remained frozen.

Therefore M3 did not fail because the runtime could not create or use new Cells.

## What failed

Growth made old-domain retention worse on every seed:

| seed | fixed A regression | growth A regression | growth minus fixed |
|---:|---:|---:|---:|
| 73411 | 0.4416 | 0.4938 | +0.0522 |
| 73412 | 0.4293 | 0.4838 | +0.0545 |
| 73413 | 0.4351 | 0.4889 | +0.0539 |

Registered failures:

```text
growth_A_retention_advantage
growth_absolute_A_retention
growth_mean_forgetting
```

## Mechanistic diagnosis

M3 inserted each child into the same global Top-K candidate pool used by the original roots. Even with the original router parameters and original route keys frozen, the candidate set itself changed.

An exact operator clone is therefore not a function-preserving birth operation under global Top-K. A child can displace another Cell or change selected gate probabilities before any child learning occurs. Once the child is trained on new-domain data, old contexts routed through it receive those changes.

Seed 73411 makes the leakage visible. The four B-born children captured about 40% of routing mass on each of A/B/C/D. After C, all eight children captured about 50% of A routing mass. High route-hit reuse was therefore not evidence of correct addressability.

The growth controller also spawned at every available cooldown opportunity until the 8-child cap was reached at step 750, before phase D. This indicates that the current pressure rule did not establish a strong reuse-vs-grow boundary.

## Updated Native-CLM boundary

```text
M1: real next-token Native CLM trainability                SUPPORTED
M2: certificate protection reduces language forgetting     PARTIAL POSITIVE
M2: fixed protected topology is sufficient                 NOT SUPPORTED
M3: global-pool context-addressed growth restores retention NOT SUPPORTED
```

New blocking proposition:

```text
safe write growth requires safe read-address growth
```

## Next milestone

Do not advance to M4 ontology analysis and do not scale to 30M.

The next distinct experiment is **M3R — Read-Preserving / Lineage-Isolated Growth**.

Preferred invariant:

```text
root route before growth == root-lineage route after growth
```

A root router should continue selecting the same original lineages. Parent/child selection should happen only inside a selected lineage.

At birth, parent gate mass should be conserved:

```text
g_p W_parent
    ->
g_p [(1-alpha) W_parent + alpha W_child]
```

with `W_child = W_parent` initially, making the birth exactly function-preserving. Later divergence can then be learned within that lineage without globally stealing traffic from unrelated roots.

M3R must use new untouched seeds and a newly frozen protocol. It is not a post-hoc M3 retry.

Canonical formal details: [`../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md`](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md).

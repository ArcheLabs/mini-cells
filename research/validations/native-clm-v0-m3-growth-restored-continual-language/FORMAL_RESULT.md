# Native CLM v0 M3 — Formal Result

Status: **NOT SUPPORTED**

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

- Scientific decision: `false`
- Protocol SHA-256: `9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658`
- Data manifest SHA-256: `38197be7396d292106700b94208cdc65cf935809889f3759caa2f2ff5e390e16`
- Formal seeds: `73411 / 73412 / 73413`
- Learner replay: `0 bytes`
- Canonical artifact commit: `e8b6a40f68862d6f01f67b125afdaeec97e6c45c`
- Hugging Face revision: `4bc1e73518f09039335a368d4352ff0201cee06c`

## Registered decision

All three formal seeds reproduced the same failure pattern.

| seed | fixed A regression | growth A regression | growth advantage | fixed mean forgetting | growth mean forgetting | final growth Cells | child reuse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 73411 | 0.4416 | 0.4938 | -0.0522 | 0.2137 | 0.2201 | 16 | 1.000 |
| 73412 | 0.4293 | 0.4838 | -0.0545 | 0.2107 | 0.2170 | 16 | 1.000 |
| 73413 | 0.4351 | 0.4889 | -0.0539 | 0.2123 | 0.2186 | 16 | 1.000 |

The following registered gates failed on every seed:

```text
growth_A_retention_advantage
growth_absolute_A_retention
growth_mean_forgetting
```

The following relevant gates passed on every seed:

```text
growth_occurs_and_is_bounded
children_are_reused
growth_phase_plasticity
growth_plasticity_preserved
sparse_compute_survives_growth
shared_and_original_router_frozen
zero_learner_replay
```

Therefore M3 is not a failure to allocate or reuse capacity. It is a failure of the registered growth/address mechanism to convert new capacity into safer continual behavior.

## Post-formal diagnosis

This section is diagnostic interpretation of the frozen artifacts; it does not modify the registered decision.

### 1. Growth saturated the allowed topology

For seed `73411`, children were created at global steps:

```text
50, 150, 250, 350, 450, 550, 650, 750
```

The model reached the registered maximum of 16 Cells before phase D. Every child later received substantial route traffic. Thus the controller behaved close to cooldown-limited repeated spawning under sustained pressure rather than demonstrating a strongly selective reuse/grow boundary.

### 2. Child reuse was real but not address-selective

For seed `73411`, after phase B the four new children accounted for approximately:

```text
A route mass  40.33%
B route mass  40.01%
C route mass  41.52%
D route mass  39.32%
```

After phase C, all eight children accounted for approximately:

```text
A route mass  50.30%
B route mass  51.79%
C route mass  57.21%
D route mass  50.59%
```

The newly created addresses therefore captured large fractions of old-domain traffic instead of isolating only the conflict contexts that caused growth.

### 3. Global Top-K insertion is not function-preserving mitosis

M3 freezes the original router parameters and route keys, but every child is inserted into the same global candidate pool. The Cellular Layer recomputes Top-K over all Cells and then normalizes the selected scores.

Consequently, adding an exact operator clone does **not** guarantee an unchanged function: a new child can displace another previously active Cell or change the Top-K mixture before it has learned anything. After the child is trained on the new stream, old contexts routed through that child can then inherit new-domain changes.

The empirical signature matches this mechanism: growth improved or preserved new-domain plasticity, slightly reduced C forgetting, but made A retention consistently worse than the matched fixed-topology control.

## Frozen interpretation

M2 established:

```text
certificate protection reduces forgetting
but fixed topology still fails absolute old-domain retention
```

M3 adds:

```text
fresh writable Cells + global-pool context keys + high child reuse
!=
safe continual expansion
```

The new blocking boundary is:

```text
capacity growth is insufficient unless read ownership is preserved during mitosis
```

or, more compactly:

```text
safe write growth requires safe read-address growth
```

This does not invalidate Core 004, Core 005, Constructive CLM-002/003, or the general value of growth. It falsifies the specific Stage-06 M3 mechanism in which children enter the same global Top-K routing pool using mean conflict-query keys.

## Next research gate

Do **not** advance directly to M4 ontology analysis and do **not** tune M3 thresholds on seeds `73411/73412/73413`.

The next experiment must introduce a genuinely new integration variable: **function-preserving / lineage-isolated read growth**.

A preferred design is hierarchical lineage routing:

```text
root router selects the same original root lineages as before growth
                    ↓
within a selected lineage, a local gate chooses parent vs child
```

At birth, parent and child should share the original parent gate mass so that an exact clone is mathematically function-preserving. A child must not globally compete with unrelated roots merely because it was added to the model.

Only after read-preserving growth restores retention should M4 Cell ontology/specialization become the main milestone.

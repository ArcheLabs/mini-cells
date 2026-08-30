# Core Validation 004 — Growth-Restored Plasticity

## Decision question

Core Validation 003 established a useful but incomplete mechanism profile. With a frozen routing plane and frozen shared state, routed local updates had zero structural escape and zero false-safe events in the favorable synthetic world. Fine granularity also reduced dependency coverage sharply. However, strict transactional rejection prevented too much new learning: safety was strong, plasticity was insufficient.

Core Validation 004 tests the missing CLM mechanism:

> **When an existing Cell cannot safely absorb a new update, can monotonic Cell growth restore plasticity without giving up dependency-scoped safety?**

The experiment does not relabel Core Validation 003. The frozen parent outcome remains:

`DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED`.

## Core state machine

For transaction data `D_t` targeting context `c_t`:

1. Route through the current sparse base Cells.
2. Train an ordinary local candidate on the existing routed base Cells.
3. Validate the candidate only on its dependency-scoped old-behavior set plus the new-task validation set.
4. If the candidate passes, commit it.
5. If it fails and `c_t` does not yet own a growth Cell, roll the candidate back, create a zero-output private Cell, add a context-scoped route, train that Cell, validate, and atomically commit **Cell + route** only if the growth candidate passes.
6. Once a context owns a private Cell, later transactions for that context train only that Cell. V1 never spawns a second Cell for the same context.

This yields the closed-loop rule:

`ROUTE -> LEARN -> VALIDATE -> COMMIT | GROW -> VALIDATE -> COMMIT/ROLLBACK`

## Why the growth route is monotonic

003 showed that router drift invalidates historical dependency indexes. 004 therefore forbids rewriting the base router. Growth may only add one explicit context-scoped route. Existing routes and shared parameters are immutable during the continual stream.

A new Cell is initialized to output exactly zero. Before its training, adding the Cell and route is therefore functionally identical to the old model. After training, it can affect only its owning context. Because the old behavior of the current target context is intentionally superseded by the new truth, the new Cell has no old-behavior dependency domain outside that target context.

This is a controlled mechanism test. It does **not** claim that a language model already knows how to infer an equally precise semantic growth address.

## Why g = 8 is frozen

004 does not repeat the 003 granularity sweep. It fixes `g=8`, where formal 003 already showed:

- roughly 12% mean dependency coverage,
- zero frozen-router false-safe rate,
- zero frozen-router structural escape,
- the strongest near-threshold direct acceptance among fine-grained configurations.

This isolates growth as the only new mechanism.

## Variants

### `local_always`

Train the two routed base Cells and always commit. This is the high-plasticity / high-interference baseline.

### `local_tx`

Train the same routed base Cells, but commit only when new gain and dependency-scoped old regression pass. This reproduces the stability / low-plasticity failure mode exposed by 003.

### `local_tx_growth`

Use the same direct transaction as `local_tx`. On direct rejection, attempt the growth transaction. Once a private Cell exists, subsequent updates train only that Cell.

## World

The synthetic world stays deliberately favorable and closely matches 003:

- 64 contexts: 16 immutable anchors and 48 mutable contexts;
- four shared nonlinear functional families;
- each context composes two families;
- each mutable context has a private nonlinear residual;
- each transaction increases that context's residual amplitude;
- 96 transactions traverse every mutable context twice.

The second visit is important: it tests whether a spawned Cell remains reusable and plastic instead of forcing one new Cell per update.

## Formal metrics

### Safety

- false-safe rate;
- maximum structural escape outside the declared dependency domain;
- cumulative positive global old-regression damage.

### Plasticity

- effective acceptance rate, counting direct or growth commits;
- cumulative committed new-learning gain;
- final mutable-context normalized MSE.

### Growth behavior

- growth rescue rate after direct rejection;
- private-Cell reuse acceptance on later visits;
- number of spawned Cells;
- spawned Cells per effective commit;
- maximum active growth Cells per input.

### Cost proxy

For every attempted candidate:

`candidate parameter fraction + dependency coverage`

is accumulated and divided by effective commits. Growth can therefore be safe but still fail if it requires unbounded state or repeated retries.

## Frozen formal gates

Every formal seed must independently satisfy all gates:

- base normalized MSE `<= 0.10`;
- false-safe rate `<= 1%`;
- maximum structural escape `= 0`;
- effective acceptance `>= 80%`;
- cumulative regression damage `<= 0.30x local_always`;
- committed new-learning gain `>= 0.80x local_always`;
- final mutable NRMSE `<= 1.10x local_always`;
- growth rescue rate `>= 80%`;
- private-Cell reuse acceptance `>= 80%`;
- spawned Cells / effective commit `<= 0.60`;
- maximum active growth Cells per input `<= 1`;
- growth must improve committed new-learning gain over `local_tx`.

All formal seeds are required:

`80411, 80412, 80413`.

Development seed `80401` was used before protocol freeze only to verify implementation and candidate-training sufficiency. It is explicitly excluded from formal decisions.

## Formal statuses

- `GROWTH_RESTORED_PLASTICITY_SUPPORTED`
- `GROWTH_RESTORED_PLASTICITY_NOT_SUPPORTED`

Smoke mode emits only `SMOKE_ONLY` and has no scientific meaning.

## Interpretation

A formal positive result would support the controlled-mechanism CLM loop:

1. sparse routing exposes a dependency domain;
2. transactional validation rejects harmful local updates;
3. rejection can act as a growth signal;
4. a new Cell creates new degrees of freedom;
5. monotonic route addition preserves the stable addressing plane;
6. the grown Cell can be reused for later learning while active computation remains sparse.

That would justify calling the CLM core logic **closed-loop mechanism validated in a controlled synthetic setting**. It would not yet validate natural-language growth addressing, production-scale MoE routing, distributed JAM execution, or arbitrary NCA topology.

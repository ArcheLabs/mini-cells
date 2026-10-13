# Native CLM v0 M3 — Growth-Restored Continual Language

Status: **FROZEN / NOT YET FORMALLY RUN**

M2 is closed as a valid fixed-topology negative result: certificate projection reduced forgetting consistently, but the protected 8-Cell model still regressed on the original TinyStories behavior by about 44%, failing the registered 20% ceiling.

M3 does not tune M2. It consumes new untouched seeds and asks a new causal question:

> When protected reusable capacity becomes insufficient, can autonomous context-addressed Cell growth restore replay-free continual-language retention without sacrificing new-domain plasticity?

## Matched causal design

Both arms restart from the exact canonical M1 checkpoint and receive the same newly pinned A/B/C/D snapshot and identical seed/batch schedule.

```text
GPU0 fixed_protected
  8 Cells forever
  protected Cell-local writes

GPU1 growth_protected
  starts with the same 8 Cells
  same protected writes
  + learner-visible pressure may spawn children
```

The shared Transformer, query projection, normalization and original eight route keys are frozen in both arms. New child route keys are created at birth from the current conflict context and then frozen.

## Registered growth rule

Every 50 learner steps, M3 may inspect only current-training quantities:

- current-window train loss;
- top-1 Cell route hits;
- parent certificate rank;
- projected/raw Cell-gradient norm ratio;
- frozen-router query vectors for the currently routed contexts.

No domain ID, phase name, evaluator loss, hidden novelty label or historical training example is visible to the growth decision.

A parent is eligible only if the current window shows sufficient usage, certificate coverage and persistent projection pressure. The eligible Cell maximizing

```text
route_hits * (1 - projected/raw gradient ratio)
```

is split, subject to a 100-step cooldown and a maximum of eight new Cells.

The child receives:

```text
operator     exact clone of parent W
route key    mean normalized frozen-router query of current conflict contexts
certificate empty (rank 0)
parent_id    lineage pointer
```

Exact operator cloning is important: if parent and child become the two active routes, their weighted combined computation initially approximates the pre-birth parent computation. The child can then diverge under new protected writes while the parent retains its historical certificate.

## Stream

```text
A TinyStories             evaluation-only M1 retention anchor
B WikiText-2 raw          train
C cleaned Python CodeParrot train
D Databricks Dolly        train
```

Training is strictly `B -> C -> D`; learner replay is exactly zero bytes.

Because the original M2 local data manifest was lost when the Kaggle session terminated, M3 does not compare against old M2 numbers as the primary causal control. Instead, M3 creates a new exact Hub-revision-pinned snapshot and runs the fixed 8-Cell control concurrently with growth on that same snapshot.

## Formal seeds

Development only:

```text
73301 / 73302 / 73303
```

Untouched formal:

```text
73411 / 73412 / 73413
```

CI never runs formal seeds. The normal `--seed` path rejects them.

## Registered success requirements

Every formal seed must independently satisfy all gates, including:

- exact same M1 parent and matched M3 data snapshot;
- zero learner replay;
- shared/original read geometry bitwise frozen;
- fixed control remains exactly 8 Cells;
- fixed control exposes an A-regression capacity limit of at least 30%;
- growth is driven only by learner-visible pressure;
- 1–8 children are spawned, final Cell count <=16;
- at least 75% of children receive at least 512 post-birth routed token hits;
- sparse Cell execution remains <=30% of dense-all-Cell execution;
- growth arm learns each B/C/D phase by at least 5%;
- final growth-arm A regression <=20%;
- A retention improves over matched fixed control by at least 10 percentage points;
- growth mean forgetting <=15%;
- growth retains at least 80% of fixed-control mean new-domain plasticity.

Positive status:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_SUPPORTED
```

Negative status:

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

A positive result would close the first real-language growth-restoration gate. It would not establish asymptotic growth, a fully learned growth policy, semantic Cell ontology, multi-layer Cellular blocks, or LLM-scale Native CLM.

# Constructive CLM-005 — Scaffold Removal / Endogenous Control

Status: **FORMAL SUPPORTED — CONSTRUCTIVE MECHANISM SEQUENCE CLOSED**

Formal decision:

```text
LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED
```

Protocol SHA-256:

```text
3188e56af52763d2de75e4d13f25c5cbff2e56b25d5b32520495148e5e65b27d
```

Formal seeds:

```text
90811 / 90812 / 90813
```

All 20 registered gates passed on all three seeds. Canonical summary: [`FORMAL_RESULT.md`](FORMAL_RESULT.md). Canonical numerical artifacts: [`../../../artifacts/experiments/constructive-clm-005-scaffold-removal/`](../../../artifacts/experiments/constructive-clm-005-scaffold-removal/).

## Mission

CLM-005 was the final registered Constructive CLM gate before **Small Native CLM v0**. It asked whether the working engineered Constructive CLM stack could replace its explicit routing/write/growth control plane with learned controllers while preserving the parent invariants from CLM-001 through CLM-004.

This experiment did not re-prove Cell formation, structure-tracking growth, replay-free certificate protection, or model-level composition.

## What became learned

### Learned router

A shared pairwise compatibility MLP scores learner-visible route context against persistent Cell route keys. It works over a dynamic Cell set, including spawned children. Its meta target is current-transaction functional fit utility, not hidden Cell identity.

### Learned write controller

A binary MLP receives current-data certificate diagnostics and decides commit/reject. The actual safe parameter delta still uses the frozen Core-005 certificate-constrained solver.

### Learned growth/reuse controller

A binary MLP receives current-only reuse/fresh-fit and route-confidence diagnostics and decides whether a rejected useful conflict should allocate a new Cell. No evaluator novelty flag is supplied.

## Retained fixed primitives

CLM-005 intentionally did not neuralize every safety mechanism. It retains:

```text
persistent Cell state / route keys
Core-005 Cell-local certificate basis Q
certificate-constrained update solver
registered linear residual Cell operator family
```

The formal claim is therefore a **learned control-plane transition over an already-supported protected Cell substrate**.

## Registered anti-degeneracy structure

The formal stream contains four useful conflicts. Correct learned behavior is:

```text
12 roots
+ 4 useful children
= 16 final Cells

4 children × 3 repeat opportunities
= 12/12 child reuse without repeat spawning
```

An always-spawn control would add 16 Cells over the same opportunities. Unsafe reuse must measurably damage historical full-composition outputs.

## Formal outcome

Across all three formal seeds:

- router/growth/write held-out meta accuracy = `1.0`;
- operator acquisition route accuracy = `1.0`;
- exactly four children were created;
- child reuse = `12/12`;
- learner replay accesses = `0`;
- final simultaneous composition MSE = `1.13e-6` to `1.26e-6`;
- final sequential composition MSE = `1.34e-6` to `1.71e-6`;
- active Cell execution fraction stayed at or below `0.2002` of dense execution;
- protected historical composition MSE stayed around `1e-32`;
- unsafe reuse historical MSE was about `2.4e-2` to `3.0e-2`;
- the safe slow-shared-substrate probe preserved historical outputs while the unsafe control produced measurable interference.

## Anti-cheating learner boundary

Learner code does not receive:

```text
hidden factor ID
hidden task/mode label as a controller target
correct Cell ID
oracle route support
oracle novelty flag
future stream information
historical raw-example replay
```

Controller training used permanently development-only meta episodes and no formal-seed data.

## Seed discipline

Development-only CLM-005 seeds:

```text
601 / 602 / 603
```

Permanently excluded after a partial pre-freeze diagnostic:

```text
90711 / 90712 / 90713
```

Formal seeds:

```text
90811 / 90812 / 90813
```

The ordinary `--seed` path rejects formal and excluded seeds. CI does not execute the formal set.

## Interpretation boundary

The supported statement is:

> Within the registered controlled Cell world, a learned router plus learned write/growth controllers can govern a persistent protected Cell substrate while preserving bounded reusable growth, replay-free protected writes, sparse simultaneous/sequential multi-Cell computation, mutation isolation, and a safe slow-plastic shared-substrate compatibility invariant.

This does **not** by itself establish arbitrary nonlinear Transformer Cells, fully learned safety geometry, natural-language continual learning, asymptotic sublinear growth, foundation-scale Native CLM, or JAM deployment.

## Next milestone

The registered Constructive CLM mechanism-validation sequence is closed. The next main experiment is **Small Native CLM v0**. Do not create a cosmetic CLM-005B that only increases synthetic Cell count, horizon, or repeat count.

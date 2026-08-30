[English] | [中文](README.zh-CN.md)

# Stage 05 — Language Validation

> Status: Current research stage

## Question

Can the CLM core loop validated in Core Validation 004 survive the transition from a controlled synthetic function world to a real autoregressive token-level language model?

The pre-0.4 research established a controlled mechanism built from stable sparse routing, dependency-scoped regression validation, transactional commit/rollback, and growth-triggered plasticity recovery. Stage 05 does **not** reopen that mechanism search. It tests whether the same loop remains useful when model state is trained by next-token prediction and continual updates arrive as controlled language tasks.

## First experiment

The first experiment in this stage is [CLM-0.4-mini Language Validation](../../validations/clm-0.4-mini-language-validation/README.md), a roughly 5M-parameter decoder-only pilot using:

- a frozen shared language backbone during the continual phase;
- two fine-grained sparse Cell-FFN layers as mutable state;
- deterministic out-of-band addresses for the formal routing plane;
- controlled mathematics and story-world curricula;
- exact dependency-scoped local validation plus a hidden full-history oracle;
- transactional rollback and monotonic private-Cell growth;
- complete transaction, Cell-lineage, routing, cost, and state-hash observability.

The explicit address plane is intentional. This experiment tests **language-level mechanism transfer**, not autonomous semantic address discovery. A shadow semantic router may be measured, but it never controls formal commits.

## Decision ladder

Stage 05 separates three questions:

1. **M0 — execution smoke:** can the end-to-end transaction journal, routing, validation, rollback, growth, checkpoint, and replay pipeline execute correctly? No scientific decision is emitted.
2. **M1 — formal 0.4-mini pilot:** does the closed CLM loop preserve safety and recover useful plasticity over 192 controlled token-level continual-learning transactions? Three unseen formal model seeds are required.
3. **M2 — scale rehearsal:** if M1 passes, can the same architecture survive a substantially longer stream while state growth, dependency scope, checkpoint size, and validation cost remain observable and bounded enough to justify a 30–50M candidate?

Only `M1 Go AND M2 Go` authorizes work on the 30–50M CLM-0.4 formal candidate.

## What a positive result would mean

A positive M1 result would support the statement:

> The dependency-scoped transactional growth loop transferred from the registered synthetic mechanism setting to a controlled token-level language model.

It would **not** establish general natural-language continual learning, autonomous semantic routing, indefinitely bounded growth, LLM-scale behavior, or JAM-native training.

## What survives from earlier stages

- From **Foundations**: token-level language modeling and local neural computation.
- From **Self-Organization**: the view of capability as structured local state rather than one monolithic parameter block.
- From **Routing and Growth**: sparse routed Cells, progressive expansion, and reuse.
- From **Continual-Learning Core**: dependency-addressed safety, transactional commit/rollback, and growth as the response to constrained plasticity.

The canonical protocol for this stage's first experiment is frozen under `research/validations/clm-0.4-mini-language-validation/` before formal seeds may be observed.

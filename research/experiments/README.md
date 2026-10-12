# Research Experiment Implementations

This tree contains experiment-specific adapters and documentation, grouped by research stage. Reusable mechanisms belong in [`src/minicells/`](../../src/minicells/). Experiment code may import `minicells`; reusable package code must never import `research`.

Formal protocols remain under [`research/validations/`](../validations/) and canonical outputs remain under [`artifacts/experiments/`](../../artifacts/experiments/).

The Cell Internal Routing development study follows this layout in
[`experiments/cell_internal_routing_002/`](../../experiments/cell_internal_routing_002/).
It imports reusable training and model components from `experiments/common/`, retains
Experiment 001's old runner/config paths for published reproductions, and records its
frozen protocol under `research/validations/routing-002-cell-internal-routing/`.

## Current routing development

[Experiment 004 — Learned Cell Graph](004-cell-graph/README.md) is registered and unrun. It reuses 003 FFN Cells, shared Attention and Pipeline v2, and compares five full-budget arms over development seeds 42, 43, 44. Historical decisions remain unchanged.

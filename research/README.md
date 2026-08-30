[English] | [中文](README.zh-CN.md)

# MiniCells Research

CLM is a sparse graph of independently routable, mutable, and verifiable neural Cells. Its controlled core loop routes an input, attempts a local update, validates the historical computations that depend on the touched Cells, and either commits or rolls back; after rejection, bounded growth can add new state and retry atomically.

The core loop is supported only in a **controlled synthetic setting**. Natural-language continual learning, language-scale emergent routing, indefinitely bounded growth, and LLM-scale behavior remain unvalidated.

## Research stages

1. [Foundations](stages/01-foundations/README.md) — Echo, NCA language dynamics, 1D/2D tissue, settling, and training mechanics.
2. [Self-Organization](stages/02-self-organization/README.md) — sparse topology, recruitment, differentiation, and trait genesis.
3. [Routing and Growth](stages/03-routing-and-growth/README.md) — Cells became routed, independently mutable computational state.
4. [Continual-Learning Core](stages/04-continual-learning-core/README.md) — failed write-addressability, dependency-scoped transactions, and growth-restored plasticity.

Start with the [research history](reports/research-history-0.4.md), then read the [CLM Core Mechanism 0.4 report](reports/clm-core-mechanism-0.4.md). Machine-readable paths and outcomes are in [`catalog.yaml`](catalog.yaml); immutable evidence is in [`../artifacts/experiments/`](../artifacts/experiments/).

Core Validation 004 passed 3/3 registered seeds. This does not establish general natural-language continual learning. The next experiment is a 5–10M-parameter controlled language pilot, followed—only after a Go—by a 30–50M formal candidate.

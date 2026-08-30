[English] | [中文](README.zh-CN.md)

# MiniCells Research

CLM is a sparse graph of independently routable, mutable, and verifiable neural Cells. Its controlled core loop routes an input, attempts a local update, validates the historical computations that depend on the touched Cells, and either commits or rolls back; after rejection, bounded growth can add new state and retry atomically.

The core loop is supported only in a **controlled synthetic setting**. Natural-language continual learning, language-scale emergent routing, indefinitely bounded growth, and LLM-scale behavior remain unvalidated.

## Research stages

1. [Foundations](stages/01-foundations/README.md) — Echo, NCA language dynamics, 1D/2D tissue, settling, and training mechanics.
2. [Self-Organization](stages/02-self-organization/README.md) — sparse topology, recruitment, differentiation, and trait genesis.
3. [Routing and Growth](stages/03-routing-and-growth/README.md) — Cells became routed, independently mutable computational state.
4. [Continual-Learning Core](stages/04-continual-learning-core/README.md) — failed write-addressability, dependency-scoped transactions, and growth-restored plasticity.
5. [Language Validation](stages/05-language-validation/README.md) — transfer the frozen core loop into an observable token-level autoregressive language model before any 30–50M scale-up.

Start with the [research history](reports/research-history-0.4.md), then read the [CLM Core Mechanism 0.4 report](reports/clm-core-mechanism-0.4.md). The next registered experiment is the [CLM-0.4-mini token-level continual-learning protocol](validations/clm-0.4-mini-language-validation/README.md). Machine-readable paths and outcomes are in [`catalog.yaml`](catalog.yaml); immutable evidence is in [`../artifacts/experiments/`](../artifacts/experiments/).

## Research assets

- [Experiment implementations](experiments/README.md) are stage-aligned adapters that import reusable code from `src/minicells/`.
- [Notebook assets](notebooks/README.md) are organized by stage; their historical experiment IDs and bytes are preserved.
- [Validations](validations/) contain frozen protocols and bilingual summaries.
- [Canonical artifacts](../artifacts/experiments/) are immutable scientific evidence.

Core Validation 004 passed 3/3 registered seeds. This does not establish general natural-language continual learning. CLM-0.4-mini is now the registered 5M-class controlled token-level pilot; only an M1 scientific Go plus an M2 scale-rehearsal Go authorizes the later 30–50M formal candidate.

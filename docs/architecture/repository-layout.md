# Repository Layout

> Status: Current
> Scope: Engineering

- `src/minicells/` contains reusable model, routing, growth, continual-learning, evaluation, release, and runtime software.
- `research/` contains scientific history, experiment adapters, protocols, notebooks, and reports. It may import `minicells`.
- `artifacts/` contains immutable scientific outputs and must not be rewritten by architecture work.
- `scripts/` contains thin operational entrypoints grouped into research, release, runtime, and maintenance tasks.
- `tests/` contains unit, integration, research-regression, and smoke validation.
- `docs/` explains current engineering behavior and historical releases; scientific history belongs under `research/`.

The allowed scientific dependency direction is:

```text
research → src/minicells
```

Reusable package code must never import experiment implementations from `research/`.

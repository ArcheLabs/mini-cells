# Repository Layout

> Status: Current
> Scope: Engineering

- `research/projects/<project>/` is the home for a complete research project. Each project owns its experiment units and historical context.
- `research/governance/` contains cross-project registries, frozen decisions, and audits.
- `shared/` contains code used by multiple projects, including Python packages and Kaggle tooling.
- `src/minicells/` contains reusable product software. Research implementations belong with their owning project unless they are genuinely shared.
- `artifacts/` contains repository-wide release assets and public engineering evidence. Experiment results live with their experiment.
- `scripts/` contains repository-level maintenance and release entrypoints.
- `tests/` contains unit, integration, research-regression, and smoke checks.
- `docs/` explains current engineering behavior and releases; scientific history belongs under `research/`.
- `apps/`, `service/`, `crates/`, and `toolchains/` contain application, service, Rust workspace, and toolchain components.

The allowed scientific dependency direction is:

```text
research/projects/*/experiments → shared, src/minicells
```

Reusable package code must never import experiment implementations from `research/`.

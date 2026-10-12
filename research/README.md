[English] | [中文](README.zh-CN.md)

# MiniCells Research

This directory is organized by research project. Start from one of the eight project folders to find its hypothesis, experiment units, notebooks, implementation, reports and available evidence.

## Projects

| Project | Scope |
|---|---|
| [`01-foundations/`](01-foundations/README.md) | Early language dynamics and training feasibility |
| [`02-self-organization/`](02-self-organization/README.md) | Tissue formation, recruitment and trait genesis |
| [`03-routing-and-growth/`](03-routing-and-growth/README.md) | Routing, growth and granularity |
| [`04-continual-learning/`](04-continual-learning/README.md) | Core continual-learning mechanisms |
| [`05-language-validation/`](05-language-validation/README.md) | Retired CLM-0.4 language-validation lineage |
| [`06-native-clm/`](06-native-clm/README.md) | Native language-model training and continual-learning tests |
| [`07-safe-model-evolution/`](07-safe-model-evolution/README.md) | Safe evolution of pretrained models |
| [`08-hybrid-clm/`](08-hybrid-clm/README.md) | HybridCLM and its linked routing experiments |

## Cross-project governance

- [`audits/CLM_CAPABILITY_CEILING.md`](audits/CLM_CAPABILITY_CEILING.md) records supported claims and explicit No-Go boundaries.
- [`audits/RESEARCH_LEDGER.md`](audits/RESEARCH_LEDGER.md) preserves family-level outcomes, limitations and engineering value.
- [`audits/HISTORICAL_RESEARCH_ASSET_MAP.md`](audits/HISTORICAL_RESEARCH_ASSET_MAP.md) explains the status of historical research lines.
- [`catalog.yaml`](catalog.yaml) and [`registry.yaml`](registry.yaml) provide machine-readable indexes.
- [`MIGRATION_MAP.md`](MIGRATION_MAP.md) records project ownership and compatibility paths.

## Evidence discipline

Frozen protocols and decisions remain authoritative. A later explanation does not promote a failed decision; a useful mechanism does not prove model-level or language-level continual learning. Keep distinct source commits and run identities intact. Move paths only with reference audits and compatibility links, and keep each experiment under one project owner.

Experiment 004's manual entrypoint is now [`08-hybrid-clm/experiments/004-cell-graph/notebook.ipynb`](08-hybrid-clm/experiments/004-cell-graph/notebook.ipynb). Seed-42 `reference` and `graph` archives remain distinct under their recorded source commits; matrix aggregation remains `DEVELOPMENT_INCOMPLETE`.

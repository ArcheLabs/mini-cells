# Research project migration map

## Canonical layout

Research-owned material now lives under its owning project directory. Each project contains real hypothesis/context files, experiment units, runner code, and available reports or evidence. See the project README and its experiment index before running or interpreting a study.

| Project | Canonical material | Main prior source families |
|---|---|---|
| `01-foundations` | `research/01-foundations/` | foundations stage notes and notebooks; experiments 001–013; associated result trees and runners |
| `02-self-organization` | `research/02-self-organization/` | stage notes and notebooks; experiments 014–024; recruitment/differentiation runners and results |
| `03-routing-and-growth` | `research/03-routing-and-growth/` | stage notes and notebooks; CLM 0.3, growth/mitosis, and experiments 025/026 |
| `04-continual-learning` | `research/04-continual-learning/` | Core validation protocols, notebooks, adapters, runners, analysis, and results |
| `05-language-validation` | `research/05-language-validation/` | retired CLM-0.4-mini protocols, notebooks, release/preview material, reports, and results |
| `06-native-clm` | `research/06-native-clm/` | constructive CLM 001–005, Native CLM code/notebooks/validations, JAM demo, and results |
| `07-safe-model-evolution` | `research/07-safe-model-evolution/` | CoW, functional-boundary, history-compression, JAM, MoE, PCU, and pretrained-model-lift studies |
| `08-hybrid-clm` | `research/08-hybrid-clm/` | routing experiments 001–004, configs, implementation, manual Kaggle notebook, registered analysis, and result archives |

Repository-wide audits, the registry/catalogue, and historical workflow snapshots remain at `research/` because they govern or index multiple projects.

## Compatibility paths

Old asset-type paths remain as symlinks to the project-owned canonical files while existing notebooks, commands, workflows, and historical records still refer to them. They are compatibility entrypoints, not independent copies. New work must link to the paths under `research/01-*` through `research/08-*`.

Moved frozen protocol and result files retain their bytes and recorded source identities. Experiment 004 seed-42 `reference` and `graph` archives remain separate under their original recorded commit identities; aggregation remains incomplete. No source SHA or scientific status was rewritten to make the migration appear cleaner.

## Ownership rules

1. Every experiment has one owning project. Cross-project audits may index it without creating another editable copy.
2. Experiment code, configs, notebooks, reports and evidence live with the project when the historical path audit allows it. Existing external entrypoints resolve through compatibility links.
3. Product APIs remain under `src/`; top-level `scripts/` remains for repository-wide maintenance, CI and release work.
4. Root `shared/` accepts only components with multiple audited project consumers. One-project utilities stay within that project.
5. Frozen protocols/results remain authoritative. Directory organization does not upgrade, weaken or alter a scientific decision.

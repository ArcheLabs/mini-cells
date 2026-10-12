# Research project layout migration

## Goal

Organize research by the question and project that owns it. Opening a project
directory should eventually lead to its hypothesis, protocol, implementation,
execution entrypoint, evidence, analysis and conclusion. Root `shared/` is for
implementation components used by multiple projects, not research logic.

## Phase 1: scaffold established

The eight project roots and root-level shared component categories exist. This
phase does not copy or move canonical evidence. Existing asset-type directories
remain authoritative until a project's reference audit and compatibility path
are ready. Project root READMEs are migration entrypoints, not duplicate
experiment reports.

## Ownership map

| New project root | Research scope | Existing canonical material to migrate later |
|---|---|---|
| [`01-foundations/`](01-foundations/README.md) | Early feasibility, language dynamics and training mechanics | `stages/01-foundations/`, `experiments/01-foundations/`, `notebooks/01-foundations/`, matching `validations/` and `../../artifacts/experiments/` |
| [`02-self-organization/`](02-self-organization/README.md) | Tissue formation, recruitment, differentiation and trait genesis | `stages/02-self-organization/`, `experiments/02-self-organization/`, `notebooks/02-self-organization/`, matching validations and `../../artifacts/experiments/` |
| [`03-routing-and-growth/`](03-routing-and-growth/README.md) | Historical routing/growth lineage, including experiments 025/026 and their notebooks | `stages/03-routing-and-growth/`, matching notebooks, source documents, validations and `../../artifacts/experiments/` |
| [`04-continual-learning/`](04-continual-learning/README.md) | Core continual-learning validations and their runners | `experiments/04-continual-learning-core/`, `notebooks/04-continual-learning-core/`, `validations/core-*`, matching `../../artifacts/experiments/` and reports |
| [`05-language-validation/`](05-language-validation/README.md) | Retired and superseded CLM-0.4-mini language-validation lineage | `stages/05-language-validation/`, `experiments/05-language-validation/`, `notebooks/05-language-validation/`, matching validations and `../../artifacts/experiments/` |
| [`06-native-clm/`](06-native-clm/README.md) | Native CLM training, continual-language tests and diagnostics | `stages/06-native-clm/`, `native-clm/`, `notebooks/06-native-clm/`, `validations/native-clm-*`, matching artifacts and scripts |
| [`07-safe-model-evolution/`](07-safe-model-evolution/README.md) | Real-model evolution, historical supervision, safe mutation and rollback | `stages/07-safe-model-evolution/`, `notebooks/07-safe-model-evolution/`, matching validations, scripts and artifacts |
| [`08-hybrid-clm/`](08-hybrid-clm/README.md) | HybridCLM research and the linked routing experiment series named in the project proposal | `stages/08-hybrid-clm/`; experiments `001-global-router` through `004-cell-graph`; matching configs, implementations, notebook, validations, reports and artifacts. The routing series is a research precursor; its ownership here does not imply a HybridCLM regression or result. |

Cross-project audits, the research catalog/registry and genuinely shared
indexes may stay at `research/`. They provide navigation and governance; they
are not experiment-owned evidence. Product code in `src/minicells/` remains
the product source unless a later audit identifies a specific research-only
component.

## Asset mapping rules

1. Give each experiment exactly one project owner. Do not retain a second
   editable canonical copy under the old asset-type tree.
2. Migrate an experiment as a unit: hypothesis/background, configuration,
   implementation, notebook entrypoint, reports and its result subtree.
3. Keep frozen protocol and result bytes unchanged. A move may update indexes
   and links, but must not rewrite content or identities recorded by a protocol,
   manifest or run.
4. Before moving a path, search code, notebooks, workflows, docs, catalogues,
   registries and serialized manifests for references. Add a compatibility
   pointer only where a historical or external consumer needs the old path;
   never leave two editable copies.
5. Put code in `shared/` only when multiple projects use it and its behavior is
   independent of one experiment's hypothesis or protocol. Keep product APIs
   under `src/`, project-specific runners with their experiment, and
   repository-wide release/CI commands under top-level `scripts/`.
6. Preserve every source identity recorded by an in-progress paired matrix;
   never relabel an archive to make its commit match another run. Experiment
   004's seed-42 reference and graph archives are both retained as observed.
   Their recorded source commits differ and aggregation remains incomplete, so
   resolve that protocol/identity mismatch before treating the seed as paired.
7. Migrate one project at a time, update machine-readable indexes in the same
   change, and use repository CI to verify references. Do not run local tests
   for this migration.

## Shared component admission

| Shared category | Suitable contents | Keep out |
|---|---|---|
| `shared/python/` | Cross-project research-only Python utilities with multiple audited consumers | Product/model package, one-experiment training logic |
| `shared/scripts/` | Reusable research maintenance helpers called by multiple projects | Repository-wide CI/release commands and single-project launchers |
| `shared/kaggle/` | Common Kaggle bootstrap/publication helpers reused by multiple projects | Notebook-specific experiment cells or protocol settings |
| `shared/visualization/` | Reusable plotting/rendering primitives | Experiment-specific figures and interpretation |
| `shared/evaluation/` | Cross-project metric/evaluator primitives with stable contracts | Frozen experiment gates or protocol-owned scoring logic |

Do not migrate a component merely because its current location is inconvenient.
Record its consumers and preserve import compatibility in the same change.

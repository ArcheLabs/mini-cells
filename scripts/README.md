# Repository Scripts

Top-level scripts manage the repository as a whole. They do not own scientific
experiment implementations.

- [`maintenance/`](maintenance/) checks repository catalogs, artifacts, branch
  provenance, and layout invariants.
- [`release/`](release/) prepares and evaluates public releases.

Experiment-specific runners live with their experiment under
[`research/projects/`](../research/README.md). Utilities shared by multiple
experiments live under [`shared/scripts/`](../shared/scripts/). Product behavior
belongs in [`src/`](../src/).

Research protocols and status are authoritative in each experiment's frozen
records and the cross-project registry in
[`research/governance/`](../research/governance/README.md).

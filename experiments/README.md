# Experimental Language Models

The experiment catalog is now `registry.yaml`. Notebook First entries are in
`001-global-router/`, `002-cell-internal-router/`, and `003-hard-cell-transition/`.
See `docs/experiments.md` for preparation, original Kaggle notebook reuse,
runtime Secrets and immutable result publication. Historical paths below
remain compatibility sources; they are not moved or rewritten.

The reusable capacity matched language model and training primitives live in `common/`.
Experiment 001's published `runners/` imports and config remain stable compatibility
paths. New protocols, model variants, evaluation and evidence for the Cell Internal
Routing study live under `cell_internal_routing_002/`; its scientific record is frozen
under `research/validations/routing-002-cell-internal-routing/`.

The old Experiment 001 config, result files and frozen protocol have not been moved or
rewritten; `runners/` preserves the published command and checkpoint paths. New
Experiment 001 routing diagnostics can be collected from an original checkpoint with the
dedicated offline repair command documented in the Experiment 002 README; that command
writes a new report and preserves old artifacts.

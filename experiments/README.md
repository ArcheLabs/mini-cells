# Experimental Language Models

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

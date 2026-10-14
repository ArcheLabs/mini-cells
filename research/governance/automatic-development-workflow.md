# One-launch development workflow

This policy replaces manual registration paperwork as a prerequisite for **development execution**.
It records the user's request on 2026-10-14 to simplify future experiments. It does not retroactively
register past runs or grant independent confirmation access.

The user imports the experiment Notebook and clicks Run All. The program prepares pinned training
assets, trains the tokenizer on training text, checks duplicates/disjointness, selects a seed against
historical metadata, records actual hashes/source/configuration, checks the return path, then runs
its declared development budget. No user-entered hash/source/registration form is required.

Execution metadata are immutable within a run. Failures preserve checkpoints and evidence.
Completed runs resume publication without repeating training. Independent workers may share a
machine but not mutable model state; performance profiling runs after all training workers exit.
Small metadata/chunked diagnostics are returned using existing authenticated publishers from a
current-main results branch; weights and raw corpus stay in the Notebook output.

Preflight means functional readiness, not scientific approval. Formal conclusions still require
adequate baselines and valid statistics. Development asset deduplication/selection is explicit
and versioned. Never claim a historical holdout conflict audit was performed if it was not.
PD-001 development reuses the proven first-occurrence pilot story selection, bounded before the
reserved confirmation boundary; this selection change is recorded in runtime identities and is
not silently presented as the original raw-row protocol.

Frozen historical studies retain their original protocols. Future research should use the shared
`experiments.common.development` bookkeeping primitives and a single automatic launch entry.

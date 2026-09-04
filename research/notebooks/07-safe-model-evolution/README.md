# 07 — Safe Model Evolution

This notebook group contains current model-level Safe Model Evolution experiments built on real pretrained MoE/CLM substrates.

Scientific authority lives in the matching frozen validation protocol and durable experiment artifacts. Notebooks are orchestration, recovery, and visualization entry points.

Current workflows:

- `jam-knowledge-mutation-001-failure-diagnostic-kaggle.ipynb` — post-hoc forward-only diagnostic for the repeated JAM001 misconception-gate failure; reuses all nine published mutation artifacts and cannot revise the upstream formal decision.
- `jam-knowledge-mutation-001-kaggle.ipynb` — completed release-oriented real-domain JAM knowledge acquisition experiment; frozen decision `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED` (0/3 formal seeds PASS).
- `history-compression-001-kaggle.ipynb` — completed replay-budget ladder after Functional Boundary Oracle 001; learner-visible historical prompt budgets `32 -> 8 -> 2 -> 0`.

Related completed evidence:

- Functional Boundary Oracle 001 — `FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED`, 3/3 formal seeds.
- History Compression 001 — `HISTORY_COMPRESSION_TO_8_SUPPORTED`; `full_32` 3/3, `tiny_8` 2/3, `tiny_2` and `zero_0` 0/3.
- JAM Knowledge Mutation 001 — `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED`, 0/3 formal seeds; repeated misconception reference-answer NLL bottleneck preserved for post-hoc diagnosis.

Rules for this group:

1. Freeze protocols/formal seeds before formal GPU execution and freeze diagnostic plans before post-hoc GPU analysis.
2. Scientific FAIL results are published, not discarded.
3. Post-hoc diagnostics may explain a frozen failure mode but may not rewrite the formal decision.
4. Notebook stdout stays compact; detailed training/coordinate/diagnostic data are durable files.
5. Withheld historical or domain evaluation data must never become learner-visible or drive checkpoint selection.
6. Visualizations are derived views; `result.json`, `decision.json`, and diagnostic result files remain canonical evidence for their respective scopes.
7. Research notebooks belong under `research/notebooks/`, not repository-level `notebooks/`.
8. If a hosted workflow publishes result commits back to its PR branch, do not merge that PR before the intended results have been durably published.

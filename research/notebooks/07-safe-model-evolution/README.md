# 07 — Safe Model Evolution

This notebook group contains current model-level Safe Model Evolution experiments built on real pretrained MoE/CLM substrates.

Scientific authority lives in the matching frozen validation protocol and durable experiment artifacts. Notebooks are orchestration, recovery, and visualization entry points.

Current workflows:

- `history-compression-001-kaggle.ipynb` — formal replay-budget ladder after Functional Boundary Oracle 001; learner-visible historical prompt budgets `32 -> 8 -> 2 -> 0`.

Related completed evidence:

- Functional Boundary Oracle 001 — `FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED`, 3/3 formal seeds. Its canonical numerical artifacts remain under `artifacts/experiments/functional-boundary-oracle-001/`.

Rules for this group:

1. Freeze protocol and formal seeds before GPU execution.
2. Scientific FAIL results are published, not discarded.
3. Notebook stdout stays compact; detailed training/coordinate data are durable files.
4. Withheld historical evaluation data must never become learner-visible.
5. Visualizations are derived views; `result.json` and `decision.json` remain canonical evidence.
6. Research notebooks belong under `research/notebooks/`, not repository-level `notebooks/`.

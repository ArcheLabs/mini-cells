# 07 — Safe Model Evolution

This notebook group contains current model-level Safe Model Evolution experiments built on real pretrained MoE/CLM substrates.

Scientific authority lives in the matching frozen validation protocol and durable experiment artifacts. Notebooks are orchestration, recovery, and visualization entry points.

Current workflows:

- `jam-knowledge-mutation-001-kaggle.ipynb` — release-oriented real-domain JAM knowledge acquisition through bounded sparse multi-coordinate mutation.
- `history-compression-001-kaggle.ipynb` — completed replay-budget ladder after Functional Boundary Oracle 001; learner-visible historical prompt budgets `32 -> 8 -> 2 -> 0`.

Related completed evidence:

- Functional Boundary Oracle 001 — `FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED`, 3/3 formal seeds.
- History Compression 001 — `HISTORY_COMPRESSION_TO_8_SUPPORTED`; `full_32` 3/3, `tiny_8` 2/3, `tiny_2` and `zero_0` 0/3.

Rules for this group:

1. Freeze protocol and formal seeds before GPU execution.
2. Scientific FAIL results are published, not discarded.
3. Notebook stdout stays compact; detailed training/coordinate data are durable files.
4. Withheld historical or domain evaluation data must never become learner-visible or drive checkpoint selection.
5. Visualizations are derived views; `result.json` and `decision.json` remain canonical evidence.
6. Research notebooks belong under `research/notebooks/`, not repository-level `notebooks/`.
7. If a hosted workflow publishes result commits back to its PR branch, do not merge that PR before the intended formal runs have been durably published.

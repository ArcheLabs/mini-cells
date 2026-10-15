# Automatic workflow verification

Current user instruction: Run All performs preflight and continues automatically on success.

- CPU suite: 71 passed (including source binding, missing-check rejection, one-invocation continuation, CPU/return/data/GPU failure preventing development, and unfrozen Admission never promoting).
- Ruff passed; Notebook cells compile and contain no approval-config/mode-selection requirement.
- GPU preflight now also checks local save/restore continuation; interrupted preflight attempts use separate checkpoint folders and never overwrite prior checkpoints.
- No real GPU experiment, real-data download or confirmation read was performed by the coding agent.
- Scientific thresholds remain unfrozen; Lambda uses the declared candidate 0.01 as recorded development configuration. No scientific gate is manufactured as passing.
- Prior CPU evidence remains attached to its original source commits; the active source manifest identifies this workflow revision.

Kaggle: reimport notebook.ipynb from this branch, select T4 x2, and Run All. Existing HF_TOKEN/GITHUB_TOKEN secret labels and metadata publisher are reused.

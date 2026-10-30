# EPCLM-002-A0 — Persistent State Recall

This project-owned launcher wraps the existing frozen protocol and experiment implementation at `research/validations/epclm-002-a0-persistent-state-recall/` and `scripts/research/epclm_002_a0/`. The protocol, thresholds, formal seeds, and formal episode count remain unchanged.

## Run the bounded development workflow in Kaggle

1. Open the repository notebook `notebook.ipynb` in Kaggle (or import this file into the existing [MiniCells Kaggle notebook](https://www.kaggle.com/code/bingjiangli/minicells)).
2. Enable Internet. CPU is sufficient; no accelerator is required.
3. Add the existing Kaggle Secret named `GITHUB_TOKEN`, with permission to create and update repository contents.
4. Click **Run All** once. The notebook checks out this experiment branch, records the actual source commit and hashes, chooses unused development seeds automatically, runs the CPU development workflow, and publishes only the report, metrics, and execution identity.

The launcher has a 6,600-second internal ceiling and reserves up to 600 seconds for result publication. The notebook kills its process group at 7,200 seconds. Checkpoints, per-episode evidence, and failure state stay in Kaggle Output; model weights are never added to the Git results branch. A publication failure leaves the Kaggle Output in place, and rerunning the same source checkout resumes completed training/evaluation work before retrying publication.

This automatic launcher is **development-only**. It never uses formal seeds and always sets `scientific_decision: false`. The frozen formal matrix contains 60,000 episodes per learned backend across the registered conditions and seeds, so it cannot be represented as complete by a two-hour launch. Formal condition runners have the same two-hour maximum, persist completed episodes, and resume them in later invocations; until every registered condition is complete and validated, the matrix stays `INCONCLUSIVE`.

## Repository layout

- `config/development.json`: immutable development budget and condition list.
- `scripts/kaggle_launch.py`: automatic preflight, seed/identity recording, training reuse/resume, bounded evaluation, analysis, and publication.
- `notebook.ipynb`: the single Run All entry point.
- `scripts/research/epclm_002_a0/`: existing frozen-protocol implementation.

For local runs, the launcher may be invoked with `python .../scripts/kaggle_launch.py`; its default limit is 6,600 seconds and it uses the same automatic identity and publication workflow.

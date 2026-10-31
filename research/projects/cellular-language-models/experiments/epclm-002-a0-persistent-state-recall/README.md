# EPCLM-002-A0 — Persistent State Recall

EPCLM-002-A0 remains a mechanism-validation experiment. The original 60,000-episode frozen protocol is preserved unchanged at research/governance/validations/epclm-002-a0-persistent-state-recall/protocol.json as the full A0-v0.1 plan. The default Kaggle notebook now runs the separate A0-v0.2 screening plan described in A0-v0.2-screening.md.

## Run A0-v0.2 screening in Kaggle

1. Open notebook.ipynb in Kaggle or import it into the existing [MiniCells Kaggle notebook](https://www.kaggle.com/code/bingjiangli/minicells).
2. Enable Internet and select a GPU accelerator.
3. Keep the existing Kaggle Secret named GITHUB_TOKEN.
4. Click **Run All** once. The notebook checks out this branch, verifies the frozen protocol and clean source identity, trains the two existing architectures, measures a high-load runtime pilot, chooses a feasible target capped at 128 episodes per cell, runs the screening matrix, and publishes the report and metrics.

The six task conditions are 1, 8, or 32 facts crossed with 0 or 64 distractors. Flat Recurrent and Cellular use the same evaluation seed, episodes, queries, and targets, producing 12 architecture-condition cells. The planned target is 128 episodes per cell (1,536 screening episodes). The separate pilot measures 32 high-load episodes per architecture and is excluded from screening counts.

The existing trainer runs on Kaggle CUDA with the frozen 500-step, batch-size-8 development settings. The evaluation and reader stay on the existing CPU path. The pilot records per-episode P50/P95 wall time, checkpoint-save time, resume-state load time, training time, GPU name, and publication time. The chosen episode target uses the measured worst-architecture P95 and a conservative per-cell overhead estimate. A target below 128 is explicitly labeled capacity-adjusted; cells unfinished at the selected target are marked INCOMPLETE and remain resumable.

## Runtime budget

- 5,400 seconds for preflight, training, runtime pilot, evaluation, and checkpoint writes.
- 600 seconds reserved for result publication.
- 1,200 seconds safety margin within the Kaggle notebook hard cap of 7,200 seconds.

The launcher retains a fixed deadline even when P95 estimates are optimistic. Training or evaluation timeout produces an incomplete status rather than a complete screening result. The existing result publisher stages only the decision, metrics, execution identity, and summary from the current result directory on a branch descending from origin/main. Checkpoints and episode evidence stay in Kaggle Output. The publisher verifies the remote result commit SHA before reporting success.

## Scope and interpretation

The screening does not modify training algorithms, state architectures, data generation, task rules, query generation, or accuracy metrics. Overwrites remain fixed at zero; multi-round correction and retention stay in 002-B, retrieval and routing in 002-C, autoregressive language prediction in NCA-LM, and efficiency in 002-E.

Predeclared stop and diagnostic rules are in A0-v0.2-screening.md. These rules do not replace formal thresholds. Every result remains development-only and supports no scientific decision. The proposed next step is a separately preregistered confirmatory run on two or three selected conditions, approximately 400 total independent episodes per condition across at least three training seeds, with paired comparisons and between-seed variance.

## Preserved development workflow

config/development.json and scripts/kaggle_launch.py remain available as the original bounded CPU development workflow. The default notebook now calls scripts/kaggle_screening_launch.py with config/screening-v0.2.json; the original frozen protocol, formal matrix, thresholds, and seeds are not edited.

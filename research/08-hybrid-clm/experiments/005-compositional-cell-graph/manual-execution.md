# Manual execution policy

Experiment 005 will use the existing manually executed Kaggle notebook pattern and the repository's immutable, source-specific results publisher. There is no application-defined wall-clock deadline for manual training. A Kaggle platform interruption is recorded as incomplete; it never counts as a completed arm.

At the current planned status, the notebook defaults to CPU preflight; `dry-run` and `analyze` are also available. It refuses training because the proposal has no registered seed matrix, independent holdout manifest, frozen edge-refresh probe procedure, or final matched-compute/runtime protocol. Once those protocol items are approved and registered, each training run pins one full source SHA and one registered seed. The notebook checks GitHub write access before preparing assets or using a GPU, then appends the immutable arm archive through the existing repository Git askpass publisher. Checkpoints and prepared assets remain outside Git; only metric curves, arm evidence and their hash manifest are committed.

The experiment slug is `005-compositional-cell-graph`. It must not use the historical `kaggle/experiment-005-results` branch, which belongs to the separate Consumer Language Bridge study.

# Research execution workflow

For new experiments and changes to active launchers, use one manual Notebook launch:
automatic functional preflight → automatic development assets and identity recording →
development experiment → isolated profiling/analysis → result return.

Do not ask users to fill hash fields, select source SHAs, register seeds or switch modes between
preflight and execution. Generate and save reproducibility metadata internally; preserve the
experiment's declared budget and use existing repository components.

A failed functional/data/return-path check stops the run with an actionable error. Resume an
existing completed checkpoint instead of retraining. Use Kaggle's original notebook and existing
secret labels. Metadata publication must descend from current main and stage results only.

Development execution does not require formal scientific preregistration. Label development
results honestly; do not claim preregistration, independent confirmation or scientific support
from engineering success. Independent confirmation data remain isolated from development.

Do not rewrite frozen historical implementations, protocols, evidence or conclusions. Apply this
workflow to future experiments and active launchers; migrate older launchers only when requested.

Implementation work must not launch real training merely to test the launcher. Use synthetic CPU
fixtures unless the current task separately requests running an experiment.

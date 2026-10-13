# Experiment 001 — Global Router

Historical development comparison. The new directory provides a six-section
evidence notebook and thin compatibility scripts; old configs, result paths,
protocols and `experiments.runners`/`experiments.common` imports remain stable.

The authoritative configuration remains
`shared/python/experiments/configs/routing/experiment-001.json`; `config.yaml` references it.
Train CLI: `python research/projects/cellular-language-models/experiments/001-global-router/implementation/runner/scripts/train.py --help`.
Evaluate CLI collects post-hoc diagnostics from an original checkpoint into a
new directory, using the existing 001 repair command. It does not rewrite results.
Load downloaded evidence in the notebook to review it without rerunning a seed.

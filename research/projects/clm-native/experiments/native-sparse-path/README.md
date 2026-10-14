# Native Sparse Path v1

Implements the supplied [CLM Native v1.0 protocol](PROTOCOL_v1.0.md). Status: SCREENING_PREREGISTERED. Formal sealed confirmation is NOT_REGISTERED.

All models have independently addressable state slots, a bounded candidate interface, and a common input encoder/message/update/readout family. C0 scores current source/candidate states, sends actual selected messages, and updates only at most eight recipients per synchronous round. Initial encoding visits N nodes and is charged. No N×N runtime attention or per-Cell Python scheduling is used. B3 uses a central query controller with the same candidate/state access rights.

## Run

From the repository root (CPU; no external dataset or model download):

```bash
python research/projects/clm-native/experiments/native-sparse-path/run.py --mode preflight
python -m pytest -q research/projects/clm-native/experiments/native-sparse-path/tests
python research/projects/clm-native/experiments/native-sparse-path/run.py --mode screen --output /tmp/clm-native-screen
```

A completed run is preserved on rerun; an interrupted run resumes its checkpoint. Changes to source, config, data or seed reject resume. Fixed training-step budgets are research controls; manual execution has no setup or one-hour wall-time deadline.

For Kaggle import [notebook.ipynb](notebook.ipynb). It pins this implementation's source commit, uses your original `bingjiangli/minicells` notebook, and runs B0/B1 and B2/C0 concurrently when two GPUs exist. It preflights the existing GITHUB_TOKEN publisher before training and publishes raw small evidence after analysis. `push_existing` retries publication without retraining. HF_TOKEN is unnecessary for these generated tasks.

Use `--mode train --arm C0 --task B --output <run>` for one arm, `--mode evaluate/profile/intervene --checkpoint <run>/checkpoint.pt --arm C0 --task B --output <analysis>` for a frozen checkpoint. Source/config must match its identity. `--mode generate` records an input/target manifest for train/dev/OOD; standalone sealed generation is blocked.

## Scope and stop gates

M0 audits archives and records hardware. M1 checks mechanisms, identity, symmetry, cost and short training. M2 runs twelve equally budgeted arm/task jobs and a pooling shortcut diagnostic. M3 is blocked until C0 has a paired Task B quality/cost trend against the strongest B0/B1/B2. B3 and three-seed repeats are required before any positive H1 claim. M4 is evaluation only at larger N/longer paths; never larger-model training.

See [reports/REPORT.md](reports/REPORT.md), [budget](BUDGET.md), [task specification](GENERATOR_SPEC.md), [fairness](reports/BASELINE_FAIRNESS.md), and [causality](reports/CAUSALITY.md). Checkpoints are retained locally/Kaggle output, with hashes in published evidence; model weights are not committed.

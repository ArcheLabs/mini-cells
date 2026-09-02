# Kaggle Research Notebooks

Notebooks are grouped by the research stage they belong to. Their filenames and historical IDs are preserved.

- [`01-foundations/`](01-foundations/): Echo, TextNCA, language dynamics, and training mechanics.
- [`02-self-organization/`](02-self-organization/): topology, recruitment, differentiation, and trait genesis.
- [`03-routing-and-growth/`](03-routing-and-growth/): CLM routing, granularity, growth, releases, and mitosis.
- [`04-continual-learning-core/`](04-continual-learning-core/): historical Core Validation notebooks.
- [`constructive-clm-005-endogenous-control-kaggle.ipynb`](constructive-clm-005-endogenous-control-kaggle.ipynb): frozen CLM-005 formal runner + exact canonical-artifact publisher.
- [`05-language-validation/`](05-language-validation/): historical CLM-0.4-mini token-level transfer and infrastructure/data preparation.
- [`06-native-clm/`](06-native-clm/): Native CLM v0 real-model workflows:
  - `native-clm-v0-m0-m1-kaggle.ipynb` — M0 execution plus canonical ~12M M1 next-token training.
  - `native-clm-v0-m2-continual-language-kaggle.ipynb` — frozen M2 replay-free continual-language validation.
  - `native-clm-v0-m3-growth-restored-continual-language-kaggle.ipynb` — consumed M3 global-pool growth formal workflow; canonical result is negative and must not be rerun as untouched evidence.
  - `native-clm-v0-m3r-read-preserving-growth-kaggle.ipynb` — consumed M3R read-preserving lineage-growth formal workflow; canonical result is negative and the published checkpoints are now diagnostic inputs.
  - `native-clm-v0-m3r-address-diagnostic-kaggle.ipynb` — checkpoint-only two-GPU lineage-local address diagnostic. Reconstructs the exact M3R data snapshot, downloads the exact published lineage checkpoints, probes query versus write/effect geometry, and performs no Native CLM training or new formal-seed run.

Canonical numerical evidence remains under [`artifacts/experiments/`](../../artifacts/experiments/); notebooks are orchestration and reproducibility assets, not the source of frozen decisions. Frozen protocols live under `research/validations/`. Native CLM failures are evidence and must not be silently discarded or converted into post-hoc threshold edits.

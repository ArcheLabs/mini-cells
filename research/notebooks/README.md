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
  - `native-clm-v0-m2-continual-language-kaggle.ipynb` — frozen M2 replay-free continual-language validation. It downloads the exact M1 checkpoint from Hugging Face, prepares A/B/C/D corpora, runs protected and unsafe arms concurrently on two GPUs for each formal seed, uploads all formal end-state checkpoints to Hugging Face, and Git-publishes lightweight positive or negative evidence.

Canonical numerical evidence remains under [`artifacts/experiments/`](../../artifacts/experiments/); notebooks are orchestration and reproducibility assets, not the source of frozen decisions. Frozen protocols live under `research/validations/`. Native CLM failures are evidence and must not be silently discarded or converted into post-hoc threshold edits.

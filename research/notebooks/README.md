# Kaggle Research Notebooks

Notebooks are grouped by the research stage they belong to. Their filenames and historical IDs are preserved.

- [`01-foundations/`](01-foundations/): Echo, TextNCA, language dynamics, and training mechanics.
- [`02-self-organization/`](02-self-organization/): topology, recruitment, differentiation, and trait genesis.
- [`03-routing-and-growth/`](03-routing-and-growth/): CLM routing, granularity, growth, releases, and mitosis.
- [`04-continual-learning-core/`](04-continual-learning-core/): historical Core Validation notebooks.
- [`constructive-clm-005-endogenous-control-kaggle.ipynb`](constructive-clm-005-endogenous-control-kaggle.ipynb): frozen CLM-005 formal runner + exact canonical-artifact publisher. It refuses duplicate formal publication and requires Kaggle Secret `GITHUB_TOKEN`.
- [`05-language-validation/`](05-language-validation/): historical CLM-0.4-mini token-level transfer and infrastructure/data preparation.
- [`06-native-clm/`](06-native-clm/): Native CLM v0 model-training workflows. The first notebook runs M0 architecture/execution, the canonical ~12M M1 next-token training run, and publishes lightweight run evidence back to the research branch.

Canonical numerical evidence remains under [`artifacts/experiments/`](../../artifacts/experiments/); notebooks are orchestration and reproducibility assets, not the source of frozen decisions. Constructive CLM formal protocols remain under `research/validations/`. Native CLM training notebooks may preserve both passing and incomplete runs: model-training failures are evidence and must not be silently discarded.

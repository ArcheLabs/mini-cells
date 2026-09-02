# Kaggle Research Notebooks

Notebooks are grouped by the research stage they belong to. Their filenames and historical IDs are preserved.

- [`01-foundations/`](01-foundations/): Echo, TextNCA, language dynamics, and training mechanics.
- [`02-self-organization/`](02-self-organization/): topology, recruitment, differentiation, and trait genesis.
- [`03-routing-and-growth/`](03-routing-and-growth/): CLM routing, granularity, growth, releases, and mitosis.
- [`04-continual-learning-core/`](04-continual-learning-core/): historical Core Validation notebooks.
- [`constructive-clm-005-endogenous-control-kaggle.ipynb`](constructive-clm-005-endogenous-control-kaggle.ipynb): frozen CLM-005 formal runner + exact canonical-artifact publisher. It refuses duplicate formal publication and requires Kaggle Secret `GITHUB_TOKEN`.
- [`05-language-validation/`](05-language-validation/): CLM-0.4-mini token-level transfer, including M0 execution smoke and M1 infrastructure/data preparation.

Canonical numerical evidence remains under [`artifacts/experiments/`](../../artifacts/experiments/); notebooks are orchestration and reproducibility assets, not the source of frozen decisions. The CLM-005 notebook is deliberately a thin formal orchestration wrapper: the scientific protocol lives under `research/validations/constructive-clm-005-scaffold-removal/`, and the first formal run publishes either the registered positive or negative result without changing gates post hoc.

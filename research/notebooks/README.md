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
  - `native-clm-v0-m3r-address-diagnostic-kaggle.ipynb` — completed checkpoint-only lineage-local address diagnostic; canonical classification is `QUERY_GEOMETRY_SEPARABLE` and it performs no Native CLM training or new formal-seed run.
  - `native-clm-v0-m3l-query-sketch-gate-kaggle.ipynb` — completed checkpoint-only M3L mechanism diagnostic; canonical classification is `QUERY_SKETCH_GATE_NOT_FEASIBLE` with rank-16 median AUC 0.8968 against the frozen 0.90 gate.
  - `native-clm-v0-m3l1-address-state-capacity-kaggle.ipynb` — active checkpoint-only M3L-1 capacity/family diagnostic sweeping diagonal, ranks 8/16/32/64/128 and dense full covariance on the exact same M3L lineage samples and oracle.

Canonical numerical evidence remains under [`artifacts/experiments/`](../../artifacts/experiments/); notebooks are orchestration and reproducibility assets, not the source of frozen decisions. Frozen protocols live under `research/validations/`. Native CLM failures are evidence and must not be silently discarded or converted into post-hoc threshold edits.

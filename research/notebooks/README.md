# Kaggle Research Notebooks

Notebooks are grouped by the research stage they belong to. Their filenames and historical IDs are preserved for reproducibility.

A notebook path is **not** a scientific-status label. Frozen decisions live under `research/validations/`; the current cross-experiment interpretation lives under `research/audits/`.

See:

- [`../audits/CLM_CAPABILITY_CEILING.md`](../audits/CLM_CAPABILITY_CEILING.md)
- [`../audits/HISTORICAL_RESEARCH_ASSET_MAP.md`](../audits/HISTORICAL_RESEARCH_ASSET_MAP.md)
- [`../audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md`](../audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md)

## Current evidence-role classification

| Notebook group | Current role | Rule |
|---|---|---|
| `01-foundations/` | **HISTORICAL EXPLORATORY** | preserve feasibility/failure history and instrumentation; do not use as current continual-learning proof |
| `02-self-organization/` | **HISTORICAL MECHANISTIC EVIDENCE** | preserve recruitment/differentiation/probation mechanisms; emergence is not a natural Cell ontology claim |
| `03-routing-and-growth/` | **ENGINEERING PRECURSOR EVIDENCE** | preserve as active engineering heritage for sparse routing, explicit mutable state, candidate/fork, probation, growth and rollback concepts |
| `04-continual-learning-core/` | **FORMAL-EXPERIMENT ORCHESTRATION / HISTORICAL RUNNERS** | exact scientific authority comes from the matching frozen validation/result, not the notebook itself |
| `05-language-validation/` | **RETIRED / SUPERSEDED PROTOCOL LINEAGE** | keep CLM-0.4-mini protocol/transaction assets; Native Stage 06 supersedes it as the active trained-model scientific path |
| `06-native-clm/` | **CURRENT TRAINED-MODEL EVIDENCE WORKFLOWS** | consumed formal results remain frozen; diagnostics cannot relabel failed milestones |

## Notebook groups

- [`01-foundations/`](01-foundations/): Echo, TextNCA, language dynamics, and training mechanics.
- [`02-self-organization/`](02-self-organization/): topology, recruitment, differentiation, and trait genesis.
- [`03-routing-and-growth/`](03-routing-and-growth/): CLM routing, granularity, growth, releases, and mitosis.
- [`04-continual-learning-core/`](04-continual-learning-core/): historical Core Validation notebooks.
- [`constructive-clm-005-endogenous-control-kaggle.ipynb`](constructive-clm-005-endogenous-control-kaggle.ipynb): frozen CLM-005 formal runner + exact canonical-artifact publisher.
- [`05-language-validation/`](05-language-validation/): historical CLM-0.4-mini token-level transfer and infrastructure/data preparation; scientifically retired/superseded, methodologically retained.
- [`06-native-clm/`](06-native-clm/): Native CLM v0 real-model workflows:
  - `native-clm-v0-m0-m1-kaggle.ipynb` — M0 execution plus canonical ~12M M1 next-token training.
  - `native-clm-v0-m2-continual-language-kaggle.ipynb` — frozen M2 replay-free continual-language validation.
  - `native-clm-v0-m3-growth-restored-continual-language-kaggle.ipynb` — consumed M3 global-pool growth formal workflow; canonical result is negative and must not be rerun as untouched evidence.
  - `native-clm-v0-m3r-read-preserving-growth-kaggle.ipynb` — consumed M3R read-preserving lineage-growth formal workflow; canonical result is negative and the published checkpoints are now diagnostic inputs.
  - `native-clm-v0-m3r-address-diagnostic-kaggle.ipynb` — completed checkpoint-only lineage-local address diagnostic; canonical classification is `QUERY_GEOMETRY_SEPARABLE` and it performs no Native CLM training or new formal-seed run.
  - `native-clm-v0-m3l-query-sketch-gate-kaggle.ipynb` — completed checkpoint-only M3L mechanism diagnostic; canonical classification is `QUERY_SKETCH_GATE_NOT_FEASIBLE` with rank-16 median AUC 0.8968 against the frozen 0.90 gate.
  - `native-clm-v0-m3l1-address-state-capacity-kaggle.ipynb` — active checkpoint-only M3L-1 capacity/family diagnostic sweeping diagonal, ranks 8/16/32/64/128 and dense full covariance on the exact same M3L lineage samples and oracle.

## Historical reuse rule

When reusing an old notebook idea, copy the **primitive**, not its historical evidence rank.

Examples:

```text
probationary mitosis -> reusable candidate/shadow/rollback primitive
old autonomous-mitosis claim -> not inherited

CLM-0.4 transaction journal -> reusable provenance primitive
unfinished CLM-0.4 formal claim -> not inherited

self-organization pressure signal -> reusable proposal signal
natural functional Cell boundary -> not established
```

Canonical numerical evidence remains under [`artifacts/experiments/`](../../artifacts/experiments/); notebooks are orchestration and reproducibility assets, not the source of frozen decisions. Frozen protocols live under `research/validations/`. Native CLM failures are evidence and must not be silently discarded or converted into post-hoc threshold edits.

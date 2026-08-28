# CLM-0.3c Counterfactual Mitosis Results

This directory contains the curated evidence from the formal 3-replicate counterfactual experiment. Training checkpoints and corpus caches are excluded.

## Formal decision

- Growth equivalence: `CLM_COUNTERFACTUAL_GROWTH_EQUIVALENCE`
- Split-regret prediction: `NO_SPLIT_REGRET_PREDICTIVE_SIGNAL`
- Counterfactual decision: `NO_COUNTERFACTUAL_DECISION_SIGNAL`
- Capacity value: `NO_COUNTERFACTUAL_CAPACITY_VALUE_SIGNAL`
- Practical growth: `NO_COUNTERFACTUAL_PRACTICAL_GROWTH_SIGNAL`
- Formal GPU experiment: `True`

## Frozen formal parameters

- Decision checkpoint: 1.5M continuation tokens
- Shadow probe horizon: 100K tokens
- Candidates: all 12 CLM-0.1 root lineages
- Confirmation horizon: 500K tokens
- Probe holdout: 32 batches / 32K target tokens
- Confirmation holdout: separate 32 batches / 32K target tokens
- Bootstrap: 2,000 paired resamples
- Practical PPL threshold: 0.995

## Immutable training provenance

- Training commit: `8ad6a799f7f390e60a842751a5c2aa62673be1dd`
- Training tree: `69584e50fbd9f891bae181d3c91230267e4538cd`
- Publishing commit: `8ad6a799f7f390e60a842751a5c2aa62673be1dd`
- Publishing branch: `codex/clm-0.3c-counterfactual-mitosis`
- Kaggle script version ID: `not recorded`

Machine-readable hashes are in `metadata.json`. The authoritative formal decision is `decision.json`.

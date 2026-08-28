# CLM-0.3 Progressive Growth Results

This directory contains the curated formal evidence from the paired 3×3 Kaggle GPU matrix.
Transient caches and training/resume checkpoints are intentionally excluded.

## Formal decision

- Growth equivalence: `CLM_GROWTH_EQUIVALENCE`
- Growth viability: `NO_PROGRESSIVE_GROWTH_VIABILITY`
- Growth utility: `NO_GROWTH_UTILITY_SIGNAL`
- Pressure selection: `NO_GROWTH_PRESSURE_SELECTION_SIGNAL`
- Formal GPU experiment: `True`

## Matrix

- Replicates: 3
- Arms: fixed4, pressure_growth, random_growth
- Continuation budget: 1,500,000 tokens per worker
- Formal control: paired fixed4 continuation within each replicate

## Provenance

- Source commit: `6d6a9ff1425c8496dfe50a8bd7fa793e45f26ae2`
- Source branch: `codex/clm-0.3-progressive-growth`
- Source results directory: `results/clm-0.3-progressive-growth`
- Kaggle script version ID: `not recorded`

Machine-readable SHA-256 hashes and runtime provenance are in `metadata.json`.
The authoritative formal decision is `decision.json`.

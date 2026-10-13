# CLM-0.3b Marginal Growth Utility Results

This directory contains the curated formal evidence from the paired 3×3 Kaggle GPU matrix.
Training/resume checkpoints and corpus caches are intentionally excluded.

## Formal decision

- Paired pre-birth: `CLM_PAIRED_PREBIRTH_EQUIVALENCE`
- Saturation regime: `NO_SATURATION_REGIME`
- Growth equivalence: `CLM_GROWTH_EQUIVALENCE_FAILURE`
- Marginal growth viability: `NO_MARGINAL_GROWTH_VIABILITY`
- Marginal growth utility: `NO_MARGINAL_GROWTH_UTILITY_SIGNAL`
- Marginal selection: `NO_MARGINAL_SELECTION_SIGNAL`
- Newborn causal utility: `NO_NEWBORN_CAUSAL_UTILITY_SIGNAL`
- Formal GPU experiment: `True`

## Matrix

- Replicates: 3
- Arms: fixed4, marginal_growth, random_growth
- Earliest saturation check: 1.5M tokens
- Latest allowed pre-birth boundary: 3.0M tokens
- Matched newborn age: 1.0M tokens
- Formal validation: 32 batches / 32K target tokens
- Auxiliary root balance weight: 0.0

## Immutable training provenance

- Training code commit: `bf4f6afe77e6660cfce41ebd82571a892b1dcd3b`
- Training code tree: `27fab655a47c474681b18d387470a8fe1070db4c`
- Publishing commit: `bf4f6afe77e6660cfce41ebd82571a892b1dcd3b`
- Publishing branch: `codex/clm-0.3b-marginal-growth-utility`
- Kaggle script version ID: `not recorded`

Machine-readable SHA-256 hashes and provenance are in `metadata.json`.
The authoritative formal decision is `decision.json`.

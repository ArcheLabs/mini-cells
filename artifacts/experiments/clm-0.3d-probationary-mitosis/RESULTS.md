# CLM-0.3d Probationary Mitosis Results

This directory contains curated evidence from the formal three-replicate, two-environment probationary-mitosis experiment. Training checkpoints and corpus caches are excluded.

## Formal decision

- Overall: `CLM_PROBATIONARY_MITOSIS_SIGNAL`
- Growth equivalence: `CLM_PROBATIONARY_GROWTH_EQUIVALENCE`
- Stationary specificity: `CLM_STATIONARY_REJECTION_SIGNAL`
- Shift sensitivity: `CLM_SHIFT_PROMOTION_SIGNAL`
- Maturation: `NO_CLM_LINEAGE_MATURATION_SIGNAL`
- Formal GPU experiment: `True`

## Frozen formal parameters

- Decision checkpoint: 1.5M TinyStories continuation tokens
- Conditions: stationary TinyStories; 50/50 Story + Arithmetic capability shift
- Shadow ages: 50K / 100K / 200K / 300K / 500K
- Initial shadows: all 12 CLM-0.1 root lineages per condition
- Shortlist: top 4 by realized 100K point utility
- Promotion gate: positive 300K and 500K LCB95, positive mean 200K/300K/500K utility, final PPL ratio <= 0.995
- Independent holdout B: required for persistent promotion
- Shift Story-retention ratio: <= 1.01
- Bootstrap: 2,000 paired resamples

## Immutable training provenance

- Training commit: `af1eed85ac674495b684c22db49e839cf433bbe0`
- Training tree: `dbe4c7ff609105cdeb2083f0269de0af17289cdb`
- Publishing commit: `af1eed85ac674495b684c22db49e839cf433bbe0`
- Publishing branch: `codex/clm-0.3d-probationary-mitosis`
- Kaggle script version ID: `not recorded`

Machine-readable hashes are in `metadata.json`. The authoritative formal decision is `decision.json`.

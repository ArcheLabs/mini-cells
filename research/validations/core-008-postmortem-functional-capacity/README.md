# Core 008 Postmortem — Functional Capacity Decomposition

This bridge diagnoses the completed negative result from Core Validation 008 without changing that result.

Core 008 showed that the online certified adaptive allocator collapsed to 32 rank-1 atoms, exhausted the full 32-rank-unit budget, and left roughly 85.7% of training writes unresolved. The missing question is whether the normalized real write demands themselves contain a compact shared structure that the online mechanism failed to discover.

The postmortem separates four questions:

1. **Intrinsic rank** — how well can each write matrix be represented by its own optimal rank-r SVD?
2. **Shared linear dimension** — how well does the optimal train-fitted PCA subspace generalize to heldout writes?
3. **Sparsity cost** — once the best 32-dimensional global subspace is known, how much residual is introduced by retaining only the top-k coefficients?
4. **Budget-matched factorized capacity** — with exactly 32 total rank units, can an offline no-certificate dictionary of shared rank-r atoms represent heldout write demands?

The first three are geometric upper bounds. The fourth is the closest capacity control to the Core 008 factor budget, but remains intentionally offline and non-deployable.

## Non-confirmatory boundary

The bridge reuses already-observed Core 008 formal seeds `80821`, `80822`, and `80823`. Therefore every output keeps `scientific_decision=false`. It may localize failure and motivate a future protocol, but it cannot rescue or overturn `CERTIFIED_ADAPTIVE_FUNCTIONAL_ATOMS_NOT_SUPPORTED`.

## Interpretation

The frozen Core 008 local-action reference `0.35` is reused only as an interpretive ruler, not as a new scientific gate.

- PCA-32 <= 0.35: strong shared low-dimensional linear geometry exists.
- per-write rank-16 <= 0.35 but PCA-32 > 0.35: writes are individually low-rank but not globally aligned.
- a 32-rank-unit factorized dictionary <= 0.35: practical low-rank shared capacity exists offline; Core 008 failure is algorithmic rather than purely representational.
- even per-write rank-16 and PCA-32 > 0.35: the current normalized `G`-matrix basis hypothesis lacks strong compression evidence.

## Run on Kaggle

Use GPU and Internet ON. The runner rehydrates the same pinned model/data identity used by Core 006–008 and verifies the exact data manifest before analysis.

```bash
python scripts/research/orchestrate_core008_postmortem_functional_capacity.py \
  --branch codex/core-008-postmortem-functional-capacity \
  --device cuda \
  --push-results
```

A Kaggle notebook entry point is also provided at:

`research/notebooks/04-continual-learning-core/core-008-postmortem-functional-capacity.ipynb`

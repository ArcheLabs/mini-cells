# Experiment 019 — gradient-oracle numerical failure and recovery

## Observed failure

The first full Kaggle run completed all three GPU workers and all six trained donors per replicate. Postprocessing then audited 32,256 proposal observations and found:

- finite `oracle_gradient`: 19,794
- non-finite `oracle_gradient`: 12,462 (38.6347%)
- `oracle_fd`: fully finite in the same audit

The failed gradient rows must not be dropped or imputed. The preregistered Experiment 019 result is therefore invalid until the gradient oracle is regenerated with stable autodiff numerics.

## Root cause

At a closed newborn gate (`e=0`), `_gated_replicator_activity` can encounter very small old-cell drive variance while the unavailable newborn has a large counterfactual drive. Two expressions are forward-safe but backward-unsafe:

```python
variance.sqrt().clamp_min(1e-4)
torch.exp(ACTIVITY_RATE * fitness).clamp_max(20.0)
```

`sqrt(0)` has an infinite derivative. More commonly, `exp(large_fitness)` overflows to `inf` before the outer clamp restores the forward value to `20`; backward then encounters a masked `0 * inf` path and returns NaN with respect to the recruitment gate.

The stable forms are mathematically forward-equivalent:

```python
variance.clamp_min(1e-8).sqrt()
torch.exp((ACTIVITY_RATE * fitness).clamp_max(log(20.0)))
```

They move the clamp before the singular/overflowing primitive, keeping the same intended forward dynamics while giving finite derivatives at the closed-gate boundary.

## Recovery policy

Two paths are intentionally separated.

### 1. Exploratory FD diagnostic — no GPU retraining

The completed worker CSVs already contain a finite one-sided probationary utility

`[L(e=0) - L(e=0.02)] / 0.02`.

Run:

```bash
python scripts/research/run_proposal_utility_fd_diagnostic.py
```

This repeats the exact leave-one-family-out estimator analysis using `oracle_fd` in a copy of the observations. Its outputs are explicitly marked `exploratory_only=true` and MUST NOT be reported as the preregistered Experiment 019 result. It exists only to determine whether a stable-gradient rerun is scientifically worth the GPU cost.

### 2. Stable-gradient official rerun

Run:

```bash
python scripts/research/run_proposal_utility_discovery_stable.py
```

This writes to a separate `results/proposal-utility-discovery-stable-v1` directory, uses the same models, seeds, corpora, epsilon, features, LOFO split and decision thresholds, and changes only the forward-equivalent autodiff numerics. Because the first worker format did not checkpoint Phase-1/donor states, regenerating the corrected gradient oracle currently requires rebuilding the donors.

## Interpretation

This failure is numerical, not evidence that marginal proposal utility is mathematically undefined at `e=0`. The underlying recruitment interpolation is smooth at the boundary; the NaNs arise from an unstable computational graph used inside the metabolic replicator. The finite-difference oracle remaining finite on every row is an independent consistency signal supporting that diagnosis.

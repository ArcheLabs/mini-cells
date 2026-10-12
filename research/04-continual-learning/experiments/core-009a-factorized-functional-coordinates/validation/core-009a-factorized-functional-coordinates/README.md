# Core Validation 009A — Factorized Functional Coordinates

## Question

Core 008 postmortem found a sharp asymmetry:

- each individual normalized functional write is almost rank-1;
- a small fixed shared matrix basis generalizes poorly.

009A tests whether the missing shared structure lives in separate left/output-effect and right/input-condition factor spaces.

For normalized write demand `G`, training writes define

```text
C_L = mean(G G^T)
C_R = mean(G^T G)
```

with top eigenspaces `L_m` and `R_n`. Heldout geometry is evaluated by

```text
G_hat = L_m L_m^T G R_n R_n^T
```

This is an oracle geometry projection. It is intentionally not a deployable router.

## What 009A does not test

009A contains no sparse addressing, router, certificate, growth, replay, continual-learning mutation, or whole-model NLL gate. If geometry is not viable, later mechanisms are not run.

## Budget

The Core 008 conceptual budget is preserved:

```text
64 * (m + n) = 4096
m + n = 64
```

Discovery may inspect the full `(m,n)` residual landscape, but winner selection is restricted to the frozen `m+n=64` split list in `protocol.json`.

## Two-stage rule

### Discovery

Fresh seeds:

```text
80901
80902
```

The best budget split is selected by mean heldout local-action residual, with deterministic tie breaks. It is viable only if it independently reaches residual `<= 0.45` on both discovery seeds.

If discovery is not viable, stop. Do not run confirmation.

When viable, publication commits `winner-lock.json` under this validation directory.

### Confirmation

Untouched seeds:

```text
80911
80912
80913
```

Confirmation is impossible unless the committed winner lock matches the current frozen protocol SHA. The exact locked split must pass all frozen gates on all three seeds.

## Kaggle execution

Use GPU + Internet and the existing `GITHUB_TOKEN` Kaggle secret.

Discovery:

```bash
python scripts/research/orchestrate_core_validation_009a.py \
  --phase discovery \
  --branch codex/core-validation-009a-factorized-functional-coordinates \
  --device cuda \
  --push-results
```

After discovery publishes a viable winner lock, start confirmation from a fresh clone or pulled checkout containing that lock:

```bash
python scripts/research/orchestrate_core_validation_009a.py \
  --phase confirmation \
  --branch codex/core-validation-009a-factorized-functional-coordinates \
  --device cuda \
  --push-results
```

Each completed seed is reported and pushed immediately. A disconnected Kaggle session can hydrate already-published seeds on rerun.

## Interpretation

A positive result supports a compact tensor-product coordinate geometry, not a finished continual-learning architecture. The next step would be a separate 009B addressing/sparsity experiment before certificates or growth return.

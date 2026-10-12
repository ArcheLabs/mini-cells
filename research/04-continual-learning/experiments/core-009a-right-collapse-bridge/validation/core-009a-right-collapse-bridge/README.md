# Core Validation 009A Bridge — Right-Side Collapse Robustness

This is a **post-confirmation diagnostic bridge** for the already-positive Core Validation 009A result.

It does **not** reopen 009A, select a new winner, change any 009A threshold, or emit a new supported/not-supported scientific decision. The source result remains:

- status: `FACTORIZED_FUNCTIONAL_COORDINATES_SUPPORTED`
- source commit: `8290d4d674a8ec9ce98d4de129043526841e5f95`
- locked split: `(left=56, right=8)`
- confirmation seeds: `80911, 80912, 80913`

## Question

009A found a strongly asymmetric factor geometry: the output-effect/left side was broad while the input-condition/right side was almost one-dimensional. This bridge asks whether that right-side collapse is:

1. a robust property of frozen-Pythia functional writes;
2. largely the activation mean direction;
3. a more general activation anisotropy/coordinate effect; or
4. amplified by sequence averaging / q-z correlation and cancellation.

## Frozen controls

All transform statistics are fit on training tokens only and then applied unchanged to heldout evaluation tokens.

- `raw`: original 009A `z` coordinates.
- `centered`: subtract the train-token mean.
- `whitened`: subtract the train-token mean and apply the inverse square root of train-token covariance, with the frozen eigenvalue floor in `protocol.json`.
- `mean_direction_removed`: project away only the normalized train-token mean direction without otherwise centering or whitening.

For every control the bridge reports both sequence-level writes

`G_i = mean_t[q_t z_t^T]`

and token-level rank-1 writes

`g_t = q_t z_t^T`.

Token diagnostics are reported in two forms:

- per-token normalized right covariance;
- energy-weighted unnormalized token-write right covariance.

## Required source reproduction

Before any diagnostic result is accepted, the `raw` path must reproduce the frozen source 009A heldout `(56,8)` local-action residual for the same seed to absolute error `<= 1e-8`.

If this fails, the seed runner raises and no bridge interpretation is published.

## Top-1 ablation

The bridge fits the raw training sequence-write right PC1 `r1` and evaluates

`G_res = G (I - r1 r1^T)`.

It reports:

- residual Frobenius fraction;
- residual local-action fraction;
- removed-component local-action fraction;
- the right spectrum of the normalized residual writes.

This distinguishes “PC1 explains covariance energy” from the stronger claim “PC1 explains the functional action that matters locally.”

## Descriptive flags

`decision.json` is always `scientific_decision=false`. Once all three seeds complete, the reporter emits frozen **descriptive** flags:

- `centering_sensitive`
- `whitening_sensitive`
- `mean_direction_sensitive`
- `sequence_aggregation_sensitive`
- `robust_common_right_direction_across_controls`
- `top1_functionally_dominant`
- `post_top1_residual_still_low_dimensional`

These labels are interpretation aids, not scientific pass/fail gates.

## How to interpret the main outcomes

| Pattern | Interpretation for the next hypothesis |
|---|---|
| raw collapses, centered/whitened do not | the apparent near-1D right factor is materially driven by activation mean/anisotropy; do not build 009B around a literal 1D key space |
| raw sequence collapses but token writes do not | sequence averaging or q-z correlation/cancellation creates the common direction; 009B should work below the sequence-average level |
| raw, centered, whitened and mean-direction-removed all collapse | evidence for a robust common-right functional geometry under strong coordinate controls; a small-condition / wide-effect architecture becomes a serious hypothesis |
| removing PC1 leaves little local action | the common right direction is not only spectral but functionally dominant |
| removing PC1 leaves large action with a broad residual spectrum | PC1 is a strong common component, but important conditional structure still lives in the residual and should be represented in 009B |

## Kaggle run

Use Kaggle GPU + Internet and provide the `GITHUB_TOKEN` secret. The one-cell notebook is:

`research/notebooks/04-continual-learning-core/core-009a-right-collapse-bridge.ipynb`

Equivalent command from a prepared checkout:

```bash
python scripts/research/orchestrate_core_validation_009a_bridge.py \
  --branch codex/core-validation-009a-right-collapse-bridge \
  --device cuda \
  --push-results
```

The orchestrator checkpoints and pushes after every completed seed. `frozen-hidden.pt` is never published.

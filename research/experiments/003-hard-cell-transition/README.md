# Experiment 003-hard-cell-transition

## Hypothesis
Selected-only hard transition can retain soft-model quality while reducing executed computation.

## Motivation and evidence boundary
Experiment 003 observed seed 42 is incomplete: hard execution reduces GEMM arithmetic but PPL worsens by 21.69% versus soft. It does not prove quality-preserving hard transition or a general CLM advantage.

## Configuration
`experiment.yaml` is the declarative entrypoint; `research/configs/` records the exact historical numerical configuration. Frozen source protocol paths remain authoritative. V2 reuses existing models, paired sampling, staged training, and measurement code.

## Metrics
Training NLL, validation PPL, observed GEMM FLOPs/token, resident-batch inference latency, routing distributions, and expected/executed path length. Missing measurements remain unavailable.

## Result
Status: **development_incomplete**. See `analysis.md` and the immutable archive index.

## Reproduction
Prepare fixed CPU assets with `kaggle/prepare_assets.py`, then prepare the notebook with `kaggle/submit.py`. Kaggle runtime needs no credentials or network. A new source identity is a development run, not an untouched confirmation seed.

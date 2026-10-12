# Experiment 004-cell-graph

## Hypothesis
Cell graphs may learn composable computation paths; this remains a planned hypothesis.

## Motivation and evidence boundary
Experiment 003 observed seed 42 is incomplete: hard execution reduces GEMM arithmetic but PPL worsens by 21.69% versus soft. It does not prove quality-preserving hard transition or a general CLM advantage.

## Configuration
`experiment.yaml` is the declarative entrypoint; `research/configs/` records the exact historical numerical configuration. Frozen source protocol paths remain authoritative. V2 reuses existing models, paired sampling, staged training, and measurement code.

## Metrics
Training NLL, validation PPL, observed GEMM FLOPs/token, resident-batch inference latency, routing distributions, and expected/executed path length. Missing measurements remain unavailable.

## Result
Status: **planned**. See `analysis.md` and the immutable archive index.

## Reproduction
Execution is blocked until model, data, seed matrix, gates, and the dual-T4 one-hour budget are registered.

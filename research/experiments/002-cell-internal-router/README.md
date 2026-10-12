# Experiment 002-cell-internal-router

## Hypothesis
Cell-internal routing can learn sequential conditional computation.

## Motivation and evidence boundary
Experiment 003 observed seed 42 is incomplete: hard execution reduces GEMM arithmetic but PPL worsens by 21.69% versus soft. It does not prove quality-preserving hard transition or a general CLM advantage.

## Configuration
`experiment.yaml` is the declarative entrypoint; `research/configs/` records the exact historical numerical configuration. Frozen source protocol paths remain authoritative. V2 reuses existing models, paired sampling, staged training, and measurement code.

## Metrics
Training NLL, validation PPL, observed GEMM FLOPs/token, resident-batch inference latency, routing distributions, and expected/executed path length. Missing measurements remain unavailable.

## Result
Status: **development_incomplete**. See `analysis.md` and the immutable archive index.

## Reproduction
This is an archived experiment. Original launchers remain available through their historical paths; v2 does not silently rerun this protocol.

# MiniCells CLM-0.1 Research Preview

MiniCells CLM-0.1 is a small recurrent cellular language model research release built from the
Experiment 006 TextNCA checkpoint by function-preserving MoE-style upcycling.

## Architecture

- Base: TextNCA, 3 recurrent NCA stages, 4 iterations per stage.
- Routing: strictly local top-1 cosine-prototype routing.
- Experts: 4 full-width inherited FFN experts per stage.
- Total expert capacity: 4x the original FFN expert capacity.
- Active expert capacity per local update: 1x the original FFN.
- Cell activation: fixed at 1.0 in this release.

## Validation

Conditionality Validation 002: `CLM_LOCAL_CONDITIONALITY_SIGNAL`.

Release-candidate replicate: `2`.

- PPL vs matched dense continuation: `0.985890`.
- Aligned route disagreement: `0.726786`.
- Normalized Dynamic advantage vs Static: `0.010406`.
- Normalized Dynamic advantage vs Shuffled: `0.010421`.
- Usage entropy: `0.996486`.

## Parameters

- Dense TextNCA parameters: `1,170,816`.
- CLM total parameters: `2,357,760`.
- Active routed expert parameters: `395,136`.
- Router parameters: `1,536`.

## Scope and limitations

This is a research preview, not a general-purpose chat model. It was trained on TinyStories using
a 10M-token TextNCA base plus 1M continuation tokens. The release demonstrates function-preserving
capacity expansion and causally useful local conditional routing. It does **not** yet demonstrate
active FLOPs below the original dense TextNCA, autonomous capacity growth, online self-learning,
phenotype, multimodality, or 100M+ token scaling.

The sparse-dispatch implementation is a correctness/reference backend; wall-clock speedups are not
claimed. Benchmark telemetry is included in `benchmark.json`.

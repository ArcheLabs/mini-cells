# CLM Sparse Runtime v2

This branch adds a Turing/Tesla-T4-oriented sparse inference runtime on top of the frozen CLM-0.3 model semantics.

## What changes

- Adds `padded_sparse`: top-1 token routing with fixed per-expert capacity and two batched FFN GEMMs.
- Overflow tokens are processed by the exact reference sparse path; no token is dropped.
- Autotuning considers several padded capacity factors plus the existing reference sparse and packed dense paths.
- Every non-reference candidate must pass a numerical parity gate before it can be timed or selected.
- Runtime telemetry reports route counts, maximum expert load, padding capacity, overflow count, and executed expert-token pairs relative to dense execution.
- The existing `batched_dense` training backend remains the only optimized training candidate; it preserves the current dense STE gradient semantics.

## What does not change

- checkpoint format;
- root/split routing decisions;
- lineage IDs or parent/child relations;
- birth equivalence requirements;
- probation/promotion rules;
- historical CLM-0.3 release evidence.

## T4 benchmark

```bash
python -m pytest \
  tests/test_clm_sparse_runtime.py \
  tests/test_clm_sparse_runtime_v2.py \
  tests/test_clm_progressive_growth.py \
  tests/test_growth_router.py \
  -q

python scripts/benchmark_clm_sparse_runtime_v2.py \
  --device cuda \
  --output results/clm-sparse-runtime-t4-v2.json
```

The v2 benchmark adds the frozen TextNCA source as an explicit inference/training baseline. The main values to inspect are:

- `speedups.optimized_over_reference_sparse`;
- `speedups.clm_inference_time_ratio_vs_textnca`;
- `speedups.batched_dense_train_over_masked_dense`;
- `speedups.clm_train_time_ratio_vs_textnca`;
- `runtime_status[*].backend`;
- `runtime_status[*].route_counts`;
- `runtime_status[*].expert_token_pair_fraction_vs_dense`;
- `runtime_status[*].autotune_v2`.

`expert_token_pair_fraction_vs_dense` is an execution-structure diagnostic, not a measured FLOP count. For a balanced four-expert top-1 route, an ideal sparse expert FFN approaches 0.25 of dense expert-token pairs before accounting for shared attention/GRU/router work and dispatch overhead.

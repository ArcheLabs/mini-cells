# CLM Sparse Runtime v3

Runtime v3 is an engineering layer over the frozen CLM-0.3 research model. It does not change checkpoint formats, routing decisions, lineage structure, birth parity, probationary growth, or the exact-STE training objective.

## Motivation

The Tesla T4 v2 benchmark established two facts:

- semantic top-1 routing executes only 25% of dense expert-token pairs;
- all new FP16 candidates were rejected before timing because the local parity gate used an FP32-like absolute tolerance, even though the observed candidate drift was only 0.5–1 FP16 ULP.

It also showed strongly skewed per-stage router traffic, so a single symmetric capacity wastes padded work.

## Runtime v3

### Precision-aware parity

Autotune candidates must still match `reference_sparse`, but the local envelope now depends on the actual inference compute dtype. FP32 remains strict. FP16/BF16 accept only the small numerical drift expected from different GEMM packing/accumulation order. The gate also records RMS and relative-L2 drift and rejects non-finite output.

This local gate does not replace model-level validation. `benchmark_clm_sparse_runtime_v3.py` also reports final-logit max error, RMS error, relative L2, argmax agreement, and synthetic-target NLL delta.

### Per-step calibration

TextNCA/CLM performs several recurrent updates per stage. Runtime v3 calibrates each `(stage, recurrent step, token shape)` separately rather than assuming the first routing distribution represents the whole stage.

Calibration is allowed to synchronize route counts to the CPU once. Stable inference does not collect route counts or rebuild the candidate set.

### Cached hot path

After calibration, the runtime stores the selected backend and executes it directly. Candidate construction, parity checks, timing, and route telemetry are not part of the stable inference loop.

### Two-tier capacity buckets

For skewed routing, runtime v3 generates load-aware plans that split experts into high-load and low-load groups. Each group has one fixed capacity and therefore one batched GEMM per FFN projection. This keeps batching while reducing padding relative to using the global maximum expert load for every expert.

Overflow remains exact: routed tokens beyond a calibrated capacity are evaluated by `reference_sparse`; no token is dropped.

### Training

Training semantics are unchanged. `masked_dense` remains the historical exact-STE backend and `batched_dense` remains the optimized exact-STE candidate. True top-1 sparse training is intentionally out of scope because it changes the current router gradient estimator.

## T4 benchmark

```bash
python scripts/benchmark_clm_sparse_runtime_v3.py \
  --device cuda \
  --output results/clm-sparse-runtime-t4-v3.json
```

Relevant regression suite:

```bash
python -m pytest \
  tests/test_clm_sparse_runtime.py \
  tests/test_clm_sparse_runtime_v2.py \
  tests/test_clm_sparse_runtime_v3.py \
  tests/test_clm_progressive_growth.py \
  tests/test_growth_router.py \
  -q
```

The main decision values are:

- `calibration_summary.selected_backends`
- `runtime_status[*].calibration_profiles[*].expert_token_pair_fraction_vs_dense`
- `speedups.optimized_over_clm_masked_dense`
- `speedups.clm_inference_time_ratio_vs_textnca`
- `numerical.optimized_sparse_vs_reference`

Runtime v3 should not be promoted as a production kernel until the T4 regression and benchmark are executed on the target hardware.

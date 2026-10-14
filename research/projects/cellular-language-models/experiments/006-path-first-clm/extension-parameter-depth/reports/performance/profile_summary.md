# CPU performance evidence

Baseline: `2e77369b0df1ce704863a815dacc48301736dde0`. Measured repair: `c9b58b08e173386885956bd6cdd673e1cb468bb8`.
Delivery implementation: `ece765188bcf94752212ae0f6fb33a3314a655aa`. All measured checkouts were clean.

5 warmup + 20 measured steps, 3 fresh-process repeats per case/arm. One CPU thread.
Synthetic fixtures: width 12, two Cells, four-token sequences, effective batch 4.
M0 fixture has two MoE layers, not the scientific six-layer model. Fixture parameter ratios
are not the scientific 50% comparison. These numbers quantify within-arm engineering overhead;
they do not establish actual-model CPU/GPU speed or language quality.

| Arm | Case | Median of repeat p50 (ms) | Repeat p50 range (ms) |
|---|---|---:|---:|
| M0 | B0 | 72.925 | 64.328–92.579 |
| M0 | B1 | 77.240 | 70.671–99.056 |
| M0 | B2 | 66.779 | 65.235–70.141 |
| M0 | B3 | 61.025 | 59.712–62.411 |
| M0 | B4 | 24.010 | 23.248–24.326 |
| M0 | B5 | 20.349 | 20.064–20.589 |
| M0 | B6-m2 | 12.027 | 11.900–12.197 |
| M0 | B6-m4 | 8.054 | 7.907–8.478 |
| M0 | LIGHT | 19.985 | 19.738–20.401 |
| P50-F16 | B0 | 724.094 | 709.286–732.489 |
| P50-F16 | B1 | 672.470 | 668.304–695.326 |
| P50-F16 | B2 | 677.791 | 676.597–701.945 |
| P50-F16 | B3 | 682.944 | 675.119–707.930 |
| P50-F16 | B4 | 229.040 | 225.951–232.866 |
| P50-F16 | B5 | 197.350 | 193.283–200.358 |
| P50-F16 | B6-m2 | 110.879 | 109.679–113.138 |
| P50-F16 | B6-m4 | 61.536 | 61.400–72.482 |
| P50-F16 | LIGHT | 196.384 | 195.227–207.524 |

## Attribution and limits

- M0: B0→LIGHT 3.65×; B3→B4 removes Dispatch, 61.025→24.010 ms. B4→B5 reduces probes, 24.010→20.349 ms.
- P50-F16: B0→LIGHT 3.69×; B3→B4 removes Dispatch, 682.944→229.040 ms. B4→B5 reduces probes, 229.040→197.350 ms.

TorchDispatch interception is the dominant confirmed instrumentation cost in these fixtures.
P50 capture removal gives a modest gain. Whole-parameter clone/gradient diagnostics and Counter
do not show a separately reliable gain in the tiny P50 fixture; B1–B3 differences are within
variation, and tiny weights cannot model the bandwidth cost of copying the actual 8.2M parameters.
M0 does not have CLM state records; its B0/B1 difference is noise/scaffold variation, not evidence
that capture benefits MoE. B0 uses the real old runner, other cases the equivalent extracted step;
this small scaffold difference and unpinned CPU clocks limit precise single-factor attribution.

| CPU operator (one profiler step, separate from timing) | M0 B0 | M0 LIGHT | P50 B0 | P50 LIGHT |
|---|---:|---:|---:|---:|
| aten::_local_scalar_dense | 120 | 44 | 1058 | 354 |
| aten::nonzero | 24 | 16 | 481 | 320 |
| aten::clone | 66 | 8 | 1577 | 846 |
| aten::multinomial | 0 | 0 | 80 | 64 |
| aten::index_copy | 0 | 0 | 789 | 535 |

Counts above are CPU profiler counts, **not CUDA host-synchronization counts**. All CUDA
kernel time, launch, synchronization and memory fields are null because no GPU was run. CPU
self-time is retained per operator in raw JSON. Chrome traces have checksums and were kept
outside Git. Profile windows are one extra step; timings do not include the profiler. LIGHT
also records one scheduled diagnostic+probe-step time outside the normal-step interval.

## Correctness and restart

80 tests passed; all 10 actual-size joint CPU fixtures passed. Source model/Router/loss
and reused parent implementations match the actual run commit. Loss/gradient replay and
multi-step old/new weights, optimizer and RNG agree. The additional M0-window regression test
ensures periodic probes do not silently disable or reinterpret its stability threshold.

Portable CPU bundles were exported, restored into a fresh directory and resumed in a separate
Python process; final state equals uninterrupted training. Corrupt files and wrong source
identities are rejected. This is **not** a retained Kaggle cloud-output rehearsal. The performance
Notebook implements that two-session rehearsal for the user to run, without full training.

## Readiness

**READY_FOR_RETRAIN = NO; GPU_RUNS = 0.**
P0/P2 have CPU-only evidence; actual-model same-T4 performance and profiler attribution remain
pending. P1 CPU equivalence is passed. P3 CPU process/file persistence passes, actual Kaggle
retention is pending. P4 requires user review and frozen source/config/assets/seed/hardware
approval, preserving the original initialization and data identities. The ordinary launcher
cannot start full retraining until this review passes. No confirmation data were accessed.

Remaining bottlenecks: hard-routing nonzero/grouping, legality Tensor-to-bool checks, small
kernels and Python gather/scatter remain. Microbatch 2/4 is benchmark-only; it is not adopted
for science. No science thresholds, parameter geometry, optimizer schedule or target budget
were changed. Old partial logs/checkpoints remain old-source evidence.

The delivery source adds the unchanged-threshold M0 stability-window compatibility fix and
explicit synthetic registration metadata after measuring c9b58b0. Its timed engine and
model/Router/loss/accounting/diagnostic files are byte-identical to the measured repair.
The bounded GPU restart rehearsal accepts FP32 state at rtol 1e-5 / atol 1e-6 while
requiring exact RNG bytes and hard action traces; CPU state remains bitwise checked.

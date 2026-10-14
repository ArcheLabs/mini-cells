# PD-001 performance repair

## Scope and current decision

Baseline: `2e77369b0df1ce704863a815dacc48301736dde0`, the actual canceled Kaggle run.
Development base: main `688f601ee5e69a3dd9c03835b45993be1a40450f`.
The new implementation is `pd001-light-v1`. Old logs and checkpoints remain separate evidence.

**READY_FOR_RETRAIN = NO. GPU_RUNS = 0 for the implementation agent.**
CPU correctness and synthetic performance evidence are engineering evidence. Actual same-T4
benchmarking, retained Kaggle Output restoration, and user review of the frozen source/config
are still required. A successful forward pass or training checkpoint proves no CLM hypothesis.

## Confirmed code boundary

`joint_loss` consumes terminal CE, trajectory log probabilities, critic values, entropy and the
model's actual `cost_units` operation proxy. `Counter`, `touched_weights`, gradient diagnostics
and `TrainingGemmCounter` do not contribute to that objective. Model, Router and loss source
are byte-identical to the actual-run commit. All original scientific configuration fields are
unchanged; only a top-level engineering performance profile was added.

The previous pipeline always captured states/retained gradients, cloned every parameter,
computed full gradient/update/path diagnostics, converted many scalars to Python, ran persistent
forward hooks and intercepted every operator, then performed greedy and sampled probes every
step. These are code facts. Their T4 time contributions are **not inferred from CPU measurements**.
The old 9.49/1.34-second observations came from different parallel GPUs and are not a controlled
algorithm-speed comparison.

## Changes

| Component | Normal light profile | Instrumented/isolated audit |
|---|---|---|
| State capture / retain_grad | only at diagnostic steps, default every 100 | every step in explicitly bounded instrumented runs |
| Whole parameter clone / update norms | only at diagnostic steps | retained for ablation |
| Gradient diagnostic transfer | aggregate norms on device; one scalar-batch transfer | old implementation preserved in baseline checkout |
| Forward hooks / touched weights | absent during training | frozen-checkpoint profile, or bounded instrumentation |
| TorchDispatch GEMM interception | absent during training | retained for isolated measurements |
| Greedy + sampled probes | every 100; private seeded generator | old every-step probes in baseline/ablation |
| Scalar loss record | one batch transfer per step; aggregate objective/norm finite checks | original scalar conversion in baseline ablations |
| Atomic checkpoint | existing 100-step default; optional more frequent saves | IO excluded from step timing, measured separately in checkpoint tests |

Unsampled probe fields are null, not carried forward. Normal training reports `OFFLINE_ONLY`
and null FLOPs/work rather than fictitious zero or cumulative counts. `Counter` remains validated;
module parameter inventories are cached per model. Unique visits are not executed parameter work.

No Cell was removed, no path shortened, no soft routing added. Self-connections execute again.
STOP, forced stop, zero-action STOP, actor/critic state dependence and J1 dual independent paths
are retained. Default microbatch remains **1**, effective batch **16**, sequence length **125**,
steps **5000**, budget **10M target tokens**, precision **float32**. Microbatch 2/4 appears only in
isolated B6 measurements and requires a new execution identity if adopted; its sampled trajectories
are not claimed bitwise equivalent. Finite gradient norms now fail closed rather than allowing
an overflowed clipping norm to silently corrupt an optimizer update.

## Measurement

`perf-audit` uses fresh processes and synthetic integer token IDs. B0 executes the original
`_train_registered` function from a clean baseline checkout; only checkpoint writes/final evaluation
are substituted with synthetic scaffolding. Wall time is bounded by original zero-grad and save
sites and excludes checkpoint IO/final evaluation. B0's separate one-step profiler also invokes
the original function. B1–B6 execute the equivalent extracted step with original-source model,
loss, accounting and diagnostic components, disabling one factor at a time:

B1 capture; B2 clone/gradient diagnostics; B3 hooks; B4 Dispatch; B5 probe frequency;
B6 microbatch 2 or 4. LIGHT additionally samples routing diagnostics and batches scalar/finite
checks, using the repaired source. CPU fixtures use width 12, two Cells, four-token sequences and
an effective batch of four, while retaining the arm's real 16-action budget. These fixture speeds
**cannot forecast actual-model T4 throughput**. The default Kaggle benchmark uses the original
full model shapes/batch. CUDA step-boundary synchronization is outside individual operators;
profiling is outside the measured interval. Unavailable CUDA metrics are null. CPU item events
are not proof of CUDA host synchronization. Kernel totals describe only the separate trace window.

Reports: `reports/performance/baseline.json`, `ablation.json`, `correctness.json`, and
`profile_summary.md`. Raw per-repeat step samples, operator summaries, source-file hashes,
implementation identity, fixture flag and measurement limits are retained. Large Chrome traces
remain local/Kaggle Output, with checksums; they are not pushed as research Git blobs.

## Run the bounded Kaggle checks

Import **performance-notebook.ipynb** into the existing minicells notebook; enable Internet and
T4, stop other GPU jobs, then Run All. It sequentially measures M0/P50-F16 on GPU 0 using
5 warmup + 20 measured steps, three repeats. No scientific dataset/confirmation assets/secrets
are used. It also creates a two-step synthetic resumable checkpoint. It never runs 5000 steps.

For a shorter first pilot use 3 warmup + 10 measured steps; output is labeled pilot. Save Version
must actually retain the output. In a **new** session attach that saved Output, set RESTORE_FROM
to the portable directory, then Run All. The benchmark is skipped, a hash-verified same-source
checkpoint resumes, and final state is compared with an uninterrupted four-step control. Record
both version links and successful receipt for review; code deliberately cannot assert provenance
of retained cloud files just because a local path exists.

CLI (repository root):

```bash
python research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/run_pd001.py --mode perf-audit --arm P50-F16 --device cpu --steps 5 --baseline-root /path/to/clean/2e77369-checkout --output /fresh/performance-output
```

The CPU command retains full model dimensions by default; use `--perf-fixture` for the documented
small engineering fixture. CUDA requires the explicit `--allow-gpu-microbenchmark` switch.
The CLI caps measurement to at most 30 measured/10 warmup steps and five repeats; these bounds
apply to engineering benchmarks, not the scientific path budget.

## Resume and review boundaries

Portable bundles contain run checkpoints/metadata, exact asset files and execution identities;
no code, Git repositories, worker secrets or credential settings. Import validates every file
hash, rejects traversal/symlinks, and requires the exact source/config identities. The existing
atomic checkpoint format retains AdamW, Python/NumPy/Torch/CUDA RNG and independent trajectory
generators. Optional `pause_after_step` and `max_session_seconds` act only at completed optimizer
steps; default is **no imposed session deadline**. Pauses save atomically, return
`PAUSED_RESUMABLE`, and suppress final profiling/publication. Completed arms verify checkpoint
hash/identity and skip optimization. New cloud sessions must import persistent Output; merely
repeating Run All does not preserve `/kaggle/working`.

`authorize()` is still called on every scientific training/evaluation/publication path. Full
retraining additionally requires a user-approved review binding source/config and correctness,
same-GPU performance, persistence and approval evidence hashes. Automatic development remains
isolated from confirmation. P4 approval should bind the actual frozen execution config, including
its generated asset hashes and seed; it must not bind a preliminary proposal then silently change
its execution identity. Rerun all four core arms from step zero with the same repaired implementation.
Old M0 results cannot be mixed with new CLM runs for the primary comparison.

## Remaining risks

Dynamic `where/nonzero`, legality Tensor-to-bool checks, tiny per-Cell kernels, index gather/scatter
and host scheduling remain in the executor. This iteration intentionally does not rewrite it,
introduce all-Cell evaluation, compile it, change precision or adopt larger microbatches. Their
remaining T4 cost must be measured before any segmented-dispatch redesign. Original model scale,
path-learning variance and language quality are unchanged by engineering speedups.

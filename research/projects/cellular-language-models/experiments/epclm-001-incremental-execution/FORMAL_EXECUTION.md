# EPCLM-001 v0.4 execution contract

The preserved user protocol is `PROTOCOL.v0.4.md`. The active inference-only
plan is `formal_config.json`, checked against its SHA-256 marker on every
invocation. Historical protocols, training code, checkpoints and reports remain
unchanged. No E1 command creates an optimizer or dispatches a training worker.
30M and additional training remain excluded.

Use `epclm001_formal_e1.ipynb` in Kaggle with **one GPU**, Internet, and the
existing `HF_TOKEN` and `GITHUB_TOKEN` secrets enabled. Run All performs the whole
bounded experiment. The notebook pins the implementation commit automatically;
there are no manual seed/hash fields or required dataset Input attachments.

The default sequence is:

1. `e1-preflight`: verify external persistence/readback, three exact HF asset
   revisions and weight hashes, tokenizer, strict model loading, original model
   state fingerprints, training/source/corpus/quality bindings, validation stream,
   fixed held-out slices and hardware identity.
2. `e1a-correctness`: trained T1/X3/X4, A0/A1, batch 1/2, all prefixes through
   128, full-reference stage/iteration KVs, causality, chunk/reset/restore/cache
   isolation and wrong-checkpoint rejection. A failure stops the sequence.
3. `e1a-benchmark`: separate short pilot then all five fixed model workloads.
4. `e1b-correctness`: all four executors, B0/B1-X3/B1-X4, 16 fixed event points,
   plus read/reset/stale-message/next-step visibility controls.
5. `e1b-benchmark`: separate pilot, full registered event matrix and separate
   profiler pass. B1 uses native-width norm/SwiGLU/residual, not attention or a
   learned event policy. Four microsteps and tile 16 are fixed.
6. `e1c-benchmark`: cached T1 parallel prefill, X3/X4 full/A0/A1, fixed teacher-
   forced inputs; separate profiler and bound quality reuse records.
7. `e1-analyze`: paired uncertainty, summary statistics, competitive table,
   independent H1/H2/H3 decisions, nine figures and English/Chinese summaries.

Each command accepts `--output`, `--assets`, `--publish-github`; `e1-run` chains
these commands, `e1-publish` returns existing evidence without rerunning inference.
For a single-stage retry, select that stage in the notebook; prerequisites are
still enforced. E1-B has no E1-A success dependency, but shares asset/environment
preflight in this first-round launcher. After an E1-A failure it may be invoked
separately if its own preflight and correctness requirements pass.

## Freeze and interpretation

First round is FP32 only, with TF32 disabled and `atol=2e-5, rtol=2e-4` frozen
before tests. A failed tolerance check blocks performance; it is not relaxed
from observed results. FP16/BF16 are explicitly not run. Source commit, actual
GPU UUID/model/driver, PyTorch/CUDA/Python, inputs, matrix and repeats bind into
`FORMAL_FREEZE.json` after pilot and before formal sampling. A different source,
config or device cannot append samples to the same experiment.

Every point uses 5 warmups and 30 retained repetitions. Model variant order
rotates and executor order alternates; no outlier is removed. Pilot uses 1/3
warmup/repeats on fixed endpoints; pilot results never select the formal matrix.
Primary H1/H3 metric is decode host time; H2 is executor host time versus D0.
Paired percentile bootstrap intervals use 10,000 draws, seed 2026103005 and
Bonferroni families 50 (20 H1 plus 30 H3 comparisons) and 96 (16 x 3 x 2 H2).
Prefill/total and D1 comparisons are descriptive. These approximate timing-
repetition intervals are hardware-specific, not independent model-seed inference.

Model timings exclude load/tokenize/sample/return. CUDA events measure stream
intervals, including submission gaps, **not** kernel-only time. Kernel counts
and event durations come from separate profiler traces. DRAM traffic and energy
are `UNAVAILABLE`, not estimated from sparse activity. FLOPs cover analytic
matrix products only. TTFT and full user end-to-end latency are `NOT_MEASURED`.
Cached storage bytes include backing allocations, not only visible tensor views;
working-memory observations and unattributed allocation overhead are explicit.
No allocator traffic or energy efficiency claim follows from partial FLOPs.

Quality can be reused only when exact weight, QUALITY/IDENTITY hashes,
validation stream, tokenizer and original evaluator bytes match. Records say
`quality_reused=true`, retain FP16 evaluation dtype, and do not claim a fresh
FP32 quality evaluation or equal-quality comparison. A changed evaluator blocks
reuse rather than silently substituting historical PPL.

## Recovery, cost and return

The new 7,200-second GPU-worker wall-time budget is independent of v0.3 training.
The shared file lock forbids simultaneous local training/inference workers.
Before each GPU stage, the remaining allowance is reserved durably on HF; only
observed cleanly completed worker time releases the unused reservation. A hard
watchdog terminates the worker at its remaining allowance. A lost process leaves
the full reservation charged and the run incomplete; it never resets budget.
CPU/network setup outside a GPU worker is not charged as GPU-worker time.
This is a conservative wall-time policy, not measured GPU utilization.

Stage outputs and individual samples have local hashes and are reused. Stage
boundaries, and each reservation before GPU work, are independently uploaded and
fresh-downloaded for SHA verification under `epclm001/e1-formal-001` in the
existing HF repository. A new cloud session restores that controller. A lost
session may lose samples since the last remote stage boundary; its outstanding
reservation is retained, so no undocumented rerun allowance is granted. If all
budget is reserved after a crash, analysis/publication remain available but new
GPU work stops for review. Do not delete/reset the controller to retry.

Reports automatically return through the existing verified GitHub publisher to
`results/epclm001-e1-formal-*`, descending from current main without merging it.
No weights go to Git. Large raw/profiler files stay in verified HF storage and
Kaggle Output, with Git coverage explicitly listing omissions. Publication can
be retried without inference. Independent result review is still required.

Initial engineering status: CPU tests can validate implementation, but formal
H1/H2/H3 remain `INCOMPLETE` until the real pinned assets and on-device gates run.
No 30M expansion is automatic, even following positive results.

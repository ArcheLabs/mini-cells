# Concurrent deterministic training v1

`echo-adamw-ce-tree32-v1` is a second execution path alongside the validated
sequential MCA1/MCF1 path.  It keeps the Echo model, CE objective, logical
batch 256, FP32, clipping, AdamW, and optimizer semantics unchanged; only the
raw-gradient reduction order changes.

Each of 32 leaves computes eight samples against the same frozen model and
returns an `MCG1`/`MCGR` raw partial gradient.  A root `MCRF1` envelope accepts
all immutable leaves, validates job/model/optimizer/batch/step and sample
ranges, reduces them with a fixed five-level pairwise tree, then performs one
normalization, one global clip, and one AdamW update.  No leaf updates weights,
Adam state, or the optimizer step.

The Native core exposes `compute_gradient_leaf`, `merge_partial_gradients`,
`reduce_32_leaves`, and `train_step_tree32`; the lab's `parallel-native`
command schedules completion orders deterministically by indexed slots.  The
MiniJAM worker branch `agent/minicells-parallel-refine` adds a wrapper-level
`ExecutionLaneId`, deterministic `work_id % lane_count` assignment, an ordered
lane executor, and refine queue metrics.  These are a concurrency compatibility
layer, not a claim of full JAM multi-core protocol conformance.

The root payload is 630,823 bytes, below the 1 MiB WorkPackage limit.  The
current runtime audit still reports a 6B block aggregate execution budget and
two duties per worker, so actual chain overlap and speedup remain a separate
fail-closed gate.  See `artifacts/parallel-training-v1/decision.json` for the
current evidence state; direct tree-guest execution is currently blocked by a
PVM panic before gas accounting, and fresh-chain E2E remains blocked when no
Docker daemon is available.

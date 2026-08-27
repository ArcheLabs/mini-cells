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

The canonical 5000-step Native tree32 run reaches 1.0 token accuracy and 1.0
exact-sequence accuracy; its trajectory and provenance are recorded under
`artifacts/parallel-training-v1/native/learning-5000.json`.

The root payload is 630,823 bytes, below the 1 MiB WorkPackage limit.  The
rebuilt MCG1 leaf guest reports 2,115,674,786 gas for the canonical eight-sample
shard and 2,310,530,456 gas for a valid all-`MAX_SEQ_LEN` shard; both are
`SHARD_PASS_FULL_COMFORTABLE`.  The full MCRF1 root bit-exact probe remains
fail-closed and is not claimed.  The current runtime audit still reports a 6B
block aggregate execution budget and two duties per worker, so actual chain
overlap and speedup remain a separate gate.  See
`artifacts/parallel-training-v1/decision.json` for the current evidence state;
fresh-chain E2E remains blocked when no Docker daemon is available.

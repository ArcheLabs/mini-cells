# Concurrent deterministic training V2

`echo-adamw-ce-hierarchical-tree128-v2` is the Stage-1-compatible execution
path alongside the sequential MCA1/MCF1 path. It keeps the Echo model, CE
objective, logical batch 256, two-sample shards, FP32 arithmetic, clipping,
AdamW, and optimizer semantics unchanged. Only the transport decomposition is
new.

The 128 immutable leaves compute two samples each against one frozen model and
return `MCG1/v2` → `MCGR/v2` records. Four ordered `MCR2` reducers each accept
exactly 32 contiguous leaves and return an `MCGR2` partial gradient. The `MCF2`
finalizer accepts the four contiguous group records, performs the same
balanced binary association as the flat 128-leaf tree, then does the only
normalization, global clip, AdamW update, and optimizer-step increment. Leaves
and reducers never carry or mutate weights, Adam state, or optimizer step.

Every intermediate record binds `job_id`, `generation`, `optimizer_step`,
`model_commitment`, `optimizer_commitment`, `batch_commitment`, group/leaf
index, and exact leaf/sample range. Reducers require canonical leaf order and
the finalizer requires canonical group ranges. Thus the hierarchy is not an
approximation: the four 32-leaf subtrees plus the two top levels are exactly the
old balanced 128-leaf tree.

The Native core exposes `compute_gradient_leaf`, `reduce_group_in_place_ref`,
`reduce_four_groups_in_place`, and `train_step_tree128`. The lab's
`hierarchical-pvm` command executes all 128 leaf guests, four reducer guests,
and the finalizer through the public `minijam-pvm-executor`.

The recorded run in `artifacts/stage1-training-compatibility/hierarchical-pvm.json`
is bit-exact against the hierarchical Native oracle. Maximum Work sizes are
18,190 bytes (leaf), 578,400 bytes (group), and 126,170 bytes (finalizer),
under the 1 MiB Formal RPC boundary. Gas maxima are 581,194,702, 13,908,329,
and 11,985,063 respectively, under the 1-billion limit.

`ParallelTrainingJobV2` models the DAG states `LeavesReady`, `LeavesRunning`,
`GroupsReady`, `GroupsReducing`, `FinalizerReady`, `Finalizing`, `Complete`,
and `Failed`, including four group commitments and exact ranges. The former
single-root `MCRF1` implementation and its 2,361,703-byte payload remain
historical evidence only. Fresh-chain service creation, a complete on-chain
training step, and restart persistence are still separate gates and are
recorded as blocked when the exact chain environment is unavailable.

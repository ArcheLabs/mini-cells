# MiniCells × MiniJAM Stage-1 compatibility

This page is the source of truth for the current protocol result. The
scientific and PVM gates now pass with the hierarchical V2 ABI. The final
decision remains blocked until the same artifacts are deployed on a fresh
Stage-1 chain and restart persistence is recorded.

| Property | Value |
| --- | --- |
| Stage-1 contract | [`configs/minijam-stage1-contract-v1.json`](../configs/minijam-stage1-contract-v1.json) |
| Contract identity | `minijam-stage1/v1`, SHA-256 recorded in [`configs/dependencies.json`](../configs/dependencies.json) |
| MiniJamSpec | v1 |
| Guest SDK ABI | 1 |
| Logical batch | 256 samples |
| Leaf topology | 128 leaves × 2 ordered samples |
| Hierarchy | 4 groups × 32 leaves, then one 4-part finalizer |
| Reduction | balanced FP32 tree: four 32-leaf subtrees, then two top levels |
| Refine/Accumulate limits | 1,000,000,000 gas |
| Work boundary | 1,048,576 bytes |
| Fresh-chain E2E | BLOCKED — exact runnable chain/credentials are unavailable |

MiniCells consumes the public `minijam-pvm-executor` contract. The dependency
lock contains only the public Stage-1 contract/ABI identity. MiniJAM source and
its internal Jambda implementation are not MiniCells CI dependencies.

## V2 execution protocol

Each leaf consumes an `MCG1/v2` payload containing the complete job,
generation, optimizer-step, model, optimizer, batch, and exact two-sample range.
It returns an immutable `MCGR/v2` raw gradient record. Four ordered `MCR2`
reducers each validate 32 records and return an `MCGR2` partial gradient. The
`MCF2` finalizer validates all four group ranges, combines them with the same
balanced-tree association as the flat 128-leaf oracle, and performs the only
normalization, clipping, AdamW update, and optimizer-step increment. Reducers
carry no weights or Adam state.

The application state machine is `ParallelTrainingJobV2`: leaves move through
`LeavesReady`/`LeavesRunning`, groups through `GroupsReady`/`GroupsReducing`,
then `FinalizerReady`/`Finalizing`, and finally `Complete` or `Failed`.
Every accepted leaf and group is bound to the same identity tuple and exact
leaf/sample range; duplicate and conflicting commitments are rejected.

## Measured local gates

The real public `minijam-pvm-executor` run is recorded in
`artifacts/stage1-training-compatibility/hierarchical-pvm.json`:

- hierarchical Native and PVM results are bit-exact;
- leaf Work is 18,190 bytes, group Work is 578,400 bytes, and finalizer Work is
  126,170 bytes — all below 1 MiB;
- leaf gas is 503,092,622–581,194,702, reducer gas is 13,850,410–13,908,329,
  and finalizer gas is 11,985,063 — all below 1 billion;
- comparison with the monolithic training step is diagnostic; the acceptance
  gate is exact parity with the frozen balanced-tree oracle.

The old single-root `MCRF1` experiment remains historical evidence: its
2,361,703-byte payload cannot cross the 1 MiB Work boundary and is not the V2
deployment path.

## Decision policy

`artifacts/stage1-training-compatibility/decision.json` deliberately reports
`BLOCKED_FRESH_CHAIN`, not PASS. A PASS requires all of the following on a
fresh chain using a released, digest-pinned MiniJAM OCI image: service
creation finalized, one complete hierarchical training step, canonical state
update, and restart persistence. No cached or unverifiable image is accepted
as chain evidence.

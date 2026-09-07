# MiniJAM Stage-1 compatibility

This file is the source of truth for the current compatibility result. It must
not be read as a claim that on-chain pretraining has passed.

| Property | Value |
| --- | --- |
| MiniJAM commit | `b90c0bffa09fa0190fb1737db876190ddd899c22` |
| Jambda gitlink | `d33e0abf8116b23bbc551c6a8d7075eacb2994ce` |
| MiniJamSpec | v1 |
| Guest SDK ABI | 1 |
| Logical batch | 256 samples |
| Leaf topology | 128 leaves × 2 ordered samples |
| Reduction | deterministic balanced tree over leaf indices 0 through 127 |
| Refine limit | 1,000,000,000 gas |
| Fresh-chain E2E | NOT RUN |

MiniCells now consumes `minijam-pvm-executor`; it does not check out a second
Jambda revision or maintain host-call numeric IDs. The dependency lock is
`configs/dependencies.json`, and MiniJAM's own Jambda gitlink is authoritative.

## Current blocking result

The existing root ABI embeds all raw FP32 leaf gradients in one Work payload.
At 128 leaves the encoded payload is 2,361,703 bytes, while the canonical
Formal RPC boundary accepts at most 1,048,576 bytes. Consequently the guest
build is an engineering result only and is marked `deployment_ready: false`.

The next compatible protocol revision must add deterministic hierarchical
reduction (for example four ordered 32-leaf reducers followed by one ordered
four-part finalizer), with every intermediate bound to the same job,
generation, model, optimizer, batch, and exact sample range. Numeric parity,
gas distribution, chain deployment, finalization, and restart persistence must
then be rerun. Until those gates produce finalized evidence, the statement
“MiniCells can perform native pretraining on MiniJAM Stage-1” is not valid.

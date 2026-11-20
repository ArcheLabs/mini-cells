# Architecture review

Status: partial implementation. CPU tests cover token causality, outer
gradients, Cell-only restore, compatibility rejection and fork ownership.

The isolated Qwen hook captures pre-RoPE features, routes four queries
per shard and blends the read before the native output projection.
The mask and cache are still supplied by the caller. Integrated bounded
local attention, independent Session restore, routing across multiple
Cells and pretrained Qwen numerical tests remain pending.

The outer step freezes Qwen and optimizes new memory and fusion
parameters. A full training runner, checkpoint/resume validation and
T4 validation remain pending. No training was run. The serial chunk
reference has no throughput claim.

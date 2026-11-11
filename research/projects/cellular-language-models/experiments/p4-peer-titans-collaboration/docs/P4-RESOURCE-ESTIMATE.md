# P4 resource estimate

M1.0 measured 5.84 seconds on CPU for 2,048 inner calls including the control and checkpoint overhead. The blended 2.85 ms per call is a planning anchor. Single-item calls have different kernel cost and checkpoint overhead does not scale linearly.

| Screening group | Calls per episode at N=64, exposure=128 | Calls across configured sizes and 20 mappings per size |
|---|---:|---:|
| Gaussian Batch | 128 | 12,800 |
| Gaussian Sequential | 8,192 | 317,440 |
| Gaussian Chunk 4 | 2,048 | 79,360 |
| Gaussian Chunk 8 | 1,024 | 40,960 |
| QR Batch through N=32 | N/A | 10,240 |
| Total | | 460,800 |

This base matrix processes 1,423,360 association presentations. Calls equal exposure times ceil(N/width). Using the blended CPU rate gives about 23 minutes before extra metrics, restore, order comparisons, forgetting or extended curves. At N=64, a single 128-exposure episode projects roughly 0.36/23.4/5.8/2.9 seconds for Batch/Sequential/Chunk4/Chunk8. Extending all four Gaussian N=32 and N=64 groups from 128 to 512 adds 1,029,120 calls at 20 mappings per configuration. Select extended configurations only after reviewing screening.

One H=64 Cell has 34,048 dynamic FP32 bytes. Two have 68,096 bytes. An H=129 near-match has 67,848 bytes, 248 bytes below the two-Cell total. M2 planning fixture budget is at most 768 four-binding update calls for A/B and fork controls, roughly 2.2 seconds acquisition at the blended rate, plus fresh interpreter, serialization and fault-fixture overhead. Measure total time in CPU smoke.

GPU time and peak VRAM are unmeasured. Budget from an actual synchronized smoke using per-width call latency multiplied by call counts plus evaluation, checkpoint and publication. Sequential tiny calls can be launch-bound. CPU timing supplies no validated T4 throughput estimate.

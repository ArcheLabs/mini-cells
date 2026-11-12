# M3-A Peer Titans CPU Smoke

Gate A PASS. Three seeds with five paired episodes each ran in both capacity modes. Each episode freshly generated eight Gaussian associations and stored its mapping, orders and tensor hashes. Each association received 128 presentations using Chunk 4. P0 committed 256 calls and each peer committed 128.

| Union width | Peer dynamic bytes | Union dynamic bytes | P0 joint Recall@1 | P2 joint Recall@1 | Swapped-query joint Recall@1 |
|---:|---:|---:|---:|---:|---:|
| 64 | 68,096 | 34,048 | 100% | 100% | 0% |
| 129 | 68,096 | 67,848 | 100% | 100% | 0% |

All 30 batch/serial pairs matched bitwise. P1, P2 and P3 agreed on both slots. P2 concatenates ordered reads. Branch ablations zero the omitted slot, mark it unavailable in metrics and preserve the retained slot. A/B state hashes and storage checks verified independent updates and read-only evaluation. After the writer exited, a new interpreter restored all 60 A/B Cells and reproduced all 30 fused output hashes. A separate historical M2 restore regression passed. Both union and peer reached the ceiling in this fixture.

Shared peer static state is 28,931 parameters and 115,724 bytes. The 129-width union has 33,156 parameters and 132,624 bytes. Each static interface has 24,576 projection parameters. Two acquisition paths across both modes used 15,360 updates and 61,440 presentations in about 24.32 seconds. Per-episode call counts, acquisition time, batch/serial dispatch times, checkpoint bytes, serialization and restore times are in the [writer](../m3-composition/peer-titans/runs/cpu-smoke/writer.json) and [reader](../m3-composition/peer-titans/runs/cpu-smoke/result.json) records. CPU batch dispatch verifies interface parity. Hardware overlap and speedup require a separate measurement.

Reproduce with `python m3-composition/peer-titans/run.py /tmp/new-peer-output --historical <completed-m2-output>`. The historical directory must contain the original M2 inventory and Cell checkpoints.

# Reuse map

Paths below are relative to the experiments directory.

| Source | Reuse |
|---|---|
| epclm-004-unified-memory-10m/m3/freeze/original-source/memory.py | Byte-identical to m3/memory.py. Direct NeuralMemory.create, recall, learn, checkpoint, restore and state_bytes. |
| epclm-004-unified-memory-10m/m3/freeze/original-source/gates.py | Exact gate5 control, including deterministic seed, 128 online updates and fresh-session restore. |
| epclm-004-unified-memory-10m/m3/freeze/ARCHIVE-STATUS.json | Archive identity and prior remote readback receipt. |

Explicit latent tensors have shape [batch, time, memory_dim]. Query equals Key. The frozen online update detaches static projections and controls from the gradient path and returns new fast weights and momentum. Read-only recall takes only query and Cell and performs no write. Checkpoint stores both components with independent hashes. M1 adds only synthetic data, scoring, state verification and run bookkeeping. Candidate values are used by scoring only.

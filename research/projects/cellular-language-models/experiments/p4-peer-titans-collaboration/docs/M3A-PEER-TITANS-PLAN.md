# M3-A Peer Titans preparation

## Verified interfaces

P4 run_m1.py uses latent [batch,time,32] keys, values and queries with NeuralMemory(256,32,64). Frozen M3 memory.py supports independent Query tensors and per-stream weights. M2 verify.py creates A/B separately, verifies disjoint tensor storage and restores each checkpoint with a validated static interface. The read helper accepts one stream and verifies dynamic state and metadata are unchanged.

Audit probe: restore A and B from the completed M2 CPU writer inventory, then stack distinct Query orders and concatenate copies of the Fast Weight and Momentum tensors along the batch axis. Frozen recall produced identical outputs to two separate reads (max absolute difference 0). Source Cell hashes remained unchanged. A read adapter can use this transient batched view for homogeneous Cells. Persist and update the original Cells independently. Heterogeneous widths require separate calls. This probe covers dispatch and isolation, while the planned Smoke covers actual composition.

## Planned experiment

Generate two independent Gaussian domains with 4 associations each for Smoke and scale to the approved larger sizes for Screening. Sample Keys and Values independently, use nonidentity random assignment and ensure no cross-domain Key collision. Generate disjoint seeds for episodes, draw orders and queries. Freeze the candidate bank and score all arms against it. Only the scorer sees the target labels. Branch dispatch uses an explicit domain/slot supplied by the task.

- Q1 reads one named Cell. Q2 takes two explicit Keys, reads A and B, and concatenates ordered output vectors. Score both slots and their joint correctness. Q3 swaps the domain dispatch and reports acceptance and retrieval errors.
- P0 trains one union Cell on A+B and reads both queries. P1 measures isolated reads with no fused prediction. P2 dispatches both reads as a packed batch and concatenates. P3 performs the same independent reads serially and concatenates, a scheduling control with equivalent expected outputs. P4 swaps queries. P2-A-only and P2-B-only zero the omitted slot and score that slot as unavailable.
- Keep P1–P4 and ablations on the same acquired A/B checkpoints. Check no cross-write, exact read-only state hashes, equivalence to independent references and fresh-process reconstruction from the M2 inventory. Different observed execution latencies require synchronized repeated timings.
- Mode A uses hidden width 64 for every Cell and reports the larger total dynamic bytes of P2. Mode B uses width 129 for P0 and 64 for each peer. M3 exposes width via its constructor. Each configuration needs its own reproducible initializer and compatibility hash. Report the residual byte and compute differences.

Run equal exposure using Chunk 4 and 128 presentations per association as a candidate Smoke budget. Interleave union chunks deterministically and preserve identical per-domain associations. Equal-compute comparisons need separately frozen calibrated time or call limits. Identical call counts do not guarantee matched compute across widths. Save actual presentations, calls, state bytes and Recall rather than equating these budgets.

Before Gate A, validate output slot ordering, swapped queries, removal of each branch, storage isolation and two-Cell restoration in a fresh process. Scientific hypotheses about interference and scalability await paired Screening.

# P4-M2 CPU isolation results

PASS. The writer process 5430 exited before a different interpreter restored A, B and both forks. All four recovered state and output hashes matched. A and B had identical initial dynamic values and disjoint storage. A-to-B and B-to-A updates preserved the peer state, output, counter and revision. The two descendants acquired different four-key assignments at 100% Recall@1 while the parent stayed unchanged.

The static interface was one frozen M3 Memory instance per process. Fast Weights and Momentum are the only dynamic optimizer state in this latent path. Active tensors retained no update graph. No static parameter gradients or writes occurred.

Ten fail-closed cases passed, including corrupt tensor bytes with recomputed outer hashes and wrong dtype with valid serialization and hashes. The first reader run surfaced an UnpicklingError for corrupt tensor bytes. The test was updated to recognize that explicit rejection and rerun against the completed writer checkpoints.

| Cell | Checkpoint bytes | Recall seconds |
|---|---:|---:|
| a | 55,755 | 0.000547 |
| b | 55,883 | 0.000646 |
| a1 | 56,601 | 0.000477 |
| a2 | 56,025 | 0.000654 |

A/B acquisition wall 0.352 seconds for 256 calls. One Cell has 34,048 dynamic FP32 bytes and two have 68,096. The H=129 near-match has 67,848 bytes. This verifies latent Memory isolation on CPU. Model-level Backbone/tokenizer integration and target-GPU timing remain pending.

Run with `python m2-isolation/verify.py /tmp/p4-m2-new` from the P4 directory. The inventory and result JSON retain the independent checkpoint hashes and rejection evidence.

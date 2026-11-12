# M5 baseline audit

Read from the archived decision and metadata on main, not from chat descriptions. Audit baseline: main ed4eb218180adeff9876f34b269f359e84caba1e; M5 result 4887883a01ffe0e5bd8515524ac13cd6a2a6b120; training source d004704ff63e8a99fc474a88f144bac4b9438894. All 50 published metadata hashes match. See m5-baseline.json for original ledger/checkpoint identities.

Gates 0–7 passed. Five arms (E3, E6, E7-Random, E7-FIFO, E7-Innovation) each committed two T4 FP32 updates, 1024 input tokens and two supervised query targets. Train bank: 16 episodes; validation: four episodes. The additional evaluation used 22 one-episode conditions and 60 queries per capacity. There was no independent test; no evidence of adequate training. All final answers were wrong. This is engineering evidence only.

Model checkpoint hashes are available, but actual M5 model and Store binaries are absent from Git; availability in the saved Kaggle Output has not been independently rechecked here. Do not claim a long-term M5 weight archive exists. SV uses fresh initialization; E3 historical reference remains explicitly smoke-only. Original checkpoints and results are never changed.

Actual representation is the original input embedding, not a contextual key/value pair. One global bounded Store per stream is shared by all SWA layers. FIFO removes oldest; Random is seeded reservoir sampling; Innovation replaces the minimum-scored record only for a strictly larger score. The predictor is parameter-free EMA (beta=.875) of the last Mamba layer output; innovation is mean squared error. Read uses Query similarity, not admission score. Summary pools channels of the previous-chunk SSM state and projects to four tokens per SWA layer. Writes occur after all readers finish each 32-token chunk.

These are approved M5 implementation choices, not exact B'MOJO paper equivalence. Input embedding storage does not encode the binding relation by itself: selection of a value token does not make its key available to the reader. Oracle admission therefore diagnoses selection capacity, not a guaranteed answer oracle or retrieval oracle. No change to this representation is authorized by SV.

Unresolved: M5 Script Version ID is null; the final publication's embedded receipt references its first publication commit; binary checkpoints have hashes but no audited long-term archive. Snapshot isolation was not part of the original multi-query evaluation. SV adds a new versioned evaluation protocol without rewriting M5 history. M4 weights are archived separately. H5 remains NOT_EVALUATED before SV runs.

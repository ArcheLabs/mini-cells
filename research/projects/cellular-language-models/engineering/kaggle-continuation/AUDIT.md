# Continuation engineering audit

Baseline: `52b1950b23debbcd3f808c10eb6e0d3ea2d317c6` (R3). This work is infrastructure only. Existing R3 Notebook, algorithm, protocol, config and results are not modified or rerun.

- Checkpoint: reuse R3 save/restore (PD-001 → Experiment 006 → Experiment 005), all model/optimizer/path RNG/data RNG/global RNG/Registry/private Candidate/Event state. Historical selection uses the exact indexed file and event revision, never a first glob match.
- Runner: preserve Trainer.update, independent RNG streams, event process and promotion. Add an explicit safe segment boundary in the engineering adapter; total target remains frozen.
- Automatic: preserve original functional/credit/beta gates for any future real run; resumed packages verify original runtime/gate evidence without recalibration. New engineering entry refuses unregistered real training.
- Monitoring: reuse initial bank and specialization snapshots. No trained-state regeneration of missing frozen probes. New engineering plans are frozen before assets; old R3 snapshots are never invented.
- Publication: reuse metadata-only authenticated helper; dedicated engineering metadata allowlist. Input copies and publication failures cannot modify parent Outputs or delete checkpoints.
- Notebook: pin a full code SHA; use mounted read-only previous Output. Only synthetic engineering 0→6→10 and historical evaluation are in scope. No R3b authorization.

The original R3 validator intentionally fixes its scientific experiment/arm identity. An engineering envelope uses its own distinct run/experiment ID while the inner algorithm fixture retains its real R3 identity. Both are explicitly recorded; it must never be reported as new R3 evidence.

Without a successful Kaggle version save and next-session mounted Input, persistence remains NOT_VERIFIED. Local copies are CPU coverage only.

# Smoke test

Run repository-owned checks with `./tools/test_all.sh`. For a running local MiniJAM stack, set `MINICELLS_SIGNER_URI` and run `./tools/smoke_local.sh`; it deploys when no service ID is supplied, initializes via status probe, performs inference, advances one generation, and submits a stale training replay.

The expected invariants are: generation starts at 0, inference is read from finalized storage, PLUS alone cannot advance generation, PLUS/MINUS advance exactly once, history records the transition, and replaying generation 0 after generation 1 does not change the current model or create a pending entry.

Because interpreted PVM evaluation and independent voting are intentionally expensive, a complete local smoke may take many minutes. Do not reduce correctness polling to a short wall-clock timeout.

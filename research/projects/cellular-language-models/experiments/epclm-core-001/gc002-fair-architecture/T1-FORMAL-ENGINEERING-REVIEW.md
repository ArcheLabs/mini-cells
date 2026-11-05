# T1 Formal engineering review

This review concerns the active `t1_formal.py` launcher. It does not certify a T4 run or a scientific result.

## Executed checks

- All experiment Python modules: global-name scope inspection passed; Notebook code syntax parsed successfully.
- Existing experiment directory tests: **9 passed, 1 failed** (13.85 seconds on this CPU environment).
- The failing test is `test_pilot_committed_update_and_exact_resume`: historical `pilot_training.PilotEngine` attempts to snapshot KV caches invalidated by an optimizer update. The active formal launcher uses `formal_training.FormalTrainer`, full-sequence teacher forcing, and no cross-optimizer training KV cache. The historical Pilot is not repaired or certified by this change.
- Added one tiny synthetic CPU integration covering the actual `_smoke` helper: 50 committed updates, resume comparison after update 25, completion reload without additional updates, FLOPs report generation, `_save_formal_checkpoint`, SHA-backed recovery bundle restoration, checkpoint reload, and a final partial-token update with another reload.
- CPU fixture uses a small vocabulary and model. Only device selection and CUDA memory reset are substituted; optimizer, scheduler, sampler, exposure ledger, FLOPs counter, checkpoint serialization and recovery bundling execute their real implementations.

## Reviewed active boundaries

- Frozen seed 173020021, 97 warmup updates, FP32, 250,000 formal prediction targets / 977 updates remain unchanged. The final formal update is 144 targets.
- Training metrics contain cumulative `prediction_tokens` and incremental `update_prediction_tokens`; checkpoint validation now verifies incremental totals and every cumulative/update/FLOPs position.
- Smoke resume verification executes one next update on each branch and returns only one committed continuation. The comparison branch is not added to exposure accounting.
- Checkpoint and recovery manifests use the existing atomic, hash-verified infrastructure. Formal helpers can resolve the shared bundler.
- Notebook fetches the current readiness branch and chooses output by source commit; checkpoints from previous source identities remain incompatible.
- Execution/publication/safety budgets remain 5400/600/1200 seconds. GPU estimation and runtime deadlines are not certified by CPU tests.
- GPU profiling, target-runtime operator coverage, T4 numerical behavior, Kaggle recovery transport and authenticated GitHub result publication still require actual target execution. Previous successful result publication does not prove training completion.

No real dataset training, GPU run, historical-result edits, or formal budget increase was performed for this review.

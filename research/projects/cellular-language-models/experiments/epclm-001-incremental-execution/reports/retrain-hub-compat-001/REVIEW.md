# Hugging Face preflight compatibility repair

Failure source: Kaggle execution of 62e8f7db1. The user log records
`TypeError: HfApi.auth_check() got an unexpected keyword argument 'write'`.
T1/X3/X4 were NOT_STARTED and training GPU charge was zero. The failure report
was successfully published at c646a9a2fde324c3c2e5562654e22d82dbaf0680;
that publication was not a training result.

## Fix

- Remove unsupported write=True from auth_check. The installed 0.36 API accepts
  repo_id, repo_type and token. Access checking alone does not prove write
  permission; the real small upload plus forced fresh readback remains mandatory.
- Convert persistence/restore integration TypeErrors into a durable
  R0_BLOCKED_NO_TRAINING report instead of leaving PENDING_R0 after a traceback.
  No gate is relaxed and no failed probe can start training.
- An existing zero-progress output can refresh its control-source hashes and
  planned source commit, retaining the previous engineering attempt in history.
  This is allowed only when all models are NOT_STARTED, no checkpoint exists
  and optimizer_updates is zero. Any trained/saved state keeps its original
  source binding. Existing data/output directories should not be deleted.
- The notebook reuses installed dependencies and installs only missing worker
  dependencies. Project installation uses --no-deps rather than the broad LM
  extra, avoiding forced Transformers-driven changes to unrelated Kaggle
  Gradio/Diffusers/Pillow packages. No training parameter is changed.

## Regression coverage

Tests emulate the exact older auth_check signature and require the sequence
access check → real-write callback → forced-readback callback. Additional tests
cover zero-progress refresh, preservation of a trained source binding, and
structured blocked R0 on persistence API errors. The complete retraining test
module and static/compilation checks are recorded alongside this report.
Mocked callbacks do not establish live write access; the Kaggle probe still does.

## Rerun

Use the updated epclm001_controlled_retrain.ipynb on codex/epclm001-incremental
and Run All with the existing output/cache paths and enabled secrets. The
notebook fetches the updated branch. No cleanup or training restart is needed:
this failed run performed no baseline updates. R0 rechecks data and persistence,
then only the already-authorized T1/X3/X4 seed-91002 sequence may start.
30M, additional seeds, formal E1-C and merging main remain unauthorized.

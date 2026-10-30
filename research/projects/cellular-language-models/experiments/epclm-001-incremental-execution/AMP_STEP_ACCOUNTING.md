# Controlled retraining: FP16 step-accounting repair

Date: 2026-10-30. This records the user-authorized repair after T1 completed its
budget and the verifier incorrectly rejected one skipped GradScaler update.
It does not change the frozen native trainer, numerical configuration, seed,
data order, learning-rate schedule or token budget. v0.3 and prior evidence
remain unchanged; amp_step_accounting.json identifies the exact predecessor
controller bundle and unchanged frozen configuration.

The historical budget is **10,000 training-loop attempts / 10M consumed tokens**.
Record independently:

- training_steps (loop attempts);
- successful_optimizer_updates (actual AdamW calls);
- skipped_optimizer_updates (attempts minus successful calls).

For the published T1 observation: 10,000 = 9,999 + 1; skipped fraction 0.01%.
No compensating update, new token, rerun, seed selection or schedule change is
allowed. The known manifest records the count, not the precise skipped step or
a found_inf trace; those missing observations remain UNKNOWN.

Acceptance also requires a completed budget, positive/in-range observed count,
matching Adam step counters for every optimizer parameter, finite Adam moments,
valid scaler state, finite model parameters, correct data/config/model identity,
complete training curve, quality checks, export parity and persistent readback.
A counter disagreement or non-finite state still blocks. Every skip count and
fraction is emitted in quality/export/asset metadata; no arbitrary post-result
skip threshold or historical-PPL target is introduced. Availability is not a
scientific quality/efficiency claim and does not authorize E1-C.

## Controller migration

Only the exact d6d5a5ae6 predecessor control bundle and unchanged retrain config
are eligible. The numerical training guard and data preparation source must be
unchanged. Existing checkpoints must have verified run/config/seed/source
identity and a completed 10,000-step budget; partial training migration is
rejected. Migration records previous/current control identities, policy hash,
checkpoint SHA-256, original training source, Run ID and charged GPU budget.
It never rewrites resume.pt or resets budget. T1 retains its actual original
training source; new X3/X4 starts record their own source commits explicitly.

Run All first verifies/exports/persists completed T1. No T1 worker is launched.
Only after T1 readback succeeds may the already-authorized X3 then X4 start,
within the remaining original budget. A fresh Kaggle session can restore the
existing controller backup; no manual manifest reset or output deletion.

## Retry and inference-only fixes

IDENTITY.json now records quality_sha256, and a publication retry verifies that
digest before reusing QUALITY.json. A failed upload can retry publication without
re-evaluation or training. The reloaded inference model is explicitly frozen,
avoiding autograd's attempt to save inference-mode token tensors for backward.
Original resume bytes remain untouched throughout export and retry.

# FP16 acceptance, export and controller-resume repair

Original observed run: fbf1959b2a364022959ded11c1e4fea6, published at
7af138908455daa49234b504b752835791fc0862. This is a repair report, not a new
training result. The original published report and checkpoint are not rewritten.

## Observed facts

The published manifest records T1 complete, worker exit code 0, 10M tokens,
10,000 loop steps, 9,999 actual AdamW calls, and 224.544535773 seconds charged.
Data identity and persistence preflight passed. Final logged PPL was 12.336.
X3/X4 were NOT_STARTED. The verifier subsequently rejected actual_updates!=10000.
One skip is consistent with historical GradScaler behavior; neither the manifest
nor original observer records its step index or direct found_inf evidence.

## Repairs

1. Validate historical loop budget instead of demanding 10,000 successful
   updates. Expose training_steps, successful_optimizer_updates,
   skipped_optimizer_updates, fraction and zero compensation. Check Adam's own
   per-parameter counters, finite moments/scaler/model and existing quality/data
   gates. Missing/invalid counters, incomplete budget or numerical failures
   remain blocking. No learning-rate or training change is made.
2. Generate quality_sha256 in the real export path. Regression tests cover the
   actual export, interrupted publication and retry without re-verification or
   any training call. Retries still reject changed verified assets.
3. Freeze the reloaded inference model: inference-mode token tensors are not
   passed into a gradient-enabled forward. This additional validation-path bug
   was found by the real-export fixture before deployment.
4. Add a narrowly audited controller migration. The published manifest's exact
   predecessor hash bundle was compared with the migration policy and matches.
   Frozen config, data code and numerical guard are unchanged. Preserve Run ID,
   checkpoint hash/bytes, T1's actual training source and spent budget; retain
   previous control identity and policy hash in history. Partial or unrecognized
   source migrations fail closed. Newly started X3/X4 get explicit source IDs.
5. Completed T1 goes directly through verification/export/persistence. A control
   flow regression proves the first possible worker afterward is X3, never T1.
   Publication/readback remains a hard gate before continuing to X3/X4.

## Validation and limits

Full correctness suite: 63 passing tests; JUnit retained. Tests include a
production-export path using a tiny synthetic fixture, invalid/mismatched Adam
counters, incomplete budgets, non-finite optimizer state, preservation of
checkpoint bytes/Run ID/source/budget during migration, partial-migration refusal,
T1-worker avoidance and publication retry. Existing CUDA fixture tests remain
engineering correctness only. Static/compilation/diff checks passed.

The actual T1 optimizer tensors were not downloaded here: the read-only HF
retrieval attempt failed with a local network URLError. Therefore this report
confirms the published count and source-bundle compatibility, not the actual
checkpoint's tensor-state verification or completed R2 readback. Those checks
execute on the restored T1 bytes in Kaggle before X3/X4 may start. No T1 training
or compensation was performed during this repair. Synthetic test artifacts
are not real model results.

## Next execution

Run the updated controlled-retraining notebook, retaining current output/cache
or restoring the existing HF controller backup. The original run ID and charged
budget persist. The sequence is T1 verification/export/readback → X3 → X4. If
any identity/state/quality/persistence check fails, preserve it and stop; never
reset the manifest or retrain completed T1. No new seed, 30M, formal E1-C or main
merge is authorized by this repair.

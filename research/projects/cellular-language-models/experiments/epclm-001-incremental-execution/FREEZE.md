# Scope and freeze status — v0.2

Round-1 scope: T1/X3/X4, approximately 1.9M parameters. Fixed by the user's
2026-10-30 amendment. No new training, formal experiment or 30M execution.

30M: DEFERRED / CONDITIONAL SCALE VALIDATION. Separate user approval required
after complete 1.9M results and a bounded scaling proposal. The original v0.1
source draft remains unchanged and does not override this amendment.

Scientific performance protocol: NOT FROZEN. Exact native checkpoints,
actual tokenizer/data identity, device/precision-specific tolerances,
holdout trace and small hardware workload matrix still need binding.

Existing FP32 tiny-fixture tolerances (atol=2e-5, rtol=2e-4) are diagnostic.
They do not certify the approximately 1.9M trained models or another precision.
H1/H2/H3 remain INCOMPLETE until their independent requirements are met.

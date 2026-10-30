# EPCLM-001 v0.3 implementation and actual R0 result

Date: 2026-10-30. Controlled retraining is authorized by the user amendment.
**Actual state: R0_BLOCKED_NO_TRAINING; 0/3 models trained.** This is an
implementation/preflight delivery, not completed model-asset generation.

## Implemented

- Independent retrain-preflight/run/verify/publish commands and one-launch
  Kaggle notebook. Fixed order T1/X3/X4, seed 91002, historical baseline profile,
  10M tokens / 10,000 original loop steps per model. M4, 30M, other seeds and
  formal E1-C are excluded. General training permission remains false; v0.3 is
  the explicit local exception. No merge to main.
- Unchanged historical single-model replication worker. Frozen original source
  hashes and explicit AdamW/GradScaler defaults; original initialization
  difference retained (T1 model seed 91003; X3/X4 model seed 91002).
- Recovered tokenizer is copied and hash-checked; original prepare_corpus is
  never called. The original stream writer rebuilds fixed-revision data and
  enforces exact historical raw hashes before training. Actual mismatches are
  retained and stop R0; no automatic tokenizer retraining or new data identity.
- Complete resume state and a separate inference export, distinct controlled
  asset IDs, source/data/config bindings, model-state/output checks, original
  validation evaluation, historical-PPL diagnostics and real-weight adapter
  integration. Historical PPL is never a rerun/selection target.
- Existing HF destination archelabsxyz/native-clm-v0 reused under an isolated
  controlled-retrain namespace. R0 requires a write/readback probe. Available
  assets require forced fresh download and a new-process strict load. Git holds
  reports/indexes only. No repository is silently created.
- Remote controller index retains run identity and resume copies. A fresh cloud
  session can recover matching bytes; a report-only empty manifest does not
  hide it. Durable resource reservation before worker start fails closed after
  an uncertain lost session instead of resetting the allowance.
- One controller/root and one GPU-worker lock. Safe-boundary periodic resume
  saves preserve sampler/scaler state; original five quality checkpoints remain.
  A 7,200-second conservative worker budget stops at an update/save boundary;
  a stuck worker has a 120-second termination grace. Observed time is recorded,
  including any overrun. No automatic budget or seed expansion.
- Publication retries can reuse verified output without retraining. Resume bytes
  are backed up before quality/export failures could prevent later recovery.

## Validation evidence

The full suite passed **47 tests**, including the existing tiny CUDA cache
correctness fixtures. After the final narrow orchestration changes, all **11
retraining tests** passed again. JUnit files are retained. Static checks and
compilation passed. Synthetic CPU observation tests compare guarded versus
original-loop parameter updates exactly; no research baseline was trained.

The supplied amendment is preserved byte-for-byte, including CRLF and Markdown
hard breaks. A file-specific whitespace attribute exempts those intentional
source-document line endings; implementation files retain normal checks. Old
protocols, original model/trainer source and historical reports are unchanged.

Live data preparation, full controlled GPU training and remote artifact
round-trip are **not validated by these fixture tests**. They remain subject
to R0 and actual execution. This report does not claim end-to-end cloud success.

## Actual preflight

TRAINING_MANIFEST.json records the executed committed-source R0 check:

| Gate | Actual result |
| --- | --- |
| Frozen native source | PASS |
| Model/configuration constraints | PASS |
| CUDA availability | PASS; no baseline worker started |
| Original tokenizer SHA-256 | Matches becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2 |
| Data reconstruction | BLOCKED: datasets/tokenizers unavailable |
| Dependency installation attempt | Failed with TLS EOF/read timeout reaching PyPI |
| Persistent storage write/readback | BLOCKED: no HF_TOKEN/cached write credential |
| T1 / X3 / X4 | NOT_STARTED / NOT_STARTED / NOT_STARTED |
| Baseline optimizer updates | 0 |
| Training GPU budget charged | 0 seconds |
| Asset readback / quality / new PPL | NOT RUN |

The training/validation stream hashes remain **unverified**, not mismatched and
not fabricated. No checkpoint hash, persistent asset URL, new PPL or training
curve exists to report. Pending asset fields remain null. No new baseline
weights are substituted for historical E0 weights.

## Next direct execution

Import epclm001_controlled_retrain.ipynb from this branch into Kaggle. Enable one
GPU, Internet and the existing HF_TOKEN/GITHUB_TOKEN secrets. HF_TOKEN must have
write access to the existing configured model repository. Run All performs the
Git return preflight, dependency preparation, source/data gates, durable storage
probe and only then the already-authorized bounded training sequence. Do not
add a second seed or change configuration when a gate fails.

A successful run must still supply actual asset hashes/URIs and independent
readback before ASSET_AVAILABLE. Scaler-skipped optimizer calls are counted;
any conflict with the required update count is retained for review rather than
compensated with extra batches. Quality differences from historical seed 91002
are diagnostic and cannot trigger seed/configuration selection.

R4 would only establish readiness to request E1 performance evaluation. E1-C
remains unauthorized, H1/H2/H3 remain INCOMPLETE, and 30M remains deferred.

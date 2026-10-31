# Implementation record

2026-10-31: Core-001 implemented as a separate development research unit. The supplied draft is preserved in SPEC.md. The initial development configuration uses 16×128 persistent state, synchronous radius-one ring messages, shared Native RMSNorm/SwiGLU residual updates, single event injection and K=4. All four controls share the next-token task and fixed token target. F0 is a low-rank flat recurrence; its parameter/compute discrepancy must be reported.

One small CPU integration test covers differentiability, finite arithmetic, locality, read purity, lane isolation, reset, corrupted snapshot rejection, exact segmented/snapshot execution, bounded L0 history, all four arms, and exact interrupted optimizer/RNG/data/state recovery. No real TinyStories training or GPU run was performed during implementation. Installed unrelated pytest plugins conflict on this host; use PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for this self-contained test.

GPU Gate 1, full-budget Gate 2, semantic causal findings, independent multi-seed confirmation and exploratory internal-compute benefits remain NOT EVALUATED. The launcher is development-only and performs no formal scientific threshold evaluation. Formal seeds, statistical power, and acceptance thresholds require a subsequent frozen confirmation protocol. GPU kernel counts are measured by the isolated profiler; analytic FLOPs are labeled estimates.

## Minimal-pilot amendment (base 5332b7a)

The default Notebook now uses only the shared Trainer's 50-update C1 pilot, with an actual save/restart boundary at update 25. Architecture and trainer configuration come from protocol.json. CUDA numerical checks use asynchronous device assertions, and fixed-shape loss masking removes a second per-token synchronization. Single-file checksummed checkpoints and immutable recovery objects close the checkpoint/sidecar and recovery-manifest update gaps. The same CPU integration test adds only critical configuration, checksum, and interrupted-save regression assertions; no local full training, CPU performance benchmark, or GPU experiment was run.

## Dataset revision fix — 2026-11-01

The first Kaggle minimal-pilot attempt stopped during tokenizer/data preparation, before real GPU training. Core had imported the historical default from native_clm_runtime directly, bypassing Native entrypoints’ existing correction of the full TinyStories commit SHA. Core now reuses native_clm_candidates.VERIFIED_DATASET_REVISION, records it in protocol.json and applies it before both Native tokenizer training and document streaming. Historical Native files and experiment results are unchanged.

Verification: the public Hugging Face revision API resolves exactly to f54c09fd23315a6f9c86f9dc80f725de7d8f9c64, and an anonymous range request for TinyStories-train.txt returned HTTP 206 / 512 bytes. The existing single CPU integration test passes with a mocked-loader regression assertion; no local full training or GPU pilot was executed.

## FP16 gradient-overflow amendment — 2026-11-01

The original T4 failure report for source 26186fed2048 is preserved byte-for-byte under reports/failures and remains INCOMPLETE. A new source identity starts fresh model/optimizer state while allowing only independently verified tokenizer/data assets to be reused. The corrected Trainer starts GradScaler at 1024, skips non-finite-gradient updates without clipping, performs normal backoff, and replays the same state/document/RNG transaction. A matched FP32 backward probe performs no optimizer update. Three consecutive overflows or a non-finite FP32 probe fail explicitly and preserve diagnostics and the last committed training position. Scheduler and logical token/update counters advance only on accepted finite updates.

The first two successful diagnostic updates count toward the original 50, followed by the update-25 checkpoint/restore boundary. No real GPU experiment was run during implementation; FP16/FP32 behavior and throughput require the next Kaggle execution. Focused CPU fault injection covers skip/backoff/rollback, FP32 classification, tolerance exhaustion and assets-only cross-source recovery; the existing small resume integration test is retained.

Validation completed: the existing CPU resume integration check, focused injected-overflow transaction test, and remote/local assets-only recovery checks pass. No repository-wide test matrix, GPU benchmark, or real GPU training was executed.

## Full FP32 pilot — 2026-11-01

The repeated FP16 failures on the same first update are preserved under reports/failures/5bd2f2c5e67a-pilot-report.json. The matched FP32 probe had finite gradients but performed no optimizer update; it did not establish training stability. The next pilot therefore explicitly selects FP32, disables Autocast/GradScaler/TF32, and validates model, loss, recurrent state and optimizer dtypes. C1, batch 4, TBPTT 64, K=4, the 50-update budget and the update-25 restore boundary are unchanged. The first two successful optimizer steps count toward those 50; no additional diagnostic training is introduced. No GPU training was executed during this implementation change.

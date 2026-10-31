# Implementation record

2026-10-31: Core-001 implemented as a separate development research unit. The supplied draft is preserved in SPEC.md. The initial development configuration uses 16×128 persistent state, synchronous radius-one ring messages, shared Native RMSNorm/SwiGLU residual updates, single event injection and K=4. All four controls share the next-token task and fixed token target. F0 is a low-rank flat recurrence; its parameter/compute discrepancy must be reported.

One small CPU integration test covers differentiability, finite arithmetic, locality, read purity, lane isolation, reset, corrupted snapshot rejection, exact segmented/snapshot execution, bounded L0 history, all four arms, and exact interrupted optimizer/RNG/data/state recovery. No real TinyStories training or GPU run was performed during implementation. Installed unrelated pytest plugins conflict on this host; use PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for this self-contained test.

GPU Gate 1, full-budget Gate 2, semantic causal findings, independent multi-seed confirmation and exploratory internal-compute benefits remain NOT EVALUATED. The launcher is development-only and performs no formal scientific threshold evaluation. Formal seeds, statistical power, and acceptance thresholds require a subsequent frozen confirmation protocol. GPU kernel counts are measured by the isolated profiler; analytic FLOPs are labeled estimates.

## Minimal-pilot amendment (base 5332b7a)

The default Notebook now uses only the shared Trainer's 50-update C1 pilot, with an actual save/restart boundary at update 25. Architecture and trainer configuration come from protocol.json. CUDA numerical checks use asynchronous device assertions, and fixed-shape loss masking removes a second per-token synchronization. Single-file checksummed checkpoints and immutable recovery objects close the checkpoint/sidecar and recovery-manifest update gaps. The same CPU integration test adds only critical configuration, checksum, and interrupted-save regression assertions; no local full training, CPU performance benchmark, or GPU experiment was run.

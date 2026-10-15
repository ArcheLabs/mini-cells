# Architecture and historical boundary audit

Engineering base: `5c4be44` on the current research line; remote main did not yet include that line. A new branch preserves the user's current work rather than rebuilding from older main. No historical experiment source, frozen result, checkpoint or research conclusion is modified.

## Reused components
| Existing component | Reuse | Boundary |
|---|---|---|
| PDCLM + exp006 Cells, ContextRead, InputMemory | Continuous inventory, alpha residuals, static causal memory | Replace PD J0 executor/router with explicit Q cost/STOP handling; no fixed Transformer backbone |
| PD `joint_loss` | J0-FULL live objective | No direct candidate Q fit to official J0 router |
| exp006 `initialize` | Same named-parameter seed initialization for both arms | Neither arm resumes an old optimizer |
| FAST Deduplicator, grouped_windows, BPE/exp006 encode_stories | New four-role story-contained assets | New CLM identity/manifest; no old FAST manifest impersonation |
| PD checkpoint / atomic 005 checkpoint | Model, optimizer, Python/NumPy/Torch/CUDA and independent generators | Add accepted registry, versioned private branches and atomic latest pointer |
| PD paired_interval | Offline paired story bootstrap | Admission uses explicitly documented equal-story weighting |
| Shadow `AcceptedModelSnapshot`, hash_accepted_state | Byte-level accepted-state witness | Historical TinyCLM sidecar architecture and positive claims are not transplanted |
| historical publish_core_validation_007 | Existing GITHUB_TOKEN / askpass / preflight / push | Fresh main-based metadata checkout, no source/workflow/weights push |

New code is necessary for full-suffix Q labels, a numerical cost-valued Connection, bounded private lifecycle and this experiment identity. Imported source hashes are recorded in `reports/read-only-source-inputs.json`; CPU tests verify those old files remain byte-identical. CI is new and scoped to CLM-001.

## Experiment 019 failure
Authoritative `research/projects/cellular-computing/experiments/019-proposal-utility-discovery-stable/artifacts/decision.json` reports **UTILITY_ORACLE_INCONSISTENT**. Oracle Pearson was `0.24125416546309048` against registered `0.95`; sign agreement `0.8773561507936508` against `0.9`; Spearman `0.7700432528118718`. These values establish failure of that gate, not its cause. No old Oracle code or targets are reused here.

The capability-ceiling No-Go conclusions remain intact. Successful copy-on-write, causality, transaction or synthetic Oracle tests are engineering contracts, not evidence for autonomous language specialization or lifelong learning.

FAST runtimes/numbers quoted in the supplied protocol are historical/user-supplied context; the original remote FAST report was not independently fetched for this implementation. They are not new benchmark measurements.

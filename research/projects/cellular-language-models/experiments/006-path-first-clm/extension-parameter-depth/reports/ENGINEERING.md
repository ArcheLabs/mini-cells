# PD-001 engineering report

Status: ENGINEERING_READY for CPU reference implementation; PROPOSED_NOT_REGISTERED for research. NO_SCIENTIFIC_RESULT_YET. GPU runs = 0; confirmation assets read = 0.

## Read-only parent audit

Implementation base is main commit `27b913cb6a9fd55904a27b97c980971600eb6e95`. Parent 006 has static-input memory, real fixed-six-step hard paths, R1/R2 state policies, detached-input critic, and an independent six-layer Top-2 MoE. Its protocol remains unregistered and rejects STOP. PD adds a new package and identity; original protocol, config, audit and artifacts remain unchanged.

Parent regression: 56 tests pass. Initial test collection exposed a historical migrated publisher import path; validation supplied the two existing canonical helper paths on PYTHONPATH without editing historical code. Existing working-tree changes to CLM Native publication recovery and the user's 005 report were kept outside this isolated branch.

Reuse: MoELM, FeedForward, ContextRead, InputMemory, deterministic component initialization, story/window checks, atomic checkpoint/RNG state, atomic JSON, parameter inventories, GEMM hooks and historical authenticated publication. New code is limited to STOP-aware execution, PD config/gates, independent whole-path J1, variable-length diagnostics, analysis and adapters.

PD automated tests: 49 pass. Ruff 0.9.10 passes. All ten actual-size models complete one CPU synthetic joint update, with additional tiny 64-step/math/replay fixtures. These are not scientific language-training results.

## Actual capacity

M0: 16,483,968. P100-F16: 16,557,539 (100.446%). P50 family: 8,200,923 (49.751%); STATE removes 336 edge-prior parameters: 8,200,587. P25 family: 4,124,631 (25.022%). Every goal passes ±5% relative tolerance. No dummy padding. Tied embedding/head counted once. Critic and STOP policy parameters are resident and counted. The exact inventories are in parameters.json.

P50 uses width384, 8 Cells, hidden688. P25 uses width256, 4 Cells, hidden800. P100 uses width384,16 Cells,hidden1024. Fixed/adaptive/J1/SELF share the same corresponding architecture; capacity is allocated to real operators. A fixed arm masks STOP and uses only the allowed step-prefix; unused conditional step rows are intentionally retained for the same-architecture depth comparison, not introduced to reach a target.

## Unresolved engineering and scientific risks

- Full non-GEMM costs and fused attention backward GEMM are not completely measured. Observed module-shape forward and dispatcher-supported forward/backward GEMM are explicit lower-bound scopes. Do not claim equal total training FLOPs or complete system superiority.
- Float32 is the audited train configuration. CPU bfloat16 replay is tested, but FP16/BF16 training, loss scaling and CUDA microbenchmarks require separate implementation validation/authorization. No CUDA runs occurred.
- Reference dispatch loops group active positions per action. Long paths may suffer scheduling overhead or memory pressure. No fused executor, hardware-cache flush or large-batch optimization is claimed.
- Static input-only context differs from MoE's dynamic layered context. P25 changes representation width. A failure is not automatically a Connection failure.
- Every step's retained trajectory diagnostics can be large; raw weights remain external. No real dataset, training trajectory, asset conflict audit or statistical-power estimate was obtained in this engineering task.
- Baseline stability, final hypothesis thresholds, hierarchical seed consistency, complete cost-matched J0/J1 checkpoint comparisons and GPU schedule remain unregistered. The report emits diagnostics, not scientific support.

The unit tests and full-size CPU fixtures are engineering evidence only. Formal language training, result publication and confirmation remain blocked.

Frozen engineering implementation: `c7ff3a4aee6e5e2b3353b7d2b6a059e93ad0b834`. The delivered notebook pins this commit; registration source remains unset.

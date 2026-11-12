# EPCLM-008 Protocol Freeze report and execution plan

**Decision: BLOCKED.** CPU engineering preflight passed. Formal training and Test access were not started.

## Evidence and remaining decisions

- EPCLM-004 M1 pins WikiText-103 raw and the 8192-token ByteLevel BPE. The dataset card's metadata and license narrative give different versions. Preserve the publisher's attribution and resolve the applicable terms for the intended distribution before declaring the license check complete. The M1 prefix adapter enforces the exact tokenizer-training prefix hash and permits only train and validation. Formal materialization needs a separate full-corpus adapter or an explicitly approved subset with fixed sampling and leakage checks.
- EPCLM-004 M2 pins E0/E3 architecture and engineering optimizer settings. Its two-update smoke provides no formal budget. The EPCLM-004 specification lists 20M screening and 100M confirmation budgets for that experiment. Neither establishes the EPCLM-008 backbone budget. The exact supervised-token budget, sequence packing, effective batch, gradient accumulation and scheduler remain open.
- EPCLM-006 canonical encoding calls the backbone on each bounded segment without a runtime-state argument. Its preflight shows that a nonempty runtime changes session-conditioned representations while the reset-only canonical path is reproducible. The 008 CPU fixture records fresh-process raw Cell output error 0. EPCLM-007 Gate P independently preserves the direct fixture's 5/8 result and complete score matrix. These are fixture results. The trained E3 test must compare raw reads for identical packets and Cell under fresh and distinct carried states. MAG-fused logits may depend on session state and are evaluated separately.
- EPCLM-007 uses a candidate-scoring retrieval head and a randomly initialized frozen E3. EPCLM-008 requires the trained E3 and MAG fusion. Verify gradient reachability and C0/C1/C2 semantics in the integrated path before running the 1024-update budget.
- Exact History scoring and arm inputs are specified in the companion protocol. Generator parameters and split identities remain open.

## Execution order after protocol decisions

1. Freeze corpus preparation, exact-history generator and training parameters. Materialize train and validation only. Verify shard, normalized text, tokenizer, token-stream, episode and ordered schedule hashes. Bind them to the source commit and reject identity changes on resume.
2. On Kaggle T4 measure synchronized train-step time and peak allocated/reserved VRAM for full-size E0, E3 and MAG training using the frozen batch and sequence settings. Measure checkpoint, evaluation and verified publication time. Project six backbone runs plus three 1024-update memory runs from measured throughput. Partition across saved sessions if needed. Report resource failure without reducing the budget.
3. Compare an uninterrupted fixture against a process-restarted run at the same committed update. Check model, optimizer, scheduler, RNG, next episode and sampler cursor, and carried Cell fast weights, momentum and update step where applicable. EPCLM-004 M2 saves its ledger and RNG and derives data position from committed updates, but its constant engineering schedule does not establish general scheduler restore. EPCLM-007 resumes at outer-update boundaries and recreates episode-local Cells. Use only fully committed boundaries or serialize the partial episode state explicitly.
4. Exercise the repository publisher and durable large-file storage with fixture artifacts. Verify branch SHA, then download the checkpoint and Cell files and verify the manifest's sizes and SHA256. Record download links and compatibility identity. Preserve Kaggle Output on failure. Publication repair must reuse the saved artifacts.
5. Train E0 and E3 per seed on identical ordered batches and token budgets. Train MAG against the matching frozen E3 checkpoint and verify its SHA before and after memory training. Run A–E on validation and preserve each seed and each failed condition independently.
6. Freeze validation results, final model hashes, evaluation configuration, artifact manifest and gate decisions. Require verified publication, successful resume/integrity checks and a documented Test release before creating or accessing Test. Evaluate once with identical conditions across arms.

## Measurement contract

Record synchronized training elapsed time, durable updates, nonpadding supervised tokens, allocated and reserved CUDA peaks, and checkpoint/evaluation time separately. Capture executed operator shapes and counts in an isolated profiling pass. Report FLOP estimates by covered operation and explicitly list unsupported operators, including reference selective scan if uncovered. Record the device and backend with each measure. A profiler subtotal with incomplete coverage does not establish actual total FLOPs.

T4 feasibility, cross-session resume and remote readback have no EPCLM-008 execution evidence yet. A resolved lock and successful preflight are required before the formal launcher can be used.

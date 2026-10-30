# EPCLM-001 v0.4 engineering delivery

Implementation source: `550861d3c6d45a616c483aff360b1612e7798183`. Branch: `codex/epclm001-incremental`.

Engineering tests: **109 passed** (63 historical regressions, 46 formal-workflow
fixtures), followed by 6 focused passing checks after report/publication guards.
The initial expanded test run caught a missing import in a test; it was corrected
before the final full-suite run. Ruff, compile checks and diff whitespace checks
passed. Notebook JSON structure/compilation and a mocked Run All bootstrap passed.
The mock performs no network, scientific model execution or training.

## What is delivered

- Exact preserved v0.4 protocol, hash-checked FP32 configuration and isolated
  inference-only commands, including independently callable stages.
- Exact HF asset revisions, strict weight/tokenizer/corpus/source/state/quality
  bindings, validation-only input reconstruction and frozen slices. No train
  corpus reconstruction or training is invoked.
- All-prefix A0/A1 parity and cache contracts, B0/B1 four-executor correctness,
  correctness-free latency paths, a separate profiler and partial-FLOPs labels.
- Fixed 5 model workloads and 16 event points per kernel, 5 warmups/30 repeats,
  paired uncertainty and full raw retention; no outcome-selected point removal.
- Automatic tables/statistics, nine figure generators and Chinese/English
  summaries. Figure generation was tested on explicitly synthetic test records;
  no synthetic graph is delivered as a formal result.
- Shared GPU-worker lock, new independent 7,200-second cap, durable reservation,
  hard watchdog, stage/sample hash guards and conservative crash recovery.
- Existing GitHub report-return component, remote verification, report-only Git
  payload, external full raw/profiler storage and separate publication retry.

## Actual execution status

The real local preflight stopped **before any GPU worker** because this session
has no existing Hugging Face persistence credential. Its recorded error is
`FORMAL_CONTROLLER_RESTORE_FORMAL_PERSISTENCE_REQUIRES_EXISTING_HF_TOKEN`.
Observed/charged formal GPU-worker time and optimizer updates are **zero**.

The three user-supplied trained asset pins have **not** been downloaded or
verified here. Training completion is not reasserted from configuration alone.
H1, H2 and H3 remain **INCOMPLETE**. No formal latency, speedup, quality regression,
real-model CUDA parity or actual GPU profiler result is claimed.

Engineering decision: **READY FOR KAGGLE PREFLIGHT**, not scientific acceptance.
Kaggle must verify the three frozen weight bytes and all correctness conditions
before sampling. Enable one GPU, Internet and existing HF_TOKEN/GITHUB_TOKEN,
then Run All in `epclm001_formal_e1.ipynb`. No manual asset Input is required.
The notebook pins the implementation commit above, independent of later report
commits on the branch. No main merge is performed.

## Cost and limitations

No reliable wall-time estimate is possible without the registered on-device
pilot. The hard ceiling is 7,200 GPU-worker seconds; setup, network transfer and
external readback add uncharged CPU/network wall time. Full-sequence CLM
reference passes and 12 sequential iteration caches are principal expected
costs, not measured bottleneck claims.

First round is FP32 only. TTFT/end-to-end, DRAM counters, energy and precisely
attributed allocator overhead remain explicitly unmeasured. FLOPs cover matrix
products only. Quality is reused only under original evaluator/data/weight hash
binding, labelled FP16 development evaluation and not equal-quality evidence.
CUPTI/kernel profiling is still unverified on Kaggle; missing CUDA profiler events
are labelled unavailable. Timing-repetition intervals are approximate and are
not model-seed quality intervals. A lost process retains its outstanding
reservation; a new session cannot silently reset the budget. Partial local
samples since the last remote boundary can be unavailable after abrupt loss.
30M requires separate future review and authorization, regardless of results.

See `FORMAL_EXECUTION.md` for exact scope, ordering, primary comparisons,
recovery behavior and independent review boundaries.

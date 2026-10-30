# EPCLM-001 — Round 1: native ~1.9M models only

Version: 0.2, 2026-10-30. Scope amendment approved by the user's current
instruction. Scientific performance protocol remains NOT FROZEN; this scope
amendment does not authorize execution. This document and `protocol.json`
supersede the **execution scope** of `PROTOCOL_DRAFT.md` v0.1. That original
user-supplied draft is preserved unchanged as historical design context.

## Scope and evidence

Round 1 uses E0-B T1, X3 and X4 only. E0-A 30M remains historical evidence,
labelled **DEFERRED / CONDITIONAL SCALE VALIDATION**. It is excluded from asset
loading, correctness runs, measurement defaults and notebook checkout paths.
No checkpoint is retrained. No historical implementation, weight or conclusion
is modified. C0/M4 are not round-1 baselines.

Three claims remain separate:

- H1: exact incremental reuse and measured same-model inference benefit.
- H2: physical sparse execution benefit under externally supplied events.
- H3: the measured gap between cached modern T1 and optimized native CLM.

Engineering fixtures do not establish these claims. Incremental KV caching is
not persistent EPCLM Cell memory. Synthetic events are not learned routing.

## Baseline contract

T1 uses the existing `ModernTransformerLM`: width 192, four blocks, six query
heads, two KV heads, SwiGLU width 480, RMSNorm and RoPE. X3/X4 use existing
`PhaseBLM`: width 188, four attention heads, three stages with four iterations
each and windows 8/32/128. X3 uses LayerNorm/SwiGLU width 638; X4 uses
RMSNorm/SwiGLU width 639. Residual scaling and all model operations stay intact.
Vocabulary is defined as 2048 and the supported context as 128. Actual
checkpoint precision, tokenizer and corpus identities must be verified rather
than inferred from these matching definitions.

Asset lock requires exact model identity/configuration, original seed, training
budget/profile, source, checkpoint hash, tokenizer hash and corpus/evaluation
identity. Reuse existing published development seed(s); never select by the
new benchmark outcome. The source declares development 91001 and replication
91002/91003. These declarations do not prove that their weights exist.
Confirmation seeds 91101/91102/91103 remain outside development access.

If multiple eligible checkpoints exist, resolve the evidence/seed binding
before selection. Missing assets are ASSET_UNAVAILABLE in the searched
locations; this is not a claim of global nonexistence. Missing exact weights
block real H1/H3 evaluation. Random initialization may test adapter engineering
only, and cannot fill this gap. H2 synthetic execution remains independently
available.

## Step 1 — Preflight, no model training

1. Inventory T1/X3/X4 checkpoints and original metadata; strict CPU loading only
   for eligible native identities. Compute hashes internally.
2. Verify actual tokenizer, data revision and shared evaluation conditions.
   The bare runtime revision differs from the corrected revision bound by the
   launchers; bind the actual saved corpus manifest, not whichever module was
   imported most recently. Do not edit historical sources to conceal this.
3. Reuse the original full-forward code, model builders, tokenizer loader,
   evaluation function, fixed validation starts, environment recorder and
   analytical FLOP estimates where applicable. Mark analytic matmul estimates
   as partial. Historical training throughput is not decode TPOT.
4. Record exact software, device, precision, supported context and evidence
   level. Stop the affected real-model path if identity is unresolved.

## Step 2 — Correctness before timing

For X3/X4, retain separate KV at each (stage, iteration), applying the original
step conditioning and RoPE absolute positions. For T1, cache its actual GQA
keys/values; do not use uncached T1 as the competitive baseline.

Cover prefix causality, teacher-forced logits, lengths 1 and window boundaries
around 8/32/128, unpadded batch 1 first, reset, chunk partition, cache identity
and restore. Padding remains explicitly unsupported until independently tested;
never silently ignore masks. Prefill predicts the next token after its prefix;
decode consumes only an already known token. No unverified context extension.

Freeze per-device/per-precision max/mean/relative error criteria before formal
measurement. Existing tiny FP32 CPU tolerances are development diagnostics,
not evidence for FP16/BF16/CUDA. All failed cases remain in the record. A
failed model's parity blocks its performance comparison.

## Step 3 — Small event microbenchmark

D0 dense reference, D1 masked dense, S0 sparse cell and S1 sparse region must
share deterministic updates and the same event trace. Read a frozen microstep
snapshot, commit only active cells, and make outgoing events visible no earlier
than the next microstep. Empty events must not invoke all-cell computation.
Regions count every physically computed slot, including inactive overcompute.

Use a small layered workload, not a Cartesian grid: native-relevant state
width 188 first; one small and one moderate cell population; dispersed and
clustered activity; low, intermediate, all-active and zero-active controls;
one initial region size followed by at most one planned comparison. The
existing width-16 CPU fixture remains an engineering test only. Native-like
FFN width 638/639 may be added as a separately identified shared kernel before
freezing; the current tanh linear synthetic kernel is not the CLM update.
Freeze exact points, trace seed/holdout and repeat count before formal timing.

Report logical/physical updates, partial FLOPs, wall time, host/device kernel
time, queue/packing/gather/scatter/message cost and available memory traffic.
Unavailable counters are null with explanations. An activity percentage is
never a speedup. Record both favourable and unfavourable distributions.

## Step 4 — Formal benchmark, gated and not authorized now

On one isolated GPU worker, compare T1 cached, X3/X4 full-prefix, X3/X4 A0
incremental, then parity-passing A1 optimization. Batch 1 first; keep precision,
hardware, supported context and fixed input sequences consistent. Use a small
frozen workload, warmups and interleaved repeated measurements.

Separate model loading, prefill, TTFT, decode TPOT p50/p95/p99, host execution,
device execution and end-to-end tokenize/sample/return latency. Unimplemented
end-to-end components cannot be relabelled as measured TTFT. Report weights,
cache capacity, peak device memory and partial-versus-measured FLOPs/token.
Use the same bound quality corpus for cross-model quality regression. Different
PPL values forbid an equal-quality speed claim. Preserve every failure/OOM.

## Go / No-Go

| Gate | GO condition | Failure consequence |
| --- | --- | --- |
| Assets | Exact eligible native weights, configuration, tokenizer/data and evidence binding | H1/H3 NO-GO; H2 remains separate |
| Correctness | Frozen device/precision criteria and all relevant cache tests pass | No speed claims for that model |
| Event semantics | State/events/consumption agree; inactive writes absent; negative controls pass | H2 NO-GO |
| Measurement | Frozen small workload, held-out trace, isolated repeated timings and honest coverage | Performance decisions INCOMPLETE |
| H3 comparability | T1 and CLM cache parity, shared quality evaluation and same measurement conditions | Competitive claim INCOMPLETE |

Record H1/H2/H3 independently as supported, not supported or incomplete under
the original decision semantics. Do not require an invented 2x/5x threshold.
No-GO is scoped to the failed dependency and never triggers replacement
training, tolerance relaxation or automatic scaling.

## Resource strategy

No GPU is needed for this amendment or source/asset inspection. Later use one
available GPU, one isolated latency worker, batch 1 and sequential model loads.
The ~1.9M FP32 weights occupy roughly 7.6 MB of raw tensor storage (FP16 about
3.8 MB); allocator/workspace/activations/profiler overhead are additional.
This is not a peak-memory estimate. No reliable GPU-hour or latency estimate
is available without recovered weights and on-device measurements. Main costs
are the 12 CLM iterations, full-prefix references, routing/gather/scatter and
kernel-launch/synchronization overhead, repeats and optional profiling.
Measure a bounded pilot point before estimating the remaining matrix. No
multi-GPU requirement or full grid is justified.

## 30M extension review — never automatic

An extension proposal requires all of: native CLM parity; measured reduction
of duplicate inference work with useful resource benefit; actual event-runtime
benefit in specified meaningful activity distributions; a reported modern
T1-vs-CLM efficiency gap; and a concrete scaling question with incremental
cost. Submit the complete 1.9M results and negative evidence first. Even when
all conditions hold, only a new explicit user decision can authorize 30M.

Current next action: locate and read-only bind the existing native trained
checkpoints and their tokenizer/corpus manifests. Synthetic executor fixtures
may be prepared separately, but no formal experiment or training starts under
this amendment task.

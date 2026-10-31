# EPCLM-001 bounded engineering closure

Review target: 1d527228. Date: 2026-10-30.
Outcome: engineering defects repaired; **formal experiment remains NO-GO**.
No training, 30M execution, trained-model quality evaluation or formal GPU
performance experiment was started. CUDA use is restricted to tiny correctness
fixtures, not a hardware performance claim. Scope/protocol are not redesigned.

## Findings and fixes

1. **Checkpoint recovery UNRESOLVED.** The original notebook explicitly kept
   weights and optimizer state outside Git. Its runtime paths are now searched:
   `/shared/kaggle/working/native-clm`, `/kaggle/working/native-clm`, the local
   `.native-clm-work` fallback, native experiment tree and configured cloud
   roots. None supplies accessible trained native bytes here. The user also
   found no verified Kaggle saved-Output locator. PR #217 / commit
   `3321a74c26965ce0c421fd14edef0f6b961cccfe` establishes historical development
   replication evidence, not present-day checkpoint availability.

   Discovery now inspects bounded Torch zip metadata without unpickling model
   tensors, including opaque .pt/.pth/.ckpt/.bin names. It does not identify a
   model from a filename. Unknown/legacy archives stay explicitly unclassified;
   safetensors and nonstandard exports are not claimed as supported. Missing
   means absent from recorded search locations, never lost globally. No 30M
   model is selected or loaded.

2. **Timing contamination repaired.** Runtime modes are explicit. Correctness
   mode retains full hashes/event traces and phase synchronization. Performance
   mode performs no hashes, per-step timing/serialization or internal GPU
   synchronization, and does not build diagnostic history. Actual queue,
   packing, allocation/transfers, kernel work, commit and message scheduling
   remain charged. Read-query outputs remain observable in both modes.

   `event_matrix` first validates every executor per microstep, then measures
   fresh performance-mode runs. Final state/consumption/queue/cost validation
   happens after timing. Host boundary timing and CUDA stream-event elapsed
   time are distinct; stream elapsed includes scheduling gaps and is **not** a
   kernel-only profiler sum. Detailed phase decomposition, DRAM bytes and energy
   remain unavailable rather than fabricated. Diagnostic microsteps in result
   files are labelled as a separate untimed pass.

3. **Transformer prefill repaired.** T1 now projects and attends a whole prefix
   in parallel, retaining unexpanded GQA KV. Zero-start uses the original RoPE
   and causal attention. Nonempty-cache chunks use absolute offsets and an
   explicit prefix-aware causal mask. A0/A1 both support this path. Tests forbid
   a hidden decode_one fallback and compare logits and KV against serial decode,
   including chunk continuation, restore, batch 1/2 and the 128-position boundary.
   X3/X4 remain the exact existing sequential reference; this is not a claim of
   an optimized CLM prefill or a production Transformer serving stack.

4. **Training source identity repaired.** `architecture_source_sha` is no longer
   current HEAD. It stays null/UNKNOWN unless explicitly recorded or bound by
   a sidecar matching the checkpoint hash. Adapter source is separate. A source
   declaration is not independent training provenance certification. Saved corpus
   hashes bind actual manifests and tokenizer bytes; nearby unrelated manifests
   no longer cause a binding assertion or supply an unverified identity.

5. **Boundary/input/tolerance path repaired.** The incremental CLI uses fixed
   128-token diagnostic IDs and hashed policy/input files. Fixed engineering
   criteria are keyed by device/dtype. Unlisted combinations stop before timing.
   The current CPU loader dtype and stored tensor dtypes are distinct from
   historical autocast precision. FP16/BF16 criteria remain uncalibrated. The
   fixed IDs are not a validation-quality corpus; H3 still needs bound shared
   quality data and its actual harness remains explicitly blocked.

6. **Complete correctness validation performed.** All current tests, including
   CUDA fixture tests when available, are run rather than only the original two
   scope checks. JUnit and machine-readable validation accompany this report.
   The first expanded run exposed an asset-binding regression; it was fixed
   before the final complete pass. No test success substitutes for tests on
   recovered trained weights. A short CPU-only engineering smoke is separately
   labelled, with zero optimizer updates and no scientific decision promotion.

## Recovery evidence and limits

The historical replication protocol, inspected directly at 3321a74c, records
T1/M4/X3/X4, development seeds 91002/91003, 10M tokens/model, corrected TinyStories
revision and corpus/tokenizer hashes. Only T1/X3/X4 are eligible now. Historical
training metrics do not become decode measurements. Training source SHA was not
saved by the original resume format; an evidence commit cannot fill that field.

The known tokenizer digest can be matched to a repository tokenizer copy;
that verifies those bytes only. It does not recover resume.pt, prove an actual
checkpoint binding or supply missing train/validation token streams. Recovery
must continue from surviving saved Kaggle Outputs or another authoritative
archive if a locator becomes available. No credentials, locator or surviving
weights are invented. No fallback training is authorized.

## Go / No-Go

- Native-only scope and bounded engineering development: retained.
- Tiny CPU/CUDA cache correctness and four-executor mode contracts: see final
  machine/JUnit results; evidence applies to fixtures only.
- Trained native cache parity: **BLOCKED**, checkpoint bytes unavailable.
- FP16/BF16 trained-model validation: **NOT RUN / NOT CALIBRATED**.
- Formal event hardware benchmark: **NO-GO**, frozen holdout/device/performance
  coverage still required. No speedup decision is made from this smoke.
- Competitive E1-C: **not implemented as a formal runnable comparison**;
  blocked by assets/shared quality conditions and measurement freeze.
- H1/H2/H3: **INCOMPLETE**. 30M remains deferred; no merge to main.

# Implementation Deviations and Open Choices

M0 only; no model mechanism has been implemented or changed.

| Item | Record |
|---|---|
| Deliverable layout | Specification §15 names `docs/EPCLM-004-IMPLEMENTATION-PLAN.md` and `configs/environment.yaml`. They live within the owning experiment directory, following repository organization. |
| Original specification | Copied byte-for-byte. Malformed Pandoc math remains in the source; implementation must consult original paper equations before M3. |
| Environment lock | No dependency installation. Samba audit commit is fixed, but its unpinned Dockerfile is not a validated runtime lock. Missing versions remain null. |
| E0/E1 parameter probe | Existing tied-head Transformer instantiated on CPU, without model computation; 10,130,688 parameters is an initial ±3% candidate, not final ±1% matching or a frozen architecture. |
| Full Attention probe | Existing model has finite `window`; probe uses 16384 for the declared maximum evaluation length. This does not implement unlimited full-KV attention. |
| Potential Mamba reference | Deferred M2 choice if extensions do not work. Must use Samba-aligned Mamba-1 equations and document any adapter/reference deviations. Mamba-3 substitution is not authorized by this specification. |
| Tokenizer | Existing HF `tokenizers` BPE is a reusable option; new 8192-vocabulary corpus/special-token protocol not yet frozen. SentencePiece suggestion has not been silently adopted or rejected. |
| Gradient horizon / token budgets | Not frozen in M0. Concrete choices must be declared before their training paths, with memory/compute implications. |
| Stage status | M1–M7 not started. M0 completion is neither model correctness nor training completion nor scientific confirmation. |

## M1 implementation decisions (2026-11-08)

- Reuse `CachedTransformer` parameters and blocks, with a small `E0.forward` override for true full causal SDPA on the supplied sequence. The old window and inference-cache detach do not define the training path; cached E0 is disabled.
- Reuse Core-001 atomic checkpoint envelope and RNG helpers. Add only an Episode sampler adapter because the old sampler hard-codes <=128-token continuous supervision. Commit counters and rollback semantics are retained; no general framework replacement.
- Short-run FP32 and constant LR are explicit Smoke choices. The specification's later cosine/AMP schedules are not silently used or claimed.
- Real WikiText-103 train-prefix BPE is frozen at 8192 entries with SHA256 and reserved symbolic IDs. ByteLevel BPE uses the existing recipe rather than introducing SentencePiece. Subsequent asset preparation reuses the frozen tokenizer and checks its original training-prefix identity.
- WikiText validation excludes exact lines found in the selected train prefix; discovered overlap required an explicit preprocessing-v2 identity. Per-line EOS-bounded LM chunks are engineering data, not full benchmark evaluation.
- Local network transport used the Windows existing proxy and LFS-verified official Parquet files. No proxy settings or alternate corpus were added to production code.
- M1 CPU training uses a declared 272,480-parameter tiny model. The actual 10,130,688-parameter E0 was instantiated/counted, but has not been trained on CPU or T4 in this task.
- No cross-Kaggle recovery platform tests were added. Same-source local checkpoint restore/rollback was tested because M1 explicitly requires it.
- T4 validation remains pending. The local CUDA entry failed before training (`GPU_UNAVAILABLE`); this is an environment precheck result, not a model failure or a scientific result.

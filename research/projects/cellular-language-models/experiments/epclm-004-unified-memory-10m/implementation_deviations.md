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

# MINICELLS-ROOT-CAUSE-001 · Phase A

The previous quality run reached the eight-token ceiling across arms.
Oracle selection also raised answer NLL on single-Cell questions. This
diagnostic checks base execution without any registered Cell. The original
held-out results remain frozen.

## Reference and gates

The target is the audited MoM 1.3B checkpoint
`5c8be94196cb1827e820077680ac2ee46ac949bc` on one Tesla T4 in FP16.
The reference calls the pinned upstream backbone directly, starting with
the upstream default empty cache, followed by its own LM head. The SDK
path uses an explicit fresh cache and the existing SDK forward. Both
paths run the same weights and FLA recurrent kernels. They share the
existing compatibility wrapper for the removed FLA `head_first` argument.

The official CausalLM no-label wrapper is probed separately with
`return_dict=True`. Its pinned implementation attempts to add an auxiliary
loss to a potentially empty loss value. A failure there is reported
independently from the reference backbone result. The diagnostic uses
no substitute checkpoint and does not patch upstream forward.

- Compare logits at all consumed positions under identical token schedules.
  Every element must satisfy absolute tolerance 0.0625 plus relative
  tolerance 0.005. Compare recurrent and convolution state tensors and
  require matching cache cursors between the SDK and the reference.
- Compare upstream whole-prefill, prefill followed by four one-token
  calls, and fully tokenwise schedules. These probes use the three
  formats of the France fixture with both BOS settings. The frozen
  tolerances are 0.25 absolute and 0.02 relative. Cache cursor values and
  consumed token counts are recorded separately.
- Compare greedy completions from the SDK and the upstream reference loop
  for the same France probes. Stop on the pinned EOS or at 32 new tokens.
- Evaluate six fixed public-knowledge facts in plain continuation,
  in-context fact completion, and question/answer formats. Run each
  with and without the tokenizer-default initial BOS. The opening word
  or integer must match the target. The BOS continuation gate requires
  at least four of six. The question/answer and in-context arms are
  diagnostics. In-context prompts include their answer by design.
- Branches use independent caches. An optional BOS appears once at the
  beginning of an input, never at a chunk or decode boundary. No input
  silently appends EOS. The audit documents the SDK's current BOS
  behavior for writes and queries. This phase performs no Cell writes.

Phase B remains gated on Phase A. Results will inform a separate single-Cell
write/read test with wrong-content and fact-in-prompt controls. Fusion and
router work remain later phases.

## Kaggle

Upload [kaggle_phase_a.py](kaggle_phase_a.py) in Script mode. Select T4,
enable Internet and use the existing `GITHUB_TOKEN` secret. The job selects
one GPU by UUID and publishes `phase_a.json`, `phase_a_rows.jsonl`,
`report.md`, the decision, and manifest to a verified result branch.
It does not run the previous MVP held-out or capacity experiments.

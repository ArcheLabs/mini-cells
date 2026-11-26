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

## Published Phase A result

**Engineering PASS · Phase A FAIL.** On the single T4, all 48 matched-schedule
SDK/reference logit comparisons had zero absolute difference. The matching
cache checks passed, and the six reference greedy comparisons agreed. The
12 cross-schedule upstream comparisons failed the frozen tolerance, with a
maximum observed logit difference of 33.789. The official no-label CausalLM
wrapper separately raised the expected None-plus-Tensor error.
[Published Phase A receipts](https://github.com/ArcheLabs/mini-cells/tree/965eb81959a7f840940b76613a62976c72c6ff95/artifacts/experiments/minicells-root-cause-001/1988d78d3166).

| Format | Without BOS | With BOS |
| --- | ---: | ---: |
| Plain continuation | 2/6 | 3/6 |
| Fact in context | 4/6 | 3/6 |
| Question/answer | 1/6 | 1/6 |

The BOS continuation arm missed its frozen 4/6 gate. The cross-schedule and
generation gates prevent advancing to Phase B.

## Phase A.1 component diagnostic

Run [kaggle_phase_a1.py](kaggle_phase_a1.py) on one T4 with Internet and the
existing GitHub secret. This runs three isolated probes with no model download.
It executes the pinned FLA Triton ShortConvolution, the fused recurrent Delta
operator on fixed synthetic Q/K/V tensors, and the pinned MoM packer on
known token positions. The convolution also has an independent Torch causal
convolution oracle. Split schedules are measured independently for outputs
and final state, with fixed absolute and relative tolerances of 0.002.

The FLA short-convolution source writes the new suffix into its cache for
multi-token updates and invokes the convolution without passing prior cache
values. The single-token path invokes the streaming update. The MoM packer
uses an unstable sort for equal expert assignments. The probes separate
these issues from the Delta recurrence. Standalone failures identify
component limits. Attribution of the full-model logit gap still needs a
layer-level intervention.

Local source replays established two concrete behaviors. The unmodified
MoM packer produced nonchronological expert streams in 6/10 CPU fixtures.
Executing the pinned ShortConvolution Python dispatch with CPU substitutes
for its CUDA operators reproduced the missing-prefix effect for a 5+7
split. Under that same substitute, 5+single-token decode and all-single
schedules matched the whole convolution. The CPU replay cannot explain
the entire model-level difference or certify native CUDA behavior.

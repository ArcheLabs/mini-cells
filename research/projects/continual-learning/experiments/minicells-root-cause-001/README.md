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

## Published Phase A.1 result

**Engineering PASS · Phase A.1 FAIL.** On one Tesla T4, whole
ShortConvolution matched the independent oracle exactly. The table reports
maximum absolute errors against whole execution. Delta values come from a
separate fused-recurrent probe under the same schedules.

| Schedule | Conv output | Conv final state | Delta output | Delta final state |
| --- | ---: | ---: | ---: | ---: |
| Prefill + decode | 2.9921875 | 0.875 | 5.96e-8 | 2.24e-8 |
| Two chunks | 0.80078125 | 0 | 5.96e-8 | 2.24e-8 |
| Tokenwise | 2.9921875 | 1.4921875 | 5.96e-8 | 2.24e-8 |

Expert packing broke chronology in 6/10 CPU and 4/10 CUDA cases. The
T4 failures occurred at lengths 5 and 13 for both assignment patterns.
Packing outcomes are recorded under `cpu_packing` and `cuda_packing`.
The published `mismatches` list contains only numerical operator failures.
[Published Phase A.1 receipts](https://github.com/ArcheLabs/mini-cells/tree/a4fa7d713e600e519dd5f05933deac2032eb7c12/artifacts/experiments/minicells-root-cause-001/a1-ad842dcd369f).

## Phase A.2 full-model trace

Run [kaggle_phase_a2.py](kaggle_phase_a2.py) in Kaggle Script mode with
T4, Internet, and the existing GitHub secret. The diagnostic uses the
audited 1.3B FP16 checkpoint and the BOS plain France continuation fixture
from Phase A. Four independent caches cover whole prefill, two chunks,
prefill plus tokenwise decode, and all-tokenwise execution. Zero-based
token positions align per-layer router logits, expert assignments, actual
packing order, and attention, block and head outputs. For both the expert
and shared paths the trace captures Q/K/V before and after convolution.
Expert tensors are compared on common (layer, expert, original token)
coordinates with padding excluded. The published JSON includes per-token
expert RMS and max absolute activation, full-vector error summaries, and
tokens that enter or leave each expert across schedules.

Convolution records include extension availability, the effective
`use_fast_conv1d` flag and the dispatched FLA Triton or causal-conv1d entry
point for each layer, projection, path and chunk. Router records include
the actual ordered top-2 selection and top-2 versus top-3 margins on logits
and FP32 softmax probabilities. Reports summarize both ordered and
set-wise agreement and the first change across layers and tokens.

Comparisons report the first measured exact difference and the first
difference outside the frozen Phase A bound. Final cache fields and
cursors use the existing Phase A comparison. Observations are limited to
the recorded points. The weights, dependencies, upstream behavior and
acceptance gates are held fixed. This short trace uses fused recurrent
execution.

A.3 is the separately gated comparison of packing-only, convolution-only
and combined interventions against the A.2 baseline. A.4 rechecks base
generation and consistency after those results, including inputs beyond
64 tokens that exercise the chunk kernel. Phase B can start after the
base runtime passes A.4.

## Phase A.3 controlled interventions

A.2 found the first out-of-tolerance stage at layer 0 expert Q after
convolution in all three split schedules. The shared Q input for the
two-chunk schedule agreed exactly before convolution. The unmodified
whole pass also packed at least one expert out of chronological order in
all 24 layers. [A.2 artifacts](https://github.com/ArcheLabs/mini-cells/tree/eb3acfe54f9cbe59dc23ac640683e173d94f8a3d/artifacts/experiments/minicells-root-cause-001/a2-e686342397c2).

Run [kaggle_phase_a3.py](kaggle_phase_a3.py) in Kaggle Script mode with
T4, Internet, and the existing GitHub secret. The four arms use the A.2
prompt and token schedules, FP16 1.3B weights and the pinned dependencies.
Each pass starts with an independent cache. The interventions are scoped to
the benchmark process and independently toggled across arms.

| Arm | Stable expert sort | Correct Triton convolution state |
| --- | --- | --- |
| Original | Off | Off |
| Packing only | On | Off |
| Convolution only | Off | On |
| Both | On | On |

The sort replaces only the tie behavior of the pinned sort. The convolution
intervention supplies width-4 weights to the existing one-token Triton
entry and prepends the last width-minus-one cached inputs to multi-token
Triton prefill, then keeps the causal suffix and rolls the full cache.
The experiment keeps the upstream expert padding, routing, Delta kernels,
and cache management outside convolution. Expert padding and kernel
rounding may still contribute to remaining schedule differences.

The T4 component preflight checks the stable packer and independently
compares the original and corrected FLA paths against a Torch causal
oracle. Both a single-lane fixture and a four-expert 1024-channel SiLU
fixture cover whole, split and tokenwise schedules. The corrected path
must meet A.1's numerical tolerances and produce the exact expected final
convolution cache before the four model arms run.

Each arm compares split against its own whole pass under the frozen Phase A
tolerance. Comparisons to the original use the same schedule. The JSON
includes final Logits differences, first numerical divergence, expert
set agreement, packing inversions, cache summaries, and paired effects on
mean absolute Logits error. A.3 measures a short fused-recurrent fixture.
The A.4 generation and long-input acceptance gate stays separate.

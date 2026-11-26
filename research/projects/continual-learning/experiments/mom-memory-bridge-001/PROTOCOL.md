# MOM-MEMORY-BRIDGE-001

This development diagnostic compares continuous native memory and single-Cell
persistence with frozen MoM 1.3B. The nine single-target MOM-ROUTE-002 facts,
questions and distractors are reused as a paired development fixture. The
340M result was 9/9 for these single-target Oracle cases and 12/12 across
single- and multi-target cases. No 340M model run is part of this study.
[MOM-ROUTE-002 results](https://github.com/ArcheLabs/mini-cells/blob/kaggle/mom-route-002-ea0f1c449700/artifacts/experiments/mom-route-002-ea0f1c449700/experiment/decision.json)

## Frozen controls

Use one Tesla T4, FP16 1.3B checkpoint
`5c8be94196cb1827e820077680ac2ee46ac949bc`, the pinned tokenizer and MoM
revision, and FLA 0.2.2. Use the bounded, corrected Fused Recurrent path
from A.3.2. Knowledge writes use at most 64 tokens per forward and every
scoring sequence is verified to fit within 64 tokens. A.4 preserved the
native Chunk acceptance failure. This bridge has a separate development
decision and cannot open the Phase B gate.

A/B/C/D/E use the pinned tokenizer without BOS to match the existing SDK
Cell write convention. Across B, C and D the memory text and write
chunking match exactly. All arms use identical question and candidate
tokens. A separate live-path control uses the historical BOS-prefixed
`Memory:\n` wrapper for both empty context and exact-record context.
Report token counts, including unequal counts if present.

| Arm | Memory path |
| --- | --- |
| A | Fresh cache, question only |
| B | Exact record processed into a live, complete native cache |
| C | Exact record written through the SDK Cell manager, then the saved recurrent bundle reopened with empty convolution state and cursor 0 |
| D | Exact record written into a full native cache, saved and reopened using the SDK stream checkpoint codec, retaining recurrence, convolution and cursor |
| E | One domain Cell with all three positive records from that domain, with the target record last |
| E live / E full | Domain-text controls through the same live and full-checkpoint paths as B and D |

Each candidate starts from a separate stream. Query scoring runs through the
same direct native backbone in every arm. Full-stream round trips compare
every native tensor and sequence cursor before scoring. Recurrent-only
bundles are checked against the recurrent fields of the matched live
prefix and inspected again after scoring. D is an experiment-local use of
the already existing stream codec. It does not change the SDK Cell format
or test a fresh-process restart. No router, multi-Cell fusion, training,
model change, or kernel optimization occurs in this study.

## Evaluation

Score the candidate at each teacher-forced answer position and archive its
token, summed and mean log probabilities. Gold NLL is negative mean
log probability. Primary margin is the difference of mean candidate log
probabilities, following MOM-ROUTE-002. Summed margins are secondary.
Report paired deltas and sign counts for B/A, C/B, D/B, E/C and the matching
domain versus exact live and full-state contrasts.

Interpret C/B only with the observed recurrent write parity. A D/B
probability divergence after field-exact restore is a separate execution
finding. Domain tests measure the effect of two additional preceding records
under the bounded write policy. The longer prefixes also introduce a
second prefill call, so this comparison does not establish a general
capacity limit or a schedule-independent interference estimate.
All effects are descriptive development evidence.

## Kaggle

Run [kaggle_bridge.py](kaggle_bridge.py) in Script mode with T4, Internet
and the existing `GITHUB_TOKEN` secret. Before loading the real model,
the job runs the CPU entry-point regression through all nine cases,
both disk codecs, JSONL reporting and checksum generation.
The job publishes case scores,
summaries, checksums and source identity to a dedicated result branch.
Temporary state bundles are kept out of the published artifacts.

# MiniCells M0 Developer Preview and M1-001 Value Swap

## Independent gates

Both lanes run the same installed candidate wheel on a single NVIDIA T4.
The native profile is pinned FP16 MoM 1.3B with corrected short Fused
Recurrent execution and a 64-token bound. It stores recurrent state only.
Native Chunk and BF16 are outside these acceptance gates.

- M0 runs the repository test suite plus the SDK integration preflight.
  The installed-wheel T4 gate covers empty and explicit reads, repeated
  read immutability, long inputs split at 64 tokens, unrelated Cell
  isolation, Fork, Append, Delete, and cross-process Export/Import with
  compatible identity, SHA-256 and no overwrite. Its decision does not
  require M1 success. Generated text is recorded as engineering evidence.
- M1-001 is a single-fact causal test. No M1-002 capacity, M1-003 memory
  persistence evaluation or M1-004 free-generation gate starts from this
  launcher. M0's cross-process codec check does not count as M1-003.

## M1-001 frozen before execution

Generate 30 independent project/code associations with seed 47031 and
the pinned tokenizer. Each project has two unique candidate codes with
equal target token counts across both query forms and equal write token
counts. Neither code is in the prompt. Preserve the actual token IDs,
dataset hash and configuration hash. The canonical tokenized dataset SHA-256 is
`413dd77f2ca9e40d0d4ab5b9d0b83dc1478ffe771b66f65c4f790232bb078913`. Both candidate values are evaluated
as the target in separate contexts and separate recurrent-only Cells.

| Arm | State before scoring |
| --- | --- |
| A | Empty native cache |
| B original / swapped | Corresponding fact then query on a live complete cache |
| C | Original-value Cell written and read through SDK storage |
| D | Swapped-value Cell written and read through SDK storage |

Score summed teacher-forced log probability and report per-token scores
and written-value NLL. Execute the same candidate scoring a second time
in reverse order, with separate working caches for every candidate.
A fact passes only if both query forms have positive A-minus-B margin
when A is written and negative margin when B is written.

- Full-context control and Cell condition each require at least 27/30
  paired flips. Full-context or order-control failure makes the run
  invalid for a Cell conclusion.
- Take the average of the two wording margin differences per fact.
  Resample 30 facts with replacement 10,000 times with seed 87293.
  The equal-tailed 95% interval lower bound for the mean must exceed zero.
- Use a two-sided exact sign test on the no-memory fact-level average
  margin at alpha 0.05. Ties are counted separately. Passing this check
  means the test did not detect a systematic A/B preference.
- Keep all frozen thresholds after the run. If Value Swap fails, inspect
  the single-Cell path before capacity. If it passes, review M1-002's
  preregistration separately.

## Launch and publication

Use [the Kaggle script](kaggle_next_stage.py) with T4, Internet and
`GITHUB_TOKEN`. The default processes both jobs and publishes separate
GitHub result branches. For independent execution set the
`MINICELLS_NEXT_STAGE` environment variable to `m0` or `m1`. The setup
runs the installed-wheel CPU test before loading model weights.

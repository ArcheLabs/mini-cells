# Functional Boundary Oracle 001

## Question

Can an unchanged Granite MoE CLM substrate expose a **sub-expert writable coordinate** that acquires a new held-out behavior while preserving unrelated historical behavior, **when the algorithm is explicitly allowed to use frozen-base historical supervision**?

This is a kill test for the weaker and more fundamental hypothesis:

> Safe functional write coordinates exist under favorable history-supervised conditions.

It is **not** a zero-replay test.

## Why this test exists

MoE Mutation 001 showed that a whole expert slice is strongly writable but can create excessive unrelated drift. That experiment did not establish whether the problem was the absence of a safe functional boundary or simply the inability to discover one without enough historical information.

This test separates those questions by granting an oracle condition: public calibration inputs plus frozen-base outputs.

## Frozen base

- `ibm-granite/granite-3.1-1b-a400m-base`
- revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- required Conversion 001 manifest identity `dd2b9c750567ff73b1d48e39eb7d1e1213eea9116a68c5164d023420f5a4d670`
- target MoE layer: 23
- formal seeds: `26090511`, `26090512`, `26090513`

The router and every parameter outside the selected aligned group remain frozen.

## Physical coordinate

Granite stores each expert as a packed gate/up tensor and a matching down tensor. For intermediate width `512`, one coordinate is a contiguous group of `32` intermediate channels:

```text
gate rows:   [g:g+32]
up rows:     [512+g:512+g+32]
down cols:   [g:g+32]
```

Thus one candidate modifies exactly 1/16 of one expert's intermediate channels. The group is a **physical mutation coordinate**, not a claimed CLM Cell.

## History supervision and leakage boundary

Two disjoint prompt sets are frozen in `protocol.json`:

- `history.selection_prompts`: visible to footprint selection and preservation training.
- `history.evaluation_prompts`: never used to select the footprint, compute the optimizer loss, choose a checkpoint, or early-stop.

The frozen base supplies next-token logits for both. Original Granite training data are not required.

A formal result is invalid if these prompt sets overlap.

## Coordinate selection

For every aligned group in every expert of layer 23:

1. Compute a new-task gradient-energy signal from target NLL.
2. Compute historical importance from the gradient energy of the frozen base's own Top-1 next-token behavior on `history.selection_prompts`.
3. Measure expert routing specificity:

   `target Top-K coverage - history-selection Top-K coverage`.

4. Score each group with the frozen formula:

```text
0.5 * log((new_group_rms + 1e-12) / (history_group_rms + 1e-12))
+ route_specificity
```

Candidates with target routing coverage below `0.25` are ineligible. Ties choose lower expert index, then lower group index.

## History-supervised training

Only the selected aligned group can change.

Training combines:

- new-task target cross-entropy;
- frozen-base KL distillation on `history.selection_prompts` with weight `12.0`.

Every two steps, the runner evaluates the candidate on the new-task training set and history-selection set. Among **nonzero** candidates with history-selection KL <= `0.05`, the final mutation is the checkpoint with maximum new-task train NLL gain; ties choose the earlier step.

No `history.evaluation_prompts` are consulted here.

## Formal gates

A seed passes only if all gates pass:

- held-out new-task NLL gain >= `0.5`;
- unseen history-evaluation mean KL <= `0.05`;
- unseen history-evaluation next-token Top-1 identity >= `0.96875` (31/32);
- target-layer router Top-K identity == `1.0`;
- selected footprint <= `0.0625` of one expert intermediate width;
- selected expert new-task routing coverage >= `0.25`;
- mutation delta is nonzero;
- exact parameter rollback succeeds;
- forward rollback error does not exceed measured base forward repeatability by more than `1e-5`;
- Conversion 001 identity matches;
- selection/evaluation history sets are disjoint.

## Decision

- **SUPPORTED**: at least 2 of 3 formal seeds pass.
- **REJECTED**: at least 2 of 3 formal seeds fail.
- Otherwise: incomplete.

A supported result establishes only:

> Under explicit frozen-base historical supervision, this construction found a substantially sub-expert aligned coordinate capable of new held-out gain with bounded drift on a disjoint historical calibration set.

A rejected result is more serious than MoE Mutation 001 failure, because the algorithm was allowed historical supervision. It still does not mathematically prove that no safe representation or optimizer exists.

## Explicit non-claims

This test does not establish zero-replay learning, data-free discovery, autonomous Cell formation, composition, mergeability, or preservation of the full Granite pretraining distribution.

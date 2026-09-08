# History Compression 001 — Replay Budget Ladder

## Question

Functional Boundary Oracle 001 established a positive-control result: with explicit frozen-base historical supervision, a 32-channel aligned sub-expert coordinate could acquire a new held-out behavior while preserving withheld historical calibration behavior.

History Compression 001 asks the next narrower question:

> How much learner-visible historical calibration input is required to rediscover and safely train such a coordinate?

This experiment deliberately compresses only the **historical prompt budget**. It does not simultaneously replace replay with Fisher, low-rank gradient certificates, activation sketches, or another preservation algorithm. Those are separate mechanisms and should be tested only after the replay-budget threshold is known.

## Frozen substrate

- Model: `ibm-granite/granite-3.1-1b-a400m-base`
- Revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- Conversion identity: `dd2b9c750567ff73b1d48e39eb7d1e1213eea9116a68c5164d023420f5a4d670`
- Layer: 23
- Expert intermediate width: 512
- Writable group: 32 aligned intermediate channels = 6.25% of one expert
- Router and every non-selected parameter remain frozen.

`expert_is_cell=false` and `group_is_cell=false` remain explicit.

## Formal seeds

- `26090611`
- `26090612`
- `26090613`

Each seed evaluates every compression mode. A mode is supported when at least 2/3 seeds pass every frozen gate.

## Compression ladder

| Mode | Learner-visible history prompts | Fraction of full calibration | Meaning |
|---|---:|---:|---|
| `full_32` | 32 | 100% | positive control |
| `tiny_8` | 8 | 25% | 4× prompt compression |
| `tiny_2` | 2 | 6.25% | 16× prompt compression |
| `zero_0` | 0 | 0% | zero-history limit under this task family |

For each seed, the 32 selection prompts are deterministically ordered by SHA-256 of `seed:index:prompt`. The 8- and 2-prompt modes use nested prefixes of that exact order. This prevents manual cherry-picking and keeps the ladder nested within a seed.

The separate 32-prompt evaluation set is never learner-visible and may not be used for coordinate selection, optimizer loss, candidate checkpoint selection, or early stopping.

## Coordinate selection

### Nonzero-history modes

For every expert × 32-channel group candidate:

1. `new_group_rms`: new-task target-NLL gradient energy;
2. `history_group_rms`: frozen-base Top-1 NLL gradient energy computed only on that mode's learner-visible historical subset;
3. routing specificity: new-task top-k expert coverage minus learner-visible-history coverage.

Score:

```text
0.5 * log((new_group_rms + 1e-12) / (history_group_rms + 1e-12))
+ route_specificity
```

### `zero_0`

No historical prompt, teacher output, routing statistic, or historical gradient is learner-visible.

Score:

```text
0.5 * log((new_group_rms + 1e-12) / 1e-12)
+ new_route_coverage
```

The minimum new-route coverage remains 0.25. Ties choose the lower expert index, then lower group index.

## Training

All modes use the same masked manual SGD primitive on one selected aligned group:

- FP32
- batch size 4
- maximum 24 steps
- learning rate 0.03
- selected-group gradient norm cap 1.0
- candidate evaluation every 2 steps.

For nonzero-history modes:

```text
L = L_new + 12 * KL(P_base || P_current)_mode-local-history
```

A candidate checkpoint is eligible only if its learner-visible-history KL is ≤ 0.05 and its new-task train gain is positive. Among eligible candidates, choose maximum train NLL gain; ties choose the earlier step.

For `zero_0`, training uses new-task cross-entropy only. Candidate selection uses positive new-task train gain only. No hidden historical information is allowed to influence candidate selection.

## Final gates

Every mode/seed must pass all of the following:

- Conversion identity exact.
- Historical selection/evaluation sets disjoint.
- New-task held-out NLL gain ≥ 0.5.
- **Withheld** history-evaluation mean KL ≤ 0.05.
- Withheld history-evaluation Top-1 identity ≥ 31/32.
- Target-layer router Top-K identity = 1.0, independently re-measured from a fresh base reload plus serialized mutation reapply.
- Selected expert fraction ≤ 1/16.
- Selected expert new-task route coverage ≥ 0.25.
- Nonzero mutation delta.
- Exact parameter rollback.
- Forward rollback excess above the measured base-repeatability floor ≤ 1e-5.

The withheld evaluation set is used only after training for scientific scoring.

## Decision

`full_32` is a required positive control. If it is not supported, the compression ladder is classified as uninterpretable rather than used to infer a minimum budget.

When the positive control is supported, record the smallest **observed supported** prompt budget among 32, 8, 2, and 0. Also record whether support is monotone with increasing historical budget. Non-monotonic results are evidence and must not be repaired post hoc.

Primary statuses:

- `HISTORY_COMPRESSION_ZERO_HISTORY_SUPPORTED`
- `HISTORY_COMPRESSION_TO_2_SUPPORTED`
- `HISTORY_COMPRESSION_TO_8_SUPPORTED`
- `HISTORY_COMPRESSION_BEYOND_FULL_NOT_SUPPORTED`
- `HISTORY_COMPRESSION_POSITIVE_CONTROL_FAILED`

A zero-history PASS is still not a general proof of replay-free continual learning. It only establishes that these learner-visible historical calibration inputs were unnecessary under this frozen task/model/protocol family.

## Logging and durable evidence

Notebook stdout is intentionally compact. Full coordinate scores, per-step training details, metrics, mutation manifests, and tensors are written to durable files instead of dumped to the notebook page.

The aggregate publisher must create:

- `decision.json`
- `summary.csv`
- `visualization/history-compression-summary.svg`
- `visualization/summary.md`

The source of scientific truth remains the durable `result.json` files and the frozen protocol, not the notebook rendering.

# Shadow Cell Validation 001 v2

Status: **REGISTERED_NOT_RUN**

This validation asks whether a candidate can learn outside the accepted model
and then receive expression authority through controlled maturity without
mutating mature Cells. It uses the checked-in CLM-0.4-mini `TinyCLMDecoder`
configuration as the Native substrate for this checkout.

## What v2 changes from v1

- accepted parameters and accepted optimizer state are immutable during Shadow training;
- the Shadow is an additive sidecar and never enters the accepted global Top-K router;
- a full native-width zero-output operator is trained;
- the complete preregistered maturity frontier is measured;
- Shadow Oracle is an architectural upper bound for admission;
- replay-free Shadow Sketch uses bounded functional sufficient statistics;
- Corrected Direct projects the realized AdamW proposal, including decay;
- Task-ID Shadow separates isolated capacity from learned input-only routing;
- accepted transitions are written as copy-on-write artifacts.

The canonical maturity grid is `[0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]`. Formal
seeds are `95311`, `95312`, and `95313`; development smoke uses `95301`.

## What v2 does not claim

This experiment does not claim autonomous continual learning, a natural Cell
ontology, autonomous mitosis, or a zero-replay global-safety theorem. A Cell is
operationally a versioned candidate functional state. The input-only gate is an
executable admission mechanism, not evidence that cosine routing is a natural
Cell ontology.

## Arms and interpretation

`corrected_direct` is the corrected mature-Cell baseline. `shadow_full` fixes
maturity at 1.0. `shadow_oracle` selects maturity using hidden historical
evaluation only and is the decisive capacity/admission upper bound.
`shadow_sketch` selects from bounded state without retaining raw history.
`task_id_shadow` is an oracle-routing capacity control and is excluded from
claims of autonomous learning.

The runner records separate optimizer, candidate-trainer, oracle-evaluator,
and hidden-evaluator historical-example counters. Shadow optimizer and learner
replay must remain zero. Every phase writes its full frontier and, if selected,
a new accepted artifact under `checkpoints/`; the parent accepted state is not
trained in place.

## Execution

```bash
python scripts/research/run_shadow_cell_validation_001_v2.py \
  --phase smoke --seed 95301 --device cpu
```

The expected smoke marker is `SHADOW_CELL_VALIDATION_001_V2_SMOKE_PASS` and it
does not emit a scientific conclusion.

For a formal run, provide the canonical checkpoint when available:

```bash
python scripts/research/run_shadow_cell_validation_001_v2.py \
  --phase formal --seed 95311 --device cuda --checkpoint /path/to/checkpoint.pt
```

Use `--preflight-only` to validate the frozen protocol, seed, checkpoint path,
device, and output directory without training.

Figures are generated from result JSON by the runner and can be regenerated with:

```bash
python scripts/research/report_shadow_cell_validation_001_v2.py \
  --results results/shadow-cell-validation-001-v2-developmental-maturation/seed-95301
```

No formal seed has been run or classified by this registration.

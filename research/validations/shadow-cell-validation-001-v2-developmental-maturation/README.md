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

Formal execution is fail-closed. It requires both a canonical accepted
checkpoint and a JSON dataset matching the registered data contract; there is
no synthetic fallback in formal mode. The dataset must contain these splits:
`A_train`, `A_calibration`, `A_eval`, `B_train`, `B_calibration`, `B_eval`,
`C_train`, `C_eval`, `D_train`, and `D_eval`. Each item uses the checked-in
`ScoredTokenExample` fields (`example_id`, `address_id`, `tokens`, and
`target_mask`). A/B train and evaluation addresses must share one complete
sparse route tuple.

Formal execution is still locked in this commit because the canonical
checkpoint and formal dataset are external artifacts. Before running any
formal seed, make a separate pre-formal commit that records their SHA-256
values in `protocol-lock.json`, sets its status to `FROZEN`, and updates the
matching SHA fields in `protocol.json`. The runner also checks the protocol
SHA-256, so any protocol change requires a new lock commit.

The lock can be generated from the exact external files with:

```bash
python scripts/research/lock_shadow_cell_validation_001_v2.py \
  --checkpoint /kaggle/input/canonical/checkpoint.pt \
  --seed-dataset 95311=/kaggle/input/shadow-v2/formal-seed-95311.json \
  --seed-dataset 95312=/kaggle/input/shadow-v2/formal-seed-95312.json \
  --seed-dataset 95313=/kaggle/input/shadow-v2/formal-seed-95313.json \
  --write
```

Review and commit the resulting protocol/lock change before starting formal
execution. The dataset lock is per seed input; all three formal dataset files
must be derived from the same registered dataset release and manifest hash.

After that lock commit, a formal run is:

```bash
python scripts/research/run_shadow_cell_validation_001_v2.py \
  --phase formal --seed 95311 --device cuda \
  --checkpoint /kaggle/input/canonical/checkpoint.pt \
  --dataset /kaggle/input/shadow-v2/formal-seed-95311.json
```

Use `--preflight-only` to validate the frozen protocol, seed, checkpoint path,
device, and output directory without training.

To publish the completed formal result to GitHub after each seed, first add a
Kaggle Secret named `GITHUB_TOKEN` containing a fine-grained token with
repository Contents read/write permission, then add `--push-results`:

```bash
python scripts/research/run_shadow_cell_validation_001_v2.py \
  --phase formal --seed 95311 --device cuda \
  --checkpoint /kaggle/input/canonical/checkpoint.pt \
  --push-results \
  --publish-branch kaggle/shadow-cell-validation-001-v2-results
```

The publisher verifies the protocol hash, aggregates all completed formal
seeds, writes provenance and SHA-256 manifests, excludes binary checkpoints
from Git, commits the curated evidence, and pushes the result branch. Re-run
the same command for the remaining registered seeds; each completed seed
updates the same result branch. The token is read only from the environment or
Kaggle Secret and is never stored in the repository or notebook.

Figures are generated from result JSON by the runner and can be regenerated with:

```bash
python scripts/research/report_shadow_cell_validation_001_v2.py \
  --results results/shadow-cell-validation-001-v2-developmental-maturation/seed-95301
```

No formal seed has been run or classified by this registration.

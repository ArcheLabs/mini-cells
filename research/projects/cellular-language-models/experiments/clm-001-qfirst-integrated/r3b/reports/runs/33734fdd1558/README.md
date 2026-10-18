# Completed training / full offline analysis

- Public directory key: `33734fdd1558`.
- Immutable internal training run: `r3b-a7792ef30582-ea3ea51c`.
- Training source: `a7792ef30582dfcaf324cca7e5245c8430df48b2`.
- Result commit: `0df519b6352412bf483945ee7e3c34e02d9ebf97`.
- HF input revision: `828585f1ff93420aa4485c725211ed28f70d15f3`.
- Source result branch: `results/clm001-r3-a7792ef30582-33734fdd1558`.

`checkpoint-index.json`, `experiment-manifest.json`, and `resource-ledger.json` are byte-for-byte copies from that result commit. They are public metadata, not the private model bundle. Historical reports are retained.

## One launch

Import this branch's `r3b/analysis-notebook.ipynb` into the existing Kaggle Notebook, keep its existing `HF_TOKEN` and `GITHUB_TOKEN` secrets enabled, select GPU, and **Run All**. No source hash, sampling budget, stage, checkpoint or mode needs editing.

The Notebook verifies its pinned analysis checkout, reads this run's immutable input, verifies the training identity and every checkpoint, generates 102 logical slots from the existing frozen schedule, reuses verified private HF results, executes remaining slots, writes the private coverage index, aggregates recorded observations, draws figures, writes performance reports, and publishes public metadata with a verified GitHub receipt. No optimizer update, Pilot, retraining or Confirmation access occurs.

If the soft session budget is reached, the current task finishes and commits before pausing. Run the same Notebook again: it reads the authoritative private HF index. A fresh local directory is safe; no manual checkpoint download or phase approval is needed.

## Status and integrity

Quality 39 + Functional 42 + Action 21 = 102 always. Coverage states: COMPLETE, UNAVAILABLE, BLOCKED, PENDING. Missing Development banks are not replaced. Blocked tasks are retried on the next launch; verified COMPLETE tasks are reused. A terminal pass has no PENDING tasks. A session boundary may leave PENDING with `SESSION_BOUNDARY_RESUMABLE`.

Input bundle corruption blocks diagnostics; the mounted input is not repaired or re-labelled. Each executed task retains original file/state/RNG/optimizer/event-cursor fingerprint checks. A private result cannot become COMPLETE without `training_state_unchanged=true` and zero updates. An interrupted result upload is adopted after immutable readback without repeating computation.

## Current evidence

The local deliverable includes real training-ledger statistics. It has **not** executed the private 102-task analysis: the checked-in coverage is 102 PENDING, not 102 completed observations. HF private contents have not been accessed in this environment. The Notebook supplies runtime analysis-source identity and actual HF completion revisions; the checked-in input reference is pinned to the implementation commit.

Training CUDA update resource sum: 4,236.461 seconds. This is not end-to-end training wall time. The ledger does not contain trustworthy full-run start/end timestamps or complete mandatory-event totals, so the two-hour verdict is NOT_VERIFIED. Missing observations and measurements are not interpreted as scientific failures. The complete study resource total is unavailable because event GPU work was not separately timed.

Public publication receipts identify a verified content commit and per-file hashes. A second commit publishes that receipt; the local result records the final receipt commit. Analysis namespaces and pointers remain separate from training `latest.json`.

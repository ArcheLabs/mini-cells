# Core Validation 008 Status

Status: **IMPLEMENTATION_COMPLETE / FORMAL_GPU_RUN_PENDING**

Scientific decision: **not yet emitted**.

## Completed

- Frozen Core 008 protocol committed before formal seeds are opened.
- Parent identity pinned to the same Pythia/SlimPajama source and exact manifest used by Core 007 / the Core 008 preflight bridge.
- Primary measurements moved away from raw whole-model NLL to normalized write reconstruction and local functional action.
- Replay-free per-atom covariance / `Q` certificate implemented.
- Safe mutation implemented as low-rank residual projected through `I - Q Q^T`.
- Safe-write infeasibility drives atom growth.
- Equal 4096-factor-scalar budget enforced across monolithic, rank-1, rank-2, rank-4 and adaptive-rank variants.
- Input-only deployable coefficient router implemented; no hard functional-mode label gate remains.
- Hidden full activation history is evaluator-only and cannot affect learner decisions.
- Three formal seeds frozen: `80821`, `80822`, `80823`.
- Resumable per-seed runner, reporter, publisher, orchestrator, Kaggle one-cell notebook, and mechanism unit tests added.
- Published hidden cache is explicitly excluded; a lost Kaggle cache is not a blocker.

## Pending

Run the formal experiment on a fresh GPU environment. Until all three formal seeds complete, `report_core_validation_008.py` must emit `CORE008_CONFIRMATION_INCOMPLETE` with `scientific_decision=false`.

The canonical execution entry point is:

```bash
python scripts/research/orchestrate_core_validation_008.py \
  --branch codex/core-validation-008-certified-functional-atoms \
  --secret-name GITHUB_TOKEN \
  --device cuda \
  --push-results
```

or the one-cell Kaggle notebook at:

`research/notebooks/04-continual-learning-core/core-008-certified-functional-atoms.ipynb`

No scientific threshold may be changed after a formal seed is opened.

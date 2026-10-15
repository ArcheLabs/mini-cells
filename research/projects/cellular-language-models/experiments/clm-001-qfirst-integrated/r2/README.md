# CLM-001-R2 — Stabilized Relative Q-First

Independent, single-seed exploratory implementation of the supplied protocol. **NOT_TRAINED**.
Historical R1 source, results and checkpoints are unchanged. R2 never resumes R1 weights.

## One Kaggle launch

Import `r2/notebook.ipynb` from branch `codex/clm001-r2` into the existing
`bingjiangli/minicells` Notebook. Select T4 ×2, enable Internet and the existing
`HF_TOKEN` / `GITHUB_TOKEN`, then Run All. Source identity is resolved and locked
internally; do not fill source/hash/seed fields. Save Version retains full Output.

The launcher performs CPU tests → return check → data identity → concurrent real
GPU Pilot. A failed check blocks the next stage. This implementation request does
**not** authorize the 500-step development run: the shipped authorization is false.
After explicit development authorization is committed, the same launcher continues
automatically from successful preflight to two-arm training, analysis and return.
No confirmation access is implemented. No process is killed on a wall-clock deadline.

## Implementation

- Same 6 accepted + 1 reserved Cell pool; self-connections, explicit CTX_READ,
  static causal input K/V, real hard execution, minimum one action and budget 16.
- QF-R2: STOP-relative Huber with genuine HT inclusion weighting, plus a capped
  confidence-filtered ranking auxiliary. Ranking is a biased sampled-pair heuristic.
- J0-R2: path actor/critic; same exploration, operators, parameter groups and data.
- Depth-stratified reservoirs retain only visited states, one detached state per
  bucket/microbatch; merge by event population, select at most two states/step.
  Alternating bucket quotas are a training distribution, not a natural-event mean.
- 2→4→8 paired CRN suffixes only on ambiguous comparisons, with a conservative
  768-operator per-step cap; reserve initial samples for all probes first.
- Continuous and Connection clipping are independent. QF critic weights remain
  resident for capacity matching but receive no gradients or optimizer updates.
- Candidate numerical/contract/no-signal failures are typed, preserve a full parent
  bundle witness, and terminate only that private branch. Oracle/integrity failures
  propagate. Native tracebacks and diagnostic hashes remain in Kaggle Output.
- Each milestone main update is checkpointed before its lifecycle transaction.
  Every committed proposal/admission unit stores its cursor and private state in
  the same atomic checkpoint. Restore skips committed units and never repeats a
  committed training update. Uncommitted private work may be deterministically replayed.
- Root aggregation verifies hashes and preserves each arm independently. GitHub
  receives aggregate allowlisted JSON only; checkpoints, tokens, target labels,
  traces and optimizer state remain local. `publish-only` never trains.

## Local / CI CPU entry

```sh
python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r2/run.py --mode cpu-preflight --output /tmp/clm001-r2-cpu
```

CI: `.github/workflows/clm001-r2-cpu.yml`; includes frozen R1 regression tests.
See `ARCHITECTURE_AUDIT.md`, `reports/engineering.json` and the verbatim `SPEC.md`.

## Research interpretation

Relative centering alone leaves action ordering unchanged; any gain must come from
supervision/optimization. Neither trained paths, finite gradients, visualizations nor
Shadow promotion prove the scientific hypothesis. Story bootstrap is not seed replication.
No external MoE training arm or cross-architecture causal claim is added.

# Core Validation 008 — Certified Adaptive Functional Atoms

## Question

Can real frozen-language-model write demands be represented as a sparse, reusable composition of low-rank functional atoms while a replay-free local subspace certificate protects previously registered behavior?

Core 008 deliberately does **not** search for another hard semantic or functional Cell label. Core 007 already showed that train-time functional mode identity was not reliably recoverable on held-out inputs, and the Core 008 preflight bridge showed that tiny whole-model NLL gaps were dominated by weak Cell effect. The primary evidence here is therefore local functional geometry.

## Mechanism

For each frozen Pythia sequence, Core 008 extracts the same foundation-path projected write demand used by Core 007:

\[
G = \operatorname{mean}_t[(U^T \partial L/\partial h_t)z_t^T].
\]

The experiment normalizes `G` by its Frobenius norm and learns an online dictionary of functional transforms `B_k`. A write is reconstructed sparsely:

\[
\hat G = \sum_{k\in S} \alpha_k B_k,\qquad |S|\le 4.
\]

Each atom keeps only bounded dependency state: a covariance of committed projected activations and the derived 99.5%-energy basis `Q_k`. When an already-used atom is changed, the proposed residual write is projected through

\[
P_{free}=I-Q_kQ_k^T
\]

and only a low-rank approximation of `R P_free` may be committed. Thus the registered certificate constraint is

\[
\Delta B_k Q_k \approx 0.
\]

If existing atoms cannot safely lower the write residual enough, the system grows a new atom. This makes growth a consequence of safe-write infeasibility rather than a semantic conflict heuristic.

## Rank comparison

The main variants are:

- `monolithic_certified`: one certified transform, up to the full 32-rank-unit budget.
- `rank1_atoms`: persistent atom rank cap 1.
- `rank2_atoms`: persistent atom rank cap 2.
- `rank4_atoms`: persistent atom rank cap 4.
- `adaptive_atoms`: atom rank may grow up to 8; each safe growth action chooses 1–4 additional rank units.

Every main variant shares the same conceptual factor budget: 4096 scalars. At `d=64`, one rank unit costs `2*d=128` factor scalars, so no variant may consume more than 32 rank units in total. Atom-count ceilings do not reduce the budget for rank-2/rank-4 variants.

Rank-1 is therefore tested as the smallest persistent functional unit, not assumed to be correct because a single-token gradient is rank-1.

## Primary measurements

Core 008 gates normalized Frobenius write reconstruction, local action reconstruction `Z G^T`, deployable input-only routing, reuse, growth, unresolved writes, budget use, certificate rank, exact certificate violation, and hidden-evaluator historical action drift.

Whole-model NLL is not a support gate. The preflight bridge established that the existing Cell intervention is too weak for raw whole-model NLL closeness to diagnose functional equivalence reliably.

The hidden evaluator may retain exact historical `z` rows only to measure false-safe drift. They are not available to sparse assignment, update selection, growth, routing, or thresholds.

## Frozen identity

Core 008 reuses the pinned Core 007 real-representation source identity:

- model: `EleutherAI/pythia-160m@step143000`
- data: `DKYoon/SlimPajama-6B@1224c66add28b96ab045cd1058e795e8d3595485`
- expected data manifest SHA-256: `d098f9172083b8de9f825b66de5277dde5b6ea0581b3a950b8f76e4f443546cc`
- Cell projection dimension: 64
- formal seeds: `80821`, `80822`, `80823`

The old Kaggle `frozen-hidden.pt` cache is **not required**. The runner deterministically re-selects the pinned source data, verifies the manifest, and regenerates hidden states when the cache is absent.

## Run

On a fresh Kaggle GPU session with Internet enabled and a `GITHUB_TOKEN` secret, run the one-cell notebook:

`research/notebooks/04-continual-learning-core/core-008-certified-functional-atoms.ipynb`

Equivalent command from a prepared checkout:

```bash
python scripts/research/orchestrate_core_validation_008.py \
  --branch codex/core-validation-008-certified-functional-atoms \
  --secret-name GITHUB_TOKEN \
  --device cuda \
  --push-results
```

The orchestrator runs the three frozen formal seeds in separate Python processes, reuses the local hidden cache across seeds when present, reports the frozen gates, copies canonical artifacts without the hidden cache, commits, and optionally pushes them back to the same branch.

## Decision rule

All three formal seeds must pass independently. No majority rescue and no post-run threshold changes are allowed.

A positive result supports sparse compositional certified functional atoms only in the present frozen-Pythia, projected-linear-write setting. A rank-1 failure with adaptive-rank success rejects rank-1 as the persistent atom boundary without rejecting rank-1 as an elementary write primitive. Failure of all variants is evidence against the current write-demand/certificate geometry and should block further architectural elaboration.

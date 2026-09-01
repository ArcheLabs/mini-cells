# Core Validation 007 — Functional Boundary Discovery

Status: **DISCOVERY_PROTOCOL_FROZEN_UNRUN**

## Decision question

Core 006 established two positive facts on real frozen Pythia representations:

1. bounded subspace certificates reduced registered forgetting without replay;
2. real hidden states showed substantial functional reuse rather than immediate full-rank saturation.

It failed because semantic/address-based mitosis did not create a useful functional separation: split-conflict reduction was effectively zero on two formal seeds and growth approached one private Cell per address.

Core 007 therefore asks:

> **What is the correct bounded functional boundary for mitosis, and can that functional identity be recovered from inference-visible representations after the split?**

The experiment does **not** retune or reopen Core 006. Its frozen semantic singleton mechanism is the control.

## Fixed real foundation

Core 007 reuses Core 006 infrastructure and pins exactly the same real foundation/data family:

- `EleutherAI/pythia-160m@step143000`, permanently frozen;
- `DKYoon/SlimPajama-6B@1224c66add28b96ab045cd1058e795e8d3595485`;
- seven SlimPajama sources;
- 128-token sequences;
- 64-D projected Cell space;
- 32 frozen global K-means addresses over eight base Cells;
- real next-token NLL;
- the exact Core 006 `unsafe`, `certificate_no_growth`, semantic-mitosis and replay implementations as confirmation baselines.

## Read geometry vs write geometry

For frozen hidden state `h`, Core 006 already defines

\[
z=U^Th\in\mathbb R^{64}.
\]

Core 007 additionally defines the foundation-path projected loss signal

\[
u_t=U^T\frac{\partial L}{\partial h_t}
\]

and sequence-level write demand

\[
G=\frac1T\sum_t u_t z_t^T.
\]

`z` describes which representation directions a sequence uses. `G` describes which writable Cell directions the next-token objective asks to change. Core 002 and Core 006 motivate treating these as distinct geometries.

## Bounded functional modes

Each global semantic address may contain at most four online functional modes. A mode retains only bounded state:

- dependency token/sequence counts;
- a `z` prototype;
- `z` second moment `Sigma`;
- running mean write matrix `D`;
- at most rank-8 orthonormal basis of `vec(G)` write directions.

No raw historical sequence is part of candidate learner state.

A new training sequence is assigned to the closest existing mode in write-demand cosine geometry, or creates a new mode if similarity falls below the frozen threshold and the address still has capacity.

## Four boundary candidates

### 1. `semantic_singleton`

The Core 006 control. Isolate the triggering semantic address/mode from its current Cell.

### 2. `activation_community`

Construct pairwise principal-subspace overlap from mode `z` covariances and cut high-overlap dependencies apart.

### 3. `write_community`

Construct pairwise principal-subspace overlap from the bounded `vec(G)` write bases.

### 4. `interference_cut`

Directly estimate cross-write damage:

\[
I_{i\rightarrow j}
=\frac{\operatorname{tr}(D_i\Sigma_jD_i^T)}
{\operatorname{tr}(\Sigma_j)\lVert D_i\rVert_F^2},
\]

symmetrize it, then use a deterministic balanced max-cut heuristic.

The scientific intent is to test whether the correct Cell boundary is a **functional interference boundary**, rather than a semantic routing boundary.

## Two routing levels

The global Core 006 K-means address router remains frozen.

Inside an address, Core 007 evaluates two identities:

- **oracle functional router** — mode identity from frozen foundation write demand `G`; this requires labels/gradients and is only an upper bound;
- **deployable z-only router** — nearest bounded mode prototype using inference-visible `z`.

A soft Top-2 z-router is reported as an evaluation diagnostic, but is not allowed to choose the scientific winner.

This separation makes the result interpretable:

- oracle succeeds + z-router fails → split boundary exists, inference routing remains unsolved;
- oracle itself fails → functional boundary definition is wrong;
- both succeed but growth remains near-linear → binary mitosis is structurally insufficient.

## Phase A — discovery

Frozen discovery seeds:

```text
80701
80702
```

Discovery builds bounded functional modes on real representations and compares all four boundary candidates using the frozen score:

\[
0.55\,\text{interference-cut fraction}
+0.30\,\text{z-routing agreement}
+0.15\,\text{partition balance}.
\]

Discovery emits only:

```text
FUNCTIONAL_BOUNDARY_DISCOVERY_COMPLETED
```

It is **not** a scientific supported/not-supported decision.

After discovery, `scripts/research/select_core_validation_007.py` deterministically writes:

```text
research/validations/core-007-functional-boundary-discovery/winner-lock.json
```

The winner lock contains the protocol hash, data-manifest identity, discovery seeds, frozen score and one selected boundary mechanism.

## Phase B — untouched confirmation

Confirmation seeds:

```text
80711
80712
80713
```

The confirmation runner refuses to start unless the winner lock already exists and matches the frozen protocol hash.

For every confirmation seed it:

1. runs the exact Core 006 baseline implementation on the same data/seed;
2. trains only the locked functional-boundary candidate without replay;
3. measures actual split conflict reduction;
4. measures growth and child reuse;
5. compares new-learning gain against replay;
6. compares registered regression against unsafe;
7. evaluates oracle functional routing vs deployable z-only routing on heldout data;
8. reports soft Top-2 routing as a diagnostic;
9. performs final Cell ablations.

Every confirmation seed must independently satisfy all frozen gates:

- registered regression <= `0.50x` unsafe;
- new-learning gain >= `0.80x` replay;
- functional-mitosis gain > no-growth certificate baseline;
- median split-conflict reduction >= `0.30` and greater than the same-seed Core 006 semantic split;
- spawned Cells <= `0.50 * 32 addresses`;
- at least four later transactions reuse children;
- oracle/deploy z-mode agreement >= `0.70`;
- deploy heldout NLL is within `2%` of oracle functional routing;
- at least one non-zero heldout causal-ablation signal.

Only two scientific statuses are allowed:

```text
FUNCTIONAL_BOUNDARY_MECHANISM_SUPPORTED
FUNCTIONAL_BOUNDARY_MECHANISM_NOT_SUPPORTED
```

## Result publication and Kaggle recovery

Core 007 fixes the result-loss failure mode encountered after Core 006.

After either phase:

```bash
python scripts/research/report_core_validation_007.py --phase discovery
python scripts/research/publish_core_validation_007.py --phase discovery --commit-results --push-results
```

or:

```bash
python scripts/research/report_core_validation_007.py --phase confirmation
python scripts/research/publish_core_validation_007.py --phase confirmation --commit-results --push-results
```

`--push-results` first creates the canonical artifact commit. It then:

1. tries existing Git credentials;
2. if needed, tries `GH_TOKEN` or `GITHUB_TOKEN` without writing the token to repository config;
3. if push still fails, raises an explicit error while keeping the local commit and printing the manual push command.

The hidden-state cache is never copied into canonical artifacts.

## Interpretation boundary

A positive Core 007 result would establish only that, under a frozen real Pythia representation and frozen coarse router, a bounded functional-mode geometry can generate a better mitosis boundary and that this identity is sufficiently predictable from inference-visible `z`.

It would not establish safe nonlinear foundation updates, learned global router drift, reconstruction of certificates for opaque historical checkpoints, or full-scale autonomous CLM growth.

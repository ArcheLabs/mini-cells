# Constructive CLM-005 — Scaffold Removal / Endogenous Control

Status: **PROTOCOL FROZEN — FORMAL SEEDS UNRUN**

## Mission

CLM-001 through CLM-004 now provide formal controlled evidence for:

```text
learned/addressable Cell coordinates
+
latent discovery under registered superposition
+
structure-tracking growth
+
replay-free protected writes
+
sparse model-level multi-Cell computation
```

CLM-005 is the final registered Constructive CLM gate before training **Small Native CLM v0**.

Its distinct question is:

> Can learned routing plus learned write/grow control replace the explicit Constructive CLM control scaffolds while preserving the previously supported protection, bounded growth, reuse, sparse compute and unseen multi-Cell composition invariants?

This experiment does **not** re-prove G1–G4.

## What “endogenous” means in CLM-005

CLM-005 removes the **control-plane rules**, not every mathematical safety primitive.

Required learned controls:

```text
fixed route selection
  -> shared learned pairwise router

hand threshold write decision
  -> learned write commit/reject controller

hand novelty/growth threshold
  -> learned growth/reuse controller
```

Retained fixed safety/substrate primitives:

```text
persistent Cell state / route keys
Core-005 Cell-local certificate basis Q
certificate-constrained update solver
linear residual Cell operator family
```

Therefore a positive result is a controlled **learned control-plane transition**. It is not a claim that the safety geometry itself has become neural, nor that an LLM-scale Native CLM is already trained.

## Learned router

Routing uses one shared pairwise compatibility network:

```text
query + candidate Cell route key
  -> compatibility statistics
  -> 5 -> 12 -> 1 MLP
  -> score
```

The scorer is reused over a dynamic Cell set, including newly spawned children.

Its meta-training target is not a hidden Cell ID. For each permanently development-only meta transaction, all candidate Cells are evaluated on the **current residual fit**, and the lowest-error candidate supplies the intrinsic utility target.

At formal inference time the router receives only learner-visible route context plus persistent Cell keys.

## Learned write controller

For a routed Cell, CLM-005 computes the already-supported certificate-constrained candidate update and exposes current-data diagnostics to a learned binary controller:

```text
constrained fit error
unconstrained fit error
free-rank ratio
certificate-rank ratio
residual energy
  -> learned commit / reject probability
```

The controller decides whether the current transaction is safe/useful to commit.

A positive result requires:

- all registered safe mutations to be committed;
- all registered conflicts to be rejected;
- learner replay accesses = 0;
- historical full-composition behavior remains protected.

The actual safe parameter delta still uses the frozen Core-005 constrained solver. CLM-005 tests learned **control**, not re-discovery of the certificate mathematics.

## Learned growth / reuse controller

When a write is rejected, a learned growth controller sees only current-data quantities:

```text
best existing Cell relative error
fresh Cell relative error
router margin
router top probability
Cell observation/load state
  -> reuse / spawn probability
```

Its development-only utility label compares the current fresh-Cell fit against the best existing fit plus a registered spawn cost.

No evaluator novelty flag is provided.

In the formal mutation stream:

```text
4 root conflicts
-> exactly 4 useful children
-> each child reused 3 additional times
-> no extra growth on those repeats
```

An always-spawn policy would add 16 Cells over the same conflict/reuse opportunities. This prevents “endogenous growth” from being satisfied by unconditional spawning.

## Parent G4 baseline

Every CLM-005 seed first runs the unchanged CLM-004 registered system on the same new seed.

The parent must still pass before CLM-005 can count as positive. This establishes that a failure after scaffold removal is attributable to the new learned-control integration rather than a broken parent world.

## Computational world

Registered scale:

```text
learned root Cells                    12
route dimension                       48
hidden dimension                      16
Cell operator spectral norm           0.40
operator acquisition batches / root   4
examples / acquisition batch         32
multi-Cell compositions in train       0

evaluation modes:
  simultaneous
  sequential

cases / mode                          64
active Cells / case                    2–4

mutation target roots                  4
history compositions / target          6
child reuse repeats                     3
expected learned children               4
expected final Cells                   16
always-spawn control additions         16
```

## Full learned-control evaluation

After the safe/conflict stream, evaluation includes both original roots and learned children.

Simultaneous:

\[
h' = h + \sum_{i\in R} W_i h
\]

Sequential:

\[
h_{t+1}=h_t+W_{i_t}h_t
\]

The sequential world remains deliberately non-commutative.

Required retained behavior includes:

- mean composition MSE <= `1e-4`;
- exact learned route-sequence accuracy >= `0.995`;
- sequential order effect >= `1e-3`;
- simultaneous permutation error <= `1e-12`;
- Cell-operator execution fraction <= `0.30` of dense execution.

## Protection and anti-degeneracy controls

### Unsafe reuse

For each conflict, the evaluator applies the same desired update to the routed existing Cell without its certificate.

Required:

```text
mean historical full-composition MSE >= 1e-4
```

This demonstrates that learned write rejection + growth is solving a real stability/plasticity conflict rather than an innocuous routing event.

### Always spawn

The registered stream exposes 16 growth opportunities if every conflict and repeat is treated as novel.

The learned system must create exactly 4 children and reuse them thereafter.

### Cell-local isolation

Accepted updates must not modify unrelated Cell parameters.

## Slow-plastic shared-substrate compatibility

CLM-005 also includes a registered shared-substrate probe.

Historical inputs from simultaneous/sequential multi-Cell paths are compressed into a shared certificate basis. A small shared residual update must:

```text
fit current data
+
preserve historical full-composition outputs
+
use zero learner replay
```

The same update without protection must measurably damage historical behavior.

This is a compatibility result for a future slow-plastic shared substrate; it is not a claim that foundation training is fully solved.

## Controller training discipline

Controllers are trained once from permanently development-only meta episodes.

```text
router train:   6501–6506
router validate:6507–6508

growth train:   6601–6606
growth validate:6607–6608

write train:    6701–6706
write validate: 6707–6708
```

Each controller must achieve >= `0.98` on its held-out meta-validation episodes.

Formal-seed data are forbidden from controller training.

Hidden factor/task/Cell IDs are not controller targets.

## Anti-cheating learner boundary

Learner code must not receive:

```text
hidden factor ID
hidden task/mode label as a controller target
correct Cell ID
oracle route support
oracle novelty flag
future stream information
historical raw-example replay
```

Evaluator identities and true operators may be used only for post-hoc scoring and control construction.

## Formal gates

Every formal seed must pass all registered gates in [`protocol.json`](protocol.json), covering:

1. CLM-004 parent validity;
2. controller held-out meta generalization;
3. formal-data-free controller training;
4. structural learned-root validity;
5. learned-router operator acquisition;
6. operator quality without retained raw examples;
7. pre-mutation multi-Cell composition;
8. learned safe-write commits;
9. learned conflict-write rejection;
10. learned growth on useful conflicts;
11. bounded growth vs always-spawn;
12. child routing/reuse without repeat spawning;
13. zero learner replay;
14. protected historical composition;
15. unsafe-reuse interference control;
16. Cell-local isolation;
17. final unseen multi-Cell composition;
18. sparse active compute;
19. slow shared-substrate plasticity;
20. slow shared-substrate retention.

## Seed discipline

Development-only CLM-005 seeds:

```text
601
602
603
```

Permanently excluded after a partial pre-freeze shared-substrate diagnostic:

```text
90711
90712
90713
```

Those numbers are **not** formal evidence.

Frozen untouched formal seeds:

```text
90811
90812
90813
```

The ordinary `--seed` path rejects both the formal and excluded diagnostic seeds. CI never executes the formal set.

## Commands

Smoke:

```bash
python scripts/research/run_constructive_clm_005.py --smoke
```

Development:

```bash
python scripts/research/run_constructive_clm_005.py --seed 601
```

Formal, only after development/CI review:

```bash
python scripts/research/run_constructive_clm_005.py --formal
```

Positive decision:

```text
LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED
```

Negative decision:

```text
LEARNED_CONTROL_PLANE_TRANSITION_NOT_SUPPORTED
```

## Kaggle formal + automatic publication

Canonical notebook:

[`../../notebooks/constructive-clm-005-endogenous-control-kaggle.ipynb`](../../notebooks/constructive-clm-005-endogenous-control-kaggle.ipynb)

Run it top-to-bottom in Kaggle with the `GITHUB_TOKEN` secret configured.

It will:

1. fresh-clone the frozen CLM-005 branch;
2. refuse to rerun if a canonical CLM-005 decision is already tracked;
3. run the three formal seeds exactly once;
4. verify status, seed set and protocol SHA-256;
5. display every per-seed gate;
6. publish the exact positive **or negative** canonical artifacts back to the branch.

The publisher stages only:

```text
artifacts/experiments/constructive-clm-005-scaffold-removal/
```

Canonical outputs:

```text
decision.json
gate-summary.csv
controller-summary.csv
stage-summary.csv
RESULTS.md
```

## Interpretation boundary

A positive CLM-005 result would support the following controlled statement:

> A shared learned router plus learned write/grow controllers can govern a persistent protected Cell substrate on unseen seeds, preserving bounded reusable growth, replay-free protected writes, sparse simultaneous/sequential multi-Cell computation and a safe slow-plastic shared-substrate compatibility invariant.

This would close the registered **Constructive CLM mechanism-validation sequence** and justify moving to **Small Native CLM v0**.

It would not by itself establish general natural-language continual learning, arbitrary nonlinear Transformer Cells, fully learned safety geometry, an asymptotic growth theorem, LLM-scale endogenous CLM, or JAM deployment.

If positive, do **not** create a cosmetic CLM-005B that only increases synthetic Cell count or horizon. The next main experiment is the first Small Native CLM v0 training run.

# Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law

Status: **PROTOCOL FROZEN — FORMAL SEEDS UNRUN**

## Decision question

> Across a long structured continual stream whose reusable latent vocabulary itself grows sublinearly, does learned Cell state track latent structure rather than transaction count while spawn probability declines and retention/composition remain usable?

This is **G2** in the CLM Feasibility Evidence Map.

Constructive CLM-001 and 001B already establish the controlled coordinate-formation bridge. 002 therefore does not ask whether a Cell can exist or whether the registered pair-superposition family can be decomposed again. The new variable is **long-horizon streaming growth**.

## Registered hypothesis

Let:

- `N` = number of continual transactions processed;
- `M(N)` = true reusable latent vocabulary exposed by the world;
- `K(N)` = committed learned Cells.

The registered world schedules latent factor introduction approximately as:

\[
M(N)-M_0 \propto (N-N_0)^{0.60}
\]

on a finite 4096-transaction horizon.

The desired behavior is not merely “few Cells”. It is:

\[
K(N) \approx M(N) \ll N
\]

with:

\[
K(N)/N \downarrow,
\qquad
P(\mathrm{spawn}\mid t) \downarrow,
\qquad
P(\mathrm{reuse}\mid t) \uparrow.
\]

A positive result is **finite-horizon scaling evidence**, not an asymptotic theorem.

## World

- horizon: `4096` transactions;
- checkpoints: `256, 512, 1024, 2048, 4096`;
- initial latent factors: `6`;
- final latent factors: `30`;
- context/effect dimensions: `48 / 40`;
- factors remain correlated/non-orthogonal;
- the initial six Cells are recovered by reusing the 001B relational mechanism;
- the remaining 24 factors appear progressively according to the registered power-law schedule;
- ordinary transactions reuse 2–3 already-active factors;
- each new factor is introduced only through four unequal-weight pair mixtures with four different known anchors.

The learner never receives the factor ID, novelty flag, pair support label, true factor count, or introduction schedule.

## Streaming growth mechanism

There is deliberately **no `max_cells` cap**.

For ordinary reuse, current Cell keys explain `x` through sparse support inference.

If existing Cells cannot explain the current context:

```text
large x-reconstruction residual
    -> generate several one-missing-factor proposals
       against possible known anchors
    -> probationary residual clusters
    -> require agreement through >=2 distinct anchors
    -> commit one new Cell
```

The registered pair geometry uses the fact that the key atoms are unit vectors. For a possible known anchor `k` and current pair mixture `x`, the candidate anchor coefficient can be solved from the unit-norm constraint without a hidden novelty label. Wrong-anchor proposals are expected to disagree across different anchor contexts; the true latent residual is expected to recur.

This is still an **engineered growth controller**. 002 does not claim a learned/endogenous growth policy.

## Why this is not a hard-cap artifact

A learner could fake a declining `K/N` simply by refusing to grow. The protocol therefore requires all of the following simultaneously:

1. no hard Cell cap exists;
2. `K(N)` tracks the true latent vocabulary `M(N)` within one Cell at every checkpoint;
3. final `K(4096)=M(4096)=30`;
4. a new Cell must still be committed after 90% of the total horizon;
5. heldout composition and old-factor retention must remain accurate.

So “growth stopped” is not itself a positive result.

## Registered gates

Every formal seed must satisfy:

- valid reused 001B bootstrap with six initial Cells;
- no hard Cell cap;
- max checkpoint `|K-M| <= 1`;
- final `K=M=30`;
- fitted finite-horizon Cell growth exponent in `[0.40, 0.80]`;
- Cell exponent within `0.05` of the oracle latent-vocabulary exponent;
- Cell exponent at most `0.85` versus transaction-memory exponent `1.0`;
- final `K/N <= 0.01` and <=30% of its N=256 value;
- windowed spawn rate declines up to 15% tolerance;
- final-half spawn rate <= `0.006`;
- final-half reuse rate >= `0.985`;
- last Cell spawn after 90% of horizon;
- final mean matched key/effect cosine >= `0.985 / 0.995`, no duplicate factor assignments;
- final pair/triple MSE <= `0.0005`, route recall >= `0.98`;
- first-six-factor retention pair/triple MSE <= `0.0005`, recall >= `0.98`;
- final transaction-to-Cell compression >= `100x`.

## Seed discipline

Full development runs already observed and permanently excluded:

```text
301
302
303
```

During final implementation review, bootstrap-only diagnostics were also inspected on:

```text
90311
90312
90313
```

Those seeds are therefore **also permanently excluded** even though the complete CLM-002 stream was never run on them.

Untouched formal seeds were frozen only after all of the above observations:

```text
90411
90412
90413
```

Do not run those seeds through `--seed`. The runner rejects that path.

## Development run

CPU is sufficient:

```bash
python scripts/research/run_constructive_clm_002.py --seed 301
```

Expected runner status for any development run:

```text
DEVELOPMENT_RUN
scientific_decision = false
```

## Formal run

Only after the branch/protocol is reviewed:

```bash
python scripts/research/run_constructive_clm_002.py --formal
```

Formal success status:

```text
LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED
```

Artifacts are written to:

```text
artifacts/experiments/constructive-clm-002-long-horizon-growth-law/
  decision.json
  gate-summary.csv
  growth-curves.csv
  RESULTS.md
```

## Interpretation boundary

A positive 002 result would support:

> On the registered structured 4096-transaction horizon, Cell state can follow reusable latent structure rather than transaction count, with declining state/spawn ratios and preserved controlled routing/composition.

It would **not** establish:

- asymptotic `K(N)=o(N)`;
- arbitrary novelty or mixing processes;
- a learned growth policy;
- replay-free certificate integration;
- language-model-scale continual learning;
- foundation plasticity;
- JAM execution.

If positive, the next main experiment is **Constructive CLM-003 — learned coordinates + existing replay-free protection**. 002 should not turn into an indefinite sequence of cosmetically longer synthetic streams.

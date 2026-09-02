# Constructive CLM-004 — Model-Level Multi-Cell Computation

Status: **PROTOCOL FROZEN — FORMAL SEEDS UNRUN**

## Question

Constructive CLM-001/001B established learned/addressable Cell coordinates. CLM-002 established finite-horizon structure-tracking growth. CLM-003 established replay-free protected writes inside learned/growing Cell lineages.

CLM-004 asks the next distinct question:

> Can multiple learned Cells become reusable **computational modules** that act on hidden state inside the same model execution, generalize to unseen compositions, preserve order when order matters, execute sparsely, and retain the G3 protected-write invariant at full-composition output?

The new variable is **model-level multi-Cell computation**. This experiment does not re-prove Cell formation, growth, or the Core-005 certificate principle.

## From effect atom to computational module

CLM-001B demonstrated controlled algebraic composition of recovered latent effects. CLM-004 promotes Cell state to an operator:

\[
g_i(h)=W_i h
\]

and evaluates two execution semantics.

### Simultaneous

\[
h' = h + \sum_{i\in R} W_i h
\]

### Sequential

\[
h_{t+1}=h_t + W_{i_t}h_t
\]

The sequential world is deliberately non-commutative, so reversing route order changes the target output. A positive result cannot be explained by treating the active set as an unordered bag.

## Reused structural bridge

Each seed first reuses the CLM-002/003 structural mechanism to obtain 12 learned root coordinates.

Required bridge diagnostics:

```text
learned root Cells = 12
covered factors = 12
duplicate assignments = 0
mean matched root-key cosine >= 0.985
```

Hidden-factor alignment is evaluator-only. G4 does not tune or re-decide G2/G3.

## Computational Cell acquisition

Each learned root owns one computational Cell. The learner receives only:

```text
noisy route context
current hidden states h
current target residual r
```

No hidden Cell ID is passed to the learner.

Each Cell accumulates bounded sufficient statistics:

```text
X^T X
Y^T X
```

and learns a residual operator `W_i`. Raw acquisition examples are not retained.

Critically:

```text
multi-Cell composition examples seen during acquisition = 0
```

All model-level pair/triple/quad compositions are held out until evaluation.

## Registered scale

```text
learned computational Cells     12
hidden dimension                16
operator spectral norm          0.40
acquisition batches / Cell       4
examples / acquisition batch    32
composition examples in train    0

evaluation modes:
  simultaneous
  sequential

cases / mode                    64
examples / case                 24
active Cells / case              2–4
```

## Sparse execution

The sparse route executes only the selected Cell operators. The dense control executes all 12 Cell operators.

The registered compute metric is **Cell-operator execution count**, not router-search complexity. At the registered active range, the main system must execute at most 30% of dense Cell-operator work on average.

A positive result may not be obtained by activating every Cell.

## Controls

### `single_cell`

Execute only the first routed Cell.

Purpose: demonstrate that multi-Cell target behavior cannot be recovered by a single module.

### `wrong_semantics`

For sequential targets, execute the same active Cells simultaneously. For simultaneous targets, execute them sequentially.

Purpose: demonstrate that the model is using the registered computation semantics rather than merely the correct support set.

### `dense_all_cells`

Execute every Cell regardless of route.

Purpose: prevent a dense-all-Cells solution from being counted as sparse composition.

### `unsafe_mutation`

After composition evaluation, fit the same one-Cell mutation without the Cell-local certificate.

Purpose: show that protected composition retention is not a trivial consequence of parameter isolation alone.

## Protected mutation inside a composition path

CLM-004 includes one integration probe from G3.

A target Cell is observed inside six sequential compositions. Its incoming hidden states are compressed into the Core-005 row-span certificate `Q`. Raw historical hidden states are then discarded from learner state.

A new underdetermined mutation is presented. The protected learner uses `constrained_update(...)` with only current mutation data plus Cell-local `Q`.

The evaluator checks the **full composition output**, not merely the local Cell output.

Required behavior:

```text
safe mutation:
  fit new residual
  historical composition output unchanged
  learner replay accesses = 0
  unrelated Cell parameters unchanged
  route keys unchanged

unsafe mutation:
  fit new residual
  measurably damage historical composition behavior
```

This does not re-prove Core-005. It tests whether the invariant survives when the protected Cell is embedded in a multi-Cell computation path.

## Formal gates

Every formal seed must pass all gates.

### 1. Structural bridge

- exactly 12 learned roots;
- all 12 evaluator factors covered;
- no duplicate factor assignment;
- mean matched root-key cosine >= `0.985`.

### 2. Operator acquisition

- route accuracy >= `0.99`;
- mean relative operator error <= `0.01`;
- max relative operator error <= `0.02`;
- learned Cell operators remain distinct;
- raw acquisition examples retained = `0`.

### 3. Unseen model-level composition

Both simultaneous and sequential modes must satisfy:

- mean MSE <= `1e-4`;
- exact route sequence accuracy >= `0.995`;
- main MSE <= `0.05x` single-Cell control;
- main MSE <= `0.05x` wrong-semantics control;
- main MSE <= `0.05x` dense-all-Cells control.

All multi-Cell compositions are unseen during operator acquisition.

### 4. Non-trivial semantics

- sequential true-order effect MSE >= `1e-3`;
- simultaneous permutation MSE <= `1e-12`.

This makes sequential order sensitivity explicit while preserving simultaneous permutation invariance.

### 5. Sparse active compute

- active Cell-operator execution fraction <= `0.30` of dense execution;
- at most 4 Cells active in any registered case.

### 6. Protected mutation under composition

- safe historical full-composition MSE <= `1e-10`;
- safe `delta W Q` change <= `1e-10`;
- safe mutation fit error <= `1e-10`;
- learner replay accesses = `0`;
- learner raw history retained = `0`;
- unsafe historical composition MSE >= `1e-4`;
- unsafe change on protected subspace >= `1e-3`;
- unrelated Cell parameter drift <= `1e-15`;
- route-key drift <= `1e-15`.

See [`protocol.json`](protocol.json) for canonical thresholds.

## Seed discipline

Development-only:

```text
501
502
503
```

Frozen untouched formal seeds:

```text
90611
90612
90613
```

Formal seeds are rejected through ordinary `--seed` invocation. CI must not execute them.

## Smoke

```bash
python scripts/research/run_constructive_clm_004.py --smoke
```

This checks operator/composition/protection mechanics with synthetic route keys and does not run the full Constructive bridge.

## Development run

```bash
python scripts/research/run_constructive_clm_004.py --seed 501
```

Expected top-level status:

```text
DEVELOPMENT_RUN
scientific_decision = false
```

## Formal run

Only after smoke/development review:

```bash
python scripts/research/run_constructive_clm_004.py --formal
```

Formal positive status:

```text
MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED
```

Canonical artifacts:

```text
artifacts/experiments/constructive-clm-004-model-level-multicell-computation/
  decision.json
  gate-summary.csv
  composition-summary.csv
  mutation-summary.csv
  RESULTS.md
```

## Interpretation boundary

A positive CLM-004 result would support the registered controlled claim:

> learned route-addressed Cell operators can act as reusable hidden-state computational modules, compose simultaneously and sequentially on unseen combinations, execute sparsely at the Cell-operator level, and preserve a replay-free protected mutation invariant through the full composition path.

It would **not** establish arbitrary nonlinear Transformer Cell operators, natural-language generation, router emergence inside G4, a learned growth controller, a learned write controller, router lookup cost proportional only to active Cells, a fully endogenous Native CLM, or JAM execution.

If positive, the next main experiment is **Constructive CLM-005 — Scaffold Removal / Endogenous Transition**.

Do not create a cosmetic CLM-004B that only increases synthetic composition count or Cell count unless formal CLM-004 identifies a specific composition failure.

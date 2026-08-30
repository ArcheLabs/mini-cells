# Experiment 019 — Proposal Utility Discovery

## Purpose

Experiments 017–018b established that a new capability can be localized into a one-cell phenotype, that the cell can be transplanted within the same genome/base checkpoint, and that keeping the cell closed protects retained language behavior. They did **not** establish an endogenous rule that knows when a particular tissue is useful. State novelty in 018 lost selectivity through sensor/actuator feedback; feedback-isolated reaction+diffusion pressure in 018b protected language but did not distinguish skill from ordinary language.

Experiment 019 stops tuning recruitment thresholds and instead defines the quantity recruitment should estimate.

For a candidate tissue with continuous recruitment strength `e`, the target quantity is its marginal validation-loss utility at the sleeping state:

```text
U*_grad = - dL/de | e=0
```

An independent finite-difference oracle is also measured:

```text
U*_FD = [L(0) - L(epsilon)] / epsilon
```

with `epsilon = 0.02` in the full experiment.

The experiment asks:

> Can label-free local or language-boundary observables predict the true marginal value of a sleeping capability-tissue proposal across held-out skill families?

019 is a signal-discovery experiment. It does not train a production recruitment gate, allow autonomous growth during utility measurement, or claim cheap-probe compute savings.

## Why this follows the earlier experiments

- 006/007: recurrent language dynamics become competitive when persistence is the default and state change is carry-biased.
- 008: the proposal actually retained must be the proposal whose utility is evaluated; proxy objectives can be misleading.
- 010: update residual contains useful stopping information.
- 012: a shrinking residual does not imply a correct attractor, so settling cannot be the sole need signal.
- 014: random-depth + stability training makes variable-depth dynamics reproducibly robust.
- 015–015c: sparse activity can specialize, while aggressive dynamic routing/overcoupling damages quality and long-range coactivity topology can be epiphenomenal.
- 017: one-cell skill localization and same-checkpoint transplantation are possible for the development skill.
- 018b: sleeping the tissue causally protects language, but old-only scalar pressure does not select when to wake it.

019 therefore tests proposal-specific utility rather than another old-organism pressure heuristic.

## Skill-family suite

`REVERSE_INC` is retained as a development/control skill but is no longer the only benchmark. The locked suite contains six different synthetic computation families:

1. `REVERSE_INC` — symbol transform;
2. `MOD_ADD` — cumulative modular arithmetic;
3. `PARITY` — finite-state running parity;
4. `LOOKUP` — associative key/value retrieval;
5. `DELAY_COPY` — sequence memory/copy;
6. `LOCAL_RULE` — local context-dependent rewrite.

All use the same fixed-length autoregressive interface and output-token loss mask, but they implement different computation structures.

## Candidate tissue bank

For each replicate:

1. train exactly one Phase-1 Growing CLM using the Experiment-016 language recipe;
2. clone the identical Phase-1 checkpoint for every skill family;
3. allocate exactly one conservative newborn cell;
4. freeze the genome and all old phenotype gradients;
5. train only the newborn phenotype for 200 local steps;
6. allow connect/prune events touching newborn tissue, but forbid additional forks;
7. retain one additional untrained one-cell tissue as a random/control candidate.

The donor bank therefore contains six trained candidate tissues plus one untrained control. Candidate tissues are evaluated one at a time against the same Phase-1 organism; they do not coexist or compete during 019 utility measurement.

## Continuous recruitment semantics

019 reuses the causal interpolation introduced by conditional recruitment:

```text
e = 0  -> exact Phase-1 graph and metabolic competition
e = 1  -> exact fully-active adapted donor organism
```

The newborn's communication conductance and metabolic participation are both scaled continuously. Tests require both limiting equivalences.

For a batch, recruitment is represented by one differentiable scalar per example. Because examples do not interact across the batch, differentiating the summed per-example losses with respect to that vector yields an example-level `U*_grad` in one backward pass.

## Oracle gate

Before interpreting any signal discovery result, the gradient and finite-difference utility definitions must agree.

Pre-registered oracle requirements:

- overall Pearson(`U*_grad`, `U*_FD`) >= 0.95;
- overall utility-sign agreement >= 0.90.

If this fails, the experiment status is `UTILITY_ORACLE_INCONSISTENT` and downstream estimator results are not treated as evidence.

## Label-free observables

No estimator receives:

- target tokens;
- input skill-family ID;
- candidate tissue ID;
- a task label;
- an expert/router index.

### Local-only feature set

These use only the candidate parent/newborn phenotype and its epsilon proposal:

- parent state RMS;
- parent final-step settling RMS;
- epsilon proposal effect at the parent;
- proposal/parent-update alignment;
- child phenotype norm;
- child/parent phenotype cosine.

### Boundary-augmented feature set

The boundary model adds information available at the language interface:

- interface settling RMS;
- epsilon proposal effect at cell 0;
- proposal/interface-update alignment;
- output entropy;
- top-1/top-2 probability margin;
- epsilon logit-shift RMS;
- KL between closed and epsilon-probe predictions;
- entropy change under the probe;
- margin change under the probe.

The distinction is intentional. If boundary features succeed but local features fail, 019 has not found a genuinely local cellular error field.

## Estimators

The report compares transparent baselines and shared estimators:

- single-feature `proposal-norm` ridge;
- single-feature `probe-kl` ridge;
- `local-ridge`;
- `boundary-ridge`;
- a small shared `local-mlp`;
- a small shared `boundary-mlp`.

The MLPs are diagnostic estimators, not production routers. They are shared across every cell/tissue and receive no IDs.

## Strong leave-one-family-out evaluation

For held-out family `S_k`, estimator training excludes both:

```text
all examples whose input family is S_k
and
all candidate rows whose donor tissue family is S_k
```

The test set then consists of examples from `S_k` evaluated against the complete candidate bank, including the previously unseen `S_k` tissue. This prevents the estimator from learning a family-specific donor signature during training.

## Metrics

Per held-out family:

- Pearson and Spearman correlation with `U*`;
- AUC for `U* > 0`;
- utility-sign accuracy;
- top-1 candidate-tissue selection accuracy;
- normalized selection regret.

Random top-1 accuracy with seven candidates is approximately 1/7, so the top-1 criterion is deliberately demanding.

A held-out family passes when all are true:

```text
Spearman >= 0.50
AUC(U*>0) >= 0.75
top-1 tissue selection >= 0.60
median normalized regret <= 0.35
```

At least four of six families must pass.

The best local estimator must also beat the proposal-magnitude baseline in top-1 accuracy or regret on at least four held-out families.

## Donor-bank validity

A skill family is considered usable for the discovery bank when the median one-cell donor improvement across replicates exceeds 0.25 NLL. At least five of six families must meet this gate for a full positive result.

This is not a claim that every skill should fit in one cell; it is only a validity condition for this controlled proposal-selection experiment.

## Decision states

### `LOCAL_PROPOSAL_UTILITY_SIGNAL`

The oracle is consistent, the donor bank is valid, and a local-only estimator passes the held-out-family criteria. This is the evidence needed before attempting a shared local utility-gated recruitment rule.

### `BOUNDARY_ONLY_PROPOSAL_UTILITY_SIGNAL`

The oracle and donor bank pass, but only the boundary-augmented estimator generalizes. Proposal utility is observable at the language interface, but a local endogenous error field is still missing.

### `PARTIAL_PROPOSAL_UTILITY_SIGNAL`

At least two held-out families pass for one estimator, but the cross-family requirement is not met.

### `NO_GENERAL_PROPOSAL_UTILITY_SIGNAL`

The oracle is valid but the recorded label-free observables do not predict proposal utility robustly across families.

### `UTILITY_ORACLE_INCONSISTENT`

Gradient and finite-difference utility disagree. Do not interpret estimator results until the continuous recruitment/oracle implementation is repaired.

## Explicit non-claims

019 does **not** establish:

- a production recruitment mechanism;
- autonomous wake/connect/fork behavior;
- cheap proposal computation;
- arbitrary cross-genome tissue transplantation;
- that one cell is sufficient for general capabilities;
- general natural-language reasoning.

The epsilon probe currently executes the same shared cellular rule as a normal forward pass. If a utility signal exists, compressing it into a genuinely cheap proposal path is a later engineering/scientific problem.

## Outputs

The full run writes:

- `decision.json`
- `task-spec.json`
- `corpus-manifest.json`
- `tokenizer.json`
- `phase1-checkpoints.csv`
- `phase1-events.csv`
- `donor-summary.csv`
- `donor-events.csv`
- `utility-observations.csv`
- `oracle-consistency.csv`
- `utility-matrix.csv`
- `feature-correlations.csv`
- `estimator-results.csv`
- `oracle-gradient-vs-fd.png`
- `candidate-utility-matrix.png`
- `heldout-spearman.png`
- `heldout-auc.png`
- `heldout-top1.png`
- `heldout-regret.png`
- `feature-oracle-correlations.png`
- one worker JSON per replicate.

## Kaggle

Run:

```text
research/kaggle/experiment-019-proposal-utility-discovery.ipynb
```

The notebook runs invariant/regression tests before launching the three-replicate GPU experiment. Result publication is handled by:

```bash
python scripts/publish_experiment_019_results.py --push
```

Default result branch:

```text
kaggle/experiment-019-results
```

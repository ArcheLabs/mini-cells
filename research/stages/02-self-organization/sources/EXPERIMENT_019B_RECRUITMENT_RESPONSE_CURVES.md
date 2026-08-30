# Experiment 019b — Recruitment Response Curves

## Motivation

Stable Experiment 019 repaired the gradient-oracle numerics but failed the preregistered oracle-consistency gate:

- gradient-vs-finite-difference Pearson: 0.2413
- sign agreement: 0.8774
- status: `UTILITY_ORACLE_INCONSISTENT`

Several trained tissues also showed the stronger qualitative contradiction that full activation improves the learned skill while the derivative at the fully closed boundary can be negative. Experiment 019b therefore asks whether recruitment is intrinsically nonlinear and whether coherent capability tissue must cross a finite activation barrier before it becomes useful.

## Question

Can the same one-cell capability tissue be harmful at tiny recruitment yet beneficial when coherently recruited, and is there a fixed finite probationary recruitment that predicts the value of full recruitment across skill families?

## Locked design

019b performs **no training**. It requires the 24 local checkpoints produced by stable 019:

- 3 Phase-1 checkpoints
- 6 trained donor checkpoints per replicate
- 1 RANDOM control checkpoint per replicate

It reuses the exact stable-019 utility examples and scans:

```text
0,
1e-4,
3e-4,
1e-3,
3e-3,
1e-2,
2e-2,
5e-2,
1e-1,
2.5e-1,
5e-1,
1
```

For tissue T:

```text
V_T(e) = L(e=0) - L(e)
```

`e=0` must recover the exact same Phase-1 loss independently of which donor checkpoint is attached. This is a hard invariant.

## Activation barrier

A replicate-level mean matching-tissue curve is called full-beneficial when:

```text
V_T(1) > 0.05 NLL
```

It has an activation barrier when some `0 < e <= 0.02` is harmful by more than:

```text
max(0.005 NLL, 5% of V_T(1))
```

A skill family supports the barrier when at least 2/3 replicate mean curves satisfy it. A common activation barrier is recorded when at least 3/6 skill families support it.

These thresholds are fixed before the 019b run and are diagnostic rather than claims of a biological phase transition.

## Finite-probe test

For every fixed grid value `0 < e <= 0.1`, the probe value `V_T(e)` is compared with full value `V_T(1)` using:

- Spearman correlation
- AUC for full-beneficial tissue/example pairs
- top-1 tissue selection accuracy within each example
- median normalized regret under full recruitment

Per-family pass requires:

```text
Spearman >= 0.50
AUC >= 0.75
top-1 >= 0.60
median normalized regret <= 0.35
```

The same fixed probe amplitude is considered general when it passes at least 4/6 families.

## Decision statuses

- `ACTIVATION_BARRIER_WITH_FINITE_PROBE_SIGNAL`
- `ACTIVATION_BARRIER_WITHOUT_GENERAL_PROBE`
- `FINITE_PROBE_SIGNAL_WITHOUT_COMMON_BARRIER`
- `MIXED_RECRUITMENT_RESPONSE`

019b does not train a router or recruitment gate and does not claim compute savings. Even a successful finite probe still executes the current full cellular rule and must later be compressed into a cheap local probationary mechanism.

## Run

The local stable-019 checkpoints must still exist in:

```text
results/proposal-utility-discovery-stable-v1/checkpoints/
```

Then run:

```bash
python -m pytest tests/test_language_recruitment_response.py -q
python scripts/run_recruitment_response_curves.py
```

The three replicate sweeps use up to two GPUs concurrently. Completed replicate response files are resumable.

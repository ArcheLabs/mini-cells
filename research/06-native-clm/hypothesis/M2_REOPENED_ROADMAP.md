# Native CLM M2 Reopened Roadmap — Safe Functional Write Primitive

Status: **M2 REOPENED / M2-R0b FROZEN UNRUN**

## Why M2 is reopened

M1 established real next-token trainability. M2 was the first end-to-end replay-free continual-language milestone and remains formally unsupported:

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
formal seeds = 73211 / 73212 / 73213  (consumed)
protected A regression mean ≈ 0.438682
protected mean forgetting ≈ 0.211502
unsafe mean forgetting ≈ 0.278987
registered A-regression target <= 0.20
```

M3, M3R, the Address Diagnostic, M3L, M3L-1, M3L-2 and M3W-0 are therefore reclassified as an **M2 failure-decomposition series**. They remain useful mechanism evidence, but they do not represent progression past M2. The active question returns to the fixed-topology write primitive:

\[
\boxed{\text{Does every realized parameter transaction remain inside the old-function safe region?}}
\]

## M2-R0 — Protected Update Invariant Audit

Status: **PUBLISHED / `INCONCLUSIVE_REFERENCE_FAILURE`**

Published commit: `c1f9e132b026efe24a0238e6ea333bdd2ae5fbdb`.

Canonical M2 projects the Cell gradient before AdamW:

\[
G_p=G(I-Q^TQ).
\]

For plain SGD, `DeltaW = -eta G_p` should preserve `DeltaW Q^T = 0` in exact arithmetic. AdamW then applies elementwise moment preconditioning plus decoupled weight decay, so the implication is not automatic.

M2-R0 measured the realized post-optimizer delta directly:

\[
\rho=\frac{\|\Delta WQ^T\|_F}{\|\Delta W\|_F+10^{-12}}.
\]

The frozen five-arm result was:

| arm | mean rho | p95 rho | max rho |
|---|---:|---:|---:|
| canonical AdamW + gradient projection | 0.089760 | 0.155416 | 0.240160 |
| AdamW no decay + gradient projection | 0.089761 | 0.155417 | 0.240162 |
| SGD no decay + gradient projection | 0.004237 | 0.015315 | 0.090235 |
| SGD with decay + gradient projection | 0.324492 | 0.425478 | 0.434843 |
| AdamW + final realized-update projection | 1.90e-6 | 5.44e-6 | 1.53e-5 |

The mechanism pattern strongly suggests that both adaptive preconditioning and weight decay can move the realized parameter transaction outside the registered certificate nullspace, while constraining the complete realized update nearly restores the invariant.

However, the algebraic SGD/no-decay reference also failed the pre-registered relative threshold. Its smallest realized updates were around `1e-6` while absolute residuals were around `1e-7`, so R0 correctly refused a scientific/mechanics classification rather than treating a possible float32 transaction floor as structural damage.

M2-R0 therefore remains optimizer-mechanics evidence only. It does not alter the historical M2 decision.

## M2-R0b — Numerical Reference Audit

Status: **FROZEN / UNRUN**

Registration: `research/stages/06-native-clm/M2_R0B_REGISTRATION.md`.

R0b does not rerun continual language. It uses the same M1 checkpoint, the same pinned WikiText-B gradient source, audit seed `75001`, the same five optimizer arms, frozen certificates/router/topology, and zero learner replay.

For every certificate-ranked Cell/step it separates:

1. projected-gradient analytic transaction `-lr * g`;
2. fp64 safe projection in the exact span represented by the stored certificate rows;
3. parameter-dtype `fl(W + Delta_safe) - W` commit;
4. raw optimizer-realized update;
5. matched-safe fp64 projection of that optimizer update;
6. matched-safe parameter-dtype commit at the same update scale;
7. actually committed update after the arm's optional final-update projection.

The numerical floor is frozen before execution as the maximum of an empirical matched-safe float-commit residual and a conservative dtype bound:

\[
8\,\epsilon_{dtype}(\|W\|_F+\|\Delta W_{safe}\|_F).
\]

R0b then uses the excess factor:

\[
E=\frac{\|\Delta W_{commit}Q^T\|_F}{\text{machine-floor envelope}}.
\]

The two references remain `SGD no decay` and `AdamW final-update projection`. Both must have sufficient coverage, fp64 ideal-projection rho `<= 1e-10`, p95 excess `<= 2`, and max excess `<= 4`. Structural optimizer violations require both p95 excess `>= 16` and committed p95 rho `>= 1e-4`.

No threshold may be changed after observing the canonical R0b run.

Registered primary classifications:

```text
INCONCLUSIVE_SGD_NUMERICAL_REFERENCE_FAILURE
INCONCLUSIVE_FINAL_PROJECTION_NUMERICAL_REFERENCE_FAILURE
R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF
```

Only the last classification closes the numerical/mechanics gate and unblocks M2-R1. It is not a continual-learning success claim.

## M2-R1 — Functional Certificate Reconstruction

Status: **PLANNED / BLOCKED ON M2-R0b**

After R0b closes the update-transaction measurement layer, rebuild certificates in the fixed final-M1 representation rather than immediately rerunning continual learning.

Known weaknesses of the historical certificate are:

1. certificate updates only cover routed top-1 Cells although `active_cells=2`;
2. every update stores only one mean input vector, which is weak for high-variance/superposed history;
3. M1 certificates were accumulated while the shared representation was still moving, rather than in the final-M1 coordinate system.

Candidate families:

1. historical top-1 mean-vector basis;
2. all-active probability-weighted activation SVD/covariance sketch;
3. importance-weighted activation subspace;
4. functional Jacobian/Fisher/Gauss-Newton sketch.

For Cell output `y_i=W_i h`, a first-order old-logit perturbation is:

\[
\delta z\approx J_i(x)p_i\Delta W_i h.
\]

A functional damage metric is:

\[
D_i(\Delta W)=\mathbb E_{old}\|J_i p_i\Delta W_i h\|^2.
\]

R1 prioritizes persistent low-rank/Kronecker approximations such as:

\[
F_i\approx A_i\otimes B_i,
\]

\[
A_i=\mathbb E[p_i^2hh^T],\qquad B_i=\mathbb E[J_i^TJ_i],
\]

so that:

\[
D_i(\Delta W)\approx\operatorname{tr}(B_i\Delta W A_i\Delta W^T).
\]

R1 must compare candidates against held-out old-function drift with sketch/evaluation separation, all-active coverage, fixed final-M1 coordinates, explicit storage budgets and capacity curves. Raw old replay is not an allowed final mechanism.

## M2-R2 — Fixed-Topology Replay-Free Continual Language

Status: **BLOCKED ON R0b + R1**

R2 is the next true continual-learning formal experiment. It keeps the original M2 boundary:

```text
exact M1 start
8 Cells fixed
active Cells = 2
shared Transformer frozen
router and route keys frozen
growth disabled
B -> C -> D
learner raw replay = 0
```

Only mechanisms independently validated in R0b/R1 may change:

- realized-update constrained optimizer;
- selected functional certificate.

The intended write transaction is a functional trust-region problem:

\[
\min_{\Delta W}\; g_{new}^T\Delta W+\frac{1}{2\eta}\|\Delta W\|^2
\]

subject to:

\[
D_{old}(\Delta W)\le\epsilon.
\]

R2 uses untouched new formal seeds. Historical M2/M3/M3R/M3L-2 seeds remain consumed.

At minimum, the end-to-end gates retain:

```text
A absolute regression <= 20%
mean forgetting <= 15%
B/C/D phase gain >= registered plasticity floor
plasticity >= 80% matched control
shared/router frozen
zero learner replay
fixed 8-Cell topology
```

Only a formal R2 pass supports:

\[
\boxed{\text{Native CLM basic replay-free continual-write primitive supported.}}
\]

## When growth reopens

No new growth/mitosis formal experiment is allowed before R2 passes.

Future growth should follow from safe-write infeasibility. Define:

\[
\mathcal S_i(\epsilon)=\{\Delta W:D_i(\Delta W)\le\epsilon\}.
\]

Reuse an existing Cell when useful new-domain gain is achievable inside its safe set. Spawn only when no eligible Cell can absorb the write safely enough. Mitosis is therefore a consequence of **safe-write infeasibility**, not an independent heuristic milestone.

## Research-governance rule

```text
M1                                  PASS
  ↓
M2 original                         NOT SUPPORTED
  ↓
M2-R0 actual-update invariant       INCONCLUSIVE_REFERENCE_FAILURE
  ↓
M2-R0b numerical reference          FROZEN / UNRUN
  ↓
M2-R1 functional certificate        BLOCKED
  ↓
M2-R2 fixed-topology continual CL   BLOCKED
  ↓ only if PASS
growth / mitosis reopened           BLOCKED
```

Until R2 passes, M4 ontology work and 30M scaling remain blocked. Local mechanism improvements must report how much of the original M2 retention gap they close; they cannot substitute for the end-to-end retention/plasticity gate.

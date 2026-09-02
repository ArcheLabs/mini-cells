# Constructive CLM-005 — Scaffold Removal / Endogenous Transition

Status: **DESIGN BOUNDARY REGISTERED — IMPLEMENTATION NOT YET FROZEN**

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

CLM-005 is the final Constructive CLM gate before training **Small Native CLM v0**.

Its distinct question is:

> Can the working engineered Constructive CLM stack progressively replace explicit routing, growth and write scaffolds with learned/endogenous control while preserving the previously supported continual-learning and computational invariants?

CLM-005 must not re-prove G1–G4.

## Required scaffold-removal axes

The experiment must register staged variants that remove at least these engineered controls one at a time:

1. **Routing scaffold**
   - replace explicit/prototype-derived route support with a learned router from learner-visible context/state;
   - no hidden factor/task/mode IDs or correct Cell IDs.

2. **Growth scaffold**
   - replace fixed residual/probation thresholds and oracle-shaped novelty logic with a learned growth/reuse decision;
   - no evaluator novelty flag.

3. **Write scaffold**
   - replace hand-selected write/update control with a learned write controller while retaining a verifiable protection constraint or safety certificate;
   - no historical raw-example replay when the replay-free guarantee is claimed.

4. **Shared substrate scaffold**
   - test whether a slow-plastic shared substrate can coexist with persistent Cell state without destroying routing, protection or composition.

The exact formal decision may stage these removals, but the protocol must specify in advance which removals are required to cross the Native-CLM boundary.

## Invariants that must survive

Every accepted scaffold-removal stage must preserve the parent evidence rather than optimizing only a new local metric:

```text
coordinate/read-address quality
historical retention
new-learning gain
bounded/improving Cell growth
zero learner replay where claimed
route-support generalization
simultaneous composition
order-sensitive sequential composition
sparse active Cell compute
Cell-local mutation isolation
```

## Anti-cheating boundary

Learner code must not receive:

```text
hidden factor/task/mode labels
correct Cell IDs
oracle route support
oracle novelty flags
evaluator-only addresses
future stream information
```

A learned controller that simply memorizes evaluator IDs does not count as endogenousization.

## Formal protocol discipline

Implementation and development seeds may be tuned before protocol freeze. Formal seeds must be generated and frozen only after:

- scaffold-removal stages are explicitly defined;
- parent invariants and degradation tolerances are registered;
- negative controls are registered;
- implementation smoke/development runs pass without observing formal seeds.

Do not assign or run CLM-005 formal seeds until the protocol is frozen.

## Interpretation boundary

A positive CLM-005 result would still be a controlled constructive result. It would justify the transition from mechanism-validation research to training the first **Small Native CLM v0**; it would not by itself establish language-scale continual learning or production-scale Native CLM.

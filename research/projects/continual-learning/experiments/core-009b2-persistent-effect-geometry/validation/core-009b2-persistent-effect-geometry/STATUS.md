# Status — Core Validation 009B-2

- Experiment: `core-validation-009b2-persistent-effect-geometry`
- Protocol: `1.0-frozen`
- Current status: `IMPLEMENTATION_COMPLETE_GPU_DISCOVERY_PENDING`
- Scientific decision: `false`
- Parent: `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`
- Parent result commit: `f2691daf5738eac0232866a46d079db3aa61b60a`
- Discovery seeds: `81101, 81102`
- Confirmation seeds: `81111, 81112, 81113`
- Confirmation opened: `false`
- Basis lock: `not yet created`
- Positive status: `PERSISTENT_EFFECT_GEOMETRY_SUPPORTED`
- Negative status: `PERSISTENT_EFFECT_GEOMETRY_NOT_SUPPORTED`

## Boundary

009B-2 tests only compact reusable geometry of natural-magnitude carrier effect vectors `a_i = Ghat_i r`.

It does not test routing, sparse coefficients, certificates, continual mutation or an end-to-end CLM.

## Stop rule

If no <=32-dimensional compact discovery winner exists, do not open confirmation.

If formal confirmation is negative, do not run 009B-3 without freezing a new effect-representation hypothesis.

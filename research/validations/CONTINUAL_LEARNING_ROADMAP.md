# MiniCells Continual-Learning Research Roadmap

Status: **Constructive CLM is the Native-CLM main line**  
Frozen evidence map: [`CLM_FEASIBILITY_EVIDENCE_MAP.md`](CLM_FEASIBILITY_EVIDENCE_MAP.md)  
Current constructive experiment: **Constructive CLM-005 — Scaffold Removal / Endogenous Transition**  
Current foundation-interface diagnostic: **Core 009D — Compositional Operator Geometry**

## Mission

```text
pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / Native CLM
```

Foundation Interface Research asks what useful writable interface already exists in a mature pretrained LLM. Constructive CLM asks whether a persistent sparse coordinate system can be learned, protected, composed and progressively endogenousized even when the checkpoint does not expose a natural Cell ontology.

## Stable sequence

```text
G1a  CLM-001   learned coordinate formation                    🟢 SUPPORTED
G1b  CLM-001B  latent discovery under superposition            🟢 SUPPORTED
G2   CLM-002   long-horizon structure-tracking growth          🟢 SUPPORTED
G3   CLM-003   protected learned/growing Cells                  🟢 SUPPORTED
G4   CLM-004   model-level multi-Cell computation               🟢 SUPPORTED
G5   CLM-005   scaffold removal / endogenous transition         🔵 ACTIVE
                                                                ↓
                                                    Small Native CLM v0
```

This order remains stable unless a registered experiment identifies a specific missing dependency.

## G1a — Learned coordinate formation

`LEARNED_COORDINATE_FORMATION_SUPPORTED` on formal seeds `90111/90112/90113`.

Boundary: registered singleton-exposure world.

## G1b — Latent discovery under superposition

`LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED` on formal seeds `90211/90212/90213`.

Boundary: registered additive pair-superposition scaffold; not arbitrary blind source separation.

## G2 — Long-horizon structure-tracking growth

`LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED` on formal seeds `90411/90412/90413`.

Registered endpoint:

```text
N_final = 4096
M_final = 30 latent factors
K_final = 30 Cells
K/N = 0.00732421875
finite-horizon fitted exponent = 0.6631226816
late spawn rate = 0.00439453125
late reuse rate ≈ 0.991
```

Boundary: finite-horizon structure-tracking evidence, not an asymptotic theorem and not a learned growth controller.

## G3 — Protected learned/growing Cells

Status: **SUPPORTED**.

Formal decision:

```text
PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED
seeds = 90511 / 90512 / 90513
protocol = 6122b8a6dd62ac69bc371909fe503c24cea319837b0a5989a4b640492aeeda86
all 15 registered gates passed on all three seeds
missing seeds = none
```

Frozen integration:

```text
learned root routing
+
Cell-local protected W/Q state
+
certificate-triggered context-keyed lineage mitosis
-> protected continual writes without learner-side replay
```

Formal summary: [`constructive-clm-003-protected-growing-cells/FORMAL_RESULT.md`](constructive-clm-003-protected-growing-cells/FORMAL_RESULT.md).

## G4 — Model-level multi-Cell computation

Status: **SUPPORTED**.

Formal decision:

```text
MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED
seeds = 90611 / 90612 / 90613
protocol = 899c466747b5bec28b548fff2fc48173524b4fba7475f59085cb5f7accc75176
all 17 registered gates passed on all three seeds
missing seeds = none
```

Registered computational form:

```text
simultaneous:
  h' = h + sum_i W_i h

sequential:
  h <- h + W_i h
  in routed order
```

The registered formal result establishes, within the controlled linear-residual world:

- held-out simultaneous multi-Cell composition;
- order-sensitive sequential composition;
- exact registered route support/order recovery;
- sparse Cell-operator execution rather than dense-all-Cells execution;
- protected one-Cell mutation that preserves full historical composition output with zero learner replay;
- explicit unsafe-mutation interference control.

Formal summary: [`constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md`](constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md).

Boundary: controlled linear residual operators with engineered routing/execution scaffolds. Do not treat this as arbitrary nonlinear Transformer Cell computation or endogenous control.

## G5 — Scaffold removal / endogenous transition

Status: **ACTIVE**.

This is now the final Constructive CLM gate before Small Native CLM v0.

Parent chain:

```text
CLM-001 / 001B learned coordinates
+
CLM-002 bounded structure-tracking growth
+
CLM-003 replay-free protected writes
+
CLM-004 sparse model-level multi-Cell computation
```

New integration variable:

```text
working engineered Constructive CLM stack
+
progressive scaffold removal
-> learned / endogenous control
```

The removal sequence should be explicit and monotonic rather than replacing everything at once:

```text
engineered route support / prototype logic
  -> learned router

residual / probation novelty-growth rule
  -> learned growth controller

engineered protected write selection/update
  -> learned write controller subject to the retained safety invariant

frozen or externally separated substrate
  -> slow-plastic shared foundation
```

Each removal step must preserve the already-supported invariants:

- reusable Cell coordinates/read addresses;
- new-learning gain and historical retention;
- zero learner-side replay where the certificate guarantee is claimed;
- bounded/improving growth;
- route-support generalization;
- simultaneous/sequential composition;
- sparse active Cell compute;
- Cell-local mutation isolation.

A positive G5 result cannot rely on hidden factor/task/mode labels, explicit correct Cell IDs, oracle novelty flags, or evaluator-only addresses being passed to learner code.

If full end-to-end learned control is too large a jump, CLM-005 may register staged ablations, but the scientific decision must be frozen before formal seeds and must specify which scaffold removals are necessary to cross the Native-CLM boundary.

## Product track

External CLM feasibility remains separate from Native CLM feasibility. A product may keep engineered persistent Cells, routing, certificates, growth, versioning and rollback even if full endogenousization later fails.

## Experimental discipline

- Observed formal seeds are not reused as untouched confirmation seeds.
- Supported mechanisms are reused rather than re-proved unless a new integration variable is registered.
- Hidden factor/task/mode labels remain evaluator-only unless a protocol explicitly allows them.
- Natural-geometry negatives do not automatically stop Constructive CLM.
- Finite streams are not described as asymptotic proofs.
- Later CLM-003/004 reruns are reproduction/artifact recovery, not new untouched formal confirmations.
- CLM-005 must test scaffold removal; it must not regress into a cosmetic 004B composition benchmark.

## Immediate order

```text
1. CLM-001  = SUPPORTED
2. CLM-001B = SUPPORTED
3. CLM-002  = SUPPORTED
4. CLM-003  = SUPPORTED
5. CLM-004  = SUPPORTED
6. CLM-005  = ACTIVE
7. If CLM-005 is positive -> train Small Native CLM v0
```

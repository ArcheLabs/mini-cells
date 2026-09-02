# MiniCells Continual-Learning Research Roadmap

Status: **Constructive CLM is the Native-CLM main line**  
Frozen evidence map: [`CLM_FEASIBILITY_EVIDENCE_MAP.md`](CLM_FEASIBILITY_EVIDENCE_MAP.md)  
Current constructive experiment: **Constructive CLM-004 — Model-Level Multi-Cell Computation**  
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
G4   CLM-004   model-level multi-Cell computation               🔵 ACTIVE
G5   CLM-005   scaffold removal / endogenous transition         ⚪ PLANNED
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

Parent chain:

```text
CLM-001 / 001B learned coordinates
+
CLM-002 structure-tracking growth
+
Core-005 replay-free subspace certificate
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

The registered controls separately verify unsafe forgetting, the no-growth stability/plasticity limit, replay-oracle history use, growth rescue, route stability, child reuse and bounded functional growth.

Formal summary: [`constructive-clm-003-protected-growing-cells/FORMAL_RESULT.md`](constructive-clm-003-protected-growing-cells/FORMAL_RESULT.md).

Boundary: controlled learned-root + linear protected-write integration. Do not treat this as arbitrary Transformer write safety or model-level multi-Cell computation.

## G4 — Model-level multi-Cell computation

Status: **ACTIVE**.

This is now the main Constructive CLM experiment.

Candidate model-level form:

\[
h_{l+1}=h_l+\sum_{i\in R(h_l)} g_i(h_l;\theta_i).
\]

New integration variable:

```text
multiple active learned Cells
+
model-level computation / composition
```

Required evidence should include:

- multiple active learned Cells in the same execution path;
- unseen combinations/compositions;
- destructive cross-Cell interference measurement;
- route-support recovery;
- sequential and/or simultaneous composition;
- compute scaling with active Cell count rather than total Cell count;
- preservation of the G3 protected-write invariant under composition.

A positive result must not be obtained by activating all Cells or collapsing back to a single monolithic adapter.

## G5 — Scaffold removal / endogenous transition

Status: **PLANNED after G4**.

```text
prototype / relational discovery -> learned router
residual / probation growth      -> learned growth controller
engineered protected writes      -> learned write controller
frozen foundation                -> slow-plastic foundation
                                 -> endogenous cellular model
```

Each removal must preserve new learning, old retention, growth efficiency, routing generalization, composition and compute/storage efficiency.

## Product track

External CLM feasibility remains separate from Native CLM feasibility. A product may keep engineered persistent Cells, routing, certificates, growth, versioning and rollback even if full endogenousization later fails.

## Experimental discipline

- Observed formal seeds are not reused as untouched confirmation seeds.
- Supported mechanisms are reused rather than re-proved unless a new integration variable is registered.
- Hidden factor/task/mode labels remain evaluator-only unless a protocol explicitly allows them.
- Natural-geometry negatives do not automatically stop Constructive CLM.
- Finite streams are not described as asymptotic proofs.
- Coordinate formation requires deployable routing; routing requires protected writes; protected writes require model-level computation before Native CLM can be claimed.
- Later CLM-003 reruns are reproduction/artifact recovery, not a second untouched formal confirmation.

## Immediate order

```text
1. CLM-001  = SUPPORTED
2. CLM-001B = SUPPORTED
3. CLM-002  = SUPPORTED
4. CLM-003  = SUPPORTED
5. CLM-004  = ACTIVE
6. If CLM-004 is positive -> CLM-005
7. If CLM-004/005 are positive -> train Small Native CLM v0
```

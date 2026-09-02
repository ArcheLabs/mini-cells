# CLM Feasibility Evidence Map

Status: **FROZEN v1.5**  
Constructive baseline through formal CLM-004 result on frozen implementation commit `a6f3fb94a68172b8a89e68e742adae43c4662510` (2026-09-02).

## Purpose

This map prevents the MiniCells research program from repeatedly re-testing mechanisms that are already supported, already falsified, or already constrained by prior experiments.

The product path remains:

```text
pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / Native CLM
```

The native route no longer depends on discovering a ready-made Cell ontology inside a pretrained checkpoint. It asks whether already-supported mechanisms can be composed into a learned, protected, bounded-growth computational coordinate system and then progressively endogenousized.

## Evidence grades

- **A — formal:** frozen multi-seed or otherwise registered scientific decision.
- **B — controlled:** repeatable mechanism evidence, but not a complete continual-learning decision.
- **C — diagnostic:** useful bridge/geometry evidence; not sufficient as a standalone CLM claim.
- **Open:** not yet established by the repository.

## Canonical evidence matrix

| Native-CLM proposition | Repository evidence | Grade | Frozen interpretation | Re-test? |
|---|---|---:|---|---|
| Useful functional organization can emerge under differential pressure | Experiments 014–024 | B | Self-organization is plausible; historical emergence is not itself validated continual learning | No toy emergence rerun |
| A Cell can be a sparse routed, independently mutable computational state unit | 025/026; CLM-0.1–0.3 | B | Operational Cell granularity, sparse routing, independent mutation and capacity growth are viable mechanisms | No basic Cell-unit rerun |
| Rejection/saturation can be converted into useful growth | Core 004 | A | Growth can restore plasticity in the controlled CLM loop | No basic growth-rescue rerun |
| Bounded Cell-local state can replace learner-side replay for protected writes and mitosis | Core 005 | A | `Q` is sufficient for registered-history protection, saturation detection and reusable growth in the frozen linear-writable world | No certificate-principle rerun |
| Replay-free protection retains useful plasticity on real pretrained representations | Core 006 | A mixed | Real Pythia states retain useful plasticity under protection; semantic/address mitosis fails as a sufficient boundary | Reuse positive bridge; do not revive semantic address |
| Natural local/sparse write addressability already exists in the pretrained model | Core 002/002B/002C | A negative | Registered natural-address hypotheses failed | Do not repeat cosmetic variants |
| Normalized functional writes expose reusable factorized geometry | Core 009A | A/C | Useful two-sided geometry exists, but later diagnostics show right-side carrier collapse | Foundation-interface evidence only |
| The common right-side carrier is causally useful | Core 009B-1 | A | Carrier-preserved writes retain roughly 97.6–98.2% of registered full-write target gain | No carrier-causality rerun |
| Pretrained effect vectors already lie in a compact persistent dictionary | Core 009B-2 | C negative | No useful compact persistent global effect subspace under the frozen discovery protocol | Natural-geometry No-Go only |
| Pretrained effect vectors expose sparse/local Cell coordinates | Core 009C | C negative | `SPARSE_LOCAL_EFFECT_GEOMETRY_NOT_FOUND` | Do not keep searching for the same natural ontology |
| Reusable Cell coordinates/read keys can form with clean singleton exposure | Constructive CLM-001; seeds `90111/90112/90113` | A | `LEARNED_COORDINATE_FORMATION_SUPPORTED` | G1a frozen |
| Latent Cell coordinates can be recovered with no singleton exposure | Constructive CLM-001B; seeds `90211/90212/90213` | A | `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED` | G1b frozen within registered additive scaffold |
| Long-horizon Cell growth can track reusable latent structure rather than transaction count | Constructive CLM-002; seeds `90411/90412/90413` | A | `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`; final `K=M=30` at `N=4096`, fitted finite-horizon exponent `0.6631`, final `K/N=0.007324` | G2 frozen; no cosmetic longer synthetic streams |
| Learned/growing Cells can host replay-free protected continual writes with bounded functional mitosis | Constructive CLM-003; seeds `90511/90512/90513`; [`FORMAL_RESULT.md`](constructive-clm-003-protected-growing-cells/FORMAL_RESULT.md) | A | `PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED`; all 15 registered gates passed on all three seeds | G3 frozen; no cosmetic certificate rerun |
| Learned Cell operators can compose at model level with sparse active compute and protected mutation | Constructive CLM-004; seeds `90611/90612/90613`; [`FORMAL_RESULT.md`](constructive-clm-004-model-level-multicell-computation/FORMAL_RESULT.md) | A | `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`; all 17 registered gates passed on all three seeds | G4 frozen; no cosmetic composition rerun |

## Reusable constructive mechanism chain

```text
sparse routed mutable state
+
transactional / growth lifecycle
+
replay-free local protection
+
real-representation writable geometry
+
causal foundation interface
+
learned addressable coordinates
+
latent discovery under registered superposition
+
finite-horizon structure-tracking growth
+
protected learned/growing writes with context-keyed mitosis
+
sparse simultaneous/sequential model-level multi-Cell computation
```

The immediate missing bridge is now **progressive scaffold removal toward learned/endogenous control**.

## Frozen gaps

### G1a — Addressable learned coordinate formation

Status: **SUPPORTED** by Constructive CLM-001.

Boundary: controlled structured continual world with singleton exposure.

### G1b — Latent coordinate discovery under superposition

Status: **SUPPORTED** by Constructive CLM-001B.

Boundary: registered additive pair-superposition discovery scaffold; not arbitrary blind source separation.

### G2 — Long-horizon structure-tracking growth

Status: **SUPPORTED** by Constructive CLM-002.

Formal seeds: `90411 / 90412 / 90413`.

This is finite-horizon scaling evidence, not an asymptotic proof that `K(N)=o(N)`.

### G3 — Learned/growing Cells + replay-free protection

Status: **SUPPORTED** by Constructive CLM-003.

Formal decision:

```text
PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED
seeds = 90511 / 90512 / 90513
protocol = 6122b8a6dd62ac69bc371909fe503c24cea319837b0a5989a4b640492aeeda86
all 15 registered gates = pass on every seed
```

Boundary: controlled learned-root + linear protected-write integration. This is not arbitrary Transformer write safety.

### G4 — Model-level multi-Cell computation

Status: **SUPPORTED** by Constructive CLM-004.

Formal decision:

```text
MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED
seeds = 90611 / 90612 / 90613
protocol = 899c466747b5bec28b548fff2fc48173524b4fba7475f59085cb5f7accc75176
all 17 registered gates = pass on every seed
```

Frozen interpretation:

```text
learned route-addressed Cell operators
+
held-out simultaneous composition
+
order-sensitive sequential composition
+
sparse active Cell execution
+
replay-free protected mutation through full composition output
-> model-level multi-Cell computation under the registered linear residual world
```

Boundary: controlled linear residual operators and engineered routing/execution scaffold. This is not arbitrary nonlinear Transformer Cell computation and not endogenous control.

### G5 — External -> endogenous transition

Status: **ACTIVE**.

Primary experiment: **Constructive CLM-005 — Scaffold Removal / Endogenous Transition**.

New integration variable:

```text
working engineered Constructive CLM stack
+
progressive removal/replacement of router, growth and write scaffolds
-> learned/endogenous control
```

The experiment must remove scaffold components one at a time and preserve the already-supported invariants:

- coordinate/read-address formation;
- protected continual learning without learner replay;
- bounded/improving growth;
- unseen multi-Cell composition;
- sparse active compute;
- routing and mutation isolation.

A positive G5 result must not be obtained by keeping hidden evaluator labels, fixed oracle routes, explicit novelty flags, or hand-coded correct Cell addresses available to the learner.

## Explicit no-repeat list

Do not create a new core validation whose primary question is only one of the following:

1. Can Cells grow?
2. Can growth restore plasticity?
3. Can Cells be independently mutable?
4. Can sparse routing exist?
5. Can conflict induce differentiation?
6. Can bounded Cell-local subspace state protect registered history without replay?
7. Do real Pythia representations contain any reusable low-dimensional structure?
8. Is the common carrier causally useful?
9. Is semantic routing address automatically the functional split boundary?
10. Is the same pretrained effect geometry secretly sparse/local under another near-equivalent fixed dictionary?
11. Can the registered singleton-exposure 001 world form six Cells again?
12. Can the registered additive pair-superposition 001B world recover the same six latent Cells again?
13. Can the registered 002 synthetic growth curve be reproduced merely by extending the horizon?
14. Can the registered 003 certificate-growth world reproduce the same protected-write result under cosmetic parameter changes?
15. Can the registered 004 linear-operator world reproduce the same multi-Cell composition result with more synthetic cases or Cells?

A new experiment touching any of these topics must name a distinct integration variable in advance.

## Research stop rule

Natural-geometry negatives do not stop Constructive CLM. The native route becomes materially threatened only after repeated constructive failure to produce the combined set of:

1. reusable learned coordinates;
2. deployable read/write alignment;
3. bounded/improving growth;
4. protected continual writes without learner replay;
5. stable model-level multi-Cell computation;
6. progressive scaffold removal.

## Product boundary

External CLM Layer feasibility and Native CLM feasibility remain separate decisions. Failure of the endogenous route does not invalidate a product built from persistent mutable Cells, routing, protection, growth, versioning and rollback on top of a mature LLM.

## Milestone boundary

If CLM-005 is supported under its registered boundary, the next main milestone is **Small Native CLM v0**, not another indefinite synthetic mechanism series.

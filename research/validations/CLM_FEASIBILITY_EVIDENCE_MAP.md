# CLM Feasibility Evidence Map

Status: **FROZEN v1.2**  
Baseline: `main@7c99ef3f4aedb8d561cf9600ea5d79da1eb99b99` plus the registered Constructive CLM-001 and 001B formal results from 2026-09-02.

## Purpose

This map prevents the MiniCells research program from repeatedly re-testing mechanisms that are already supported, already falsified, or already constrained by prior experiments.

The product path is treated as:

```text
pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / native CLM
```

The native-CLM research question is therefore **not** whether every cellular mechanism can be rediscovered from scratch. It is whether the already-supported mechanisms can be composed with learned read/write coordinates and whether those coordinates remain useful with bounded/improving growth over long continual streams.

## Evidence grades

- **A — formal:** frozen multi-seed or otherwise registered scientific decision.
- **B — controlled:** repeatable mechanism evidence, but not a complete continual-learning decision.
- **C — diagnostic:** useful bridge/geometry evidence; not sufficient as a standalone CLM claim.
- **Open:** not yet established by the repository.

## Canonical evidence matrix

| Native-CLM proposition | Repository evidence | Grade | Frozen interpretation | Re-test? |
|---|---|---:|---|---|
| Useful functional organization can emerge under differential pressure | Experiments 014–024; especially conflict-driven differentiation / trait-genesis work | B | Self-organization is plausible; historical emergence is **not** itself validated continual learning | **No** toy emergence rerun |
| A Cell can be a sparse routed, independently mutable computational state unit | Stage 03; 025/026; CLM-0.1–0.3; progressive/probationary growth | B | Operational Cell granularity, sparse routing, independent mutation and capacity growth are viable engineering mechanisms | **No** basic Cell-unit rerun |
| Rejection/saturation can be converted into useful growth | Core Validation 004 | A | Growth can restore plasticity in the controlled CLM loop | **No** basic growth-rescue rerun |
| Bounded Cell-local state can replace learner-side historical replay for protected writes and mitosis | Core Validation 005 | A | In the registered linear-writable fixed-feature world, `Q` is sufficient for exact registered-history protection, saturation detection and reusable growth | **No** certificate-principle rerun |
| Replay-free subspace protection retains useful plasticity on real pretrained representations | Core Validation 006 | A (mixed) | Real Pythia states did not immediately saturate; certificate writes reduced forgetting while retaining about 0.84–0.89x replay gain | **No** basic real-representation certificate rerun |
| Semantic/routing address is a sufficient Cell split boundary | Core Validation 006 | A negative | **False under the registered mechanism.** Address-based mitosis did not sufficiently reduce functional conflict and growth covered ~65.6–68.8% of addresses | **Do not** revive as main hypothesis |
| Natural local/sparse write addressability is already present in the pretrained model | Core 002 / 002B / 002C | A negative | Locality, wider sparse assemblies and oracle tomography did not establish adequate natural write addressability | **Do not** repeat with cosmetic variants |
| Normalized functional writes have reusable factorized geometry | Core 009A; result commit `8290d4d674a8ec9ce98d4de129043526841e5f95` | A/C | Two-sided factor geometry exists, but later diagnostics show the right side largely collapses to a common activation carrier | Use as interface evidence only |
| The common right-side carrier is causally useful | Core 009B-1; result commit `f2691daf5738eac0232866a46d079db3aa61b60a` | A | Carrier-preserved writes retain roughly 97.6–98.2% of full-write target gain at the locked causal scale | **No** carrier-causality rerun |
| Pretrained effect vectors already lie in a compact persistent shared dictionary | Core 009B-2 discovery | C negative | No viable compact `<=32D` persistent effect subspace was found under the frozen discovery protocol | Natural-geometry No-Go only |
| Pretrained effect vectors already expose sparse/local Cell coordinates | Core 009C; result commit `3a524abd5b29a42c425dddab7df497cbfadfeecd` | C negative | `SPARSE_LOCAL_EFFECT_GEOMETRY_NOT_FOUND` | **Do not** keep searching for the same natural ontology |
| Full write operators retain compositional organization lost by carrier compression | Core 009D | Open | Representation-level operator question remains independent of the constructive route | Do not block Constructive CLM |
| Reusable Cell keys/effects can form without task/factor labels when factors receive singleton exposure | Constructive CLM-001; formal seeds `90111/90112/90113` | A | `LEARNED_COORDINATE_FORMATION_SUPPORTED`; six Cells covered six hidden factors and growth stopped after coverage | **No** singleton-world rerun; G1a frozen |
| Latent Cell keys/effects can be recovered when no hidden factor is ever presented alone | Constructive CLM-001B; formal seeds `90211/90212/90213`; result commit `55071bc7fd01e7c61df02846cd8f4205b906814f` | A | `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`; 3/3 seeds recovered six Cells from twelve pair prototypes and generalized to unseen pairs/triples | **No** equivalent pair-superposition rerun; G1b frozen |

## Reusable constructive mechanism chain

```text
sparse routed mutable state
+
transactional/growth lifecycle
+
replay-free local protection
+
real-representation writable geometry
+
causal foundation interface
+
addressable coordinate formation
+
latent coordinate discovery under registered superposition
```

The immediate missing bridge is now growth scaling rather than Cell existence or latent decomposability.

## Frozen gaps

### G1a — Addressable learned coordinate formation

Status: **SUPPORTED** by Constructive CLM-001.

Boundary: controlled structured continual world with singleton exposure.

### G1b — Latent coordinate discovery under superposition

Status: **SUPPORTED** by Constructive CLM-001B.

Frozen boundary:

```text
correlated/non-orthogonal latent factors
+ no singleton training
+ additive pair-superposition discovery scaffold
-> latent Cell keys/effects
-> x-only unseen pair/triple composition
```

This is not arbitrary blind source separation.

### G2 — Long-horizon growth law

Status: **ACTIVE**.

Primary question:

> Does learned Cell state track the amount of reusable latent structure rather than the number of continual-learning transactions?

Primary metric family:

```text
K(N), K(N)/N,
latent-vocabulary tracking error,
windowed spawn probability,
reuse rate,
Cell lifetime,
retention/composition quality
```

Primary experiment: **Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law**.

A finite positive result may establish sublinear-like scaling on the registered horizon, but must not be described as an asymptotic theorem.

### G3 — Learned coordinates + existing protection

Can learned coordinates coexist with the already-supported Core-005 certificate mechanism without collapsing plasticity or forcing near-linear growth?

Planned experiment: Constructive CLM-003.

### G4 — Multi-Cell composition

Can multiple learned Cells activate together in a model-level execution setting without destructive interaction? Controlled algebraic composition in 001/001B is reusable evidence but not the final model-level claim.

Planned experiment: Constructive CLM-004.

### G5 — External -> endogenous transition

Can handcrafted components be removed one at a time while preserving validated behavior?

```text
prototype/relational/residual-growth scaffolds
  -> learned router
  -> learned write controller
  -> learned growth controller
  -> slow-plastic foundation
  -> endogenous cellular model
```

Planned experiment: Constructive CLM-005.

## Explicit no-repeat list

Do not create new validations whose primary question is only one of the following:

1. Can Cells grow?
2. Can growth restore plasticity?
3. Can Cells be independently mutable?
4. Can sparse routing exist?
5. Can conflict induce differentiation?
6. Can bounded subspace state protect registered history without replay?
7. Do real Pythia representations contain any reusable low-dimensional structure?
8. Is the common carrier causally useful?
9. Is semantic routing address automatically the functional split boundary?
10. Is the same pretrained effect geometry secretly sparse/local under another near-equivalent fixed dictionary?
11. Can the registered singleton-exposure 001 world form six Cells again?
12. Can the registered additive pair-superposition 001B world recover the same six latent Cells again?

A new experiment touching one of these topics must name the new integration variable in advance.

## Research stop rule

A negative result in natural-geometry characterization does **not** stop Constructive CLM. The native route stops only after repeated constructive failure to produce all three of:

1. reusable learned coordinates under progressively weaker scaffolds;
2. deployable read/write alignment;
3. bounded or improving growth behavior.

## Product boundary

External CLM Layer feasibility and Native CLM feasibility remain separate decisions. Failure of the endogenous route does not invalidate a product built from persistent mutable Cells, routing, protection, growth, versioning and rollback on top of a mature LLM.

# CLM Feasibility Evidence Map

Status: **FROZEN v1.0**  
Frozen against `main@474f7f1c736a61a433e4d5a01691edc6795177da` (2026-09-02).

## Purpose

This map prevents the MiniCells research program from repeatedly re-testing mechanisms that are already supported, already falsified, or already constrained by prior experiments.

The product path is treated as:

```text
pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / native CLM
```

The native-CLM research question is therefore **not** whether every cellular mechanism can be rediscovered from scratch. It is whether the already-supported mechanisms can be composed with a learned read/write coordinate system and remain bounded over a long continual stream.

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
| Pretrained effect vectors already lie in a compact persistent shared dictionary | Core 009B-2 discovery, branch `codex/core-validation-009b2-persistent-effect-geometry` | C negative | Discovery found no viable compact `<=32D` persistent effect subspace; confirmation was forbidden | Treat as natural-geometry No-Go, not native-CLM No-Go |
| Pretrained effect vectors already expose sparse/local Cell coordinates | Core 009C; result commit `3a524abd5b29a42c425dddab7df497cbfadfeecd` | C negative | `SPARSE_LOCAL_EFFECT_GEOMETRY_NOT_FOUND`; no confirmation | **Do not** keep searching for the same natural ontology |
| Full write operators retain compositional organization lost by carrier compression | Core 009D | Open | Protocol/implementation exists at the frozen map baseline; no scientific result is imported into this map | Await result; do not duplicate |

## What is already strong enough to reuse as a component

The repository already supports this mechanism chain strongly enough that new native-CLM experiments should **compose** it rather than re-prove it:

```text
sparse routed mutable state
        +
transactional/growth lifecycle
        +
replay-free local protection
        +
real-representation writable geometry
        +
simple causal foundation interface
```

The missing bridge is:

```text
continual pressure
    -> learned Cell coordinates
    <-> learned read address
    -> reuse before spawn
    -> bounded long-run growth
```

## Frozen open gaps

### G1 — Learned coordinate formation and read/write alignment

Can the system form reusable Cell coordinates and a deployable read key from data **without task/factor labels** and without assuming a pretrained natural Cell ontology?

Primary experiment: **Constructive CLM-001 — Learned Coordinate Formation**.

### G2 — Long-horizon growth law

After G1, does growth become sublinear or at least show a declining spawn probability on a structured continual stream?

Primary metric family:

```text
K(N), K(N)/N, late-window spawn rate, reuse rate, Cell lifetime
```

Planned experiment: Constructive CLM-002.

### G3 — Learned coordinates + existing protection

Can G1 coordinates coexist with the already-supported Core-005 certificate mechanism without collapsing plasticity or forcing near-linear growth?

This is an **integration** test, not a new certificate-principle test.

Planned experiment: Constructive CLM-003.

### G4 — Multi-Cell composition

Can multiple learned Cells activate together on unseen combinations without destructive interaction? Core 009D may inform the representation, but it does not substitute for model-level compositional execution.

Planned experiment: Constructive CLM-004.

### G5 — External -> endogenous transition

Can handcrafted components be removed one at a time while preserving the validated behavior?

```text
prototype/read-key scaffold
  -> learned router
  -> learned write controller
  -> learned growth controller
  -> slow-plastic foundation
  -> endogenous cellular model
```

Planned experiment: Constructive CLM-005.

## Explicit no-repeat list

Do not create new core validations whose primary question is only one of the following:

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

A new experiment touching one of these topics must state the **new integration variable** that makes it non-duplicative.

## Research stop rule

A negative result in natural-geometry characterization (009B-2/009C/009D) does **not** stop Constructive CLM. The native route stops only if repeated constructive experiments fail to produce all three of:

1. reusable learned coordinates;
2. deployable read/write alignment;
3. bounded or improving growth behavior.

## Product boundary

External CLM Layer feasibility and Native CLM feasibility are separate decisions. Failure of the endogenous route does not invalidate a product built from persistent mutable Cells, routing, protection, growth, versioning and rollback on top of a mature LLM.

# MiniCells Continual-Learning Research Roadmap

Status: **two-track roadmap; Constructive CLM is the Native-CLM main line**  
Frozen evidence map: [`CLM_FEASIBILITY_EVIDENCE_MAP.md`](CLM_FEASIBILITY_EVIDENCE_MAP.md)  
Current constructive experiment: **Constructive CLM-003 — Protected Learned/Growing Cells**  
Current foundation-interface diagnostic: **Core 009D — Compositional Operator Geometry**

## Mission

\[
\boxed{
\text{pretrained LLM}
\rightarrow
\text{external CLM layer}
\rightarrow
\text{hybrid CLM}
\rightarrow
\text{endogenous / Native CLM}
}
\]

The program keeps two questions separate:

1. **Foundation Interface Research** — what useful writable interface already exists in a mature pretrained LLM?
2. **Constructive CLM Research** — can a persistent sparse coordinate system be learned, protected, composed and progressively endogenousized even when the checkpoint does not expose a natural Cell ontology?

Track-A negatives do not stop Track B.

---

# Track A — Foundation Interface Research

Reusable conclusion:

\[
\boxed{
\text{useful pretrained write interface}
\neq
\text{ready-made natural Cell ontology}
}
\]

Evidence reused by the product/constructive path:

- Core 005: replay-free Cell-local subspace certificates and reusable mitosis.
- Core 006: useful real-representation plasticity under protection; semantic/routing address is not a sufficient split boundary.
- Core 008: individual normalized writes are nearly rank-1; a small fixed shared matrix basis is rejected.
- Core 009A: asymmetric factorized functional geometry.
- Core 009B-1: carrier-only writes preserve roughly 97.6–98.2% of registered target gain.
- Core 009B-2 / 009C: no deployable compact persistent sparse/local natural Cell ontology under the registered discovery hypotheses.

Core 009D may continue only as a non-blocking interface/operator diagnostic.

---

# Track B — Constructive CLM Research

## Stable sequence

```text
G1a  CLM-001   learned coordinate formation                    🟢 SUPPORTED
G1b  CLM-001B  latent discovery under superposition            🟢 SUPPORTED
G2   CLM-002   long-horizon structure-tracking growth          🟢 SUPPORTED
G3   CLM-003   protected learned/growing Cells                  🔵 ACTIVE
G4   CLM-004   model-level multi-Cell computation               ⚪ PLANNED
G5   CLM-005   scaffold removal / endogenous transition         ⚪ PLANNED
                                                                ↓
                                                    Small Native CLM v0
```

This order is stable unless a registered experiment fails and exposes a specific dependency that must be repaired.

## G1a — Addressable learned coordinate formation

Status: **SUPPORTED**.

Constructive CLM-001 formal result:

```text
LEARNED_COORDINATE_FORMATION_SUPPORTED
seeds = 90111 / 90112 / 90113
Cells = 6 / 6 / 6
pair route recall = 1.0 / 1.0 / 1.0
```

Boundary: every hidden factor received singleton exposure.

## G1b — Latent discovery under superposition

Status: **SUPPORTED**.

Constructive CLM-001B formal result:

```text
LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED
seeds = 90211 / 90212 / 90213
training singleton count = 0
latent Cells = 6 / 6 / 6
pair route recall = 1.0 / 1.0 / 1.0
triple route recall = 1.0 / 1.0 / 1.0
```

Boundary: registered additive pair-superposition scaffold; not arbitrary blind source separation.

## G2 — Long-horizon structure-tracking growth

Status: **SUPPORTED**.

Constructive CLM-002 formal result:

```text
LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED
seeds = 90411 / 90412 / 90413
N_final = 4096
M_final = 30 latent factors
K_final = 30 Cells
K/N = 0.00732421875
finite-horizon fitted exponent = 0.6631226816
late spawn rate = 0.00439453125
late reuse rate ≈ 0.991
```

This is finite-horizon evidence that Cell state tracks reusable structure rather than transaction count. It is **not** an asymptotic theorem and does not establish a learned growth controller.

Do not create a cosmetic CLM-002B whose only change is a longer synthetic horizon.

## G3 — Protected learned/growing Cells

Status: **ACTIVE**.

Experiment: **Constructive CLM-003 — Protected Learned/Growing Cells**.

Parent components:

```text
CLM-001 / 001B learned coordinates
+
CLM-002 structure-tracking growth
+
Core-005 replay-free subspace certificate
```

New integration variable:

```text
learned root routing
+
Cell-local W/Q protected state
+
certificate-triggered context-keyed lineage mitosis
```

Registered comparison:

```text
unsafe
certificate_no_growth
certificate_growth
replay_growth_oracle
```

The key question is no longer whether `Q` works in isolation. It is:

> Can replay-free protection coexist with learned routing/growth without destroying plasticity, route stability or bounded functional growth?

A positive result must show all of:

- certificate learner historical replay accesses = 0;
- old behavior retained;
- acquisition gain ≈ replay oracle;
- safe writes reuse existing Cells;
- certificate saturation creates a context-addressable child rather than overwriting old behavior;
- old contexts continue routing to old Cells;
- new contexts reuse their child;
- long-tail writes produce no further unnecessary spawning.

Validation: [`constructive-clm-003-protected-growing-cells/README.md`](constructive-clm-003-protected-growing-cells/README.md).

## G4 — Model-level multi-Cell computation

Status: **PLANNED after G3**.

001/001B already contain controlled algebraic composition, but G4 must promote Cell from memory/write atom to computational module.

Candidate model-level form:

\[
h_{l+1}
=
h_l+
\sum_{i\in R(h_l)} g_i(h_l;\theta_i).
\]

Required evidence should include:

- multiple active learned Cells;
- unseen compositions;
- destructive-interference measurement;
- route-support recovery;
- sequential/simultaneous composition;
- compute scaling with active rather than total Cell count.

## G5 — Scaffold removal / endogenous transition

Status: **PLANNED after G4**.

Remove scaffolding one component at a time:

```text
prototype / relational discovery
  -> learned router
residual/probation growth
  -> learned growth controller
engineered write update
  -> learned write controller
frozen foundation
  -> slow-plastic foundation
  -> endogenous cellular model
```

Each removal is accepted only if it preserves or improves:

- new learning;
- old retention;
- growth efficiency;
- routing generalization;
- composition;
- compute/storage cost.

---

# Product Track — External CLM Layer

Product feasibility remains independent of Native-CLM feasibility.

\[
\boxed{
\text{mature frozen/slow LLM}
+
\text{persistent sparse mutable Cells}
+
\text{router}
+
\text{certificate}
+
\text{growth}
+
\text{version/rollback}
}
\]

The external product may keep engineered LoRA/rank-1/low-rank Cells, practical routing and explicit lifecycle controls even if full endogenousization fails.

---

# Experimental discipline

- Frozen formal seeds are never silently replaced; any observed seed is permanently excluded from later confirmation.
- Existing supported mechanisms are reused, not re-proved, unless a new integration variable is registered in advance.
- Hidden factor/task/mode labels are evaluator-only unless explicitly allowed by a protocol.
- Natural-geometry negatives do not become Constructive-CLM No-Gos.
- Short/finite streams are never described as asymptotic theorems.
- Coordinate recovery is insufficient without deployable read routing.
- Read routing is insufficient without protected writes and bounded growth.
- Protected writes are insufficient without model-level multi-Cell computation.
- A fully engineered constructive system is insufficient to claim Native CLM until scaffolds are progressively removed.

## Immediate order

```text
1. CLM-001  = SUPPORTED.
2. CLM-001B = SUPPORTED.
3. CLM-002  = SUPPORTED.
4. Run CLM-003 protection integration on frozen seeds after smoke/development validation.
5. If CLM-003 is positive -> CLM-004 model-level multi-Cell computation.
6. If CLM-004 is positive -> CLM-005 scaffold removal/endogenous transition.
7. If CLM-003/004/005 are all positive -> train Small Native CLM v0.
```

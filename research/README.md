[English] | [中文](README.zh-CN.md)

# MiniCells Research

MiniCells separates the **product architecture** from the stronger **endogenous / Native CLM research hypothesis**.

```text
mature pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous / Native CLM
```

The external product path may use engineered persistent Cells, sparse routing, replay-free protection, growth, versioning and rollback. Native-CLM research asks how far those mechanisms can become learned, compositional and endogenous.

## Native CLM core progress

This table is the stable high-level research scoreboard. When a registered experiment is completed, update the evidence and status here instead of redefining the roadmap.

Status legend:

- 🟢 **Supported / reusable** — formal or strong enough to treat as a component rather than re-test.
- 🟡 **Partial evidence** — useful evidence exists, but the stronger Native-CLM claim remains open.
- 🔵 **Active** — current main Constructive CLM experiment.
- ⚪ **Planned** — later gate in the frozen sequence.
- 🔴 **Not supported / blocked** — the registered hypothesis failed or should not be used as the main path.

| # | Native CLM proposition | Current evidence | Status |
|---:|---|---|---|
| 1 | Functional organization can emerge under pressure | Experiments 014–024 | 🟡 Strong emergence evidence; not itself continual-learning proof |
| 2 | Sparse Cells can be independently mutable computational units | 025/026, CLM-0.1–0.3 | 🟢 Reusable mechanism |
| 3 | Conflict can trigger differentiation / growth | 021–024, Core 004 | 🟢 Reusable mechanism |
| 4 | Growth can restore plasticity | Core 004 | 🟢 Formally supported |
| 5 | Historical behavior can be protected without learner-side replay | Core 005; real-representation bridge in Core 006 | 🟢 Certificate principle supported |
| 6 | Mature LLMs expose a useful writable interface | Core 006, 009A, 009B-1 | 🟢 Strong foundation-interface evidence; **not** a natural Cell ontology |
| 7 | Reusable Cell coordinates / read addresses can form from experience | Constructive CLM-001 and 001B | 🟢 Controlled constructive formation supported, including no-singleton superposition discovery |
| 8 | Long-horizon Cell growth can track reusable structure rather than transaction count | Constructive CLM-002 | 🟢 Finite-horizon structure-tracking growth supported; not an asymptotic theorem |
| 9 | Learned/growing Cells can support replay-free protected continual writes | **Constructive CLM-003** | 🔵 **Active** |
| 10 | Multiple learned Cells can perform stable model-level computation/composition | Constructive CLM-004 | ⚪ Planned |
| 11 | Router/write/growth scaffolds can be removed toward endogenous control | Constructive CLM-005 | ⚪ Planned |
| 12 | Train a Small Native CLM v0 | after 003–005 | ⚪ Milestone |

The important negative boundary remains:

```text
pretrained semantic/routing address
!=
automatically correct functional Cell boundary
```

Core 006 and the 002/009 natural-geometry line prevent us from returning to that assumption.

## Current main experiment

### Constructive CLM-003 — Protected Learned/Growing Cells

Current question:

> Can learned/growing Cell coordinates be combined with the already-supported Core-005 replay-free certificate so that new writes retain old behavior, preserve plasticity, and create bounded context-addressable children instead of destructive overwrite or replay?

CLM-003 directly reuses:

```text
Constructive CLM-001 / 001B
  learned Cell coordinates
+
Constructive CLM-002
  structure-tracking growth
+
Core 005
  replay-free subspace certificate
```

The new integration variable is:

```text
learned hierarchical routing
  + protected mutable W/Q state
  + certificate-triggered context-keyed mitosis
```

Validation: [Constructive CLM-003 — Protected Learned/Growing Cells](validations/constructive-clm-003-protected-growing-cells/README.md).

## Constructive CLM sequence

```text
G1a  CLM-001   addressable learned coordinate formation      🟢
G1b  CLM-001B  latent discovery under superposition          🟢
G2   CLM-002   long-horizon structure-tracking growth        🟢
G3   CLM-003   protected learned/growing Cells                🔵
G4   CLM-004   model-level multi-Cell computation             ⚪
G5   CLM-005   scaffold removal / endogenous transition       ⚪
                                                          ↓
                                              Small Native CLM v0
```

The canonical evidence-reuse/no-repeat policy is frozen in the [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md), with a machine-readable companion at [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml).

See the [Continual-Learning Research Roadmap](validations/CONTINUAL_LEARNING_ROADMAP.md) for detailed experiment boundaries.

## Two research tracks

### Track A — Foundation Interface Research

Core 006–009 characterize a mature pretrained LLM as a writable substrate for an external CLM layer.

Current conclusion:

```text
useful low-dimensional/factorized write interface
!=
ready-made natural Cell ontology
```

Core 009D remains a non-blocking operator-geometry diagnostic. Track-A negatives do not stop Constructive CLM.

### Track B — Constructive CLM Research

This is the main Native-CLM feasibility line. It is allowed to **construct and learn the Cell coordinate system** even when the pretrained checkpoint does not expose one naturally.

Formal constructive parent evidence now includes:

- **CLM-001** — `LEARNED_COORDINATE_FORMATION_SUPPORTED`, seeds `90111/90112/90113`.
- **CLM-001B** — `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`, seeds `90211/90212/90213`.
- **CLM-002** — `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`, seeds `90411/90412/90413`; final registered growth tracked 30 latent factors with 30 Cells at `N=4096` and `K/N=0.007324`.

## Research stages

1. [Foundations](stages/01-foundations/README.md) — Echo, NCA language dynamics, 1D/2D tissue, settling, and training mechanics.
2. [Self-Organization](stages/02-self-organization/README.md) — sparse topology, recruitment, differentiation, and trait genesis.
3. [Routing and Growth](stages/03-routing-and-growth/README.md) — routed, independently mutable Cell state and capacity growth.
4. [Continual-Learning Core](stages/04-continual-learning-core/README.md) — write-addressability failures, growth-restored plasticity, replay-free certificates, real-representation constraints, foundation-interface geometry, and Constructive CLM.
5. [Language Validation](stages/05-language-validation/README.md) — historical token-level transfer/scale-readiness work; later Native-CLM language validation will resume after the constructive core is integrated.

## Research assets

- [Experiment implementations](experiments/README.md) are stage-aligned adapters that import reusable code from `src/minicells/`.
- [Notebook assets](notebooks/README.md) preserve historical experiment IDs and workflows.
- [Validations](validations/) contain frozen protocols, evidence maps and decision documents.
- [Canonical artifacts](../artifacts/experiments/) are immutable scientific evidence once formal runs are published.
- Machine-readable historical paths/outcomes remain in [`catalog.yaml`](catalog.yaml).

## Current boundary

The repository does **not** yet establish general natural-language continual learning, an asymptotic `K(N)=o(N)` theorem, arbitrary latent-source discovery, a fully learned router/growth controller, simultaneous model-level multi-Cell computation, or an endogenous LLM-scale CLM.

If CLM-003, CLM-004 and CLM-005 succeed under their registered boundaries, the next milestone is no longer another toy mechanism validation: it is training the first **Small Native CLM v0**.

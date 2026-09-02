[English] | [中文](README.zh-CN.md)

# MiniCells Research

MiniCells separates the **product architecture** from the stronger **endogenous CLM research hypothesis**.

The product path is:

```text
pretrained LLM
  -> external CLM layer
  -> hybrid CLM
  -> endogenous/native CLM
```

The external layer may use engineered persistent Cells, sparse routing, replay-free protection, growth, versioning and rollback. Native-CLM research asks whether the coordinate system, read/write alignment and eventually the growth/write controllers can themselves become learned/endogenous.

## Current research split

### Track A — Foundation Interface Research

Core 006–009 characterize what a mature pretrained LLM already provides as a writable substrate. Current evidence supports useful low-dimensional/factorized write structure and carrier causal sufficiency, but does **not** support treating pretrained semantic/routing addresses or carrier-effect vectors as a ready-made natural Cell ontology.

Core 009D remains a non-blocking operator-geometry diagnostic.

### Track B — Constructive CLM Research

This is the main native-CLM feasibility line.

Two coordinate-formation gaps are now formal parent evidence:

- **G1a / Constructive CLM-001 — SUPPORTED** on untouched formal seeds `90111/90112/90113`: addressable learned Cells formed in the registered singleton-exposure world.
- **G1b / Constructive CLM-001B — SUPPORTED** on untouched formal seeds `90211/90212/90213`: six latent Cells were recovered with no singleton training, from twelve additive pair-superposition prototypes, and x-only routing generalized to unseen pairs/triples.

The active question is now growth scaling rather than Cell existence:

> Does learned Cell state track the amount of reusable latent structure rather than the number of continual-learning transactions over a long stream?

Current experiment: [Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law](validations/constructive-clm-002-long-horizon-growth-law/README.md).

The canonical reuse/no-repeat policy is frozen in the [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md), with a machine-readable companion at [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml).

Read the [Continual-Learning Research Roadmap](validations/CONTINUAL_LEARNING_ROADMAP.md) for the constructive sequence.

## Research stages

1. [Foundations](stages/01-foundations/README.md) — Echo, NCA language dynamics, 1D/2D tissue, settling, and training mechanics.
2. [Self-Organization](stages/02-self-organization/README.md) — sparse topology, recruitment, differentiation, and trait genesis.
3. [Routing and Growth](stages/03-routing-and-growth/README.md) — Cells became routed, independently mutable computational state.
4. [Continual-Learning Core](stages/04-continual-learning-core/README.md) — write-addressability failures, dependency-scoped transactions, growth-restored plasticity, replay-free certificates, real-representation constraints, and foundation-interface geometry.
5. [Language Validation](stages/05-language-validation/README.md) — historical token-level transfer/scale-readiness work; it no longer blocks the constructive native-CLM line.

## Evidence already reused by Constructive CLM

- Core 004: growth can restore plasticity in the controlled CLM loop.
- Core 005: bounded Cell-local subspace state can replace learner-side replay for registered-history protection, saturation detection, and reusable growth in the frozen linear-writable world.
- Core 006: real pretrained representations contain reusable structure and replay-free certificate writes retain useful plasticity, while semantic/routing address is not a sufficient mitosis boundary.
- Core 009A/009B-1: a compact foundation write interface exists; carrier-only writes preserve most of the tested causal target gain.
- Core 009B-2/009C: the tested pretrained carrier-effect representation does not expose the desired compact persistent sparse/local Cell ontology. These are natural-geometry No-Gos, not Constructive-CLM No-Gos.
- Constructive CLM-001: addressable learned Cell formation is supported under the registered singleton scaffold.
- Constructive CLM-001B: latent Cell discovery is supported under the registered additive pair-superposition scaffold with no singleton training.

## Research assets

- [Experiment implementations](experiments/README.md) are stage-aligned adapters that import reusable code from `src/minicells/`.
- [Notebook assets](notebooks/README.md) preserve historical experiment IDs and workflows.
- [Validations](validations/) contain frozen protocols, evidence maps and decision documents.
- [Canonical artifacts](../artifacts/experiments/) are immutable scientific evidence once formal runs are published.
- Machine-readable historical paths/outcomes remain in [`catalog.yaml`](catalog.yaml).

No current repository result establishes general natural-language continual learning, asymptotically sublinear Cell growth, arbitrary latent-source discovery, a fully learned growth policy, write-certificate integration for the constructive coordinates, or an endogenous LLM-scale CLM. Constructive CLM-002 is deliberately narrower: it measures finite-horizon structure-tracking growth before certificate integration and later scaffold removal.

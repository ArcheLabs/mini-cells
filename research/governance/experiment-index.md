# Research Experiments

- **EPCLM-003/004** — [Unified Memory Architecture, 10M](../projects/cellular-language-models/experiments/epclm-004-unified-memory-10m/README.md). M0 audit complete; E0–E5 implementation and training pending. No training performed; H1–H6 NOT_EVALUATED. Source specification and observed environment are preserved with the experiment.

Every experiment belongs to one project and lives at:

```text
research/projects/<project>/experiments/<experiment>/
```

That directory is the working unit for the study. It may contain a README,
protocol/configuration, implementation, notebook, reports, and experiment
artifacts. It need not contain empty placeholder subdirectories for assets it
does not use.

Cross-project policy and frozen validation records remain in
[`governance/`](./). Shared implementation utilities live under
[`shared/`](../../shared/); repository-level release and maintenance entrypoints
live under [`scripts/`](../../scripts/).

The registry is only a discovery index. The owning experiment's frozen protocol
and recorded source commit determine what was run and what can be concluded.

- **EPCLM-CORE-001-C1-DIAGNOSTIC** — [Frozen C1 mechanism diagnostic](../projects/cellular-language-models/experiments/epclm-core-001/reports/mechanism-diagnostic/README.md). Post-training exploratory development; original C1 frozen; D0–D3 implementation ready, real asset recovery/GPU execution pending; H1–H5 NOT_EVALUATED.

- **EPCLM-CORE-001-CP-001** — [Communication Propagation-001](../projects/cellular-language-models/experiments/epclm-core-001/reports/communication-propagation/README.md). Frozen C1, spatial propagation/cross-event persistence/predictive utility measured separately. Exploratory implementation ready; GPU unrun; no training or formal confirmation.

- **EPCLM-GC-001** — [Generative Core and baseline pipeline](../projects/cellular-language-models/experiments/epclm-core-001/reports/generative-core-001/README.md). Repository-audited extension of Runtime; C1/F0 frozen reuse, center-only controlled training, optional Native T1 reference. Implementation ready, GPU unrun, development-only holdout; H1–H5 NOT_EVALUATED.

- **EPCLM-GC-001-BASELINE-REPAIR** — [Matched broadcast/independent/Flat controls and generation diagnosis](../projects/cellular-language-models/experiments/epclm-core-001/gc001-baseline-repair/README.md). Separate immutable-source development study; P1→P2→P3, single-seed, no independent Test; implementation ready, GPU unrun.

- **M3C-SERIES** — [Mamba-3 Cellular series proposal](../projects/cellular-language-models/experiments/mamba3-cellular-series/README.md). Proposed, not registered; Mamba-3 baseline, bounded Workspace, stateless Cells, dynamic routing and State-Memory Workspace are staged as separate gates; no training authorized.

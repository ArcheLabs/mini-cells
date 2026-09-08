# HybridCLM overview

HybridCLM is the MiniCells v0.1 research toolkit for attaching independently
manageable Cell mutations to a pretrained MoE. A frozen foundation supplies
the mature computation; a mutation supplies a reversible, explicitly scoped
change.

```text
pretrained Granite MoE + Cell mutation
              │
              ├── alpha = 0   mutation OFF
              ├── alpha = 1   full learned mutation
              └── detach      exact rollback to the cellularized baseline
```

The public API is intentionally small: inspect, explicit cellularization,
attach, detach, alpha scaling, serialization, compatibility validation, and
rollback reporting. v0.1 does not claim automatic optimal placement, generic
MoE support, standalone CLM conversion, or a continual-learning solution.

The primary research direction is progressive dependency withdrawal from the
parent model. HybridCLM is also a candidate modular post-training mechanism,
but that secondary hypothesis requires a matched future comparison.

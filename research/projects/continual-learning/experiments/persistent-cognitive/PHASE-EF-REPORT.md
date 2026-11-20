# Phase E/F verification

Inference mode remains active for Qwen. Only online surprise-gradient
calculation enters a normal tensor context. Generation returns the final
unconsumed token and the advanced Session. Resume restores model,
optimizer, RNG and a deterministic episode cursor at committed steps.

The root pytest wrapper isolates research module names. CPU CI installs
the pinned text backend and checks syntax and runtime tests. Local
execution covers inference-mode online writes and exact fixture
checkpoint continuation. Tiny Qwen tests require Transformers 5.3.0.

The T4 notebook checks real inference-mode generation and a two-update
resume fixture. This is an engineering test. Long-run 10,000-token
boundedness and measured runtime memory remain pending.

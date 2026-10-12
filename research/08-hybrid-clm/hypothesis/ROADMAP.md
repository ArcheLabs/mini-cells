# HybridCLM roadmap

1. Keep the supported Granite API small, explicit, and provenance-checked.
2. Build a progressive dependency-withdrawal experiment around a separately
   registered protocol; do not overload alpha with parent-dependency semantics.
3. Run a matched HybridCLM vs LoRA vs strong continual/mixture-LoRA kill test
   only after the infrastructure and budgets are frozen.
4. Add other MoE backends only after independent compatibility and numerical
   tests pass.

Auto-placement, router training, arbitrary MoE mutation, and standalone
conversion remain research work rather than v0.1 defaults.

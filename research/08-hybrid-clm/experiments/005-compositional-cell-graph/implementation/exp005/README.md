# Experiment 005 implementation

This package is owned by the HybridCLM research project. It reuses the project's `experiments.common.models.FeedForward` and `GlobalRouter`, plus the shared `make_batch` and `write_json` training utilities. Experiment-specific graph routing, causal blocks, accounting and gates stay here.

The proposed config is deliberately non-runnable for full training until the protocol status changes from `proposed` and its registered seed/holdout/edge-refresh identities are filled. `preflight` and `dry-run` are CPU-only. The old Consumer Language Bridge experiment numbered 005 has its own historical artifact publisher and is not modified here.

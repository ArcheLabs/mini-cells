# Manual execution policy amendment — 2026-10-12

The user authorized removing the one-hour restriction for manually executed Kaggle notebooks. The canonical Experiment 004 notebook now uses `manual=True` in the existing runtime. This explicit execution-policy override replaces the historical time limits for new manual runs; the original registered protocol remains available for interpreting earlier observations.

Manual execution imposes no preparation, session, per-model training, validation or measurement deadline. Each arm completes its registered 10M training tokens, then checkpointing and evaluation. There is no reserved time window or idle-time rejection. The existing runtime and workers are reused; `--manual` is propagated to each independent GPU worker. Historical automatic callers retain their original default behavior.

The runtime records `execution_policy: manual-token-budget` in run provenance and the evidence manifest, and records `manual_execution: true` in the effective run configuration. No infinite number is serialized into evidence. Source, data, configuration and protocol hashes, seeds, validation population, model definition and development gates remain checked. The original GPU queue allocation is retained for this amendment.

Already observed evidence retains its original source and policy. Recover the archived seed 42/reference result under source `7c1d8b23adaf047195d38a6827f46f3e97ae1faa` without retraining or changing its identity. New manual runs use their own source-specific archive and matrix; the collector must not relabel or silently combine results from different source revisions.

Kaggle infrastructure may independently terminate its own session. Such an interruption remains incomplete evidence; removing application timers does not make missing training or validation pass a development gate.

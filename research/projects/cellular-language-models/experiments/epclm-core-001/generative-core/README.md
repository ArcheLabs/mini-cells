# Generative Core-001

Start with [before-edit audit](generative-core-audit.md), [fixed GC protocol](../generative-core-protocol.json), and [implementation/execution record](../reports/generative-core-001/README.md). Code reuses the existing implementation directory and Runtime rather than duplicating the engine.

Primary run: [generative-core-001.ipynb](../generative-core-001.ipynb). Optional later Native T1 batch: [generative-core-001-transformer.ipynb](../generative-core-001-transformer.ipynb).

Generation sessions are created by `Runtime(core).generation(tokenizer, model_identity, tokenizer_sha, seed=713)`. The launcher supplies audited identities automatically. Use `prefill(prompt)`, `generate(...)`, `read()`, optional `evolve(steps)`, `snapshot()`, `restore(snapshot)` and `reset()`. `generate` returns new IDs and full continuation text. Snapshots contain state, sampling RNG and UI output metadata, never model weights.

Current status: implementation ready, actual GPU evaluation/training not run; no independent Test in pinned assets.

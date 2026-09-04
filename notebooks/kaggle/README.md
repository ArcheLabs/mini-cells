# Kaggle launchers

Kaggle is the primary hosted GPU execution environment for experiments that are too expensive or unsuitable for GitHub Actions.

## Required secrets

Formal publishing uses the existing repository publisher path and expects:

- `GITHUB_TOKEN` — GitHub token with write access to `ArcheLabs/mini-cells`.
- `HF_TOKEN` — optional for public Hugging Face models, recommended to avoid anonymous rate limits.

Secrets are read through Kaggle's `UserSecretsClient`; they must never be written into notebook cells, logs, or committed files.

## Formal execution rules

1. Start from a fresh Kaggle session when possible.
2. Clone the exact experiment branch.
3. Run GitHub write preflight **before GPU work**.
4. Confirm CUDA and record the GPU model.
5. Invoke only the canonical runner under `scripts/research/`.
6. Run one formal seed at a time.
7. Publish that seed immediately after it produces `result.json`, even when the scientific status is `FAIL`.
8. On restart, skip seeds already present under `artifacts/experiments/<experiment>/seed-*` on the remote branch.
9. Never wait for all seeds before publishing.

## Current formal launcher

- `moe-mutation-001.ipynb` — Granite 3.1 1B-A400M first isolated CLM mutation test.

The notebook intentionally uses one CUDA device even if Kaggle assigns two GPUs. Multi-GPU execution would change the execution path and is outside the frozen Mutation 001 protocol.

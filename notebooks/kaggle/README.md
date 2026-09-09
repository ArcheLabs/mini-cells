# Kaggle launchers

Kaggle is the primary hosted GPU execution environment for experiments that are too expensive or unsuitable for GitHub Actions.

## Required secrets

HybridCLM publication uses only:

- `HF_TOKEN` — a fine-grained Hugging Face write token for the configured Model,
  Space, and Collection.

The notebook does not create tags, GitHub releases, or write to the repository.
GitHub/PyPI software release is handled independently by the tag-driven
`release.yml` workflow.

Secrets are read through Kaggle's `UserSecretsClient`; they must never be written into notebook cells, logs, or committed files.

## Engineering publication rules

1. Start from a fresh Kaggle session when possible.
2. Clone the exact immutable release tag.
3. Validate the release identity and formal seed guard **before GPU work**.
4. Confirm CUDA and record the GPU model.
5. Invoke only the canonical runner under `scripts/research/`.
6. Export and validate the engineering Cell mutation before any Hugging Face upload.
7. Never execute formal seeds or change `research/formal_seed_registry.json`.

## Current launchers

- `hybrid_clm_release_v0_2_0a1.ipynb` — immutable HybridCLM publication launcher.
- `moe-mutation-001.ipynb` — historical experiment launcher.

The notebook intentionally uses one CUDA device even if Kaggle assigns two GPUs. Multi-GPU execution would change the execution path and is outside the frozen Mutation 001 protocol.

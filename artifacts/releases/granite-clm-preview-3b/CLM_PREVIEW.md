# Granite-CLM-Preview-3B

Granite-CLM-Preview-3B is a lossless CLM substrate lift of
`ibm-granite/granite-3.1-3b-a800m-base`.

The upstream Granite MoE execution path and checkpoint bytes are preserved.
The CLM layer adds provenance, tensor identities, and deterministic expert-slice
addresses around the immutable substrate.

## Preview boundary

- No training or weight mutation is performed.
- Expert addresses are canonical mutation coordinates only.
- This release does **not** claim that pretrained MoE experts are Cells.
- It does **not** claim safe model evolution, composability, or replay-free learning.

## Identity

- Upstream model: `ibm-granite/granite-3.1-3b-a800m-base`
- Upstream revision: `d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a`
- CLM manifest identity: `6e52357f49d578420f43fabe1e6f6da51788dda680ab1ad9e1e50f3bf747117f`

## Release environment

- Transformers: `4.46.0` (pinned)
- Execution policy: `single_gpu_deterministic_reference_parity`
- Requested device: `cuda:0`
- Visible CUDA devices: **2**
- Deterministic algorithms: **enabled**
- Attention reference backend: `eager`

The release intentionally uses a single GPU even when two GPUs are visible.
Granite 4.46 uses CUDA `index_add` in its MoE expert accumulation path, so
ordinary FP16 CUDA inference is not used as a bitwise-equivalence oracle.
Instead the release uses a deterministic reference path and first proves that
the source checkpoint reproduces itself before comparing it with the CLM bundle.

## Baseline controls

- Same loaded source, repeated forward: **PASS**
- Source checkpoint after independent reload: **PASS**

## Release gates

- Bundle integrity: **True**
- Checkpoint byte identity: **True**
- Forward parity: **True**
- Router Top-K identity: **True**
- Greedy token identity: **True**

Release status: **PASS**.

The exact upstream Hugging Face checkpoint is stored under `substrate/` so the
CLM bundle remains independently verifiable with `verify_clm_moe_bundle`.
Machine-readable evidence is in `metrics.json`, `parity_report.json`,
`provenance.json`, and `clm_moe_manifest.json`.

# MiniCells HybridCLM release-preparation report

## Source

- Source branch: `codex/hybrid-clm-release-preparation-v0.1`
- Release-preparation implementation commit: `0fbfb74`
- Final report commit: the commit containing this report
- Package version: `0.2.0a1`

## Repository changes

- Deterministic public fixtures moved from `fixtures/` to `tests/fixtures/`;
  Rust/Python consumers were updated and provenance is documented.
- Root `results/.gitkeep` was removed. `results/` is ignored as transient local
  output; durable evidence remains under `artifacts/`.
- CLM-0.4 Preview/Release originals are preserved under
  `research/archive/clm-0.4/{preview,release}/`; old paths remain lightweight
  compatibility pointers.
- PCU-KILL-001 protocol template is available under the validation family;
  the historical protocols path has a compatibility pointer.
- `research/audits/BRANCH_ARCHIVE.json` and `.md` inventory local branch heads.
  No branches were deleted; current HybridCLM, Stage-1, and research branches
  are explicitly marked keep/review.
- `research/stages/08-hybrid-clm/` establishes the current strategy, evidence
  boundary, and roadmap; `research/catalog.yaml` indexes it.

## HybridCLM API

The public package exports:

```python
from minicells import HybridCLM, CellMutation, CellPlacement, ModelInspection
```

Implemented operations are structural inspection, explicit placement,
cellularization, attach/detach, alpha scaling, context-managed expression,
zero-state diagnostics, restoration checks, safe serialization, and strict
compatibility validation. Placement search, router training, and speculative
dependency-withdrawal beta semantics are intentionally not implemented.

## Supported backend

`GraniteMoEBackend` is the only `SUPPORTED` backend. It reuses the validated
router-preserving fused `gate_up_proj`/`down_proj` SwiGLU partition. Mixtral,
Qwen MoE, DeepSeek MoE, dense Transformers, and unknown layouts are
`UNSUPPORTED` and fail closed.

## Artifact schema

Mutation directories contain `mutation.safetensors`, `cell_config.json`,
`manifest.json`, `provenance.json`, and `evaluation.json`. Schema version 1
records placements and tensor targets, pins base model/revision and architecture
hash, records tensor shapes/dtype, and verifies the mutation SHA256. Public
weights are safetensors-only; arbitrary pickle deserialization is not used.

## Tests and CI

- Focused HybridCLM + PCU + fixture parity: **45 passed, 1 skipped** (the skip
  is the optional safetensors roundtrip because this host does not have that
  package installed).
- Rust workspace: **passed**, including Stage-1 hierarchical reduction and
  MiniJAM/PVM parity.
- Full Python collection: **blocked by pre-existing host environment issues**
  (Python 3.8 incompatibilities in older research modules, missing
  `tokenizers`, and a broken auto-loaded web3 pytest plugin). No formal runner
  was invoked.
- `.github/workflows/hybrid-clm-ci.yml` adds Python 3.11 install, ruff,
  focused tests, notebook JSON checks, and formal-seed integrity checks. It
  does not publish automatically.

## Hugging Face assets

Prepared under `artifacts/releases/hybrid-clm-v0.1/`: a model-card template,
Space data bundle with frozen v3 engineering metrics, publication README, and
`scripts/release/prepare_hybrid_clm_release.py` for staging a validated real
mutation artifact. No foundation checkpoint is copied and no synthetic binary
mutation is presented as a trained release.

## Known limitations

The first backend currently cellularizes the verified expert runtime for a
selected MoE block while mutation targets provide the fine-grained declared
scope. A real Granite mutation artifact still requires a compatible trained
run and immutable Hub revision. The v3 locality threshold remains unresolved;
engineering evidence is not formal validation. Hosted notebooks are launchers,
not scientific protocols.

Formal Hybrid CLM execution: NOT STARTED
Formal seeds: RESERVED_UNTOUCHED

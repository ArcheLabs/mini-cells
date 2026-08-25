# MINI Cells V0.2 implementation status

Compatibility target: JAM semantics 0.7.2; MiniJAM
`5947c50699863948c51028bc346980481d839884`; Jambda
`e52307a726868205a151e6917a0a70a79965a028`; JamScript
`79347ca2435ca21a08cbd257bc9c3dce8ed77f4b`. The machine-readable source of
truth is [`artifacts/implementation-status.json`](artifacts/implementation-status.json).

| Area | Status | Evidence |
|---|---|---|
| Wallet challenge/session | PASS | Canonical SS58/32-byte decoding, random 32-byte nonce, genesis/service binding, one-use five-minute challenge, sr25519 verification, bounded in-memory 12-hour opaque cookie sessions, logout and operator mapping are implemented and unit-tested. |
| Operator/CORS boundary | PASS | Mutable training is operator-only; unset operator configuration yields no operator sessions; credentialed CORS accepts only `MINICELLS_WEB_ORIGIN`; no permissive wildcard layer remains. |
| Authenticated SSE | PASS_CODE_PATH | Authenticated route, immediate snapshot, bounded broadcast, heartbeat keep-alive, required event names, and fresh current snapshot on lag are implemented. Full HTTP integration requires a running Keeper fixture. |
| Canonical model gateway | PASS | `/v1/model` reads finalized `META` and `mc:v1:model`, checks exact 8,952-byte length, decode, and canonical hash, and fails closed on mismatch. |
| Browser local inference | PASS | `minicells-wasm` is a bounded raw ABI over `minicells-core`; release wasm32 artifact is built and copied by `npm run build`; Vitest loads the real artifact and verifies deterministic inference, hash/length/vocabulary rejection. |
| Wallet web flow | PASS_CODE_PATH | Polkadot extension `web3Enable`/`web3Accounts`/`signRaw` challenge flow, credentialed fetches, in-memory model cache, model refresh/stale display, SSE event consumption, logout, and no normal-user training controls are implemented. |
| PVM verification inference | PASS_CODE_PATH | `/v1/verify/infer` waits for a finalized inference-ring record matching the submitted request ID instead of treating an execution receipt as completion. |
| Dependency portability | PASS | Absolute MiniJAM Cargo paths were removed. `tools/bootstrap_deps.sh` pins repository-relative `.deps` sources and exact MiniJAM/Jambda refs; `.deps` is ignored and never committed. |
| Artifacts/refs | PASS | Service ELF/blob/PolkaVM artifacts rebuilt; manifest records actual dependency refs, toolchain, target hash, and WASM artifact provenance. |
| Artifact provenance ledger | PASS | `tools/build_service.sh` now refuses dirty reproducibility builds unless explicitly opted in, captures source ref/tree/dirty identity, records JamScript as unused, and generates a clean non-sticky manifest. Stale-field and dependency-ref tests pass. |
| Historical deployment evidence | PASS_CODE_PATH | Prior block-8/`0x3748…` attempt is preserved under `artifacts/deployments/`; generated artifact manifests no longer carry mutable deployment history. |
| Rust verification | PASS | `cargo test --offline --workspace` passed: core, protocol, simulator, chain, Keeper auth/model tests, and WASM ABI tests. |
| Web verification | PASS | `npm test -- --run` passed (5 tests); `npm run build` passed with current WASM artifact. |
| Python validation | BLOCKED_EXTERNAL | `python3 -m pip install --upgrade pip setuptools wheel` could not reach PyPI (`ConnectTimeoutError: pypi.org`); no Python result is claimed. |
| Service receipt / fresh deployment | BLOCKED_CASE_A | Historical block-8 attempt remains preserved. A genuinely fresh chain (genesis `0x05cc868745308c2718e08c3e4cd0cb9fc84796a5540fa575d3ceef670ee1e7f7`) was used for the rebuilt artifact; runtime `validate_transaction` panicked with `Codec error` before CreateService inclusion, so no receipt or Service ID is claimed. Evidence is append-only under `artifacts/deployments/2026-08-25T035154Z-create-service.json`. |
| Worker / generation | BLOCKED_EXTERNAL | Because the canonical creation receipt never materialized, no receipt-derived Service ID exists for initialization. The available direct worker also reported `processed=0`; no PLUS/MINUS, 0→1, gas, multi-generation, or restart result is fabricated. |

## Commands and evidence

```text
cargo test --offline --workspace                         PASS
npm --prefix apps/web test -- --run                     PASS (5 tests)
npm --prefix apps/web run build                         PASS
./tools/build_service.sh                                 PASS (69,414-byte blob)
minicells deploy ...                                     BLOCKED_EXTERNAL_RECEIPT (60s timeout)
status-probe / worker finalization                       BLOCKED_EXTERNAL (no receipt-derived ID)
```

The external chain blocker does not weaken the browser integrity invariant:
the WASM runtime recomputes the canonical model hash before every forward pass,
and the Keeper independently verifies the finalized model before serving it.

## Experiment 002 — persistent local training status (2026-08-25)

| Requirement | Status | Evidence |
|---|---|---|
| Persistent native trainer, arbitrary generation | PASS | `minicells-sim::trainer` drives the production no-std `refine`/`accumulate` entry points and completed a release 0→1000 run in 25.31 s (checkpointing at 250/500/750/1000); the recorded genesis evaluation was 1/80 tokens and the generation-1000 selected evaluation was 2/48. |
| Checkpoint and resume | PASS | `run.json`, append-only `metrics.jsonl`, `model.bin`, `meta.bin`, and identity-checked checkpoint directories; uninterrupted vs 3→5 resumed runs ended with the same model hash. |
| Real JSONL dataset compiler | PASS | `minicells-dataset`: NFKC/lowercase/space normalization, fixed punctuation mapping, unsupported-character rejection, 32-byte segmentation, hash sort/dedupe, deterministic split, domain-separated Merkle root, deterministic batch selection and membership proofs. |
| Research CLI | PASS | `minicells-lab dataset build/inspect`, `train`, `resume`, `evaluate`, `compare`, and `benchmark`; dataset-backed native training uses the compiled batch rather than silently falling back to synthetic data. |
| Batch identity / protocol V2 groundwork | PASS_CODE_PATH | `BatchIdentityV1`, `PendingV2`, V2 pending keys, and dataset `BatchSelection`/Merkle proofs are additive; V1 synthetic wire vectors remain unchanged. Full guest V2 wiring is reserved for the direct executor adapter. |
| Local PVM host surface | PASS_CODE_PATH | `minicells-pvm::LocalPvmHost` implements payload/results/storage/external-data/yield surfaces and verifies the real `service/artifacts/service.pvm` artifact. |
| Direct PVM PLUS/MINUS/ACCUMULATE | BLOCKED_EXTERNAL_ADAPTER | The pinned MiniJAM/Jambda public path only runs `VmEngine` through chain-oriented `RefineCtx`/`StateView`. Running `minicells-lab train --backend pvm` loads the real artifact and returns this typed blocker; no native fallback or fabricated PVM parity is reported. |
| Native ↔ PVM parity | BLOCKED_DEPENDS_ON_DIRECT_PVM | Native V1 golden vectors and generation transitions pass; PVM byte parity cannot be evaluated until the demonstrated executor adapter blocker is removed. |

Experiment 002 commands:

```text
cargo run --release --offline -p minicells-lab -- train --generations 1000 --checkpoint-every 250 --output .local/runs/native
cargo run --offline -p minicells-lab -- resume --generations 1000 --output .local/runs/native
cargo run --offline -p minicells-lab -- dataset build input.jsonl --output .local/datasets/echo-real-v1
cargo run --offline -p minicells-lab -- train --backend native --dataset .local/datasets/echo-real-v1 --generations 10
```

The PVM blocker is intentionally explicit: `service.pvm` was loaded and hashed as
`0xe1ebe71a3dabdab59135a21e7d71c6d28a2a3f5aaacf3693c872d3dbdf0d8bb4` by the
local harness, but the pinned executor still requires a chain `StateView`.

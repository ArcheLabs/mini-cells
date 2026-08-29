# MINI Cells V0.2 implementation status

Compatibility target: JAM semantics 0.7.2; MiniJamSpec v1 MiniJAM
`c4dec2db5d59ab40f8293335e29c94dd82b8eaf4`; Jambda exact gitlink
`fe67ecf5ccbe16b3490d73cc4d8b1e48eb7bea86`; JamScript is validated separately
against this MiniJAM pin. The machine-readable source of
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
| Service receipt / fresh deployment | POOL_ACCEPTED_BUT_LIFECYCLE_UNKNOWN | The fresh node returned the CreateService hash, proving transaction-pool acceptance. The old evidence used Alice although `--dev` genesis configures the local playground relayer, and `author_pendingExtrinsics` cannot prove rejection because it only exposes the ready queue. Inclusion/dispatch watch evidence is now required. |
| Worker / generation | BLOCKED_EXTERNAL | Because the canonical creation receipt never materialized, no receipt-derived Service ID exists for initialization. The available direct worker also reported `processed=0`; no PLUS/MINUS, 0→1, gas, multi-generation, or restart result is fabricated. |

## Commands and evidence

```text
cargo test --offline --workspace                         PASS
npm --prefix apps/web test -- --run                     PASS (5 tests)
npm --prefix apps/web run build                         PASS
./tools/build_service.sh                                 PASS (69,414-byte blob)
fresh rebuilt CreateService against MiniJAM 5947c           POOL_ACCEPTED_BUT_LIFECYCLE_UNKNOWN (hash returned; no watch evidence)
status-probe / worker finalization                       BLOCKED_EXTERNAL (no receipt-derived ID)
```

The external chain blocker does not weaken the browser integrity invariant:
the WASM runtime recomputes the canonical model hash before every forward pass,
and the Keeper independently verifies the finalized model before serving it.

## Experiment 002 — persistent local training status (2026-08-25)

## Production training repair and conditional gates (2026-08-26)

The Training result ABI is version 2 (156-byte maximum), carrying BASE and
candidate metrics. The production Refine/Accumulate path now implements
guarded SIGN-SPSA v2 with explicit `Keep`/`Plus`/`Minus` decisions, retained
model metrics, generation advancement on `Keep`, and pair invariant checks.
The real PVM trainer and shared accumulation envelope encoder are available in
`minicells-lab`/`minicells-pvm`, and the rebuilt service artifact is recorded
in `artifacts/local-training-gate/manifest.json`.

The fixed 512-generation Native gate was executed without a dataset. It
correctly stopped at the first failed gate: final fixed-probe loss was
573,303 from 607,901 (5.69% improvement), while the required best loss was at
most 90% of the initial loss. Solved-model regression passed. Per the
non-stop phase contract, PVM parity and fresh-chain E2E were not started.
Machine-readable evidence is in `artifacts/local-training-gate/`.

| Requirement | Status | Evidence |
|---|---|---|
| Persistent native trainer, arbitrary generation | PASS | `minicells-sim::trainer` drives the production no-std `refine`/`accumulate` entry points and completed a release 0→1000 run in 25.31 s (checkpointing at 250/500/750/1000); the recorded genesis evaluation was 1/80 tokens and the generation-1000 selected evaluation was 2/48. |
| Checkpoint and resume | PASS | `run.json`, append-only `metrics.jsonl`, `model.bin`, `meta.bin`, and identity-checked checkpoint directories; uninterrupted vs 3→5 resumed runs ended with the same model hash. |
| Real JSONL dataset compiler | PASS | `minicells-dataset`: NFKC/lowercase/space normalization, fixed punctuation mapping, unsupported-character rejection, 32-byte segmentation, hash sort/dedupe, deterministic split, domain-separated Merkle root, deterministic batch selection and membership proofs. |
| Research CLI | PASS | `minicells-lab dataset build/inspect`, `train`, `resume`, `evaluate`, `compare`, and `benchmark`; dataset-backed native training uses the compiled batch rather than silently falling back to synthetic data. |
| Batch identity / protocol V2 groundwork | PASS_CODE_PATH | `BatchIdentityV1`, `PendingV2`, V2 pending keys, and dataset `BatchSelection`/Merkle proofs are additive; V1 synthetic wire vectors remain unchanged. Full guest V2 wiring is reserved for the direct executor adapter. |
| Local PVM host surface | PASS | `minicells-pvm::LocalPvmHost` implements payload/results/storage/external-data/yield surfaces and drives the real converted `service/artifacts/service.blob` through Jambda's production `VmEngine`. |
| Direct PVM PLUS/MINUS/ACCUMULATE | PASS | Chain-free Jambda execution covers refine at PC 0 and accumulate at PC 5; status, PLUS, MINUS, exact accumulate storage/yield, and a full 0→1 transition are exercised against the tracked service artifact. |
| Native ↔ PVM parity | PASS | Status, PLUS, and MINUS outputs match the production native `minicells-runtime` byte-for-byte; ACCUMULATE matches exact storage/yield state and final META/MODEL/history for genesis→generation 1. |

Experiment 002 commands:

```text
cargo run --release --offline -p minicells-lab -- train --generations 1000 --checkpoint-every 250 --output .local/runs/native
cargo run --offline -p minicells-lab -- resume --generations 1000 --output .local/runs/native
cargo run --offline -p minicells-lab -- dataset build input.jsonl --output .local/datasets/echo-real-v1
cargo run --offline -p minicells-lab -- train --backend native --dataset .local/datasets/echo-real-v1 --generations 10
cargo test --offline -p minicells-pvm
cargo run --offline -p minicells-lab -- train --backend pvm
```

The direct PVM boundary is separate from the fresh-chain lifecycle issue: the
converted service blob is decoded by Jambda's production predecoder and
executed by `VmEngine<InterpBackend>` with only the local host surface. The
historical CreateService hash proves pool acceptance, but its missing watcher
stream leaves inclusion and dispatch status unknown; it also used Alice rather
than the configured development ingress relayer.

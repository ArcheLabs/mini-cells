# MINI Cells V0.2 implementation status

Compatibility target: JAM semantics 0.7.2; MiniJAM
`c980212cb7fa7d0d178e7c173169f41878a27b81`; Jambda
`90d93f79d9a899e07667e7734fbb9c778618b831`; JamScript
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
| Rust verification | PASS | `cargo test --offline --workspace` passed: core, protocol, simulator, chain, Keeper auth/model tests, and WASM ABI tests. |
| Web verification | PASS | `npm test -- --run` passed (5 tests); `npm run build` passed with current WASM artifact. |
| Python validation | BLOCKED_EXTERNAL | `python3 -m pip install --upgrade pip setuptools wheel` could not reach PyPI (`ConnectTimeoutError: pypi.org`); no Python result is claimed. |
| Fresh service deployment | PARTIAL_EXTERNAL | A new service was submitted from the rebuilt blob on a fresh local `--dev` chain (service id 0 observed at finalized RPC). Initialization/status Work could not finalize with the available direct worker setup. |
| Worker signature / generation | BLOCKED_EXTERNAL | Direct worker follow-up execution remains unavailable in this local chain; no PLUS/MINUS, 0→1, gas, multi-generation, or restart result is fabricated. |

## Commands and evidence

```text
cargo test --offline --workspace                         PASS
npm --prefix apps/web test -- --run                     PASS (5 tests)
npm --prefix apps/web run build                         PASS
./tools/build_service.sh                                 PASS (69,414-byte blob)
fresh local service submission                            PASS (service id 0 observed)
status-probe / worker finalization                       BLOCKED_EXTERNAL
```

The external chain blocker does not weaken the browser integrity invariant:
the WASM runtime recomputes the canonical model hash before every forward pass,
and the Keeper independently verifies the finalized model before serving it.

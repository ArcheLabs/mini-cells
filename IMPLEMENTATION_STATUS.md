# MINI Cells V0.1 MiniJAM / Keeper and JamScript State Plane Status

Compatibility baseline: MiniJAM `1dceda20d501b19207fc33252f180e658dc064d7`, Jambda `0fb3591ed6f3a7479e3fa0c672711aa80827d486`, JamScript `82e88087c86442f81ac5bf526f4c5b57cba454b5`, JAM semantics 0.7.2.

The machine-readable source of truth is `artifacts/implementation-status.json`.

| Phase | Status | Evidence |
|---|---|---|
| 0 — inspect and pin | PASS | Current SDK ABI, converter, target/toolchain lock, native launcher and JamScript guest path inspected |
| 1 — Rust workspace/core | PASS | `cargo test --offline --workspace` core/protocol/runtime/sim suites |
| 2 — protocol/runtime | PASS | Deterministic codecs, rings, SPSA, genesis, runtime tests |
| 3 — simulator/golden | PASS | 5 simulator tests and checked-in golden vector |
| 4 — PVM guest/artifacts | PASS | Official PolkaVM target, ELF/blob/PVM converter artifact manifest |
| 5 — CLI | PASS | Deploy/status/probe/infer/train/replay commands compile |
| 6 — direct chain / Keeper | PASS_CODE_PATH | Reusable chain client, WorkPackage builder, Bulletin verification, direct CLI, Keeper recovery/scheduler/SSE and worker gateway compile |
| 7 — web | PASS | Keeper-only UI; Vitest and Vite production build pass |
| 8 — JamScript State Plane M0 | PASS | `service-state-plane` facade, ProofState adapter, forged-root rejection test and architecture document pass |
| 9 — real MiniJAM | PARTIAL | Existing local worker finalized inference and candidate Refine; paired generation remains blocked by repeated bad-signature response |
| 10 — full verification | PARTIAL | Rust/JamScript/Web/build checks pass; Python pytest unavailable in environment |

Known blockers are recorded explicitly: the environment has no `pytest` module, and the
current native local chain accepts the candidate Refine report but leaves the work
operation in `tracking_work` because the local worker repeatedly receives
`Transaction has a bad signature` while submitting the follow-up chain transaction.
The PLUS pending record is finalized on service 1677269814; the MINUS side and paired
generation are therefore externally blocked. No result is marked PASS without command
evidence.

V0.1 evidence: `cargo test --offline --workspace` PASS (including direct Bulletin
round-trip); `cargo test --offline -p service-state-plane` PASS; Keeper/CLI direct
crate checks PASS; `npm --prefix apps/web test && npm --prefix apps/web run build`
PASS. The remaining real-chain generation failure is the pre-existing external
worker signature failure, not a local compilation or state-provider failure.

Python follow-up: `python3 -m pip install -e '.[dev]'` is blocked because this
repository has no `setup.py` editable backend; the non-editable install stalled
on dependency processing and was cancelled. `python3 -m pytest -q` therefore
remains a demonstrated environment blocker (`No module named pytest`). A direct
CLI status probe against the existing service also rejected its legacy state as
`invalid MetaV1`; this is recorded as an incompatible pre-existing chain state,
not silently treated as a direct-path pass.

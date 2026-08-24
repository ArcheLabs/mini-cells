# Keeper API

`minicells-keeper` exposes `/healthz`, finalized `/v1/status` and
`/v1/history`, the authenticated wallet flow (`/v1/auth/challenge`,
`/v1/auth/verify`, `/v1/auth/me`, `/v1/auth/logout`), authenticated `/v1/model`,
authenticated `/v1/events`, and the authenticated `/v1/verify/infer` debug path.
Normal browser inference never submits a Work item. The model endpoint fails
closed unless finalized `META` and `mc:v1:model` have the exact canonical length,
decode, and hash. SSE starts with a current snapshot, emits `snapshot`,
`chain`, `training`, `generation`, `model`, and `error` events, sends heartbeats,
and sends a fresh current snapshot when a bounded subscriber falls behind.

Training is serial and operator-controlled. Set `MINICELLS_OPERATOR_ACCOUNT` to
enable the authenticated `/v1/admin/training/start`, `/pause`, and `/step`
routes; an unset variable makes every web session non-operator. Sessions are
opaque 256-bit in-memory cookies (12-hour TTL), and challenges are one-use
five-minute records. Set `MINICELLS_WEB_ORIGIN` to the exact browser origin and
`MINICELLS_COOKIE_SECURE=1` in HTTPS deployments; wildcard credentialed CORS is
never enabled.

Trust boundary: MiniJAM remains the canonical source of model state, while the
Keeper is the V0 trusted finalized-state gateway for browser clients. The
browser/WASM runtime independently checks `H(modelBytes) == modelHash` for the
delivered response and executes the deterministic Rust kernel locally. It does
not prove that the Keeper's supplied hash came from MiniJAM; that requires a
future State Plane/proof-RPC path. UI language therefore describes “local model
integrity verified”, not a trustless chain proof.

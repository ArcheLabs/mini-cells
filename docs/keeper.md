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

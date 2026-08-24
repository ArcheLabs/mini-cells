# Keeper API

`minicells-keeper` exposes `/healthz`, `/v1/status`, `/v1/history`,
`/v1/training`, `/v1/infer`, `/v1/infer/:id`, and `/v1/events`. The SSE stream
starts with a snapshot, emits state/training/inference events, sends heartbeats,
and emits `resync` when a bounded subscriber falls behind. Training is serial:
it submits at most one side at a time, observes finalized state, and retries
with bounded exponential backoff. On restart it derives the next action from
finalized `META`, `PENDING_PLUS`, and `PENDING_MINUS`, so PLUS-only, MINUS-only,
and pending-generation recovery are safe.

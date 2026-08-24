# MINI Cells V0.1 architecture

The canonical path is direct MiniJAM: the CLI or Keeper reads one finalized
context, loads `META` and `MODEL` as ordered external data, builds a WorkPackage
with the reusable MiniJAM builder, stores and verifies the Bulletin bundle, and
submits it with `minijam-chain-client`. Finalized state is the only source of
truth. The browser is a view of the Keeper and does not sign or poll storage.

The Keeper is intentionally non-canonical. It provides recovery, serialized
training scheduling, inference request tracking, SSE snapshots/events, and a
local Bulletin HTTP gateway for workers. Native learning quality is not claimed
by this release; only deterministic execution and state-transition integrity are
part of the acceptance surface.

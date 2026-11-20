# MA-001 preliminary engineering results

**Retrieval: BLOCKED in the local cache-only run. Scientific status: NOT_EVALUATED.**

- The engineering fixture has 10 cells, 120 single-cell events and one
  additional event projected to two cells. All 120 fixture writes reached the
  expected owner; transaction tests separately cover Inbox, binding, explicit
  priority, tombstones and injected rollback.
- A fresh Python process reopened the SQLite registry and recovered all 122
  address records unchanged. This checks index persistence; semantic routing
  across sessions has not passed.
- The pinned E5 encoder was not cached. No fallback was used. Development
  thresholds were not frozen and no holdout retrieval metrics were produced.
- The fixture's event texts remain short and formulaic. A meaningful
  descriptor-versus-raw comparison requires richer authored event bodies and
  descriptors before the formal evaluation.

See [manifest](artifacts/manifest.json), [readback](artifacts/persistence.json)
and [blocked run](artifacts/decision.json). The manifest's Git commit identifies
the repository base; the experiment implementation was still uncommitted at
execution.

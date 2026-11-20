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

## Enriched fixture — completed CPU evaluation

Pinned multilingual E5, 24 development queries and 56 holdout queries.
Thresholds were saved before holdout evaluation. See
[configuration](artifacts/enriched-cpu/frozen.json),
[all metrics](artifacts/enriched-cpu/metrics.json) and
[model identity](artifacts/enriched-cpu/encoder.json).

| Arm | Recall@3 | Cross-category Recall@3 | Multi-cell AllRelevant@3 | No-match FPR |
|---|---:|---:|---:|---:|
| BM25 | 59.5% | 72.7% | 54.5% | 0/6 |
| Dense | 59.5% | 81.8% | 72.7% | 0/6 |
| Entity | 64.3% | 72.7% | 54.5% | 0/6 |
| Hybrid Raw | 88.1% | 90.9% | 81.8% | 0/6 |
| Hybrid Descriptor | 82.1% | 86.4% | 72.7% | 1/6 |

**Decision: PARTIAL PASS on this synthetic fixture.** Write routing and
fresh-process persistence passed. The unbound Session C example returned
all three expected Cell IDs from the two writer sessions. The persistence
check and the semantic query were separate; this does not establish a
fresh-process end-to-end routing result.

Hybrid Raw gave the strongest recall but missed the 90% overall gate.
Descriptor did not improve it: paraphrase Recall@3 dropped from 7/11 to 5/11,
and multi-cell exact coverage from 9/11 to 8/11. Both hybrids routed one
of eight underspecified queries. No-match rates have only six holdout
examples, so 0/6 does not establish a population false-positive rate below 10%.

Raw notes reuse per-Cell context, while descriptors are shorter and contain
some explicit aliases. The difference measures these fixture representations,
not whether descriptors universally hurt. Retain the holdout and investigate
coverage or rejection failures on development data before evaluating any
new design on a new independent holdout. No Titans integration is warranted
on this evidence.

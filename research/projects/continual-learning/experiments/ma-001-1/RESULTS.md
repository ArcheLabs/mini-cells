# MA-001.1 — Development results

The frozen Raw Hybrid returned exactly the expected set on 20/24 development
queries. The three positive cases with missing cells (q018, q029, q044) all
had the target Cell indexed and ranked within the top three; the frozen
threshold excluded them. This is not evidence that all relevant event records
were retrieved: the fixture only has Cell-level ground truth. The record
trace shows each candidate's raw and descriptor Dense scores and Dense,
BM25 and Entity ranks. Do not infer missing descriptor information or an
embedding failure from Cell labels alone.

| Variant | Exact development sets | Weak probes routed |
|---|---:|---:|
| Raw Hybrid, frozen | 20/24 | 3/3 |
| Raw Dense + BM25 | 19/24 | 3/3 |
| Raw Dense + Entity | 17/24 | 0/3 |
| Raw BM25 + Entity | 16/24 | 0/3 |
| Raw + structured entity metadata | 20/24 | 3/3 |

Ablation thresholds other than the frozen baseline were fitted on the
same 24 cases; these are descriptive, not generalization results. The extra
no-match and ambiguous probes abstained under the baseline. The weak
probes routed despite insufficient evidence: retention decision → two
Cells, account → finance, and model update → linear models. The score and
margin are in the trace; no margin rule was chosen. Retain the frozen Raw
Hybrid and Descriptor as baselines.

An independent reader process verified persisted SQLite and embedding
hashes, checked address row order, encoded only the query, and found the
three expected Cell IDs across Sessions A and B. It did not rebuild Cell
records or re-encode passages. This is one end-to-end engineering scenario,
not a routing success rate.

Before MA-002, add event-level relevance labels to development diagnostics
and investigate evidence for resolving weak matches. The independent
180-query confirmation set is specified but not built, opened, or run.

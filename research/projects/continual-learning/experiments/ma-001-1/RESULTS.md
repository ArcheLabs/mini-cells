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

## Event-level follow-up

The [development annotations](event-labels.json) and
[joined trace](artifacts/event-diagnosis.json) distinguish primary records
from supporting context. These annotations were added after observing
development errors; they are diagnostic labels, not independent evidence.

| Query | Primary record | Dense rank | BM25 rank | Winning record in the missing Cell |
|---|---|---:|---:|---|
| q018 | Portable Cell | 21 | 1 | Session reset (supporting) |
| q029 | Source map | 4 | 11 | Async call (not annotated relevant) |
| q044 | Execution trace | 13 | 2 | Execution trace (primary) |

All three primary records exist and had nonzero Dense and lexical scores;
none matched an entity. q018 and q029 show a record-level ranking problem
in addition to final rejection. q044 isolates rejection of the primary
record even though its Cell is ranked second. The fixture is insufficient
to attribute either ranking error to encoder capacity rather than content.

A monotone minimum score alone cannot admit q018 or q029 while rejecting
all three weak probes: each weak probe scores higher than both valid
queries. A margin minimum alone cannot admit q018 while rejecting the
account probe (margin 0.00384 vs 0.00269). These counterexamples do not
rule out joint rules. Do not fit a joint rule to just three probes.

Next: build paired development cases that preserve topic words but vary
whether the query identifies an actual event, and assess event-level
evidence rather than merely lowering the Cell score threshold. MA-002
remains unopened.

## Topic-matched development probes

Ten authored pairs share an event term. The valid query seeks documented
material; its paired weak query asks for unsupported specifics or requires
missing context. These are adversarial development probes, not a representative
estimate of false-positive rate.

The frozen Raw Hybrid routed to the intended Cell on 10/10 valid probes, but
routed 10/10 weak probes. No pair passed both routing requirements. The valid
query had a strictly higher top RRF score on only 1/10 pairs. Its Dense score
against the annotated target event was higher in 10/10 pairs. That comparison
uses a known target and its matched counterpart; it does not establish a
deployment-ready threshold or show that the target can be identified without
labels. See [paired traces](artifacts/paired-probes.json).

Keep the current baselines frozen. A development candidate should compare
query-to-record evidence among retrieved candidates, including hard competing
records, and test whether it can reject unsupported requests without losing
valid paraphrases. Do not advance to MA-002 confirmation yet.

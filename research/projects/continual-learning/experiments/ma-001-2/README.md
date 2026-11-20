# MA-001.2 — Event-Level Evidence Routing

Owner: Continual Learning. Development only; MA-002 remains sealed.

Preserve MA-001 Raw Hybrid and its published threshold as a Cell baseline.
For evidence routing consume a wider record-level candidate set before
Cell rejection. Report grade-3 event candidate recall before reporting
evidence quality. Gold event IDs and grades are evaluation data only.

The prototype accepts normalized (subject, relation, object) claims. Exact
coverage supports a facet, same subject alone is related, and absent subject
is ungrounded. It can combine covered facets from different Cells. If any
required facet is uncovered it does not return primary Cells. RELATED does
not assert that the Cell lacks the requested information.

## Development candidate audit

Run `python diagnose.py E5_SNAPSHOT MA001_1_INDEX NEW_OUTPUT`. Uses the
existing Raw Hybrid channel scoring before Cell aggregation and rejection.
The original baseline and cutoff are untouched. Gold labels join only after
retrieval. The 23 observed development queries comprise three annotated
misses and ten topic-matched pairs. Unjudged records remain null.

| Measure | Result |
| --- | --- |
| Annotated direct events in first 8 | 12/14 |
| Annotated direct events in first 16 | 14/14 |
| Annotated direct events in first 32 | 14/14 |
| Dense top-ranked candidate annotated direct | 11/23 |
| Dense top-ranked candidate unjudged | 2/23 |

See [candidate traces](artifacts/candidate-traces.json) for channel scores,
ranks and graded matches. This establishes candidate coverage of the
labeled positives on this development subset, not a support classifier.
The structured prototype still takes pre-normalized claims; grade-0
annotations and a pinned semantic support reranker are pending. MA-002
has not been accessed.

## Structured evidence audit

Run `python compare_structured.py` against the saved top-16 traces.
The five selected cases use authored normalized claims and separate grades
0–3. Candidates and their order are unchanged. On q018 the Dense winner
is supporting Session Reset; on q029 it is unjudged Standard Library; on
q044 it is direct Debug Symbols but a single record covers only one facet.
The structured replay supports all three valid queries and identifies both
Cells on q044. It marks the two unsupported probes RELATED with no primary
Cell. Missing claim annotations are not negative labels. The structured
router still does not infer grade 2; grade 2 here is a human annotation.

This is an oracle-input diagnostic with partial claim coverage and five
cases selected from observed development errors. It is not evidence of
automatic query parsing or a fair accuracy comparison with Dense. The
next comparison needs a pinned automatic text scorer on the same candidates
and an independent development annotation pass before any threshold choice.
No MA-002 data were accessed.

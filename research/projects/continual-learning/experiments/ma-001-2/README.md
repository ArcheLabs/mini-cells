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

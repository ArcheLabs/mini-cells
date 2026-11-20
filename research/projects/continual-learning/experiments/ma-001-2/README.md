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

Next implementation work: generate actual Raw Hybrid record candidates and
record each channel's score/rank; author independent four-grade event
annotations including auxiliary evidence and competitors; compare Dense
and structured claim coverage plus an explicitly pinned cross-encoder.
The prototype does not yet extract claims from natural language, recognize
grade-2 auxiliary evidence, or establish end-to-end evidence quality.

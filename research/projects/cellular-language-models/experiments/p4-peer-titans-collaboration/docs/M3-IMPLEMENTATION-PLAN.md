# P4-M3 implementation plan

Base de4f6c0 on research/p4-m11-m2-preparation. Scope is source and checkpoint audit, experiment design and Smoke preparation. This stage ends at the review checkpoint in specification §17.

| Workstream | Reuse | Thin adapter | New work | Open gate |
|---|---|---|---|---|
| Peer Titans | Frozen M3 learn/recall/restore and M2 strict persistence | Copy two read-only state streams into a batched view | Paired domain generator, query decomposition, ordered concatenation and ablations | Composite recall, independent branch contribution and fresh-process composition |
| SSM–Titans | EPCLM-004 M2 E2 Mamba-1 implementation and frozen M3 | E2 pre-logit hidden tap and common causal event interface | Sequence tasks, fusion head, split manifests and S0–S5 comparisons | Freeze an approved SSM source and initialization, then verify branch independence and resource cost |

Use 3 seeds × 5 episodes per configuration for Smoke. Paired arms receive the same maps, events and query order. Use distinct seed namespaces for development, fusion fitting, validation and sealed confirmation. Retain failed episodes. Only begin Screening after the Smoke evidence is reviewed. Screening proposes 20 paired episodes per configuration across at least 3 seeds. Confirmation has a separate frozen protocol with at least 5 seeds × 100 episodes.

For statistical analysis, record Q1 domain-level accuracy, Q2 joint exact match and component accuracy, Q3 false accepts, old-knowledge retention and cost. Bootstrap paired episode differences, report per-seed summaries and intervals, and stratify by distance and distractors where applicable. Gate A and Gate B are engineering gates. Gate C requires evidence about capacity, compute and causal branch contributions. Smoke does not establish synergy.

Dependencies: [Peer plan](M3A-PEER-TITANS-PLAN.md), [SSM plan](M3B-SSM-TITANS-PLAN.md), [cost estimates](M3-RESOURCE-ESTIMATE.md), [deviations](M3-IMPLEMENTATION-DEVIATIONS.md).

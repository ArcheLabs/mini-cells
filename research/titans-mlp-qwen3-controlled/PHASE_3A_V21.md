# Phase 3A v2.1 — Dataset correction

The v2 smoke and its checkpoint remain historical artifacts. The full v2
design is archived in PHASE_3A_V2_SPEC.md.

The v2.1 generator samples from all 35 complementary pairs of four-of-eight
relation subsets. Every complementary pair contributes one count to each
relation. Shuffling complete cycles and taking complementary pairs for the
remainder yields exact marginal balance and diverse entity profiles.
The audit returns relation counts, combination counts and pairwise
cooccurrence counts. Entity and value identifiers are disjoint across splits.

Two seen and two held templates per role preserve entity-before-relation
and value-after-relation ordering. Held-out predicates use separate
surface forms. The initial wording review focused on grammatical embedded
questions and preserved the original relation semantics. Tokenizer-level
verification and real-model multi-view smoke remain pending.

Status: dataset implementation and surface-span tests passed. Multi-view
optimization, explicit hard negatives, separate loss logging, early stopping,
baseline runs, formal five-seed evaluation and Kaggle publication remain pending.

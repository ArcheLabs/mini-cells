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

## Kaggle entry

Paste kaggle_phase3a_v21.py into a Python notebook with one T4, Internet,
and GITHUB_TOKEN. It runs the validation-only Smoke, using two seen
views, sampled Query/Key view combinations, additive logit weighting
for same-entity or same-relation negatives, and separate QK/Value losses.
The expected result is SMOKE_ONLY. It does not open the formal test split.
The repository publisher records the result, adapter and SHA256 manifest.
Use publication.json to check the verified GitHub branch and commit.

For a local CPU run:
```sh
python phase3a_v21.py /path/to/verified/qwen/snapshot /path/to/new-output
```

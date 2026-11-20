# MA-001.3 — Event-Level Evidence Verification

Continual Learning, development only. The 18 authored facts specify actors,
actions, context and outcomes, including close competitors and explicit
non-support. Each of six queries has a complete 0–3 judgment over all 18
events (108 judgments). These are synthetic authored annotations, not an
independent rater study. Discovery uses a fixed Dense + lexical RRF recipe
on this fixture; it is separate from the published MA-001 Raw Hybrid.
All comparisons use the same 16 candidates per query. No thresholds were fit.

Run: `python run.py results.json --e5 /path/to/pinned/e5`.
The recorded CPU run used the MA-001 snapshot of multilingual E5 small at
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Optional `--nli` takes a local
snapshot of DeBERTa NLI at `fa2804872c3b4bd748f38c0185cc85775361e735`; output
preserves all three label probabilities per authored claim and candidate.

## Initial results

- Candidate coverage: 6/6 labeled direct records.
- Dense and lexical each ranked direct evidence first on 2/6 queries. For the
  restart question, both ranked a supporting record ahead of direct evidence.
- Structured replay matched 6/6 expected statuses. It joined two events in
  one Cell and two events across Cells. It returned no primary Cells when
  only one of two required facets was covered.
- Structured input was authored on both sides. This is a controlled coverage
  check; it does not validate automatic fact extraction or give a fair
  head-to-head classifier accuracy comparison.
- NLI has not been run on this new fixture locally: its pinned snapshot is
  still missing weights. Prior MA-001.2 scores cannot be transferred to it.

See [candidate scores and judgments](results.json). MA-002 remains sealed;
Titans integration remains out of scope.

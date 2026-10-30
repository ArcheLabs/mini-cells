# EPCLM-001 — ~1.9M native-only first round

The active scope is **T1 / X3 / X4 only**, defined by
[protocol v0.2](PROTOCOL.v0.2.md) and [machine-readable configuration](protocol.json).
The original [v0.1 draft](PROTOCOL_DRAFT.md) is preserved unchanged; its broader
E0-A execution scope is superseded. 30M is **DEFERRED / CONDITIONAL SCALE
VALIDATION**, with no automatic loader or CLI selection in round 1.

No training, formal benchmark or 30M execution is authorized by this scope
amendment. Historical E0 results and checkpoints remain unchanged.

Order: native asset preflight → numerical/cache correctness → small event
microbenchmark → gated formal competitive benchmark. H1/H2/H3 remain separate.
See [scope review](reports/scope-v0.2/REVIEW.md) for availability, cost and blockers.

## Engineering tools

- X3/X4 per-stage/per-iteration A0 KV cache and A1 preallocated variant.
- Modern T1 GQA KV cache, original full-forward reference untouched.
- Four event executors and deterministic synchronous event/cost records.
- Native-only read-only discovery; hashes computed automatically.
- Existing report-only GitHub return with the existing `GITHUB_TOKEN` secret.

The CPU notebook is an optional engineering launcher, **not** an instruction
to run this amendment or a formal experiment. Its default scope excludes 30M.
`audit-assets` performs read-only discovery. `engineering` uses clearly labelled
tiny random-weight fixtures; it never supplies missing trained-model evidence.
`bench-event` / `bench-incremental` are development tools, not frozen formal
results. `bench-competitive` remains blocked pending assets and shared quality
conditions. `publish-existing` retries result return without rerunning models.

```sh
python research/projects/cellular-language-models/experiments/epclm-001-incremental-execution/run.py audit-assets --output /tmp/epclm001-assets
```

Missing checkpoints mean absent from the recorded search roots, not globally
nonexistent. Matching tokenizer size or source defaults does not prove common
training/evaluation provenance. See [freeze status](FREEZE.md) and
[claim limits](WHAT_THIS_DOES_NOT_PROVE.md).

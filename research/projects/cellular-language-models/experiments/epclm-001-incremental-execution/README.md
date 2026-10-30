# EPCLM-001 — ~1.9M native-only first round

The active model scope is **T1 / X3 / X4 only**, defined by
[protocol v0.2](PROTOCOL.v0.2.md) and [machine-readable configuration](protocol.json).
The original [v0.1 draft](PROTOCOL_DRAFT.md) is preserved unchanged; its broader
E0-A execution scope is superseded. 30M is **DEFERRED / CONDITIONAL SCALE
VALIDATION**, with no automatic loader or CLI selection in round 1.

The v0.2 scope amendment did not authorize training. The new v0.3 amendment
authorizes only the controlled retraining exception described below; formal
E1 benchmarks and 30M remain unauthorized. Historical E0 results and checkpoints
remain unchanged.

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


## Engineering closure after review of 1d527228

T1 now uses parallel causal prefill, with offset-aware chunk continuation and
GQA KV construction checked against full forward and serial decode. X3/X4
retain the original per-iteration sequential reference path.

Event correctness and performance are separate passes. Correctness retains
hashes, traces and synchronized phase diagnostics. Performance disables those
observations and internal synchronization; outer host and CUDA-event intervals
include actual queue/packing/compute/commit costs. GPU-event intervals are not
labelled as summed kernel-only time. Detailed phase timings remain diagnostic,
not a performance decomposition.

Checkpoint discovery inspects bounded Torch archive metadata rather than model
names in paths, includes the original notebook's output/cache roots, and keeps
legacy/unrecognized files unclassified. Training source stays UNKNOWN unless
recorded in the checkpoint or bound by a matching checkpoint-hash sidecar;
the current adapter source is recorded separately. Nearby tokenizer/corpus
files are accepted only through saved hashes, not directory proximity.

`validation_policy.json` fixes engineering device/dtype criteria before the
checks, and `validation_inputs.json` provides a fixed 128-token diagnostic
sequence. Unlisted precision/device combinations stop rather than inheriting
FP32 tolerances. These fixed token IDs are not a model-quality corpus.
See `reports/engineering-closure-001/REVIEW.md` for the full test record and
unresolved trained-asset/low-precision/formal-benchmark gates.

## v0.3 controlled retraining exception

The [user amendment](PROTOCOL.v0.3.md) authorizes only T1 → X3 → X4,
seed 91002, 10M tokens / 10,000 original loop steps per model. It does not
reopen architecture search, seed expansion, 30M, or formal E1-C. Original
v0.1/v0.2 protocols and engineering reports remain unchanged.

Use `epclm001_controlled_retrain.ipynb` in Kaggle, enable one GPU/Internet and
the existing `HF_TOKEN` and `GITHUB_TOKEN` secrets, then Run All. No manual hash,
seed or source registration is needed. The existing Native CLM artifact repo
`archelabsxyz/native-clm-v0` is reused under a new controlled-retrain namespace;
R0 verifies write access and fresh-download readback before GPU training.
No new repository is silently created. Git receives reports/indexes only.

The CLI routes `retrain-preflight`, `retrain-run`, `retrain-verify` and
`retrain-publish` through `run.py`. `retrain-run` performs R0 automatically.
Reusing the retained output resumes matching state; it never intentionally
restarts a completed model. Failed publication can be retried without training.
Controller metadata and verified resume bytes are retained remotely for a later
cloud session. An unresolved active reservation remains charged, failing closed
rather than resetting the resource allowance after a lost session.

The original single-model replication worker, sampler, loss, initialization,
AdamW and FP16 GradScaler remain in use. Wrapper observation forces the historical
optimizer/scaler defaults explicitly and saves after complete update boundaries.
Extra 100-step resume saves do not add evaluations or updates; the original
five quality checkpoints are retained. Actual AdamW calls are counted separately
from loop steps. Scaler skips trigger review, never extra training to compensate.
The original NumPy trapezoid report calculation has an equivalent reporting-only
fallback for older NumPy; it does not alter training.

The 7,200-second cap conservatively charges worker wall time. Validation and
checkpoint IO are recorded separately. A running update may finish before safe
save; a stuck process has a bounded 120-second termination grace. Actual observed
time is retained, including any overrun, and never treated as a budget increase.
Only one GPU worker may hold the shared lock. CPU synthetic tests do not use the
real baseline data or train a research model.

`ASSET_AVAILABLE` requires strict completed-state checks, inference export,
HF upload, forced fresh readback and independent-process strict loading. It is
an asset-availability result, not H1/H2/H3 support. Full resume and inference
weights stay outside Git; Save Version also retains the local output. R4 can
only request the separately authorized inference benchmark.

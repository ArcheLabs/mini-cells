# CLM-001-R3b — deferred diagnostics

Read [the execution specification](DIAGNOSTIC_EXECUTION_OPTIMIZATION.md),
[audit](AUDIT.md), [versioned plan](config/diagnostic-execution-v1.json),
[resource review](reports/diagnostic-optimization/PERFORMANCE_REVIEW.md).
Original R3 model/training/Admission code and historical results remain unchanged.

## Two entries, independent identities

- `notebook.ipynb`: automatic source/CPU evidence and permission/data preflight.
  This engineering release reports HOLD after preflight; it never repeats the
  accepted Pilot or starts unapproved500-step training. Event decisions, every
  checkpoint and verified HF ACK remain synchronous; general explanations are
  deferred. No pytest or fault suite runs on Kaggle.
- `analysis-notebook.ipynb`: **Run All** performs the complete read-only 102-task
  analysis of completed run directory `33734fdd1558`, training source
  `a7792ef30582dfcaf324cca7e5245c8430df48b2`, immutable HF revision
  `828585f1ff93420aa4485c725211ed28f70d15f3`. It automatically reuses verified
  private HF results, batches at task boundaries, aggregates, plots and publishes.
  No Pilot, training update, manual stage approval or Confirmation access.
  [Run-specific identity, coverage and measured performance](reports/runs/33734fdd1558/README.md).

Existing `HF_TOKEN` / `GITHUB_TOKEN`, same private repo
`libingjiang/minicells-research-checkpoints`. No Add Input, checkpoint download,
old Kaggle version URL or reconstructed asset/probe is required. The prior user
manual acceptance is recorded honestly; removed Gate C is not reinstated.

## Independent analysis CLI

The Notebook uses `full-analysis`: fixed Quality39 + Functional42 + Action21,
with COMPLETE / UNAVAILABLE / BLOCKED / PENDING counts always summing to102.
Development is never replaced by Calibration. A session boundary can leave
PENDING; the next Run All resumes. A terminal pass has no PENDING cells.

Advanced legacy commands remain available through `python r3b/analyze.py --help`:
`list-checkpoints`, `analyze-checkpoint`, `analyze-batch`, `aggregate-analysis`,
`publish-only`. They do not change the Notebook's full-analysis default. Use an existing trusted
GitHub checkpoint-index or pinned local reference; arbitrary URLs are rejected.
Task budgets apply between tasks without killing processes. The private remote
index, not local output existence, determines verified completion on reentry.
Stage quality evaluates exact declared policies; functional retains original
local autograd, MC and bootstrap; action reuses original STOP/CTX definitions.
All are FROZEN_INITIAL_STATE measurements, not recomputed online state claims.

Tasks bind training run/source/runtime/asset/tokenizer/protocol, exact checkpoint
and event revision, analysis source/protocol, policies/roles/window/seed/probe.
Step400 pre selects revision0; post requires the final committed complete Event
chain. Missing history is NOT_EVALUATED, never zero or a latest-state substitute.
New analysis protocols/sources produce separate immutable task IDs. Public
summaries have sample counts and single-seed caveats; raw scores/probes stay HF.

`aggregate-analysis`/`publish-only` only retrieve verified task results and draw
figures; they never run diagnostics or training. Aggregate namespace and HF index
are separate from training latest. Publication retry cannot relaunch training.

## Release status

Production parameters remain1,075,492 per arm (1,070,819 requires_grad). Original
seed151001301,500×8×64=256,000 live tokens per arm, all semantic Shadow costs and
original data boundaries remain. Formal authorization is false. New GPU delta,
mandatory Event budget, <=2h training performance and total analysis GPU cost are
NOT_VERIFIED. No full Pilot or formal training is executed by this work.

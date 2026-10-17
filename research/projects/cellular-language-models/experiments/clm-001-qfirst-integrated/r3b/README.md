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
- `analysis-notebook.ipynb`: independent read-only analysis source. Defaults to
  listing the accepted Pilot's exact pinned HF checkpoints. It starts no GPU
  task or optimizer update. An approved bounded task can use the same entry,
  `analyze-checkpoint`, QC-R3b Step3/revision0, Calibration Action, max-tasks1.
  That task is an engineering measurement, not formal Development evidence.

Existing `HF_TOKEN` / `GITHUB_TOKEN`, same private repo
`libingjiang/minicells-research-checkpoints`. No Add Input, checkpoint download,
old Kaggle version URL or reconstructed asset/probe is required. The prior user
manual acceptance is recorded honestly; removed Gate C is not reinstated.

## Independent analysis CLI

`python r3b/analyze.py --help` exposes `list-checkpoints`, `analyze-checkpoint`,
`analyze-batch`, `aggregate-analysis`, `publish-only`. Use an existing trusted
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

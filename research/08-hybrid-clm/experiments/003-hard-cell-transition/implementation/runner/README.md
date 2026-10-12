# Experiment 003 — Hard Cell Transition

Status: **registered, not run**. Read `protocol.json` before changing configuration.
The canonical entry is `notebooks/experiment.ipynb`; implementation lives in
`src/minicells/{models,routing,training,utils}`. `config.yaml` is JSON-compatible
YAML 1.2. The thin scripts use the same runner and evaluator as the notebook.

## Registered comparison

Four capacity matched arms: four fixed residual Cells, Global Router top-2,
Soft Cell Router, Hard Cell Transition. Each sees the same train-only BPE,
pinned TinyStories revision, seeded windows, 10M training tokens and 32k
validation tokens. Seeds 42/123/456 are **development**, already observed in
earlier work; they cannot be advertised as untouched confirmation seeds.

Hard training is 1,250 soft steps at T=2, 1,250 soft annealing steps from T=2
to T=0.1, then 2,500 genuinely selected-only hard steps at T=0.1. Percentages
refer to token budget rather than wall time, keeping stage exposure paired.
STOP is absorbing. At the fourth hop, termination is labeled FORCED_STOP.

The chosen next probability uses `gate = 1 + p_selected - stop_gradient(p_selected)`.
Its forward value is one; its derivative provides a **biased selected-value
straight-through estimator**. It has no unselected Cell outputs/gradients and
does not reproduce a dense mixture gradient. A differentiable survival cost
encourages fewer visits. This is a concrete hypothesis to evaluate, not a claim
that the train/inference mismatch is already solved.

## Compute and latency

Every arm uses SDPA **math** during training, quality, profiling and timing.
Executed mm/addmm/bmm input shapes are observed for every validation window;
the counter includes actual attention GEMMs and repeated Cell execution.
The metric is **executed GEMM FLOPs**, 2 FLOPs/MAC, excluding normalization,
softmax, activation, dispatch and backward. It is not total hardware FLOPs.
An uncovered fused attention operation makes the efficiency gates incomplete.
Latency uses an identical resident FP16 batch, five warmups, twenty forwards,
with synchronization. Instrumentation is excluded from that timing. CUDA
activity duration is collected separately, not substituted for host latency.

Minimum: hard PPL within ±1% of soft and fewer executed GEMM FLOPs.
Strong: hard PPL no worse than global +1%, latency no worse than fixed.
Exploratory signal: hard PPL within ±1% of fixed, <30% fixed GEMM FLOPs,
and better latency. All are architectural development gates; none establish
general superiority over MoE/Transformer or prove CLM.

## Bounded execution

One seed/session; dual T4 independent workers, not DDP. GPU A runs soft then
fixed; GPU B runs hard then global. Placement alternates by seed. Each GPU has
30 minutes for its long arm and 10 minutes for its baseline, each including
three minutes reserved for saving/evaluation. Preparation receives 11 minutes,
orchestration one minute, publication three minutes: **55-minute application
cap**, 60-minute server timeout. This is a hard budget, not a runtime estimate.
An unfinished arm remains incomplete; its token budget is never silently reduced.

Use the existing **Kaggle Experiment** workflow and select Experiment 003 with
one registered development seed. It updates and runs the existing
`bingjiangli/minicells` script kernel; it does not create another notebook.
Inside Kaggle, the launcher uses the existing `GITHUB_TOKEN` secret for the
repository preflight and result push. It also uses `HF_TOKEN` when available;
if that optional secret cannot be read, the pinned public TinyStories dataset
is fetched anonymously. The canonical notebook remains the source of the
uploaded script.

## Evidence

Outputs live in `/kaggle/working/results-003`, per-arm checkpoints beside metrics.
The report aggregates metrics, routing_stats.json, path_length.json, logs.jsonl,
plots and experiment_report.md. STOP and FORCED_STOP are distinguished in sampled
hard paths; soft path lengths represent probability mass, not sampled trajectories.
Training loss/PPL in per-step logs are training batch statistics; validation is
separate. Fixed entropy is zero by definition.

The existing Kaggle publisher performs the GitHub write preflight before GPU work.
Its `GITHUB_TOKEN` secret must be a GitHub PAT authorized for ArcheLabs/mini-cells
with **Contents read/write** and automatically granted Metadata read. Checkpoints
stay in Kaggle outputs; small immutable evidence and checkpoint hashes are published to `results/routing-003-<source SHA prefix>` under
`artifacts/experiments/003-hard-cell-transition/<full SHA>/seed-<seed>`.
Failed publication preserves files in Kaggle outputs. No seed directory is overwritten.

Offline repair: `scripts/evaluate.py --checkpoint ... --original-output ... --output NEW_DIR`.
It verifies data identity and keeps the original scientific decision unchanged.

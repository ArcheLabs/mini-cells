# Experiment 002 — Cell Internal Routing

This is a development mechanism study. It does not establish natural-cell behavior,
autonomous computation, continual learning, or a general CLM claim.

## Frozen setup

The registered comparison uses the Experiment 001 data revision, tokenizer rules,
Transformer width (512), six layers, 8,192 vocabulary, 10 million training tokens per
arm, and an AdamW schedule. It compares four deterministic width-2,048 Cells, the
existing external Top-2 Global Router with eight width-1,024 Cells, and eight width-1,024
Cells that predict the next Cell and an absorbing STOP after each update. Total model
capacity must match within 1%.

There are three development seeds: 42, 123 and 456. Choose one seed per manual Kaggle
workflow run. The two T4s run the internal-router arm alongside the fixed arm; the free
GPU then runs the global-router arm. GPU placement alternates by seed. This is three
bounded sessions, not one nine-run session, and is not a reserved formal confirmation.

The application deadline is 55 minutes, with Kaggle's one-hour server limit leaving a
five-minute shutdown margin. Its frozen budget allows up to 11 minutes for source setup
and data preparation, 40 minutes for the longest training/evaluation schedule, one minute
for orchestration, and three minutes for result publication. The two baseline arms share
one GPU sequentially; their combined wall budget also caps at 40 minutes. GitHub write
access and `HF_TOKEN` are checked before dataset preparation or GPU use. A missing token
or write access ends the run before training.

## Routing semantics and measurements

Soft training uses a differentiable, absorbing-STOP mean-field relaxation. It carries a
probability-weighted conditional mean state for still-active histories; it does not
enumerate every possible graph path. Every supported Cell is computed at later soft
hops, even when its probability is small, so expected path length is not actual compute
saving. The report records both expected depth and executed Cell visits.

After the 10M-token soft-trained checkpoint, evaluation runs native soft inference and
Top-1 hard inference on the same checkpoint. There is no hard-routing fine-tuning in this
protocol. The sparse evaluator executes only the selected Cell for each live token,
allows self transitions, enters Cell 0, and forces termination after at most four Cell
updates. Reported hard quality therefore measures the soft-to-hard quality gap.

Every final checkpoint is evaluated on the same 32,000 validation tokens, then timed on
a resident batch with FP16 inference. FLOPs count matmuls and repeated Cell/router calls;
they exclude dispatch, norms, activations and softmax. A separate CUDA profiler records
device activity events when supported, and `nvidia-smi` samples GPU utilization during
training. These measures have different scopes and are not interchangeable.

The fixed arm performs four sequential width-2,048 FFNs. The global arm selects two of
eight Cells in parallel. Because these execution graphs differ, the comparison measures
the full architectures; it does not isolate router location as a causal factor.

## Preregistered interpretation

The thresholds and allowed development statuses are in
[`../../validation/routing-002-cell-internal-routing/protocol.json`](../../../../research/projects/cellular-language-models/experiments/002-cell-internal-router/validation/routing-002-cell-internal-routing/protocol.json).
Minimum support requires every seed to finish the full token budget, hard PPL within 1%
of global, hard FLOPs within 5%, hard latency within 5%, and noncollapsed sampled paths.
Medium and strong support use stricter per-seed gates in the protocol. The three-seed
summary reports means and sample standard deviations without a significance claim.
Missing or invalid evidence is `DEVELOPMENT_INCOMPLETE`; any successful status remains a
development result, not scientific confirmation.

## Running and reporting

All experiment families reuse `https://www.kaggle.com/code/bingjiangli/minicells`
with title `MiniCells`, preserving the notebook's existing `HF_TOKEN` and
`GITHUB_TOKEN` secret permissions. Do not create a new notebook per experiment.

Use the repository's **Kaggle Experiment** manual GitHub Actions workflow. Choose
Experiment 002 and one of the three seeds. The workflow pins the source commit, verifies
the returned Kaggle kernel/version reference, and leaves both metrics and checkpoint
artifacts in the Actions run. Kaggle also appends a small, hash-manifested seed record to
`research/projects/cellular-language-models/experiments/002-cell-internal-router/artifacts/<source-commit>/` through the repository's
existing GitHub publisher and `GITHUB_TOKEN` secret. `HF_TOKEN` authenticates the pinned
Hugging Face dataset read. Checkpoints are not committed to Git.

The workflow never starts the other seeds automatically. Already-published seeds are
verified and skipped; an observed failed or incomplete seed is preserved and cannot be
overwritten. Matrix aggregation requires all three complete runs with identical source,
protocol, and dataset identities.

Use the unified development reporter after downloading seed folders:

```text
python shared/scripts/research/report.py routing-002 --inputs <seed-42> <seed-123> <seed-456> --output <matrix-report.json>
```

The offline Experiment 001 repair helper can recalculate routing statistics from a
checkpoint when supplied its original `data.json` output directory. It rebuilds the
immutable dataset and tokenizer, checks the text, token and tokenizer hashes, and writes
a new destination; it never edits the original evidence:

```text
python -m experiments.cell_internal_routing_002.evaluate --checkpoint <checkpoint.pt> --original-output <old-results> --output <new-repair>
```

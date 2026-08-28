# Experiment 025 — 30M Story→Math Developmental Shift

## Question

When two ~30M-scale language models have already learned stories and then receive a math-heavy continual-learning stream, does a growing CLM reach a better **new-capability / old-knowledge** tradeoff than a fixed LLM?

The public comparison is intentionally limited to:

- **LLM** — the matched ~30M Transformer from Experiment 007.
- **CLM** — a function-preserving upcycle of the retained ~30M TextNCA source from Experiment 007, with persistent program-cell growth enabled.

TextNCA appears only as the CLM source in the internal starting-condition panel. It is not a third public comparison arm.

## Hardware budget

The formal job is designed for **8 hours of Kaggle wall time on Tesla T4 ×2** and for unattended **Save Version → Run All** execution.

- GPU0: reproduce the Experiment-007 Transformer to 100M TinyStories tokens, then run the Story→Math shift.
- GPU1: reuse the retained 30M TextNCA@100M checkpoint, function-preservingly upcycle it to CLM, then immediately run the same shift.
- The orchestrator reserves 30 minutes for final evaluation, reporting, figures, and result publication.
- Incomplete workers run in at most 2.5-hour slices and automatically resume from periodic checkpoints within the same Kaggle job.
- Automatic new rounds stop when less than 20 minutes of training budget remain.
- The existing worker already requires at least 30 minutes remaining before starting a 2M-token growth probation, so a counterfactual growth decision is not intentionally started at the end of a worker slice.
- The first 100M TextNCA Story training is not repeated.
- The 72-shadow CLM-0.3d formal mechanism experiment is not repeated at 30M scale.

### Runtime estimate

The estimate is deliberately based on previously measured T4 data rather than assuming ideal scaling:

- Experiment 007 Transformer training throughput was about 28.4k tokens/s. Reproducing 100M Story tokens is therefore about **1.0 hour** of pure training, and the 50M shift is another **~0.5 hour** before evaluation/checkpoint overhead.
- Experiment 007 30M TextNCA throughput was about 12.6k tokens/s. Using the already measured batched-dense CLM training-time ratio of about 2.07× as a conservative proxy gives roughly 6.1k tokens/s for the CLM path.
- The CLM selected trajectory is 50M shift tokens. Two 2M control/shadow probations add at most one duplicated 2M future branch per decision relative to the selected trajectory, so the maximum physical CLM training budget is approximately **54M tokens**, not 58M.
- At the conservative 6.1k tokens/s proxy, that is about **2.5 hours of pure CLM training**. Allowing evaluation, checkpointing, routing calibration, shadow restore, and birth bookkeeping, the expected CLM wall time is **roughly 3–4 hours**.

Because the two arms run concurrently, CLM is expected to dominate wall time. Eight hours therefore provides substantial margin; the hard budget remains authoritative if real 30M CLM throughput is worse than this proxy.

## Fairness boundary

The Story→Math boundary is the start of the comparison that matters.

At that boundary:

1. Both models have approximately matched Story-language quality.
2. **Both optimizer states are reset.** The retained CLM source has no retained Adam state, so the freshly reproduced Transformer is not allowed to carry its pretraining Adam moments into the shift.
3. Both arms use fresh AdamW with the same LR schedule, betas, weight decay, batch size, sequence length, gradient clipping, and mixed precision.
4. Both arms receive the exact same deterministic training-example schedule.
5. Both arms optimize standard next-token cross entropy only. CLM receives no teacher-KL retention advantage.

The CLM `batched_dense` training backend is allowed because it preserves the existing exact-STE training semantics and only removes an obvious implementation inefficiency. No model-scale-specific CUDA/Triton kernel is part of this experiment.

## Panel A — internal starting comparability

Panel A plots the Story pretraining trajectory of:

- the matched 30M Transformer;
- the 30M TextNCA source that becomes CLM.

At 100M Story tokens the CLM source is upcycled. Age-zero logits are checked before the shift.

Panel A is an internal validity check, not the main product claim. Starting Story PPL is tagged comparable when the symmetric ratio is within 3%.

## Main shift

Default post-Story continuation budget: **50M tokens**.

Training mixture:

- 90% synthetic integer arithmetic;
- 10% TinyStories rehearsal.

The 10% Story component is deliberate. A 100% Math stream would confound structural learning interference with the trivial absence of any old-domain rehearsal.

The arithmetic generator contains six families:

- integer addition;
- non-negative subtraction;
- integer multiplication;
- `x + a = b`;
- `x - a = b`;
- one-variable linear forms `k*x + a = b`.

This experiment therefore tests **synthetic arithmetic adaptation**, not general mathematical reasoning.

## Measurements

At preregistered checkpoints we record:

### Old knowledge retained

- TinyStories validation NLL;
- TinyStories validation PPL.

### New capability learned

- arithmetic validation NLL/PPL;
- teacher-forced exact-answer accuracy on 256 fixed held-out arithmetic examples;
- answer-token accuracy.

The exact-answer metric scores all answer tokens correctly under the correct causal prefix. It is not claimed as unrestricted chain-of-thought or broad mathematical reasoning.

### Structure and cost

- persistent program-cell count;
- stored parameters;
- active-parameter proxy;
- training elapsed time;
- physical training tokens, including counterfactual shadow work;
- peak VRAM;
- birth proposals, promotions, and rejections.

## Preregistered Pareto crossover

A public **crossover** occurs at the first shared evaluation checkpoint where:

1. `CLM math exact-answer accuracy >= LLM math exact-answer accuracy`; and
2. `CLM Story PPL <= LLM Story PPL`; and
3. at least one inequality is strict.

No post-hoc weighted score is used to manufacture a crossover.

If no checkpoint satisfies both dimensions, the experiment reports that no Pareto crossover was observed.

## Budgeted developmental controller

The 30M run does not repeat CLM-0.3d's exhaustive 72-shadow validation. That mechanism has already been studied separately.

The performance experiment permits at most two growth decisions:

- 10M shift tokens;
- 25M shift tokens.

Each decision works as follows:

1. Collect a short routing-usage window from the current math-heavy environment.
2. Select the highest-usage eligible active lineage as a cheap **proposal**. Proposal is not birth.
3. Save the common checkpoint.
4. Train a **no-growth control** for the next 2M tokens.
5. Restore the same checkpoint, create a function-preserving child shadow, and train it on the exact same 2M future tokens.
6. Evaluate both branches on balanced Story and Arithmetic holdouts.
7. Promote only if all preregistered conditions pass:
   - mean balanced NLL utility is positive;
   - the 80% paired-bootstrap lower bound is positive;
   - Math NLL is not worse;
   - Story PPL is no more than 1% worse than the no-growth control.
8. Continue the main organism from the promoted shadow or rejected control branch.

A proposal or shadow does **not** increase the persistent cell count. Only promotion is recorded as birth.

Across two decisions, the counterfactual procedure adds at most 4M physical training tokens beyond the selected 50M shift trajectory, for a maximum of approximately 54M CLM training tokens.

## Hypotheses

### H1 — Starting comparability

The LLM and CLM source begin the shift with competitive Story quality. This is a validity condition, not the main advantage claim.

### H2 — Growth under structural pressure

A sustained 90% arithmetic shift may produce positive future utility for at least one persistent CLM birth. Growth is not forced; rejection remains a valid result.

### H3 — Adaptation/retention advantage

After enough shift experience, the growing CLM may reach a Pareto-superior point: at least as much held-out arithmetic exact-answer capability while retaining at least as much Story quality as the fixed LLM.

If H3 is not observed, the result must be reported as such even if CLM grows.

## Outputs

Internal research outputs:

- `panel-a-starting-comparability.csv/png`
- per-arm `metrics.csv`
- per-arm `worker-summary.json`
- CLM `events.json`
- `llm-vs-clm-trajectory.csv`
- `decision.json`

Public-facing outputs:

- `story-math-performance.png` — Math learning and Story retention.
- `growth-timeline.png` — Math capability, Story retention index, and persistent CLM cells on one synchronized timeline.
- `growth-animation.gif` — generated when Pillow animation support is available.

## Kaggle unattended run

The canonical notebook is:

```text
research/kaggle/experiment-025-story-math-growth.ipynb
```

With Kaggle configured for **T4 ×2** and Internet access, choose **Save Version → Run All**. The notebook:

1. checks out the experiment branch;
2. runs the regression tests;
3. verifies two visible GPUs;
4. launches the 8h one-shot orchestrator;
5. automatically resumes incomplete worker slices while budget remains;
6. generates `decision.json`, Panel A, the performance figure, growth timeline and optional GIF when complete;
7. automatically publishes curated non-checkpoint outputs to `kaggle/experiment-025-story-math-growth-results` when complete and when the existing `GITHUB_TOKEN` Kaggle secret is available.

The core command is:

```bash
python scripts/run_experiment_025_story_math_growth.py \
  --total-wall-hours 8 \
  --finalization-reserve-minutes 30 \
  --round-wall-hours 2.5
```

If the global 8h budget is exhausted before completion, the run does not manufacture a formal decision. Available checkpoints and partial metrics are preserved.

A short preflight remains a systems check only and is not formal Experiment-025 evidence.

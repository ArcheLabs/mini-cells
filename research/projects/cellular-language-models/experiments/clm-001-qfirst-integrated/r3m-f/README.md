# R3m-F — Arithmetic Learnability & Cell Specialization

**Exploratory development. Original R3m registration remains BLOCKED.**

## Run on Kaggle

Import [kaggle-learnability.ipynb](notebooks/kaggle-learnability.ipynb) from this research branch. Select **T4 ×2**, enable Internet and the existing **HF_TOKEN** and **GITHUB_TOKEN** Secrets, then **Save Version → Save & Run All**. The Notebook pins an immutable implementation, not `main`.

One launch performs functional/identity preflight → allowed asset preparation → **200 Tiny Overfit updates on eight fixed A1 Train facts** → free-generation evaluation and complete checkpoint restore verification → serialized private HF checkpoint return and GitHub metadata publication. No repeated Notebook test suite, performance benchmark matrix or old Gate 2 rehearsal. Two visible GPUs retain the historical CUDA RNG layout; Tiny training itself occupies one lane.

Rerun the **same** pinned Notebook after interruption. It restores verified private HF state, resumes remaining updates and skips completed training. All checkpoints also remain in the printed Kaggle Output directory. Reports are inside the run-ID subdirectory. HF upload and GitHub publication have separate statuses; local existence is not remote success. Preserve Output even when a return channel fails.

**Next decision:** read `TINY_OVERFIT_REPORT.json` and its eight actual predictions. The frozen engineering reference is **6/8 (75%)** after 200 updates. `TINY_OVERFIT_FAILED` retains the negative result and stops without extending the budget. A pass also stops for independent review, as required by SPEC §15.4. The nine-job Pilot is implemented and CPU-tested, but this release provides no override to start it before review and authorization. No A2/A3 training or Connection reorganization.

## Frozen design

| Phase / Arm | Seeds | Updates per job | Story / new task tokens |
|---|---|---:|---|
| Tiny A1 memorization | 151001301 | 200 | 0 / 100% |
| F0 Story | 151001301–303 | 900 | 100 / 0% |
| F1 Copy | 151001301–303 | 900 | 50 / 50% |
| F2 A1 | 151001301–303 | 900 | 50 / 50% |

Pilot: **8,100 updates / 4,147,200 tokens**, excluding Tiny and historical training. Fixed evaluation at relative **0, 100, 300, 600, 900**. All models restore the same seed's original R3h B-H Step 300 weights, AdamW and RNG; Connection, embedding and head remain frozen. Six Cells execute eight hard calls per token with self reuse. No task label enters routing.

Actual unchanged model: **1,006,694 total parameters**, **380,928 trainable Cell parameters**, **864 frozen Connection parameters**. All three arms have exactly the same structure; no parameters are added.

## Task identities and evaluation

The selection plan and explicit split were committed **before generator execution** in `b603dc2c`. A1: **44 Train / 11 Dev** canonical facts, both operand orders in the same role, equal weighting within facts. Two training templates; a third held out. The generator checks original Tokenizer identity and exact prompt-prefix/answer-token alignment. Answers contain digits followed by EOS, without a period. Answer NLL excludes EOS; EOS NLL is separate. Complete EM requires actual freely generated numeric text and EOS; teacher-forced first-token accuracy is separately identified. Every prediction and template/order variant is retained, aggregated at fact level. Wilson intervals use the stricter “all frozen variants correct” fact indicator, not an inflated template sample size.

Copy: **15 Train / 4 Dev numerical identities**, deliberately number-disjoint. Distribution weights follow A1 Train frequencies where possible. Residual total variation is **0.25**; one-digit probability is approximately **54.55% A1 vs 54.29% Copy**. This is not a perfectly matched numeric-frequency control. No balancing based on model results. The predefined Train-majority answer is `6` (tie broken by numeric order); its Dev fact accuracy is 0/11. These are descriptive exploratory references, not independent confirmation.

All allowed Story Development is explicitly reused historical data. Original R3m Main/Trigger/Pilot candidates and Confirmation are never materialized. The historical global content-isolation proof remains unresolved and is not claimed here.

## Mechanism analysis and cost

At all five Pilot checkpoints, report actual capability and all six Cells: total/stage usage, full signed Train-probe gradient comparisons, and legal fixed-ID, usage-matched and seeded-random Cell replacements. Mechanism panels are frozen to the first four preselected Story groups/one window, all 11 A1 Dev facts and four Copy Dev numbers at template 0/order 0; full capability evaluation uses every frozen variant. Replacement loss is normalized by the unmodified task NLL. Analyses preserve training state and run after training, with resumable per-checkpoint results.

Cell 1 means **zero-based index 1**, selected from prior R3m-E evidence. The primary exploratory contrast is final-minus-anchor specificity change in F2 minus **both** F0 and F1 (usage matched), reference ≥0.005 in ≥2 seeds with positive A1 replacement harm. Report every Cell, time, control and seed. Sensitivity without arithmetic competence is never labeled successful arithmetic specialization; correlations across five times are associations. No Connection-value claim.

Cost reports separate continuation updates/tokens/calls, measured training time, observed executed-work counters, analysis time and parameter counts. CPU timings are not GPU estimates. Full Pilot wall time remains unmeasured; the implemented budget is fixed and no runtime process termination is introduced.

## Engineering validation

CPU checks are in `tests/research/r3mf`, with R3m-E regression tests and GitHub Actions. They use tiny synthetic models and mocked private transport; they do not access historical model weights, Confirmation or GPUs. See `engineering/VALIDATION.md` for measured checks and unresolved items.

Reusable components: R3h model/trainer, R3m immutable checkpoint/continuation engine in its stable no-event branch, R3m-E continuous scheduling/batching, legal replacement method, state-preserving analysis helpers, serialized immutable HF snapshots, selective restore and current-main metadata-only GitHub publication. Historical sources/configs/results are unmodified.

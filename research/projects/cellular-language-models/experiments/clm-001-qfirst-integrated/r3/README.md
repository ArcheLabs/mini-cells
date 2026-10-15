# CLM-001-R3 — Competitive Credit Assignment

Independent, exploratory three-arm experiment. R1/R2 sources, checkpoints and results are read only.
The supplied protocol is retained verbatim in `SPEC.md`; `BASELINE_EVIDENCE.json` records inspected R2 evidence.

## One Kaggle Run All

Import `r3/notebook.ipynb` from `codex/clm001-r3-competitive-credit` into the original
`bingjiangli/minicells` Notebook. T4 ×2, Internet, existing `HF_TOKEN` / `GITHUB_TOKEN`.
Run All once; source IDs, hashes, beta and runtime identity are generated internally.

Flow: CPU Gate → return check → identical data/tokenizer identity → real CUDA gradient
Pilot → independent fixed-suffix direction audit → freeze beta once from Train-only
gradient scales → automatically run Q/QC/QS → isolated profiling → story-paired analysis → return.
A failed, resource-infeasible or statistically inconclusive gate returns evidence and does not train.
The default source configuration has development=false and beta=null; these are frozen automatically
only after valid same-source preflight evidence. No manual approval, mode change or second Run All.
Independent confirmation and seeds2/3 remain sealed. No hard wall-clock process termination.

GPU0 queue: Q-R3 then QS-R3. GPU1: QC-R3. Same seed, named parameter initialization,
main sample order, effective tokens and Shadow policies. A failed worker retains sibling evidence.
Already verified same-identity preflight is reused on resume; committed checkpoints skip training updates.
Save Version retains private checkpoints, RNG/optimizer state, frozen Probe bank, full diagnostics.
`publish-only` retries publication without training. Never load R2 checkpoints as R3 warm starts.

## Treatment and controls

- Q-R3: unchanged hard CE + R2 relative-Q/Rank objective.
- QC-R3: extra detached local first-order credit to the existing cost scorer, with preference logits -q.
- QS-R3: same candidate forward and gradient audit/backward computation, applied competition gradient zero.
  This is the predeclared zero-credit sham variant, not a new noisy training algorithm.

Cells/Memory/head only receive hard-path CE gradients; all arms retain the same frozen unused critic
capacity. No soft expert mixture, new attention backbone, context summary or target feature enters inference.
Self-connections, explicit CTX_READ and STOP remain intact. Candidate values are detached, used only in training.
32 actual Cell events/step, K=3 including the executed Cell. Reservoirs are stratified by depth, prior CTX and
chosen Cell with logged inclusion probabilities. Extra candidate and Oracle work are separately charged.

## Scientific gates and limits

The alignment gate requires ≥100 identifiable fixed-suffix pairs, ≥32 stories, multiple depth buckets,
story-bootstrap local agreement lower95 > .5 and positive improvement over output-permutation control.
Resampled long-horizon agreement is reported independently. No claim of an unbiased hard-policy gradient.
Beta is selected once using the predeclared median gradient-ratio target .5 and [.25,1] protection band;
no Development result tunes it. An excessive or unidentifiable beta stops the run.

Calibration spans token quartiles and depth quartiles, including explicit missing/STOP/CTX coverage.
Snapshots 0/50/100/200/300/400/500 separate frozen initial h/M probes from live recomputed states.
Conditional crossover uses predeclared Cell pairs and target-free input-token parity groups, conservative
multiplicity correction and independent Development stories for validation. Token parity is a narrow,
low-power grouping, not semantic expert labels. A/R curves and route JS/flip rates are descriptive.
Causal matched-shuffle training and multi-seed replication are **NOT TESTED** by default; H2 is never inferred
from correlation or routing visualization. Optional extra diagnostic arms/mixed-domain training are not enabled.

P1: pre-admission QC-vs-Q and QC-vs-QS difference-in-differences of greedy minus own-uniform utility.
Post-lifecycle comparisons are separate, because promotions can confound capacity. Report every arm's
four deployment policies, STOP/CTX/path costs, failure state and scientific_decision=false.
Compute-Pareto evidence includes isolated real GEMM observations and candidate-call accounting;
a complete compute-matched extra-training experiment is separately labelled NOT_RUN.

## CPU / CI

```sh
python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3/run.py --mode cpu-preflight --output /tmp/clm001-r3-cpu
```

CI `.github/workflows/clm001-r3-cpu.yml` runs R3 tests and frozen R1/R2 regressions.
Actual GPU timings, beta, local-direction verdict and asset lock remain pending until Kaggle execution.

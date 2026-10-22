# R3g — Cell–Connection Co-Adaptation Attribution

Experiment: `CLM-001-R3g-CoAdaptation-Attribution-001`. Read-only, one seed, A-C/B-C. No optimizer construction, optimizer updates, candidate regeneration, adaptation replay, Development or Confirmation evaluation.

## One launch

Import `notebooks/kaggle-readonly.ipynb` into the existing Kaggle Notebook. Enable Internet and the existing `HF_TOKEN` and `GITHUB_TOKEN`, select one T4, and Run All. No Kaggle version, source/config hash fields, CPU suite, Pilot, Smoke or manual stage changes are required.

The code first tries the original local output for execution `31633d7661954111aa3727535f044d00`, otherwise resolves the private pointer for run `6b79bb4c6c29694e10e5ad77a20285613e3b98ff247b58845f218b48ded8c39b`. The actual HF revision is pinned before downloads. Exact source/config/run/execution identities, manifest hashes, object sizes and Store receipts are checked. A currently resolved revision is never labelled as the historical head revision.

The paired branches come from ADAPTING(10), not COMMITTED/Selected. Step 100 is taken from the archived official proposal state. Initial candidate Cells and optimizer origin, final paired data/route RNG, unchanged old Connection, actual scaled new Connection, margin/tolerance, frozen weights and Selected model/optimizer are verified. Intermediate paired snapshots 1..10 support the Cell-difference timeline. Missing paired states are reported; they are never reconstructed with training.

## Measurements

Each arm evaluates OO/ON/NO/NN and 0O/0N with the original historical Calibration evaluator, tokenizer, windows and FP32. Two fixed read-only evaluations per unique model hash measure numerical repeatability: twelve primary labelled configurations across both arms, plus at most twelve repeats. Bitwise-identical model configurations reuse their already-measured result and explicitly record that reuse.

The numerical resolution is the maximum of measured repeat error, the configured floor, and FP32 epsilon times maximum absolute NLL. Two-term and four-term effects use corresponding conservative error bounds. Historical OO/NN must match the protocol and private original results within 1e-6; otherwise quality attribution stops for that arm.

All six Cells are compared per tensor, with exact hashes and numerical differences. Calibration states from the initial old organization are shared by both analytic routing-distribution comparisons. Raw entropy, epsilon-mixture entropy, hard-action probability and stage/predecessor summaries are recorded for the ten historical epsilon values. No Gumbel Monte Carlo is run. Fixed-state action probability is explicitly not a closed-loop path probability.

Training routes are read from the ADAPTING(10) record, immutable observation mirror and private original job result; these must agree exactly. The original public `TRAINING_ROUTE_DIFFERENCE.json` is independently read from verified merged commit `f87ad21a4104c010ba3eb0c091328aeb2cc46f4c` (or the original local report) and compared. This avoids depending on a deleted result branch. Original summary identity is verified, not inferred from a folder name. Publication-evidence failure retains all computed measurements but marks the audit incomplete.

## Storage and publication

HF access is read-only. Only necessary checkpoint/receipt files are downloaded; repeated requests for the same object reuse the local path. No new weights or token data are published. Comparing all ten paired snapshots may require a few hundred MB of cloud-to-cloud checkpoint reads; it does not download them to the user's computer.

GitHub receives only the named aggregate files in this unit's reports directory, on a main-descended `results/r3g-*` branch. Remote SHA is compared with the local result commit. A failed publication can be retried with `scripts/run_r3g.py --publish-existing <retained-output>` without reevaluation. Save Version retains the local output.

## Interpretation

Route-only benefit, Cell adaptation, interaction, exact parameter differences and training/inference policy mismatch are separate findings. If effects are below numerical resolution, report that instead of a signed benefit. One seed and previously selected Calibration data cannot establish general convergence, functional specialization or independent quality confirmation. No subsequent training is launched.

## Engineering checks

`checks.py` uses only synthetic CPU fixtures. It checks real Store/receipt restoration, missing paired-state refusal, frozen-weight and selected-optimizer mismatch refusal, six configurations and identical-Cell identities, interaction arithmetic, numerical repeats, analytic entropy/probabilities, private trajectory consistency, unchanged input files, no optimizer construction/steps/backward, and pinned HF read caching. Kaggle does not rerun this suite.

Engineering base: repaired R3f-v2 research branch at `57906beece8d2f0034fa6305c05d510954f0d586`. At audit implementation time, main contains merged R3f-v2 result artifacts but not its implementation. Historical implementation is reused from the research branch; public input is independently pinned to the verified main commit.

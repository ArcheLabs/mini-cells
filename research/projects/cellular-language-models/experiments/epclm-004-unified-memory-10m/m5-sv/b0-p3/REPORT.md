# B0-P3 pre-execution review and implementation report

Status: implemented and CPU-validated; actual T4 benchmark, frozen Stage2 budget, Stage1 learnability gate and conditional Stage2 results remain pending. No canonical P3 GPU updates occurred locally. Test was not generated, loaded or evaluated. H1–H6 remain NOT_EVALUATED.

## Existing implementation audit

Remote main remains ce9f1905b8d9a8badd2dbed111f68e180e71a619. The actual P2 source is fe31d2695675eec1c33b09c70f681dd331a07782. The published P2 summary65c7589708e0aa057afbf583c368258cb51713dd records640 updates per arm,4922.42 charged GPU seconds,4/256 Training Diagnostic hits and2/512 Validation hits in every arm. Value mass is about0.998 while conditional NLL remains above ln256 and full-vocabulary top prediction is constant. This review reads the published evidence; it does not claim another complete independent audit of the old P2 binaries.

The actual P2 optimizer protocol was reconstructed from frozen source and its640-update budget. Its SHA256 exactly matches the published bf5827a453601dab7ee1141a9f4af53ec2f49b1cda6671d3dd8c094299706263. P3 preserves FP32 AdamW, LR0.0003, betas0.9/0.95, weight decay0.1, global clipping1, constant LR and TF32 off. P2 uses one parameter group, with no special architecture group. P3 deliberately versions the objective, logical batch, sequence/data task, checkpoint cadence and curriculum; it does not silently scale LR.

Three existing features required an explicit P3 adapter:

- P2 repeatedly samples a fixed epoch bank; P3 needs new mappings for every epoch. The new thin sampler wraps the existing transactional sampler separately for each counter-derived512-Episode epoch and preserves order/RNG/cursor in checkpoints.
- P2 computes full-vocabulary Query loss. P3 selects the current factual candidate columns outside the unchanged model and differentiates only that conditional CE. Other losses are detached diagnostics.
- E6/E7 reject PAD IDs, and letting PAD update their state would violate the specification. A common adapter omits only trailing PAD, retaining every real token/EOS and every original position. Storage64/128 and processed59/115 counts are separately recorded. This is neither input truncation nor streaming prefill.

The original E6/E7 code does support per-stream native batch state. Tests verify Batch8 independence and equivalence to eight separately accumulated gradients. Training and primary evaluation use native Batch8 rather than an unnecessary serial fallback. All original transformer, Mamba, SWA, Summary and Store computations remain unchanged.

Model sizes remain E0 10,130,688 and E6/E7 10,021,120. New model seed75043 and Store metadata seed78043 are frozen. E6/E7 share the complete initialization; E7 has no exclusive trainable parameters, so their initialization is NOT_APPLICABLE. Actual target-environment initialization hashes are generated/checked within that environment rather than comparing CPU and T4 package-dependent initializer bytes. No previous trained checkpoint initializes P3.

## New data audit

Master93031; independent SHA256-derived Train/Validation/level/epoch/index RNG namespaces. Keys are sampled without replacement and Values with replacement. Fact/query permutations are separate RNG draws. Current IDs and8192-tokenizer SHA d1ed85988c25e27618ea8de951c9005758d57fef9e46832fa623a0812afe2d04 are preserved; SEP aliases reserved HOLE7.

| Level | Prepared training mappings | Validation mappings | Train/Validation full-map overlap | Repeated-Value training Episodes | Actual random candidate EM | Actual random candidate NLL |
|---|---:|---:|---:|---:|---:|---:|
|8|1024|256|0|106|0.1275111607|2.0609436643|
|16|3072 maximum|256|0|1201|0.0645071400|2.7420638188|

All256 training Keys are observed with multiple different Values at both levels. Validation candidate counts are{6:3,7:29,8:224} and{13:1,14:27,15:62,16:166}. Individual Train/Validation Pair overlap is223/2038 and is allowed in this mapping-local task. Full Mapping, full Episode and seed overlap are zero. Exact fact lookup reproduces every validated answer. Prepared Stage2 data and temporary timing prototypes do not mean Stage2 training has started.

Frozen dataset identity: ed9c3d0fdefffb187297b59a8b9dc7009d090764780a7cf900f6914f84c88e52. Source-bank files and hashes are generated automatically; actual consumed Stage2 prefix is determined only by the measured common budget. Test namespace access is denied. Because Test cannot be opened/generated, Train/Test mapping intersection remains NOT_CHECKED_TEST_SEALED, not a manufactured zero or PASS. Older pair-disjoint Test semantics are not asserted for this redesigned task.

## Owner-frozen gate clarifications

Online Stage1 EM is cumulative across all8192 Queries, measured before each respective update, not a selected final window. Severe collapse concerns active-candidate predictions only: among K>1 Episodes, at least50% having at least90% of predictions on one candidate blocks advancement. Full/value collapse is diagnostic and cannot veto a candidate-only learning objective. These owner decisions are reflected in configs/protocol.json and SPEC.md before GPU execution.

The cohort advances only after every128-update Stage1 ledger, order, checkpoint and all five conditions pass. Each Stage2 worker rechecks all arms' saved gate metrics, protocol and checkpoint hashes. Stage2 restores trained model/optimizer/RNG/global step and creates only the authorized new stage sampler. Python process restoration is not a learned-state reinitialization. No arm can advance alone, increase exposure or change LR.

## CPU correctness evidence

Fourteen focused tests pass. They cover generator randomness/replacement and causal layout; exact current-fact oracle; mapping/Episode/namespace isolation; candidate-only masked gradients and stable three-part NLL; variable1/K/lnK baselines; all original model classes' native Batch8 independence, causality and PAD exclusion; native-versus-accumulated gradients; real optimizer descent; fresh-mapping epoch-boundary replay; exact interrupted continuation; optimizer/global-step curriculum continuity; independent payload/checkpoint/log-probability reproduction; corrupted-label rejection; all gate criteria; isolated-query clustering; and every common-budget downgrade/rejection case.

Gradient averaging tolerance is atol3e-6/rtol3e-5. Tiny controlled one-query update examples:

| Model | Candidate loss before | After | Gold probability before | After |
|---|---:|---:|---:|---:|
|E0|1.066631675|1.061361194|0.344165840|0.345984513|
|E6|1.120556831|1.111988425|0.326098152|0.328904313|
|E7|1.120556831|1.111988425|0.326098152|0.328904313|

These are small CPU fixtures, not canonical10M training or evidence of generalization. E7's controlled short fixture does not require Store use. Detailed gradient errors, parameter changes, logs and delivery hashes are archived under audit/. Source/Notebook syntax is checked separately.

## Persistence, monitoring and integrity

Complete new checkpoints use the original atomic checksum envelope and plain tensor model state, preserving the portability fix without modifying historical files. Distinct final stage checkpoints, optimizer moments, RNG, stage sampler, global/stage steps, ledger and cumulative online totals are preserved. Save every64 updates plus evaluation/final boundaries. Raw update/resume/rollback events are appended/fsynced; the checkpoint's committed history is authoritative after interruption. No complete update/evaluation is repeated merely to repair publication.

Candidate, value and full prediction spaces, decomposition, entropy/concentration/ranks and Mapping-cluster intervals are recorded. Candidate/value arrays exist at intermediate evaluations; complete full-space arrays are retained at final checkpoints for independent replay. State/Store observations are read-only, scoped per stream and refer to actual selected positions. Missing intermediate state-update norms, calibrated saturation, retrieval scores, semantic binding retrieval and causal contribution are NOT_AVAILABLE. Joint fact-position hits and occupancy do not prove useful association storage.

Query-isolated diagnostics use64 fixed Validation mappings with4 separate Queries each. Fresh state is enforced in each batch stream. Moving a Query to a single-query prefix also changes its position/chunk/local-window/Store availability. Differences are secondary observations, not a clean causal estimate of previous-answer reliance. Their all-arm/time-only decision is frozen before publication and may be NOT_RUN_BUDGET.

New bundles include actual banks, identities/config/source/environment/init hashes, requested metrics/log views and all available final checkpoints. Existing publication/archive utilities are reused; repeated identical checkpoint paths are verified tar hard links. Each full bundle is independently downloaded, all members and mathematical/data/ledger invariants checked, then final checkpoints are restored and Validation/predictions/log-probability arrays reproduced within2e-5. Final companion indexes include that proof and raw verifier logs. The summary preserves cohort gate state, paired Mapping-cluster analysis and the cumulative GPU phase ledger.

## Pending actual execution

Local Python3.11/Torch2.6 is CPU-only and has no authenticated Kaggle execution session. Canonical GPU time, chosen Stage2 budget, learnability gates, stage outcomes, GPU memory/cost and real remote P3 artifacts are pending. They are not simulated or marked PASS.

One Notebook launch in the existing T4 environment performs GPU isolation/causality probes, native Batch8 benchmarks at both lengths, and a common384/320/256/192 Stage2 budget freeze before any formal Stage1 update. Forecast must fit6480 seconds; all arms/stages/benchmarks/restore/optional diagnostics share a persisted7200-second cap. Parent process deadlines and a worker watchdog prevent unbounded GPU work; partial results never pass. No unverified precision/model/context/optimizer fallback exists.

The final experiment must distinguish candidate-conditioned recall from full-vocabulary generation, use the actual random baselines, account for8→16 difficulty plus continued training, and preserve single-seed uncertainty. A Stage1 No-Go returns failure diagnostics and prohibits Stage2 for every arm. No12M,32–64-pair scaling,formal multi-seed experiment,Test access or M6 is authorized.

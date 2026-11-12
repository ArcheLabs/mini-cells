# B0-P3 — Minimum Associative Recall and Memory Scaling

Version v0.1 with two owner-approved gate clarifications. Single-seed learnability calibration; H1–H6 remain NOT_EVALUATED. This English protocol implements the supplied P3 specification. No historical B0/P2 source, configuration, weight or result is modified.

## Frozen model and execution semantics

Use the existing E0 Full Context Transformer (10,130,688 parameters), E6 Mamba-1/SWA/State Summary and E7 Innovation Store (10,021,120 each). Preserve all layers, embeddings, heads, intrinsic SWA window32, chunk32, Summary4, state20, global per-stream Store capacity64, Top-8 read/admission rules and predictor beta0.875. E7 has zero exclusive trainable parameters, so their independent initialization is NOT_APPLICABLE. Fresh model seed75043 and Store metadata seed78043; E6/E7 full initialization hashes must match. No historical trained weights are loaded.

Each complete logical Episode has64/128 slots, of which59/115 are real tokens. Only trailing5/13 PAD tokens are omitted at the common model adapter: all real tokens, positions, facts, queries, previous answers and EOS are retained. This is padding exclusion, not bounded context or external streaming. E6/E7 reject PAD IDs in their frozen forward, so passing padding through their recurrences would violate the specification. All arms receive exactly the same complete valid prefix.

Use existing native independent Batch8 states. CPU tests compare native versus eight separate gradient accumulations and test cross-stream invariance; target GPU benchmarks additionally test native isolation and causality. No unverified fallback is selected on failure. Candidate losses are summed across all Queries in the8 Episodes and divided by their total Query count before one backward/update. Native evaluation also uses Batch8. A fresh Runtime is created for every batch, with separate recurrent state slices and Store objects for every stream. No Episode state crosses updates or stages.

## Episode-local data

Keep the frozen8192 tokenizer and all existing token IDs. Keys8–263 and Values264–519 are disjoint256-token sets. BOS2, EOS3, PAD0, FACT4, QUERY5 and ANSWER6 retain their meanings. SEP aliases the existing reserved HOLE token7; no token ID or embedding is remapped or added.

Data master seed93031. Derive each Episode's RNG seed by SHA256 of master, namespace, level, epoch and index. Train/Validation use separate namespaces. Keys are uniform without replacement; Values are independent uniform draws with replacement. Independently permute Fact order and Query order and query each Key once. Repeated Values are allowed and deduplicated only for the active candidate set.

Sequence: BOS; N triples FACT/KEY/VALUE; SEP; N quadruples QUERY/KEY/ANSWER/VALUE; EOS; trailing PAD. Loss at each ANSWER predicts the following VALUE. The current gold VALUE occurs only in the future input position; causal masks/recurrences prevent self leakage. Earlier gold answers are teacher-forced standard causal history. Primary evaluation is teacher-forced scoring, not free-running generation.

Stage1:8 pairs,64 slots,2 epochs of512 new mappings,1024 Episodes,128 updates,8192 targets,65536 logical input slots and60416 processed tokens. Stage2 target:16 pairs,128 slots,6 epochs/3072 new mappings,384 updates,49152 targets,393216 logical slots and353280 processed tokens. Maximum per-arm totals:512 updates,4096 Episodes,57344 targets,458752 logical slots and413696 valid processed tokens. All arms share exactly the same counter-derived mappings, permutations, batches and stage order. The existing transactional sampler is reused independently for each new-mapping epoch; it never repeats a fixed512-Episode mapping bank.

Audit complete Mapping hashes over sorted token-ID pairs, full Episode hashes and seed namespaces. Train/Validation complete mappings and Episodes must not overlap. Individual pairs may coincide and are not leakage. Audit all prepared mappings, including maximum Stage2 capacity before budget selection; only the frozen authorized prefix is consumed. Oracle baseline is exact lookup from current facts. Test generation/loading/evaluation is forbidden. The test namespace is sealed; Train/Test Mapping intersection is NOT_CHECKED_TEST_SEALED because inspecting it would violate that boundary. No claim of compatibility with older pair-disjoint Test semantics is made for this redesigned task.

## Objective and diagnostics

For each Episode use its distinct factual Values C as the active candidates. Candidate IDs are not supplied as model input and no new head is added. Optimize only L_assoc=logsumexp(z_C)-z_gold at ANSWER positions. Preserve original AdamW: LR0.0003, betas0.9/0.95, decay0.1, clipping1.0, constant LR, FP32 and TF32 off. No batch-size LR scaling or auxiliary-loss weights. The original comparable parameter policy is one AdamW group; no architecture-specific group is silently added.

Report candidate-,256-value- and8192-full-vocabulary predictions, EM, NLL, entropy, concentration, diversity, Top-5, gold rank/probability and type/candidate masses. Stable log-sum-exp must satisfy L_full=L_type+L_candidate+L_assoc. Random candidate accuracy/loss are mean1/K and mean lnK using actual per-Episode K, alongside256/8192 uniform references and a100% exact-context oracle. Never substitute ln8/ln16 for variable-cardinality baselines.

Stage1 Validation at0/32/64/96/128 uses fixed256 mappings/2048 targets. Stage2 Validation at start/25%/50%/75%/100% uses fixed256 mappings/4096 targets. Curves include pre-update online training metrics, not an easier separately rescored training set. Bootstrap95% intervals and paired differences resample complete Mapping/Episode clusters; all Queries from one mapping stay together. Empirical intervals can collapse at a floor/ceiling and do not measure across-seed uncertainty.

Full final log-probability arrays are retained for independent checkpoint reproduction. Intermediate checkpoints retain candidate/value arrays and complete full-space metrics/predictions. Invalid candidate-array padding uses ID-1/log probability-infinity and is excluded from normalization and statistics; it is not model input padding.

## All-arm Stage1 gate

After each arm completes128 updates, verify its complete ledger/order/checkpoint before computing the gate. Every arm must satisfy:

1. Validation active-candidate EM≥0.60.
2. Episode-bootstrap EM95% lower bound exceeds actual mean1/K.
3. Mean lnK minus active NLL≥0.5 nat.
4. Online training active EM≥0.75 across all8192 Stage1 Queries scored before their respective updates. No final-window selection.
5. Finite numerics and no severe active-candidate collapse.

Owner-approved collapse definition: among Validation Episodes with K>1, at least50% having at least90% of Query predictions on one active candidate is severe. Full/value-vocabulary collapse is diagnostic and does not veto this candidate-only experiment.

If any arm blocks, Stage2 is not trained for any arm. Finish and publish the Stage1 diagnostics with STAGE_1_GATE=BLOCKED and STAGE_2_STARTED=FALSE. This predefined No-Go is a complete calibration result, not a scientific impossibility verdict. If all pass, each Stage2 worker rechecks the cohort gate and all Stage1 checkpoint hashes, then restores its own Stage1 model, optimizer, RNG and global step with a new Stage2 namespace/sampler. The trained state is continuous across process restoration; no statistical reinitialization or LR reset occurs.

Descriptive stability means two consecutive scheduled Validation points meeting the three Stage1 Validation conditions. It is reported separately and never adds a new advancement gate. First observed above-random and first CI-above-random points are checkpoint-resolution observations, not exact learning-change times.

## GPU plan and time cap

Measure actual T4 native Batch8 forward/backward/update costs at both lengths, full256-Episode Validation and a complete checkpoint. Each independent temporary run performs one warmup/two measured updates and functional GPU isolation/causality probes; its weights never initialize formal training. All benchmark time is charged.

Before formal Stage1, freeze the largest common Stage2 budget384,320,256 or192 updates whose whole three-arm forecast≤6480 seconds. Stage1 remains128 for all. Forecast includes both stages, scheduled Validation, checkpoint, final independent restoration,25% margin and process overhead. If192 does not fit, stop before formal training. Never reselect the budget during training or publication retries.

Cumulative GPU-worker wall across all arms/stages/sessions is capped at7200 seconds. It conservatively includes imports and CPU work inside workers, benchmarks, forwards/backwards/updates, Validation, raw-update log/checkpoint I/O, independent re-evaluation and optional diagnostics. CPU preparation/hashes and external artifact transfers outside workers are excluded and recorded separately. Reuse the proven P2 persisted clock/process-group killer plus a worker watchdog. Exhaustion yields TIME_BUDGET_EXCEEDED; incomplete work cannot pass. No precision, model-size, context-length or algorithm fallback.

After mandatory training and before archive freeze, optional Query-isolated diagnostics are planned as all-arm groups using time only, while reserving independent restores and the6480 planning line. Freeze that decision for retries. Each level selects64 fixed Validation mappings and4 Queries each; every Query has fresh native stream state and Facts+one Query. Insufficient time yields NOT_RUN_BUDGET. Single-query placement changes chunk/window/Store history as well as answer history; differences cannot uniquely identify causal reliance on earlier answers.

## Persistence, telemetry and return

Use existing atomic checkpoint envelopes, transactional sampler and optimizer observations. Save every64 updates plus initial/evaluation/final boundaries for exact evaluation reuse. Stage1 and Stage2 final checkpoints remain distinct. Include plain tensor model state, optimizer, RNG, sampler, dataset/protocol/model identity, global/stage steps, epoch cursor, ledger and online totals; constant scheduler state is None. Normalized model export avoids inherited bound-method metadata. Never rewrite older weights.

Record every update's stage/epoch/global step, pre-update NLL/EM, actual LR, gradient norm, complete parameter-change norm, post-update parameter norm, logical/valid tokens, targets, duration and GPU allocation. Append/fsync raw update, resume and rollback events; final committed history comes from the authoritative checkpoint. A zero gradient on zero-loss/single-candidate solved data is distinguished from a disconnected nontrivial loss.

Read-only telemetry records per-stream Summary/state norms, activations/resets/resources, Store writes/rejections/occupancy/evictions, selected-position Top-1/5, factual Key/Value/joint hits and EMA statistics. Missing intermediate state-update norms, calibrated saturation, retrieval scores, semantic binding retrieval and causal Store contribution are NOT_AVAILABLE. Position co-selection and occupancy are not proof of useful binding memory.

Publish separate immutable bundles and main-derived result indexes. Include actual datasets, identity/config/source/environment/init hashes, all logs/metrics/arrays, available stage-final checkpoints and SHA256 inventory. Reuse full independent download/member-hash verification and independently restore/re-evaluate each available final checkpoint; compare labels, predictions and log-probability arrays within2e-5. A separate summary index preserves cohort No-Go/Go, paired cluster results, resource accounting and interpretation. No Test,12M main experiment, formal multi-seed experiment,32–64-pair scaling or M6 is authorized.

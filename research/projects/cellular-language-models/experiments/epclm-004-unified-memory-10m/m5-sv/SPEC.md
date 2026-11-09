# EPCLM-004 M5-SV — Scientific Validation, v0.1

## Scope and evidence boundary

This English implementation specification follows the supplied M5-SV v0.1 task. M5 engineering completion is not H5 support. Keep H1–H6, original model definitions, historical results and checkpoint bytes unchanged. Do not implement Titans, routers, metamemory, future-forgetting predictors, new representations or learned innovation scores. No M6 authorization is granted.

The frozen implementation uses Mamba-1, SWA W=32, four previous-chunk SSM summary tokens per SWA layer, chunk=32, parameter-free EMA beta=.875, mean squared prediction error, original input embeddings in one global bounded Store per stream, and Query-similarity read Top-8. Random is standard seeded reservoir sampling; FIFO is chronological; Innovation uses strict replacement of the minimum score. Capacity64 and128 share all other mechanisms. No mechanism is silently repaired or replaced.

## Baseline audit and identity

Read `audit/m5-baseline-audit.md` and its JSON. The source/result/data/tokenizer identities are frozen in `configs/baseline-lock.json`. M5 has five two-update Smoke models, not adequately trained models. Real checkpoint hashes exist, but binary availability is unverified outside saved Output. New scientific runs start fresh. E3 remains an explicitly untrained historical reference and is not a fifth training arm.

## SV0 — Protocol audit

Reuse M1 integer tokens and pair partition: (key index + value index) modulo3 is0/1/2 for train/validation/test. Use separate model/data/store seeds and independent split seeds. Keys and values share the known vocabulary; held-out bindings are distinct from held-out symbols. Episode IDs and actual content must not overlap. Freeze generated Test bytes and SHA256 before Pilot. Generated Test is not evaluation access. All policies and training seeds share the frozen Test bank.

Each evidence record has record_id, key_id, value_id and source_position. Each Query has query_id, required_record_ids, expected_value_id and query_position. Metadata is never a model input. Multi-query history is prefetched once; each Query receives an independent clone of the identical Runtime/Store snapshot and only `[QUERY, key, ANSWER]`. Gold answers are labels, never subsequent inputs. The additional metadata cannot change the history token sequence.

Controls: reproducible random guess over256 value tokens, actual no-history inference, E6 no Store, and diagnostic E7-Oracle. Oracle admission alone sees future-required source positions, prioritizes those value records within the same capacity, and preserves original embedding representations, positions and Query-similarity read. It cannot train or change Query inputs. Oracle is a selection upper bound, not a guaranteed answer/retrieval upper bound. Oracle failure to improve EM is evidence to diagnose the task/representation/read interface; it is not silently replaced by answer injection.

## SV1 — Innovation validity

Measure score versus recoverability: use E6 no-Store to probe each evidence record from an independent history snapshot. R_i=1 only for a correct answer. The score's target is1-R_i; labels never enter the EMA predictor. Report raw scores, recoverability labels, group score distributions, AUROC and average precision (AUPRC), with explicit nulls for single-class data. Report by sequence length as well as overall.

For S=stored record identifiers, Gq=query-required records, Ge=union of all required records: Coverage=|S∩Gq|/|Gq|, Admission Precision=|S∩Ge|/|S|, Admission Recall=|S∩Ge|/|Ge|. Non-evidence stored tokens count in the precision denominator. Compare Random/FIFO/Innovation/Oracle with identical static weights and identical Episodes at64/128; distinguish this fixed-checkpoint intervention from separately trained policies. SV1/2 preflight uses development data only; complete measurements are repeated on adequately trained models.

## SV2 — Retrieval and utilization

Report actual ranked candidates separately for each SWA layer. The primary read metric uses the last SWA layer's actual Top-K candidates; never claim success from an arbitrary layer union. Hit@1/Hit@K, Recall@K and reciprocal rank follow required_record_ids. Missing hits have RR=0.

Store On/Off must clone the same prefill Runtime and Store from the same checkpoint. Off removes Store reads only during the Query; history, summary, local KV, record contents and predictor remain identical. Report paired delta EM, conditional EM given Hit@K, rescue versus E6 wrong queries, and regression versus E6 correct queries. Zero-denominator conditional metrics are null. Attention mass alone is not utilization evidence.

## SV3 — Pilot, freeze and independent evaluation

Train E6, Random, FIFO, Innovation; never train Oracle. `configs/protocol.json` declares three independent model seeds, separately recorded data/store seeds, paired initialization/data/sampler, FP32 and TF32=false, AdamW constant LR. Reuse M2 atomic Model/Optimizer/RNG/Sampler/Ledger checkpoints and rollback. All successful optimizer updates, input tokens and supervised/query targets are recorded separately. Do not count a failed or rolled-back update.

First run a separate four-arm Pilot,16,384 input tokens per arm, model seed50003 (not a main seed). Check finite updates and held-out learnability: each arm must improve validation NLL by at least.02 and answer at least2/16 validation queries correctly. A failed Pilot stops with INCONCLUSIVE; it never expands its budget or accesses Test. This is a conservative execution gate, not a claim that the mechanism is disproved.

After Pilot, persist an immutable freeze receipt containing source/config/dataset/tokenizer hashes, all decision thresholds and three-seed budget. Main runs start fresh, not from Pilot. Each arm/seed receives exactly1,000,000 input tokens;12M total, plus the separately bounded65,536-token Pilot. A1954-episode single-pass bank has1953×512+64 tokens; the existing sampler shuffles the complete bank once. Every arm consumes it exactly once. The64-token remainder is explicit and identical across arms. No Test-driven change or budget extension is permitted.

Preserve validation curves at25/50/75/100% of updates. Continuing improvement beyond.01 NLL or.01 EM in the last quarter makes adequacy INCONCLUSIVE, not success. Three seeds are displayed separately, with mean, sample standard deviation, seed-level95% interval and paired hierarchical bootstrap (resample paired seeds, then paired Episodes). Query means are nested within Episode, never treated as independent observations.

Test uses500 independent Episodes per representative condition, not a Cartesian product: lengths256/512/2048/8192/16384; bindings8/16/32/64; queries1/8/32; interference0/.5/1; fixed-initial-embedding cosine strata; repeated values; delays16/32/64/256/512 around W=32. Query isolation keeps history length constant when changing Query count. Full evaluation is resumable at Episode granularity. Every case must finish before a scientific conclusion; timeout is PARTIAL, not a reduced-size PASS. The fixed cosine reference is an engineering similarity control, not a semantic similarity claim.

## Resources and decision

Record unique static parameters, weights, Mamba/KV/Predictor/Summary bytes, Store payload and serialized metadata, checkpoint bytes/write time, training peak/compute throughput, history prefill time, per-query and per-input-token latency, Store read/update timings. Profile inference separately from training; give64/128 costs. Do not call residual allocator estimates exact activation memory. Reliable training FLOPs are null with an explicit reason unless actually measured. Iso-parameter comparisons use identical static weights within E7. Iso-resource assessment reports measured dynamic storage and latency ratios without adding an unauthorized training group.

Before Test, freeze these MVP thresholds: Innovation coverage difference≥.05 versus both Random and FIFO, paired95% intervals positive; actual Hit@K/MRR improvements; same-checkpoint On-Off delta EM≥.02 with positive paired95% interval; consistent final long-range EM gains against E6 and both Store controls. Also require evidence that innovation predicts recovery failure, rather than novelty alone. Resource bounds: Query latency ratio versus E6≤2 and additional dynamic state≤2MiB for the same capacity/condition; report all raw ratios. These are project choices, not universal standards.

H5 can be SUPPORTED only when all requirements, adequate training, complete three-seed Test and reasonable resources hold. Classify admission without utilization, Oracle success without Innovation, Oracle failure, and insufficient evidence separately. Incomplete evaluation, single-class recoverability evidence or unstable training yields INCONCLUSIVE. Complete stable evidence without incremental value yields NOT_SUPPORTED. Until measurements exist, retain NOT_EVALUATED. Never advance automatically to M6.

## Execution and delivery

One Kaggle Notebook invocation runs cached functional SV0 checks, freezes data, runs development SV1/SV2, Pilot, budget freeze, main training and sealed Test in order. At most3 hours per session:9600 seconds experiment,600 seconds publication reserve,10800 hard cap. Resume exact-identity checkpoints from working Output or attached saved Outputs, preserve completed units, and publish reports/metrics/hash manifests through the existing isolated publisher with remote SHA verification. Missing prior Output is BLOCKED rather than silently restarting an ongoing run. Kaggle Script Version ID may remain explicitly pending until the saved version is supplied; never invent one.

All new documentation, code comments and report text are English. Existing Chinese historical documents receive English addenda only. `REPORT.md` contains actual run status, not simulated science. Each implementation phase has its own commit. Raw logs, configuration and code/data/result hashes are retained. Weight files stay outside Git.

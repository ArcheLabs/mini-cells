# B0-P2 — Balanced Data and Extended Training Pilot

Version: v0.1, with the user's explicit marginal-tolerance clarification. Controlled single-seed development experiment. H1–H6 remain NOT_EVALUATED. This English protocol implements the supplied B0-P2 specification; original B0 experiments, configurations and weights remain unchanged.

## Scope and models

Use the existing E0 Full Context Transformer (10,130,688 static parameters), E6 Mamba-1/SWA/State Summary and E7 Innovation Store (10,021,120 each). Every primary training/evaluation call consumes the complete512-token Episode and supervises only its8 Query answer positions. E6/E7 retain their intrinsic SWA window32 and chunk32; no input truncation, streaming substitution,509+3 path or new architecture is introduced. Every call creates fresh Episode state. E6/E7 common initialization must match exactly. E7 has zero exclusive trainable parameters: separate exclusive-parameter initialization is NOT_APPLICABLE. All formal P2 models are freshly constructed; prior Overfit/Pilot and temporary benchmark weights are never used as initialization.

## Data

B0-P2-DATASET-v2 uses only the union of the previous Pilot Training and Validation unique pairs. Reconstruct the original frozen banks and check their original content hashes before forming the pool. The actual pool has4252 pairs:3746 from Training,506 from Validation, with zero overlap, all256 Keys and256 Values. No Test generation, inspection or evaluation is authorized.

Deterministic split seed75029. Select506 Validation pairs through two disjoint symbol-covering bipartite matchings, remove six edges while preserving coverage, and compare32 deterministic candidates using marginal JSD. Preserve the nearest-integer old-source proportions: Validation446/60 and Training3300/446 from the former Train/Validation sources. Modulo labels are used only as historical source strata; both new splits contain both strata. Pair membership is not assigned by modulo.

Both banks cover all256 Keys and Values with zero pair overlap. Validation unique degree is1 or2 per symbol. The original candidate degrees make Training degree14–15 everywhere impossible. The user accepted preserving the full pool and treating14–15 as an ideal, with these explicit limits:

- Key and Value JSD on unique pairs: each≤0.02 bit.
- Key and Value JSD on actual Episode observations: each≤0.01 bit.

Preserve512 Training Episodes/4096 targets and64 Validation Episodes/512 targets,512 input tokens and8 distinct Query bindings per Episode. Cover all assigned pairs once, then allocate350/6 supplemental observations to reduce marginal imbalance and preserve source proportions. Pack eight pairs with distinct Keys and Values per Episode. Reuse original noise, token IDs, Query ordering and answer masks, rewriting every fact/query binding consistently. Duplicate rates, maximum multiplicity, per-symbol counts, source proportions, JSD and total variation must be reported. Current Training maximum pair multiplicity is7, versus3 in the previous Pilot; this allocation is part of the changed data distribution and is not an isolated split-only intervention.

The fixed Training Diagnostic subset is the first32 newly generated Training Episodes (256 Queries). Its diagnostic forwards use no gradients or optimizer updates; those Episodes remain members of the512-Episode training bank. Validation has64 Episodes/512 Queries. Test remains reserved and unaccessed; no physical sealed Test artifact is claimed where none has been generated.

## Training and budget

Retain the prior protocol's FP32 AdamW: LR0.0003, betas0.9/0.95, weight decay0.1, clipping1.0, constant LR, TF32 disabled, batch1 and Query-only mean loss. Model seed75017, Store metadata seed78017, sampler seed76003, statistics seed86003. The original sampler reshuffles after every complete512-Episode epoch and is shared across arms.

Choose the largest common budget from1024,768,640 updates before any formal P2 update. These correspond to524288/393216/327680 input tokens and8192/6144/5120 supervised target observations per arm. Distinct training pair count remains3746, regardless of repeated exposure.

First independently download prior result bundles and read their actual timings. Benchmark each exact model with one warmup and four measured temporary updates, full64-Episode Validation,32-Episode Training Diagnostic and a real checkpoint write. Temporary benchmark updates are separate from formal ledgers; all their time is charged. Forecast using the slower of new maximum measured step and prior mean step,25% margin, measured evaluation/checkpoint costs and process overhead. Include final independent restore/evaluation. Only a forecast≤6480 seconds permits training; if640 does not fit, stop without fallback.

All three GPU phases cumulatively share a7200-second hard cap, including benchmarks, train/diagnostic/Validation forwards, checkpoint I/O and independent GPU restore/re-evaluation. Controller subprocess wall is conservatively charged, including import/CPU work within a GPU phase. CPU data preparation, hashing, report aggregation, uploads and downloads outside GPU workers are excluded and recorded separately in end-to-end wall time. Persist charges across sessions. Kill a running GPU worker at exhaustion and mark TIME_BUDGET_EXCEEDED; partial runs cannot pass. Retain at least720 seconds between planning and hard limits.

## Monitoring and persistence

Per update record loss, gradient norm, full parameter-change norm, post-update parameter norm, actual LR, epoch/cursor, cumulative tokens/targets, compute time and GPU allocation/reservation. Evaluate both fixed Training Diagnostic and full Validation at0/256/512/768/1024, retaining only boundaries within the chosen budget and adding its final boundary (including640).

Record full8192-way EM/NLL, constrained-value EM/NLL, Top-5, gold rank/probability, valid-value probability mass, entropy, predicted-token counts/diversity, dominant-token rate and invalid-output rate. Stable log-softmax decomposition must satisfy full NLL=type NLL+value-conditional NLL. ln256 is the uniform conditional-value reference, not an unconditional full-vocabulary baseline. Preserve complete probability/log-probability arrays and prediction records.

Read-only hooks record Summary norms, final state activations and resources, Store admissions/rejections/occupancy/evictions, actual selected positions and overlap with factual Key/Value positions, and EMA/innovation statistics. Missing state-update norms, retrieval scores and causal Store-effect measurements are explicitly NOT_AVAILABLE. Position hits/occupancy do not establish semantic or causal benefit.

Use existing atomic checkpoint/rollback/RNG/sampler primitives. Save complete checkpoints every128 updates and at final/valid-stop boundaries. Preserve Model, Optimizer, RNG, sampler order/cursor, global step, epoch cursor, ledger and exact source/config/data/environment identity. There is no stateful scheduler. New P2 checkpoints normalize model state to a plain tensor dictionary to remove inherited bound-method metadata; old files are untouched. Completed runs and evaluation boundaries are reused.

## Integrity and analysis

Each arm has a separate b0-p2 Run ID, full configuration/data/init/environment/source identities, archived actual datasets, logs, monitoring, complete checkpoint, probability arrays and SHA256 inventory. Export latest/final checkpoint paths; archive repeated identical weight payloads using verified tar hard links. Reuse the existing GitHub Release publisher and main-derived metadata-return utilities.

After publication, independently download the full bundle, verify every member and the complete archive hash/size, check ledgers/order/labels and probability metrics, restore a fresh model from the downloaded final checkpoint and re-evaluate Training Diagnostic and Validation. Compare complete probabilities and predictions within2e-5 tolerance. Final completion requires all three independent checks and remote metadata SHA verification. The frozen bundle's report states when independent verification is pending; the companion verified result index contains the final restore evidence and report.

Report Episode-cluster bootstrap95% intervals and paired differences on the same64 Validation Episodes. Eight Queries within an Episode are not independent. Single-seed differences do not prove architectural superiority. Changes to split, within-bank sampling and exposure are joint interventions and cannot isolate failure causes. No12M main training, main three-seed experiment or M6 is authorized. Test remains unaccessed.

# B0-2 Learnability Pilot

Development execution only. No12M main training, three-seed experiment, Test evaluation or M6. B0-1 Overfit artifacts and all frozen model/data/training code remain unchanged. This directory adds only a Pilot entry and fail-closed checks around the existing B0/M2 trainer, evaluation, checkpoint and publication components.

## Frozen data audit

| Bank | Episodes | Binding/target observations | Distinct pairs | Duplicate observations | Maximum multiplicity |
|---|---:|---:|---:|---:|---:|
| Pilot training |512|4096|3746|350 (8.5449%)|3|
| Validation |64|512|506|6 (1.1719%)|2|

Training multiplicity:3414 pairs appear once,314 twice,18 three times. Validation:500 appear once,6 twice. Train/Validation pair overlap is zero. All generated bank hashes match the existing dataset-lock.json. No data, partition, seed or sampling condition was changed. This is unseen-pair generalization within shared symbols, not unseen-symbol generalization. Repeated Query observations within an Episode are not independent statistical units. Modulo-three partitioning also changes conditional value support across splits; the existing nominal binomial diagnostics must not be treated as scientific confirmation.

Test remains RESERVED_NOT_GENERATED_NOT_ACCESSED under the existing frozen protocol. No Test bank is generated, mounted or evaluated by this entry. This is a reserved deterministic partition, not a claim that an independent Test file has already been materialized and archived.

## Execution contract

Use notebook.ipynb in the existing bingjiangli/minicells Notebook with Tesla T4, Internet and the existing GITHUB_TOKEN secret. One Save & Run All performs tiny CPU resume fixtures, exact frozen dependency/data checks, the three published Overfit prerequisite checks, fresh initialization checks, then sequential E0/E6/E7 Pilot execution and binary Release read-back. There is no rerun of canonical Smoke or Overfit.

Every arm starts from construct() and its recorded initial-state hash, never an Overfit checkpoint. E6/E7 initial hashes must match exactly. Only phase=pilot checkpoints with identical source/protocol/data/software/arm identity may resume. The startup check deliberately rejects differing initial hashes; do not replace expected hashes to bypass an environment difference.

E0 has10,130,688 parameters; E6/E7 each10,021,120. The original forward implementations are imported. Each update processes one complete512-token Episode; masked loss supervises only its8 Query answer positions. Full512-token forward is also used for every primary Validation evaluation. E6/E7 preserve chunk32, SWA window32, preceding-chunk Summary and read-before-write Store semantics. E0 uses full causal sequence attention, with no finite training window. No Prefill+Query primary metric is used.

Each arm consumes512 distinct Episodes exactly once,512 committed updates,262144 input tokens,4096 supervised/query target observations. All use identical training order. FP32, TF32 off, standard AdamW, constant LR and every16-update/120-second checkpoints remain unchanged. Validation occurs initially and at128/256/384/512. No early stopping, threshold tuning or budget extension.

## Integrity and return

Before publication, verify final checkpoint SHA and ledger, exact sequential update history, all512 input identities, all four full8192-way probability artifacts and their hashes/targets/Episode order/normalization/EM/NLL. Restore a fresh model from the final checkpoint and reproduce held-out Validation without optimizer updates. CSV curves, per-query probabilities, metrics, resources, parameter counts, exact configuration, model/optimizer/RNG/sampler/ledger checkpoint and a SHA256 manifest are included in each arm's Release bundle. Existing publication utilities verify every archive member after remote download and verify the metadata branch SHA. Pilot indexes use their own schema and run IDs.

The child entry reserves at least session time beyond its9000-second training deadline for publication; the Notebook stops its process group by10500 seconds, within the10800-second session cap. Preserve complete Output; attach it to the same Notebook for exact-identity resume. Completed curves/checkpoints are reused. Missing completed checkpoint Output stops execution rather than retraining. Publication retries reuse saved results.

## Local verification and pending hardware check

Four focused tests passed: frozen data/count/leakage and budget rejection, plus exact uninterrupted-versus-resumed tiny Pilot training for each original model class. Completed checkpoint re-entry performs zero extra updates and leaves checkpoint bytes unchanged. These are small CPU engineering fixtures, not Pilot results.

Local data preflight passes. CPU Torch2.6 initialization does not match the frozen Torch2.11 T4 initialization hashes, so the complete local preflight stops before any canonical update. Matching must be demonstrated in the existing T4 environment before real training. No fallback or hash substitution is authorized. The local machine has no CUDA or authenticated Kaggle API session; the supplied Notebook must be launched in Kaggle. Actual Validation curves, resource costs, final restore checks and remote Pilot artifacts remain pending execution.

E6/E7 inherit the known checkpoint metadata portability issue; loading uses the frozen repository classes. No historical checkpoint was rewritten. H1–H6 remain NOT_EVALUATED.

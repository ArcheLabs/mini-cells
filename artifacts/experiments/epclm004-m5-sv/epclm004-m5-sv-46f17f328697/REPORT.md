# M5-SV actual execution report

## Executive summary

Status: BLOCKED
H5_STATUS: INCONCLUSIVE

M6 is not authorized. Engineering/protocol PASS does not establish scientific support.
The frozen M5 baseline was a two-update engineering Smoke, not scientific validation.

## Experimental setup

Source: 46f17f3286976e63354fa78eb8b7f2ff2ec893f9
Config hash: d269de14b38ad7ce72a5a0fe5d85f78340b9394ea06ab12dc0f03355527be83e
Tokenizer hash: d1ed85988c25e27618ea8de951c9005758d57fef9e46832fa623a0812afe2d04
Hardware: {"gpu": "Tesla T4", "torch": "2.11.0+cu130", "cuda": "13.0", "precision": "fp32", "tf32": false}
Seeds: [{"model_seed": 51017, "data_seed": 62017, "store_seed": 72017}, {"model_seed": 51031, "data_seed": 62031, "store_seed": 72031}, {"model_seed": 51047, "data_seed": 62047, "store_seed": 72047}]
Data: independent modulo-three binding partitions; Test is hashed before Pilot and sealed until all main checkpoints finish.
Budgets: 1,000,000 inputs per arm/seed, 12M main total; separately bounded 65,536-input Pilot.

## Actual training and resource evidence

| Run | Updates | Input tokens | Targets | Validation EM | Peak bytes | Compute tokens/s | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot/50003/E6 | 32 | 16384 | 32 | 0.0 | 739846656 | 289.4841063720711 | 175175789 |
| pilot/50003/E7-FIFO | 32 | 16384 | 32 | 0.0 | 786957312 | 243.46418875523295 | 175206381 |
| pilot/50003/E7-Innovation | 32 | 16384 | 32 | 0.0 | 786957312 | 246.635960112793 | 175188013 |
| pilot/50003/E7-Random | 32 | 16384 | 32 | 0.0 | 786957312 | 244.9250140585437 | 175155565 |

Training FLOPs are not claimed: no validated complete operator counter is installed.
Static module counts, checkpoint write timings, optimizer/RNG/sampler identities and validation curves are retained beside each run.

## SV1 and SV2

Raw per-record innovation scores and E6 recoverability labels, including undefined single-class AUROC/AUPRC, are retained in Output.
Development summaries are in metrics/development/. Independent Test reports include admission precision/recall/coverage,
Hit@1/Hit@K/Recall@K/MRR, per-layer candidates, paired On/Off EM, conditional EM, rescue and regression.
Primary retrieval uses the last SWA layer, not an optimistic union. Oracle is diagnostic admission only.

## SV3 independent evaluation

Independent scientific evaluation is incomplete. No final accuracy or H5 support is inferred.

## Resource analysis and limitations

Each capacity report includes weights, SSM, local KV, derived Summary, predictor, payload and metadata bytes, plus actual query/prefill/read/update timing.
Same-checkpoint policy interventions are iso-parameter. Resource ratios compare measured dynamic bytes and Query latency to E6; no extra training group is added.
Query latency includes snapshot cloning and three-token computation. It is an instrumented diagnostic, not a production benchmark.
Capacity128 is inference-only on capacity64-trained weights; cosine strata are not semantic similarity. E3 remains an original Smoke reference.
Full query traces and Model/Optimizer/Runtime binaries stay in Kaggle Output; Git receives reports and hash manifests.

## Scientific conclusion

Innovation validity, admission, retrieval, utilization, final accuracy and resource acceptability must all pass the frozen criteria before SUPPORTED.
The actual H5_STATUS above is the only current verdict. No automatic M6 decision or budget expansion follows.

```json
"RuntimeError: PILOT_INCONCLUSIVE: held-out learnability gate failed; no main training or Test access, no budget extension"
```

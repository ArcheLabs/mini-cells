# M5-SV implementation and execution report

## Executive summary

Status: IMPLEMENTATION_IN_PROGRESS. H5: NOT_EVALUATED. No M5-SV T4 training or independent Test evaluation has run in this development environment. The frozen M5 run was a successful engineering Smoke, not evidence of H5. Actual measurements must come from the ordered Notebook and its archived outputs.

## Experimental setup

The versioned protocol declares unchanged M5 Mamba-1/SWA/EMA/embedding Store semantics, the frozen8192 tokenizer, independent pair partitions, three separately recorded model/data/store seeds, four trained arms, an evaluation-only Oracle and a sealed common Test bank. Main training is12M inputs; a separate Pilot is65,536 inputs total. Hardware target: Tesla T4, FP32, TF32 disabled. The run manifest records actual source/config/data/tokenizer hashes and execution identity.

## Measurements pending

SV1: per-record scores/recoverability, AUROC/AUPRC, admission precision/recall/coverage. SV2: per-layer ranked candidates, final-layer retrieval, Store On/Off, conditional EM, rescue/regression. SV3: all seeds/conditions, validation adequacy, paired Episode bootstrap, independent Test. Resource analysis: full static/runtime/Store breakdown, training peak, inference timings and checkpoint cost at both capacities. None of these scientific results are inferred from M5's Smoke.

## Scientific conclusion

Innovation validity: not evaluated. Admission advantage: not evaluated. Retrieval and utilization gains: not evaluated. Stable final accuracy gains: not evaluated. Resource tradeoff: not evaluated. Do not advance automatically to M6.

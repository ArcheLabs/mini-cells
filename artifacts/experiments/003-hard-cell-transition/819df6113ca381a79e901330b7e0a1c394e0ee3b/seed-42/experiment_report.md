# Experiment 003 — Hard Cell Transition

Status: **DEVELOPMENT_INCOMPLETE**

| Model | Tokens | PPL | Executed GEMM FLOPs/token | Latency ms |
|---|---:|---:|---:|---:|
| fixed-cell | 8904000 | missing | missing | missing |
| global-router | 8446000 | missing | missing | missing |
| soft-cell | 10000000 | 12.18219189302685 | 338020352.0 | 93.03747474998545 |
| hard-cell | 10000000 | 14.824803405492208 | 43833797.952 | 62.44784659998004 |

Gates: `null`

Development evidence only. All arms use SDPA math; actual GEMMs include attention. Non-matmul operations are excluded.
Soft routing uses a conditional-mean relaxation; hard training uses a biased selected-value STE.
Missing/failed arms do not pass gates. Three paired development seeds are required for a matrix summary.
No general superiority claim about CLM, Transformer or MoE follows from this experiment.

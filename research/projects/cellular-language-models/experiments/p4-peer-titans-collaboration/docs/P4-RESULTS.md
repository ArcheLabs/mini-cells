# P4-M1 single Cell competence

Status: M1_COMPLETE. CPU FP32 synthetic development, PyTorch 2.6.0+cpu.

The frozen M3 Gate 5 control reached 100% Recall@1 after 128 online updates. This control uses the deterministic initializer rather than the smoke-trained weights.

| Bindings | Recall@1 across seeds | Random | Mean cosine | Mean MSE | Unknown false positive |
|---:|---:|---:|---:|---:|---:|
| 4 | 100.00% (100.00%–100.00%) | 25.00% | 0.9983 | 0.0037 | 8.33% |
| 8 | 100.00% (100.00%–100.00%) | 12.50% | 0.9971 | 0.0063 | 0.00% |
| 16 | 100.00% (100.00%–100.00%) | 6.25% | 0.9288 | 0.1389 | 0.00% |
| 32 | 29.17% (21.88%–34.38%) | 3.12% | 0.3140 | 0.9030 | 0.00% |
| 64 | 5.21% (3.12%–9.38%) | 1.56% | 0.1665 | 0.9714 | 0.00% |

2,048 inner updates including the control. Wall time 5.84 seconds. Maximum restore error 0. Each read checked the fast weights, momentum, update counter and metadata for mutations. Scoring uses the candidate bank only after the Memory returns its vector.

Each size has three independent mappings. Keys and values use separate QR draws up to 32 bindings and independent normalized Gaussian draws at 64. The change in vector geometry and equal update count across sizes limit interpretation as an intrinsic capacity bound. Unknown-key false positives use a fixed maximum candidate cosine threshold of 0.8.

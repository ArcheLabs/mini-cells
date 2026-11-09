# EPCLM-004 M2 — Baseline Architectures

**M1 engineering: PASS and frozen. M2: implemented / CPU validated; T4 validation pending.**

- [Implementation and validation report](REPORT.zh-CN.md)
- [One-launch Kaggle Notebook](notebook.ipynb) — T4 + Internet + existing `GITHUB_TOKEN`.
- [Fixed protocol / model configurations](configs/protocol.json)
- [Actual parameter counts and common initialization hashes](audit/parameter-counts.json)
- [Official source pins and adaptations](configs/upstream-lock.json)
- [CPU evidence](audit/validation.json)
- [M1 acceptance and saved Kaggle ID](../docs/M1-ACCEPTANCE.zh-CN.md)

| Arm | Definition | Static parameters | Difference from frozen E0 |
|---|---|---:|---:|
| E0 | Frozen full causal Attention | 10,130,688 | 0% |
| E1 | Same blocks/initial weights, causal W32 SWA | 10,130,688 | 0% |
| E2 | Samba-aligned Mamba-1, 18 residual blocks | 10,092,288 | −0.379% |
| E3 | 4 Mamba-1 / 4 SWA alternating blocks, MLP after each | 10,135,808 | +0.051% |

The matched target is the frozen 10.13M E0, rather than modifying E0 to exactly
10M. Both SSM arms use state size 20, expansion 2, convolution 4; E3 FFN is 848.
These declared matching choices differ from Samba's default state size 16.

E2/E3 reuse the pinned official **PyTorch Mamba-1 reference**, not Mamba-3 and not
an optimized CUDA kernel. Reference costs must not be presented as native Mamba
performance. No Mamba/causal-conv1d/FlashAttention installation is required.

The Notebook runs common CPU regressions, then GPU causality/cache/gradient
checks and **two fresh updates per arm**, with identical training inputs,
8192 tokenizer, FP32, TF32 off, AdamW, LR and checkpoint policy. This is a new
bounded M2 correctness run, not extra M1 training to remedy zero recall.
No quality ranking, 20M Screening or 100M Confirmation is authorized.

```bash
# From this directory, project Python >=3.11 environment.
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python -m pytest -q tests
OMP_NUM_THREADS=1 python count_parameters.py
OMP_NUM_THREADS=1 python run.py --device cpu --tiny \
  --assets /tmp/epclm004-m1-assets --output /tmp/epclm004-m2-tiny
```

Checkpoint policy: every 8 updates or 120 seconds, plus final/timeout flush.
Before-step rejection preserves the valid prefix; unsafe optimizer mutations
restore the last durable model/optimizer/RNG/sampler/ledger together. A hard
process crash may lose up to 7 updates. Do not confuse an in-memory committed
prefix with a durable checkpoint. Completed arms are not trained again.

`--publish-only` republishes saved reports through the existing main-based
publisher. Weights stay in Kaggle Output. A retry uses the same source/config
and verified arm checkpoints; a new source does not silently reuse old weights.

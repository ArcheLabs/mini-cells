# EPCLM-003/004 — Unified Memory Architecture, 10M

**Current milestone: M1 implemented and CPU validated; target T4 execution pending.**
E0 and unified data are available; M2–M7 remain pending. Scientific hypotheses
H1–H6 remain `NOT_EVALUATED`.

- [M1 实施与实际结果报告](docs/M1-IMPLEMENTATION-REPORT.zh-CN.md)
- [Kaggle Notebook](notebook.ipynb): T4 + Internet + existing `GITHUB_TOKEN`, Save & Run All.
- [Data / Batch / Run contracts](docs/M1-DATA-CONTRACT.md)
- [Frozen 8192 BPE identity](configs/tokenizer-lock.json)
- [CPU evidence and target GPU limitation](audit/m1/validation-summary.json)

The Notebook runs bounded CPU tiny validation, E0 MQAR (32 GPU updates), and
E0 WikiText (4 GPU updates). It never starts 20M Screening or 100M Confirmation.
Each stage publishes through the existing GitHub result publisher. Private
checkpoints stay in Kaggle Output; `run.py --publish-only` retries publication.

```bash
# Run from this directory using the Python >=3.11 project environment.
python -m pip install -r requirements-m1.txt
OMP_NUM_THREADS=1 python -m pytest -q tests
python scripts/count_parameters.py --config configs/model-e0.json
OMP_NUM_THREADS=1 python run.py --profile cpu-tiny --task mqar \
  --assets /tmp/epclm004-assets --output /tmp/epclm004-cpu
```

Input, supervised targets, query targets, padding, and committed updates are
reported separately. Source/config/data/tokenizer mismatches fail closed. E0
uses full causal attention over each supplied sequence and no training KV cache.

## Preserved M0 audit

- [原始实施规格 v0.1](SPEC-v0.1.zh-CN.md) — preserved verbatim; §20 limits this first delivery to M0.
- [M0 实施计划、风险与 M1 最小范围](docs/EPCLM-004-IMPLEMENTATION-PLAN.md)
- [Architecture reuse report / 代码复用报告](docs/ARCHITECTURE-REUSE.md)
- [Observed environment manifest / 实测环境清单](configs/environment.yaml) — observations, not an installation lock.
- [Audit evidence / 审计证据](audit/evidence.json)
- [Implementation deviations and unresolved choices](implementation_deviations.md)
- [EPCLM-002 理论文档（中英）](../../literature/README.md)

The study compares full attention, SWA, Samba-aligned Mamba-1, Titans MAG-lite
and a three-path candidate at approximately 10M static parameters. B'MOJO-inspired
storage and Gated-SWA are later stages. No architecture or resource advantage is
assumed. The passed M3C Mamba-3 reference is a distinct implementation and study.

本次已经完成 M1 代码、CPU 小样本真实更新与恢复验证。实际 T4 短训练、显存和吞吐仍待用户运行上面的新 Notebook；本机 CPU / K620 环境不能替代该结果。不要运行历史实验入口来代替本研究。

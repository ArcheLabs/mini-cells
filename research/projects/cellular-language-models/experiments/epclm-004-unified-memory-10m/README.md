# EPCLM-003/004 — Unified Memory Architecture, 10M

**Current milestone: M0 audit complete; M1–M7 pending.** No training or new model
implementation was performed in this milestone. Scientific hypotheses H1–H6
remain `NOT_EVALUATED`.

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

下一步是按规格审查 M0 后进入 M1：复用现有 Transformer 和执行基础设施，先完成统一数据、预算记账与 E0 的最小 harness。当前没有可运行的新 Kaggle 训练入口；不要运行历史实验入口来代替本研究。

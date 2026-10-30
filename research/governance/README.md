# Research Governance

This directory contains shared records and controls that span projects. It is
not a second home for experiment code or results.

- [`catalog.yaml`](catalog.yaml) lists the research projects.
- [`registry.yaml`](registry.yaml) indexes registered studies and their status.
- [`validations/`](validations/) preserves frozen protocols and decisions.
- [`audits/`](audits/) records cross-study reviews and evidence boundaries.
- [`archive/`](archive/) preserves retired workflows and historical records.
- [`experiment-index.md`](experiment-index.md) and
  [`notebook-index.md`](notebook-index.md) explain where active material lives.
- [`historical-migration-map.md`](historical-migration-map.md) maps the former
  stage folders into current project ownership.

Treat project experiment records and frozen protocol files as authoritative.
The registry and indexes help locate them; they do not change their status.

Current development launch policy: [one-launch automatic workflow](automatic-development-workflow.md). Manual registration fields are not a prerequisite for development execution.

## MiniCells Evidence Map

统一研究证据入口：[MINICELLS_EVIDENCE_MAP.md](MINICELLS_EVIDENCE_MAP.md)。它汇总原始事实、研究解释与历史决策，不取代项目协议、结果或现有 registry，也不授予实验执行权限。

- 正式实验启动前，按现有治理流程完成预注册，并关联稳定 Claim ID、Research Line 与已有证据；探索性研究记录目的、范围和预算。例行软件测试无需重复登记。
- 实验关闭 PR 必须同步更新 Evidence Map 的 Experiment Registry、对应研究线、相关主张及发生实质变化的问题/评估；失败、无效、中止同样登记。
- 设计、数据、Seed、指标、预算或分析变更保留原版本，并记录结果可见前后；不得补造历史预注册或覆盖旧结果。
- 原始记录优先。UNKNOWN/NOT_RECORDED 保留缺失信息；冲突使用 CONFLICT_REVIEW_REQUIRED，并列原来源，不擅自选边。
- 主要研究阶段结束时复核 Map；历史迁移覆盖与尚未完成的语义核验见 [完整性审计](audits/EVIDENCE_MAP_COMPLETENESS.json)。不将自动目录清单当作最终科学验收。

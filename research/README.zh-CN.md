[English](README.md) | [中文](README.zh-CN.md)

# MiniCells 研究项目

研究资料按项目组织。进入对应项目目录，即可查阅研究问题、实验单元、Notebook、实现代码、报告和已有证据。

## 研究项目

| 项目 | 范围 |
|---|---|
| [`01-foundations/`](01-foundations/README.md) | 早期语言动力学与训练可行性 |
| [`02-self-organization/`](02-self-organization/README.md) | 组织形成、招募、分化与特征生成 |
| [`03-routing-and-growth/`](03-routing-and-growth/README.md) | 路由、增长与粒度 |
| [`04-continual-learning/`](04-continual-learning/README.md) | 持续学习核心机制 |
| [`05-language-validation/`](05-language-validation/README.md) | 已退役的 CLM-0.4 语言验证线 |
| [`06-native-clm/`](06-native-clm/README.md) | 原生语言模型训练与持续学习测试 |
| [`07-safe-model-evolution/`](07-safe-model-evolution/README.md) | 预训练模型的安全演化 |
| [`08-hybrid-clm/`](08-hybrid-clm/README.md) | HybridCLM 及其关联路由实验 |

## 跨项目研究治理

- [`audits/CLM_CAPABILITY_CEILING.md`](audits/CLM_CAPABILITY_CEILING.md)：当前证据支持的结论及明确的 No-Go 边界。
- [`audits/RESEARCH_LEDGER.md`](audits/RESEARCH_LEDGER.md)：实验结果、限制与工程价值记录。
- [`audits/HISTORICAL_RESEARCH_ASSET_MAP.md`](audits/HISTORICAL_RESEARCH_ASSET_MAP.md)：历史研究线的证据定位。
- [`catalog.yaml`](catalog.yaml) 和 [`registry.yaml`](registry.yaml)：机器可读索引。
- [`MIGRATION_MAP.md`](MIGRATION_MAP.md)：项目归属和旧路径兼容映射。

## 研究纪律

冻结协议与正式决策具有权威性。后续解释不能把失败决策升级；有用的机制也不能证明模型级或语言级持续学习。保留每次运行的 source commit 与身份；路径迁移前核对引用并保留兼容入口；每项实验只归属一个研究项目。

Experiment 004 的手动入口现为 [`08-hybrid-clm/experiments/004-cell-graph/notebook.ipynb`](08-hybrid-clm/experiments/004-cell-graph/notebook.ipynb)。Seed 42 的 `reference` 与 `graph` 归档仍保留各自记录的 source commit，矩阵汇总仍为 `DEVELOPMENT_INCOMPLETE`。

[English](README.md) | 中文

# MiniCells 研究

研究资料按长期研究项目组织。每个项目集中保存研究问题、实验记录、代码、Notebook、报告和项目专属结果。


## 研究项目

| 项目 | 研究范围 |
|---|---|
| [细胞计算](projects/cellular-computing/README.md) | 细胞计算、自组织、路由、增长与稀疏执行。 |
| [持续学习](projects/continual-learning/README.md) | 持续学习机制、验证及其已知限制。 |
| [细胞语言模型](projects/cellular-language-models/README.md) | 原生与混合细胞语言模型架构及语言验证。 |
| [安全模型演化](projects/safe-model-evolution/README.md) | 对预训练模型进行安全、可测量的修改。 |
| [CLM Native](projects/clm-native/README.md) | Bounded-candidate dynamic sparse computation and causal/cost validation. |

## 共享研究治理

[`governance/`](governance/README.md) 保存实验注册表、冻结协议、审计和历史索引。研究项目目录拥有研究内容；治理目录记录状态与规则。

## 研究纪律

- 每项实验只归属一个研究项目。
- 保留冻结协议、源代码提交、运行身份和负面结果。
- 实验结果与所属实验放在一起；仓库级发布产物保留在顶层 `artifacts/`。
- 跨项目共享代码放在 [`../shared/`](../shared/)；项目专属逻辑留在项目或实验目录中。
- 运行成功、机制有用或路径可视化，都不能单独证明科学主张成立。

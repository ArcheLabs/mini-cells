[English](README.md) | 中文

# MiniCells

MiniCells 是一个 Cellular Language Model（CLM，细胞语言模型）研究项目。CLM 将模型组织为稀疏激活、可独立更新和验证的神经 Cell；模型可以局部学习，在更新破坏既有行为时拒绝提交，并在现有 Cell 无法安全吸收新知识时生长新的 Cell。

## MiniCells 是什么

MiniCells 研究能否把模型状态划分为具有明确依赖边界和事务边界的路由计算单元：

$$\boxed{\text{Cell}=\text{可独立路由}+\text{可独立修改}+\text{可独立验证的模型状态}}$$

NCA 提供了局部状态、局部交互、生长和自组织的原始视角。当前 CLM 不要求字面意义上的二维网格；稀疏动态 Cell 图是更一般的抽象。

## 为什么使用 Cell

传统 MoE 主要提供稀疏计算；CLM 还把稳定稀疏路由用作依赖索引，以确定 Cell 更新后必须复查哪些历史计算。生长会增加可独立修改的新状态，而不是迫使旧状态吸收不兼容的学习。

## 当前研究状态

| 层次 | 状态 |
|---|---|
| Cellular/NCA 语言动力学 | 历史实验支持 |
| 稀疏路由 Cell | 实验支持 |
| 依赖域回归安全 | 受控合成环境中得到支持 |
| 事务回滚 | 作为安全机制得到支持 |
| 生长恢复可塑性 | **支持——Core Validation 004，3/3 seeds** |
| 自然语言持续学习 | **尚未验证** |
| 5–10M CLM-0.4 pilot | 下一步 |
| 30–50M 正式 CLM-0.4 模型 | pilot 通过后计划 |
| JAM 原生分布式 CLM | 未来工作 |

## CLM 核心闭环

$$\boxed{\mathrm{CLM}=\text{稀疏路由}+\text{依赖验证}+\text{事务学习}+\text{自适应 Cell 生长}}$$

路由后先尝试更新已有 Cell；若依赖域验证安全则提交，否则回滚并尝试生成、训练和验证新 Cell，最后原子提交或回滚。

## 实验证据

Core Validation 002、002B、002C 分别否定了精确单地址、稀疏组装和 oracle 稀疏组装写入作为前提。003 表明依赖域验证可以拒绝不安全候选，但仅靠拒绝不能提供足够可塑性，因此官方结论仍为 No-Go。004 加入事务式生长，并通过全部正式 seeds：`80411`、`80412`、`80413`。

**CLM 核心学习闭环已在受控合成环境中完成实验验证。**详见[最终机制报告](research/reports/clm-core-mechanism-0.4.zh-CN.md)与[规范实验产物](artifacts/experiments/)。

## 已验证与未验证

已支持：稳定路由下的历史依赖范围、安全回滚，以及被拒已有 Cell 更新后的生长恢复可塑性。尚未证明：通用自然语言持续学习、彻底解决灾难性遗忘、无限期有界生长或 LLM 规模有效性。

## 仓库结构与复现

- [`research/`](research/README.zh-CN.md)：四阶段历史、目录、报告、协议和历史来源。
- [`artifacts/experiments/`](artifacts/experiments/)：不可变的规范实验证据。
- [`src/minicells/`](src/minicells/)：研究实现；[`research/notebooks/`](research/notebooks/)：保持稳定路径的 notebooks。

运行 `python -m pytest -q` 与 `./tools/test_all.sh`。协议、notebook 和 artifact 索引见 [`research/catalog.yaml`](research/catalog.yaml)。文档整理不应重新生成正式结果。

## MiniJAM / JAM 与 CLM-0.4

候选更新、验证、提交/回滚和状态迁移适合映射为 JAM 风格的确定性状态迁移，但 004 是链下受控研究结果；JAM/MiniJAM 是目标执行环境，不是 004 结论的一部分。下一步是 5–10M 参数的数学+故事 pilot；通过后才考虑 30–50M 正式候选。本次整理不实现训练。

## 许可证

见 [LICENSE](LICENSE)。

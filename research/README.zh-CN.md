[English](README.md) | 中文

# MiniCells 研究

CLM 是由可独立路由、修改和验证的神经 Cell 构成的稀疏图。受控核心闭环会路由输入、尝试局部更新、验证依赖于被修改 Cell 的历史计算并提交或回滚；拒绝后可通过有界生长增加新状态并原子重试。

该核心闭环仅在**受控合成环境**中得到支持。自然语言持续学习、语言规模的语义路由涌现、长期有界生长和 LLM 规模行为均尚未验证。

## 四个研究阶段

1. [基础](stages/01-foundations/README.zh-CN.md)：Echo、NCA 语言动力学、1D/2D tissue、settling 与训练机制。
2. [自组织](stages/02-self-organization/README.zh-CN.md)：稀疏拓扑、招募、分化和 trait genesis。
3. [路由与生长](stages/03-routing-and-growth/README.zh-CN.md)：Cell 转变为可路由、可独立修改的计算状态。
4. [持续学习核心](stages/04-continual-learning-core/README.zh-CN.md)：写地址假设失败、依赖域事务和生长恢复可塑性。

建议先读[研究历史](reports/research-history-0.4.zh-CN.md)，再读[最终机制报告](reports/clm-core-mechanism-0.4.zh-CN.md)。机器可读索引见 [`catalog.yaml`](catalog.yaml)，不可变证据见 [`../artifacts/experiments/`](../artifacts/experiments/)。

## 研究资产

- [实验实现](experiments/README.md) 是按阶段组织的适配层，复用 `src/minicells/` 中的代码。
- [Notebook 资产](notebooks/README.md) 按阶段组织，历史实验 ID 与内容保持不变。
- [Core validations](validations/) 保存冻结协议和双语摘要。
- [规范 artifacts](../artifacts/experiments/) 是不可变科学证据。

Core Validation 004 通过 3/3 正式 seeds，但这不等于自然语言持续学习已经成立。下一步是 5–10M 参数受控语言 pilot；只有 pilot Go 后才进入 30–50M 正式候选。

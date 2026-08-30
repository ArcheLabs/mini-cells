[English](README.md) | 中文

# 阶段 05 — 语言级验证

> 状态：当前研究阶段

## 核心问题

Core Validation 004 已在受控合成函数环境中验证 CLM 核心闭环。阶段 05 要回答：这一闭环迁移到真正的自回归 token 级语言模型后是否仍然成立？

pre-0.4 阶段冻结的核心机制由以下部分组成：稳定稀疏路由、依赖域回归验证、事务式提交/回滚，以及由拒绝触发的生长恢复可塑性。阶段 05 不重新搜索这套机制，而是测试它在 next-token prediction 与连续语言任务下能否继续工作。

## 第一个实验

本阶段首先执行 [CLM-0.4-mini 语言级验证](../../validations/clm-0.4-mini-language-validation/README.zh-CN.md)。它是约 5M 参数的 decoder-only pilot，包含：

- continual phase 冻结共享语言 backbone；
- 两层细粒度 sparse Cell-FFN 作为可修改状态；
- 正式实验使用确定性的 out-of-band address plane；
- 受控数学与故事世界 curriculum；
- 精确 dependency-scoped local validation 与隐藏 full-history oracle；
- transaction rollback 与 monotonic private-Cell growth；
- 完整 transaction、Cell lineage、routing、成本与 state hash 可观测性。

显式地址是有意保留的控制变量。本实验验证的是**语言级机制迁移**，不是自动语义寻址。可以运行 shadow semantic router 诊断，但它不得控制正式 commit。

## 三段决策流程

1. **M0 — execution smoke**：验证 transaction journal、routing、validation、rollback、growth、checkpoint 与 replay 全流程能正确执行；不产生科学结论。
2. **M1 — 正式 0.4-mini pilot**：在 192 个受控 token-level continual-learning transactions 中验证 CLM 闭环；要求三颗未参与开发的正式 model seeds 全部通过。
3. **M2 — scale rehearsal**：仅在 M1 通过后运行更长 stream，观察 state growth、dependency scope、checkpoint size 与 validation cost，判断是否具备放大到 30–50M 的工程条件。

只有：

`M1 Go AND M2 Go`

才允许进入 30–50M CLM-0.4 正式候选。

## 正结果允许表达什么

如果 M1 通过，可以说：

> 依赖域事务式生长闭环已从受控合成机制环境迁移到受控 token 级语言模型。

不能据此声称通用自然语言持续学习、自动语义路由、无限期有界生长、LLM 尺度行为或 JAM 原生训练已经成立。

## 与此前阶段的关系

- **基础**：token 级语言建模与局部神经计算。
- **自组织**：能力可以形成结构化局部状态，而不必依赖单一整体参数块。
- **路由与生长**：稀疏 routed Cells、渐进扩容与 Cell reuse。
- **持续学习核心**：dependency-addressed safety、transactional commit/rollback，以及当可塑性受约束时通过 growth 引入新自由度。

本阶段首个实验的正式协议必须在查看 formal seeds 结果前冻结在 `research/validations/clm-0.4-mini-language-validation/`。

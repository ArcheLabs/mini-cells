# Core Validation 003：依赖域事务学习

冻结结论：`DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED`。注册合成环境中 dependency-scoped gate 显示 structural escape 与 false-safe 为零，粒度降低验证覆盖；但安全候选拒绝也阻止了过多有效学习，因此总体 No-Go 不变。证据见规范 [artifact](../../../artifacts/experiments/core-validation-003-dependency-scoped-transactional-learning/)。

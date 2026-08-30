# 持续学习核心：从写地址到依赖域事务式生长

**阶段问题：**同时提供回归安全与足够可塑性的最小受控机制是什么？

001/001B 研究知识包含与残余记忆。002、002B、002C 依次得到 `WRITE_ADDRESSABILITY_NOT_SUPPORTED`、`SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED`、`ORACLE_SPARSE_ASSEMBLY_NOT_SUPPORTED`，从而排除精确语义写地址作为前提。

003 将地址从知识内容转向执行依赖；在冻结路由/共享状态下 structural escape 与 false-safe 都为零，但官方结论仍是 `DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED`：安全存在，可塑性不足。004 加入回滚触发的上下文域生长，三个 seeds 全部通过，结论为 `GROWTH_RESTORED_PLASTICITY_SUPPORTED`。

这是受控合成证据，不是自然语言持续学习。详见[最终报告](../../reports/clm-core-mechanism-0.4.zh-CN.md)、[验证协议](../../validations/)与[规范 artifacts](../../../artifacts/experiments/)。

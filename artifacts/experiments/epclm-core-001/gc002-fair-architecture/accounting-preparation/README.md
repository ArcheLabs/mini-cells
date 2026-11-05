# GC002 后续准备：账本、参考算术计数与运行候选

**PREPARATION_ADVANCED_FREEZE_CANDIDATE_NOT_AUTHORIZED**。无真实语料训练，无正式模型权重更新。

本附件基于当前 main 的独立结果分支，不覆盖此前 CPU/GPU 准备结果或 GC001。执行来源和逐文件 SHA 记录在 JSON identity；本次完整准备执行于 `3fa58b6`，随后 `339ec9d` 只增加相同消费流的回归断言。代码分支 `codex/epclm-gc002-fair-architecture`。

- [阶段结论](reports/research-decision.json)
- [四模型完整形状算子清单](profiling/training-flops.json)：Batch 4 × TBPTT 64，前向、反向、裁剪、零学习率 AdamW，权重不变。当前本地算子覆盖已通过；这是参考标量算术口径，**不是实测硬件 FLOPs，也不是正式 Compute-Matched 就绪证明**。
- [已复用的 CUDA 工程证据](audit/reused-engineering-evidence.json)：固定 `187d7f498...` 结果提交和相同 KV 代码哈希，不重复测量。
- [Token 审计](audit/token-exposure.json)：原始语料没有复制到本地主机，完整期望流重建由准备 Notebook 执行。历史真实逐 Update 输入哈希未记录，无法事后补证；不把期望流重建冒充运行时观测。
- [具体运行计划](RUN-PLAN.zh-CN.md)、[冻结候选](freeze-candidate.json)。

A/B 候选为 4 模型 × 3 Seed，每作业 250K Token，共 12 作业、3M Token。每作业两小时资源上限意味着最多预留 24 GPU 小时，**不是实测成本**，不含 C/D/Pilot，当前批准 Token 仍是 0。

独立人工评审保持 PENDING，独立 Test 未访问。真实 live Update/FLOPs 计量、目标环境验证、数值 Compute 预算及记忆矩阵仍待完成。没有生成 PROTOCOL_FROZEN 或训练授权。

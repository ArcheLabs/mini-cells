# M1 工程验收：PASS

按用户审计意见冻结 M1，源提交 `b089f18f13893920e9e996ce91ce39bec642bc6d`。

原始 CPU/MQAR/LM 三个结果分支保持不变；对应发布文件 SHA256 已与 metadata.json 核对。[验收记录](../audit/m1/acceptance.json)登记分支、不可变 commit、参数/Token 记账和原始零召回。

T4 FP32 上 10,130,688 参数 E0 已完成 MQAR 32 updates / 128 Query targets，LM 4 updates / 893 targets，均有有限梯度、真实参数更新和新实例 Checkpoint 恢复。MQAR 0/32，常量预测 474，去 binding 后同样零召回。该结果不用于否定记忆机制，也不追加补救训练。

原 decision.json 的 publication=PENDING 是推送前写入的旧字段；运行日志含远端 SHA 核验，且结果分支已经实际读取、文件哈希通过。此次通过附加验收记录说明发布完成，不覆盖历史 decision。

同 Seed 证据：本地 torch2.6 与 Kaggle torch2.11 的 CPU tiny 运行，全部训练输入 SHA 一致、Ledger 一致，逐次 Loss 最大差 9.54e-7。它支持数据/CPU路径可复现，不等于已经完成同一 T4 环境的独立重复训练。已有中断/恢复测试另外给出逐元素一致证据。不额外运行 GPU 重复来弥补零召回。

用户补充本次已保存版本 ID `366932788`，对应 `bingjiangli/minicells?scriptVersionId=366932788`。验收记录按 USER_SUPPLIED_SAVED_VERSION_ID 补录三个同次 Notebook 阶段的关联；未独立调用平台 API 复核，不回填或覆盖原始 ID_PENDING manifest。

M1 代码、配置、冻结 Tokenizer 和原始实验记录均保留。M2 新目录只复用 M1 组件，不修改冻结 E0。完整 Checkpoint 保存频率优化属于 M2。H1–H6 始终为 NOT_EVALUATED。

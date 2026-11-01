# GC002 下一阶段运行计划（待批准，不授权训练）

当前准备不再把历史缺失的逐 Update 输入哈希伪装成可追溯证据。已有 GC001 的 250K/977 计数、最后段 192K→250K 和原始资产身份已核验；完整实际消费顺序未记录，作为明确历史限制保留。新运行必须记录、保存并恢复实时 exposure ledger。

[freeze-candidate.json](freeze-candidate.json) 为具体冻结候选，非 `PROTOCOL_FROZEN`。`freeze-candidate.sha256` 仅校验候选内容，不是训练许可。

## A/B 候选矩阵

4 模型 × 3 独立 Seed，共 12 个训练作业，每个 250,000 prediction events，总计 3,000,000 Token。A 与 B 中相同配置只执行一次，不把重复引用当独立实验。每次 Kaggle 只运行一个作业，保留现有 5400/600/1200 秒分配，总上限 7200 秒。预算未获批准，当前实际授权为 0 Token。

三个拟定 Seed 为 173020021、173020123、173020227；冻结前再次核对 registry。每个 Seed 内共享数据顺序，各模型独立初始化；C1/IC 保持对应的相同参数初值设计。RMSNorm/输出共享权重与模型活跃参数按已有报告计算，T1-PM 为 967,808 参数。

训练用相同数据/Tokenizer/文档顺序、Batch=4、TBPTT=64、FP32、AdamW lr=3e-4、betas=.9/.95、weight_decay=.1、clip=1，97 Update warmup 后 cosine；本版调参次数为 0。Transformer 不沿用 GC001 逐 Token 重算前缀的低效训练包装器；平行 Teacher-Forced Chunk 与跨 TBPTT detach KV 的训练正确性仍需独立验证，不能拿推理 KV 通过代替该验证。

## Token 与 FLOPs 账本

unique 指 (Document ID, target position)，不是不同词表 Token ID。live ledger 记录输入/目标 ID、时序顺序、accepted Update、unique/repeated 数、数据顺序链哈希和 Token-equivalent epochs。溢出/rollback 不提交；checkpoint 必须将 ledger、采样器位置、状态、optimizer/scheduler/RNG 一并原子提交。`exposure.py` 提供计数与恢复组件，尚未接入获批的正式 Trainer。

`flops_audit.py` 覆盖前向、反向、梯度裁剪及 AdamW 运算清单。合成数据、零学习率且核验权重不变；这些不是训练模型，也不消费真实训练 Token。FMA=2，显存拷贝/索引/比较排除，transcendental 与 fused normalization 使用公开的参考算子口径，报告完整 operator inventory 和未知浮点算子。

**该算子标量操作口径不等同于 GPU 硬件指令 FLOPs。** 所有未知浮点算子 fail closed；即使 inventory 已覆盖，也不能自动宣布正式 Compute-Matched 就绪。目标环境的 live Update/文档边界/Mask、反向算子和计量标准仍须验证。正式目标采用验证后的 C1 250K 对应累计预算，25/50/75/100% 检查点；离散 Update 超过 ±5% 时不能标完成，不准临时改变预算。

## 记忆与推理

M1–M5 与 LM 分轨。距离点保留 128/512/1024/4096/8192；拟定持久字节预算 16KiB/64KiB/256KiB/1MiB，包含每 Session 的 KV/State、保留 Token 缓冲、计数器与明确元数据。模型权重和共享 runtime 单独报告。Cellular 固定 8192 张量字节不通过空填充或裁剪伪装预算曲线；KV 长度由实际存储与元数据导出。episode 数、实体生成器、输出词表、重置/跨 Chunk 策略仍需冻结，当前不能运行正式记忆矩阵。

## GPU 小时与审批边界

12 作业 × 每次最多 2 小时，A/B 可申请的会话预留上限为 **24 GPU 小时**；这是资源上限，不是实测训练时间预测，不含 C/D 与 Pilot。历史未优化 T1 的 4013 秒/250K 仅是历史参考，不能当新 PM/并行训练吞吐率。实际 GPU 小时需在独立批准的最小正确性/吞吐 Pilot 后给出；Pilot 结果不得用于修改已冻结的效应判定或模型选择规则。

下一次准备 Notebook 仅复用既有 CUDA KV/推理证据、核验/重建数据流并列出 full-shape 算子。它不会训练、不会扩大预算，也不替代仍待完成的训练路径验证。全部门禁满足后再冻结总协议并申请 Pilot/正式训练。独立 Test 保持封存，Human Review 仍为 PENDING。

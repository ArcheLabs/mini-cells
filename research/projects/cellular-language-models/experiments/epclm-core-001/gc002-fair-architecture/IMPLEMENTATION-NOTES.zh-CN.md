# GC002 v1.0 草案：实施说明与阶段边界

## 历史事实澄清

1. Native T1 为 dim=192、layers=4、heads=6、kv_heads=2、ffn=480、tied embeddings。所有独立可训练参数计数为 1,894,080；共享 Embedding/Head 只算一次，但二者运算成本分别计入 FLOPs。
2. 采用紧凑 GQA 时 FP32、128 Token 的 KV 载荷是 `2 × 4 × 128 × 2 × 32 × 4 = 262,144 Bytes`。原文 786,432 Bytes 是按全部 6 个头扩展后的 K/V 载荷。扩展张量只作为临时 Attention 工作区；实际持久 backing storage 单独测量。旧 516 Bytes 是 Token 包装器，不能当成 KV。
3. 每个 GC001 新模型最终消费计数均为 250,000、977 Updates；976×256+144=250,000。416,027 是整个可用训练语料的 prediction tokens，并非实际消费预算。
4. `learning()` 在 250/500/750/977 Updates 分段调用 Trainer，最后段从当前运行的 750 Update Checkpoint 起点 192,000 继续，新增 58,000。`elapsed_seconds` 已跨段求和；返回的保存、恢复耗时字段仍是最后段，不能作为总耗时。
5. 源码与完整曲线支持这些统计口径；“未跨运行复用”仍只有执行政策记录，未读取私有 Checkpoint 独立证明。实际 unique/repeated 预测事件和完整消费序列未发布，保持 null，不从数据集大小推算。unique 指 (Document ID, target position)，不是词表不同 Token ID 数。
6. 历史 T1 对每个事件重放前缀；训练期也走这种逐事件实现。缺少推理 KV 不是正常并行 Teacher-Forced 训练慢的解释。后续正式训练需独立审计向量化 Teacher-Forced 路径，不能把历史 wall time 当成优化后的 Transformer 架构效率。

## 新 Transformer 语义

`CachedTransformer` 复用 Native 层组件与 state_dict 名称，Native 尺寸可 strict-load 原权重布局；本轮只用随机权重验证，不下载旧权重。使用从序列开始即启用的因果 Sliding-Window Mask，绝对 RoPE Position ID 连续增长。逐 Token 缓存只处理新 Token，每层只保留紧凑 KV，持久存储有界，不保存 Token 历史。

未缓存参照使用**相同模型、相同绝对位置和相同逐层窗口 Mask**对整个输入执行因果前向。它并不是历史 GC001 的“截取最后 128 Token，从头重算并将位置清零”路径。新旧路径只要求未截断前一致；不能在窗口滑动后声称完全复现历史结果。新语义必须在训练前冻结，未来模型不得悄然替换历史结果。

每个独立文档必须新建 cache，结束丢弃；持续记忆任务跨 Chunk 传递同一个 cache，并保持绝对位置。snapshot 绑定配置/权重哈希，拒绝其他模型快照，恢复后验证继续计算。正常 decode 路径不做权重哈希或历史重算。

## 参数与内存

参数搜索只依据结构计数，不看验证/留出效果。候选限制合理的头维度、GQA 分组、2–4 层和 SwiGLU FFN；目标 966,784、误差 ≤1%，不自动放宽到 2%。逐模块计数去重共享参数，再用四 Token 合成反向检查所有参数有有限非零梯度；不创建 Optimizer，不更新任何参数。

KV 报告区分 tensor payload、底层 storage、Python/Tensor wrapper 的浅层元数据、序列化大小和模型权重。浅层 Python 计数不是完整 runtime 内存。临时扩展 K/V、Attention 矩阵及 allocator/driver 成本不包含在 KV 载荷中。GPU 报告 allocated/reserved 高水位，明确进程口径；无硬件的字段保持未知。CPU 不做吞吐量 Benchmark。

## 冻结前必须解决

- 获得并核验原始语料、Tokenizer 全文件 SHA，按真实 Document ID/位置重放消费流，区分 accepted update 与 overflow rollback/retry；只给已提交 Optimizer Update 累加 Token 和 FLOPs。历史同运行 milestone 恢复与跨运行 resume/restart 分别记录。
- 正式轨道 A/B 保持同一 Seed 内数据顺序、有效 Batch Token、EOS/文档重置及重复规则。A4 与 B2 重合的同配置运行可以共享，不能作为两个独立 Seed 计数。
- 训练 FLOPs 必须覆盖 forward/backward/optimizer，声明算子覆盖、未支持算子和 dtype；线性乘加估算、6P/token、wall time 及推理 cache 不能替代完整计量。完整覆盖未验证时禁止标 Compute-Matched PASS。冻结累计预算点及 ≤5% 的离散 Update 末步处理，不能看结果再改。
- A/B/C/D 分开；Memory M1–M5 采用同输入、问题、输出规则，自然预算与多个持久字节预算点分开。按原文距离 128/512/1024/4096/8192，episode 数、提示/答案词表、不同状态预算的模型适配仍须冻结。更改状态维度的条件可能需独立训练，不能直接裁剪现有 State 冒充公平容量曲线。
- 至少 3 个正式独立训练 Seed，确认性建议 5 个；实际 Seed、Token/FLOPs 预算、超参数搜索次数、学习率、选择规则、统计单位与效应阈值均待冻结。独立 Test 保持封存，不能把开发留出改名 Test。
- 两名独立人工盲评者及一致性指标尚未落实，AI 附件不能替代。未授予新训练预算；GPU 小时不能从历史逐事件 T1 外推为优化路径保证。

所有后续阶段只在相应门禁通过且取得独立训练授权后实施。本次准备不伪造已完成的 Memory Benchmark、完整 FLOPs、GPU profile 或确认性数据审计。

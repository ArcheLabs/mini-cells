# M5 实施报告

## 当前验收状态

本次交付源码、自动门禁及单一 Kaggle 入口。**尚未运行真实 M4 归档和 T4 Smoke，不能标记 M5_ENGINEERING_COMPLETE。** 当前环境仅 CPU，可运行 Tiny 组件检查；无可用 Kaggle 执行凭据。H5 保持 NOT_EVALUATED。

| 验证 | 本地证据 | 正式门禁 |
|---|---|---|
| State Summary、Predictor、Store、Attention、跨 Session 恢复 | 7 项测试通过，14.36 秒 | 待 Gate 0 后自动执行 |
| 五臂真实优化器更新 | 每臂 2 次，64 Input Tokens、2 Target Tokens | 仅 Tiny 开发集成，非 Gate 6 正式验收 |
| 参数审计 | E3 9,939,200；E6/E7 10,021,120 | 实例化核验通过 |
| M4 权重归档 | 固定 Version 187 和原始 GPU 文件 Hash | NOT_RUN |
| 真实 T4 五臂训练与资源统计 | 已实现入口 | NOT_RUN |

## 数学和梯度

固定数量 Summary = P(mean_channels(上一 Chunk 的 SSM State))，每个 SWA 层 4 个 D=256 Token。Mamba-1 复用冻结 M2 实现，分块路径显式补入初始 State 的齐次响应，支持完整 Episode 反向传播。训练不借用 inference-only detach Cache。

Predictor 预测值为历史 EMA：m[t]=0.875 m[t-1]+0.125 h[t]；score[t]=mean((h[t]-m[t-1])²)。m 初始为零，无可训练参数；评分和选择不传梯度。写入记录为原始输入表示、位置、创新度。选中记录的表示保留当前 Episode 的梯度；观测快照和持久化状态 detach。

所有层先读历史 Store，再在 Chunk 完成后写入。FIFO 淘汰最旧记录；Random 使用固定 Seed reservoir sampling；Innovation 仅在新分数严格大于当前最小值时替换，平分不替换。Recall 用 Query 相似度 Top-k，再与 Local KV 和 Summary 联合 Attention，分数不参与检索排序。

## 因果性、保存与恢复

局部 Attention 为因果窗口 W=32。Summary 仅使用上一 Chunk 状态，Store 仅包含过去 Chunk。跨 Session 恢复分别校验 Mamba、KV、Predictor、Store 的独立文件 Hash；Store-only 模式清空其他三种状态，仅用真实 Attention 读取持久记录。损坏文件拒绝加载，旧 Runtime 在 Reset 后失效。

CPU 开发检查使用解析 Oracle 核验真实 Attention，校验 Full/Chunk 数值一致、未来信息隔离、随机策略重现及独立恢复；Tiny 五臂训练复用 M2 Checkpoint、Rollback、RNG、Sampler 和 Ledger，不改冻结基础设施。原始输出见 audit/。

## 实际运行与资源

T4 入口严格要求 Tesla T4、FP32、TF32 关闭，不自动换算法或精度。每臂 2 更新、序列 512，五臂合计 5120 Input Tokens；额外长序列仅开发评估，不计训练。保存模型/优化器/RNG/Sampler/Ledger 和分离 Runtime。报告记录静态分项、动态状态、Store 元数据与 Payload、峰值显存、读写耗时、Checkpoint 大小及耗时、实际吞吐。

评估包含长度至 16384、Binding/Query/干扰变化、相似 Key、重复值、延迟检索、Selective Copy、Induction；WikiText 使用 M1 冻结版本、Tokenizer 和 Validation 适配器，不替换语料。每例样本有限，只用于工程执行诊断。容量 128 是推理干预，不是额外训练组。

Gate 0 归档保存已冻结 M4 实际权重和 Cell；版本序号 187 不作为 Script Version ID。只有实际 Gate 0–7 全通过及发布 Hash/SHA 核验通过后，运行入口才标记 M5_ENGINEERING_COMPLETE。

## 未验证事项

尚无实际 T4 成本、正式模型质量、创新度与遗忘程度的统计关系或 H5 证据；不能把 Tiny 零召回或优化器更新解释为机制有效。M4 历史源码和结果保持不变。详见 implementation_deviations.md。

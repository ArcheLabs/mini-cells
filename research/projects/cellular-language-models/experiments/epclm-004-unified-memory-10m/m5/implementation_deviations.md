# 实施决策与边界

- 仅新增 m5/，复用 M1 数据和 Tokenizer、M2 模型与训练、M3 归档函数及通用发布器，不修改 M1–M4。
- E3 为 M5 新鲜控制，FFN=816，以给 Summary 参数留出预算；不是重训或改写冻结 E3。五臂静态参数差约 0.82%。
- Summary 将实际 SSM State 按通道平均后投影为固定 4 Token，不声称完整保留 State 信息，也不等同原论文的所有细节。
- Predictor 是过去全历史的 EMA，有效时间尺度约 8 Token；创新度是参数自由代理，不声称已证明能预测遗忘。
- Store 写入按 Chunk=32 延迟，整个 Chunk 不读取自己的写入。所存表示为输入 embedding，不是标签或预测答案。全 Episode 训练 State 不截断；离散写入/Top-k 无梯度，选中表示可回传。
- Random 采用 reservoir sampling；FIFO 按原位置淘汰；三种策略使用相同容量、输入表示和 Recall 路径。
- Capacity128 仅评估同一 Capacity64 训练模型；不是独立训练对照。相似 Key 按固定初始化 embedding 的余弦构造，不代表自然语言语义相似。
- 合成任务和 WikiText 开发诊断每条件样本很少，不提供显著性或正式泛化结论。自然语言仅 Validation，不开放 Test。
- 所有序列无 PAD，训练 Batch=1；不同 Episode 独立 Runtime。长序列只在 Query 位置投影词表以节约显存；不改变监督定义。
- 推理读准备时间覆盖 Store 投影/Top-k/Gather，不单独归因联合 Softmax；Activation/workspace 残差显存为估计，不能当精确分项。Smoke 吞吐不作正式性能。
- M4 Script Version ID 尚未知；仅使用用户确认的 Version187和冻结文件 Hash。归档失败即停止，不猜 ID、不搜最新、不重训。
- 当前 CPU Tiny 证据不是按 Gate0→7 完成的正式验收；真实运行仍需 Kaggle T4。依赖仅新增小型 einops，缺失的数据工具沿用现有固定版本，不安装 Mamba CUDA/FlashAttention。

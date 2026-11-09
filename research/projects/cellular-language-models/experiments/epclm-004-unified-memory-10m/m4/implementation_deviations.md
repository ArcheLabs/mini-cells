# M4 实施选择

- 原规格三路并行，不改为串行 Mamba→Titans。只设置一个融合阶段，三路均从公共 Embedding 开始。
- SWA 5 层 / FFN664，Mamba-1 8 层 / state20；共同维度256。实际静态参数匹配10M与E0±1%，对E4约−1.0171%，如实报告。
- 完整复用 M2 的可微分分支 forward，仅把 LM Head 换为 Identity 暴露归一化隐藏表示，Embedding共享。
- 复用 M3 NeuralMemory，不解冻其历史权重；新模型从公共未训练初始化构造。
- Memory 不分层共享更新，避免“先处理完整上一层序列，再泄漏到下一层早期位置”的时间问题。
- Gate 为三个标量概率；已同维的 SSM 不增加额外投影；保留融合后 Residual / Norm / MLP。
- Cache 仅用于在线/只读推理，调用必须 Chunk 对齐。Runtime 与持久 Cell 分别管理；不提供跨 Chunk 的训练状态续接。
- 只交付两次更新的工程 Smoke，不进行正式模型质量或长期任务比较。
- M1/M2/M3 的源码、配置、历史结果和归档权重均保持不变。没有新增通用训练或传输框架。

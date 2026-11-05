# Chunk-Parallel v0.3

MECHANISM_VERSION = chunk-parallel-v0.3。新机制与独立源码/结果目录，不修改 v0.2，也不加载 v0.2 训练 Checkpoint。

运行本目录 run.ipynb：Tesla T4、Internet、已有 GITHUB_TOKEN，Save & Run All。全程180分钟，训练累计105分钟硬截止；选预算时同时限制固定80分钟训练窗口并保留20%余量。共同候选1M/500K/250K为完整文档组上限，实际不得少于250K；最低档向下取整不足250K会明确不适用，不悄悄降低最低预算。

## 已冻结的计算语义

- 输入 Embedding→RMSNorm→QKV；当前输入 Token 可以读自身和过去，不能读未来。Cell 最终输出不写 Token KV。
- 同一逻辑 Chunk 内两个步骤都读 Chunk 初始 State。每步骤将选中的 Identity/State Gather 后批量执行共享 CellCore，不构造所有 Cell 的候选输出。
- 共享 Linear(3d,128)+tanh Writer；对 Token/两步的调用按 Cell Scatter-Add，再按真实调用次数取均值。共享 FP32 GRU(128,128) 在 Chunk 末尾更新，未选中 Cell 数值保持不变。GRU 实际批量计算16槽再屏蔽，不声称对 GRU 也做稀疏 Kernel 执行。
- API 分段不改变逻辑边界。部分 Chunk 保留 State Snapshot、未提交 sum/count、KV 和位置；Session.finish() 才提交文档末尾残块。Snapshot 验证权重、配置和模式并恢复 Router RNG。
- 完整文档组更新优化器；H=2为不重叠两 Chunk 的图窗口，窗口内一次 backward，窗口边界同时 detach KV/State，保留数值。最后/窗口末尾 Writer 不一定获得后续普通反传，记录具备下一 Chunk 梯度条件的写入数量和实际 Writer 梯度。
- 为给窗口末 Chunk 的路由提供下一 Chunk 的 detached CE，先生成该文档的分窗口计算图，再按独立窗口 backward；图在窗口边界已断开，但文档内多个窗口会同时驻留，真实显存由 Warm-up 测量，不声称常量窗口内存。
- Return为当前 Token CE加下一逻辑 Chunk平均CE，最后Chunk未来项零；PG=.1×所有动作的平均score-function目标。两步独立EMA baseline，初值0、decay=.98，在整个optimizer group使用旧值；成功step后更新。TRUNCATED_ROUTER_CREDIT_ASSIGNMENT，不是全文档目标无偏估计。
- CE按组有效预测目标加权；PG按组有效动作数2N加权。训练Categorical、主验证Argmax；固定Seed采样副验证仅作诊断。统一FP32，无Autocast/Scaler/TF32；非有限梯度失败、丢弃组baseline/counters，不重试接受NaN。

复用固定Tokenizer和真实完整文档数据；选定Chunk后发布重新生成的文档Chunk索引及哈希。按32/64/128各测四模型真实前后向和更新，优先64若满足最低共同预算，否则按四模型总成本选最快候选；不依据PPL挑Chunk，全部重新初始化后训练。

文档之间当前顺序执行，Chunk内部并行；支持同长度稠密文档Batch与末尾Padding写入屏蔽，但尚不提供跨多次请求的ragged批次KV调度。Padding只能是尾部，不参与Writer；正式文档流无Padding。采样路径在不同API分段下可能消耗不同RNG顺序；确定性Argmax的流式/并行等价为本轮验收口径，Snapshot则保证同一分段续执行采样可复现。

## 检查及证据

只运行SPEC要求的16项小型CPU检查，不运行完整CPU训练或性能矩阵。覆盖32/64/128因果性、输入KV、并行/流式等价、两步State Snapshot、Scatter写入、无效Cell不变、跨Chunk Writer梯度、稀疏Core、部分Chunk与Snapshot。

GPU记录三候选吞吐、显存、前后向CUDA时间，正式评估固定延迟测量和一次冻结forward的Kernel/host热点。FLOPs为单独声明的稳态矩阵解析参考，绝不冒充GPU硬件计数。Head梯度是CE-only代表，Router query梯度来自Policy目标，其他模块可能混合两者，不把它们标为纯CE总梯度。

结果仅发布报告/配置/指标，权重留Kaggle Output；旧输出目录不覆盖。发布使用现有基础设施、独立Receipt和远端SHA核验；失败可用 `run.py --publish-only <run目录>`，不重复训练。没有真实GPU结果时，所有训练、速度和科学结论为NOT_RUN；16项正确性通过不等于模型研究成功。

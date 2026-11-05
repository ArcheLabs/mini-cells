# Stateful Computation & Routing 001

独立实现，不修改历史 A2/X4C、训练结果或冻结协议。原规范见 SPEC.md，用户修订见 AMENDMENTS.md；最终可执行定义以 protocol.json 和下列冻结条款为准。

## 一次启动

Kaggle 打开 run.ipynb，选择 Tesla T4、Internet、已有 GITHUB_TOKEN，Save & Run All。自动执行配置审计→关键测试→真实文档资产→四模型 GPU 测速→统一预算冻结→重新初始化训练→双模式评估→冻结消融→报告及 SHA 核验发布。

总上限三小时，包括 Notebook 初始化和依赖安装。累计截止 10/15/25/105/150/165/180 分钟。训练/评估各用独立硬截止，不向后挪用。预算不可行停止，不强行做满表格。没有本地 GPU 训练预检或完整 CPU 训练。

复用固定 TinyStories revision 和已有 2048 Tokenizer；重新建立有真实文档边界的开发子集，不声称旧随机片段协议复现。只纳入不超过 4096 targets 的完整文档，过滤规则显式记录。训练候选为 1M/500K/250K 上限下的完整文档组前缀，实际不得少于 250K；250K 档因组取整可能不适用。四模型同一实际目标数、文档组及顺序，不截断最后一篇冒充完整文档。验证 32 篇固定完整文档，独立 Holdout 不加载。

## 六项训练前冻结条款

1. **梯度与 Credit Horizon**：State、Workspace KV、T0 全部层 KV 在 Chunk 内保留图；Chunk 反向后同时 detach，保持数值。Router 每次动作仅归因当前 Token 到本 Chunk 末尾 loss-to-go；不使用先前 Token 损失或未来 Chunk 收益。TBPTT 估计不是完整文档目标的无偏梯度。
2. **T0 Attention**：标准 Pre-Norm、RMSNorm、RoPE、SwiGLU、三层四头 MHA；窗口 128。当前输入 Token 可以读自身 K/V，但不读其预测目标或未来输入。每层独立 KV 跨 Chunk 保留数值并 detach。Cellular Workspace 则严格只读此前 Token，先完成 K=2 再 append 最终 KV。两个模型族语义差异如实报告。
3. **双模式**：训练 Categorical，温度 1；主评估 Argmax，副评估固定 Seed Categorical。四模型相同文档起点、前缀、Chunk 和计分目标。冻结消融按主模式评估，不把训练随机策略收益直接解释为 Argmax 收益。
4. **消融路径**：CellCore-State-Off 用零替换输入 State；Router-State-Off 清除 State 地址项；Recurrence-Off 仅清除 GRU 的 old-state 输入，h/u 仍可能携带历史，不能声称完全移除历史。All-Off 同时关闭三者。M1 All-Off 对预测与 CellCore-Off 冗余，不重复运行；M2 保留联合关闭。Write-Off 完全不写；Reset-at-Chunk 只重置 State，不重置 KV。Shuffle 是 CellCore 读取索引固定循环移位，Router/写回索引不变，仅诊断此读取绑定。
5. **Token-weighted Group Loss**：每组八篇完整文档，最后一组可较少。组目标数 N 为所有有效 targets（BOS 不计、EOS 计、无 Padding）。每 Chunk n 个目标，CE=sum(loss)/n，乘 n/N 后反向；PG=sum_{t,k}[(loss_to_go_t-b_old*(n-t))*log p(t,k)]/n，也乘 n/N。所有 Chunk 累积后只裁剪/更新一次。LR 以已提交 targets 调度，Warmup 占实际预算 10%，之后 cosine 到零。EMA 初值 0、beta=.99，每 Chunk 用旧值构造目标，再更新候选值；整组成功才提交。
6. **初始化及失败**：M0/M1/M2 公共组件在同 Seed fork 内按同顺序初始化，新增 State 及 M2 State地址使用独立 Seed，公共 Hash 自动核验。统一 FP32，禁用 Autocast/GradScaler/TF32。故不执行 FP16 Skip/Backoff；非有限 Loss/梯度在 step 前失败，清梯度，恢复 RNG，丢弃整组 counters/baseline。不会盲目重试同一 FP32 故障或接受 NaN 参数；step 后非有限参数使模型失效，不保存成完成 Checkpoint。Warm-up 所有模型、优化器、baseline、RNG、数据游标全部丢弃重建。

State 采用 16×128 零初始化，共享 GRUCell(2d,128)；Router 维度128；CellCore复用 d=188、SwiGLU639。只给每 Cell Identity 和动态 State，不给独立 FFN。M2 新增 State→Router 投影，M1 无此参数。Group 顺序处理文档以保证隔离，吞吐包含真实顺序执行成本，不宣称等于 Batch8 并行吞吐。

## 证据边界与输出

记录每组 NLL、梯度、targets、耗时、输入文档 ID；每组仅一次参数更新。模型最终权重保留 Output，报告中记 SHA/大小/保存时间，不入 Git。验证输出每文档 NLL、Argmax/Sampled 差异、路由次数/熵/转移/位置分布、State 范数/余弦/奇异值和更新幅度。实测三次预热、十次重复的固定 batch/context prefill/decode；无可靠 FLOPs 时明确 NOT_MEASURED，不用推测冒充实测。

配对 Bootstrap 2000 次、Seed91002、按文档重采样并重算 Token-weighted NLL 比值，改善至少1%且区间支持才记初步改善，明显反向则记退化；不代替 Seed 方差，不做多重比较确认。动态选择、功能分化、State不可替代性及架构效率优势均不能由单 Seed 或多样性直接推出。

不增加第五模型、Graph、消息传递、新 State 类型、参数匹配训练、多 Seed 或长程任务。当前实现只选择语言机制筛选；其他项目 NOT_RUN。

结果使用现有 GitHub 发布器，从 origin/main 派生独立结果分支，只 stage 结果。decision 在 SHA 清单生成前冻结；追加独立 receipt，并再次验证远端 SHA；不回写 decision。发布失败保留 Output，可执行 `run.py --publish-only <原 run 目录>` 重试，不训练。

## 最小检查

只运行 tests/test_core.py 的七项小型 CPU 检查（其中四项为四模型关键因果测试）。它们包括一次合成文档组更新及 Checkpoint roundtrip，无真实语料训练或 CPU Benchmark。

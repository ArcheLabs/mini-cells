# EPCLM-005 M4-H4 Fast Screening 实施报告

状态为 **L1 已实施，CPU 预检通过，正式 T4 运行待执行**。H4 保持 `NOT_EVALUATED`。

## 基线与预算

远端 main 核验为 ce9f1905b8d9a8badd2dbed111f68e180e71a619。历史 M4 验证记录的结果提交为 f0ab964502c1caa15d09ea1185ce16bec1674a20。冻结 M4 融合、M2 Trainer 和 M3 Memory 原样复用。本轮从公共初始化开始，模型 Seed 51017，数据 Seed 51001。

| 实验臂 | SSM | Titans | 外层更新 | 输入 Token | Query Targets |
|---|---|---|---:|---:|---:|
| F00 | 关 | 关 | 32 | 65,536 | 1,024 |
| F10 | 开 | 关 | 32 | 65,536 | 1,024 |
| F01 | 关 | 开 | 32 | 65,536 | 1,024 |
| F11 | 开 | 开 | 32 | 65,536 | 1,024 |

四组都保留 SWA。每组驻留参数 10,065,158，活跃参数及逐组件初始化 Hash 见审计文件。L1 使用 128 个训练 Episode、32 个 Validation Episode，另有三个独立评估数据 Seed，每个 64 个 Episode。训练和留出集 Pair 重叠为 0。最短事实到 Query 监督位置距离为 435 Token，高于五层 SWA 的 156 Token 传播范围。独立 Test 保持封存。

## 门禁进度

| Gate | 当前证据 |
|---|---|
| 0 源码与环境 | 源码、历史证据及 Tokenizer 核验通过。T4 实测预算在 Notebook 开始时检查。本机 GPU 是 Quadro K620。 |
| 1 Factorial | 四组规范模型公共初始化 Hash 一致。CPU 小张量验证分支跳过、冻结参数、有限梯度、Gate 归一化及形状。 |
| 2 数据 | 预算、配对身份、留出 Pair 隔离、无 Query 答案输入均通过。 |
| 3 CPU | 合成因果性和完整/缓存路径通过，最大差 1.20e-7。新进程验证 F00/F10/F01 自然状态与 F11 的四个持久化条件，恢复误差为 0。 |
| 4 T4 | 待执行。 |
| 5 跨 Session | CPU 合成矩阵通过。正式训练后权重及评估集验证待执行。 |
| 6 分析与交付 | 配对交互计算的专项测试通过。真实指标与结果发布待执行。 |

两个专项单元测试通过。另用真实 M2 Trainer 对四个小张量 CPU Fixture 各提交一次更新，验证了冻结权重、记账和 Checkpoint 内容。Fixture 的更新不计入正式预算。

## 实现和研究边界

Session A 从首个 Query 的起点切分，使用 HOLE 对齐。Session B 开始时 SWA KV 为空，Titans 使用只读模式。Runtime 恢复用新模型的 owner、参数版本和 Session 绑定来重建 wrapper。F11 的 C00/C10/C01/C11 共享静态权重和 Query。

T4 入口先运行小张量门禁，再对实际 L1 batch 做无优化器更新的耗时和显存探测。运行后保存配对 Episode、负 NLL 和 Exact Match 交互区间、状态、Gate 和推理消融证据。正向质量候选应用 3 个百分点的实用阈值，Gate 行为和资源价值仍需结合结果审查。

L1 触发共同地板或天花板时记录升级触发。L2/L3 的上下文长度与固定 32 次、batch 4、65,536 输入 Token 预算无法同时保持，升级前需冻结其记账口径。当前交付仅实现 L1。尚无正式 Delayed Recall、Cross-session、交互效应或 H4 正向信号可报告。

[运行入口和说明](fast-screen/README.md) · [CPU 与数据审计](fast-screen/audit)

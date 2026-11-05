# EPCLM-X4C-001

本目录将 Native X4 的深度、持久状态和稀疏 Cell 计算分阶段验证，保留原始 X4/T1/Core-001 实现与结果。完整请求见 [SPEC.md](SPEC.md)，机器可读约束见 [protocol.json](protocol.json)。本地资源不符合目标 T4，正式训练状态为 **NOT_RUN_RESOURCE_UNAVAILABLE**。CPU 正确性不代表语言能力或效率证据。

## 一次 Kaggle 启动

使用本分支 `codex/epclm-x4c-001` 的 [run.ipynb](run.ipynb)，开启 Tesla T4 / Internet / 已有 `GITHUB_TOKEN` Secret，Save & Run All。数据来自固定 HF TinyStories Revision；公开数据下载无需新的 Secret，若环境已有 HF_TOKEN 可由既有库读取。没有凭据写入代码、输出或 Git。

默认 `auto / dev / tokens`：自动正确性检查 → G0 目标 GPU 等价 → 固定数据与 Tokenizer 字节身份验证 → G1 各组最多 1M 预测目标（100K 为同一次训练的首个检查里程碑）→ 只在实测 G1 通过阈值时执行 G2 → 只在 G2 跨 Chunk 机制证据通过时执行 G3 → 发布全部已执行结果。预算不足诚实返回 INCOMPLETE。没有自动执行 10M 或三 Seed 确认。

G1 开发筛选预先固定为 Validation PPL ≤A0 的 1.02 倍、解析 Forward FLOPs 至少降低 20%、实测 Decode Median ≤A0 的 1.05 倍；其中 5% 是对“不明显恶化”的运行前量化，不是多 Seed 非劣证明。

若 G1 的全部减深候选在完整开发筛选中退化，则记录 DEPTH_REDUCTION_NOT_SUPPORTED，后续阶段不运行。Smoke 100K 或单独 Compute-Matched Job 不用于冻结 G2 基础架构。模块可用不等于研究假设成立。

每个 Job 为 5400 秒执行、600 秒发布、1200 秒余量。Notebook 把初始化时间计入两小时总限制。失败模型从头开始；本实验不跨源码恢复 Optimizer。最后权重仅保存一次到各模型 `private/final-model.pt`，Profiler 原始 trace 在同目录，Git 返回 JSON/Markdown。已发布报告含原始输入位置、曲线、全部候选和失败信息。

## 复用与历史边界

- `architecture.X4C` 直接继承 PhaseBLM、复用 PhaseBStage 参数和原 Stage 组件；A0 参数名与形状可直接 strict-load 原 X4。原残差系数是 sigmoid(-2)，不是 Core 的 0.1。
- G1 缩减 Step Embedding 行数并设置 iterations，不删除 Stage。D_ATTN4_FFN1、D_ATTN1_FFN4 是独立训练的低预算机制诊断，不能把即时裁剪结果当成重训质量。
- 原增量实现 `epclm001.incremental.IncrementalLM` 的 `_attend`、`_append`、`rope_at`、Stage/Iteration 布局、缓存校验与内存统计用于 Session。训练的并行 Chunk Attention 使用相同位置/窗口定义并经 Cached 对比，不复制旧模型或完整 Runtime 框架。
- 原 Phase-C Seed 91001 输出目录不在当前仓库；另定位到了 Controlled Retrain Seed 91002 的报告和 pinned HF 资产。它们不是同一次实验。新 A0 始终从零初始化。
- 实际 Phase-C wrapper 临时令 base.C0_NAME 等于当前模型名，因此历史 model_seed=data_seed=91001，不能直接套用 Native 通用函数的非 C0 seed+1 分支。
- 数据严格复用 `epclm001.retrain_data.prepare` 与已存储的 retrain_config 历史 Manifest，保留原固定 Tokenizer，原 writer 重建后核对全部输入 Hash；不调用会重新训练 Tokenizer 的 prepare_corpus。
- Native runtime 的文件默认 Revision 与实际 Phase-C 覆盖后的 Revision 不同；执行依照 pinned Manifest，冲突和来源记在 REPOSITORY-AUDIT。

## 持久状态与调用语义

State 为 FP32，Flat `[B,2048]` 与 Cellular `[B,16,128]` 元素数相同，Query Read 地址空间都仅为 16 个 128 维 State 槽。Flat 使用全局非线性低秩 gated residual 更新，Cellular 使用共享非线性 gated residual 写入、固定局部混合与不同坐标的 Chunk 摘要，避免所有 Cell 永远同值。参数与状态分别管理。

固定 Chunk=32；进入 Chunk 时的 State 用于本 Chunk 的所有预测，在最后一个已观察 Token 的因果表示完成后才写 State。32/8/1 Token API 分块不改变边界。G2/G3 的无状态对照同样按 32 Token 分块执行，防止把分块执行本身的 Kernel 差异归因于 State；G1 保留原并行参考。训练不会在 Chunk 写入后 Detach。原语言训练每个独立 125 Token 随机流样本 Reset，TBPTT 最多跨四个 Chunk；这不声称训练到了跨 125 Token 的任务。

新的连续执行语义为绝对 RoPE + 每 Stage/Iteration 原窗口 8/32/128，最多 4096 Session Token。与原模型等价的范围是起始位置 0、长度≤128；超过该长度比较的是明确扩展的同一因果参考，不是截断历史再重置位置的重算函数。

Runtime 用 `Session(model.eval(), tokenizer_identity)`，`prefill`、`step` 继续同一文档；独立文档默认调用 `prefill_document` 自动 Reset。`snapshot/restore` 保存权重/配置/Tokenizer 身份、绝对位置、逐迭代 KV、动态 State、未完成 Chunk 表示与 Session RNG，不重放历史 Token。独立任务使用新 Session。原 Native 数据为拼接 Token 流，报告的语言评价单位是独立流片段，不冒称独立文档。

## G2/G3 机制和公平性

M1–M4 是明确分离的随机 Token-ID 延迟、更新、干扰和约束任务，不冒充自然语言能力。延迟超过配置 Attention 的最坏传播范围，训练/评价随机组合不相交；每种模型另给 128 个 Query 预测目标，观察输入 Token 数单列。整个机制 Episode 保留梯度直到延迟 Query，提供跨 Chunk Write 的学习信号。B3/B4 是同一任务训练后 B2 的冻结 Read/Write 消融，不是新结构训练。M5 验证持续执行与真正 Snapshot/Restore。

G3 仅替换 Stage 2 的原 dense SwiGLU。共享小 SwiGLU + rank-16 Cell-specific adapter 只在 Top-2 行执行；router 计算全 16 个得分，其成本单列。Chunk Writer 仍可轻量更新所有 State，并计入成本；不声称未路由 Cell 完全零成本。报告 activation frequency、entropy、dead Cells、特定参数梯度、按输入 Token 的路由、实际 FFN 执行行数以及真实训练 trace。

rank 16 是训练前按结构参数量选定，使 C2 位于原 X4 的 ±5% 目标内，不是按数据搜索。Stateless C1 没有约 10 万个 State 参数，参数量仍明显少于原 X4；该差异保留并报告，不塞入闲置参数。C3 保留与 C2 相同的 Cellular State，按整数宽度匹配 C2 的普通 dense FFN 控制；Compute 差异另报，不能同时宣称两种匹配。

四条比较轴分开：参数量；同 Token/同输入顺序；解析 Compute 预算；同 T4/FP32/Batch 的推理。`--match compute` 是独立 Job，以原 A0 的解析 matmul-heavy 训练预算为参照，完整 1000 预测目标步向下取整，误差上界为一个完整 Update 的解析成本。被 GradScaler 拒绝的前向/反向尝试也消耗 Compute 预算，但不增加已训练 Token、Optimizer 或 LR 步号；因此实际成功更新数可能低于名义最大值，二者分别记录。Warmup 按计划 Compute 步数比例缩放，不在 Token-Matched Job 隐式加 Token。

解析 FLOPs 严格复现旧公式，新增 State/Router/Cell 路径分别估算；Backward=2×Forward 是显式估算。Profiler 只报告它能计数的算子 FLOPs、Kernel 事件、分配与设备时间，原 trace 保留。未计数的归一化、RoPE、softmax、top-k、Optimizer、真实 DRAM 流量不会被当作零成本；本实验不伪称完整硬件 Compute-Matched；局部 Attention Pair 公式也可能低估 dense masked SDPA 后端的实际矩阵计算。延迟包含当前 Python Session 与诊断开销，不是理想化 Kernel 延迟。

## 单独入口与阶段身份

```bash
python train.py --stage G1 --profile smoke --match tokens
python train.py --stage G1 --profile dev --match tokens
python train.py --stage G1 --profile dev --match compute
```

之后单独执行 G2/G3 时，可用 `--selection <原 selection.json>`；要求同 Source、实现、Protocol 和 Dataset 身份，且原阶段已真正满足条件。默认 Notebook 自动逐阶段处理，无需填写 SHA 或手动选胜者。

最终完整开发训练用 `--profile development`（10M/模型），三 Seed 复核需要另行明确 Seed；默认不消耗 91101–91103 Confirmation Seeds。所有 GPU Job 都受同一两小时限制，未完成不填 PASS。开发阈值按协议固定，结果可为负。

## 测试和报告

`tests/test_correctness.py` 仅实现文档要求的 15 项正确性测试，主要用微型 CPU 模型；实际 188 维/47 维奇数 Head 的原模型等价、完整参数与旧 FLOPs 公式也验证。没有本地完整训练或 CPU 性能 Benchmark。

每个模型单独保存 config、training metrics、完整 Validation/Holdout、生成文本、FLOPs、推理、显存、State 诊断和 decision；缺失结果只在失败结束后标明 NOT_RUN，不用文件存在性跳过评估。多个评估独立处理错误。阶段总表、选择依据和中文研究报告都保留，权重身份用于绑定 Snapshot 和评估。

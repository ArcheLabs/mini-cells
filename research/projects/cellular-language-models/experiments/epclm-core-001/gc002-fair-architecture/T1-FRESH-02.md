# GC002-T1-FRESH-02

运行 `t1-formal.ipynb`，Tesla T4、Internet，启用已有 `GITHUB_TOKEN` 和 `KAGGLE_API_TOKEN` Secrets，Save & Run All。

Notebook 的 `t1_formal.py` 命令入口现在委托 `t1_fresh.py`。旧 Formal 函数保留供历史源码与测试参考，新的命令入口不会调用占位文件、恢复、FLOPs 审计或 Smoke 门禁。

已逐字段核对上次结果分支 `kaggle/epclm-gc002-t1-formal-63cb0a48c2f5` 的 `training-config.json`：与冻结协议完全一致。新运行沿用其模型、Seed 173020021、FP32、AdamW、学习率 0.0003、97 Warmup、采样/Mask/EOS 和 250000 预测目标/977 更新配置。新配置报告仅包含运行需要的普通配置，不带历史 Hash/Checkpoint/Recovery 配置。

每次创建 UUID 后缀的新目录，版本 142 仅下载 Tokenizer、Manifest、Train、Development 数据，不下载模型或优化器。读取已存储的数据划分：完整 Validation 为 Development 前 16 个文档，完整 Development Holdout 为其余文档；不重新划分，也不访问 Independent Test。数据普通结构、BOS/EOS 和跨划分重复检查继续执行，不计算文件 SHA256。

训练采用原 FullSequenceSampler 与 teacher_forced_sequences，明确初始化新模型、优化器、调度器与采样器。上下文 Token 不进入有效预测目标预算。没有恢复、周期保存、恢复包或独立预训练门禁；训练完成后保存一次 `final-model.pt`。中断后下次重新初始化，不自动恢复。

训练后明确调用五项评估，每项独立处理异常，覆盖已有 NOT_RUN 文件。评估有失败时发布其他完成结果，整体不标记 COMPLETE。训练未完成时，在结束后才写未执行项目的 NOT_STARTED 报告，供统一发布，不把其标记为已执行。

生成沿用 GC001 的 16 个 Prompt，两种模式、每个最多 128 Token、基础 Seed 713（按原协议每 Prompt 加 index）。完整生成文本和 Token ID 位于 generation-samples.json。PPL 包含文档数、目标数、总/平均 NLL、PPL 与时间。推理包含 32/128 Prefill、单 Token Decode、滑窗、KV 占用和 Cached/Noncached 数值比较。推理失败不阻塞 PPL。

总预算仍为 5400 秒执行、600 秒发布、1200 秒安全余量。超时产生失败/未完成报告，不扩展预算。

结果使用已有 GitHub 发布器与认证方式，写入独立 `artifacts/experiments/epclm-core-001/gc002-fair-architecture/t1-fresh-02/`，每次独立结果分支。发布器的普通报告元数据机制沿用现有基础设施；不生成模型、Checkpoint 或数据完整性 Hash。权重仅存在本次 Kaggle Output 的 `<run_id>/final-model.pt`，私有 Output 需要 Notebook 所有者权限，Git 不包含权重。

本地只运行一个微型 CPU 回归：三次更新、精确尾批、真实 PPL/生成/推理/显存报告，覆盖已有占位文件及单项失败继续执行。未启动 T4 正式训练；当前环境为 Quadro K620，且没有 Kaggle API 凭据。完成实施不等于训练或评估已完成。

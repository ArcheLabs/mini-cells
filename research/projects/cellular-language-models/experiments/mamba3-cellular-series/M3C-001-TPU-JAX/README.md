# M3C-001-TPU-JAX

独立实验：使用社区候选实现 `jax-state-spaces/mamba3-jax`，不修改官方 PyTorch `BLOCKED_HARDWARE` 历史记录。源码固定 `ffa55848fc513ca91aa330bda4c33c86dab74827`，Apache-2.0，作者 Cosmo Santoni；非官方背书实现。

## 一次运行

Kaggle 选择 TPU、Internet、已有 GITHUB_TOKEN Secret，运行 notebook.ipynb 的 Save & Run All。自动准备独立 Python 环境、锁定候选源码、记录设备与依赖、SISO → MIMO → 发布结果。禁止 CPU 静默冒充 TPU；没有 TPU 时发布 BLOCKED_TPU_UNAVAILABLE。

单次 Notebook 上限 120 分钟，审计进程上限 1800 秒，依赖安装 600 秒；无真实语料、训练预算、优化器或 Checkpoint。多 TPU 核只使用默认设备进行最小正确性检查，不声称多核扩展性能。依赖遵循候选约束，实际解析版本写入 pip freeze，不声称已冻结跨设备依赖组合。Python >=3.11。

## 冻结门禁

1. 完整源码 Commit 与核心实现、上游测试、NumPy golden 哈希一致，实际导入来自固定 checkout。
2. SISO 先复用候选上游 logits-vs-NumPy-golden 和 SSD-vs-sequential-reference 两项测试。参考由候选作者编写，**不是官方 CUDA 内核对照**。
3. 同一随机模型，FP32 / highest matmul precision，固定 Seed18001、Batch2、长度9、Chunk4（跨两次边界），检查 JIT/Eager、Full/Chunk/Decode logits 及最终四类 State、因果性、批次隔离、缓存重置和内存复制恢复、有限非零梯度。无参数更新。
4. 误差 atol=rtol=1e-4，运行前固定，失败不自动放宽。梯度必须逐叶有限、总范数非零。
5. 仅 SISO 全部通过后执行 MIMO Rank4；增加上游 MIMO SSD-vs-NumPy-sequential reference，执行相同模型门禁。SISO 失败则 MIMO NOT_RUN。

配置仅小型合成工程验证：vocab64、hidden32、state8、head8、expand2、1 layer、ngroups1、rope_fraction0.5、is_outproj_norm false。不代表原提案8层模型已通过，也不扩大全配置测试矩阵。缓存每叶 shape/dtype/bytes 记录，并验证不同前缀长度下大小不变。

## 判定与后续

- TPU_CANDIDATE_GATES_PASS：两个模式在实际 TPU 上通过本协议范围，具备 M3C-002 候选底座设计审查资格。
- INCOMPLETE：失败保留异常及已完成阶段报告，不产生科学结论。
- official_kernel_equivalence 始终 NOT_VERIFIED；社区 README 的 parity 声明不直接当作 MiniCells 证据。
- M3C-002 在 TPU 门禁通过后沿用此候选实现、源码身份、State/缓存语义和门禁；另行明确有界 Workspace 插入边界与实验预算。本次不实现 Cells，不自动训练，不把正确性通过解释为架构收益。

若发布失败，保留 Kaggle Output。入口支持 `--publish-only` 复用报告，不重复审计。既有 PyTorch 审计和历史结果不变。

来源：[候选仓库](https://github.com/jax-state-spaces/mamba3-jax)、[锁定源码](https://github.com/jax-state-spaces/mamba3-jax/tree/ffa55848fc513ca91aa330bda4c33c86dab74827)、[JAX TPU 安装说明](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu)。

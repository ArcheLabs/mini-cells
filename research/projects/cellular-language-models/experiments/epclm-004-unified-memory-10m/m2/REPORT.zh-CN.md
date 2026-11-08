# M2 实施与验证报告

状态：`CPU_VALIDATED_TARGET_GPU_PENDING`。未运行真实 GPU 训练；未运行 20M / 100M；未实现 Titans、MAG、B'MOJO 或三路融合。H1–H6 均为 NOT_EVALUATED。

## 1. M1 已冻结

[M1 验收记录](../docs/M1-ACCEPTANCE.zh-CN.md)登记三个原始结果分支和文件哈希，T4 FP32 真实更新与恢复已通过。MQAR 0/32、常量预测 474 和无 binding 的零召回保留，不进行补救性训练。旧 decision 中的 publication=PENDING 解释为推送前写入值；附加验收记录证明发布完成，不覆盖原始文件。

用户提供 Script Version ID **366932788**，按用户提供来源补入验收记录；未独立向 Kaggle API 核验。三个阶段属于同一次 Notebook 的关联按用户回复记录，原始 ID_PENDING manifest 不改写。

本地/Kaggle CPU tiny 同 Seed 的训练输入哈希与 Ledger 全部一致，Loss 最大差 9.54e-7。这是跨运行环境的 CPU 一致性证据，不是独立 T4 训练重复，也不是多 Seed 科学确认。没有为重复性额外消耗 GPU 训练资源。

`configs/source-lock.json` 核验冻结 E0、M1 运行器、数据、配置和 Tokenizer 的文件 SHA。M2 没有修改这些源文件。

## 2. 模型定义与公平性

实际实例化结果：E0/E1 10,130,688；E2 10,092,288；E3 10,135,808。相对冻结 E0 分别为 0%、0%、−0.379%、+0.051%，全部在 ±1% 内。这里不是声称 E0 本身已达到相对整数 10M 的 ±1%。

- **E0：**直接复用冻结 M1 E0，不改其数学定义、参数与初始化。
- **E1：**复用现有 CachedTransformer 的显式局部 Mask。窗口含当前 Token，允许 `max(0,t-W+1)..t`，最多 W 个 Key。训练为可微 uncached full-sequence 路径；推理缓存另行管理，不把 detach cache 当训练路径。
- **E2：**18 个 Mamba-1 residual blocks，每层 RMSNorm + Mamba，不添加 Transformer FFN。SSM 本身的输出投影/非线性仍保留。
- **E3：**8 个 mixing blocks，4 个 Mamba-1 与 4 个 SWA 交替；每个 mixer 后有 Norm / Residual / SwiGLU。对应 Mamba + MLP + SWA + MLP 的 Samba-lite 层级组合，不是原论文模型复现。

共同 dim=256、vocab=8192、共享 Embedding/Head、Head4、KVHead4。E0/E1 全部初始参数逐元素相同。其他模型与冻结 E0 中名称/形状相同的公共参数复制相同初值。E2/E3 对应逻辑 SSM 层也使用独立、相同 Seed 流，保证 Mamba 特殊 A/D/dt 初始化一致，不被通用 Linear 初始化覆盖。

参数匹配选择：E2/E3 state size 都为 20（默认 16 的明确调整），expand=2、conv=4、dt_rank=auto；E3 FFN=848（E0/E1=704）。这些设置写入固定配置，实际计数和公共 Embedding / SSM 哈希见 [parameter-counts.json](audit/parameter-counts.json)。不同层数和运算构成不等于计算预算匹配，必须同时报告系统成本。

工程 Smoke 用 **W32、序列 128**，确保实际跨越局部窗口，避免 W256 在长度 128 下与 E0 完全等效。W32 是明确的工程设置，不宣称是论文最优窗口。该范围是逐层直接读取的范围；多层 SWA 可以间接传播更早信息，不能宣称整个模型只含最近 32 Token 的信息。后续改变窗口必须形成新协议身份。

## 3. 官方版本与参考后端

- Samba 官方仓库 commit：`617c7a0f8c71f1b7cb6180b86f9543d146f5c66f`，复用 `lit_gpt/mamba_simple.py`。
- 官方 state-spaces/mamba **v1.2.2** commit：`a07faffa36a7b89e754b5de972418475bcdd77b6`，抽取未修改的 `selective_scan_ref` 函数。
- 许可证和原文件/提取函数/适配文件 SHA 均保留于 vendor/ 与 upstream-lock.json。

Samba 模块仅替换扩展导入、显式禁用 fused fast path，并用官方参考函数接替 selective scan。原始卷积、输入依赖 B/C、dt softplus、A=-exp(A_log)、D skip、SiLU gate、forward / step 状态结构均保留。没有编造兼容 mamba_ssm 包或自行写 GPU kernel。

该后端满足本轮先有 correctness reference 的执行顺序；**不能把它的吞吐或延迟当作优化后的原生 Mamba 性能**。原生 Kernel 的数值等价和 T4 性能尚未验证。Mamba-3 未参与。

## 4. 统一协议与基础设施

所有模型共用冻结 Tokenizer SHA `d1ed85988c25e27618ea8de951c9005758d57fef9e46832fa623a0812afe2d04`、WikiText revision 与 MQAR 规则。GPU Smoke 共用 bindings8、length128、单 Query、data seed41001、model/sampler seed41017、train128/validation32，Test 封闭。完成后检查逐更新 input hash 完全相同。

训练均对完整 Episode 反向传播；Mamba 的 scan 状态在该序列内保留梯度，不做 chunk detach；不同 Episode/optimizer update 不携带激活或 Runtime State。推理 full/cache 检查单独在 no_grad 下执行，不将其缓存作为训练路径。

所有模型为 AdamW、LR=3e-4、betas=(0.9,0.95)、WD=0.1、clip=1、FP32、TF32 off、恒定 LR、Batch4、2 updates。为与 M1 统一，WD 应用全部静态参数；不声称复现原论文的参数分组/优化器策略。字段实际生效，拒绝不支持的精度/优化器/Scheduler 配置。

每臂 GPU 入口仅 1024 input tokens / 8 query targets；四臂总计 4096 / 32。它是同预算的微型工程检查，不能对质量排序。既不延长历史 M1，也不加载其训练后权重到某一臂造成不公平。所有模型从相同公共初始参数开始一个独立 M2 run。

## 5. Checkpoint 优化及故障语义

复用 M1 的 Loss、Episode Sampler、Core 原子 checksum envelope 与 RNG 工具，只新增本阶段的短训练适配器。

- 每 8 updates 或 120 秒保存完整 Checkpoint；结束、软超时、可处理的退出保存最新安全前缀。
- 非有限梯度在裁剪/参数更新前拒绝，恢复当前尝试的 Sampler/RNG，之前正确但尚未写盘的更新可以一起 flush。
- 参数更新可能已突变时，恢复最后 durable checkpoint，包括模型、Optimizer、Sampler、RNG、Ledger 与 history；同时丢弃其后的未持久化更新，不在报告中保留虚假的 Token 数。
- 无法处理的断电/kill 最多丢失 7 次更新；不假装每个内存已提交的 update 都已经落盘。
- 最后刚保存的步骤不重复写相同完整文件。保存次数与耗时单独记录，计算时间不包含写盘。
- COMPLETE arm 验证 source/config/data/checkpoint 后直接复用结果，不重复训练。发布重试另有 `--publish-only`。

12 更新测试只写 **3 次**完整文件（初始、8、12），旧 M1 为 13 次。恢复后参数逐元素相同。注入第三次非有限梯度时，正确保留/落盘前 2 次更新；注入第十次 optimizer NaN 时，退回第 8 次 durable checkpoint，模型与 Ledger 同时回退。没有改变冻结 M1 trainer。

## 6. 实际本地验证与未完成项

M2 共 4 项合并测试：

1. E1 同初始化、窗口边界、未来泄漏、全序列/缓存一致性。
2. 官方参考扫描闭式 Oracle；四模型相同 MQAR、因果性、有限梯度、full/cache、Batch 隔离、reset 以及 SSM/KV 分开计量。
3. 定期保存次数、中断精确恢复、相同输入顺序、零预算退出。
4. 失败梯度 safe-prefix flush、参数突变后的 durable rollback。

完整本地入口在 Python3.11 / torch2.6 CPU 上执行 `--tiny`，**每臂只有 2 次更新、256 输入 Token、8 个 Query target**，四臂输入顺序相同。该 fixture 为 dim32、2层、state4，仅验证路径，不进行约 10M 的 CPU 训练或性能 Benchmark。fixture 参数未做 ±1% 匹配，不能用其质量或速度排名。

| Arm | CPU fixture 参数 | 初始→结束 NLL | 完整 Checkpoint 写入 |
|---|---:|---|---:|
| E0 | 282,784 | 9.0000 → 9.0052 | 2 |
| E1 | 282,784 | 9.0234 → 9.0298 | 2 |
| E2 | 277,472 | 9.0015 → 9.0082 | 2 |
| E3 | 286,304 | 9.0453 → 9.0528 | 2 |

短训没有展示收益，不改变 H1–H6 状态。CPU 小张量 correctness gate 的 full/cache 最大误差在 1.8e-7 以下；完整 tiny 入口的误差另见各臂 resource_profile。重复调用已完成入口、禁止训练回调后仍成功，四个 checkpoint 字节完全不变。原始 JSON 见 [audit/cpu-tiny/](audit/cpu-tiny/)。实际执行源码为 `542683712db31c55c58a379fa961b4017c8d8f35`；后续只增加配置严格校验、源锁/实际环境元数据与文档，实际优化器数值设置保持相同，没有重训已完成 fixture。

**T4 的 M2 四臂正确性、2次真实更新、运行时显存与延迟尚待执行。** [Notebook](notebook.ipynb)自动 CPU 回归→目标 GPU correctness→四臂等预算 Smoke→资源报告→既有发布器。无大依赖构建、无 20M/100M。7200 秒硬上限，5400 秒执行、600 秒发布、1200 秒安全余量，符合单次最多三小时限制。

原生 Mamba Kernel 性能、正式模型质量、多 Seed 结论均不在本次完成范围。不能因 PyTorch reference 较慢而否定 SSM 机制，也不能因四模型能训练而声称架构有效。

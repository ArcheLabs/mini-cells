# CLM v0.4 — Correctness → Smoke

本轮交付范围按用户确认：单次 Kaggle 最多三小时，支持中断恢复，先完成 Smoke。主实验32M targets × seeds17/29/43尚未冻结，不自动运行，也不生成正式科学结论。原v0.3源码、权重和结果不覆盖。

## 运行

使用本目录 run.ipynb，Tesla T4、Internet、沿用 GITHUB_TOKEN；Save & Run All。
自动执行：8项关键CPU检查 → 配置审计和两类基线匹配 → 固定Tokenizer/完整文档资产 → C0/C1/C2/C3/T0-P/T0-F共同短训练 → 独立验证、冻结反事实及性能记录 → 结果回传和SHA核验。

每模型Seed17、至多32768个有效targets，按共同完整文档组向下取整，至少3组；全部模型相同文档顺序、Chunk64、组8文档、上下文256。沿用v0.3验证过的AdamW配置：3e-4、betas(.9,.95)、decay.1、clip1、Token-based 10% warmup+cosine。目标是检查可训练性，不进行Smoke PPL排序后单独追加训练。

执行总起点从Notebook启动计时：数据准备最多20分钟，训练至130分钟，评估至160分钟，发布及余量至180分钟。提前完成立即退出；未完成如实标记。没有本地完整训练或CPU性能Benchmark。

## 架构和匹配口径

- d256、4头、8个独立SwiGLU Cell（中间维512）、Top-2、两步；LayerNorm、RoPE、零Dropout。
- 一个因果Transformer Workspace Encoder；其内部KV与向Cell提供的Workspace KV分别存储和统计。每步共享Query/Output投影，Query只依赖当前h，不直接依赖Controller。
- Cell独立参数，按Expert收集实际选中Token，逐Expert批量FFN，Scatter回原顺序；无Token丢弃，无全量专家Soft Mixing。辅助balance用全Router softmax统计，不代表执行全部专家。
- Controller为Token内GRU256，输入h_before/实际delta/r/稀疏门控向量；每Token零初始化，依次更新两次，仅通过后续Router影响预测。最后一次更新按规范保留并计入实际计算成本，不能影响不存在的第三步路由。
- C3第二步复用第一步r；其余参数和机制与C2相同。
- C0每步独立固定SwiGLU1024，使FFN激活矩阵成本等于两个512专家；总参数并未匹配C1。
- T0-P三Block，FFN1768，实际参数5,912,064；C2为5,913,096，误差约0.0175%。
- T0-F三Block，FFN1488，在声明的全窗口、MAC=2前向矩阵口径下精确匹配C2。此口径排除elementwise/norm/softmax/Dispatch及反向，不称为完整硬件FLOPs匹配。真实GPU延迟和算子分项另测，正式预算须结合实测冻结。

训练从开始使用Top-2，选中Logit上的Softmax门控可微，索引本身不求导；未选Expert不获得该Token的FFN任务梯度。负载辅助系数.001，噪声0，不使用REINFORCE或ST。共同参数使用相同Seed初始化，Controller和State-Router独立Seed。

BF16硬件优先，否则FP16+GradScaler1024；Controller GRU强制FP32。FP16非有限梯度跳过整组参数更新并Backoff，最多连续5次，同一组同一RNG重试；不裁剪非有限梯度，不推进Scheduler/Token计数。非有限Loss、非FP16梯度或step后NaN参数明确失败，不以恢复掩盖数值问题。

## 断点和回传

每个完整文档组结束原子保存模型、Optimizer、Scaler、Token调度进度、Python/NumPy/Torch/CUDA/Router RNG及组游标。恢复需source/config/data身份完全一致，已完成模型不重新训练。

默认复用本地同源目录。跨Kaggle版本使用仓库既有版本化Output恢复设施，不要求挂载旧Output：最多3个版本、45秒恢复窗口；同源已有结果却无法恢复时停止，不自动重训。外层运行身份固定，发布分支每次独立UUID。可用 `--resume-run <已恢复run目录>` 指定本地恢复，用 `--publish-only <run目录>` 只重试发布。`--no-recovery`是明确重新执行的选项，不作为恢复失败后的自动回退。

恢复清单引用不可变Hash对象，Checkpoint替换不会破坏旧清单；新清单原子提交后才清理不再引用的对象。大型Checkpoint与恢复对象只留Kaggle Output，Git只发布元数据、指标、校验和及定位信息。decision在Hash清单前冻结，发布后只追加独立Receipt并核验远端SHA。

## 诊断和研究边界

Smoke验证仅取固定8篇开发文档；包含normal、均匀无放回random Top-2、三个fixed pair、shuffled及Controller State Off。Shuffle从同一Chunk正常路径记录每步的配对和门控，再分别打乱Token对应，保留正常路径每步总体使用频率；实际反事实Controller依据强制动作和真实输出更新。

记录每步Logit均值/标准差、熵、Top1概率、概率Gap、所选Cell、State范数及其Logit贡献；记录模块梯度、每Expert激活数与参数张量更新幅度。Z0=0，Z1范数同时给出第一次状态变化量。性能包括训练CUDA前后向时间、总吞吐/显存，冻结推理的Encoder/Read/Router/Dispatch/Cell/Controller算子分项及两个物理KV成本。

本轮不是完整研究报告：三Seed主训练、固定训练FLOPs曲线、条件性参数匹配C0及Stateless Router复核、Oracle、完整Cell替换/余弦诊断、反事实专门延迟实验和正式Bootstrap研究决策暂未执行，不能用Smoke填充这些正式结果。H1–H4保持NOT_EVALUATED。

# 授权、实现解释及待冻结项

用户已明确授权“立即实施工程，仅运行 CPU 测试”（2026-10-20）。覆盖 Phase 0、Phase 1、Phase 2 前置代码；不授权实际 GPU、500-Step 开发训练或 Confirmation。

## 固定解释

- A：Cell 首 ID=(绝对 Token position+2×stage)%6，第二 ID=(首 ID+3)%6；批次/窗口近似均衡，不声称微批每阶段严格均分。B 使用与数据/全局 RNG 分离的 CPU 随机流，以置换均衡首 ID 和非零循环偏移保证双 Cell 不重复。
- Top-2 按 logits 降序稳定排序，同分时 Cell ID 升序。只对选中 logits 归一化；未选成员身份没有普通任务梯度。
- 成员调度允许跨 Stage 连续复用 Cell；Stage 内两个 ID 必须不同。
- 反事实枚举替换 slot0，保留基线 slot1；如重复，partner 使用原 slot0；重新计算选中双权重。每候选仍是四阶段双 Cell。固定后缀重放已执行的 ID 和权重；重规划后缀重新路由，两者分开报告，不能混合为同一估计量。
- 无标签分组：固定随机输入状态投影，在 Calibration 上冻结 median 阈值；Development 用同一规则且不与 discovery state ID 重叠。真实分组及机制阈值未冻结。Oracle 的目标仅用于只读评分，不进入 Gate 或分组输入。
- 交叉估计接口要求匹配状态的独立 selection/validation 反事实值。仅不同字符串不能证明统计独立；实际分析必须保存采样流和窗口身份证明。不满足则不能解释为无偏 Oracle 收益。
- Cell Swap 与 Route ID 置换对共享 Cell 池可能等价，不将其算作两项独立机制证据。
- CE 和 balance 分别 autograd.grad 后相加，避免第三次 backward，保持端到端状态依赖。只有 D 混合权重 detach。逐 Stage 报告对残差更新张量的损失梯度，不冒称已分离共享权重每阶段的全部梯度。
- 矩阵成本来自真实执行 Linear hooks、tied head 与 QK/AV；训练硬件 FLOPs、优化器成本和 GPU 时间尚未测量，统一 NOT_MEASURED。不是“活跃参数=FLOPs”。训练日志未安装 TorchDispatch/profiler。
- 训练诊断是本规范要求的短程工程版本；真实吞吐只能由获准 Pilot 测量，不能预先保证两小时。未设置进程硬终止计时器。

## 已知数值边界

复用 ContextRead 的注意力 Softmax 使用 float32。双精度合成测试中，将 CE/balance 的梯度分开再相加，与单个总损失反向有约 1e-12 的差异；对接近零梯度的 Adam 更新可放大至约 5e-8。测试分别约束梯度 rtol=1e-7/atol=3e-12、更新后权重 rtol=1e-5/atol=1e-7。相同实现的 CPU 中断/恢复须完整指纹精确一致；不借用上述容差放宽恢复要求。B 的随机均衡依赖批次组成，不能声称任意 full-sequence 与增量分块共享同一采样轨迹；路径固定重放可单独比较。

## Phase 2 候选工作量（未运行、未批准）

预案：三个 balance 候选各五次更新，再五组各五次，总计最多 40 次更新、20,480 Train Token。只有 Train 更新，Calibration 固定窗口选择 lambda，Development 不参与筛选。无有效系数则 NO_GO，不自动延长预算。`minimum_cell_share=null`，必须先冻结操作性门槛再单独授权 GPU；不随意填数字。

GPU smoke 入口先验证授权与配置 hash，才分配设备/读资产。授权是实施人员依据用户实际批准制作的配置绑定记录，不是安全签名或自动获得许可。真实资产复用旧私有 HF bundle 仅下载 assets，不下载/续训原 Checkpoint。五组从同 Seed 初始化。

## 正式训练之前仍需完成

真实 GPU 时间/显存/算子 FLOPs 校验；有依据的 balance 候选/曝光门槛及最终系数；数据及 Tokenizer 固定 hash 的实际读取验证；独立 Kaggle Session 的对象回读和完整恢复比较；资源投影；正式训练 Seed/预算/检查点矩阵；实践收益、非劣容差和互补机制阈值；正式源码/协议锁定与单独授权。

当前不提供 500-Step 正式调度器，不生成 `frozen.json`，不声称工程或科学全面通过。统计模块仅支持探索性、Seed 内 Story 配对 Bootstrap 和前三项 Holm；Seed 间重复独立报告，三 Seed 不等于高功效 Confirmation。

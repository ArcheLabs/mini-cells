# 双T4 / Seed17 / 2M Pilot 实施审计

代码提交：93987918df7bdd75e730707cdd9da8f9ddea31ed。本地检查使用小型CPU夹具；没有运行真实GPU训练。

六模型各2M，有效目标总计12M；不运行32M/三Seed。中央事务队列动态调度两个隔离Subprocess；精度、因果缓存、真实并发吞吐、Dry Run均是启动前阻断门禁。预算不通过不降低Targets。

本地硬件：Quadro K620, 5.0。状态：PILOT_BLOCKED（DUAL_GPU_BLOCKED: exactly two GPUs required; no single-GPU fallback）。真实双T4吞吐、利用率和两小时可行性未验证。

C2声明前向矩阵FLOPs由10,264,576降至8,679,424；重新计算T0-F，FFN=1144，声明成本误差=0.0。这是解析矩阵口径，不是硬件测量。

跨Kaggle恢复验证按用户指示删除，复用已有设施；本地Checkpoint/Scheduler一致性检查保留。旧Smoke/v0.3结果不修改。实际运行入口为pilot-2m.ipynb，必须选择GPU T4×2。

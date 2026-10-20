# R3c 工程交付

实现源码：`0b04e89ef6f14f7c9ad3aa6da5692df8daef9b9b`。Notebook 固定读取此版本。

## 完成范围

Phase 0 公开产物核查；Phase 1 新架构、五组对照、差异梯度/真实稀疏执行、RNG、完整 Checkpoint 恢复、标签隔离及统计接口；Phase 2 授权检查、Train/Calibration 系数选择和有界短 Pilot 前置入口；CPU Notebook、聚合报告、GitHub CPU CI。

五组均 1,005,830 参数，其中 Gate 9,734，核心 996,096；无 reserve slot。A/B/D 的任务梯度可达参数为996,096；C/E 为1,005,830。D 的均衡梯度是否启用取决于最终系数，当前未冻结。

CPU 合成报告显示：C/E Gate task 梯度非零，A/B/D 为零；D 保留 balance 梯度；各组实际 8 Cell +1 Context/Token；局部恢复的模型、优化器、RNG、步数和游标完整指纹精确一致。真实 Kaggle 跨 Session 验证没有运行，不以模拟存储测试替代。

## 未完成的实测与冻结

- 实际 GPU Pilot、硬件时间/显存/优化器成本、正式运行预算投影：NOT_MEASURED。
- balance 曝光门槛及系数、机制/实践收益阈值：待批准测量后冻结。
- 真实私有资产读取、HF 上传及独立 Kaggle 回读：未运行。
- 正式500-Step、多Seed研究及 Confirmation：未授权，未运行。
- CF-001 新私有块持久化：按已读公开状态单列 BLOCKED_PERSISTENCE，不依赖其数据。

因此状态仅为 **CPU_ENGINEERING_COMPLETE_GPU_PENDING**，不是完整 Engineering PASS 或 Scientific PASS。请勿把合成收敛、可视化或训练梯度正确性当作互补/专业化得到证明。当前提交不包含正式训练 Notebook、frozen.json 或自动放行。

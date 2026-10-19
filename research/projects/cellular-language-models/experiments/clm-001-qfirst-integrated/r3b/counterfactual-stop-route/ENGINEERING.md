# CF-001 工程交付记录

## 实施范围

仅新增独立 counterfactual-stop-route 研究单元、CPU 测试和 Actions。
未改原 R3/R3b 算法、训练器、Checkpoint、冻结结果、102项诊断身份或Confirmation。

复用：原 Model、Registry、STOP锚定 Snapshot.suffix、输入/资产核验器、只读RNG隔离、Kaggle Output检索与断点下载、原Git认证和推送工具。
新代码负责策略临时掩码、实际访问状态复算、采样统计、私有chunk和聚合发布。

## CPU 证据

- 独立 CF-001 合成测试：25项通过。
- 加上已有 Kaggle Output 恢复回归：59项通过，最终一轮耗时19.38秒。
- Ruff检查通过。
- 合成完整流程覆盖128个窗口、六种干预、双后缀审计、同目录幂等恢复；这不是实际数据诊断。
- Git发布使用临时本地bare远端，验证提交仅包含CF汇总/回执，且远端SHA一致。

## 真实环境状态

| 验证 | 状态 |
| --- | --- |
| 旧59项Saved Output原始结果回读 | NOT_EXECUTED |
| 原57条Checkpoint及真实资产 | NOT_EXECUTED |
| 真实Greedy baseline parity | NOT_EXECUTED |
| GPU Calibration耗时/显存 | NOT_EXECUTED |
| 新CF1/CF2/CF3 | NOT_EXECUTED |
| 新Output下一Session回读 | NOT_EXECUTED |

没有本机Kaggle凭据；未启动GPU任务。Notebook在原私有Notebook中一次启动，顺序自动执行。

## 已知限制

- 历史逐路径指纹未记录，不伪造历史逐路径相等证据。新执行器与原执行器严格比较，历史核对窗口、逐Story NLL、统计和转移矩阵。
- MC8置信区间是有限样本近似；UNRESOLVED单列。单Seed、部分状态抽样不能形成正式机制确认。
- GPU浮点运算可能导致严格状态复算不通过；失败会保存报告，不自动放宽或继续发布改善结论。
- Calibration只能估计耗时，不能保证整次云端耗时；分段只在已完成任务边界打包，不杀进程。
- 依赖已有Kaggle Output API、私有版本保存和权限；网络/存储失败均须在真实Notebook验收。没有把临时文件存在称作跨Session持久化。

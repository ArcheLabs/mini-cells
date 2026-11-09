# M4 — E5 实际 T4 工程验收报告

状态：**M4_ENGINEERING_COMPLETE**。已核对远端完整决策、源码 Hash、13 个发布文件
Hash 及训练/恢复/状态/消融记录；本次复核没有追加训练，也没有启动 M5/M6。

- 原训练源码：`c812f1797022934d951c713389883abb8a1c6715`。
- 结果提交：`f0ab964502c1caa15d09ea1185ce16bec1674a20`。
- Run：`epclm004-m4-c812f1797022-3e2c894a`。
- 原始指标：[audit/t4-smoke.json](audit/t4-smoke.json)。
- 发布与身份核验：[audit/t4-result-verification.json](audit/t4-result-verification.json)。

## 实际执行

| 项目 | 实测 |
|---|---:|
| GPU / 数值路径 | Tesla T4 / FP32，TF32 off |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| 静态参数 | 10,065,158 |
| 真实外层更新 | 2 |
| Input Tokens | 1,024 |
| Supervised / Query Targets | 8 / 8 |
| 内层 Batch Chunk 更新 | 16（64 次独立 Stream 更新） |
| 训练 Loss | 8.79299 / 9.01272（两个不同 Batch） |
| 裁剪前梯度范数 | 24.52437 / 17.72170，有限 |
| 静态 Embedding 改变量范数 | 0.42946 / 0.36269 |
| Fast Weights 改变量范数 | 6.49829 / 6.53885 |
| Momentum 范数 | 0.70118 / 0.71098 |
| Checkpoint 恢复最大误差 | 0 |
| Full / Chunk Cache 最大误差 | 2.4437904e-6 |
| 训练峰值显存 | 785,603,584 bytes，749.21 MiB |
| 推理峰值显存 | 85,832,192 bytes，81.86 MiB |
| 模型完整 Checkpoint | 175,911,149 bytes |
| Cell Checkpoint | 56,580 bytes |

两次真实优化器更新与动态 Memory 变化分别有记录。模型/Optimizer/RNG/Sampler/Ledger
由继承的原子 Checkpoint 保存，恢复检查通过。两次输入 Hash 与 M3 相同，数据身份也相同，
但这里只是相同 Smoke 数据，不构成质量比较实验。

## 融合与状态

CPU 已通过六种关闭单路/两路配置；T4 通过关闭 SWA、SSM、Memory 的三种消融。
关闭项权重均为 0，剩余项归一化；关闭 Memory 不修改 Cell。

一次验证探测的 Gate 均值：SWA **25.29%**，SSM **24.03%**，Memory **50.68%**。
它不是分支贡献度、重要性排名或互补性的统计证据。

单 Stream：SWA KV **327,680 bytes**，Mamba conv+SSM **393,216 bytes**，
Fast Weights / Momentum 各 **17,024 bytes**。Cell 与两种 Runtime 独立。
完整/分块执行检查、Session Reset 与跨 Batch 隔离检查通过。

推理探测约 0.16058 秒；关闭 SSM / Memory / SWA 的探测分别约
0.03456 / 0.15256 / 0.15162 秒。该固定顺序单次参考实现探测只作工程成本记录，
不是正式吞吐、质量保持或 Pareto 比较。

## 研究解释

MQAR Validation NLL：**9.26472 → 9.24502**，Query Exact Match 仍为 **0%**（4 个验证目标）。
两个不同训练 Batch 的 Loss 不能当同一批次上的收敛曲线；八个监督目标也不足以判断可学习性。

受控潜空间关联测试从 0% 达到 **100%**，保存后新会话恢复误差为 0，静态 Memory 参数未修改。
这证明实际 Memory 的学习/持久化/召回链路能在本 E5 环境中执行，**不是 MQAR 成功**。
目前没有证据证明 Mamba 与 Neural Memory 长程互补，也没有证明优于 E3/E4/Transformer。
**H1–H6，尤其 H4，仍为 NOT_EVALUATED。**

## 保留事项

- 不重复本次训练；原始结果分支保持不动。
- Kaggle Script Version ID 本次没有由运行时暴露，不能以 Git SHA 或版本序号替代。
- 模型和 Cell 权重仍保存在本次 Kaggle Output；尚无 M4 永久权重归档回执。
  当前工程完成不等于权重保全完成。进入下一阶段前应单独保存实际权重。
- 此次只复核已发布文件及运行时报告，没有重新下载二进制 Checkpoint。
- M5/M6、20M/100M 未启动，M1–M3 的冻结资料与权重均未修改。

最初 CPU 阶段报告保存在 [audit/REPORT-before-t4.zh-CN.md](audit/REPORT-before-t4.zh-CN.md)。

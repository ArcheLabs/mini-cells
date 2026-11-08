# EPCLM-004 M3 — T4 验收及发布修复

## 当前结论

**Gate 1–6 均 PASS。** 首次 T4 结果已发布至
`kaggle/epclm004-m3-2b5362884492-d8942eb1`，提交
`7d0c3eb2a146da9d3bc60cd3c5944cf73d4bdb58`。
第二次补写完成状态时发生发布分支切换冲突。训练没有失败，**不需要重训**。
最终发布修复记录位于 `kaggle/epclm004-m3-publication-repair-d8942eb1`。
仅在该修复分支推送并核验远端 SHA 后，完成状态才成立。
H3 仍为 **NOT_EVALUATED**；没有启动 M4–M6 或扩大 Token Budget。

## T4 实际结果

| 项目 | 结果 |
|---|---|
| GPU / 精度 | Tesla T4，FP32，TF32 关闭 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| 静态参数 | 10,168,580；比 E1 增加 0.3740% |
| 外层真实更新 | 2 |
| Input / Supervised / Query Tokens | 1,024 / 8 / 8 |
| 内层更新 | 16 次 Batch Chunk 转移，即 64 次独立 Stream 转移 |
| Fast Weights 改变量 | 两次分别为 6.4982867 / 6.5231202 |
| Momentum 范数 | 0.7011780 / 0.7142604 |
| 模型 Checkpoint 恢复误差 | 0 |
| Optimizer 恢复 | PASS |
| 跨会话正向召回 | 100%，恢复误差 0 |
| 训练峰值显存 | 411,926,016 bytes，约 392.84 MiB |
| 推理峰值显存 | 84,866,048 bytes，约 80.93 MiB |
| Cell Checkpoint | 56,507 bytes；写入约 0.01636 秒 |
| Memory 更新 / Recall | 128 次潜空间适应合计约 0.27659 秒 / 约 0.000624 秒 |

时间和显存只代表本次 Smoke；潜空间关联正向检查不等于语言 MQAR 或 H3 证据。
Gate 1–5 的原始数学、因果、状态、集成记录保存在已发布目录的 `audit/`。
完整高阶元梯度及有限差分定义见 [README](README.md)。

## 故障与最小修复

共享发布器会切换当前 Checkout。首次发布完成后，第二次准备产物修改了已跟踪
的结果文件，随后切回 main 基线发生冲突。该问题发生在已完成的实验之后。
M3 现在为每次发布创建独立的临时 Git Checkout，继续复用原有产物整理、认证、
提交和 SHA 核验机制。实验源码目录不再被发布过程切换。`--publish-only`
也会完成最终状态补写，不进入 Gate 或训练路径。

增加一个有针对性的真实 Git 回归测试：连续发布两次，确认最终状态、远端文件
和原始实验 Checkout 均正确。结果：**1 passed in 1.85s**。没有重跑已通过的
数学测试，没有重训 CPU 或 GPU 模型。M1、M2 和共享发布器均未修改。

本次只复核已发布文件 Hash 及运行时记录；权重仍保留在 Kaggle Output，
未重新下载或重新训练。原始结果分支和首次 decision 均保留；修复在新分支交付。

`diffusers` 的依赖提示不是这次 Git 异常的原因。未修改冻结依赖以处理无关包。
首次 CPU 阶段报告保存在 [audit/REPORT-before-t4.zh-CN.md](audit/REPORT-before-t4.zh-CN.md)。

# EPCLM-006 实施与 CPU 预检报告

当前状态为 **ENGINEERING_PASS（CPU）**。T4 门禁、预算探测、正式训练与召回质量评价待执行。正式优化器更新为 0。

## 工程证据

- 冻结 E3 实例化为 10,135,808 参数。CPU 完整与分段缓存结果最大误差约 7.39e-6。因果性、固定形状下的批内隔离和 W32 缓存边界通过。
- Canonical 输入只接受有限 Token 片段，走 E3 的无 Runtime 前向。相同 Query Packet 逐位复现。带 Session 状态的 I0 与重置编码 I1 的最大表示差约 3.47。该差异说明状态依赖，未用于推断质量。
- 一个真实 M2 CPU 小模型事务完成 256 输入 Token、16 个目标和一次更新。Fast Weight 学习路径对共享初始化及 K/V/Q 投影的梯度非零。它仅用于验证工程入口。
- C0、C1、C2 分别使用新建 Target Runtime。正确与无关 Cell 由不同支持事实获得。C3 保留创建者上下文，C3-reset 用于隔离导入影响。
- 新进程恢复 Cell 后的 Query 输出、以及分别读取 Target Mamba 与 SWA 状态后的续读输出，最大误差均为 0。Cell Artifact 的文件只有 Manifest、张量和完整性记录。它不携带 E3 权重、Session Runtime 或源会话历史。
- 替换、失败替换、只读哈希、Fork、卸载和本地删除均通过。新进程复验依赖独立写出的 Target 状态文件，Cell 本身不包含这些文件。卸载保留已有 Session Runtime。
- 两项专项测试通过，覆盖分割隔离、记账、封存 Test、重新计算 Hash 后的非法张量形状及损坏 Manifest 拒绝。实现过程中发现新入口名称与历史 preflight 模块冲突，已通过改用独立模块名修复。

## 冻结的首轮设计

采用 EPCLM-004 M2 E3 配置与一个 Titans Cell，单训练 Seed、32 次外层更新、8,192 个输入 Token、512 个 Answer Targets。每 Episode 为 8 个源事实、16 个 Target 前缀 Token 和 4 个 Query Packet。E3 的 Canonical Query 编码额外处理每 Episode 16 个 Token 位置。原始输入记账与实际额外计算分开。

训练与留出集的实体和值使用不同的 Token-ID 池，正式评价为 3×64 Episode，C0–C3 共享一个 Checkpoint。Test 保存生成规则摘要，未构造 Test 样本，因此没有声称持有实际 Test 样本 Hash。Cell Dropout 使用 0.15 的初始工程配置，采用保留一个有效 lane 的策略，未做额外训练消融校准。来源事实单 Token EM 的两项主要差异均须达到 3 个百分点，且配对 Bootstrap 下界大于 0。另报告联合四答案 Episode EM。

## 待执行

本机 GPU 是 Quadro K620。Notebook 将在 Tesla T4 上检查 E3、Portable Cell 及进程恢复，再用真实 batch 做无更新预算探测。正式运行前无法回答 C1 是否优于 C0/C2、与 C3 的差距、部署侧导入成本或是否值得进入 Confirmation。

[运行说明](README.md) · [CPU 证据](audit/cpu-preflight.json) · [源码与数据清单](audit/source-and-data.json)

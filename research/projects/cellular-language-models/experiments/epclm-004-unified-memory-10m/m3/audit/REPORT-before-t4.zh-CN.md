# EPCLM-004 M3 — 工程实施报告

## 结论与边界

当前状态：**`BLOCKED_GPU_ACCESS`**。完整 E4 实现、Gate 1–5 的实际 CPU
验证和一键 T4 入口已交付，但 **没有执行 T4 Smoke**，不能标记
`M3_ENGINEERING_COMPLETE`。H3 始终为 **`NOT_EVALUATED`**。

本环境没有 Kaggle API 凭据；浏览器控制工具在连接前因工作区路径错误失败。
本机 CPU 验证不替代 Tesla T4，不使用其他 GPU、降阶算法或未经验证的精度制造 PASS。
无 M4–M6 实现，无 Screening/Confirmation，无历史 M1/M2 修改。

## 实现及公式来源

公式依据 [Titans 原论文 §3.1–3.2、§4.2](https://arxiv.org/html/2501.00663v1)，
未采用原规格中损坏的排版公式。完整定义及偏离见 [README](README.md)
和 [implementation_deviations.md](implementation_deviations.md)。

- 投影 K/V/Q，2 层 GELU Memory MLP、Residual、LayerNorm。
- Fast Weights 使用瞬时误差梯度，Momentum 保留历史 Surprise，Adaptive
  Forgetting 逐 Chunk 衰减；内层没有 AdamW。
- `meta_train` 使用完整高阶 Functional Update，保留整个 Episode 的梯度。
  `online_adapt` 显式冻结静态模型并截断每次动态更新；`read_only` 不写状态。
- MAG-lite 复用未改动的 M2 E1，增加一个共享 Memory 和可观测的标量 Gate。
  关闭 Memory 的输出与 E1 完全一致。继承的 SWA-only Cached API 已禁用。
- 所有当前 Chunk 的 Recall 完成后才 Learn；Chunk 写入只影响后续 Chunk。
- 静态模型与动态 Cell 分离。Fast Weights、Momentum 分开存储并由最后原子
  写入的 Manifest 选择同一个 Generation。Hash、接口及模型身份不匹配即拒绝。

## 自动门禁实测

| 门禁 | 状态 | 证据 |
|---|---|---|
| 1 数学正确性 | PASS | 三步递推 Oracle；FP64 元梯度有限差分 |
| 2 因果性 | PASS | 未来输入影响、Read-before-write、跨 Batch 隔离误差均为 0 |
| 3 状态管理 | PASS | Read-only、Online Adapt、双组件恢复、损坏拒绝、Fork 隔离 |
| 4 MAG 集成 | PASS | 分支消融、有限梯度、两次更新、Checkpoint 恢复、Runtime Reset |
| 5 Cross-session Recall | PASS | 真实潜空间关联学习后保存，清空动态状态，新会话恢复召回 |
| 6 T4 Smoke | **NOT_RUN / BLOCKED** | 未获得 Kaggle 执行访问，未填写 GPU 数据 |
| 结果发布 | 源码与 CPU 证据随本分支交付 | T4 结果须由 Notebook 实跑并校验远端 SHA |

[CPU 原始门禁记录](audit/cpu/) 包含误差、容差、更新证据和 Checkpoint Hash；
[参数统计](audit/parameter-counts.json) 是真实实例化统计，非 GPU 运行数据。
测试使用一个顺序套件，直接复用运行入口中的五个 Gate，不另建重复测试框架。

有限差分：初始化 W1、LayerNorm 权重、K/V/Q Projection 及控制投影的代表性
坐标均通过；最大绝对误差 **6.021997900340637e-10**，每项容差为
`2e-6 + 2e-4 * abs(numerical_gradient)`，中心差分步长 `1e-5`。
三步 Fast Weights / Momentum 独立递推最大误差 **4.163336342344337e-17**。
这检查选定坐标，不宣称逐元素穷尽完整 Jacobian。

## 实际训练与恢复证据

CPU Tiny 集成模型：**4,036** 静态参数。使用原样复用的 M2 `training.fit`，
两次真实外层更新，中间保存并由新模型恢复。恢复过程只执行第二次更新，
保留第一条训练历史。Model、Optimizer、RNG、Sampler、Ledger 均在同一原子载荷中。

- Committed Updates：**2**。
- Input / Tensor Tokens：**32 / 32**，Padding：**0**。
- Supervised / Query Targets：**4 / 4**。
- 实际内层更新：**8 个 Batch Chunk 更新**，与单次检查 Forward 的计数分开。
- Optimizer 步数均为 2，恢复后输出最大误差 **0**。
- 这是工程合成 Fixture，不是 10M E4 训练，也不是质量评估。

跨会话正向检查使用真实 **Memory Dim 32 / Hidden 64**，四个 Key 与非恒等
排列的 Value。通过 **128 次内层梯度更新**使召回准确率从 **0% 到 100%**。
保存后丢弃动态状态，在新会话只恢复 Cell；Recall 不输入答案。
恢复后误差 **0**，静态 Memory 参数没有修改。Fast Weights 改变量和非零
Momentum 范数记录在 Gate 5 JSON。此任务直接检查潜空间记忆，不是语言 MQAR。

## 资源与运行环境

| 项目 | 实测 / 状态 |
|---|---|
| 10M E4 静态参数 | **10,168,580** |
| 冻结 E1 静态参数 | **10,130,688** |
| 参数差异 | **+0.3740%**，公共参数初始化逐元素一致 |
| Fast Weights / Cell | **4,256 参数，17,024 字节（FP32）** |
| Momentum / Cell | **17,024 字节** |
| KV / Mamba State 依赖 | **0** |
| 活跃 Runtime Cell 快照 | Fast Weights + Momentum；可清除，独立持久 Cell 保留 |
| Cell Checkpoint | Gate 5 JSON 记录真实字节数和写入耗时 |
| Memory 更新与 Recall 耗时 | Gate 5 JSON，CPU 正向检查范围；非 T4 性能 |
| GPU 训练 / 推理峰值显存 | **NOT_MEASURED** |
| T4 训练更新与吞吐 | **NOT_RUN** |
| 本地环境 | Python 3.11.18，PyTorch 2.6.0+cpu；不新增依赖 |

## T4 一次性入口与验收

运行 [notebook.ipynb](notebook.ipynb)：T4、Internet、现有 `GITHUB_TOKEN`，
Save & Run All。入口不在阶段之间要求确认：CPU Gate → T4 Gate → 发布。

冻结配置 [configs/smoke.json](configs/smoke.json)：真实 10M E4，FP32，TF32
关闭，B4、长度 128、Chunk 16、两次 AdamW 外层更新；所有内层更新仍为 Titans
Functional Rule。复用冻结的 8192 Tokenizer 和 MQAR 数据协议，Test 不访问。
预期仅 1,024 Input Tokens / 8 Supervised Targets，不扩大训练规模。

执行上限 5,400 秒，发布预留 600 秒，Notebook 硬上限 7,200 秒。
失败即停止后续 Gate，保留有效检查点；成功记录按身份与 Hash 复用。
真实 GPU 结果将记录静态/动态变化、两次提交、完整恢复、跨会话召回、显存、
耗时及 Hash。只有 Gate 6 实际 PASS 且产物发布验证成功，才写入完成状态。

已知限制：MAG-lite 与完整 Titans 不等价；Chunk-mean 更新、共享 Memory 的
Embedding 输入、标量 Gate 均已显式声明。没有官方数值等价性、长程任务质量、
大预算性能或 H3 证据。Checkpoint 不保存跨会话高阶计算图。

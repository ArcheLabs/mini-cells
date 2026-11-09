# M4 — 当前实施报告

状态：**M4_CPU_ENGINEERING_PASS_T4_PENDING**。真实 T4 训练尚未执行，不标记 M4 完成。

E5 三路融合已实现。实际静态参数 **10,065,158**，详见参数审计。公共 Embedding、
形状兼容的 M3 初始化和 M2 前八个 Mamba 初始化均核对一致；未使用历史训练权重。

CPU 新行为测试：**1 passed**。小模型 8022 个静态参数。

- 因果性最大误差：0.0。
- Batch 隔离误差：5.960464477539063e-08。
- Full 与 Chunk Cache 最大误差：7.450580596923828e-08。
- Checkpoint 恢复误差：0.0。
- 六种单路/两路关闭配置均通过，关闭 Memory 时 Cell 不变。
- Read-only 不写 Cell；在线适应不修改静态参数；Reset 使旧 Session Runtime 失效。
- 通过原 M2 训练器完成两次 Tiny 真实更新，中断后新模型只执行剩余一次。
- Input Tokens：32；Supervised / Query Targets：4 / 4；无 Padding。
- 本轮没有重新执行 M2/M3 完整测试，也没有重训旧模型。

CPU 训练损失：[3.8970186710357666, 3.8846049308776855]。
这些仅是工程 Fixture 数值，不能解释为模型质量。

目标 T4 协议已固定为同一 8192 Tokenizer、MQAR、FP32、两次真实外层更新，
发布包含完整 Gate、静态/动态资源、Checkpoint/Restore 和新 E5 Cell 正向召回证据。
在真实运行之前，GPU 显存、成本、10M 训练结果及最终状态均为待验证。

当前环境无法直接启动 Kaggle；使用 notebook.ipynb 一次运行自动完成余下流程。
M5/M6、20M/100M 未启动；H1–H6，尤其 H4，均为 **NOT_EVALUATED**。

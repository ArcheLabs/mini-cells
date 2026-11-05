# 本地验证

已执行最小检查：

- 硬件门禁：SM75 拒绝，SM80/SM90 进入运行时审计。
- Mock 无 CUDA：生成 BLOCKED_HARDWARE 报告，0 训练 Token，0 Optimizer Updates，无 Checkpoint。
- 官方 Commit 不匹配：生成 INCOMPLETE 并保留诊断，不执行模型。
- Notebook/Python 语法、JSON 源码锁和 Git whitespace 检查。

未执行 GPU、内核安装、吞吐测量、反向传播、训练或真实 GitHub 结果发布。发布复用仓库现有实现；真实远端回执仍须由运行时验证。当前锁定的是源码，安装解析依赖由实际运行 freeze 记录。

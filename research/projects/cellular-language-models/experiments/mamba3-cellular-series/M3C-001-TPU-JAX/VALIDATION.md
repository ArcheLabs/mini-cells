# 本地验证（CPU 合成工程证据）

Python3.11.18、JAX/JAXlib0.10.2、Flax0.12.8。独立临时环境，无真实语料、训练或参数更新。

- 复用上游三项参考检查：SISO logits-vs-NumPy-golden、SISO SSD-vs-sequential、MIMO SSD-vs-sequential，全部通过；未运行完整上游测试矩阵。
- 最小模型核心门禁 SISO → MIMO 均通过：JIT/Eager、Full/Chunk/Decode logits 与最终 State、因果性、缓存重置/内存恢复、批次隔离、固定缓存大小及有限非零梯度。
- 最大比较绝对误差：SISO 1.430511474609375e-6，MIMO 2.86102294921875e-6。
- 合成 Loss / Gradient Norm：SISO 4.7174305916 / 4.4461521641；MIMO 4.7415161133 / 3.7342101816。仅数值执行证据，不是训练结果。
- CPU 环境调用正式 TPU 入口：BLOCKED_TPU_UNAVAILABLE，SISO/MIMO NOT_RUN，未静默降级。
- Notebook/Python 语法及 JSON 检查通过。

未运行 TPU，也未运行官方 CUDA 对照。official_kernel_equivalence=NOT_VERIFIED；科学假设 NOT_EVALUATED；M3C-002 仍等待实际 TPU 门禁。

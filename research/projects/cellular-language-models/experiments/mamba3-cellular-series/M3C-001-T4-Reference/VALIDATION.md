# 最小本地验证

Python3.11.18、PyTorch2.6.0+cpu、einops0.8.2，固定合成样本，单线程。未运行 GPU、正式训练、FP16 或性能 Benchmark。

SISO → MIMO 两模式门禁通过，完整前向与前缀+Cached Step 的 logits / 最终State、因果性、缓存重置、内存恢复、批次隔离与有限非零梯度均通过。

| 模式 | 最大比较误差 | 合成 CE | Gradient Norm |
|---|---:|---:|---:|
| SISO | 3.5762786865234375e-7 | 4.1812982559 | 2.2249266836 |
| MIMO Rank4 | 6.258487701416016e-7 | 4.1748743057 | 2.8171765657 |

首次构造检查仅初始化 A_log/dt_bias，未处理构造器 empty 的D，未通过数值门禁；已改为完整复用上游 create_toy_model 初始化规则，再次检查通过。未修改第三方源码或放宽容差。

额外关键边界：Mock 无 GPU → BLOCKED_GPU_UNAVAILABLE，SISO/MIMO NOT_RUN，不产生 GPU_PASS。Python/Notebook语法、JSON及源码哈希检查通过。没有重复运行完整测试矩阵。

本地结果仅 CPU_ENGINEERING_PASS；T4 实测、官方 Kernel 等价性及科学假设尚未验证。0 参数更新，0 正式训练Token（每模式22个合成预测Targets用于反向检查）。

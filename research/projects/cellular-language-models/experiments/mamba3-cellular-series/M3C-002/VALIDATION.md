# 本地最小验证

Python3.11.18、PyTorch2.6.0+cpu、einops0.8.2。仅合成正确性检查，无GPU或CPU性能Benchmark。

三项核心检查通过，并通过一次完整入口CPU集成调用：

- Base与Reference原生logits、Adapter bypass差异均为0，公共Reference权重逐Tensor一致。
- Full/Cached Decode logits最大误差1.7881393432617188e-7；全部Cache/输出比较最大误差6.556510925292969e-7，低于冻结1e-4容差。
- 未来扰动、Batch隔离、重置、缓存输入不被修改、Batch重排和解析滑动平均oracle均通过。
- 每个Workspace参数梯度有限且非零；原生参数梯度均存在且有限。合成CE=4.1388816833，0参数更新。
- B2/Layers2/Dim32/Window4/FP32的Workspace KV logical/retained storage均为4096字节，原生Mamba存储另计。
- Mock无GPU：BLOCKED_GPU_UNAVAILABLE，correctness NOT_RUN，不发布成GPU成功。
- Python/Notebook语法、协议JSON和源码身份检查通过。

GPU推理延迟、VRAM、Kernel活动尚未实测；MQAR、PPL、跨Seed科学结论仍NOT_EVALUATED。未修改第三方核心、历史报告或既有审计入口，未新增完整单元测试矩阵。

# M3C-001：官方模块源码与推理审计

当前交付：审计入口已实现，真实 GPU 结果未运行。训练授权 0 Token。

运行 notebook.ipynb，开启 Internet，沿用 GITHUB_TOKEN Secret。入口自动锁定官方 Commit、核对源码哈希、检查硬件、运行可用路径并复用仓库结果发布器；不需训练数据或历史 Checkpoint。

## 已发现的限制

- 官方源码锁定 `e9594ce1c732d97440f0332fdc43170a2294dbfa`，依赖 `triton>=3.5.0`、`tilelang==0.1.8`、`quack-kernels>=0.3.4`。
- Triton 当前 NVIDIA 支持范围为 Compute Capability 8.0+；T4 为 SM75，因此入口在安装内核前标记 BLOCKED_HARDWARE。此状态是依赖支持门禁，不声称做过 T4 内核失败实测。不要为此反复运行 T4。
- 官方 Mamba3.step 注释明确仅在 H100 上测试。优先在 H100 执行；其他 CC>=8.0 设备仍需实际门禁，不保证兼容。
- 官方单模块 forward 的扫描调用固定 Input_States=None，不暴露多 Token 分块续接。本次仅检查 Full Sequence 与 Prefill 后逐 Token Decode，不能声称多 Token Chunk 续接通过。
- 官方 step 输入为二维 [B,D]，输出同形；调用时不把单 Token 保留为 [B,1,D]。

## 输出与范围

SISO/MIMO 独立报告，不可用路径保留异常。检查随机初始化单模块的有限输出、因果性、Full/Decode 一致性、批次隔离、新建缓存重置、内存快照恢复。恢复仅复制内存 Tensor，不生成 Checkpoint，也不验证跨 Kaggle 恢复。

State 共四项：angle_dt、SSM（FP32），K、V（参数精度）；形状和实际字节运行时记录。Prefill/Decode 3 次 warmup、5 次采样，CUDA 同步，报告样本中位数与最大值作为小样本 P95；此样本量不支持精密性能结论。测量仅单模块，不是端到端 LM。

容差预先固定 atol=rtol=0.02（BF16），记录实际最大误差，禁止失败后自动放宽。训练数据、反向、参数更新、LM 包装、参数匹配对照均不在本次授权内。任何 MODULE_INFERENCE_PASS 不等于完整 Gate001 或科学假设通过。

安装阶段遵循官方依赖约束并记录安装后完整 pip freeze；运行配置固定，依赖不是跨平台完全冻结环境，后续复现应使用运行报告中的解析版本。安装超时保留 INCOMPLETE 并尝试发布。历史实验文件不变。

来源：[官方锁定源码](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py)、[官方依赖](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/pyproject.toml)、[Triton compatibility](https://github.com/triton-lang/triton#compatibility)。

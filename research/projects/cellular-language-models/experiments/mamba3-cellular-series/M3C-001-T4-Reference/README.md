# M3C-001-T4-Reference

当前 M3C-001 执行路径，独立运行记录。使用 VikramLex/mamba3-minimal，源码锁定 `15ba9983123703d9f4f7d6f8709152d0a07edbd9`，Apache-2.0。第三方教学/Reference 实现，非官方实现；不声称官方 Kernel 数值等价。

Kaggle 选择 T4、Internet，沿用 GITHUB_TOKEN，运行 notebook.ipynb 的 Save & Run All。无需 Triton、TileLang、自定义 CUDA Kernel、训练语料或 Checkpoint；仅依赖已有 PyTorch 与 einops。首轮只执行 FP32，关闭 TF32，不自动启用 AMP、FP16 或训练。总 Notebook 上限120分钟，审计600秒。

## 固定最小门禁

先 SISO，再 MIMO Rank4；失败保留 INCOMPLETE 和诊断。模型 hidden32、1层、state8、head8、expand2、chunk4、vocab64、Batch2、长度12（跨三 Chunk），固定 Seed18001。构造器的 A_log、D、dt_bias 使用 empty；审计完整复用上游 create_toy_model 的初始化规则（A_log uniform(-4,-1)、D=1、dt_bias uniform(.001,.1)、普通矩阵 normal std.02、保留 BC bias 和 MIMO 初始化），避免随机内存和默认大幅嵌入影响数值检查。

- FP32 Full Forward 输出有限，Next-token CE Backward；每个可训练参数梯度必须存在且全部有限，总范数非零。
- Full Sequence vs 前缀4 Token +8次 Cached Step，比较全部 logits 与最终 State。
- atol=rtol=1e-4，运行前冻结，失败不自动放宽；记录最大误差。
- 因果性、批次隔离、缓存重置、内存快照恢复、不同历史长度下固定 State 形状。
- State 逐字段记录形状、精度和字节数。0 Optimizer Updates、0 正式训练 Token；报告另列22个合成 Prediction Targets，避免把反向检查当成正式训练。

未执行性能矩阵、完整上游测试、Parity Task训练、正式训练、FP16、Checkpoint 或跨 Kaggle 恢复。上游 tests/test_mimo.py 的 Full/Step 思路被复用并扩展为带断言的两模式单入口；不重复执行该上游脚本。

## 已核查的实现边界

本实现采用两个 Chunked SSD 分支（当前项/前一项）之和，非逐 Token Full Forward。Full 长度必须整除 chunk_size；有 Cache 时只支持单 Token Step，不声称多 Token Cached Chunk 续接已通过。

其 `A=-exp(A_log)` 为静态每 Head 参数，官方锁定实现使用输入依赖 A；Reference 使用 SSM State、prev_Bx（上一 Token 的 B⊗x）、cum_angle 三项缓存，而官方使用 angle、SSM、K、V 四项。Reference 包装还包含 SwiGLU MLP。不能把“同属梯形递推/旋转状态思路”写成“参数化、State 布局、数学定义或数值行为完全相同”。本次不改第三方核心以强求一致。

GPU_REFERENCE_GATES_PASS 仅表示实际 GPU 上通过本协议。仍记录 official_kernel_equivalence=NOT_VERIFIED、scientific_hypotheses=NOT_EVALUATED；设备型号保留，非T4设备不声称T4实测。CPU_ENGINEERING_PASS 仅为本地合成检查。

通过后沿用原 M3C-002 至006研究路线，以该 Reference 为标明身份的底座。原官方 PyTorch BLOCKED_HARDWARE 和 TPU/JAX 历史记录保留；不要求等待H100，不将旧记录改为成功。

复用仓库发布器自动推送独立结果并核验远端SHA。失败保留Output；`--publish-only`只重试发布，不重复检查。

来源：[锁定源码](https://github.com/VikramLex/mamba3-minimal/blob/15ba9983123703d9f4f7d6f8709152d0a07edbd9/mamba3.py)、[上游一致性脚本](https://github.com/VikramLex/mamba3-minimal/blob/15ba9983123703d9f4f7d6f8709152d0a07edbd9/tests/test_mimo.py)。

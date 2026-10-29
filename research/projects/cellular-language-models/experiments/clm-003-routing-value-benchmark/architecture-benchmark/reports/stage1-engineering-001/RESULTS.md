# Stage 1 工程结果

**工程验证 PASS；阶段 2 NO-GO。** 仅极小 CPU fixture，不能作架构质量或资源优势结论。

最终验证：2 项测试通过；每组 2 主更新/32 目标 Token，CLM 另 1 次候选优化及完整 22 尺度扫描。Confirmation 模型访问为 0。

|组别|总参数|可训练参数|完整路由参数|Cell 调用/目标 Token|
|---|---:|---:|---:|---:|
|CLM|35054|1680|318|8.0|
|MoE|34784|1584|48|16.0|
|Dense-AllCell-Mean|34736|1536|0|48.0|

上述仅为 width=8 的工程模型，不是科学配置。未声称计算匹配。FLOPs 仅覆盖 Linear GEMM；ContextRead/functional head 等未覆盖。延迟仅 CPU forward 5 次采样，内存和能耗未测量。

完整方法及剩余门禁见 ../../STAGE1_IMPLEMENTATION.md。正式 Seed、预算、统计门槛、科学 split 和自然语料均未冻结。停止并等待审阅及阶段 2 明确授权。

# GC002 — CLM 与 Transformer 公平架构比较

**PROTOCOL DRAFT — NOT AUTHORIZED TO TRAIN。** 只执行 Phase 0/1 工程准备，训练授权 Token 数为 0。历史 GC001 和 AI/人工评审状态保持不变。

- [原始需求文档](SPEC.zh-CN.md)，原文保留。
- [机器可读草案](protocol.json)：不是冻结协议；不生成表示冻结完成的 `protocol.sha256`。
- [实施说明与待决项](IMPLEMENTATION-NOTES.zh-CN.md)。
- [唯一准备 Notebook](prepare.ipynb)：原有 Kaggle Notebook、Internet、GITHUB_TOKEN 和原有 KAGGLE_API_TOKEN，Save & Run All。CPU 可用；若已有 GPU，仅运行无训练的工程推理检查。自动从已知版本 142 只恢复四份数据资产并核验，不加载模型权重；无需 Kaggle Output 挂载或手填 SHA。

本地只执行合成正确性与元数据审计，不运行训练或 CPU 性能 Benchmark：

```bash
python research/projects/cellular-language-models/experiments/epclm-core-001/gc002-fair-architecture/run.py --prepare-only --output /tmp/gc002-preparation
```

使用既有 RMSNorm、SwiGLU、GQA 投影、Native T1 权重布局、历史结果读取和 GitHub 发布器。仅新建实验目录，不改原始 Native/CLM 模型、Trainer 或 GC001 协议。`--train`、Pilot 或预算扩展入口不存在。

两项必要测试覆盖 Native 和参数匹配候选的缓存等价性及参数可达性。缓存比较 32/128/385 Token、batch=1、FP32、逐 Token Logits/Argmax、滑动、绝对位置、分块 Prefill、EOS 停止和保存恢复，最大误差门限 1e-5。CPU 正确性通过不替代 GPU 环境验证。

## 门禁

- Phase 0：历史报告与训练曲线的元数据核验；原始 Tokenizer/语料字节、实际 Token 消费顺序未独立复核，不能标 `BASELINE_AUDIT_PASS`。
- Phase 1：KV 合成正确性与实际 CPU 张量存储可检查；无 GPU 时延迟/显存显式 NOT_MEASURED。参数候选不是正式冻结配置。
- Phase 2：正式 Seed、Token/FLOPs 预算、预算检查点、记忆任务与统计口径未冻结；不宣称 PROTOCOL_FROZEN。
- Phase 3/4：没有训练入口，不自动启动。门禁完成后提交完整预算与 GPU 小时计划，再获得单独授权。

准备输出位于 `artifacts/experiments/epclm-core-001/gc002-fair-architecture/preparation/`，并为每份 JSON 附源码/文件/协议身份。未知身份为 null 且写明原因，不填写虚构的 Tokenizer、模型或性能哈希。随机权重的 KV 检查不构成历史训练 Checkpoint 等价性或科学效果证据。

## 后续准备补充

[运行计划](RUN-PLAN.zh-CN.md) 和 [冻结候选](freeze-candidate.json) 现已给出 A/B 12 个作业、3M Token 的**待批准提案**；当前授权仍为 0。历史逐 Update 输入哈希未发布，审计将此缺口保留为 `HISTORICAL_RUNTIME_ORDER_UNVERIFIABLE`，不再声称能事后验证实际顺序。原始 `SPEC.zh-CN.md` 和 GC001 均未改变。

后续 Notebook 按完整 Update 形状检查四个模型的前向/反向/裁剪/零学习率 AdamW 清单，复用已完成的 CUDA KV 检查。合成夹具调用 Optimizer 仅验证运算路径，模型权重不变，不消费真实语料 Token。`flops_audit.py` 报告的是有明确规则的标量等价算术计数，不是实测硬件 FLOPs；未知算子、实际提交计量和目标环境验证仍会阻断正式 Compute-Matched 就绪。

`exposure.py` 提供 commit-only、unique/repeated、顺序链哈希及身份绑定的恢复；`training_path.py` 提供事务采样和逐 lane 重置的并行 Chunk Teacher Forcing。只有采样 resolve(True) 才提交 ledger；溢出 rollback 恢复同一候选输入。正式 Checkpoint 必须将采样器、状态、ledger、Optimizer/Scheduler/RNG 作为同一原子事务保存。没有新增正式训练入口，也未把准备结果标记为 PROTOCOL_FROZEN。

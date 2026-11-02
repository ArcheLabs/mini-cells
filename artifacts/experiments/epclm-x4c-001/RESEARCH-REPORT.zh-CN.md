# EPCLM-X4C-001 研究报告

运行状态：STOPPED_AT_STAGE_DECISION。训练：COMPLETE；评估：COMPLETE。

单个开发 Seed。Independent Test 未访问；多 Seed 复核未执行。未执行项目不构成正面或负面研究结论。

| 模型 | 训练状态 | 参数 | 预测目标 | Validation PPL | Holdout PPL | 前向 FLOPs/Token（解析） | Batch-1 Decode Median |
|---|---|---:|---:|---:|---:|---:|---:|
| G1/A0 | COMPLETE | 1893912 | 1000000 | 47.230121100037 | 47.03330914058547 | 13109790.464 | 0.017347946500024136 |
| G1/A1 | COMPLETE | 1892784 | 1000000 | 48.616837564995315 | 48.50062874158141 | 6939919.232 | 0.009204017999991265 |
| G1/A2 | COMPLETE | 1892220 | 1000000 | 51.3971830310497 | 51.11548121728749 | 3854983.616 | 0.004428263500017238 |
| G1/A3 | COMPLETE | 1892408 | 1000000 | 50.25728338351117 | 50.02975988845723 | 4879607.68 | 0.006153122500052177 |
| G1/D_ATTN4_FFN1 | COMPLETE | 1893912 | 1000000 | 51.78795336920579 | 51.49950644854813 | 6622662.464 | 0.015374341999972785 |
| G1/D_ATTN1_FFN4 | COMPLETE | 1893912 | 1000000 | 44.986926363057876 | 44.80975967873175 | 10342111.616 | 0.01013908349995063 |

## 原始 X4 的计算瓶颈

3×4 重复 Attention 与 FFN；解析拆分与实际训练 trace 分开记录。

## 哪些迭代可减少

{"status": "DEPTH_REDUCTION_NOT_SUPPORTED", "selected": null, "base_iterations": null, "checks": {"A1": {"ppl": false, "flops": true, "decode": true}, "A2": {"ppl": false, "flops": true, "decode": true}, "A3": {"ppl": false, "flops": true, "decode": true}}, "selection_data": "validation only; holdout not used for choosing architecture", "scientific_confirmation": false, "stage_complete": true}

## 减少迭代的质量损失

仅依据上表中同 Token/数据顺序的实测 PPL；未训练或不完整时无法判断。

## Persistent State 是否有独立价值

{"status": "NOT_EVALUATED"}

## Cellular 是否优于等规模 Flat

见 G2 的 B2−B1 机制指标；相同 State 元素数不等于相同更新网络。单 Seed 不确证组织优势。

## Cell Compute 是否有效替代 FFN

{"status": "NOT_EVALUATED"}

## 训练/推理成本是否降低

解析 FLOPs、Profiler 已知算子 FLOPs、真实 Tokens/s 与延迟分别报告；禁止用估算替代硬件测量。

## 是否质量退化

记录全部模型，包含失败、不完整和未获晋级者；原始完整曲线、采样文本见各模型 JSON。

## 哪些假设得到支持或否定

阶段阈值仅产生开发候选信号。研究假设未获多 Seed 确认，不将门禁通过解释为科学支持。

## 是否扩大训练

不自动执行 10M 或三 Seed 复核。只有已有阶段的完整质量/成本/机制证据满足阈值，才值得另行评估更大预算。

## 证据边界

Native 原 Phase-C Seed 91001 的完整原始输出未在本地找到。已定位独立 Controlled Retrain Seed 91002 的报告、固定外部资产与原始数据 Manifest；新 A0 始终是全新初始化，不冒充历史权重。
训练保持 Native 的 FP16 Autocast/GradScaler 与 FP32 参数；推理统一 FP32。溢出重试不提交 Token、Optimizer 或调度步。
语言样本保持原始 125 Token 连续数据流目标；每个独立样本 Reset，125 Token 内没有 Chunk Detach。其本身不证明跨 125 Token 的 State 价值。独立 M1–M4 使用超出配置 Attention 传播范围的延迟、未见随机组合与单独任务训练预算。
原始 Token 数据流不是逐文档数据集；验证/开发 Holdout 为固定、不重叠位置样本。真实独立文档使用 Session.prefill_document 默认 Reset，任务之间也 Reset。
State 更新的 16 个 Cell 在 Chunk 边界执行轻量共享写入；G3 仅 Top-2 执行主要 Cell FFN，其余 Cell 的路由/状态成本仍计入。
参数匹配、Token 匹配、解析 Compute 匹配与同硬件推理匹配互不替代。Compute 模式为独立 Job；不会在 Token 匹配 Job 中为更快模型增加训练量。
最终模型和 Profiler 原始 trace 保存在 Kaggle Output 的各模型 private/，Git 返回报告与完整 JSON 曲线；私有 Output 需要 Notebook 所有者权限。
运行时 Snapshot/Restore 是缓存与动态 State 生命周期测试，不是跨源码版本的训练 Checkpoint 恢复。

## 阶段与错误原始记录

```json
{
  "format": "minicells.epclm-x4c-001.v1",
  "source_commit": "bc1288d34748038455b2e9b441095821da6d5bc3",
  "protocol_sha256": "0958e1282974b06d95d09d732799cc0c7c0932d56a9177d9b51fc9e11b580496",
  "seed": 91001,
  "profile": "dev",
  "match": "tokens",
  "independent_test_accessed": false,
  "run_id": "epclm-x4c-001-bc1288d34748-b8cc5876",
  "implementation_sha256": "231bb733c978d4a16104c0766bcb144d5e553366abc247a0ecd5ce9775697327",
  "status": "STOPPED_AT_STAGE_DECISION",
  "training_status": "COMPLETE",
  "evaluation_status": "COMPLETE",
  "publication_status": "PENDING",
  "scientific_confirmation": false,
  "phases": {
    "G1": "COMPLETE",
    "G2": "NOT_RUN_G1_SELECTION_REQUIRED",
    "G3": "NOT_STARTED"
  },
  "dataset_manifest_sha256": "dfc7eeee33aaab5693a43bf72c88a991ee572b84cc93a5793e6901bf2c9cb6dc",
  "tokenizer_sha256": "becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2",
  "train_sha256": "3c8edf7a559e2e677be15ecc25eb5da90c19096bad82ab4e7a0c28b44b426c93",
  "validation_sha256": "4a8c15e6a53657488c78209e37fdd98e67e4e159875946cce1fc8663221f8d8f",
  "environment": {
    "gpu": "Tesla T4",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "training_precision": "float16_autocast_with_fp32_parameters_and_GradScaler"
  },
  "scope": "Completed phases only; blocked later phases and unperformed confirmation remain unproven"
}
```

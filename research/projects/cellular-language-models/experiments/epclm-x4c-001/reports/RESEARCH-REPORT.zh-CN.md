# EPCLM-X4C-001 研究报告

运行状态：IMPLEMENTATION_READY_NOT_RUN_RESOURCE_UNAVAILABLE。训练：NOT_RUN_RESOURCE_UNAVAILABLE；评估：NOT_RUN_RESOURCE_UNAVAILABLE。

单个开发 Seed。Independent Test 未访问；多 Seed 复核未执行。未执行项目不构成正面或负面研究结论。

| 模型 | 训练状态 | 参数 | 预测目标 | Validation PPL | Holdout PPL | 前向 FLOPs/Token（解析） | Batch-1 Decode Median |
|---|---|---:|---:|---:|---:|---:|---:|

## 原始 X4 的计算瓶颈

3×4 重复 Attention 与 FFN；解析拆分与实际训练 trace 分开记录。

## 哪些迭代可减少

{"status": "NOT_EVALUATED"}

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
  "status": "IMPLEMENTATION_READY_NOT_RUN_RESOURCE_UNAVAILABLE",
  "training_status": "NOT_RUN_RESOURCE_UNAVAILABLE",
  "evaluation_status": "NOT_RUN_RESOURCE_UNAVAILABLE",
  "phases": {
    "G0": "SOURCE_AND_CPU_CORRECTNESS_PASS; TARGET_GPU_NOT_RUN",
    "G1": "IMPLEMENTED_NOT_TRAINED",
    "G2": "IMPLEMENTED_CORRECTNESS_TESTED; BASE_SELECTION_PENDING_G1_MEASUREMENTS",
    "G3": "IMPLEMENTED_CORRECTNESS_TESTED; EXECUTION_REQUIRES_G1_G2_EVIDENCE"
  },
  "scientific_confirmation": false,
  "independent_test_accessed": false
}
```

## 已执行的源码计算核对

| 模型 | 参数量 | 解析前向 MFLOPs/Token（125 长度） |
|---|---:|---:|
| A0 | 1893912 | 13.109790 |
| A1 | 1892784 | 6.939919 |
| A2 | 1892220 | 3.854984 |
| A3 | 1892408 | 4.879608 |

以上为源码公式重算，不是实测 GPU 延迟。必要 CPU 测试 15 项通过，耗时 15.39 秒。当前 GPU 为 Quadro K620，不符合 Tesla T4 运行要求。六份冻结 Native 源码 Hash 全部与既有 Controlled Retrain 记录一致。

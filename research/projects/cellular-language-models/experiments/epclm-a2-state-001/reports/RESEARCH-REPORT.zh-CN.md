# EPCLM-A2-State-001 研究报告

状态：IMPLEMENTATION_READY_GPU_NOT_RUN。单 Seed 1M Development Pilot；不构成科学确认。

| 指标 | A2（历史） | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| Validation PPL | 51.3971830310497 | NOT_RUN | NOT_RUN | NOT_RUN |
| Parameters | 1892220 | NOT_RUN | NOT_RUN | NOT_RUN |
| Forward FLOPs/Token | 3854983.616 | NOT_RUN | NOT_RUN | NOT_RUN |
| Training Tokens/s | 见历史原始曲线 | NOT_RUN | NOT_RUN | NOT_RUN |
| Decode Median | 历史报告；未重新评估 | NOT_RUN | NOT_RUN | NOT_RUN |
| Peak VRAM allocated bytes | 见历史训练记录 | NOT_RUN | NOT_RUN | NOT_RUN |
| State Bytes | 0 | NOT_RUN | NOT_RUN | NOT_RUN |

## Baseline 与初始化

A2 仅引用指定历史 G1 的实测指标，不是本次重新评估。已定位配置、曲线、目标位置与最终权重身份；最终权重位于原 Kaggle Private Output，Git 结果中没有权重。
历史初始权重/初始 Hash 未归档：按原源码和 Seed 重建，并与独立 Native 构造数值 Hash 比较。三组内部基础权重完全相同，但不宣称已有独立历史初始 Hash 支撑的严格复现。

## B1 / B2 / B3

B1：既有四头 Attention 增加 State Memory Slots，FFN 无独立 State Read。B2：Token Attention 不接入 State，单头 State Context 仅进入原 n2 后的 FFN 输入。B3：两条路径读取同一 Chunk 初始 State。
Writer 分别只汇聚 Attention 输出、只汇聚 FFN 输出、二者等权均值；FP32 共享 GRU，在完整 Chunk 和序列结束的 29 Token 残块结束后更新。

## 机制判定

1. State 是否参与 Attention：查看 B1/B3 的 Memory Mass、Entropy、Read Disabled 与 Attention-only Read Disabled。
2. State 是否参与 FFN：查看 B2/B3 的 FFN Read 梯度、Read Disabled 和 FFN-only Read Disabled。
3. 哪条路径贡献更明显：结合完整验证 NLL 差异与相同冻结模型的路径消融，不按单个梯度大小排序。
4. 两条路径是否互补：B3 对比 B1/B2 及两条单独消融，不能预设 B3 最优。
5. 是否需要持久化：若 Reset Every Chunk 不退化，不声称跨 Chunk 保留有效。Write Disabled 与 Reset Every Chunk 可以数值相同，因为两者读取初始槽。
6. 额外成本是否合理：分别报告新增 State Read/Write FLOPs 与实测延迟，不把参数增加当作状态贡献。
7. 是否进入下一轮：仅有积极开发信号后才考虑三个独立 Seed，再考虑 10M；本次不自动执行。

## 科学结论

- H1: NOT_RUN；单 Seed，未科学确认。
- H2: NOT_RUN；单 Seed，未科学确认。
- H3: NOT_RUN；单 Seed，未科学确认。

完整训练/消融失败和未完成结果均保留。不同独立 125 Token 样本之间 Reset，不构成超过 128 Token 的长期记忆证据。

模型 Checkpoint 按本次 SPEC 第 16 节 Step 8 明确要求进入独立结果分支；没有历史权重热启动或 Optimizer Resume。decision.json 在 SHA 清单生成前冻结。最终发布验证写独立 Receipt，不改 decision。

```json
{
  "status": "IMPLEMENTATION_READY_GPU_NOT_RUN",
  "gpu_training": "NOT_RUN_RESOURCE_UNAVAILABLE",
  "local_gpu": "Quadro K620",
  "hypotheses": {
    "H1": "NOT_RUN",
    "H2": "NOT_RUN",
    "H3": "NOT_RUN"
  }
}
```

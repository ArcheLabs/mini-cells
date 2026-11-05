# EPCLM-A2-State-001 研究报告

状态：DEVELOPMENT_PILOT_COMPLETE。单 Seed 1M Development Pilot；不构成科学确认。

| 指标 | A2（历史） | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| Validation PPL | 51.3971830310497 | 51.053014685221974 | 51.34527075754049 | 50.85439790981691 |
| Parameters | 1892220 | 2040445 | 2040445 | 2040445 |
| Forward FLOPs/Token | 3854983.616 | 4257130.944 | 4681258.944 | 4717354.944 |
| Training Tokens/s | 见历史原始曲线 | 9530.593453146961 | 9546.719700103953 | 8262.055358011667 |
| Decode Median | 历史报告；未重新评估 | 0.005123484000023382 | 0.0050317010000071605 | 0.005632017500033726 |
| Peak VRAM allocated bytes | 见历史训练记录 | 124253184 | 127732224 | 131073024 |
| State Bytes | 0 | 8192 | 8192 | 8192 |

### B1 实测开发信号

训练 COMPLETE；评估 COMPLETE。
相对历史 A2 PPL 变化 -0.67%；2% 非劣筛选参考：范围内。历史对照并非此次重新评估，无多 Seed 确认证据。
read_disabled：状态 COMPLETE，NLL=4.536099523305893；相对正常 NLL 差值=0.6032349268595376。
write_disabled：状态 COMPLETE，NLL=3.934731215238571；相对正常 NLL 差值=0.001866618792215835。
reset_every_chunk：状态 COMPLETE，NLL=3.934731215238571；相对正常 NLL 差值=0.001866618792215835。
Reset 后出现开发集退化，存在持久化贡献的初步信号，仍需复核。

### B2 实测开发信号

训练 COMPLETE；评估 COMPLETE。
相对历史 A2 PPL 变化 -0.10%；2% 非劣筛选参考：范围内。历史对照并非此次重新评估，无多 Seed 确认证据。
read_disabled：状态 COMPLETE，NLL=3.9395901511112847；相对正常 NLL 差值=0.0010173171758651733。
write_disabled：状态 COMPLETE，NLL=3.9388491064310074；相对正常 NLL 差值=0.00027627249558781486。
reset_every_chunk：状态 COMPLETE，NLL=3.9388491064310074；相对正常 NLL 差值=0.00027627249558781486。
Reset 后出现开发集退化，存在持久化贡献的初步信号，仍需复核。

### B3 实测开发信号

训练 COMPLETE；评估 COMPLETE。
相对历史 A2 PPL 变化 -1.06%；2% 非劣筛选参考：范围内。历史对照并非此次重新评估，无多 Seed 确认证据。
read_disabled：状态 COMPLETE，NLL=4.526717921098073；相对正常 NLL 差值=0.5977513144413629。
write_disabled：状态 COMPLETE，NLL=3.931166728337606；相对正常 NLL 差值=0.0022001216808957835。
reset_every_chunk：状态 COMPLETE，NLL=3.931166728337606；相对正常 NLL 差值=0.0022001216808957835。
attention_read_disabled：状态 COMPLETE，NLL=4.524171431859334；相对正常 NLL 差值=0.5952048252026239。
ffn_read_disabled：状态 COMPLETE，NLL=3.9293159494797387；相对正常 NLL 差值=0.00034934282302856445。
Reset 后出现开发集退化，存在持久化贡献的初步信号，仍需复核。

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

- H1: INCONCLUSIVE；单 Seed，未科学确认。
- H2: INCONCLUSIVE；单 Seed，未科学确认。
- H3: INCONCLUSIVE；单 Seed，未科学确认。

完整训练/消融失败和未完成结果均保留。不同独立 125 Token 样本之间 Reset，不构成超过 128 Token 的长期记忆证据。

模型 Checkpoint 按本次 SPEC 第 16 节 Step 8 明确要求进入独立结果分支；没有历史权重热启动或 Optimizer Resume。decision.json 在 SHA 清单生成前冻结。最终发布验证写独立 Receipt，不改 decision。

```json
{
  "format": "minicells.epclm-a2-state-001.v1",
  "source_commit": "6b95b93a733d3789efa4d66f8235e03395fe6a91",
  "protocol_sha256": "dfcbe9794c60b37865224b5a20e5b42f7887b7abea5a2322ca746c69ae90f75c",
  "seed": 91001,
  "state_seed": 9100101,
  "run_id": "epclm-a2-state-001-6b95b93a733d-34a787e8",
  "profile": "dev",
  "match": "tokens",
  "independent_test_accessed": false,
  "status": "DEVELOPMENT_PILOT_COMPLETE",
  "training_status": "COMPLETE",
  "evaluation_status": "COMPLETE",
  "publication_status": "SEE_INDEPENDENT_RECEIPT",
  "scientific_confirmation": false,
  "hypotheses": {
    "H1": "INCONCLUSIVE",
    "H2": "INCONCLUSIVE",
    "H3": "INCONCLUSIVE"
  },
  "implementation_sha256": "8b0bf733446af7a2f6dd3e56539322e33c3e2f1a172ac2036f44a996c6e6422c",
  "dataset_manifest_sha256": "dfc7eeee33aaab5693a43bf72c88a991ee572b84cc93a5793e6901bf2c9cb6dc",
  "tokenizer_sha256": "becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2",
  "train_sha256": "3c8edf7a559e2e677be15ecc25eb5da90c19096bad82ab4e7a0c28b44b426c93",
  "validation_sha256": "4a8c15e6a53657488c78209e37fdd98e67e4e159875946cce1fc8663221f8d8f",
  "environment": {
    "gpu": "Tesla T4",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "precision": "FP16 autocast, FP32 parameters, FP32 GRU update"
  },
  "historical_environment_match": true
}
```

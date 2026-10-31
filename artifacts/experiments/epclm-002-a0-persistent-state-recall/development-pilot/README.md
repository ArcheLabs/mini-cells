# EPCLM-002-A0 开发期试运行

这些输出只用于检查代码和模型是否按预期运行，不是正式科学结论。使用的开发模型种子为 `98100`，评估生成种子为 `80100`；未使用正式模型种子 `98211–98213` 或正式评估种子 `98311–98313`。

| 后端 | 条件 | 评估查询 | Exact Match | 隔离/状态结果 |
|---|---|---:|---:|---|
| Direct Address | 8 facts、0 distractors | 128 | 128/128 | 状态与读者隔离审计通过；错误 Episode 状态 0/128；无状态读者 abstain |
| Learned Cells，100 steps | 8 facts、0 distractors | 64 | 0/64 | Reader 隔离审计通过；验证损失 0.9970；固定扫描 128 Cells |
| Learned Cells，500 steps | 8 facts、0 distractors | 32 | 0/32 | Reader 隔离审计通过；验证损失 0.7414；固定扫描 128 Cells |

Direct Address 仅验证显式 KV 工程参考与隔离协议，不能支持学习型状态主张。学习型试运行虽然验证损失下降，但精确回忆仍为 0；因此当前结果没有支持 H2。该开发样本很小，也不是三个正式种子、每条件 1,000 Episode 的正式测试，故 H2 的正式状态仍为 `INCONCLUSIVE`。该 Cell 候选每次都评分全部 128 Cells，不能支持 H3。

可复现文件：

- [Direct Address 条件结果](direct-address-8f-0d/result.json) 与 [逐 Episode 记录](direct-address-8f-0d/episodes.jsonl)
- [100-step Learned Cells checkpoint](learned-cells-100/checkpoint.pt)、[元数据](learned-cells-100/checkpoint.pt.json)、[条件结果](learned-cells-100/result.json)
- [500-step Learned Cells checkpoint](learned-cells-500/checkpoint.pt)、[元数据](learned-cells-500/checkpoint.pt.json)、[条件结果](learned-cells-500/result.json)

正式运行前不得根据这些试运行结果降低注册阈值。若修改模型结构或正式训练配置，必须先发布协议修订并重新计算协议哈希；正式评估种子仍保持未触碰。

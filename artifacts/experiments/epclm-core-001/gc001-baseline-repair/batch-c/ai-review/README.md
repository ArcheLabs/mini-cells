# Batch C：AI 辅助评分与研究分析附件

**状态：AI_REVIEW_COMPLETE_HUMAN_REVIEW_PENDING。** 这是一名 AI 评审者对 256 条已有匿名样本的六维评分，不是两名独立人工评审者的盲评。评审前已见模型汇总信息，不能声称严格独立盲评。用户无法自行复核，本附件不据此改变为“已人工复核”。

- [完整中文分析](analysis.zh-CN.md)
- [256 条逐样本评分 CSV](ai-review-scores.csv)
- [模型与解码模式汇总](ai-review-model-summary.json)
- [来源、方法与限制](ai-review-provenance.json)
- [文件 SHA256 清单](manifest.json)

来源为 [Batch C 原始不可变结果提交](https://github.com/ArcheLabs/mini-cells/tree/0df8a8f4ec9d9d921e55da2c94fba92af79555b5/artifacts/experiments/epclm-core-001/gc001-baseline-repair/batch-c)。已核验原发布清单中全部 17 份文件的字节数与 SHA256。

评分冻结后以精确提示和续写关联模型；相同提示、相同续写的评分已核验一致，无需使用私有盲评密钥。模型均值仅为单个 AI 的描述性评分，不含评审者间方差或确认性推断。

本附件保留原结果 `ANALYSIS_COMPLETE_REVIEW_PENDING` 和 H1–H5 `NOT_EVALUATED`，不覆盖原报告，不启动训练，也不增加训练预算。请勿将 AI 评分文件改名导入 `human-review-scored.csv`。

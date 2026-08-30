# CLM-0.4-mini M1-v2 — Stage 05

执行顺序：

1. 用固定 TinyStories revision 生成 v2 数据；
2. 将 `asset-summary.json` 的 hash 锁入 `asset-lock.json`；
3. 合并 data-lock commit；
4. 再用新 development seed `90402` 执行 calibration；
5. 生成报告；
6. 使用 `scripts/research/publish.py` 将 curated result 推送到独立 Kaggle result branch；
7. 只有 calibration PASS 且 protocol-lock 被单独提交后，才允许 formal seeds。

v1 的 `90401` 失败不会被删除或复用。Dense baseline 只用于比较，不参与 CLM 科学判定。

# CLM-0.4-mini M1-v2 语言验证

状态：**PROTOCOL_FROZEN_DATA_LOCK_PENDING**

M1-v2 是在 v1 开发种子 `90401` 得到
`CALIBRATION_BASE_PREREQUISITES_FAILED` 后进行的显式协议修订。v1 结果保存在
`v1-development-failure.json`，不会被覆盖，也不属于 formal scientific result。

## v2 修订

- Story base 数据改为与评估同构的 `Context → Question → Answer` 检索 QA；
- Math base 四个家族统一为 QA 形式，但不提前训练 M1 continual curriculum 的新家族；
- base primary gate 改为 teacher-forced answer exact，阈值仍为 `0.85`；
- greedy exact、answer-token accuracy、answer NLL 仅作诊断；
- 每个 domain 使用 64 个 held-out route-balanced address，覆盖 L3/L4 全部 32 Cells；
- 新 development seed 为 `90402`；`90401` 只属于 v1；
- formal seeds 仍为 `90411/90412/90413`；
- 增加 equal-parameter dense 与 equal-active-compute dense 对照；
- dense baseline 永远不能控制 CLM commit、growth、candidate selection 或 formal decision。

## 数据锁

当前 `asset-lock.json` 故意处于 `DATA_LOCK_PENDING`。必须先在 Kaggle 生成 v2
的 30M token 数据，取得新的 tokenizer/base-corpus/curriculum hashes，并把它们
提交到仓库后，runner 才允许打开 `90402`。

## 结果回传

训练/校准结束后：

```bash
python scripts/research/publish.py clm-0.4-mini-m1-v2-calibration \
  --results /path/to/results \
  --push
```

只会提交 JSON/JSONL/CSV/MD/PNG 等分析证据，不会把 checkpoint 或 30M token
原始数据推入 Git。

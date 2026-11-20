# Phase 3A v2 — CPU development smoke

Formal decision: **NOT TESTED**.

The smoke uses verified Qwen3-0.6B-Base weights, frozen CPU FP32 SDPA,
and offset-mapped entity, relation and value spans from block index 22.
The write value follows the relation anchor. The query and key encoders
are independent linear projections. The value encoder learns across bare
and contextual views. Only the adapter was optimized. The formal test
was not generated or read.

Seed 42 uses 32 train and 32 validation facts. Eight adapter AdamW updates
lowered combined contrastive training loss from 6.904 to 4.658. The adapter
Safetensors checkpoint was strictly reloaded with exact projected outputs.
Validation Recall@1 was 43.75% seen, 25% held query, 15.625% held both,
and 15.625% value.

```sh
python phase3a_v2.py /path/to/verified/snapshot /path/to/new-output
python -m pytest -q test_phase3a_v2.py
```

The smoke has one view per condition and uses in-batch negatives. The
formal template/view matrix, explicit hard-negative sampling, shared and
MLP controls, pooling ablations, validation-based early stopping and
rejection thresholds, 128/512-way evaluations, five-seed statistics and
Kaggle result publication remain outstanding.

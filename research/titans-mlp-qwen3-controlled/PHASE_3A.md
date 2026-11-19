# Phase 3A — Semantic Addressing

Status: **NOT TESTED at formal scale**. The CPU Smoke used seed 42 and 32
facts per split. It used the fixed Qwen snapshot in FP32 and evaluated
last-token and mean pooling without training the Backbone or Titans.
The linear Query-to-Write ridge diagnostic fit train only and selected
regularization on validation only.

| Last-token 32-way | Raw Recall@1 | Probe Recall@1 | Random Recall@1 |
| --- | ---: | ---: | ---: |
| Seen | 12.50% | 28.13% | 6.25% |
| Held-out query | 3.13% | 6.25% | 6.25% |
| Held-out write and query | 3.13% | 3.13% | 6.25% |

The oracle returned 100%. These smoke metrics do not open the next gate.
The raw artifact includes all ranks, cosine and margin summaries,
value-context comparison, negative query scores and snapshot hashes.

Run the full five-seed matrix and frozen thresholds:

```sh
python phase3a.py /path/to/verified/snapshot /path/to/full-result
python summarize_phase3a.py /path/to/full-result/results.json
```

Formal mode uses 1024/256/512 train/validation/test facts per seed.
The primary gate uses last-token scores, with mean pooling and held-out
write/query pairs reported separately.

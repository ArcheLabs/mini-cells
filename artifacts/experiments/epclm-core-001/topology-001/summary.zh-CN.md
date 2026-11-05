# Topology-001 开发筛选

| Arm | Training | Tokens | Updates | Validation PPL | Holdout PPL | Tokens/s |
|---|---|---:|---:|---:|---:|---:|
| R1 | COMPLETE | 250000 | 977 | 325.55188056848345 | 343.66393250191965 | 389.80647282582487 |
| IC | COMPLETE | 250000 | 977 | 326.50607250265097 | 343.8801379613052 | 443.5535369872042 |
| R2 | COMPLETE | 250000 | 977 | 326.63768491519784 | 343.64383055031146 | 377.1428950522478 |
| SR | COMPLETE | 250000 | 977 | 326.6687732276179 | 344.25344416159294 | 376.2766980862835 |
| T4 | COMPLETE | 250000 | 977 | 326.64241398890243 | 344.41731820217956 | 376.7153003142884 |

```json
{
  "format": "minicells.epclm.topology-001.v1",
  "experiment": "EPCLM-CORE-001-TOPOLOGY-001",
  "run_id": "topology-001-3e4270973141-5a604561",
  "source_commit": "3e4270973141f0cd69a96a0c8e52f4fc2d90b26f",
  "status": "DEVELOPMENT_SCREEN_COMPLETE",
  "training_status": "COMPLETE",
  "evaluation_status": "COMPLETE",
  "propagation_status": "COMPLETE",
  "compute_status": "COMPLETE",
  "publication_status": "PENDING",
  "research_review_status": "PENDING",
  "scientific_confirmation": false,
  "independent_test_accessed": false,
  "result_branch": "kaggle/epclm-core-001-topology-001-3e4270973141",
  "environment": {
    "gpu": "Tesla T4",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "precision": "fp32",
    "tf32": false
  },
  "screen": {
    "status": "NO_CLEAR_TOPOLOGY_WINNER",
    "selected": null,
    "checks": {
      "R2": {
        "status": "EVALUATED",
        "criteria": {
          "holdout_improvement": false,
          "paired_ci": false,
          "validation": true,
          "causal_propagation": true,
          "latency": true
        },
        "eligible": false
      },
      "SR": {
        "status": "EVALUATED",
        "criteria": {
          "holdout_improvement": false,
          "paired_ci": false,
          "validation": true,
          "causal_propagation": true,
          "latency": true
        },
        "eligible": false
      },
      "T4": {
        "status": "EVALUATED",
        "criteria": {
          "holdout_improvement": false,
          "paired_ci": false,
          "validation": true,
          "causal_propagation": true,
          "latency": true
        },
        "eligible": false
      }
    },
    "rules": {
      "minimum_holdout_ppl_improvement": 0.01,
      "delta_nll_ci_upper_max": 0.0,
      "maximum_validation_ppl_ratio": 1.01,
      "maximum_decode_latency_ratio": 2.0,
      "threshold_note": "operational definitions for unspecified obvious validation degradation and reasonable latency; development screen only"
    },
    "scientific_confirmation": false,
    "next_phase_automatic": false,
    "recommendation": "multi-seed follow-up only after review; no automatic training"
  }
}
```

单 Seed 文档 Bootstrap 不估计跨 Seed 方差。传播不等于预测收益。最终模型仅在各组记录的 Kaggle Output，不在 Git。未自动进入下一阶段。

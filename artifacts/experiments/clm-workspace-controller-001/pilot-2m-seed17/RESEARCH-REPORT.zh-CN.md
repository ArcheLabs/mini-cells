# CLM v0.4 双 T4 2M Pilot

{
  "format": "minicells.clm-workspace-controller-001.v04",
  "run_id": "clm-v04-pilot-2m-seed17-7c94a3724475-edc54d46",
  "experiment": "clm-workspace-controller-001-pilot-2m-seed17",
  "status": "PILOT_BLOCKED",
  "source_commit": "7c94a37244758c890184d43d96f3a0adfb1a4337",
  "seed": 17,
  "targets_per_model": 2000000,
  "total_targets": 12000000,
  "main_training_started": false,
  "pilot_training_started": false,
  "scientific_confirmation": false,
  "hypotheses": {
    "H1": "NOT_EVALUATED",
    "H2": "NOT_EVALUATED",
    "H3": "NOT_EVALUATED",
    "H4": "NOT_EVALUATED"
  },
  "recovery_validation": "REMOVED_BY_USER; existing infrastructure reused",
  "selected_precision": "fp32",
  "precision_fallback": {
    "from": "fp16",
    "to": "fp32",
    "reason": "GPU cached/full/decode correctness failed; upstream AMP error can still change hard Top-2 after FP32 Router"
  },
  "error": "RuntimeError: cannot freeze exact target budget for benchmark",
  "blocking_gate": "ENGINEERING_GATE_BLOCKED",
  "elapsed_before_publication": 168.41322934200002,
  "wall_clock_limit_seconds": 7200,
  "cumulative_pilot_active_seconds": 168.41322934200002
}

| Model | Committed Targets | Training | Validation NLL |
|---|---:|---|---:|
| T0-P | 0 | NOT_RUN | NOT_RUN |
| T0-F | 0 | NOT_RUN | NOT_RUN |
| C0 | 0 | NOT_RUN | NOT_RUN |
| C1 | 0 | NOT_RUN | NOT_RUN |
| C2 | 0 | NOT_RUN | NOT_RUN |
| C3 | 0 | NOT_RUN | NOT_RUN |

本轮只包含六模型、Seed17、每模型2M；缺失诊断为PENDING，不视为门禁通过。不执行32M×三Seed，不宣称单Seed统计显著。跨Kaggle恢复验证按用户要求删除。

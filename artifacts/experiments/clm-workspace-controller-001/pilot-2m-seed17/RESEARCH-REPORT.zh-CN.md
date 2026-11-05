# CLM v0.4 双 T4 2M Pilot

{
  "format": "minicells.clm-workspace-controller-001.v04",
  "run_id": "clm-v04-pilot-2m-seed17-c114ac0a548b-e4e99a54",
  "experiment": "clm-workspace-controller-001-pilot-2m-seed17",
  "status": "PILOT_BLOCKED",
  "source_commit": "c114ac0a548b41ce709b099c01eb4a4fdb78fa1c",
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
  "error": "RuntimeError: engineering GPU gate failed or blocked",
  "blocking_gate": "ENGINEERING_GATE_BLOCKED",
  "elapsed_before_publication": 139.36311571799996,
  "wall_clock_limit_seconds": 7200,
  "cumulative_pilot_active_seconds": 139.36311571799996
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

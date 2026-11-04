# CLM v0.4 双 T4 2M Pilot

{
  "format": "minicells.clm-workspace-controller-001.v04",
  "run_id": "clm-v04-pilot-2m-seed17-cf777887532a-25d18724",
  "experiment": "clm-workspace-controller-001-pilot-2m-seed17",
  "status": "PILOT_PASS",
  "source_commit": "cf777887532a021826afb05f063addc651af5570",
  "seed": 17,
  "targets_per_model": 2000000,
  "total_targets": 12000000,
  "main_training_started": false,
  "pilot_training_started": true,
  "scientific_confirmation": false,
  "hypotheses": {
    "H1": "INCONCLUSIVE",
    "H2": "INCONCLUSIVE",
    "H3": "INCONCLUSIVE",
    "H4": "INCONCLUSIVE"
  },
  "recovery_validation": "REMOVED_BY_USER; existing infrastructure reused",
  "selected_precision": "fp32",
  "precision_fallback": null,
  "committed_targets": 12000000,
  "elapsed_before_publication": 2233.7983470679997,
  "wall_clock_limit_seconds": 7200,
  "cumulative_pilot_active_seconds": 2233.7983470679997
}

| Model | Committed Targets | Training | Validation NLL |
|---|---:|---|---:|
| T0-P | 2000000 | COMPLETE | 3.140125461512851 |
| T0-F | 2000000 | COMPLETE | 3.1727486848632394 |
| C0 | 2000000 | COMPLETE | 3.205274489793814 |
| C1 | 2000000 | COMPLETE | 3.199339233161172 |
| C2 | 2000000 | COMPLETE | 3.2031325787449947 |
| C3 | 2000000 | COMPLETE | 3.2081840842172107 |

本轮只包含六模型、Seed17、每模型2M；缺失诊断为PENDING，不视为门禁通过。不执行32M×三Seed，不宣称单Seed统计显著。跨Kaggle恢复验证按用户要求删除。

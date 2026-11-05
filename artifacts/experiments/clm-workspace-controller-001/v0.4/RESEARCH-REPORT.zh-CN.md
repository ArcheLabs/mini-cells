# CLM v0.4 Smoke Report

{
  "format": "minicells.clm-workspace-controller-001.v04",
  "run_id": "clm-v04-smoke-f50509786a9b-fb1381bd",
  "status": "SMOKE_COMPLETE_MAIN_NOT_FROZEN",
  "phase": "SMOKE",
  "main_training": "NOT_RUN_NOT_FROZEN",
  "scientific_confirmation": false,
  "hypotheses": {
    "H1": "NOT_EVALUATED",
    "H2": "NOT_EVALUATED",
    "H3": "NOT_EVALUATED",
    "H4": "NOT_EVALUATED"
  },
  "source_commit": "f50509786a9bdc9c408577b69e9da9d8ea255256",
  "independent_test_accessed": false,
  "publication_status": "SEE_INDEPENDENT_RECEIPT",
  "elapsed_seconds_before_publication": 220.322162826,
  "models": {
    "C0": {
      "training": "COMPLETE",
      "smoke_gate": "PASS",
      "validation": "COMPLETE"
    },
    "C1": {
      "training": "COMPLETE",
      "smoke_gate": "PASS",
      "validation": "COMPLETE"
    },
    "C2": {
      "training": "COMPLETE",
      "smoke_gate": "PASS",
      "validation": "COMPLETE"
    },
    "C3": {
      "training": "COMPLETE",
      "smoke_gate": "PASS",
      "validation": "COMPLETE"
    },
    "T0-P": {
      "training": "COMPLETE",
      "smoke_gate": "PASS",
      "validation": "COMPLETE"
    },
    "T0-F": {
      "training": "COMPLETE",
      "smoke_gate": "PASS",
      "validation": "COMPLETE"
    }
  }
}

| Model | Targets | Smoke Gate | NLL | PPL |
|---|---:|---|---:|---:|
| C0 | 31721 | PASS | 6.692012785600414 | 805.9428107626011 |
| C1 | 31721 | PASS | 6.699127403431293 | 811.697231835121 |
| C2 | 31721 | PASS | 6.697986162625826 | 810.7714182210876 |
| C3 | 31721 | PASS | 6.698196570500139 | 810.9420288600309 |
| T0-P | 31721 | PASS | 6.516135633149894 | 675.9611697195569 |
| T0-F | 31721 | PASS | 6.602051067099331 | 736.6044660844294 |

仅为17 Seed短训练和机制检查，不是三Seed正式结果；H1–H4均未评估。32M预算、固定训练FLOPs比较、条件性参数匹配复核和Oracle诊断尚未执行，不作架构有效性判断。
完整研究报告需主训练及三Seed完成后另行生成。

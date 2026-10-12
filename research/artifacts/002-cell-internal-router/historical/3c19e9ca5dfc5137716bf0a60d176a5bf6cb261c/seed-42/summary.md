# Research run summary

Status: **DEVELOPMENT_NOT_SUPPORTED**

| Model | Tokens | Final PPL | Matmul FLOPs/token (scope) | Latency ms |
|---|---:|---:|---:|---:|
| fixed-cell | 10000000 | 12.289763510450271 | 123170816 (estimate) | 20.300971000028767 |
| global-router | 10000000 | 12.34814312168731 | 47722496 (estimate) | 20.42686724997793 |
| cell-internal | 10000000 | 16.010517598174484 | 52622897.63200001 (estimate) | 147.85528824995708 |

Frozen decision: `{"status": "DEVELOPMENT_NOT_SUPPORTED", "scientific_decision": false, "comparisons": {"hard_to_global_ppl_ratio": 1.2965931347244322, "hard_to_global_flops_ratio": 1.1026853589552401, "hard_to_global_latency_ratio": 7.238275279344013, "hard_mean_sequential_depth": 2.384197916666667, "sample_path_count": 74, "noncollapse": true, "minimum": false, "medium": false, "strong": false}}`

Derived view only; unavailable measurements remain unavailable. Training PPL and validation PPL are distinct.
GEMM FLOPs exclude non-matmul costs; latency scope follows the source evidence.
Soft depth is expected probability mass; hard depth is actual executed visits.
Single development seeds do not establish general CLM/MoE superiority.
Unavailable plots: loss, throughput, memory, routing_entropy

## Input and execution provenance
```json
{
  "experiment_id": "002-cell-internal-routing",
  "seed": 42,
  "source_commit": "3c19e9ca5dfc5137716bf0a60d176a5bf6cb261c",
  "protocol_sha256": "8a1a425a98bce99b6e92620cd782cbd302ee73600b3de726a622d53246040993",
  "source_config_sha256": "a00fd4a4eb90d5a8bdbfcb2784d7634616e8a89931769bdb9cde5ec73daa11bd",
  "started_at_unix": 1791767073.533643,
  "python": "3.12.13",
  "torch": "2.10.0+cu128",
  "dependencies": {
    "numpy": "2.0.2",
    "datasets": "5.0.0",
    "tokenizers": "0.22.2"
  },
  "capacities": {
    "fixed-cell": 60894720,
    "global-router": 60919296,
    "cell-internal": 61115904
  },
  "execution": "one seed/session; independent arms queued on two GPUs, not DDP",
  "precision": "FP16 with GradScaler",
  "scientific_decision": false,
  "gpus": [
    "Tesla T4",
    "Tesla T4"
  ],
  "status": "completed",
  "elapsed_seconds": 1555.0343770980835
}
```

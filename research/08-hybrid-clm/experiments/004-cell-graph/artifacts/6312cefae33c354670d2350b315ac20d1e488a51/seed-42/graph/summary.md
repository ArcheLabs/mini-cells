# Research run summary

Status: **DEVELOPMENT_SESSION_COMPLETED**

| Model | Tokens | Final PPL | Matmul FLOPs/token (scope) | Latency ms |
|---|---:|---:|---:|---:|
| graph-cell | 10000000 | 13.070590205256561 | 55557695.072 (measured) | 78.0375972499769 |
| hard-cell | 10000000 | 14.824803405492208 | 43833797.952 (measured) | 65.75463105004928 |

Frozen decision: `{"status": "DEVELOPMENT_SESSION_COMPLETED", "scientific_decision": false, "gates": null}`

Derived view only; unavailable measurements remain unavailable. Training PPL and validation PPL are distinct.
GEMM FLOPs exclude non-matmul costs; latency scope follows the source evidence.
Soft depth is expected probability mass; hard depth is actual executed visits.
Single development seeds do not establish general CLM/MoE superiority.
Unavailable plots: none

## Input and execution provenance
```json
{
  "schema_version": 2,
  "experiment": "004-cell-graph",
  "source_commit": "6312cefae33c354670d2350b315ac20d1e488a51",
  "seed": 42,
  "session": "graph",
  "scientific_decision": false,
  "capacities": {
    "fixed-cell": 60894720,
    "global-router": 60919296,
    "soft-cell": 61115904,
    "hard-cell": 61115904,
    "graph-cell": 61116336
  },
  "config_sha256": "c4cfd632485795260834c6bfb577d9e6ad02927a24d544dd44565511b15808f2",
  "protocol_sha256": "f49229ee09f5967cb95ab2a7de61479c56bfb540ac47deb3d75ab06465aa2a57",
  "definition_sha256": "50d1e0385ea7e856c98cbc7eaefeb13c3ca732ca8f4f385d943375d61d426f28",
  "assets_manifest_sha256": "0462d013cc9d90335777febde41722a462d94c0bcc0bfe3635e32277f00bc27f",
  "torch": "2.10.0+cu128",
  "python": "3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]",
  "dependencies": {
    "numpy": "2.0.2",
    "matplotlib": "3.10.0"
  },
  "execution": "two independent T4 workers; no DDP; offline mounted Dataset",
  "attention_backend": "SDPA math in every arm and measurement",
  "status": "DEVELOPMENT_SESSION_COMPLETED",
  "execution_policy": "manual-token-budget",
  "gpus": [
    "Tesla T4",
    "Tesla T4"
  ],
  "elapsed_seconds": 1659.874531507492
}
```

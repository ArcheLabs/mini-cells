# CLM-001-R3b — Posthoc performance

Training run directory: `33734fdd1558`.

Training end-to-end wall time: **NOT_MEASURED**. Two-hour verdict: **NOT_VERIFIED**.

CUDA resource time and serial HF costs are not end-to-end wall time.

| Arm | Steps (measured) | CUDA seconds (measured) | Update P50 / P99 seconds (measured) |
| --- | ---: | ---: | --- |
| Q-R3b | 500 | 1384.9299638671876 | 2.729327752000131 / 3.1446000248502513 |
| QC-R3b | 500 | 1472.3996115722657 | 2.901038840999945 / 3.3431646469501786 |
| QS-R3b | 500 | 1379.131448486328 | 2.7132864595000683 / 3.1448823792798475 |

## Recorded storage and resource costs

| Quantity | Value | Classification | Source |
| --- | ---: | --- | --- |
| Training CUDA resource seconds | 4236.461023925782 | DERIVED | resource-ledger.json/training/*/gpu_training_seconds |
| Posthoc CUDA resource seconds | 1575.0231879882813 | DERIVED | coverage-ledger.json/tasks/gpu_stream_seconds |
| Posthoc session wall seconds | None | NOT_MEASURED | private HF coverage/sessions |
| seconds | 1318.7243573899996 | DERIVED | resource-ledger.json/storage/uploads |
| commit_seconds | 116.58928657899929 | DERIVED | resource-ledger.json/storage/uploads |
| readback_seconds | 137.6983179709996 | DERIVED | resource-ledger.json/storage/uploads |
| new_content_object_bytes | 1264190064 | DERIVED | resource-ledger.json/storage/uploads |
| payload_bytes_submitted | 1269151967 | DERIVED | resource-ledger.json/storage/uploads |
| final_logical_bundle_bytes | 1284580587 | MEASURED | resource-ledger.json/storage/uploads/-1 |
| final_full_readback_seconds | 124.5893255810006 | MEASURED | resource-ledger.json/storage/uploads/-1 |

Coverage: `{'complete': 42, 'unavailable': 0, 'blocked': 60, 'pending': 0}`.

Mandatory event totals and ACK-only time: NOT_RECORDED. Oracle time is only a component.
Checkpoint serialization values describe the final checkpoint, not all saves.
All values, classifications, timing groups and source paths: `performance-summary.json`.
Missing analysis observations are not scientific failures. Confirmation remains sealed.
No training update is performed by this pipeline.

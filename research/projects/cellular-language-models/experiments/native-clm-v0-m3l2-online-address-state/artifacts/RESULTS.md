# Native CLM v0 M3L-2 — Online Historical Address-State Integration

- Status: `NATIVE_CLM_V0_M3L2_ONLINE_ADDRESS_STATE_NOT_SUPPORTED`
- Scientific decision: `False`
- Protocol SHA-256: `b5f95f9a4d920577d5ff0ddbaf120631f7ed0faf04e1c5ba39492e28d3b20adb`
- Data manifest SHA-256: `ed0b50c8bf3a822bf13c1a542a78cb9ad4b15daa09c612c4c1cfff782f30f409`
- Formal seeds: `[74211, 74212, 74213]`
- Learner replay after continual start: `0 bytes`

| seed | control A reg | treatment A reg | advantage | treatment forgetting | result |
|---:|---:|---:|---:|---:|---|
| 74211 | 0.4782 | 0.4250 | 0.0532 | 0.1945 | FAIL |
| 74212 | 0.4737 | 0.4214 | 0.0523 | 0.1975 | FAIL |
| 74213 | 0.4702 | 0.4216 | 0.0487 | 0.1948 | FAIL |

Boundary: the A bootstrap is explicitly pre-continual and sidecar-only; its one-shot learner handle is released before B begins. No bootstrap token/query is available to the B→C→D learner.

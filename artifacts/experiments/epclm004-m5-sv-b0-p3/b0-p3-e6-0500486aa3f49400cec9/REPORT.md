# B0-P3 E6

Stage1 gate: BLOCKED. Stage2 started: False. Single-seed development;no H1–H6 confirmation.

| Stage | Active EM | Active NLL | Actual random EM | Actual random NLL | Value EM | Full EM |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.1719 | 2.040750 | 0.1275 | 2.060944 | 0.0039 | 0.0000 |

## Recorded training calls

| Stage | Updates | Logical slots | Valid processed tokens | Targets | Fit wall this call, s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 65536 | 60416 | 8192 | 78.93 | 709.6865234375 |

Fit time/peak cover the completing call; cumulative subprocess charges include interrupted calls and are the time-budget authority.
An individual passing arm does not enter Stage2 when another arm blocks the cohort.

Gate checks: {"status": "BLOCKED", "checks": {"validation_accuracy": false, "ci_above_actual_random": true, "nll_advantage": false, "all_stage_online_accuracy": false, "no_severe_active_collapse": true, "numerical_status": true}, "online_scope": "all8192 Stage1 targets before each optimizer update", "checkpoint_sha256": "63a33ae750d3a0a83b03a1b906924ce9e6673d6e39ecfd4a26508aec642b416c"}

Query-isolated diagnostic: {"1": {"status": "COMPLETED"}}

Primary evaluation teacher-forces earlier gold answers; current targets are strictly causal next tokens. Query-isolated evaluation changes answer history, position and chunk/Store availability, so its difference does not isolate a causal answer-history effect.
Only active-candidate loss is optimized. Full/value output collapse is diagnostic, not a Stage1 veto. Actual per-Episode candidate random baselines are used. Empirical cluster intervals can collapse at floor/ceiling and do not include across-seed variance.
All valid59/115 tokens are retained; only trailing5/13 PAD slots are omitted. Logical slot counts64/128 are distinct from processed token counts. E7 stores raw token representations; selected fact-position co-hits do not prove semantic binding retrieval or causal benefit.
Test is SEALED_NOT_ACCESSED;Train/Test mapping intersection is not inspected. No12M,32–64-pair scaling,main multi-seed training or M6.

Independent verification: {"download": {"status": "PASS", "sha256": "07d0b9213c0af8d60cb66956470725eee94f42fb11c4e7a1191b3421bd8c3f66", "bytes": 184019915, "files_verified": 69, "scope": "All archived files, including checkpoint/probabilities/metrics/report; no trusted status substitution"}, "payload": {"status": "PASS", "stages": [1], "global_steps": 128, "test_comparison": "NOT_CHECKED_TEST_SEALED"}, "restore": {"status": "PASS", "stages": {"1": {"status": "PASS", "checkpoint_sha256": "63a33ae750d3a0a83b03a1b906924ce9e6673d6e39ecfd4a26508aec642b416c", "probability_log_max_abs_error": {"full_log_probabilities": 0.0, "value_log_probabilities": 0.0, "candidate_log_probabilities": 0.0}, "tolerance": 2e-05}}, "new_optimizer_updates": 0, "test_accessed": false}, "stage1_gate": "BLOCKED", "stage2_started": false, "charged_worker_phases": [{"action": "benchmark1", "arm": "E6", "started_epoch": 1794323836.1732032, "elapsed_seconds": 17.816179818000023, "status": "COMPLETE"}, {"action": "benchmark2", "arm": "E6", "started_epoch": 1794323853.9907691, "elapsed_seconds": 26.840268616000003, "status": "COMPLETE"}, {"action": "stage1", "arm": "E6", "started_epoch": 1794323987.8889155, "elapsed_seconds": 89.01736208300002, "status": "COMPLETE"}, {"action": "isolated1", "arm": "E6", "started_epoch": 1794324225.105946, "elapsed_seconds": 8.589189781999949, "status": "COMPLETE"}, {"action": "verify", "arm": "E6", "started_epoch": 1794324576.4578164, "elapsed_seconds": 17.518584838000038, "status": "COMPLETE"}], "independent_worker_log": ""}

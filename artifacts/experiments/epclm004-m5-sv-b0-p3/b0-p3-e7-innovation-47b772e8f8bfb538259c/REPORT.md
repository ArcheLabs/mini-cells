# B0-P3 E7-Innovation

Stage1 gate: BLOCKED. Stage2 started: False. Single-seed development;no H1–H6 confirmation.

| Stage | Active EM | Active NLL | Actual random EM | Actual random NLL | Value EM | Full EM |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.1543 | 2.042384 | 0.1275 | 2.060944 | 0.0024 | 0.0000 |

## Recorded training calls

| Stage | Updates | Logical slots | Valid processed tokens | Targets | Fit wall this call, s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 65536 | 60416 | 8192 | 115.72 | 744.5205078125 |

Fit time/peak cover the completing call; cumulative subprocess charges include interrupted calls and are the time-budget authority.
An individual passing arm does not enter Stage2 when another arm blocks the cohort.

Gate checks: {"status": "BLOCKED", "checks": {"validation_accuracy": false, "ci_above_actual_random": true, "nll_advantage": false, "all_stage_online_accuracy": false, "no_severe_active_collapse": true, "numerical_status": true}, "online_scope": "all8192 Stage1 targets before each optimizer update", "checkpoint_sha256": "0cad1365830a638587e365e1064fe844c60d6793341102076de22d8757ee2efe"}

Query-isolated diagnostic: {"1": {"status": "COMPLETED"}}

Primary evaluation teacher-forces earlier gold answers; current targets are strictly causal next tokens. Query-isolated evaluation changes answer history, position and chunk/Store availability, so its difference does not isolate a causal answer-history effect.
Only active-candidate loss is optimized. Full/value output collapse is diagnostic, not a Stage1 veto. Actual per-Episode candidate random baselines are used. Empirical cluster intervals can collapse at floor/ceiling and do not include across-seed variance.
All valid59/115 tokens are retained; only trailing5/13 PAD slots are omitted. Logical slot counts64/128 are distinct from processed token counts. E7 stores raw token representations; selected fact-position co-hits do not prove semantic binding retrieval or causal benefit.
Test is SEALED_NOT_ACCESSED;Train/Test mapping intersection is not inspected. No12M,32–64-pair scaling,main multi-seed training or M6.

Independent verification: {"download": {"status": "PASS", "sha256": "5a687d58424817d7b7a7789e33d2c6c4b5ea94a53f7225fd3303137daef15e0c", "bytes": 187122178, "files_verified": 69, "scope": "All archived files, including checkpoint/probabilities/metrics/report; no trusted status substitution"}, "payload": {"status": "PASS", "stages": [1], "global_steps": 128, "test_comparison": "NOT_CHECKED_TEST_SEALED"}, "restore": {"status": "PASS", "stages": {"1": {"status": "PASS", "checkpoint_sha256": "0cad1365830a638587e365e1064fe844c60d6793341102076de22d8757ee2efe", "probability_log_max_abs_error": {"full_log_probabilities": 0.0, "value_log_probabilities": 0.0, "candidate_log_probabilities": 0.0}, "tolerance": 2e-05}}, "new_optimizer_updates": 0, "test_accessed": false}, "stage1_gate": "BLOCKED", "stage2_started": false, "charged_worker_phases": [{"action": "benchmark1", "arm": "E7-Innovation", "started_epoch": 1794323880.8323493, "elapsed_seconds": 22.572704338999984, "status": "COMPLETE"}, {"action": "benchmark2", "arm": "E7-Innovation", "started_epoch": 1794323903.4066272, "elapsed_seconds": 37.268293137, "status": "COMPLETE"}, {"action": "stage1", "arm": "E7-Innovation", "started_epoch": 1794324077.2011266, "elapsed_seconds": 129.27383541899997, "status": "COMPLETE"}, {"action": "isolated1", "arm": "E7-Innovation", "started_epoch": 1794324233.6963177, "elapsed_seconds": 9.388494541, "status": "COMPLETE"}, {"action": "verify", "arm": "E7-Innovation", "started_epoch": 1794324771.2895024, "elapsed_seconds": 20.87197566399982, "status": "COMPLETE"}], "independent_worker_log": ""}

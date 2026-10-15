# Single-launch workflow repair

The user reported a missing `pilot/pilot.json` after a new Kaggle cloud session. The staged launcher incorrectly
assumed a previous session's ephemeral `/working` files remained present. No scientific training had started.

The user's later instruction replaces the proposal's manual stages for normal development:

1. Run CPU functional checks and validate T4 hardware.
2. Prepare/check independent development assets.
3. Perform a short same-GPU check at the unchanged training microbatch, or reuse matching recorded identity.
4. Record source/config/Pilot identity internally and train S16/S8, then M0 on the first free device.
5. Join all workers before serial evaluation/interventions/report and portable export.

No mode switch, external Pilot file, manual hash/seed/approval form or runtime-confirmation switch is required.
Estimated runtime is advisory. Failures still stop execution; confirmation remains sealed, and completed
checkpoints cannot be overwritten. No wall-time kill or training-budget reduction is installed.

Validation: 118 CPU tests passed (35 FAST, 83 PD regression), including automatic prerequisite ordering,
failed-preflight prevention of GPU/training work, and reuse of an existing matching config/checkpoints.
No real GPU job, TinyStories training or confirmation read was launched during this repair.
Original model geometry, J0 objective, fixed 8/16 hard actions and 500-step token budgets are unchanged.

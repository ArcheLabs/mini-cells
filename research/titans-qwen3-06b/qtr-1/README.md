# QTR-1 — Gate 1 training recipe rescue

Retains Gate 1 layer index 22, native Q/K/V and RoPE, one-token updates,
128-token cache boundaries and 384-token episodes. No Titan-Llama model is
loaded. T imports only the pinned FP32AdamW implementation.

E0–E3 cross caps 0.0025/1.0 and Q/T optimizer profiles. Engineering runs
the same two episodes per arm, then compares a fresh-process resume from
step 1 against continuous step 2. It publishes the pairing and resume
checks; it neither ranks arms nor opens holdout.

Copy kaggle_qtr_1.py into Kaggle (T4, Internet, GITHUB_TOKEN). Default mode
is engineering. Formal segments use a published successful engineering
result from the same source. Keep each arm/seed at 500 steps overall;
STOP_AFTER is an absolute step, not a reduced training budget. Resume
from the previous release asset ID and published checkpoint SHA-256.
Formal natural-language Holdout selection/evaluation is not enabled;
the existing per-step evaluation is synthetic development only.

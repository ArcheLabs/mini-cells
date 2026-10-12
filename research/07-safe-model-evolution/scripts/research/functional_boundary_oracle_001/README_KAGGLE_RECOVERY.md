# Functional Boundary Oracle 001 — Kaggle recovery

If a formal seed exits before emitting `result.json`, classify it as an infrastructure interruption, not a scientific FAIL.

Recovery procedure:

1. Restart the Kaggle runtime/session to clear possible orphan GPU subprocesses.
2. Re-run the notebook from the first cell so it fetches the latest branch.
3. The launcher runs `kaggle_preflight.py` before each formal seed and requires at least 12,000 MiB of free GPU memory.
4. If the formal child fails again, inspect the streamed launcher log under `results/functional-boundary-oracle-001-launcher/` plus the automatically printed `nvidia-smi` diagnostics.
5. Do not change `protocol.json`, formal seeds, gates, prompt splits, or training rules in response to an infrastructure failure.

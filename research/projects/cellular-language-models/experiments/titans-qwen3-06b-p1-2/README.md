# P1.2 — Fast Weight Content Attribution

Frozen diagnostic replay of P1.1. No optimizer, training, or changes to Titans or Qwen. The three historical chunks use the original `NativeMemory` and its deferred commits. The last chunk is queried under six interventions with identical Q/K/V; it is never committed. Historical NLL and the first 16 published read diagnostics are reproduction checks. An instrumented vs uninstrumented comparison checks committed state and prediction equality.

## Run

- **Kaggle**: use the Python source of `kaggle_p1_2.py` in the existing notebook, with Tesla T4, Internet and `GITHUB_TOKEN`. A single run executes the preflight, then the nine-checkpoint replay only if it passes. No attached Output is required. GitHub publication and remote SHA verification use the shared publisher. A missing Kaggle version is recorded separately from the verified source commit; it is a provenance gap.
- **Local CUDA**: extract `artifacts/experiments/titans-p1-1-13c8dbc1f67b` from commit `0cc6870ded45abd3a0f5835cc9700f0fb52f8784` with `git archive`; check out upstream `sarahpannn/titans-pytorch` at `c60b4efaa3a3c6df4deebd57cbccf75947b87cad`. Use the dependency pins in the Kaggle launcher. From this directory:

```sh
python -m unittest -v
python replay.py /path/to/titans /path/to/p1-1-artifact /path/to/new-output
```

Add `--smoke` for seed 42 at step 0, document 0 and its three original donors. It validates all nine checkpoint file hashes but restores only the smoke checkpoint, and reports `SMOKE_ONLY`. The formal run restores parameters from all nine checkpoints by name, shape and dtype, without loading optimizer state into an optimizer.

## Interpret the output

`write_dynamics.jsonl` separates unit-weight gradients, the actual adaptive-LR-weighted gradients, clipped surprise, momentum carry and decay toward the learned slow map. The recurrence residual metrics compare the measured state to those terms. `fast_weight_pairs.jsonl` compares paired histories (including per-KV-head fast deltas). `read_diagnostics.jsonl` compares fixed-query reads before and after gate scaling, including all 16 Q heads. `prediction_replay.jsonl` records target log probabilities, NLL, logit divergence and top-1 flips. `statistics.json` preserves breakdowns per seed and step, with donor-mean NLL confidence intervals resampled by seed and document. The donor mean averages three NLLs. Diagnostic completion does not revise P1.1's scientific status.

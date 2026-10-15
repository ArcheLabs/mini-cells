# FAST-001 engineering delivery

Implementation commit: `0462816af47c82359f66e78830f106b22477c3b4`. CPU only; no GPU training, no confirmation data.

Tests: 31 FAST + 83 PD regression = 114 passed; no skips.

| Arm | Resident parameters | CLM/MoE |
|---|---:|---:|
| M0-TINY | 2,013,440 | 1.000000 |
| CLM-S8 | 1,011,000 | 0.502126 |
| CLM-S16 | 1,011,000 | 0.502126 |

S8/S16 parameter-name/shape inventories and complete initial state hashes match exactly.

## Measured CPU optimizer steps

3 warmup + 10 measured steps per case, one CPU thread, FP32, effective batch8, sequence64.
Same process sequentially measures all three arms. Separate representative one-microbatch dispatch/profiler work is excluded.

| Arm | Microbatch | p50 ms | p90 ms |
|---|---:|---:|---:|
| M0-TINY | 1 | 337.714 | 418.588 |
| CLM-S8 | 1 | 623.060 | 691.308 |
| CLM-S16 | 1 | 1252.281 | 1298.621 |
| M0-TINY | 2 | 239.905 | 267.752 |
| CLM-S8 | 2 | 376.618 | 404.733 |
| CLM-S16 | 2 | 665.201 | 696.322 |

Pilot end-to-end CPU wall time: 62.356 seconds including initialization and offline instrumentation.
These CPU numbers are not T4 estimates. CUDA peak bytes, kernel time and T4 total runtime remain unmeasured.

## Acceptance mapping

| Requirement | Automated evidence |
|---|---|
| Three real sparse arms, matched count/shape/init | test_models actual parameter / MoE / sparse-call tests |
| Cell, Connection, Context, head jointly train | sparse gradients + exact Cell/Router expectation enumeration |
| No STOP / self-connections / real repeat cost | fixed router shape, replay trace, Counter verification |
| Causal input memory and cached decode | future-token and full/prefill/decode replay parity for all arms |
| Token/trajectory loss normalization, detached critic | exact gradient test, actor depth scaling, critic isolation |
| Reproducible data and isolated windows | train-only real BPE test, exact/near filtering, raw-range boundary, window checks |
| No resident training instrumentation | light vs dispatch/capture gradient, parameter and RNG equality |
| Safe checkpoint / actual separate Python process restart | pause + portable export/import + child-process resume state equality |
| Interrupted checkpoint/receipt transaction | prior committed recovery-pair test |
| CLMs first; free GPU takes M0; failure preserves others | mocked concurrent scheduler tests, no GPU launched |
| Symmetric real interventions and report integrity | both CLMs, empty self-loop subset, full synthetic report, mismatch rejection |
| Historical protection and gated GPU phases | append-only Git diff, CPU Notebook default, freeze mutation and fixture CUDA rejection |

## Remaining measured-stage requirements

- User-authorized T4 pilot must measure both microbatches, three arms, single idle device; target is about 30 minutes for the complete 500-step development run.
- Freeze candidate config only after reviewing that measured projection. Over-target results require explicit acceptance; code never silently changes tokens/model/objective.
- New real TinyStories source/tokenizer/window hashes and exact retained story counts await explicit asset preparation; candidate metadata checks pass and confirmation remains sealed.
- Actual saved Kaggle Output recovery and CUDA numeric/device-state restoration have not been measured. CPU separate-process portability is not a cloud-persistence claim.
- Dynamic indexing, Python Cell dispatch and tiny kernels remain by design; half resident parameters do not imply half wall time.
- 256K target tokens and one development seed are exploratory, insufficient to conclude CLM succeeds or fails as a general language architecture.

No development run has been launched, and no frozen-development config granting GPU training is committed.

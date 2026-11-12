# B0-P2 Controlled Learnability Pilot

Status: B0_P2_COMPLETE_REQUIRES_RESEARCH_REVIEW. Single-seed development; H1–H6 NOT_EVALUATED. Test unaccessed.

Cumulative charged GPU-phase wall: 4922.42/7200 seconds. Controller wall through analysis: 5874.71 seconds (before final summary return).

## Quality

| Arm | Training Diagnostic | Validation | Full NLL | Type NLL | Conditional NLL | Value mass | Dominant rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 4/256 | 2/512 | 5.632760 | 0.001404 | 5.631357 | 0.998597 | 1.0000 |
| E6 | 4/256 | 2/512 | 5.637418 | 0.001892 | 5.635526 | 0.998110 | 1.0000 |
| E7-Innovation | 4/256 | 2/512 | 5.634478 | 0.001977 | 5.632501 | 0.998025 | 1.0000 |

Training Diagnostic uses32 Episodes already in Training. Validation uses64 Episodes/512 clustered Queries. Uniform valid-value conditional NLL is ln256=5.545177; it is not the full-vocabulary random baseline. Complete initial/scheduled curves and prediction distributions are in each verified bundle.

## Resources

| Arm | Static parameters | Updates | Fit wall seconds | Compute input tokens/s | Peak allocated MiB | Checkpoint MiB |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 10130688 | 640 | 110.46 | 6209.09 | 379.09 | 169.62 |
| E6 | 10021120 | 640 | 2026.66 | 194.06 | 753.32 | 167.46 |
| E7-Innovation | 10021120 | 640 | 2311.59 | 171.63 | 798.25 | 167.44 |

Timing includes parameter observers. Compute throughput excludes evaluation/checkpoint/publication; fit wall includes completed training/evaluation/checkpoint calls. The cumulative worker clock also charges restore and imports conservatively. These are implementation-level pilot costs, not universal architecture complexity claims.

## Paired uncertainty

| Paired comparison | Metric | Mean difference | Episode-bootstrap95% CI |
|---|---|---:|---|
| E6 minus E0 | exact_match | 0.000000 | [0.000000, 0.000000] |
| E6 minus E0 | nll | 0.004657 | [-0.003420, 0.012584] |
| E6 minus E0 | type_nll | 0.000488 | [0.000482, 0.000494] |
| E6 minus E0 | conditional_nll | 0.004170 | [-0.003793, 0.012353] |
| E7-Innovation minus E6 | exact_match | 0.000000 | [0.000000, 0.000000] |
| E7-Innovation minus E6 | nll | -0.002939 | [-0.007918, 0.002172] |
| E7-Innovation minus E6 | type_nll | 0.000086 | [0.000078, 0.000093] |
| E7-Innovation minus E6 | conditional_nll | -0.003025 | [-0.007628, 0.001814] |
| E7-Innovation minus E0 | exact_match | 0.000000 | [0.000000, 0.000000] |
| E7-Innovation minus E0 | nll | 0.001718 | [-0.006145, 0.009604] |
| E7-Innovation minus E0 | type_nll | 0.000573 | [0.000560, 0.000585] |
| E7-Innovation minus E0 | conditional_nll | 0.001145 | [-0.007240, 0.009120] |

Empirical bootstrap intervals can collapse at a floor/ceiling; they do not quantify across-seed variation.

## Previous Pilot reference

| Arm | B0-2 Validation | B0-2 NLL | P2 minus B0-2 EM |
|---|---:|---:|---:|
| E0 | 1/512 | 5.705709 | 0.001953 |
| E6 | 3/512 | 5.723018 | -0.001953 |
| E7-Innovation | 3/512 | 5.720505 | -0.001953 |

## Memory observations

E7 final Validation: 9105/32768 accepted write attempts; 268/2048 Query-layer reads select the corresponding factual Key position and 1189/2048 select its Value position.
These are actual selected-position overlaps across all SWA layers, not causal attribution. Retrieval scores, calibrated state saturation and a causal Store-effect intervention are NOT_AVAILABLE. Per-Episode Summary/state norms, activations, reset evidence, EMA statistics, admissions and evictions are archived.

## Interpretation and execution boundary

The quality and conditional-NLL curves describe whether E0 learns the task and how E6/E7 compare under the same input/data/update conditions. No post-hoc learnability threshold is invented. Training-versus-Validation gaps and concentration/type diagnostics distinguish memorization, answer-type learning and possible retrieval progress.
Differences from B0-2 cannot isolate the cause: pair membership, supplemental observation allocation and exposure changed jointly. A single seed and small differences cannot establish architecture superiority or a general Store benefit. Full-episode runtime, parameter counts and initialization were kept fixed.
Convergence comparisons use common update boundaries and the resource table. Additional controlled ablations and independent replication would be required before a scientific mechanism claim. Main12M training, the main three-seed experiment and M6 remain unauthorized.

## Verified artifacts

- E0: [complete Release bundle](https://github.com/ArcheLabs/mini-cells/releases/tag/b0-p2-e0-2993a008267e653c7f92), SHA256 8a1a75ab0af096f22f7856a75ee72818285f769339225c6b637fa4da32593d9c. Independent full download, all member hashes and downloaded-checkpoint re-evaluation passed.
- E6: [complete Release bundle](https://github.com/ArcheLabs/mini-cells/releases/tag/b0-p2-e6-1fe1e18770a69dd0774d), SHA256 68e7b3b1cf7cc97adc655845a8d402599f0c58d71dc3c205600a81e065a4e15b. Independent full download, all member hashes and downloaded-checkpoint re-evaluation passed.
- E7-Innovation: [complete Release bundle](https://github.com/ArcheLabs/mini-cells/releases/tag/b0-p2-e7-innovation-7eb46dd6fd8871691630), SHA256 b203865874e161ad2afb447e009172f0fa063f8c8b39c597e8cc119af7fe5543. Independent full download, all member hashes and downloaded-checkpoint re-evaluation passed.

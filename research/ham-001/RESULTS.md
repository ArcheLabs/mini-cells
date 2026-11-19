# HAM-001 initial CPU screening

Three paired seeds: 42, 43, 44. Primary metric: four or more candidate Top-1, except XOR sign accuracy.

| Random KV count | Local | Delta | Gated Delta | Full-history attention | Full KV |
|---:|---:|---:|---:|---:|---:|
| 8 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 16 | 100.00% | 97.92% | 95.83% | 100.00% | 100.00% |
| 32 | 50.00% | 79.69% | 77.08% | 100.00% | 100.00% |
| 64 | 25.00% | 47.14% | 41.15% | 100.00% | 100.00% |
| 128 | 12.50% | 23.70% | 21.09% | 100.00% | 100.00% |
| 256 | 6.25% | 9.57% | 8.72% | 100.00% | 100.00% |
| 512 | 3.12% | 4.88% | 4.49% | 100.00% | 100.00% |

Three CPU unit tests passed. Raw results include per-seed metrics, Student t intervals, interference matrices and transfer retention.

This is an initial screening. B1/B4 are unavailable in this run; H1 is incomplete. No conclusions about nonlinear Titans capacity or hybrid complementarity.

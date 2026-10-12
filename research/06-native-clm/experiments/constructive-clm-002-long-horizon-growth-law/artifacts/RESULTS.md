# Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law

- Status: `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`
- Scientific decision: `True`
- Protocol SHA-256: `eca97a71c5251519b43bda087e69a262241d388215f2da22ce8470db857b6454`
- Completed seeds: `[90411, 90412, 90413]`
- Missing seeds: `[]`

| seed | pass | growth exponent | oracle exponent | final Cells | final K/N | late spawn | late reuse | compression |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 90411 | True | 0.6631 | 0.6631 | 30 | 0.007324 | 0.004395 | 0.990723 | 136.53x |
| 90412 | True | 0.6631 | 0.6631 | 30 | 0.007324 | 0.004395 | 0.991211 | 136.53x |
| 90413 | True | 0.6631 | 0.6631 | 30 | 0.007324 | 0.004395 | 0.990723 | 136.53x |

A positive result is finite-horizon evidence only. It means learned Cell state tracks the registered sublinearly growing latent vocabulary across N=256..4096 while retention and composition remain usable. It is not an asymptotic proof of K(N)=o(N), a language-scale result, or a learned growth-policy result.

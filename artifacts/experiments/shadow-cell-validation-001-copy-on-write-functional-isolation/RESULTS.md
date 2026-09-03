# Shadow Cell Validation 001 — Copy-on-Write Functional Isolation

- Classification: `INCONCLUSIVE_DIRECT_PLASTICITY`
- Phase: `formal`
- Scientific decision: `True`
- Independent of Native CLM M2/M3 conclusion chain: `True`
- Protocol SHA-256: `8c1e4b16de9b6af83abb5e56043fed3dba1b8b58f94bd6be1d390d821946596e`
- Implementation SHA-256: `d860fd1c3badb581422d9f0f12cbeaad106cfeeb24cdaf4715173e14fed9ffb7`

| seed | base A acc | parent A share | parent B share | direct B gain | gate AUC | primary m | primary A damage | primary B gain/direct | HV gain vs direct | shuffled A damage advantage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 95211 | 1.0000 | 1.0000 | 1.0000 | 0.2638 | 0.7467 | 1.0000 | 0.0001 | 0.0133 | -0.8874 | 0.0033 |
| 95212 | 1.0000 | 1.0000 | 1.0000 | 0.1382 | 0.6946 | 1.0000 | 0.0003 | 0.0036 | -0.9725 | 0.0010 |
| 95213 | 1.0000 | 1.0000 | 1.0000 | 0.1454 | 0.7278 | 1.0000 | 0.0003 | 0.0014 | -0.9908 | 0.0007 |

Boundary: this experiment uses fresh synthetic data, fresh base checkpoints and fresh seeds. A/B calibration data may train the expression probe, but old examples never enter Shadow/direct operator weight training. The experiment does not modify any Native CLM M2/M3 scientific decision.

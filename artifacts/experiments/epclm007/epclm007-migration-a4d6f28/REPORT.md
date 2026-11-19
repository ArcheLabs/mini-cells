# EPCLM-007 Gate P migration report

**Decision: PORTABILITY_PASS**

Implementation: `a4d6f288aca4b1e1c3241af67ade1ff388ee3100`. CPU, Python 3.11.18, PyTorch 2.6.0+cpu,
FP32, deterministic execution with one PyTorch thread per worker.

Source PID 3106 exited with status 0 before Target
PID 3107 started. Target exited with status 0.
Source reproduced the Direct Gate fixture at seed 41035 with 128 full-set replay
updates over eight associations. Target performed zero updates and used the Cell
artifact as its only source of dynamic memory. Static weights were initialized
deterministically from the same configuration in both workers. The fixture uses
identity query vectors and the original cosine scoring function, with no external
backbone checkpoint or tokenizer execution.

| Metric | Source | Target |
| --- | ---: | ---: |
| Top-1 correct | 5/8 | 5/8 |
| Top-2 correct | 5/8 | 5/8 |
| Mean true-candidate rank | 2.625 | 2.625 |
| Mean top-two margin | 0.025308132 | 0.025308132 |

All eight candidate rankings matched. Readout and score maximum absolute and
relative errors were zero (atol 1e-6, rtol 1e-5). Fast weights, momentum and step
matched at export, import, and before and after retrieval. The portable artifact
contains 38765 bytes. Export took 0.002373 s and
verified import took 0.003043 s. The artifact identity is
`e521075d1175911d78ee122bd28ce7af37338cb99c47460f52de84c1aee4878c`, computed from the per-file hashes using the existing
Cell integrity format.

See `source.json` and `target.json` for raw tensors, rankings, process identities
and compatibility metadata. `decision.json` records the gate comparison.
The focused pytest suite passed all three tests, including rejection of corrupted
state and mismatched backbone or projection identities.

Gate P covers this fixed CPU fixture. The separate 8/8 Direct Gate remains blocked.
E3 training, Test access and T4 Screening were outside this run.

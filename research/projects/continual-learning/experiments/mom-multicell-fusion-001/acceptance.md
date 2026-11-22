# Acceptance

CPU fixtures exercise the fusion equations, joint softmax dimensions, Cell
permutations, Shared Memory, extreme input scales, the empty path, native-read
adapter boundaries and frozen parameters. They also run the existing Cell
serialization, lifecycle, stream and recall tests. FP32 factorization uses
an absolute tolerance of 1e-6 on ordinary fixture outputs. The real-model
preflight records individual FP16 hidden-state maximum errors and applies
the predeclared tolerance (atol 0.025, rtol 0.01). Separate subprocesses check
fresh-process restore with the pinned model and identical persistent hashes.

Awaiting a real GPU run for whole-model parity and fresh restoration,
Gate convergence, QA quality, failure cases and measured performance. This
pilot does not confer product acceptance. Retain legacy equal fusion as the
default until an independent result establishes a better configuration.

Run artifacts include `preflight.json`, `preflight-restored.json`,
`datasets.json`, `training.json`, `test-queries.json`, `v2-replay.json`,
`quality.json` and `benchmark.json`. The repository publisher adds the
decision, source identity, SHA-256 manifest and `RESULTS.md`.

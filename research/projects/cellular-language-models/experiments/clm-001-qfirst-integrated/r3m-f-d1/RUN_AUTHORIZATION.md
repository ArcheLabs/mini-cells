# R3m-F-D1 execution authorization

Date: 2026-10-25 (Asia/Shanghai).

The human user explicitly authorized starting the diagnostic after receiving independent review and the audit closeout: “好的，现在授权启动，我应该如何运行？”

Authorized execution source: `16df822e95ac2466529f424a8ebfec63ee93c295`.
Configuration SHA-256: `656da576ad7e6908825c9c496a5434155957f591aadcf4fee00ddbd14f67648b`.
Audit closeout evidence commit: `e1d0f6e66baf7f294a992c488d8e0eada1add0d3`.

Scope: B/C/D only; each arm at most 200 logical optimizer updates from the pinned R3h B-H Step300 checkpoint, with frozen sampling, parameter mask, optimizer continuation and evaluation barriers. A is reused without retraining. Failed-attempt cumulative work may exceed the logical trajectory budget and remains explicitly unmeasured. No full R3m-F Pilot, Main or Confirmation is authorized.

This record supplies the later separate human authorization contemplated by the original engineering configuration. The frozen scientific configuration is not rewritten. The Notebook supplies the existing `--authorize-d1-run` flag, defaults to `RUN_BCD=True` and automatically performs preflight before execution. The code checkout directory includes its pinned source prefix to avoid reusing an old Notebook checkout.

This authorization record and launch preparation do not claim that the cloud run has already started or succeeded.

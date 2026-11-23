# MINICELLS-PYTHON-SDK-001 acceptance

| Stage | Evidence |
| --- | --- |
| Audit | Native runtime, state, selector, generation and GPU limits mapped in [architecture](architecture.md) |
| Runtime and API | 29 targeted CPU tests passed. SDK fixtures cover serialized access, Cell management, revisions, resumable writes, independent score/fusion caches, no-memory modes and limits |
| Portable storage | Real file tests cover native and collection ZIPs, checksum failure, model mismatch, conflict and interrupted-commit rollback, cross-process fixture restoration and append, and unchanged sibling state |
| Packaging | Wheel built and public API imported from the installed wheel outside the checkout |
| Web | Existing web lifecycle test exercises the SDK adapter, with automatic and manual query paths using SDK logic |
| Real GPU | Kaggle run pending. Local Quadro K620 is compute capability 5.0 with 2 GiB. |

Use [the Kaggle Python Script](../../apps/mom-native-cells/kaggle_sdk_acceptance.py)
with T4, Internet and the existing `GITHUB_TOKEN`. It installs the SDK, runs the
CPU tests and exercises real generation, single/collection restore, continued
writes, teacher-forced parity and complete-query profiling. It publishes the
generated text for manual readability review. An engineering smoke status
does not resolve the historical free-generation issue. The product MVP remains
pending real GPU smoke and generated-text review.

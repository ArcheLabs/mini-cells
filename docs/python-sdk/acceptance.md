# MINICELLS-PYTHON-SDK-001 acceptance

| Stage | Evidence |
| --- | --- |
| Audit | Native runtime, state, selector, generation and GPU limits mapped in [architecture](architecture.md) |
| Runtime and API | 29 targeted CPU tests passed. SDK fixtures cover serialized access, Cell management, revisions, resumable writes, independent score/fusion caches, no-memory modes and limits |
| Portable storage | Real file tests cover native and collection ZIPs, checksum failure, model mismatch, conflict and interrupted-commit rollback, cross-process fixture restoration and append, and unchanged sibling state |
| Packaging | Wheel built and public API imported from the installed wheel outside the checkout |
| Web | Existing web lifecycle test exercises the SDK adapter, with automatic and manual query paths using SDK logic |
| Real GPU | Tesla T4 smoke completed. The product generation check remains open. |

Use [the Kaggle Python Script](../../apps/mom-native-cells/kaggle_sdk_acceptance.py)
with T4, Internet and the existing `GITHUB_TOKEN`. It installs the SDK, runs the
CPU tests and exercises real generation, single/collection restore, continued
writes, teacher-forced parity and complete-query profiling. It publishes the generated text for manual readability review.

## Tesla T4 run 8b17779e1abb

The installed wheel passed all 29 targeted tests. The GPU run completed native,
single-Cell, manual multi-Cell and automatic inference, export/import in a new
process, and continued writing. Import preserved bundle hashes. Queries
preserved those hashes, and the append left the sibling Cell unchanged.
The largest teacher-forced token log-probability change was 0.1664 and passed
the combined absolute/relative tolerance (0.15, 0.03).

| Registered Cells | Complete query (s) | Peak allocated (GiB) |
| ---: | ---: | ---: |
| 1 | 3.21 | 1.09 |
| 2 | 5.30 | 1.21 |
| 4 | 6.84 | 1.21 |
| 8 | 9.88 | 1.21 |

Each row is one warmed 12-token generation with forked copies of one Cell.
Automatic selection uses at most two Cells for final fusion. The profile
measures registry size and scoring cost.

**Product MVP remains open.** The reported GPU status is
`GPU_SMOKE_COMPLETE` and generation quality is `REVIEW_REQUIRED`.
The free-form founder query returned “Wangvera B-which B” followed by
fragmented text, and the no-memory capital query also returned fragments.
The founder cloze teacher-forced probe favored the correct candidate,
but it uses a different prompt and evaluation protocol. A matched-prompt
comparison of teacher forcing and stepwise decoding is needed to localize
the generation issue.

[Published decision, generated text, restore evidence and profile](https://github.com/ArcheLabs/mini-cells/tree/a3eba0d85993e09f7b176abd10c6a276b37a8a9b/artifacts/experiments/minicells-python-sdk-001-8b17779e1abb)

# M0-001 repository audit

## Release baseline

The candidate is based on `main` commit
`32048a03fa297cd8ab481d34ed86f115de534169`. At that commit the research
distribution is `mini-cells 0.2.0a1`, and the native MoM Python package
and its SDK documentation are absent from that branch.

The bridge source was `133ca503e50adc1adb4e96b0bad21d8134c8d246`.
The only change from the initial bridge source `5072ff79` to that
executed source was the reporting fix, CPU entry-point test, and launch
gate. Its published nine-case output is independent development evidence.
The earlier MVP quality branch `46de2806` and the 340M SDK smoke
`117c6e45` belong to separate histories and do not qualify this release
candidate. [Bridge evidence](https://github.com/ArcheLabs/mini-cells/tree/6f3f5dd8072685c657cafe4753c3f0a92bdd5faf/artifacts/experiments/mom-memory-bridge-001/0b117b5493e8).

## SDK differences and evidence boundaries

- The bridge branch exposed create, append, fork, delete, export and import.
  Existing codec checks cover schema, identity, hashes, shapes, finite
  tensors, and conflict rejection. Its native Cell format restores
  recurrence and resets stream history. The cumulative token offset is
  already stored as `tokens_written`.
- The former SDK defaulted to automatic selection and offered legacy model
  and precision choices. The audited A.3.2 corrections lived in scoped
  diagnostic hooks. The M0 API now uses an explicit supported preview
  profile, with an empty selection by default. Its profile is an identity
  field, so earlier unprofiled bundles require an explicitly compatible
  legacy runtime or reconstruction.
- Historical CPU tests covered registry isolation, forks, writes, archive
  handling, selection, and a small mock restore. A previous T4 smoke
  tested a different SDK source and mostly 340M. The bridge test covered
  one-process state persistence and candidate scoring. None of these
  substitutes for a clean-wheel run on this release candidate.
- The supported model is the integrity-checked MoM 1.3B revision
  `5c8be94196cb1827e820077680ac2ee46ac949bc`; its tokenizer uses the same
  revision. MoM source is pinned at `125038087b71b635c171cec857bd0f21d10955b6`.
  The previous T4 baseline used PyTorch 2.11.0+cu130, Triton 3.6.0 and FLA
  0.2.2. The candidate pins FLA 0.2.2, Transformers 4.51.3 and tokenizer
  packages, retains the installed PyTorch/Triton in Kaggle and records
  their actual versions.
- A.4's native Chunk parity failure remains in historical evidence. The
  current execution plan defines bounded FP16 recurrence as the preview
  release path. MVP quality, free-form recall, and fusion gains remain
  unproven.

## Implementation and release sequence

1. Integrate the native package on a main-based preview branch. Build a
   separately versioned Native SDK 0.1.0 wheel.
2. Reuse the validated stable packing, ShortConvolution, and inert expert
   padding behavior. Exercise state lifecycle, compatibility and ordering
   in CPU tests.
3. Publish CPU and T4 evidence on the exact final candidate SHA. The T4
   gate runs an installed package, with independent writer and reader
   processes and a bounded long prompt.
4. Publish a release only after M0 tests and the T4 gate pass. M1-001 runs
   independently and owns the next research decision.

# Kaggle Output recovery — R3b

## One launch

Import `r3b/kaggle-output-analysis-notebook.ipynb` into the existing **private** `bingjiangli/minicells` Notebook, with its existing `KAGGLE_API_TOKEN` and `GITHUB_TOKEN` Secrets enabled, then **Save & Run All**. `HF_TOKEN` is optional and used only for read-only recovery of old raw observations. No new training or scientific approval is requested.

The automatic flow discovers exact numbered Saved Outputs, checks all pages, finds the 57 frozen Checkpoints and original assets/banks, verifies them before loading PyTorch state, recovers original observations, computes only missing slots, saves a cumulative private result package, draws frozen-engine summaries, and publishes public projections.

Version65 is a discovery candidate, not a trusted training source. The web `scriptVersionId=359735394` is recorded separately and is not converted to65 by assumption. The actual training and prior-analysis versions are recorded independently after API and file verification. Missing historical versions do not end discovery.

## Scientific identity

- Training source: `a7792ef30582dfcaf324cca7e5245c8430df48b2`.
- Internal run: `r3b-a7792ef30582-ea3ea51c`; public directory: `33734fdd1558`.
- Measurement engine: **`ed2accbc90f541325ca2f6c53229e9c81475768d`** in a separate immutable checkout.
- The original HF reference remains an identity field only. Training bytes come from verified Kaggle files, never an HF fallback.
- The new transport/persistence package is `kaggle_posthoc`, outside the frozen measurement package. The launcher verifies the engine checkout and materialized tracked bytes before importing it. Task identities continue to use the actual old scientific engine, not a new module falsely relabelled as old.
- Quality39 / Functional42 / Action21, original policies, windows, event revisions, sampling, gradients and statistics remain unchanged. Confirmation stays sealed.

## Persistence and limits

Inputs, code checkouts and publication checkouts are under `/kaggle/temp`, outside Saved Output. Only the cumulative private observations, indexes, manifests, provenance and reports are under `/kaggle/working/minicells-r3b-posthoc`.

Every raw task is written atomically before any network operation. COMPLETE means result data were verified; `cross_session_persistence_verified` separately states whether the bytes were read back from a saved Kaggle version or immutable HF revision. New local observations are `COMPUTED_LOCAL` until Kaggle actually retains the Output. The running Notebook prints `OUTPUT_VERSION_PENDING_SAVE`; it cannot invent its future saved version number.

At a soft time boundary or SIGTERM request, the current task finishes and commits locally. A new Save & Run All imports cumulative results from verified saved versions. No external controller is available here: the Notebook **does not start future cloud sessions itself**. `NEXT_RUN_REQUIRED` is explicit. No optimizer update or process time-limit termination is introduced.

HF recovery performs **zero commits**, including for orphan results. HF429/offline recovery never prevents new observations being computed and saved locally. Optional HF backup is disabled by default. Logs/HTML/coverage counts without raw observations are not reused as COMPLETE.

The pinned official Kaggle CLI commit is `6894864ff4095cf98e9e36d3a694ff5fc9ba3818`. It supports `owner/slug/version` and maps numeric versions to API `v<N>` labels. Source references: [official CLI commands](https://github.com/Kaggle/kaggle-cli/blob/6894864ff4095cf98e9e36d3a694ff5fc9ba3818/docs/kernels.md), [implementation](https://github.com/Kaggle/kaggle-cli/blob/6894864ff4095cf98e9e36d3a694ff5fc9ba3818/src/kaggle/api/kaggle_api_extended.py). If current metadata exposes no latest version number, the inventory explicitly records the bounded numbered-version search; it never labels an unversioned Output as frozen input.

## Current execution evidence

This environment has no Kaggle/HF credentials and cannot verify the private historical Output. Therefore the real training version, version65/page-ID mapping, 57/57 actual input verification, old42 recovery count and completed diagnostic counts are **NOT_EXECUTED**, not inferred from the user's logs. Runtime generates the real inventory, checkpoint/recovery ledgers, complete task matrix, research report, timings, private snapshot hashes and verified public publication receipts.

Training ledger facts remain unchanged: three500-step arms; recorded CUDA update resource sum4236.461s; full training wall, complete semantic-event resources and two-hour verdict remain NOT_MEASURED / NOT_VERIFIED. Old and new analysis time is not fabricated from one another.

## Duplicate listing repair

Kaggle listings may repeat the same path. Equal path/size entries are counted once; different sizes fail closed. Listing equality does not establish content equality: all frozen file SHA checks still run before loading checkpoints. The inventory records page and repeated-entry counts, and progress logs show each successfully enumerated version. No real Kaggle rerun has been claimed.

## Authoritative path repair (supersedes display-list de-duplication)

`ListKernelFiles` provides `name`/`size`; these display entries are not used as unique download paths. Discovery now uses **ListKernelSessionOutput.file_name**, exactly as the pinned official `kernels_output` downloader does. This preserves separate Arm directories even when filenames match. The official SDK schema was audited at [f983c972](https://github.com/Kaggle/kaggle-sdk-python/blob/f983c97287bf274ebab506aac85eda6efebe3b32/kagglesdk/kernels/types/kernels_api_service.py).

The authoritative response has URLs and paths, not file lengths. Signed URLs remain in memory only. For selected files, a streamed GET reads Content-Length and closes before consuming the file body (HEAD is not assumed to be authorized by a GET signature). Capacity, completed-download length and frozen file hashes are then checked. Unselected byte counts stay unknown; enumeration does not issue an HTTP request per unrelated file. Inventory SHA covers sorted authoritative paths, with lengths independently recorded once resolved. Scientific task identity and frozen hashes are unchanged.

28 offline CPU transport tests passed, including two identically named files under different Arm directories, lazy length resolution, no signed URL in inventory, and streaming-header behavior. Real private Output execution remains unverified here.

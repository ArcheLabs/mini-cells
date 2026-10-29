# Independent review of recovered historical metadata

Disposition: recovery provenance and metadata structure verified; R3m registration and Window Gate remain BLOCKED. No candidate reselection or scientific-protocol amendment.

## Verified inputs and identity

The three user-supplied files are retained byte-for-byte in this directory. Their SHA256 and sizes match `INDEPENDENT_REVIEW.json`. The recovered manifest matches the fixed original R3h manifest SHA256 `d8a0fd0da03e6c15bec8506c4e217f7e9c52111ac792470cf271036352a51193`. Its window object is `0ecfd4848bb99b01bf87cb50ec02265b7fcc2a1506fbec6bfc28b2a1f40d9291`. The recovery auditor hash, HF repository, fixed revision `828585f1ff93420aa4485c725211ed28f70d15f3`, bundle path/hash and original tokenizer identity all match the frozen recovery implementation/configuration.

This independently checks the submitted proof's consistency with pinned identities; local authenticated HF access was not used. The proof attests the exact original tokenizer bytes on Kaggle, SHA256 `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`. The tokenizer file itself was not included in this submission. Local byte hashing/re-tokenization therefore remains not performed, rather than falsely marked PASS. No substitute tokenizer was trained.

## Verified historical membership and window structure

The full cached original Parquet shard was checked against its fixed SHA256. All 8,704 recovered raw story IDs and text hashes match that shard and the manifest's original membership. The four roles have no shared raw IDs or identical raw story hashes. Confirmation is explicitly unmaterialized and absent from the window roles.

| Role | Stories | Eligible windows | Evaluation windows |
|---|---:|---:|---:|
| Train | 8,192 | 23,940 | 0 |
| Calibration | 128 | 330 | 32 |
| Admission | 128 | 317 | 32 |
| Development | 256 | 620 | 128 |

For sequence length64, every start/group pair exactly matches the historical nonoverlapping enumeration derived from the recovered story offsets. Input plus target stays within one story. All192 evaluation selections exactly reproduce the original FAST `grouped_windows` function with seed151001304. That pure function is reused from its original file without importing a model; its source hash is recorded.

These checks establish the internal consistency of the recovered window metadata and raw membership. They do not independently confirm token lengths until original tokenizer bytes are supplied, nor recover actual optimizer-consumed starts. Train's23,940 entries are the eligible pool, not a declaration that each window was trained on.

The unchanged512 Main candidates have no raw-ID or full-story exact-hash collision with these recovered four roles. This is a scoped whole-story result, not the global fragment/window leakage result required for registration.

## Remaining evidence and decision

- Supply the existing `tokenizer.json` from the same retained Kaggle Output (265,240 bytes). No new Notebook execution is required. Then independently hash and re-tokenize the fixed historical records to verify offsets and window definitions.
- Actual R3k/R3l research-role asset manifests and windows remain unrecovered; these files are the historical R3 four-role assets, not those later research-role outputs.
- Observed optimizer starts or an independently reviewed/approved conservative-source containment alternative are still needed for the applicable Window Gate. Whole-story nonoverlap alone is insufficient.
- Unknown historical reservation extents and global coverage limitations remain. Scientific review is pending; Path B has not been enabled.

Earlier amendments, candidates, source/config identities, checkpoints and Confirmation remain unchanged. Zero model evaluations and zero training updates were executed.

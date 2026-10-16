# Hugging Face checkpoints + GitHub research evidence

Engineering only. No real-data R3 rerun, no R3b authorization, no science changes.

## One launch

Import `notebook.ipynb` into the existing Kaggle notebook, use **Accelerator None**
and enable existing Secrets `HF_TOKEN` and `GITHUB_TOKEN`. No Add Input, local
file transfer or `kaggle kernels output` is used. Default repo slug is the requested
`minicells-research-checkpoints`; HF_TOKEN's account namespace is resolved by
`whoami`. The resolved private repo ID is printed. A full `organization/repo`
can be used instead; no existing historic repository is modified or made public.
Default setup creates this dedicated private repo if absent, then actual tiny
write/pinned-read and existing GitHub write preflight run before any CPU update.
Token creation rights may be needed for first creation; an existing private repo
only needs its read/write permissions. Tokens remain in memory/environment only.

`ACTION=auto` starts an absent engineering run or resumes its recorded HF state.
It never falls back to zero on missing/corrupt/forbidden requested state. Explicit
`train` rejects an existing run, `resume` requires a reference/key; `evaluate` is
read-only; `fork` requires a new run identity/config and explicit optimizer policy.
Source SHA and run identity are frozen. Notebook resolves identities internally.

The source-pinned CPU fixture serially reuses three original R3 arms. A single
publisher uploads at each completed safe checkpoint/event boundary, including all
committed event revisions. Failure stops the segment at that boundary, retaining
local state and publishing STORAGE_BLOCKED when GitHub is available. No hard
runtime termination is introduced. Final upload performs complete remote readback;
every resume always checks every SHA256 before deserialization.

## Real A / B / C acceptance (pending)

1. **A:** defaults, segment end 6. Run All. HF receives the tiny complete resume
   package, GitHub receives index/status. Retain version link + HF revision log.
2. **B:** NEW cloud Session, same Notebook/source/repo/run-id; change only
   `SEGMENT_END_STEP=10`. Run All. Auto finds remote state; all arms must report
   restored step 6 and random-init false. No assets regeneration for remote resume.
3. **C:** NEW cloud Session, set `ACTION="evaluate"`. It resolves HF latest at a
   precise revision, evaluates historical QC-R3 step6 (no optimizer update), and
   compares B's resumed full state with an independent continuous CPU 0→10 control.

All three stages are synthetic CPU engineering only. No additional Kaggle suite,
Pilot, real dataset or GPU budget. HF_CROSS_SESSION_RESUME remains NOT_VERIFIED
until actual private HF I/O, separate sessions, state equality and clean GitHub
metadata evidence have been reviewed. Gate D includes recovery through the published
GitHub checkpoint-index: supply its saved JSON via `--checkpoint-ref`; it carries
repo/revision/bundle hashes rather than arbitrary download URLs. The first small
JSON can be fetched from the GitHub result branch; weights never pass via GitHub.

## Layout and concurrency

HF uses model-independent `objects/sha256/<digest>`, `runs/<identity-hash>/bundles/`
and per-run/per-arm latest pointers. This mapping preserves the existing byte-level
continuation schema, probes, assets and all historical event revisions without
reformatting checkpoint pickle. Immutable assets deduplicate within the repo.
The exact HF revision is the durability boundary; index is produced only after
commit/path verification. Historical indices keep working as latest advances.

CAS `parent_commit` prevents lost updates. Same-step/event unequal hashes and stale
or missing-arm snapshots fail closed. The current adapter uses one serialized
publisher. Parallel GPU integration must send snapshots to that queue, never treat
independent partial bundles as a safe state merge. Frozen historical R3 remains
unchanged. This launcher exposes synthetic CPU engineering only; a future real
R3b launcher and frozen scientific budget require a separate review.

GitHub publisher reuses R3 main-descended result branches; explicit aggregate
projection contains checkpoint-index/status only. No `.pt`, assets, raw Oracle,
private probes, optimizer/RNG or token values are staged. STORAGE_BLOCKED can coexist
with an earlier valid DURABLE_PARTIAL index; failed new uploads never replace it.
Scientific conclusions and durable storage states are separate.

## Evidence and limitations

See `reports/engineering.json` for CPU results and real-I/O status. Local offline
transport tests cannot verify actual account permissions, quota, HF throughput or
Kaggle session persistence. We have no local HF credentials; no real HF repository
has been created from this workstation. Actual repo/revision will be recorded by
your Kaggle run. Old R3 Kaggle outputs have not been retrieved/migrated and remain
UNAVAILABLE/NOT_VERIFIED. Large-run upload interval must be measured before real
training; synthetic upload timing is not an estimate of full checkpoint throughput.
No storage quota number is assumed. Trusted private repo and pinned verified bytes
are required before loading pickle. A private token/repo alone is not proof of trust.

References: [HF atomic commits and parent_commit](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#create_commit),
[pinned revision downloads](https://huggingface.co/docs/huggingface_hub/guides/download).

# Existing implementation reuse

| Existing component | Reused contract | Extension |
|---|---|---|
| Native CLM M2/M3 publishers in context/06-native-clm/scripts/research | HfApi(token), auth_check(write=True), model repository, hashes and exact revision | One atomic create_commit with parent_commit instead of separate mutable uploads; private target required |
| fetch_native_clm_v0_m1_checkpoint.py and fetch_native_clm_v0_m3w0_checkpoints.py | hf_hub_download at resolved immutable revision plus full SHA256 verification | Entire continuation bundle, not just final model weights |
| Frozen R3 checkpoint.py / events.py | Original torch state, optimizer and independent RNG, data cursor, Registry, candidates, event revisions | No changes to historical implementation; shared continuation adapter safe-boundary callback |
| Existing minicells_runtime store/r3/evaluation/fork | Commit-last manifest, full identity checks, exact event revision, immutable probes, independent evaluation and explicit fork | Remote object mapping reconstructs byte-identical local package |
| R3 publication.prepare/helper | main-descended metadata result branch and authenticated preflight/push | Explicit checkpoint-index/storage-status projections; no binaries/private raw samples |

HF transport is model independent and never deserializes pickle. The trusted R3
adapter supplies event cursor metadata; the generic transport only verifies bytes.
A single publisher owns the whole bundle. Competing publishers use repository CAS;
a stale or incomplete bundle cannot roll back any arm. It fails closed rather than
silently merging independent model states. Future parallel GPU workers must route
safe-boundary snapshots to this single publisher; this CPU launcher does not start them.

Historical R3 remains unchanged. Earlier Kaggle checkpoints not actually retrieved
and uploaded remain UNAVAILABLE / NOT_VERIFIED. Old Native CLM HF repos are untouched.

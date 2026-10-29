# Amendment 003 — content selection, registration blocked

The selector protocol and shared implementation were committed in `c3da320d` before candidate generation. No model quality results are inputs. The pool is explicitly the entire fully verified first frozen train shard, not a newly moved continuous evaluation range. Selection ranks normalized content by domain-separated SHA256 and emits 512 individual original indices, with three content hashes per record. A different pool or algorithm requires a new amendment, not retries until a desired outcome appears.

## Evidence interpretation

`MAIN_CANDIDATE_INDICES.json` is an immutable **provisional** audit list, not a qualified or registered Main dataset. All candidates exclude exact/format/word-sequence collisions in the recorded index, including the full conservative original 007 training prefix, its validation prefix, the known development/reservation union and the pending Trigger/Pilot content. This does not prove complete global historical coverage. The historical declarations and their original source hashes are retained in `HISTORICAL_COVERAGE.json`; unresolved revisions and reservation endpoints remain UNKNOWN. The SQLite index contains per-record source, role and raw index. It is a local reusable artifact, not a claimed HF upload.

The index deliberately includes skipped empty/terminal records from the 007 raw prefix. That over-exclusion is safe but must not be described as actual optimizer usage. Individual historical declarations covered by a conservative union have not thereby acquired verified identity. Other frozen Revisions or unknown loader selections are not silently mapped to this snapshot.

## Artifact recovery

Twenty-one tracked tokenizer paths were checked; none matches the required original R3m SHA256 `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`. The fixed upstream original bundle at HF revision `828585f1ff93420aa4485c725211ed28f70d15f3` returned HTTP401 without authenticated access. This does not establish that the artifact is missing. No replacement tokenizer was trained. Exact historical sampled-start/window manifests have not been recovered. Existing corpus hashes and a saved cursor are not substituted for those records.

Required next evidence is authenticated read-only retrieval of the pinned original bundle and tokenizer, plus historical selection/window manifests or sufficient saved sampler state to reproduce observed windows. Use the existing HF_TOKEN through the existing artifact store; do not paste credentials. Recovered files must match pinned hashes before use. No GPU or model quality evaluation is needed for this recovery.

## Independent gates

- Recorded-index content exclusion: pass only within stated coverage.
- Global content isolation: UNKNOWN, blocked by unresolved historical identities/scopes.
- Exact original tokenizer: unrecovered, blocked.
- Exact historical token-window nonoverlap: unrecovered, blocked; no PASS claimed.
- Scientific review: pending even if all technical gates later pass.
- Trigger/Pilot: pending, not registered.
- Old Main `[500512,501024)`: abandoned; Amendment001/002 evidence unchanged.

Scientific config and releases lock remain unchanged. No new scientific Source/Config identity has been minted. No historical model, checkpoint or Confirmation was modified or evaluated. The 512 records are not exposed to any model. No GPU training was started.

## Reproduction

Run `audit_content_selection.py` on CPU with pyarrow and the two exact cached Parquet files named in the script. It verifies full file hashes, uses an exclusive new SQLite/output path and refuses to overwrite evidence. For a fresh independent reproduction use an isolated machine with a clean output directory and absent default SQLite path; compare candidate/protocol hashes. Existing proof files are never overwritten. `registry_checks.py` exercises deterministic order, duplicate exclusions, retained provenance, insufficient pools and UNKNOWN/review blocking on synthetic data only.

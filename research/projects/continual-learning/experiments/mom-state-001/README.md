# MOM-STATE-001 — Oracle-only sequential baseline

ROUTE-002 Oracle scored 12/12 (minimum 10/12). Run `kaggle_state.py`
as a Kaggle Python Script with T4, Internet and GITHUB_TOKEN.

The writer appends A, B and C to one MoM cache, queries deep copies,
and saves the complete ABC state. It then appends D for the reference.
The writer exits before the reader starts. The reader checks the bundle
hash, restores without replay, queries ABC, appends D and queries ABCD.
Compare every cache field and both candidate log probabilities with zero
tolerance. Negative margins remain visible even when restore is identical.
Routing is not used; expert telemetry consists of per-layer counts.
Checkpoint tensors remain in Kaggle Output; JSON/JSONL reports return to GitHub.
This baseline does not adjudicate the full protocol's conflict, interference
or version-update cases. The CUDA baseline completed with BASELINE_MISMATCH; see RESULTS.md.

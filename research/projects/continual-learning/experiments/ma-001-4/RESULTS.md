# MA-001.4 — Multi-Cell Runtime Feasibility

**Final development engineering verdict: PASS (7/7).**
The Kaggle CPU run completed in about 117 seconds. See the archived
[publishing decision](artifacts/ma-001-4-runtime-5e905aff6b57/decision.json)
and [runtime report](artifacts/ma-001-4-runtime-5e905aff6b57/runtime/decision.json).

- Three independent states shared one seeded upstream Titans module.
- Staged multi-write rolled back on injection and read-only access was isolated.
- Fresh-process restore and directory relocation matched continuous execution
  exactly, including the next write and subsequent reads.
- The frozen MA-001 index routed the known valid query to MiniJAM, MiniCells
  and AI Memory, and all three real Titans states were read.
- The state bundle was 50,241 bytes. Parent-process peak RSS was 684,880 KiB;
  this is not a combined parent-and-worker peak.

This closes the runtime feasibility study. It does not resolve the MA-001.3
evidence verifier's false support for an absent approving vendor. The next
development experiment should target missing actors, facts and relations
with facet-level annotations and matched distractors. Keep the established
runtime as a reference; defer MA-002 confirmation and learned memory quality
claims until evidence verification is evaluated on fresh data.

## Earlier local attempt

**BLOCKED locally.** The CPU attempt stopped importing the pinned upstream
Titans implementation because `assoc_scan` is missing. No core acceptance
case executed. See [the local receipt](artifacts/local-blocked.json); its
source SHA is the base of the uncommitted implementation at the time.

The experiment reuses the HAM-001 adapter at the pinned Titans revision.
Writes call upstream `store_only`; reads call upstream `read_only`. A single
seeded module serves each Cell's independently copied NeuralMemState. The
checkpoint codec includes every field of the upstream named tuple, not just
fast weights. No mock implementation is used for runtime acceptance.

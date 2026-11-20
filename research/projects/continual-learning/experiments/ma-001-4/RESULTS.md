# MA-001.4 — Multi-Cell Runtime Feasibility

**BLOCKED locally.** The CPU attempt stopped importing the pinned upstream
Titans implementation because `assoc_scan` is missing. No core acceptance
case executed. See [the local receipt](artifacts/local-blocked.json); its
source SHA is the base of the uncommitted implementation at the time.

The experiment reuses the HAM-001 adapter at the pinned Titans revision.
Writes call upstream `store_only`; reads call upstream `read_only`. A single
seeded module serves each Cell's independently copied NeuralMemState. The
checkpoint codec includes every field of the upstream named tuple, not just
fast weights. No mock implementation is used for runtime acceptance.

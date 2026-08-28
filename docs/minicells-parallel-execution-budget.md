# Worker execution lanes and budget

The concurrent-training v1 path uses four application-level execution lanes
and leaves canonical MiniJAM protocol bytes unchanged.  Lane assignment is
deterministic (`work_id % lane_count`) and candidate submission remains a
single ordered queue so account nonces cannot race.

The current MiniJAM runtime audit found:

- `MaxWorksPerRound = 4`;
- `MaxDutiesPerWorkerPerRound = 2`;
- `MaxExecutionReports = 4`;
- `MaxExecutionGas = 6,000,000,000`.

`MaxExecutionGas` is a block-level aggregate check: report projection sums
Refine and Accumulate gas and rejects the block when the total exceeds the
configured value.  It is not a per-report ceiling.  Consequently, four
2.3B-gas training leaves cannot currently be claimed as one same-block,
four-way runtime execution; raising the aggregate limit or changing topology
is deliberately out of scope for concurrent-training v1.  The worker lane
types and metrics are a compatibility scaffold, not a claim that MiniCells
owns ChainSpec or full JAM multi-core conformance. The consumer config records
`execution_lanes = 4` for this workload while the MiniJAM worker default remains
one lane.

The production decision remains fail-closed until a real worker executor
records overlapping wall-clock intervals (`peak_refine_concurrency >= 2`) and
the fresh-chain one-step gate is available.

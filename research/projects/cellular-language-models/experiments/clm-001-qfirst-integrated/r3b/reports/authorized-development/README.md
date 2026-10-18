# User-authorized estimated-budget development — 2026-10-18

The user explicitly approved the estimated budget, including the possibility of
exceeding two hours, and requested starting the three500-step development arms.
`config/authorization.json` records that decision. It does not claim a measured
PERFORMANCE_GATE_PASS. The previous strict two-hour release condition is replaced
only for this approved development execution; Confirmation remains sealed.

## Concrete execution

Same original science, seed151001301, beta4552.793066973983, exact recorded
TinyStories assets/tokenizer. GPU0 runs Q then QS; GPU1 runs QC. One Run All targets
500, retaining all50-step checkpoints, milestones200/350/500, original event
revisions, source/runtime identities and HF ACKs. No hard timeout or process kill.

The automatic controller looks up the exact source/run before allocating workers.
An existing run resumes its immutable HF checkpoint; a completed run returns
without training. Network/permission failures never count as a missing run.
Local leftovers are retained rather than deleted or silently used over HF state.
The same pinned Notebook must be used for recovery.

## Evidence transfer, not relabelling

The original gradient/alignment Pilot remains source67b3dd47ab88. Independent
analysis remains c29d29486105; the exact one-update comparison remains9c396fcf2d89.
Their byte hashes and actual HF references are retained. Frozen R3 science and
R3b runner/observer bytes match the successful delta source. This release changes
only admission to execution and automatic start/resume control; CPU/CI checks
cover that change. Source-code attestation binds the approved runtime, not a mutable
branch or an arbitrary environment boolean. Documentation-only commits need not
repeat GPU verification; the actual pinned Git HEAD is still the training identity.

No new-source full Pilot is claimed. Production event/HF history cost is not fully
measured. The historical ~66min scheduling proxy and~100min planning margin are
estimates; separate offline diagnostics add to total study GPU usage.

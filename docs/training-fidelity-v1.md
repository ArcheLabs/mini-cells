# Training fidelity v1

This path freezes the validated Python Echo experiment before any optimizer or
model search. `tools/export_training_fidelity.py` imports the production
`research/minicells` model, data, metrics, and trainer modules and exports the
exact FP32 initial parameters, deterministic batches, AdamW metadata, and
step-1/step-2/step-4/step-16 tensors. The checked-in fixture records Python
3.8.10, torch 2.4.1+cu121, device and validation provenance.

`crates/minicells-training-ref` is the single `#![no_std]` FP32 implementation
used by the Native runner and by `crates/minicells-training-service`. The
logical batch remains 256; gradients are summed over all samples, normalized by
the total valid-token count, globally clipped once, and passed to one AdamW
step. The padding embedding row is not updated, matching PyTorch. All large
forward/backward scratch arrays live in a caller-provided `TrainingWorkspace`,
and `GradientAccumulator`/`finalize_adamw_step` allow exact sequential shards
without changing sample order or optimizer semantics.

The independent guest is built with:

```text
tools/build_training_fidelity_service.sh
```

It writes `service/artifacts/training-fidelity.blob`. Authoritative gas is
reported only by `minicells-lab pvm-gas` from
`StandaloneExecutionResult.gas_used`; a decode or panic is recorded as
`NOT_MEASURED`, never classified as a gas ceiling.

Every real MiniJAM measurement is normalized with
`tools/summarize_minijam_telemetry.py`. The input records measured Refine and
Accumulate gas, wall time, peak memory, batch count, and sample count; the
summary emits p50/p95/max and Refine/Accumulate headroom against the canonical
1B limits. Synthetic or diagnostic-only runs cannot be promoted to this
telemetry record.

For memory debugging only, `pvm-gas --diagnostic-stage payload|decode|batch|train|forward|backward|full-batch|adamw|return`
wraps the payload in the diagnostic ABI and writes `pvm-diagnostic.json`; those
numbers are never promoted to final gas evidence. A valid synthetic payload can
be generated with `tools/make_synthetic_training_payload.py`.

The current machine evidence is in
`artifacts/pvm-algorithm-fidelity/`. Native parity passes at steps 1/2/4/16
with the existing 5e-4 tolerance, and the 5000-step Python/Rust learning gate
reaches 1.0 token and exact-sequence accuracy on both sides. Native chunked
accumulation and the corrected MCA1→MCF1 PVM path are bit-exact. MCA1 only
deserializes and accumulates a shard with frozen weights/optimizer state; MCF1
alone performs normalization, clipping and AdamW. Independent MCA1 measurements
for 1/2/4/8/16 samples are recorded in the evidence directory; 8 samples is
comfortably below the MiniJamSpec v1 1B Refine ceiling. The completed 256-sample production
envelope (32 sequential shards plus one finalize) is recorded in
`full-logical-batch-gate.json`: max canonical shard is 2,256,027,812 gas,
worst-case valid 8×32 shard is 2,311,013,084 gas, and the final result is
bit-exact. The authorized MiniJAM integration is pinned to MiniJamSpec v1:
Refine and Accumulate are each 1B, and unrelated topology limits remain unchanged.

Production integration is pinned to the exact MiniJAM commit recorded in the
artifact manifest, with the dedicated non-diagnostic artifact
described by `service/artifacts/minicells-training-v1.manifest.json`. The
local `TrainingRoundStateV1` gate proves 32 ordered MCA1 transitions followed
by one MCF1 finalize, with weights, Adam state, loss, gradient norm, token
count, and step all bit-exact against the monolithic Native step. The fresh
chain CreateService/one-step gate remains `NOT_STARTED_BY_POLICY` because the
required Season 2 probe exits 77 when no Docker daemon is available; no
receipt or service ID is fabricated.

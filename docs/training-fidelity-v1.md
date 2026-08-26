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
comfortably below the 5B Full profile. The full production logical-batch
envelope remains a separate measurement, so no production gas limit or
optimizer/model choice is changed before it is rerun.

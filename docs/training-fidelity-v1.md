# Training fidelity v1

This path freezes the validated Python Echo experiment before any optimizer or
model search. `tools/export_training_fidelity.py` imports the production
`research/minicells` model, data, metrics, and trainer modules and exports the
exact FP32 initial parameters, deterministic batches, AdamW metadata, and
step-1/step-2 tensors. It fails closed when PyTorch is unavailable.

`crates/minicells-training-ref` is the single `#![no_std]` FP32 implementation
used by the Native runner and by `crates/minicells-training-service`. The
logical batch remains 256; gradients are summed over all samples, normalized by
the total valid-token count, globally clipped once, and passed to one AdamW
step. The padding embedding row is not updated, matching PyTorch.

The independent guest is built with:

```text
tools/build_training_fidelity_service.sh
```

It writes `service/artifacts/training-fidelity.blob`. Authoritative gas is
reported only by `minicells-lab pvm-gas` from
`StandaloneExecutionResult.gas_used`; a decode or panic is recorded as
`NOT_MEASURED`, never classified as a gas ceiling.

The current machine evidence is in
`artifacts/pvm-algorithm-fidelity/`. It is blocked because this environment has
no PyTorch and the dedicated guest currently panics in the Direct Jambda
harness. No production gas limit or optimizer/model choice is changed until
both blockers are cleared.

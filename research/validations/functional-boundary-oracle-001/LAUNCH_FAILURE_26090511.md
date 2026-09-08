# Functional Boundary Oracle 001 — seed 26090511 launch interruption

Scientific classification: **NO RESULT / INFRASTRUCTURE-ONLY**.

The first hosted attempt for formal seed `26090511` terminated before producing `result.json` or any publishable scientific artifact. It must not count as PASS or FAIL.

## Confirmed root cause

The Oracle runner attempted to identify Granite packed expert tensors by comparing their shapes against `model.config.intermediate_size`.

The real Granite 3.1 1B-A400M layer-23 packed tensors observed on Kaggle are:

- `model.layers.23.block_sparse_moe.input_linear.weight`: `[32, 1024, 1024]`
- `model.layers.23.block_sparse_moe.output_linear.weight`: `[32, 1024, 512]`

The per-expert intermediate width is therefore `512`: the input tensor contains two aligned `512`-wide gate/up blocks, while the output tensor projects from `512` expert channels. The model-level config value used by the failed role classifier was not a valid substitute for this per-expert width.

Mutation 001 did not have this bug because it discovered exactly two rank-3 packed expert tensors and mapped them using their actual tensor/slice geometry rather than assuming the model-level `intermediate_size` was the expert width.

## Recovery

Formal execution now identifies gate/up and down roles from the two packed tensors themselves using the invariant:

- `gate_up.shape[1] == 2 * down.shape[2]`
- `gate_up.shape[2] == down.shape[1]`

For the observed Granite layout this yields expert intermediate width `512`, matching the frozen Oracle protocol.

A CPU regression test reproduces the exact observed shapes while deliberately setting the fake model-level `intermediate_size` to `1024`. The test requires the runtime detector to recover width `512` from tensor geometry.

## Scientific invariants

No scientific protocol field is changed by this repair. Formal seed `26090511` must be rerun under the same frozen protocol. The interruption remains neither PASS nor FAIL.

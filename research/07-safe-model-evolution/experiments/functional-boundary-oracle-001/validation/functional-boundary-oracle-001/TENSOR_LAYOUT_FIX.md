# Functional Boundary Oracle 001 — Granite packed tensor layout repair

The formal seed `26090511` launch failure exposed a runner-only role-identification bug.

Observed Granite layer-23 tensors:

- `input_linear.weight`: `[32, 1024, 1024]`
- `output_linear.weight`: `[32, 1024, 512]`

The expert intermediate width is `512`, inferred from the packed tensor relation, not the model-level `config.intermediate_size` field.

Formal execution now identifies the unique orientation satisfying:

- `gate_up.shape[1] == 2 * down.shape[2]`
- `gate_up.shape[2] == down.shape[1]`

This is an engineering repair only. `protocol.json`, formal seeds, data splits, optimizer, selection rule, gates, and decision rule are unchanged. The interrupted seed remains `NO RESULT` and must be rerun as `26090511`.
